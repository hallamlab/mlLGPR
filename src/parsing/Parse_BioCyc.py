'''
This is considered the main entry point to preprocesses
BioCyc PGDBs and to construct samples.
'''

import numpy as np
import os
import os.path
import sys
import time
import traceback
from prep_biocyc.DataObject import DataObject

try:
    import cPickle as pkl
except:
    import pickle as pkl


def _parseData(b_arg):
    '''
    Create training data by calling the Data class
    '''

    ##########################################################################################################
    ######################                   PREPROCESSING DATABASES                    ######################
    ##########################################################################################################

    objData = DataObject()
    if b_arg.save_builddata_kb:
        print('\n*** PREPROCESSING DATABASES...')
        objData.ExtractInfoFromDatabase(kbpath=b_arg.kbpath, dataFolder='', constraintKB='metacyc')
        objData.SaveData(data=objData, fname=b_arg.objectname,
                         savepath=b_arg.ospath, tag='the data object')

        print(
            '\n*** CONSTRUCTING THE FOLLOWING INDICATOR (OR ADJACENCY) BINARY MATRICES...')
        print('\t>> Creating the following mapping matrices...')
        reaction_ec = objData.MapDataMatrices(rowdata=objData.reaction_id, coldata=objData.ec_id,
                                              rowdataInfo='.reaction_info', mapRowBasedDataID=3, mapColID=3,
                                              constrainKB='metacyc', tag='ECs to Reactions')
        pathway_ec = objData.MapDataMatrices(rowdata=objData.pathway_id, coldata=objData.ec_id,
                                             rowdataInfo='.pathway_info', mapRowBasedDataID=4, mapColID=15,
                                             constrainKB='metacyc', tag='ECs to Pathways')

        print('\t>> Constructing the following indicator (or adjacency) binary matrices...')
        rxn_ec_spmatrix, rxn_ec_id = objData.CreateIndicatorMatrix(rowdata=reaction_ec, ncoldata=len(objData.ec_id),
                                                                   tag='reaction-ec', removeZeroEntries=True)
        ptw_ec_spmatrix, ptw_ec_id = objData.CreateIndicatorMatrix(rowdata=pathway_ec, ncoldata=len(objData.ec_id),
                                                                   tag='pathway-ec', removeZeroEntries=True)
        print('\t>> Deleting the mapping matrices...')
        del reaction_ec, pathway_ec
        print('\t>> Saving the following indicator (or adjacency) binary matrices in: \n\t{0:s}'.format(
            b_arg.ospath))
        objData.SaveData(data=(rxn_ec_spmatrix, rxn_ec_id), fname='reaction_ec.pkl', savepath=b_arg.ospath,
                         tag='reaction-ec')
        objData.SaveData(data=(ptw_ec_spmatrix, ptw_ec_id), fname='pathway_ec.pkl', savepath=b_arg.ospath,
                         tag='pathway-ec')

        print('\n*** MAPPING LABELS WITH FUNCTIONS...')
        objData.MapLabelswithFunctions(rowDataMatrix=ptw_ec_spmatrix, colIDx=ptw_ec_id, y=None, nSamples=b_arg.nsample,
                                       mapAll=True, useEC=True, constructRxn=False, fName=None,
                                       savepath=b_arg.ospath)

        print('\n*** EXTRACTING PATHWAY PROPERTIES...', )
        pathway_properties = objData.ExtractPathwayProperties(ptwECMatrix=ptw_ec_spmatrix, ecIDx=ptw_ec_id,
                                                              nFeatures=27)
        fileDesc = '#File Description: number of pathway x number of features\n'
        objData.SaveData(data=fileDesc, fname=b_arg.pathwayfeature, savepath=b_arg.ospath,
                         tag='the pathway properties', mode='w+b')
        objData.SaveData(data=('nPathways:', str(pathway_properties.shape[0]), 'nFeatures:', str(
            pathway_properties.shape[1])),
                         fname=b_arg.pathwayfeature, savepath=b_arg.ospath, mode='a+b', printTag=False)
        objData.SaveData(data=pathway_properties, fname=b_arg.pathwayfeature, savepath=b_arg.ospath,
                         mode='a+b', printTag=False)
        del pathway_properties

        print('\n*** EXTRACTING REACTION (EC) PROPERTIES...', )
        ec_properties = objData.ExtractReactionProperties(ptwECMatrix=ptw_ec_spmatrix, rxnECMatrix=rxn_ec_spmatrix,
                                                          pECIDx=ptw_ec_id, rECIDx=rxn_ec_id,
                                                          nFeatures=25)
        fileDesc = '#File Description: number of reaction (ec) x number of features\n'
        objData.SaveData(data=fileDesc, fname=b_arg.ecfeature, savepath=b_arg.ospath,
                         tag='the reaction (ec) properties', mode='w+b')
        objData.SaveData(data=('nECs:', str(ec_properties.shape[0]),
                               'nFeatures:', str(ec_properties.shape[1])),
                         fname=b_arg.ecfeature, savepath=b_arg.ospath, mode='a+b', printTag=False)
        objData.SaveData(data=ec_properties, fname=b_arg.ecfeature, savepath=b_arg.ospath,
                         mode='a+b', printTag=False)
        del ec_properties

    ##########################################################################################################
    ######################                 CONSTRUCT SYNTHETIC CORPORA                  ######################
    ##########################################################################################################

    if b_arg.save_dataset:
        print('\n*** CONSTRUCTING SYNTHETIC DATASET...')
        print('\t>> Loading files...')
        objData = objData.LoadData(fname=b_arg.objectname, loadpath=b_arg.ospath, tag='the data object')
        ptw_ec_spmatrix, ptw_ec_id = objData.LoadData(fname=b_arg.pathway_ec, loadpath=b_arg.ospath)
        objData.BuildSyntheticDataset(rowDataMatrix=ptw_ec_spmatrix, colIDx=ptw_ec_id, nSamples=b_arg.nsample,
                                      averageNitemsPerSample=b_arg.average_item_per_sample,
                                      nComponentsToCorrupt=b_arg.ncomponents_to_corrupt,
                                      exception=b_arg.lower_bound_nitem_ptw,
                                      nComponentsToCorruptOutside=b_arg.ncomponents_to_corrupt_outside,
                                      addNoise=b_arg.add_noise, displayInterval=b_arg.display_interval,
                                      constraint_kb='metacyc', useEC=True,
                                      constructRxn=False, ptwConstraint=True,
                                      provided_lst=None, fName='synset', savepath=b_arg.dspath)
        X = objData.FormatCuratedDataset(nSamples=b_arg.nsample, rowDataMatrix=ptw_ec_spmatrix, colIDx=ptw_ec_id,
                                         useEC=True, constructRxn=False,
                                         minpathDataset=False, minpathMapFile=False,
                                         fName='synset', loadpath=b_arg.dspath)
        fName = os.path.join(b_arg.ospath, b_arg.ecfeature)
        with open(fName, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    ec_properties = data
                    break
        feature_lst = [42, 68, 32]
        matrixList = [ptw_ec_spmatrix] + [ec_properties]
        fName = 'synset' + '_' + str(b_arg.nsample) + '_X.pkl'
        objData.BuildFeaturesMatrix(X=X, matrixList=matrixList, colIDx=ptw_ec_id, featuresList=feature_lst,
                                    displayInterval=b_arg.display_interval, XName=fName, savepath=b_arg.dspath)

    ##########################################################################################################
    ######################                   CONSTRUCT GOLDEN CORPORA                   ######################
    ##########################################################################################################

    nSamples = 63
    if b_arg.build_golden_dataset:
        print('\n*** CONSTRUCTING GOLDEN DATASET...')
        print('\t>> Loading files...')
        objData = objData.LoadData(fname=b_arg.objectname, loadpath=b_arg.ospath, tag='the data object')
        ptw_ec_spmatrix, ptw_ec_id = objData.LoadData(fname=b_arg.pathway_ec, loadpath=b_arg.ospath)
        objData.BuildGoldenDataset(rowDataMatrix=ptw_ec_spmatrix, KB_lst=objData.lst_kbpaths,
                                   displayInterval=b_arg.display_interval, constructRxn=False,
                                   constraint_kb='metacyc', fName='golden', savepath=b_arg.dspath)
        X = objData.FormatCuratedDataset(nSamples=nSamples, rowDataMatrix=ptw_ec_spmatrix, colIDx=ptw_ec_id,
                                         useEC=True, constructRxn=False, pathologicInput=False,
                                         minpathDataset=False, minpathMapFile=False, fName='golden',
                                         loadpath=b_arg.dspath)

        print('\n*** EXTRACTING FEATURES FROM GOLDEN DATASET...')
        fName = 'golden' + '_' + str(nSamples) + '_Xm.pkl'
        print(
            '\t\t## Loading the golden dataset (X) from: {0:s}'.format(fName))
        fName = os.path.join(b_arg.dspath, fName)
        with open(fName, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    X = data
                    break
        print('\t\t## Loading the ec properties from: {0:s}'.format(
            b_arg.ecfeature))
        fName = os.path.join(b_arg.ospath, b_arg.ecfeature)
        with open(fName, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    ec_properties = data
                    break

        feature_lst = [42, 68, 32]
        matrixList = [ptw_ec_spmatrix] + [ec_properties]
        fName = 'golden' + '_' + str(nSamples) + '_X.pkl'
        objData.BuildFeaturesMatrix(X=X, matrixList=matrixList, colIDx=ptw_ec_id, featuresList=feature_lst,
                                    displayInterval=b_arg.display_interval, XName=fName, savepath=b_arg.dspath)


def BioCycMain(b_arg):
    try:
        timeref = time.time()
        _parseData(b_arg)
        print('\n*** THE PREPROCESSING CONSUMED {0:f} SECONDS\n'.format(round(time.time() - timeref, 3)),
              file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
