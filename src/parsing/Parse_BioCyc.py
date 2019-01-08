'''
This is considered the main entry point to preprocesses
BioCyc PGDBs and to construct samples.
'''

import os
import os.path
import sys
import time
import traceback

import numpy as np
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
    print('\n*** PREPROCESSING DATABASES...')
    if b_arg.save_builddata_kb:
        objData.ExtractInfoFromDatabase(kbpath=b_arg.kbpath, dataFolder='', constraintKB=b_arg.constraint_kb)
        objData.SaveData(data=objData, fname=b_arg.objectname, savepath=b_arg.ospath, tag='the data object')
    else:
        objData = objData.LoadData(fname=b_arg.objectname, loadpath=b_arg.ospath, tag='the data object')

    ##########################################################################################################
    ######################          CREATING INDICATOR AND ADJACENCY MATRICES           ######################
    ##########################################################################################################

    print('\n*** CONSTRUCTING THE FOLLOWING INDICATOR (OR ADJACENCY) BINARY MATRICES...')
    if b_arg.save_indicator:
        print('\t>> Creating the following mapping matrices...')
        reaction_ec = objData.MapDataMatrices(rowdata=objData.reaction_id, coldata=objData.ec_id,
                                              rowdataInfo='.reaction_info', mapRowBasedDataID=3, mapColID=3,
                                              constrainKB=b_arg.constraint_kb, tag='ECs to Reactions')

        pathway_ec = objData.MapDataMatrices(rowdata=objData.pathway_id, coldata=objData.ec_id,
                                             rowdataInfo='.pathway_info', mapRowBasedDataID=4, mapColID=15,
                                             constrainKB=b_arg.constraint_kb, tag='ECs to Pathways')

        print('\t>> Constructing the following indicator (or adjacency) binary matrices...')
        rxn_ec_spmatrix, rxn_ec_id = objData.CreateIndicatorMatrix(rowdata=reaction_ec, ncoldata=len(objData.ec_id),
                                                                   tag='reaction-ec', removeZeroEntries=True)
        ptw_ec_spmatrix, ptw_ec_id = objData.CreateIndicatorMatrix(rowdata=pathway_ec, ncoldata=len(objData.ec_id),
                                                                   tag='pathway-ec', removeZeroEntries=True)
        print('\t>> Deleting the mapping matrices...')
        # del gene_go, gene_product, reaction_gene, reaction_ec, pathway_gene, pathway_reaction, pathway_ec
        del reaction_ec, pathway_ec

        print('\t>> Saving the following indicator (or adjacency) binary matrices in: \n\t{0:s}'.format(
            b_arg.ospath))
        objData.SaveData(data=(rxn_ec_spmatrix, rxn_ec_id), fname=b_arg.reaction_ec, savepath=b_arg.ospath,
                         tag='reaction-ec')
        objData.SaveData(data=(ptw_ec_spmatrix, ptw_ec_id), fname=b_arg.pathway_ec, savepath=b_arg.ospath,
                         tag='pathway-ec')

        print('\n*** MAPPING LABELS WITH FUNCTIONS...')
        objData.MapLabelswithFunctions(rowDataMatrix=ptw_ec_spmatrix, colIDx=ptw_ec_id, y=None, nSamples=b_arg.nsample,
                                       mapAll=b_arg.mapall, useEC=b_arg.use_ec, constructRxn=b_arg.construct_reaction,
                                       fName=None, savepath=b_arg.ospath)
    elif not b_arg.save_indicator:
        rxn_ec_spmatrix, rxn_ec_id = objData.LoadData(fname=b_arg.reaction_ec, loadpath=b_arg.ospath)
        ptw_ec_spmatrix, ptw_ec_id = objData.LoadData(fname=b_arg.pathway_ec, loadpath=b_arg.ospath)
    else:
        print('\t>> Building or loading is not applied...')

    ##########################################################################################################
    ################################          EXTRACTING PROPERTIES           ################################
    ##########################################################################################################

    print('\n*** EXTRACTING PATHWAY PROPERTIES...', )
    if b_arg.build_pathway_properties:
        pathway_properties = objData.ExtractPathwayProperties(ptwECMatrix=ptw_ec_spmatrix, ecIDx=ptw_ec_id,
                                                              nFeatures=b_arg.num_pathwayfeatures)
        fileDesc = '#File Description: number of pathway x number of features\n'
        objData.SaveData(data=fileDesc, fname=b_arg.pathwayfeature, savepath=b_arg.dspath,
                         tag='the pathway properties', mode='w+b')
        objData.SaveData(
            data=('nPathways:', str(pathway_properties.shape[0]), 'nFeatures:', str(pathway_properties.shape[1])),
            fname=b_arg.pathwayfeature, savepath=b_arg.dspath, mode='a+b', printTag=False)
        objData.SaveData(data=pathway_properties, fname=b_arg.pathwayfeature, savepath=b_arg.dspath,
                         mode='a+b', printTag=False)
        del pathway_properties
    else:
        print('\t>> Building pathway properties is not applied...')

    print('\n*** EXTRACTING REACTION (EC) PROPERTIES...', )
    if b_arg.build_ec_properties:
        ec_properties = objData.ExtractReactionProperties(ptwECMatrix=ptw_ec_spmatrix, rxnECMatrix=rxn_ec_spmatrix,
                                                          pECIDx=ptw_ec_id, rECIDx=rxn_ec_id,
                                                          nFeatures=b_arg.num_ecfeatures)
        fileDesc = '#File Description: number of reaction (ec) x number of features\n'
        objData.SaveData(data=fileDesc, fname=b_arg.ecfeature, savepath=b_arg.dspath,
                         tag='the reaction (ec) properties', mode='w+b')
        objData.SaveData(data=('nECs:', str(ec_properties.shape[0]),
                               'nFeatures:', str(ec_properties.shape[1])),
                         fname=b_arg.ecfeature, savepath=b_arg.dspath, mode='a+b', printTag=False)
        objData.SaveData(data=ec_properties, fname=b_arg.ecfeature, savepath=b_arg.dspath,
                         mode='a+b', printTag=False)
        del ec_properties
    else:
        print('\t>> Building ec properties is not applied...')

    print('\n*** BUILDING PATHWAY SIMILARITY MATRIX ...', )
    if b_arg.build_pathway_similarities:
        objData.BuildSimilarityMatrix(ptwECMatrix=ptw_ec_spmatrix, fName=b_arg.pathway_similarity,
                                      savepath=b_arg.ospath)
    else:
        print('\t>> Building pathway similarity matrix is not applied...')

    ##########################################################################################################
    ######################                 CONSTRUCT SYNTHETIC CORPORA                  ######################
    ##########################################################################################################

    print('\n*** CONSTRUCTING SYNTHETIC DATASET...')
    if b_arg.save_dataset:
        objData.BuildSyntheticDataset(rowDataMatrix=ptw_ec_spmatrix, colIDx=ptw_ec_id, nSamples=b_arg.nsample,
                                      averageNitemsPerSample=b_arg.average_item_per_sample,
                                      nComponentsToCorrupt=b_arg.ncomponents_to_corrupt,
                                      exception=b_arg.lower_bound_nitem_ptw,
                                      nComponentsToCorruptOutside=b_arg.ncomponents_to_corrupt_outside,
                                      addNoise=b_arg.add_noise, displayInterval=b_arg.display_interval,
                                      constraint_kb=b_arg.constraint_kb, useEC=b_arg.use_ec,
                                      constructRxn=b_arg.construct_reaction, ptwConstraint=b_arg.constraint_pathway,
                                      provided_lst=None, fName=b_arg.syntheticdataset_ptw_ec, savepath=b_arg.dspath)
        X = objData.FormatCuratedDataset(nSamples=b_arg.nsample, rowDataMatrix=ptw_ec_spmatrix, colIDx=ptw_ec_id,
                                         useEC=b_arg.use_ec, constructRxn=b_arg.construct_reaction,
                                         minpathDataset=b_arg.minpath_ds, minpathMapFile=b_arg.minpath_map,
                                         fName=b_arg.syntheticdataset_ptw_ec, loadpath=b_arg.dspath)

    if b_arg.build_synthetic_features:
        print('\n*** EXTRACTING FEATURES FROM SYNTHETIC DATASET...')
        print('\t\t## Loading the ec properties from: {0:s}'.format(b_arg.ecfeature))
        fName = os.path.join(b_arg.dspath, b_arg.ecfeature)
        with open(fName, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    ec_properties = data
                    break

        feature_lst = [b_arg.num_reaction_evidence_features] + [b_arg.num_ec_evidence_features] + [
            b_arg.num_ptw_evidence_features]
        matrixList = [ptw_ec_spmatrix] + [ec_properties]
        fName = b_arg.syntheticdataset_ptw_ec + '_' + str(b_arg.nsample)
        objData.BuildFeaturesMatrix(X=X, matrixList=matrixList, colIDx=ptw_ec_id, featuresList=feature_lst,
                                    displayInterval=b_arg.display_interval, XName=fName, savepath=b_arg.dspath)
    else:
        print('\t>> Building features is not applied...')

    ##########################################################################################################
    ######################                   CONSTRUCT GOLDEN CORPORA                   ######################
    ##########################################################################################################

    nSamples = 63
    print('\n*** CONSTRUCTING GOLDEN DATASET...')
    if b_arg.build_golden_dataset:
        startTime = time.time()
        objData.BuildGoldenDataset(rowDataMatrix=ptw_ec_spmatrix, KB_lst=objData.lst_kbpaths,
                                   displayInterval=b_arg.display_interval, constructRxn=b_arg.construct_reaction,
                                   constraint_kb=b_arg.constraint_kb, fName=b_arg.goldendataset_ptw_ec,
                                   savepath=b_arg.dspath)
        X = objData.FormatCuratedDataset(nSamples=nSamples, rowDataMatrix=ptw_ec_spmatrix, colIDx=ptw_ec_id,
                                         useEC=b_arg.use_ec, constructRxn=b_arg.construct_reaction,
                                         pathologicInput=b_arg.pathologic_input, minpathDataset=b_arg.minpath_ds,
                                         minpathMapFile=b_arg.minpath_map, fName=b_arg.goldendataset_ptw_ec,
                                         loadpath=b_arg.dspath)
        print('\t\t## Constructing golden dataset consumed {0:f} seconds'
              .format(round(time.time() - startTime, 3)))
    else:
        print('\t>> Constructing golden dataset is not applied...')

    if b_arg.build_golden_features:
        print('\n*** EXTRACTING FEATURES FROM GOLDEN DATASET...')
        fName = b_arg.goldendataset_ptw_ec + '_' + str(nSamples) + '_Xm.pkl'
        print('\t\t## Loading the golden dataset (X) from: {0:s}'.format(fName))
        fName = os.path.join(b_arg.dspath, fName)
        with open(fName, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    X = data
                    break
        print('\t\t## Loading the ec properties from: {0:s}'.format(b_arg.ecfeature))
        fName = os.path.join(b_arg.dspath, b_arg.ecfeature)
        with open(fName, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    ec_properties = data
                    break

        feature_lst = [b_arg.num_reaction_evidence_features] + [b_arg.num_ec_evidence_features] + [
            b_arg.num_ptw_evidence_features]
        matrixList = [ptw_ec_spmatrix] + [ec_properties]
        fName = b_arg.goldendataset_ptw_ec + '_' + str(nSamples)
        startTime = time.time()
        objData.BuildFeaturesMatrix(X=X, matrixList=matrixList, colIDx=ptw_ec_id, featuresList=feature_lst,
                                    displayInterval=b_arg.display_interval, XName=fName, savepath=b_arg.dspath)
        print('\t\t## Building golden dataset features consumed {0:f} seconds'
              .format(round(time.time() - startTime, 3)))
    else:
        print('\t>> Building features is not applied...')


def BioCycMain(b_arg):
    try:
        if os.path.isdir(b_arg.kbpath):
            timeref = time.time()
            print('*** THE KNOWLEDGE BASES ARE LOCATED IN: {0:s}'.format(b_arg.kbpath))
            _parseData(b_arg)
            print('\n*** THE PREPROCESSING CONSUMED {0:f} SECONDS\n'
                  .format(round(time.time() - timeref, 3)), file=sys.stderr)
        else:
            print('\n*** PLEASE MAKE SURE TO PROVIDE THE CORRECT PATH FOR THE DATABASES', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
