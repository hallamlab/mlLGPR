'''
This file preprocesses the input data in PathoLogic File Format (.pf).
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


def _ParseInput(m_arg):
    '''
    Create training objData by calling the Data class
    '''

    ##########################################################################################################
    ######################            LOAD DATA OBJECT AND INDICATOR MATRICES           ######################
    ##########################################################################################################

    print('*** THE DATA OBJECT AND THE ASSOCIATED PARAMETERS ARE LOCATED IN: {0:s}'.format(
        m_arg.ospath))
    objData = DataObject()
    nSamples = 0
    objData = objData.LoadData(fname=m_arg.objectname, loadpath=m_arg.ospath)
    ptw_ec_spmatrix, ptw_ec_id = objData.LoadData(fname=m_arg.pathway_ec, loadpath=m_arg.ospath)

    ##########################################################################################################
    ######################      EXTRACTING INFORMATION FROM METEGENOMICS DATASET        ######################
    ##########################################################################################################

    print('\n*** EXTRACTING INFORMATION FROM METEGENOMICS DATASET...')
    if m_arg.extract_info_mg:
        X = objData.ExtractInputFromMGFiles(colIDx=ptw_ec_id, useEC=m_arg.use_ec, folderPath=m_arg.inputpath,
                                            processes=m_arg.n_jobs)
        nSamples = X.shape[0]
        file = m_arg.metegenomics_dataset + '_' + str(nSamples) + '_Xm.pkl'
        fileDesc = '# Metagenomics dataset representing a list of data components (X)...'
        objData.SaveData(data=fileDesc, fname=file, savepath=m_arg.dspath, tag='the metagenomics dataset (X)',
                         mode='w+b')
        objData.SaveData(data=X, fname=file, savepath=m_arg.dspath, mode='a+b', printTag=False)

        y, sample_ids = objData.ExtractOutputFromMGFiles(folderPath=m_arg.inputpath, processes=m_arg.n_jobs)
        file = m_arg.metegenomics_dataset + '_' + str(nSamples) + '_y.pkl'
        fileDesc = '# Metagenomics dataset representing a list of data components (y) with ids...'
        objData.SaveData(data=fileDesc, fname=file, savepath=m_arg.dspath, tag='the metagenomics dataset (y)',
                         mode='w+b')
        objData.SaveData(data=(y, sample_ids), fname=file, savepath=m_arg.dspath, mode='a+b', printTag=False)

        objData.BuildMinPathDataset(X=X, colIDx=ptw_ec_id, useEC=m_arg.use_ec, nSamples=X.shape[0],
                                    fName=m_arg.metegenomics_dataset, savepath=m_arg.dspath)
    else:
        print('\t>> Extracting information from metegenomics dataset is not indicated...')

    if m_arg.mapping:
        print('\n*** MAPPING LABELS WITH FUNCTIONS...')
        if y:
            objData.MapLabelswithFunctions(rowDataMatrix=ptw_ec_spmatrix, colIDx=ptw_ec_id, y=y, nSamples=X.shape[0],
                                           mapAll=False, useEC=m_arg.use_ec, constructRxn=m_arg.construct_reaction,
                                           fName=m_arg.metegenomics_dataset, savepath=m_arg.dspath)
        else:
            print('\t>> A list of pathways is not provided...')

    ##########################################################################################################
    ################################             BUILDING FEATURES             ###############################
    ##########################################################################################################

    print('\n*** EXTRACTING FEATURES FROM METEGENOMICS DATASET...')
    if m_arg.build_mg_features:
        if nSamples != 0:
            nSamples = y.shape[0]
        else:
            nSamples = 418

        fName = m_arg.metegenomics_dataset + '_' + str(nSamples) + '_Xm.pkl'
        print('\t\t## Loading the metagenomics dataset (X) from: {0:s}'.format(fName))
        fName = os.path.join(m_arg.dspath, fName)
        with open(fName, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    X = data
                    break

        print('\t\t## Loading the ec properties from: {0:s}'.format(fName))
        fName = os.path.join(m_arg.dspath, m_arg.ecfeature)
        with open(fName, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is np.ndarray:
                    ec_properties = data
                    break

        feature_lst = [m_arg.num_reaction_evidence_features] + [m_arg.num_ec_evidence_features] + [
            m_arg.num_ptw_evidence_features]
        matrixList = [ptw_ec_spmatrix] + [ec_properties]
        fName = m_arg.metegenomics_dataset + '_' + str(nSamples)
        objData.BuildFeaturesMatrix(X=X, matrixList=matrixList, colIDx=ptw_ec_id, featuresList=feature_lst,
                                    displayInterval=m_arg.display_interval, XName=fName, savepath=m_arg.dspath)
    else:
        print('\t>> Building features is not applied...')


def InputMain(m_arg):
    try:
        if os.path.isdir(m_arg.ospath):
            timeref = time.time()
            _ParseInput(m_arg)
            print('\n*** THE DATASET PROCESSING CONSUMED {0:f} SECONDS'.format(round(time.time() - timeref, 3)),
                  file=sys.stderr)
        else:
            print('\n*** PLEASE MAKE SURE TO PROVIDE THE CORRECT PATH FOR THE CONSTRUCTED CORPORA '
                  'AND THE ASSOCIATED PARAMETERS', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
