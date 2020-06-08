'''
This file is the main entry used to train the input dataset
using mlLGPR train and also report the predicted pathways.
'''

import numpy as np
import os
import os.path
import sys
import time
import traceback
from model.mlLGPR import mlLGPR
from model.mlUtility import ListHeaderFile, DetailHeaderFile, ComputePathwayAbundance
from model.mlUtility import PrepareDataset, LoadData, SaveData, Score
from prep_biocyc.DataObject import DataObject

try:
    import cPickle as pkl
except:
    import pickle as pkl


def _saveFileName(clf, saveFileName, time=False):
    if clf.l1_ratio == 1:
        saveFileName = saveFileName + '_l1_ab'
    elif clf.l1_ratio == 0:
        saveFileName = saveFileName + '_l2_ab'
    else:
        saveFileName = saveFileName + '_en_ab'

    if clf.useReacEvidenceFeatures:
        saveFileName = saveFileName + '_re'
    if clf.useItemEvidenceFeatures:
        saveFileName = saveFileName + '_pe'
    if clf.usePossibleClassFeatures:
        saveFileName = saveFileName + '_pp'
    if clf.useLabelComponentFeatures:
        saveFileName = saveFileName + '_pc'

    return saveFileName


def _report(clf, use_tCriterion, dataFile, saveFileName, t_arg):
    saveFileDetails = saveFileName + '.details'
    saveFileLists = saveFileName + '.lists'
    SaveData(data=ListHeaderFile(), fname=saveFileLists, savepath=t_arg.rspath,
             tag='class labels', mode='w', wString=True)
    SaveData(data=DetailHeaderFile(), fname=saveFileDetails, savepath=t_arg.rspath,
             tag='detail class labels', mode='w', wString=True)
    SaveData(data='>> Predicted class labels for: {0:s}...\n'.format(dataFile),
             fname=saveFileLists, savepath=t_arg.rspath, mode='a',
             wString=True, printTag=False)
    SaveData(data='>> Predicted class labels for: {0:s}...\n'.format(dataFile),
             fname=saveFileDetails, savepath=t_arg.rspath, mode='a',
             wString=True, printTag=False)
    X_file = os.path.join(t_arg.dspath, dataFile)
    startTime = time.time()
    y_pred_prob = clf.predict(X_file=X_file, applyTCriterion=use_tCriterion, estimateProb=True)
    pathwayAbun = ComputePathwayAbundance(X_file=X_file,
                                          labelsComponentsFile=os.path.join(t_arg.ospath, t_arg.pathway_ec),
                                          classLabelsIds=clf.classLabelsIds, mlbClasses=clf.mlb.classes,
                                          nBatches=clf.nBatches, nTotalComponents=clf.nTotalComponents)
    y_pred = np.copy(y_pred_prob)
    y_pred[y_pred_prob >= clf.threshold] = 1
    y_pred[y_pred_prob != 1] = 0
    labels = clf.mlb.inverse_transform(y_pred)
    SaveData(data=ListHeaderFile(header=False), fname=saveFileLists, savepath=t_arg.rspath,
             mode='a', wString=True, printTag=False)
    SaveData(data=DetailHeaderFile(header=False), fname=saveFileDetails, savepath=t_arg.rspath,
             mode='a', wString=True, printTag=False)
    for sidx in np.arange(len(labels)):
        sampleInit = False
        for pid in labels[sidx]:
            sampleId = ""
            nLabels = ""
            labelProb = str("{0:.4f}").format(y_pred_prob[sidx, clf.mlb.classes.index(pid)])
            labelAbun = str("{0:.4f}").format(pathwayAbun[sidx, clf.mlb.classes.index(pid)])
            if not sampleInit:
                sampleId = str(sidx + 1)
                nLabels = str(len(labels[sidx]))
                sampleInit = True
            data = " {1:10}{0}{2:15}{0}{3:40}{0}{4:15}{0}{5:18}\n".format(" | ", sampleId, nLabels,
                                                                          str(pid), labelProb, labelAbun)
            SaveData(data=" {1:10}{0}{2:15}{0}{3:40}\n".format(" | ", sampleId, nLabels, str(pid)),
                     fname=saveFileLists, savepath=t_arg.rspath, mode='a', wString=True, printTag=False)
            SaveData(data=data, fname=saveFileDetails, savepath=t_arg.rspath,
                     mode='a', wString=True, printTag=False)
        SaveData(data="{0}\n".format("-" * 60), fname=saveFileLists,
                 savepath=t_arg.rspath, mode='a', wString=True, printTag=False)
        SaveData(data="{0}\n".format("-" * 111), fname=saveFileDetails, savepath=t_arg.rspath,
                 mode='a', wString=True, printTag=False)


def _train(t_arg, channel):
    '''
    Create training objData by calling the Data class
    '''

    ##########################################################################################################
    ###################        TRAINING DATA USING MULTI-LABEL LOGISTIC REGRESSION         ###################
    ##########################################################################################################

    if t_arg.train:
        print('\n*** BEGIN TRAINING USING MULTI-LABEL LEARNING...')
        print('\t>> Loading files...')
        objData = LoadData(fname=t_arg.objectname, loadpath=t_arg.ospath, tag='data object')
        data_id = objData.pathway_id
        del objData
        if t_arg.useLabelComponentFeatures:
            print('\t>> Retreiving labels components mapping file from: {0:s}'.format(t_arg.pathwayfeature))
            labelsComponentsMappingFile = os.path.join(t_arg.ospath, t_arg.pathway_ec)
        else:
            labelsComponentsMappingFile = None

        print('\t>> Constructing a training dataset from: {0:s} and {1:s}'.format(t_arg.X_name, t_arg.y_name))
        file_name = t_arg.file_name + '_values.pkl'
        if t_arg.save_prepared_dataset is True:
            value_lst = PrepareDataset(dataId=data_id, useAllLabels=t_arg.all_classes, trainSize=t_arg.train_size,
                                       valSize=t_arg.val_size, datasetPath=t_arg.dspath,
                                       X_name=t_arg.X_name, y_name=t_arg.y_name,
                                       file_name=t_arg.file_name)
            SaveData(data=value_lst, fname=file_name, savepath=t_arg.dspath, tag='prepared dataset')
        else:
            value_lst = LoadData(fname=file_name, loadpath=t_arg.dspath, tag='prepared dataset')

        print('\t>> Building multi-label logistic regression train...')

        alpha = t_arg.alpha
        l1_ratio = t_arg.l1_ratio

        clf = mlLGPR(classes=value_lst[0], classLabelsIds=value_lst[1],
                     labelsComponentsFile=labelsComponentsMappingFile,
                     binarizeAbundance=t_arg.binarize,
                     useReacEvidenceFeatures=t_arg.useReacEvidenceFeatures,
                     usePossibleClassFeatures=t_arg.usePossibleClassFeatures,
                     useItemEvidenceFeatures=t_arg.useItemEvidenceFeatures,
                     useLabelComponentFeatures=t_arg.useLabelComponentFeatures,
                     nTotalComponents=value_lst[2], nTotalClassLabels=value_lst[3],
                     nTotalEvidenceFeatures=value_lst[4],
                     nTotalClassEvidenceFeatures=value_lst[5],
                     penalty=t_arg.penalty, alpha=alpha,
                     l1_ratio=l1_ratio, max_inner_iter=t_arg.max_inner_iter,
                     nEpochs=t_arg.nEpochs, nBatches=t_arg.nBatches,
                     testInterval=t_arg.test_interval, adaptive_beta=t_arg.adaptive_beta,
                     threshold=t_arg.threshold, n_jobs=t_arg.n_jobs)
        print('\t\t## The following parameters are applied:\n\t\t\t{0}'.format(clf.print_arguments()),
              file=sys.stderr)
        print('\t>> train...')
        startTime = time.time()
        clf.fit(X_file=os.path.join(t_arg.dspath, value_lst[6]),
                y_file=os.path.join(t_arg.dspath, value_lst[7]),
                XdevFile=os.path.join(t_arg.dspath, value_lst[8]),
                ydevFile=os.path.join(t_arg.dspath, value_lst[9]),
                savepath=t_arg.mdpath)

    ##########################################################################################################
    ###################         PREDICTING LABELS USING MULTI-LABEL LEARNING MODEL         ###################
    ##########################################################################################################

    if t_arg.predict:
        if t_arg.parse_input:
            print('\n*** EXTRACTING INFORMATION FROM DATASET...')
            print('\t>> Loading files...')
            objData = LoadData(fname=t_arg.objectname, loadpath=t_arg.ospath, tag='data object')
            ptw_ec_spmatrix, ptw_ec_id = objData.LoadData(fname=t_arg.pathway_ec, loadpath=t_arg.ospath,
                                                          tag='mapping ECs onto pathway')
            X = objData.ExtractInputFromMGFiles(colIDx=ptw_ec_id, useEC=True, folderPath=t_arg.dspath,
                                                processes=t_arg.n_jobs)
            nSamples = X.shape[0]
            fName = os.path.join(t_arg.ospath, t_arg.ecfeature)
            with open(fName, 'rb') as f_in:
                while True:
                    data = pkl.load(f_in)
                    if type(data) is np.ndarray:
                        ec_properties = data
                        break
            feature_lst = [42, 68, 32]
            matrixList = [ptw_ec_spmatrix] + [ec_properties]
            X_name = t_arg.file_name + '.pkl'
            objData.BuildFeaturesMatrix(X=X, matrixList=matrixList, colIDx=ptw_ec_id, featuresList=feature_lst,
                                        displayInterval=t_arg.display_interval, XName=X_name, savepath=t_arg.dspath)
        else:
            X_name = t_arg.X_name

        print('\n*** PREDICTING USING MULTI-LABEL LOGISTIC REGRESSION...')
        if t_arg.useLabelComponentFeatures:
            print('\t>> Retreiving labels components mapping file from: {0:s}'.format(t_arg.pathwayfeature))
            labelsComponentsMappingFile = os.path.join(t_arg.ospath, t_arg.pathway_ec)
        else:
            labelsComponentsMappingFile = None
        print('\t>> Loading pre-trained multi-label train...')
        modelFileName = t_arg.model
        clf = LoadData(fname=modelFileName, loadpath=t_arg.mdpath,
                       tag='the multi-label logistic regression train')
        clf.labelsComponentsFile = labelsComponentsMappingFile
        clf.nBatches = t_arg.nBatches
        clf.n_jobs = t_arg.n_jobs
        clf.adaptive_beta = t_arg.adaptive_beta
        saveFileName = t_arg.predict_file
        saveFileName = _saveFileName(clf, saveFileName)
        print('\t>> Predicting class labels for: {0:s}...'.format(X_name))
        _report(clf, t_arg.use_tCriterion, X_name, saveFileName, t_arg)


def TrainMain(t_arg, channel):
    try:
        if os.path.isdir(t_arg.ospath):
            timeref = time.time()
            _train(t_arg, channel)

            print('\n*** TRAINING AND EVALUATION CONSUMED {0:f} SECONDS'.format(round(time.time() - timeref, 3)),
                  file=sys.stderr)
        else:
            print('\n*** PLEASE MAKE SURE TO PROVIDE THE CORRECT PATH FOR THE DATA OBJECT', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
