'''
This file is the main entry used to train the input dataset
using mlLGPR train and also report the predicted pathways.
'''

import os
import os.path
import sys
import time
import traceback

import numpy as np
from model.mlLGPR import mlLGPR
from model.mlUtility import ListHeaderFile, DetailHeaderFile, ComputePathwayAbundance
from model.mlUtility import PrepareDataset, LoadData, SaveData, Score

try:
    import cPickle as pkl
except:
    import pickle as pkl


def _datasetType(t_arg):
    if t_arg.ds_type == "syn_ds":
        fObject = t_arg.syntheticdataset_ptw_ec
    elif t_arg.ds_type == "meta_ds":
        fObject = t_arg.metegenomics_dataset
    else:
        fObject = t_arg.goldendataset_ptw_ec
    return fObject


def _saveFileName(clf, saveFileName, time=False):
    tag = ''
    if time:
        tag = '_pred_time'

    if clf.adjustCoef:
        saveFileName = saveFileName + '_adjusted_' + clf.coef_similarity_type

    if clf.l1_ratio == 1:
        saveFileName = saveFileName + tag + '_l1_ab'
    elif clf.l1_ratio == 0:
        saveFileName = saveFileName + tag + '_l2_ab'
    else:
        saveFileName = saveFileName + tag + '_en_ab'

    if clf.useReacEvidenceFeatures:
        saveFileName = saveFileName + '_re'
    if clf.useItemEvidenceFeatures:
        saveFileName = saveFileName + '_pe'
    if clf.usePossibleClassFeatures:
        saveFileName = saveFileName + '_pp'
    if clf.useLabelComponentFeatures:
        saveFileName = saveFileName + '_pc'

    return saveFileName


def _report(clf, dataFile, saveFileName, t_arg, tag='meta'):
    saveFileDetails = saveFileName + '_' + tag + '.details'
    saveFileLists = saveFileName + '_' + tag + '.lists'
    saveTimeFilename = _saveFileName(clf, t_arg.mllr, time=True) + '_' + tag + '.txt'

    SaveData(data=ListHeaderFile(), fname=saveFileLists, savepath=t_arg.rspath,
             tag='class labels', mode='w', wString=True)
    SaveData(data=DetailHeaderFile(), fname=saveFileDetails, savepath=t_arg.rspath,
             tag='detail class labels', mode='w', wString=True)
    SaveData(data='>> Predicted class labels for: {0:s}...\n'.format(dataFile + '_X.pkl'),
             fname=saveFileLists, savepath=t_arg.rspath, mode='a',
             wString=True, printTag=False)
    SaveData(data='>> Predicted class labels for: {0:s}...\n'.format(dataFile + '_X.pkl'),
             fname=saveFileDetails, savepath=t_arg.rspath, mode='a',
             wString=True, printTag=False)
    X_file = os.path.join(t_arg.dspath, dataFile + '_X.pkl')
    startTime = time.time()
    y_pred_prob = clf.predict(X_file=X_file, estimateProb=True)
    elapsedTime = str((_saveFileName(clf, t_arg.mllr, time=True), time.time() - startTime))
    print('\t\t## Inference consumed {0:f} seconds'.format(round(time.time() - startTime, 3)))
    SaveData(data=elapsedTime, fname=saveTimeFilename, savepath=t_arg.rspath, tag='time performance', mode='w',
             wString=True)
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
    ######################                   LOADING DATA OBJECT                        ######################
    ##########################################################################################################

    print('*** THE DATA OBJECT IS LOCATED IN: {0:s}'.format(t_arg.dspath))
    objData = LoadData(fname=t_arg.objectname, loadpath=t_arg.ospath, tag='data object')
    data_id = objData.pathway_id
    del objData

    ##########################################################################################################
    ######################                  PREPROCESSING DATASET                       ######################
    ##########################################################################################################

    print('\n*** PREPROCESSING DATASET USED FOR TRAINING AND EVALUATING...')
    if t_arg.useItemfeatures:
        print('\t>> Retreiving items properties file from: {0:s}'.format(t_arg.pathwayfeature))
        ptwFeaturesFile = os.path.join(t_arg.dspath, t_arg.pathwayfeature)
    else:
        ptwFeaturesFile = None

    if t_arg.useLabelComponentFeatures:
        print('\t>> Retreiving labels components mapping file from: {0:s}'.format(t_arg.pathwayfeature))
        labelsComponentsMappingFile = os.path.join(t_arg.ospath, t_arg.pathway_ec)
    else:
        labelsComponentsMappingFile = os.path.join(t_arg.ospath, t_arg.pathway_ec)

    print('\t>> DONE...')

    ##########################################################################################################
    ###################        TRAINING DATA USING MULTI-LABEL LOGISTIC REGRESSION         ###################
    ##########################################################################################################

    if t_arg.train:
        print('\n*** BEGIN TRAINING USING MULTI-LABEL LEARNING...')

        if t_arg.adjust_by_similarity:
            print('\t>> Retreiving items similarity score matrix file from: {0:s}'.format(t_arg.pathway_similarity))
            if t_arg.similarity_type == "sw":
                similarityFile = t_arg.pathway_similarity + '_sw.pkl'
            elif t_arg.similarity_type == "chi2":
                similarityFile = t_arg.pathway_similarity + '_chi2.pkl'
            elif t_arg.similarity_type == "cos":
                similarityFile = t_arg.pathway_similarity + '_cos.pkl'
            elif t_arg.similarity_type == "rbf":
                similarityFile = t_arg.pathway_similarity + '_rbf.pkl'
            similarityScoreFile = os.path.join(t_arg.ospath, similarityFile)
        else:
            similarityScoreFile = None

        nSample = t_arg.nsample
        fObject = _datasetType(t_arg)
        dataFileName = fObject + '_' + str(nSample)

        print('\t>> Constructing a training dataset from: {0:s} and {1:s}'.format(dataFileName + '_X.pkl',
                                                                                  dataFileName + '_y.pkl'))
        file = dataFileName + '_values.pkl'

        if t_arg.save_prepared_dataset is True:
            value_lst = PrepareDataset(dataId=data_id, fObject=fObject, nSample=nSample,
                                       useAllLabels=t_arg.all_classes, trainSize=t_arg.train_size,
                                       valSize=t_arg.val_size, datasetPath=t_arg.dspath)
            SaveData(data=value_lst, fname=file, savepath=t_arg.dspath, tag='prepared dataset')
        else:
            value_lst = LoadData(fname=file, loadpath=t_arg.dspath, tag='prepared dataset')

        print('\t>> Building multi-label logistic regression train...')

        if t_arg.grid == True:
            alpha = np.logspace(np.log10(0.0001), np.log10(1), num=5)
            l1_ratio = np.logspace(np.log10(0.15), np.log10(1), num=5)
            sigma = np.linspace(1.0, 3.0, num=5)
            if t_arg.adjust_by_similarity:
                alpha = np.logspace(np.log10(0.01), np.log10(1), num=5)
        else:
            alpha = t_arg.alpha
            l1_ratio = t_arg.l1_ratio
            sigma = t_arg.sigma

        clf = mlLGPR(classes=value_lst[0], classLabelsIds=value_lst[1],
                     labelsComponentsFile=labelsComponentsMappingFile,
                     itemPrintFeaturesFile=ptwFeaturesFile,
                     similarityScoreFile=similarityScoreFile,
                     scaleFeature=t_arg.scale_feature,
                     sMethod=t_arg.norm_op, binarizeAbundance=t_arg.binarize,
                     grid=t_arg.grid, useReacEvidenceFeatures=t_arg.useReacEvidenceFeatures,
                     usePossibleClassFeatures=t_arg.usePossibleClassFeatures,
                     useItemEvidenceFeatures=t_arg.useItemEvidenceFeatures,
                     useLabelComponentFeatures=t_arg.useLabelComponentFeatures,
                     nTotalComponents=value_lst[2], nTotalClassLabels=value_lst[3],
                     nTotalEvidenceFeatures=value_lst[4],
                     nTotalClassEvidenceFeatures=value_lst[5],
                     penalty=t_arg.penalty, adjustCoef=t_arg.adjust_by_similarity,
                     coef_similarity_type=t_arg.similarity_type,
                     customFit=t_arg.customFit, useClipping=t_arg.useClipping, alpha=alpha,
                     l1_ratio=l1_ratio, sigma=sigma, fit_intercept=t_arg.fit_intercept,
                     max_inner_iter=t_arg.max_inner_iter, nEpochs=t_arg.nEpochs,
                     nBatches=t_arg.nBatches, testInterval=t_arg.test_interval,
                     shuffle=t_arg.shuffle, adaptive_beta=t_arg.adaptive_beta,
                     threshold=t_arg.threshold, learning_rate=t_arg.learning_rate,
                     eta0=t_arg.eta0, power_t=t_arg.power_t,
                     random_state=t_arg.random_state, n_jobs=t_arg.n_jobs)
        print('\t\t## The following parameters are applied:\n\t\t\t{0}'.format(clf.print_arguments()),
              file=sys.stderr)
        print('\t>> train...')
        startTime = time.time()
        clf.fit(X_file=os.path.join(t_arg.dspath, value_lst[6]),
                y_file=os.path.join(t_arg.dspath, value_lst[7]),
                XdevFile=os.path.join(t_arg.dspath, value_lst[8]),
                ydevFile=os.path.join(t_arg.dspath, value_lst[9]), subSample=t_arg.sub_sample,
                subSampleShuffle=t_arg.shuffle, subsampleSize=t_arg.sub_sample_size, savename=t_arg.mllr,
                savepath=t_arg.mdpath)

        if t_arg.adjust_by_similarity:
            baseName = 'mllg_time_fu'
        else:
            if t_arg.l1_ratio == 1:
                baseName = 'mllg_time_l1'
            elif t_arg.l1_ratio == 0:
                baseName = 'mllg_time_l2'
            else:
                baseName = 'mllg_time_en'
        elapsedTime = str((baseName, time.time() - startTime))
        saveTimeFilename = baseName + '.txt'
        SaveData(data=elapsedTime, fname=saveTimeFilename, savepath=t_arg.rspath, tag='time performance', mode='w',
                 wString=True)
        if clf.grid:
            gridPara = clf.get_best_grid_params()
            print('\t\t## Best hyper-parameters with counts: ({0}, {1})'.format(gridPara[0], gridPara[1]))
        print('\t>> DONE...')

    ##########################################################################################################
    ###################      PERFORMANCE EVALUATION USING MULTI-LABEL LEARNING MODEL       ###################
    ##########################################################################################################

    if t_arg.evaluate:
        print('\n*** EVALUATING THE MULTI-LABEL LEARNING MODEL...')
        print('\t>> Loading pre-trained multi-label train...')
        modelFileName = t_arg.model
        clf = LoadData(fname=modelFileName, loadpath=t_arg.mdpath,
                       tag='the multi-label logistic regression train')
        if clf.grid:
            gridPara = clf.best_grid_params()
            print('\t\t## Best hyper parameters: (l1_ratio: {0}, alpha: {1})'.format(gridPara[0], gridPara[1]))
        clf.itemPrintFeaturesFile = ptwFeaturesFile
        clf.labelsComponentsFile = labelsComponentsMappingFile
        clf.nBatches = t_arg.nBatches
        clf.n_jobs = t_arg.n_jobs

        saveFileName = t_arg.score_file
        saveFileName = _saveFileName(clf, saveFileName)
        saveFileName = saveFileName + '.txt'

        SaveData(data='', fname=saveFileName, savepath=t_arg.rspath, tag='detailed scores', mode='w',
                 wString=True)

        ### Synset Dataset
        fObject = _datasetType(t_arg)
        dataFileName = fObject + '_' + str(t_arg.nsample)
        file = dataFileName + '_values.pkl'
        value_lst = LoadData(fname=file, loadpath=t_arg.dspath, tag='prepared dataset')
        print('\t>> Computing scores for: {0:s}...'.format(value_lst[6]))
        Score(clf=clf, X_file=os.path.join(t_arg.dspath, value_lst[6]),
              y_file=os.path.join(t_arg.dspath, value_lst[7]),
              mode='a', fname=saveFileName, savepath=t_arg.rspath)

        print('\t>> Computing scores for: {0:s}...'.format(value_lst[8]))
        Score(clf=clf, X_file=os.path.join(t_arg.dspath, value_lst[8]),
              y_file=os.path.join(t_arg.dspath, value_lst[9]),
              mode='a', fname=saveFileName, savepath=t_arg.rspath)

        print('\t>> Computing scores for: {0:s}...'.format(value_lst[10]))
        Score(clf=clf, X_file=os.path.join(t_arg.dspath, value_lst[10]),
              y_file=os.path.join(t_arg.dspath, value_lst[11]),
              mode='a', fname=saveFileName, savepath=t_arg.rspath)

        ### Metagenomics Dataset
        dataFileName = t_arg.metegenomics_dataset + '_' + str(418)
        X_file = os.path.join(t_arg.dspath, dataFileName + '_X.pkl')
        y_file = os.path.join(t_arg.dspath, dataFileName + '_y.pkl')
        print('\t>> Computing scores for: {0:s}...'.format(dataFileName + '_X.pkl'))
        Score(clf=clf, X_file=X_file, y_file=y_file, loadBatch=True, mode='a',
              fname=saveFileName, savepath=t_arg.rspath)

        ### Gold Dataset
        dataFileName = t_arg.goldendataset_ptw_ec + '_' + str(63)
        X_file = os.path.join(t_arg.dspath, dataFileName + '_X.pkl')
        y_file = os.path.join(t_arg.dspath, dataFileName + '_y.pkl')
        print('\t>> Computing scores for: {0:s}...'.format(dataFileName + '_X.pkl'))
        Score(clf=clf, X_file=X_file, y_file=y_file, loadBatch=True, mode='a',
              fname=saveFileName, savepath=t_arg.rspath)
        print('\t>> Computing scores for: {0:s}...'.format(dataFileName + '_X.pkl'))
        Score(clf=clf, X_file=X_file, y_file=y_file, loadBatch=True, sixDB=True, mode='a',
              fname=saveFileName, savepath=t_arg.rspath)

        if t_arg.use_tCriterion:
            adaptive_beta_series = np.linspace(start=0.01, stop=1, num=20)
            for b in adaptive_beta_series:
                clf.adaptive_beta = b
                nSample = t_arg.nsample
                fObject = _datasetType(t_arg)
                dataFileName = fObject + '_' + str(nSample)
                file = dataFileName + '_values.pkl'
                value_lst = LoadData(fname=file, loadpath=t_arg.dspath, tag='prepared dataset')
                print('\t>> Computing scores for: {0:s}...'.format(value_lst[10]))
                Score(clf=clf, X_file=os.path.join(t_arg.dspath, value_lst[10]),
                      y_file=os.path.join(t_arg.dspath, value_lst[11]), applyTCriterion=t_arg.use_tCriterion,
                      mode='a', fname=saveFileName, savepath=t_arg.rspath)

                ### Metagenomics Dataset
                dataFileName = t_arg.metegenomics_dataset + '_' + str(418)
                X_file = os.path.join(t_arg.dspath, dataFileName + '_X.pkl')
                y_file = os.path.join(t_arg.dspath, dataFileName + '_y.pkl')
                print('\t>> Computing scores for: {0:s}...'.format(dataFileName + '_X.pkl'))
                Score(clf=clf, X_file=X_file, y_file=y_file, applyTCriterion=t_arg.use_tCriterion,
                      loadBatch=True, mode='a', fname=saveFileName, savepath=t_arg.rspath)

                ### Gold Dataset
                dataFileName = t_arg.goldendataset_ptw_ec + '_' + str(63)
                X_file = os.path.join(t_arg.dspath, dataFileName + '_X.pkl')
                y_file = os.path.join(t_arg.dspath, dataFileName + '_y.pkl')
                print('\t>> Computing scores for: {0:s}...'.format(dataFileName + '_X.pkl'))
                Score(clf=clf, X_file=X_file, y_file=y_file, applyTCriterion=t_arg.use_tCriterion,
                      loadBatch=True, mode='a', fname=saveFileName, savepath=t_arg.rspath)
                print('\t>> Computing scores for: {0:s}...'.format(dataFileName + '_X.pkl'))
                Score(clf=clf, X_file=X_file, y_file=y_file, applyTCriterion=t_arg.use_tCriterion,
                      loadBatch=True, sixDB=True, mode='a', fname=saveFileName, savepath=t_arg.rspath)

        print('\t>> DONE...')

    ##########################################################################################################
    ###################         PREDICTING LABELS USING MULTI-LABEL LEARNING MODEL         ###################
    ##########################################################################################################

    if t_arg.predict:
        print('\n*** PREDICTING USING MULTI-LABEL LOGISTIC REGRESSION...')
        print('\t>> Loading pre-trained multi-label train...')
        modelFileName = t_arg.model
        clf = LoadData(fname=modelFileName, loadpath=t_arg.mdpath,
                       tag='the multi-label logistic regression train')
        if clf.grid:
            gridPara = clf.best_grid_params()
            print('\t\t## Best hyper parameters: (l1_ratio: {0}, alpha: {1})'.format(gridPara[0], gridPara[1]))
        clf.itemPrintFeaturesFile = ptwFeaturesFile
        clf.labelsComponentsFile = labelsComponentsMappingFile
        clf.nBatches = t_arg.nBatches
        clf.n_jobs = t_arg.n_jobs
        saveFileName = t_arg.predict_file
        saveFileName = _saveFileName(clf, saveFileName)

        ### Synset Dataset
        # print('\t>> Predicting class labels for: {0:s}...'.format(value_lst[6]))
        # X_file = os.path.join(t_arg.dspath, value_lst[6])
        # y_pred = clf.Predict(X_file=X_file)
        # labels = clf.mlb.inverse_transform(y_pred)
        # SaveData(data='>> Predicted class labels for: {0:s}...\n'.format(value_lst[6]),
        #          fname=saveFileName, savepath=t_arg.rspath, mode='a',
        #          wString=True, printTag=False)
        # for sidx in np.arange(len(labels)):
        #     SaveData(
        #         data='\t{0})- Total set of pathways for the sample {0}: {1}...\n'.format(sidx + 1, len(labels[sidx])),
        #         fname=saveFileName, savepath=t_arg.rspath, mode='a',
        #         wString=True, printTag=False)
        #     for pid in labels[sidx]:
        #         SaveData(data='\t\t' + pid + '\n', fname=saveFileName, savepath=t_arg.rspath,
        #                  mode='a', wString=True, printTag=False)
        #
        # print('\t>> Predicting class labels for: {0:s}...'.format(value_lst[8]))
        # X_file = os.path.join(t_arg.dspath, value_lst[8])
        # y_pred = clf.Predict(X_file=X_file)
        # labels = clf.mlb.inverse_transform(y_pred)
        # SaveData(data='>> Predicted class labels for: {0:s}...\n'.format(value_lst[8]),
        #          fname=saveFileName, savepath=t_arg.rspath, mode='a',
        #          wString=True, printTag=False)
        # for sidx in np.arange(len(labels)):
        #     SaveData(
        #         data='\t{0})- Total set of pathways for the sample {0}: {1}...\n'.format(sidx + 1, len(labels[sidx])),
        #         fname=saveFileName, savepath=t_arg.rspath, mode='a',
        #         wString=True, printTag=False)
        #     for pid in labels[sidx]:
        #         SaveData(data='\t\t' + pid + '\n', fname=saveFileName, savepath=t_arg.rspath,
        #                  mode='a', wString=True, printTag=False)
        #
        # print('\t>> Predicting class labels for: {0:s}...'.format(value_lst[10]))
        # X_file = os.path.join(t_arg.dspath, value_lst[10])
        # y_pred = clf.Predict(X_file=X_file)
        # labels = clf.mlb.inverse_transform(y_pred)
        # SaveData(data='>> Predicted class labels for: {0:s}...\n'.format(value_lst[10]),
        #          fname=saveFileName, savepath=t_arg.rspath, mode='a',
        #          wString=True, printTag=False)
        # for sidx in np.arange(len(labels)):
        #     SaveData(
        #         data='\t{0})- Total set of pathways for the sample {0}: {1}...\n'.format(sidx + 1, len(labels[sidx])),
        #         fname=saveFileName, savepath=t_arg.rspath, mode='a',
        #         wString=True, printTag=False)
        #     for pid in labels[sidx]:
        #         SaveData(data='\t\t' + pid + '\n', fname=saveFileName, savepath=t_arg.rspath,
        #                  mode='a', wString=True, printTag=False)

        ### Metagenomics Dataset
        dataFileName = t_arg.metegenomics_dataset + '_' + str(418)
        print('\t>> Predicting class labels for: {0:s}...'.format(dataFileName + '_X.pkl'))
        _report(clf, dataFileName, saveFileName, t_arg)

        ### Gold Dataset
        dataFileName = t_arg.goldendataset_ptw_ec + '_' + str(63)
        print('\t>> Predicting class labels for: {0:s}...'.format(dataFileName + '_X.pkl'))
        _report(clf, dataFileName, saveFileName, t_arg, tag='gold')
        print('\t>> DONE...')


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
