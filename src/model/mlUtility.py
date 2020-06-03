'''
This is a utility module for the mlLGPR file.
'''

import os
import sys
import traceback
from collections import OrderedDict

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import coverage_error, label_ranking_loss, label_ranking_average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import jaccard_similarity_score, hamming_loss

try:
    import cPickle as pkl
except:
    import pickle as pkl

EPSILON = np.finfo(np.float).eps

def LoadItemFeatures(fname, components=True):
    try:
        with open(fname, 'rb') as f_in:
            while True:
                itemFeatures = pkl.load(f_in)
                if components:
                    if type(itemFeatures) is tuple:
                        break
                else:
                    if type(itemFeatures) is np.ndarray:
                        break
        return itemFeatures
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(fname), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def LoadYFile(nSamples, y_file):
    y_true = list()
    sidx = 0
    try:
        with open(y_file, 'rb') as f_in:
            while sidx < nSamples:
                tmp = pkl.load(f_in)
                if type(tmp) is tuple:
                    y_true.append(tmp[0])
                    sidx += 1
                if sidx == nSamples:
                    break
        return y_true
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(y_file), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def LoadyData(fname, loadpath, tag='labels data'):
    try:
        print('\t\t## Loading {0:s} from: {1:s}'.format(tag, fname))
        fname = os.path.join(loadpath, fname)
        with open(fname, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is tuple and len(data) == 2:
                    y, sample_ids = data
                    break
            return y, sample_ids
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(fname), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def LoadData(fname, loadpath, tag='data'):
    '''
    Save edus into file
    :param data:
    :param loadpath: load file from a path
    :type fname: string
    :param fname:
    '''
    try:
        fname = os.path.join(loadpath, fname)
        with open(fname, 'rb') as f_in:
            data = pkl.load(f_in)
        print('\t\t## Loading {0:s} from: {1:s}'.format(tag, fname))
        return data
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(fname), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def SaveData(data, fname, savepath, tag='', mode='wb', wString=False, printTag=True):
    '''
    Save data into file
    :param mode:
    :param printTag:
    :param data: the data file to be saved
    :param savepath: location of the file
    :param fname: name of the file
    '''
    try:
        file = fname
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        fname = os.path.join(savepath, fname)
        if printTag:
            print('\t\t## Storing {0:s} into the file: {1:s}'.format(tag, file))
        with open(file=fname, mode=mode) as fout:
            if not wString:
                pkl.dump(data, fout)
            elif wString:
                fout.write(data)
    except Exception as e:
        print('\t\t## The file {0:s} can not be saved'.format(fname), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def ReverseIdx(value2idx):
    idx2value = {}
    for key, value in value2idx.items():
        idx2value.update({value: key})
    return idx2value


def ExtractLabels(dataId, useAllLabels, nSample, fName, datasetPath):
    classlabels = list()
    classLabelsIds = OrderedDict()
    if useAllLabels:
        classlabels = [item for item, idx in dataId.items()]
        classLabelsIds = dataId
    else:
        fName = fName + '_' + str(nSample) + '_mapping.txt'
        fileClasslabels = os.path.join(datasetPath, fName)
        with open(fileClasslabels, 'r') as f_in:
            for tmp in f_in:
                try:
                    if not tmp.startswith('#'):
                        label = tmp.split('\t')[0]
                        classlabels.append(label)
                        classLabelsIds.update({label: dataId[label]})
                except IOError:
                    break
    return classLabelsIds, classlabels


def PrepareDataset(dataId, useAllLabels=False, trainSize=0.8, valSize=0.2, datasetPath='', 
                   X_name = "synset_X.pkl" , y_name = "synset_y.pkl", file_name="synset"):
    X = os.path.join(datasetPath, X_name)
    y = os.path.join(datasetPath, y_name)
    classLabelsIds = OrderedDict()
    labels = OrderedDict()
    classlabels = list()

    if useAllLabels:
        classlabels = [item for item, idx in dataId.items()]
        classLabelsIds = dataId

    with open(X, 'rb') as f_in:
        while True:
            try:
                data = pkl.load(f_in)
                if type(data) is tuple and len(data) == 10:
                    nTotalSamples = data[1]
                    nTotalComponents = data[3]
                    nTotalClassLabels = data[5]
                    nTotalEvidenceFeatures = data[7]
                    nTotalClassEvidenceFeatures = data[9]
                    break
            except IOError:
                break

    with open(y, 'rb') as f_in:
        while True:
            try:
                data = pkl.load(f_in)
                if type(data) is tuple:
                    y_true, sample_ids = data
                    for sidx in np.arange(y_true.shape[0]):
                        for label in y_true[sidx]:
                            if label not in labels:
                                labels.update({label: [sidx]})
                                if not useAllLabels:
                                    classlabels.append(label)
                                    classLabelsIds.update({label: dataId[label]})
                            else:
                                labels[label].extend([sidx])
                    break
            except IOError:
                break

    trainSamples = list()
    devSamples = list()
    testSamples = list()
    for label in labels.items():
        trn = np.random.choice(a=label[1], size=int(np.ceil(trainSize * len(label[1]))), replace=False)
        dev = np.random.choice(a=trn, size=int(np.ceil(valSize * len(trn))), replace=False)
        if len(dev) > 0:
            trn = [x for x in trn if x not in dev]
        tst = [i for i in label[1] if i not in trn and i not in dev]
        for sidx in trn:
            if sidx not in trainSamples and sidx not in devSamples and sidx not in testSamples:
                trainSamples.append(sidx)
        for sidx in dev:
            if sidx not in trainSamples and sidx not in devSamples and sidx not in testSamples:
                devSamples.append(sidx)
        for sidx in tst:
            if sidx not in trainSamples and sidx not in devSamples and sidx not in testSamples:
                testSamples.append(sidx)

    trainSamples = np.unique(trainSamples)
    devSamples = np.unique(devSamples)
    testSamples = np.unique(testSamples)

    ## X training set and y label set
    fileDesc = '# The training set is stored as X\n'
    XtrainFile = file_name + '_Xtrain' + '.pkl'
    SaveData(data=fileDesc, fname=XtrainFile, savepath=datasetPath, tag='X training samples', mode='w+b')
    SaveData(data=('nTotalSamples', len(trainSamples),
                   'nTotalComponents', nTotalComponents,
                   'nTotalClassLabels', nTotalClassLabels,
                   'nTotalEvidenceFeatures', nTotalEvidenceFeatures,
                   'nTotalClassEvidenceFeatures', nTotalClassEvidenceFeatures),
             fname=XtrainFile, savepath=datasetPath, mode='a+b', printTag=False)
    fileDesc = '# This file stores the labels of the training set as (y, ids)\n'
    ytrainFile = file_name + '_ytrain' + '.pkl'
    SaveData(data=fileDesc, fname=ytrainFile, savepath=datasetPath, tag='y training samples', mode='w+b')

    ## X development set and y label set
    fileDesc = '# The development set is stored as X\n'
    XdevFile = file_name + '_Xdev' + '.pkl'
    SaveData(data=fileDesc, fname=XdevFile, savepath=datasetPath, tag='X development samples', mode='w+b')
    SaveData(data=('nTotalSamples', len(devSamples),
                   'nTotalComponents', nTotalComponents,
                   'nTotalClassLabels', nTotalClassLabels,
                   'nTotalEvidenceFeatures', nTotalEvidenceFeatures,
                   'nTotalClassEvidenceFeatures', nTotalClassEvidenceFeatures),
             fname=XdevFile, savepath=datasetPath, mode='a+b', printTag=False)
    fileDesc = '# This file stores the labels of the development set as (y, ids)\n'
    ydevFile = file_name + '_ydev' + '.pkl'
    SaveData(data=fileDesc, fname=ydevFile, savepath=datasetPath, tag='y development samples', mode='w+b')

    ## X test set and y label set
    fileDesc = '# The test set is stored as X\n'
    XtestFile = file_name + '_Xtest' + '.pkl'
    SaveData(data=fileDesc, fname=XtestFile, savepath=datasetPath, tag='X test samples', mode='w+b')
    SaveData(data=('nTotalSamples', len(testSamples),
                   'nTotalComponents', nTotalComponents,
                   'nTotalClassLabels', nTotalClassLabels,
                   'nTotalEvidenceFeatures', nTotalEvidenceFeatures,
                   'nTotalClassEvidenceFeatures', nTotalClassEvidenceFeatures),
             fname=XtestFile, savepath=datasetPath, mode='a+b', printTag=False)
    fileDesc = '# This file stores the labels of the test set as (y, ids)\n'
    ytestFile = file_name + '_ytest' + '.pkl'
    SaveData(data=fileDesc, fname=ytestFile, savepath=datasetPath, tag='y test samples', mode='w+b')

    with open(X, 'rb') as f_in:
        sidx = 0
        while True:
            try:
                item = pkl.load(f_in)
                if type(item) is np.ndarray:
                    if sidx in trainSamples:
                        SaveData(data=item, fname=XtrainFile, savepath=datasetPath, mode='a+b',
                                 printTag=False)
                        SaveData(data=(y_true[sidx], sample_ids[sidx]), fname=ytrainFile, savepath=datasetPath,
                                 mode='a+b',
                                 printTag=False)
                    elif sidx in devSamples:
                        SaveData(data=item, fname=XdevFile, savepath=datasetPath, mode='a+b',
                                 printTag=False)
                        SaveData(data=(y_true[sidx], sample_ids[sidx]), fname=ydevFile, savepath=datasetPath,
                                 mode='a+b',
                                 printTag=False)
                    else:
                        SaveData(data=item, fname=XtestFile, savepath=datasetPath, mode='a+b',
                                 printTag=False)
                        SaveData(data=(y_true[sidx], sample_ids[sidx]), fname=ytestFile, savepath=datasetPath,
                                 mode='a+b',
                                 printTag=False)
                    sidx += 1
                if sidx == nTotalSamples:
                    break
            except IOError:
                break

    return [classlabels, classLabelsIds, nTotalComponents, nTotalClassLabels,
            nTotalEvidenceFeatures, nTotalClassEvidenceFeatures,
            XtrainFile, ytrainFile, XdevFile, ydevFile, XtestFile, ytestFile]


def DetailHeaderFile(header=True):
    if header:
        desc = "{0}{1}\n".format("# ", "=" * 52)
        desc += "# Description of attributes in this file.\n"
        desc += "{0}{1}\n".format("# ", "=" * 52)
        desc += "# Sample-id: a unique identifier of the sample.\n"
        desc += "# Total Pathways: total set of pathways for the \n#\tassociated sample.\n"
        desc += "# Pathway Frame-id: a unique identifier of the \n#\tpathway (as in the PGDB).\n"
        desc += "# Pathway Score: a number from 0-1 indicating the \n#\tstrength of the evidence " \
                "supporting the \n#\tinference of this pathway, where 1.0 means \n#\tvery strong evidence.\n"
        desc += "# Pathway Abundance: the abundance of the pathway \n#\tgiven the abundance values of the " \
                "enzymes \n#\tfor this pathway in the annotation file.\n"
        desc += "{0}{1}\n\n".format("# ", "=" * 52)
    else:
        desc = "{0}\n".format("-" * 111)
        desc += " {1:10}{0}{2:15}{0}{3:40}{0}{4:15}{0}{5:18}\n".format(" | ", "Sample-id", "Total Pathways",
                                                                       "Pathway Frame-id", "Pathway Score",
                                                                       "Pathway Abundance")
        desc += "{0}\n".format("-" * 111)
    return desc


def ListHeaderFile(header=True):
    if header:
        desc = "{0}{1}\n".format("# ", "=" * 47)
        desc += "# Description of attributes in this file.\n"
        desc += "{0}{1}\n".format("# ", "=" * 47)
        desc += "# Sample-id: a unique identifier of the sample.\n"
        desc += "# Total Pathways: total set of pathways for the \n#\tassociated sample.\n"
        desc += "# Pathway Frame-id: a unique identifier of the \n#\tpathway (as in the PGDB).\n"
        desc += "{0}{1}\n\n".format("# ", "=" * 47)
    else:
        desc = "{0}\n".format("-" * 60)
        desc += " {1:10}{0}{2:15}{0}{3:40}\n".format(" | ", "Sample-id", "Total Pathways", "Pathway Frame-id")
        desc += "{0}\n".format("-" * 60)
    return desc


def ComputePathwayAbundance(X_file, labelsComponentsFile, classLabelsIds, mlbClasses,
                            nBatches, nTotalComponents):
    '''
    Predict a list of pathways for a given set
        of features extracted from input set

    :param forTraining:
    :type X_test: list
    :param X_test: feature list generated by DataObject, shape array-like
    '''

    labelsComponents = LoadItemFeatures(fname=labelsComponentsFile)[0]

    with open(X_file, 'rb') as f_in:
        while True:
            tmp = pkl.load(f_in)
            if type(tmp) is tuple and len(tmp) == 10:
                nSamples = tmp[1]
                exp2labelsAbun = np.empty(shape=(nSamples, len(mlbClasses)))
                batchsize = int(nSamples / nBatches)
                break

        pred_idx = 0
        for batch in np.arange(nBatches):
            start_idx = 0
            if batch == nBatches - 1:
                batchsize = nSamples - (batch * batchsize)
            final_idx = start_idx + batchsize
            compFeature = np.empty(shape=(batchsize, nTotalComponents))
            print('\t\t## Generating report for {2:d} samples: {0:d} batch (out of {1:d})'.format(
                batch + 1, nBatches, batchsize))
            while start_idx < final_idx:
                try:
                    tmp = pkl.load(f_in)
                    if type(tmp) is np.ndarray:
                        tmp = tmp[:, : nTotalComponents]
                        compFeature[start_idx] = np.reshape(tmp, (tmp.shape[1],))
                        start_idx += 1
                except Exception as e:
                    print(traceback.print_exc())
                    raise e

            for classIdx, classLabel in enumerate(mlbClasses):
                refLabel = labelsComponents[classLabelsIds[classLabel], :]
                tmp = np.copy(compFeature)
                compFeature[:, np.where(refLabel == 0)[0]] = 0
                preAbun = np.divide(compFeature, refLabel + EPSILON)
                np.nan_to_num(preAbun, copy=False)
                exp2labelsAbun[pred_idx:pred_idx + final_idx, classIdx] = np.sum(preAbun, axis=1)
                compFeature = np.copy(tmp)
            pred_idx += final_idx
    return exp2labelsAbun


def Score(clf, X_file, y_file, applyTCriterion=False, loadBatch=False, sixDB=False, mode='w',
          fname='results.txt', savepath=''):
    if not clf.fit:
        raise Exception("This instance is not fitted yet. Call 'fit' with "
                        "appropriate arguments before using this method.")
    y_pred = clf.predict(X_file=X_file, applyTCriterion=applyTCriterion)

    if not loadBatch:
        y_true = LoadYFile(nSamples=y_pred.shape[0], y_file=y_file)
    else:
        y_true = LoadYFile(nSamples=1, y_file=y_file)[0]

    idx_lst = [1]
    item_lst = [np.size(y_true)]
    tag = 'samples'

    if sixDB:
        tag = 'sample'
        if sys.platform != 'darwin':
            item_lst = ['EcoCyc', 'HumanCyc', 'AraCyc', 'YeastCyc', 'LeishCyc', 'TrypanoCyc']
            idx_lst = [5, 0, 3, 4, 1, 2]
        else:
            item_lst = ['EcoCyc', 'HumanCyc', 'AraCyc', 'YeastCyc', 'LeishCyc', 'TrypanoCyc']
            idx_lst = [2, 1, 4, 0, 5, 3]

    SaveData(data='*** COMPUTING scores for: {0:s}...\n'.format(X_file), fname=fname, savepath=savepath,
             mode='a', wString=True, printTag=False)
    for i, idx in enumerate(idx_lst):
        y = y_true
        y_hat = y_pred

        if sixDB:
            y = y_true[idx]
            y_hat = y_pred[idx]
            y = y.reshape((1, y.shape[0]))
            y_hat = np.reshape(y_hat, (1, len(y_hat)))
            print('\t>> Scores for {0:s} {1:s}...'.format(str(item_lst[i]), tag))
            SaveData(data='\t>> Scores for {0:s} {1:s}...\n'.format(str(item_lst[i]), tag),
                     fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)

        if applyTCriterion:
            print('\t\t## Scores for beta: {0:f}...\n'.format(clf.beta))
            SaveData(data='\t\t## Scores for beta: {0:f}...\n'.format(clf.beta),
                     fname=fname, savepath=savepath, mode='a', wString=True, printTag=False)
        else:
            print('\t\t## Scores for beta: False...')
            SaveData(data='\t\t## Scores for beta: False...\n', fname=fname, savepath=savepath, mode=mode,
                     wString=True, printTag=False)

        y = clf.mlb.fit_transform(y)

        hl_samples = hamming_loss(y, y_hat)
        print('\t\t1)- Hamming-Loss score: {0:.4f}'.format(hl_samples))
        SaveData(data='\t\t1)- Hamming-Loss score: {0:.4f}\n'.format(hl_samples),
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)

        print('\t\t2)- Precision...')
        pr_samples_average = precision_score(y, y_hat, average='samples')
        pr_samples_micro = precision_score(y, y_hat, average='micro')
        pr_samples_macro = precision_score(y, y_hat, average='macro')
        print('\t\t\t--> Average sample precision: {0:.4f}'.format(pr_samples_average))
        print('\t\t\t--> Average micro precision: {0:.4f}'.format(pr_samples_micro))
        print('\t\t\t--> Average macro precision: {0:.4f}'.format(pr_samples_macro))
        SaveData(data='\t\t2)- Precision...\n',
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> Average sample precision: {0:.4f}\n'.format(pr_samples_average),
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> Average micro precision: {0:.4f}\n'.format(pr_samples_micro), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> Average macro precision: {0:.4f}\n'.format(pr_samples_macro), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)

        print('\t\t3)- Recall...')
        rc_samples_average = recall_score(y, y_hat, average='samples')
        rc_samples_micro = recall_score(y, y_hat, average='micro')
        rc_samples_macro = recall_score(y, y_hat, average='macro')
        print('\t\t\t--> Average sample recall: {0:.4f}'.format(rc_samples_average))
        print('\t\t\t--> Average micro recall: {0:.4f}'.format(rc_samples_micro))
        print('\t\t\t--> Average macro recall: {0:.4f}'.format(rc_samples_macro))
        SaveData(data='\t\t3)- Recall...\n',
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> Average sample recall: {0:.4f}\n'.format(rc_samples_average), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> Average micro recall: {0:.4f}\n'.format(rc_samples_micro), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> Average macro recall: {0:.4f}\n'.format(rc_samples_macro), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)

        print('\t\t4)- F1-score...')
        f1_samples_average = f1_score(y, y_hat, average='samples')
        f1_samples_micro = f1_score(y, y_hat, average='micro')
        f1_samples_macro = f1_score(y, y_hat, average='macro')
        print('\t\t\t--> Average sample f1-score: {0:.4f}'.format(f1_samples_average))
        print('\t\t\t--> Average micro f1-score: {0:.4f}'.format(f1_samples_micro))
        print('\t\t\t--> Average macro f1-score: {0:.4f}'.format(f1_samples_macro))
        SaveData(data='\t\t4)- F1-score...\n',
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> Average sample f1-score: {0:.4f}\n'.format(f1_samples_average), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> Average micro f1-score: {0:.4f}\n'.format(f1_samples_micro), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> Average macro f1-score: {0:.4f}\n'.format(f1_samples_macro), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)

        print('\t\t5)- Jaccard similarity score...')
        js_samples_normalize = jaccard_similarity_score(y, y_hat)
        js_samples_not_normalize = jaccard_similarity_score(y, y_hat, normalize=False)
        print('\t\t\t--> Jaccard similarity score (normalized): {0:.4f}'.format(js_samples_normalize))
        print('\t\t\t--> Jaccard similarity score (not-normalized): {0:.4f}'.format(js_samples_not_normalize))
        SaveData(data='\t\t5)- Jaccard similarity score...\n',
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(
            data='\t\t\t--> Jaccard similarity score (normalized): {0:.4f}\n'.format(js_samples_normalize),
            fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(
            data='\t\t\t--> Jaccard similarity score (not-normalized): {0:.4f}\n'.format(js_samples_not_normalize),
            fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)

        ce_samples = coverage_error(y, y_hat)
        print('\t\t6)- Coverage error score: {0:.4f}'.format(ce_samples))
        SaveData(data='\t\t6)- Coverage error score: {0:.4f}\n'.format(ce_samples),
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)

        lrl_samples = label_ranking_loss(y, y_hat)
        print('\t\t7)- Ranking loss score: {0:.4f}'.format(lrl_samples))
        SaveData(data='\t\t7)- Ranking loss score: {0:.4f}\n'.format(lrl_samples),
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)

        lrap_samples = label_ranking_average_precision_score(y, y_hat)
        print('\t\t8)- Label ranking average precision score: {0:.4f}'.format(lrap_samples))
        SaveData(data='\t\t8)- Label ranking average precision score: {0:.4f}\n'.format(lrap_samples),
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)

        print('\t\t9)- Confusion matrix...')
        tn, fp, fn, tp = confusion_matrix(y.flatten(), y_hat.flatten()).ravel()
        print('\t\t\t--> True positive: {0}'.format(tp))
        print('\t\t\t--> True negative: {0}'.format(tn))
        print('\t\t\t--> False positive: {0}'.format(fp))
        print('\t\t\t--> False negative: {0}\n'.format(fn))
        SaveData(data='\t\t9)- Confusion matrix...\n',
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> True positive: {0}\n'.format(tp), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> True negative: {0}\n'.format(tn), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> False positive: {0}\n'.format(fp),
                 fname=fname, savepath=savepath, mode=mode, wString=True, printTag=False)
        SaveData(data='\t\t\t--> False negative: {0}\n'.format(fn), fname=fname,
                 savepath=savepath, mode=mode, wString=True, printTag=False)
