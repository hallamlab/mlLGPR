'''
This file is the main entry used to explore various
features of the input-set.
'''

import os
import os.path
import re
import sys
import time
import traceback
from collections import Counter, OrderedDict, defaultdict
from itertools import chain
from operator import itemgetter

import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from feature.ParseFeaturesList import ExtractFeaturesNames
from matplotlib import pyplot, patches
from model.mlUtility import ComputePathwayAbundance, ListHeaderFile, DetailHeaderFile
from scipy import sparse
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

try:
    import cPickle as pkl
except:
    import pickle as pkl
from sklearn.metrics import precision_recall_fscore_support, hamming_loss, confusion_matrix


###***************************          Accessing Data          ***************************###

def _saveData(data, fname, savepath, tag='', mode='wb', wString=False, printTag=True):
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


def _loadData(fname, loadpath, tag='data'):
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


def _load_X_data(fname, loadpath, properties=True, tag='data'):
    nTotalSamples = 0
    nTotalComponents = 0
    nTotalClassLabels = 0
    nEvidenceFeatures = 0
    nTotalClassEvidenceFeatures = 0

    try:
        print('\t\t## Loading {0:s} from: {1:s}'.format(tag, fname))
        fname = os.path.join(loadpath, fname)
        sidx = 0
        with open(fname, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is tuple and len(data) == 10:
                    nTotalSamples = data[1]
                    nTotalComponents = data[3]
                    nTotalClassLabels = data[5]
                    nEvidenceFeatures = data[7]
                    nTotalClassEvidenceFeatures = data[9]
                    if properties:
                        break
                if not properties:
                    if type(data) is np.ndarray:
                        if sidx == 0:
                            X = np.ndarray(
                                (nTotalSamples, nTotalComponents + nEvidenceFeatures + nTotalClassEvidenceFeatures))
                        if sidx < nTotalSamples:
                            X[sidx] = data
                            sidx += 1
                        if sidx == nTotalSamples:
                            break
        if properties:
            return nTotalSamples, nTotalComponents, nTotalClassLabels, nEvidenceFeatures, nTotalClassEvidenceFeatures
        else:
            return X
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(fname), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def _load_y_data(fname, loadpath, tag='labels data'):
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


def _loadItemFeatures(fname, datasetPath, components=True):
    filePtwfeatures = os.path.join(datasetPath, fname)
    with open(filePtwfeatures, 'rb') as f_in:
        while True:
            itemFeatures = pkl.load(f_in)
            if components:
                if type(itemFeatures) is tuple:
                    break
            else:
                if type(itemFeatures) is np.ndarray:
                    break
    return itemFeatures


def _loadPredictedData(fname, loadpath, tag='data'):
    try:
        fname = os.path.join(loadpath, fname)
        header = False
        newsample = False
        lst_ptwys = list()
        samples_info = OrderedDict()
        print('\t\t## Loading {0:s} from: {1:s}'.format(tag, fname))
        with open(fname, 'r') as f_in:
            for line in f_in.readlines():
                line = line.strip()
                if not (line.startswith('#') or line.startswith('>>') or line.startswith('-') or line == ""):
                    if not header:
                        header = True
                    else:
                        data = line.split('|')
                        if data[0] != "":
                            if newsample:
                                samples_info.update({sidx: [nptwys, lst_ptwys]})
                                lst_ptwys = list()
                                newsample = False
                            sidx = int(data[0].strip())
                            nptwys = int(data[1].strip())
                            lst_ptwys.append(data[2].strip())
                        else:
                            lst_ptwys.append(data[2].strip())
                            newsample = True
        samples_info.update({sidx: [nptwys, lst_ptwys]})
        return samples_info
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(fname), file=sys.stderr)
        print(traceback.print_exc())
        raise e


###***************************             Utilities            ***************************###

def _reverseIdx(value2idx):
    idx2value = {}
    for key, value in value2idx.items():
        idx2value.update({value: key})
    return idx2value


def _datasetType(args):
    if args.ds_type == "syn_ds":
        fObject = args.syntheticdataset_ptw_ec
    elif args.ds_type == "meta_ds":
        fObject = args.metegenomics_dataset
    else:
        fObject = args.goldendataset_ptw_ec
    return fObject


###***************************          Analyzing Data          ***************************###

def _topComponents(idx2value, maxPerColumnXC, nCount, tag=''):
    print('\t>> The top {0} {1} operation for components with values:'.format(nCount, tag))
    sortedIDx = np.argsort(maxPerColumnXC)[::-1]
    ec_lst = [idx2value[idx] for idx in sortedIDx[:nCount] if idx in idx2value]
    val_lst = maxPerColumnXC[sortedIDx[:nCount]]
    print('\t\t## The top {0} components with values (descending):'.format(nCount))
    for idx, ec in enumerate(ec_lst):
        print('\t\t{0})-    EC: {1};    value: {2}'.format(idx + 1, ec, val_lst[idx]))
    ec_lst = list()
    val_lst = list()
    for idx in sortedIDx[::-1]:
        if maxPerColumnXC[idx] != 0:
            ec_lst.append(idx2value[idx])
            val_lst.append(maxPerColumnXC[idx])
            if len(ec_lst) == nCount:
                break
    print('\t\t## The top {0} components with values (ascending):'.format(nCount))
    for idx, ec in enumerate(ec_lst):
        print('\t\t{0})-    EC: {1};    value: {2}'.format(idx + 1, ec, val_lst[idx]))


def _check_baseline_mapping(args, objData, X, X_file, y, nTComponents, threshold=0.5,
                            beta=np.logspace(np.log10(0.1), np.log10(1), num=1), adaptivePrediction=False,
                            onlyPredict=True, report=True):
    ### Useful for creating dataset through finding the frequencies
    X_components = X[:, :nTComponents]
    ecidx2value = _reverseIdx(objData.ec_id)
    true_ecs_per_ptw = _loadData(fname=args.pathway_ec, loadpath=args.ospath)
    # X_components = X_components[5]  # EcoCyc
    # y = y[5]
    # X_components = X_components[0]  # HumanCyc
    # y = y[0]
    # X_components = X_components[3]  # AraCyc
    # y = y[3]
    # X_components = X_components[4] # YeastCyc
    # y = y[4]
    # X_components = X_components[1] # LeishCyc
    # y = y[1]
    # X_components = X_components[2] # TrypanoCyc
    # y = y[2]
    # X_components = X_components.reshape((1, X_components.shape[0]))
    # y = y.reshape((1, y.shape[0]))

    true_ecs_per_ptw = true_ecs_per_ptw[0]
    preprocessing.binarize(X_components, copy=False)
    preprocessing.binarize(true_ecs_per_ptw, copy=False)
    ptwidx2value = _reverseIdx(objData.pathway_id)
    mlb = preprocessing.MultiLabelBinarizer(tuple(objData.pathway_id))
    y = mlb.fit_transform(y)

    startTime = time.time()
    for b in beta:
        result = np.dot(X_components, true_ecs_per_ptw.T)
        result = np.divide(result, np.sum(true_ecs_per_ptw, axis=1))
        np.nan_to_num(result, copy=False)

        if adaptivePrediction:
            maxval = np.max(result, axis=1) * b
            for sidx in np.arange(result.shape[0]):
                result[sidx][result[sidx] >= maxval[sidx]] = 1

        result[result >= threshold] = 1
        result[result != 1] = 0
        y_pred = np.ndarray((result.shape[0],), dtype=np.object)
        for sidx in np.arange(result.shape[0]):
            ptw_lst = list()
            for pidx in np.nonzero(result[sidx])[0]:
                ptw_lst.append(ptwidx2value[pidx])
            y_pred[sidx] = np.unique(ptw_lst)
        y_pred = mlb.fit_transform(y_pred)
        print('\t\t## Inference consumed {0:f} seconds'.format(round(time.time() - startTime, 3)))

        if not onlyPredict:
            if not report:
                if adaptivePrediction:
                    print('\t>> Scores for beta: {0:f}...'.format(b))
                else:
                    print('\t>> Computing scores...')

                pr_s, rc_s, f1_s, _ = precision_recall_fscore_support(y, y_pred, average='samples')
                pr_mi, rc_mi, f1_mi, _ = precision_recall_fscore_support(y, y_pred, average='micro')
                pr_ma, rc_ma, f1_ma, _ = precision_recall_fscore_support(y, y_pred, average='macro')
                hloss = hamming_loss(y, y_pred)

                print('\t\t1)- Hamming loss score using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, hloss))

                print('\t\t2)- Precision...')
                print('\t\t\t--> Average sample precision using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, pr_s))
                print('\t\t\t--> Average micro precision using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, pr_mi))
                print('\t\t\t--> Average macro precision using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, pr_ma))

                print('\t\t3)- Recall...')
                print('\t\t\t--> Average sample recall using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, rc_s))
                print('\t\t\t--> Average micro recall using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, rc_mi))
                print('\t\t\t--> Average macro recall using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, rc_ma))

                print('\t\t4)- F1-score...')
                print('\t\t## Average sample f1-score using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, f1_s))
                print('\t\t## Average micro f1-score using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, f1_mi))
                print('\t\t## Average macro f1-score using threshold {0:.2f}: {1:.4f}'
                      .format(threshold, f1_ma))

                print('\t\t5)- Confusion matrix...')
                tn, fp, fn, tp = confusion_matrix(y.flatten(), y_pred.flatten()).ravel()
                print('\t\t\t--> True positive: {0}'.format(tp))
                print('\t\t\t--> True negative: {0}'.format(tn))
                print('\t\t\t--> False positive: {0}'.format(fp))
                print('\t\t\t--> False negative: {0}'.format(fn))
            else:
                lst_ptws = mlb.inverse_transform(y_pred)
                X_file = os.path.join(args.dspath, X_file)
                labelsComponentsFile = os.path.join(args.ospath, args.pathway_ec)
                pathwayAbun = ComputePathwayAbundance(X_file=X_file, labelsComponentsFile=labelsComponentsFile,
                                                      classLabelsIds=objData.pathway_id, mlbClasses=mlb.classes,
                                                      nBatches=1, nTotalComponents=nTComponents)
                _generateReport(X_file=X_file, lst_ptws=lst_ptws, mlb=mlb, pathwayAbun=pathwayAbun,
                                saveFileListsName='baseline.lists', saveFileDetails="baseline.details",
                                rspath=args.rspath)
    return X_components, ecidx2value, y


def _generateReport(X_file, lst_ptws, mlb, pathwayAbun, saveFileListsName, saveFileDetails, rspath):
    _saveData(data=ListHeaderFile(), fname=saveFileListsName, savepath=rspath,
              tag='class labels', mode='w', wString=True)
    _saveData(data=DetailHeaderFile(), fname=saveFileDetails, savepath=rspath,
              tag='detail class labels', mode='w', wString=True)
    _saveData(data='>> Generate class labels for: {0:s}...\n'.format(X_file),
              fname=saveFileListsName, savepath=rspath, mode='a',
              wString=True, printTag=False)
    _saveData(data='>> Generate class labels for: {0:s}...\n'.format(X_file),
              fname=saveFileDetails, savepath=rspath, mode='a',
              wString=True, printTag=False)
    _saveData(data=ListHeaderFile(header=False), fname=saveFileListsName, savepath=rspath,
              mode='a', wString=True, printTag=False)
    _saveData(data=DetailHeaderFile(header=False), fname=saveFileDetails, savepath=rspath,
              mode='a', wString=True, printTag=False)
    for sidx in np.arange(len(lst_ptws)):
        sampleInit = False
        for pid in lst_ptws[sidx]:
            sampleId = ""
            nLabels = ""
            labelProb = str("{0:4}").format("-" * 4)
            labelAbun = str("{0:.4f}").format(pathwayAbun[sidx, mlb.classes.index(pid)])
            if not sampleInit:
                sampleId = str(sidx + 1)
                nLabels = str(len(lst_ptws[sidx]))
                sampleInit = True
            data = " {1:10}{0}{2:15}{0}{3:40}{0}{4:15}{0}{5:18}\n".format(" | ", sampleId, nLabels,
                                                                          str(pid), labelProb, labelAbun)
            _saveData(data=" {1:10}{0}{2:15}{0}{3:40}\n".format(" | ", sampleId, nLabels, str(pid)),
                      fname=saveFileListsName, savepath=rspath, mode='a', wString=True, printTag=False)
            _saveData(data=data, fname=saveFileDetails, savepath=rspath, mode='a', wString=True, printTag=False)
        _saveData(data="{0}\n".format("-" * 60), fname=saveFileListsName, savepath=rspath, mode='a', wString=True,
                  printTag=False)
        _saveData(data="{0}\n".format("-" * 111), fname=saveFileDetails, savepath=rspath, mode='a', wString=True,
                  printTag=False)


def _dataset_properties(X, y, f_name, n_total_components):
    XComponents = X[:, :n_total_components]
    # XComponents = XComponents[5]  # EcoCyc
    # y = y[5]
    # XComponents = XComponents[0] # HumanCyc
    # y = y[0]
    # XComponents = XComponents[3]  # AraCyc
    # y = y[3]
    # XComponents = XComponents[4] # YeastCyc
    # y = y[4]
    # XComponents = XComponents[1] # LeishCyc
    # y = y[1]
    # XComponents = XComponents[2] # TrypanoCyc
    # y = y[2]
    # y = y.reshape((1, y.shape[0]))
    # XComponents = XComponents.reshape((1, XComponents.shape[0]))
    # L_S = len(y)
    # DL_S = np.unique(L_S).size
    L_S = np.sum([len(i) for i in y])
    LCard_S = L_S / len(y)
    LDen_S = LCard_S / L_S
    DL_S = np.unique([j for i in y for j in i]).size
    PDL_S = DL_S / len(y)
    print('\t1)- CLASS PROPERTIES...')
    print('\t>> Number of labels for {0}: {1:f}...'.format(f_name, L_S))
    print('\t>> Label cardinality for {0}: {1:f}...'.format(f_name, LCard_S))
    print('\t>> Label density for {0}: {1:f}...'.format(f_name, LDen_S))
    print('\t>> Distinct label sets for {0}: {1:f}...'.format(f_name, DL_S))
    print('\t>> Proportion of distinct label sets for {0}: {1:f}...'.format(f_name, PDL_S))
    R_S = np.sum(XComponents)
    RCard_S = R_S / XComponents.shape[0]
    RDen_S = RCard_S / R_S
    DR_S = np.nonzero(np.sum(XComponents, axis=0))[0].size
    PDR_S = DR_S / XComponents.shape[0]
    print('\t2)- COMPONENT PROPERTIES...')
    print('\t>> Number of component for {0}: {1:f}...'.format(f_name, R_S))
    print('\t>> Component cardinality for {0}: {1:f}...'.format(f_name, RCard_S))
    print('\t>> Component density for {0}: {1:f}...'.format(f_name, RDen_S))
    print('\t>> Distinct component sets for {0}: {1:f}...'.format(f_name, DR_S))
    print('\t>> Proportion of distinct component sets for {0}: {1:f}...'.format(f_name, PDR_S))


def _generate_upset(args, obj_data, X, y, n_total_components):
    ### Useful for creating dataset through finding the frequencies
    XComponents = X[:, :n_total_components]
    trueECsperPtw = _loadData(fname=args.pathway_ec, loadpath=args.ospath)
    ecoX = XComponents[5]
    humanX = XComponents[0]
    araX = XComponents[3]
    yeastX = XComponents[4]
    leishX = XComponents[1]
    trypanoX = XComponents[2]
    ecoLabels = y[5]
    humanLabels = y[0]
    araLabels = y[3]
    yeastLabels = y[4]
    leishLabels = y[1]
    trypanoLabels = y[2]
    classes = [set(ecoLabels), set(humanLabels), set(araLabels), set(yeastLabels), set(leishLabels),
               set(trypanoLabels)]
    XComponents = [ecoX, humanX, araX, yeastX, leishX, trypanoX]
    dsnames = ['ecocyc', 'humancyc', 'aracyc', 'yeastcyc', 'leishcyc', 'trypanocyc']
    classesList = set([ptw for cls in classes for ptw in list(cls)])

    dfAll = pd.DataFrame(data=np.zeros((len(classesList), len(dsnames))), index=list(classesList),
                         columns=tuple(dsnames))
    dfAll.index.names = ['Pathways']

    for idx, sample in enumerate(classes):
        df = pd.DataFrame(data=np.zeros((len(classesList), 2)), index=classesList,
                          columns=('Present', 'Abundance'))
        df.index.names = ['Pathways']
        for ptwy in sample:
            df.loc[ptwy, 'Present'] = 1
            dfAll.loc[ptwy, dsnames[idx]] = 1
            identifiedECs = (XComponents[idx].T * trueECsperPtw[0][obj_data.pathway_id[ptwy]])
            identifiedECs = np.divide(identifiedECs, np.sum(trueECsperPtw[0][obj_data.pathway_id[ptwy]]))
            df.loc[ptwy, 'Abundance'] = np.sum(np.nan_to_num(identifiedECs))
        print('\t\t## Storing {0:s} into the file: {1:s}'
              .format(dsnames[idx], dsnames[idx] + '_upset.csv'))
        df.to_csv(path_or_buf=os.path.join(args.dspath, dsnames[idx] + '_upset.csv'), sep='\t')
    print('\t\t## Storing {0:s} into the file: {1:s}'.format("sixdb", 'sixdb_upset.csv'))
    dfAll.to_csv(path_or_buf=os.path.join(args.dspath, 'sixdb_upset.csv'), sep='\t')


def _corrAnalysis(args, rho=0.15):
    clf = _loadData(fname=args.model, loadpath=args.mdpath)
    mlb = clf.mlb
    coef_intercept = np.hstack((clf.coef, clf.intercept))
    del clf
    # Perform correlation analysis assuming each label as samples.
    # This needs to be added to the optimized cost function.
    coefCorr = np.corrcoef(coef_intercept)
    del coef_intercept
    np.nan_to_num(coefCorr, copy=False)
    # Keep only the upper triangular
    coefCorr = np.triu(coefCorr, k=1)
    coefCorr[coefCorr < 0.] = 0.
    # Any values above rho is set to +1 as positive correlation.
    # Do not set below rho as -1.
    ptwCorr = [np.argwhere(coefCorr[idx] > rho) for idx in np.arange(coefCorr.shape[0])]
    ptwCorr = [np.concatenate(ptw) for ptw in ptwCorr if len(ptw) > 0]
    for pidx in np.arange(len(ptwCorr)):
        ptw_lst = []
        for idx in ptwCorr[pidx]:
            ptw_lst.append(mlb.classes[idx])
        ptwCorr[pidx] = ptw_lst
    return ptwCorr


def _coeffAnalysis(args, objData, nTotalClassLabels, nTotalComponents, itemEvidFeaturesSize, reacEvidFeaturesSize,
                   kBestFeatures=20, computeAuc=False):
    clf = _loadData(fname=args.model, loadpath=args.mdpath)
    mlb = clf.mlb
    coef_intercept = np.hstack((clf.coef, clf.intercept))
    del clf
    ptwidx2value = _reverseIdx(objData.pathway_id)

    ## Abundance Features Analysis
    ecidx2value = _reverseIdx(objData.ec_id)
    trueECsperPtw = _loadData(fname=args.pathway_ec, loadpath=args.ospath)
    preprocessing.binarize(trueECsperPtw[0], copy=False)
    tmptrueECsperPtw = [trueECsperPtw[1][np.nonzero(trueECsperPtw[0][idx])[0]] for idx in
                        np.arange(trueECsperPtw[0].shape[0])]
    topECsperPtwFromCoeff = np.ndarray((trueECsperPtw[0].shape[0],), dtype=np.object)
    topECsperPtwFromCoeffScores = np.ndarray((trueECsperPtw[0].shape[0],), dtype=np.object)
    ecIDXperPtw = np.array(
        [np.argsort(coef_intercept[idx, :nTotalComponents])[::-1] for idx in np.arange(trueECsperPtw[0].shape[0])])

    recall = list()
    precision = list()
    fscore = list()
    intersectedECsPerPtw = list()
    kBest = kBestFeatures
    if computeAuc:
        kBest = np.arange(start=1, stop=nTotalComponents, step=3)

    for key, val in enumerate(kBest):
        tp = 0
        fp = 0
        fn = 0
        for pidx in np.arange(ecIDXperPtw.shape[0]):
            lst_ec = list()
            lst_ec_score = list()
            for eidx in ecIDXperPtw[pidx, :val]:
                tidx = trueECsperPtw[1][eidx]
                if tidx:
                    lst_ec.append(ecidx2value[int(tidx)])
                    lst_ec_score.append(coef_intercept[pidx, eidx])
            topECsperPtwFromCoeff[objData.pathway_id[mlb.classes[pidx]]] = lst_ec
            topECsperPtwFromCoeffScores[objData.pathway_id[mlb.classes[pidx]]] = lst_ec_score

        for pidx, item in enumerate(tmptrueECsperPtw):
            lst_ec = [ecidx2value[eidx] for eidx in item]
            intersect_lst_ec = np.intersect1d(lst_ec, topECsperPtwFromCoeff[pidx])
            intersectedECsPerPtw.append(intersect_lst_ec)
            tmp_1 = val - len(lst_ec)
            tmp_2 = len(lst_ec) - len(intersect_lst_ec)
            tmp_3 = val - len(intersect_lst_ec)
            tmp_4 = 0
            tmp_5 = 0
            if tmp_1 <= 0:
                tmp_4 = tmp_3
            elif tmp_1 > 0:
                tmp_4 = tmp_2
                tmp_5 = tmp_1
            fn += tmp_4
            fp += tmp_5
            tp += len(intersect_lst_ec)
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
        fscore.append(2 * (precision[key] * recall[key]) / (precision[key] + recall[key]))

    if computeAuc:
        ## Common Information for plots
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        plt.rc('axes', labelsize=16)
        plt.rc('pdf', fonttype='TrueType')
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
        plt.plot(list(kBest), precision, 'o-.', color='blue', markersize=10, lw=1, label='Average Precision')
        plt.plot(list(kBest), recall, 'o-.', color='red', markersize=10, lw=1, label='Average Recall')
        plt.plot(list(kBest), fscore, 'o-.', color='green', markersize=10, lw=1, label='Average F1')
        plt.ylim([0.0, 1.05])
        ax.set_xlabel('Number of features')
        ax.set_ylabel('Performance')
        plt.legend(loc="best")
        fig.set_size_inches(10, 6)
        fig.savefig('ecs_roc_curve.eps')

    _saveData(data='# Top {0:s} ECs and their scores for each pathway...\n\n'
              .format(str(kBestFeatures)), fname="mlLGPR_en_top_ECs_pathway.txt",
              savepath=args.rspath, tag='ECs and their scores',
              mode='w', wString=True)
    _saveData(data="{0}\n".format("-" * 110), fname="mlLGPR_en_top_ECs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    datum = " {1:10}{0}{2:40}{0}{3:20}{0}{4:20}{0}{5:8}\n".format(" | ", "Pathway ID", "Pathway Name", "EC", "Score",
                                                                  "TP")
    _saveData(data=datum + "{0}\n".format("-" * 110), fname="mlLGPR_en_top_ECs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)

    for pidx in np.arange(nTotalClassLabels):
        pathwayInit = False
        for eidx, ec in enumerate(topECsperPtwFromCoeff[pidx]):
            pathwayName = ""
            pathwayID = ""
            if not pathwayInit:
                pathwayName = ptwidx2value[pidx]
                pathwayID = pidx + 1
            pathwayInit = True
            trueEC = False
            if ec in intersectedECsPerPtw[pidx]:
                trueEC = True
            datum = " {1:10}{0}{2:40}{0}{3:20}{0}{4:20.4f}{0}{5:8}\n".format(" | ", pathwayID, pathwayName, ec,
                                                                             topECsperPtwFromCoeffScores[pidx][eidx],
                                                                             str(trueEC))
            _saveData(data=datum, fname="mlLGPR_en_top_ECs_pathway.txt", savepath=args.rspath,
                      mode='a', wString=True, printTag=False)
        _saveData(data="{0}\n".format("-" * 110), fname="mlLGPR_en_top_ECs_pathway.txt",
                  savepath=args.rspath, mode='a', wString=True, printTag=False)

    dictFeatures = ExtractFeaturesNames(path=os.path.join(os.getcwd(), 'feature'))

    ## Reaction Evidence Features Analysis
    lst_re_features = dictFeatures['Reaction-Evidence Features']
    top_re_features = Counter()
    start_idx = nTotalComponents
    REvidence = coef_intercept[:, start_idx:start_idx + reacEvidFeaturesSize]
    _saveData(data='# Top {0:s} Reaction evidence features and their scores for each pathway...\n\n'
              .format(str(kBestFeatures)), fname="mlLGPR_en_top_REs_pathway.txt",
              savepath=args.rspath, tag='REs and their scores', mode='w', wString=True)
    _saveData(data="{0}\n".format("-" * 150), fname="mlLGPR_en_top_REs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    datum = " {1:10}{0}{2:40}{0}{3:78}{0}{4:10}\n".format(" | ", "Pathway ID", "Pathway Name",
                                                          "Reaction Evidence Feature", "Score")
    _saveData(data=datum + "{0}\n".format("-" * 150), fname="mlLGPR_en_top_REs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    for pidx in np.arange(nTotalClassLabels):
        pathwayInit = False
        reIDXperPtw = np.argsort(REvidence[pidx, :])[::-1][:kBestFeatures]
        reScoreperPtw = np.sort(REvidence[pidx, :])[::-1][:kBestFeatures]
        for idx, ridx in enumerate(reIDXperPtw):
            pathwayName = ""
            pathwayID = ""
            if not pathwayInit:
                pathwayName = ptwidx2value[pidx]
                pathwayID = pidx + 1
                pathwayInit = True
            top_re_features[lst_re_features[idx]] += 1
            datum = " {1:10}{0}{2:40}{0}{3:78}{0}{4:6.4f}\n".format(" | ", pathwayID, pathwayName,
                                                                    lst_re_features[ridx],
                                                                    reScoreperPtw[idx])
            _saveData(data=datum, fname="mlLGPR_en_top_REs_pathway.txt", savepath=args.rspath,
                      mode='a', wString=True, printTag=False)
        _saveData(data="{0}\n".format("-" * 150), fname="mlLGPR_en_top_REs_pathway.txt",
                  savepath=args.rspath, mode='a', wString=True, printTag=False)
    print("\t\t## Most {0} common reaction evidence features across pathways: {1}".format(kBestFeatures,
                                                                                          top_re_features.most_common(
                                                                                              kBestFeatures)))

    ## Pathway Evidence Features Analysis
    lst_pe_features = dictFeatures['Pathway-Evidence Features']
    top_pe_features = Counter()
    start_idx = nTotalComponents + reacEvidFeaturesSize
    PEvidence = coef_intercept[:, start_idx:start_idx + itemEvidFeaturesSize]
    _saveData(data='# Top {0:s} Pathway evidence features and their scores for each pathway...\n\n'
              .format(str(kBestFeatures)), fname="mlLGPR_en_top_PEs_pathway.txt",
              savepath=args.rspath, tag='PEs and their scores', mode='w', wString=True)
    _saveData(data="{0}\n".format("-" * 135), fname="mlLGPR_en_top_PEs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    datum = " {1:10}{0}{2:40}{0}{3:63}{0}{4:10}\n".format(" | ", "Pathway ID", "Pathway Name",
                                                          "Pathway Evidence Feature", "Score")
    _saveData(data=datum + "{0}\n".format("-" * 135), fname="mlLGPR_en_top_PEs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    for pidx in np.arange(nTotalClassLabels):
        pathwayInit = False
        peIDXperPtw = np.argsort(PEvidence[pidx, :])[::-1][:kBestFeatures]
        peScoreperPtw = np.sort(PEvidence[pidx, :])[::-1][:kBestFeatures]
        for idx, eidx in enumerate(peIDXperPtw):
            pathwayName = ""
            pathwayID = ""
            if not pathwayInit:
                pathwayName = ptwidx2value[pidx]
                pathwayID = pidx + 1
                pathwayInit = True
            top_pe_features[lst_pe_features[eidx]] += 1
            datum = " {1:10}{0}{2:40}{0}{3:63}{0}{4:6.4f}\n".format(" | ", pathwayID, pathwayName,
                                                                    lst_pe_features[eidx],
                                                                    peScoreperPtw[idx])
            _saveData(data=datum, fname="mlLGPR_en_top_PEs_pathway.txt", savepath=args.rspath,
                      mode='a', wString=True, printTag=False)
        _saveData(data="{0}\n".format("-" * 135), fname="mlLGPR_en_top_PEs_pathway.txt",
                  savepath=args.rspath, mode='a', wString=True, printTag=False)

    print("\t\t## Most {0} common pathway evidence features across pathways: {1}".format(kBestFeatures,
                                                                                         top_pe_features.most_common(
                                                                                             kBestFeatures)))


def _featuresFisherScore(args, objData, fObject, nTotalComponents, itemEvidFeaturesSize, reacEvidFeaturesSize,
                         kBestFeatures=5):
    # Preprocess data
    fName = fObject + '_' + str(args.nsample)
    Xfile = fName + '_X.pkl'
    yfile = fName + '_y.pkl'
    X = _load_X_data(Xfile, args.dspath, properties=False, tag='X of ' + _datasetType(args))
    y, sample_ids = _load_y_data(yfile, args.dspath, tag='y of ' + _datasetType(args))
    ptwidx2value = _reverseIdx(objData.pathway_id)

    ## Abundance Component Features Analysis
    X_train = X[:, :nTotalComponents]
    ecidx2value = _reverseIdx(objData.ec_id)
    trueECsperPtw = _loadData(fname=args.pathway_ec, loadpath=args.ospath)
    preprocessing.binarize(trueECsperPtw[0], copy=False)
    topECsperPtwFromFScores = np.ndarray((trueECsperPtw[0].shape[0],), dtype=np.object)
    topECsperPtwFromFpValues = np.ndarray((trueECsperPtw[0].shape[0],), dtype=np.object)
    topECsperPtw = np.ndarray((trueECsperPtw[0].shape[0],), dtype=np.object)

    # Iterate for each pathways to get the Fisher score
    for idx, label in ptwidx2value.items():
        y_this_class = list()
        for lst_labels in y:
            if label in lst_labels:
                y_this_class.append(1)
            else:
                y_this_class.append(0)

        kBestFisher = feature_selection.SelectKBest(score_func=feature_selection.f_classif,
                                                    k=kBestFeatures).fit(X=X_train, y=y_this_class)
        topKsupports = np.argsort(kBestFisher.scores_)[::-1][:kBestFeatures]
        lst_ec = [ecidx2value[trueECsperPtw[1][ecidx]] for ecidx in topKsupports]
        topECsperPtwFromFScores[idx] = kBestFisher.scores_[topKsupports]
        topECsperPtwFromFpValues[idx] = kBestFisher.pvalues_[topKsupports]
        topECsperPtw[idx] = lst_ec

    trueECsperPtw = [trueECsperPtw[1][np.nonzero(trueECsperPtw[0][idx])[0]] for idx in
                     np.arange(trueECsperPtw[0].shape[0])]
    for idx, item in enumerate(trueECsperPtw):
        lst_ec = list()
        for eidx in item:
            lst_ec.append(ecidx2value[eidx])
        trueECsperPtw[idx] = lst_ec
    intersectedECsPerPtw = list()
    for idx, item in enumerate(trueECsperPtw):
        intersectedECsPerPtw.append(np.intersect1d(item, topECsperPtw[idx]))

    _saveData(data='# Top {0:s} ECs and their scores for each pathway...\n\n'
              .format(str(kBestFeatures)), fname="fisher_top_ECs_pathway.txt",
              savepath=args.rspath, tag='ECs and their scores',
              mode='w', wString=True)
    _saveData(data="{0}\n".format("-" * 110), fname="fisher_top_ECs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    datum = " {1:10}{0}{2:40}{0}{3:20}{0}{4:20}{0}{5:8}\n".format(" | ", "Pathway ID", "Pathway Name", "EC", "Score",
                                                                  "TP")
    _saveData(data=datum + "{0}\n".format("-" * 110), fname="fisher_top_ECs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    for idx in np.arange(topECsperPtw.shape[0]):
        pathwayInit = False
        for eidx, ec in enumerate(topECsperPtw[idx]):
            pathwayName = ""
            pathwayID = ""
            if not pathwayInit:
                pathwayName = ptwidx2value[idx]
                pathwayID = idx + 1
            pathwayInit = True
            trueEC = False
            if ec in intersectedECsPerPtw[idx]:
                trueEC = True
            datum = " {1:10}{0}{2:40}{0}{3:20}{0}{4:20.4f}{0}{5:8}\n".format(" | ", pathwayID, pathwayName, ec,
                                                                             topECsperPtwFromFScores[idx][eidx],
                                                                             str(trueEC))
            _saveData(data=datum, fname="fisher_top_ECs_pathway.txt", savepath=args.rspath,
                      mode='a', wString=True, printTag=False)
        _saveData(data="{0}\n".format("-" * 110), fname="fisher_top_ECs_pathway.txt",
                  savepath=args.rspath, mode='a', wString=True, printTag=False)

    dictFeatures = ExtractFeaturesNames(path=os.path.join(os.getcwd(), 'feature'))

    ## Reaction Evidence Features Analysis
    lst_re_features = dictFeatures['Reaction-Evidence Features']
    start_idx = nTotalComponents
    REvidence = X[:, start_idx:start_idx + reacEvidFeaturesSize]
    _saveData(data='# Top {0:s} Reaction evidence features and their scores for each pathway...\n\n'
              .format(str(kBestFeatures)), fname="fisher_top_REs_pathway.txt",
              savepath=args.rspath, tag='REs and their scores', mode='w', wString=True)
    _saveData(data="{0}\n".format("-" * 150), fname="fisher_top_REs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    datum = " {1:10}{0}{2:40}{0}{3:78}{0}{4:10}\n".format(" | ", "Pathway ID", "Pathway Name",
                                                          "Reaction Evidence Feature", "Score")
    _saveData(data=datum + "{0}\n".format("-" * 150), fname="fisher_top_REs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    for pidx, label in ptwidx2value.items():
        y_this_class = list()
        pathwayInit = False
        for lst_labels in y:
            if label in lst_labels:
                y_this_class.append(1)
            else:
                y_this_class.append(0)

        kBestFisher = feature_selection.SelectKBest(score_func=feature_selection.f_classif,
                                                    k=kBestFeatures).fit(X=REvidence, y=y_this_class)
        reIDXperPtw = np.argsort(kBestFisher.scores_)[::-1][:kBestFeatures]
        reScoreperPtw = kBestFisher.scores_[reIDXperPtw]
        for idx, ridx in enumerate(reIDXperPtw):
            pathwayName = ""
            pathwayID = ""
            if not pathwayInit:
                pathwayName = ptwidx2value[pidx]
                pathwayID = pidx + 1
                pathwayInit = True
            datum = " {1:10}{0}{2:40}{0}{3:78}{0}{4:6.4f}\n".format(" | ", pathwayID, pathwayName,
                                                                    lst_re_features[ridx],
                                                                    reScoreperPtw[idx])
            _saveData(data=datum, fname="fisher_top_REs_pathway.txt", savepath=args.rspath,
                      mode='a', wString=True, printTag=False)
        _saveData(data="{0}\n".format("-" * 150), fname="fisher_top_REs_pathway.txt",
                  savepath=args.rspath, mode='a', wString=True, printTag=False)

    ## Pathway Evidence Features Analysis
    lst_pe_features = dictFeatures['Pathway-Evidence Features']
    start_idx = nTotalComponents + reacEvidFeaturesSize
    PEvidence = X[:, start_idx:start_idx + itemEvidFeaturesSize]
    _saveData(data='# Top {0:s} Pathway evidence features and their scores for each pathway...\n\n'
              .format(str(kBestFeatures)), fname="fisher_top_PEs_pathway.txt",
              savepath=args.rspath, tag='PEs and their scores', mode='w', wString=True)
    _saveData(data="{0}\n".format("-" * 135), fname="fisher_top_PEs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    datum = " {1:10}{0}{2:40}{0}{3:63}{0}{4:10}\n".format(" | ", "Pathway ID", "Pathway Name",
                                                          "Pathway Evidence Feature", "Score")
    _saveData(data=datum + "{0}\n".format("-" * 135), fname="fisher_top_PEs_pathway.txt",
              savepath=args.rspath, mode='a', wString=True, printTag=False)
    for pidx, label in ptwidx2value.items():
        y_this_class = list()
        pathwayInit = False
        for lst_labels in y:
            if label in lst_labels:
                y_this_class.append(1)
            else:
                y_this_class.append(0)

        kBestFisher = feature_selection.SelectKBest(score_func=feature_selection.f_classif,
                                                    k=kBestFeatures).fit(X=PEvidence, y=y_this_class)
        peIDXperPtw = np.argsort(kBestFisher.scores_)[::-1][:kBestFeatures]
        peScoreperPtw = kBestFisher.scores_[peIDXperPtw]
        for idx, eidx in enumerate(peIDXperPtw):
            pathwayName = ""
            pathwayID = ""
            if not pathwayInit:
                pathwayName = ptwidx2value[pidx]
                pathwayID = pidx + 1
                pathwayInit = True
            datum = " {1:10}{0}{2:40}{0}{3:63}{0}{4:6.4f}\n".format(" | ", pathwayID, pathwayName,
                                                                    lst_pe_features[eidx],
                                                                    peScoreperPtw[idx])
            _saveData(data=datum, fname="fisher_top_PEs_pathway.txt", savepath=args.rspath,
                      mode='a', wString=True, printTag=False)
        _saveData(data="{0}\n".format("-" * 135), fname="fisher_top_PEs_pathway.txt",
                  savepath=args.rspath, mode='a', wString=True, printTag=False)


def _buildSimilarityMatrix(obj_data, ptwPosition=4, kb='metacyc', fName='pathway_similarity', savepath='.'):
    print('\t>> Building pathway similarities from {0}...'.format(kb))
    ptw_info = obj_data.processedKB[kb][ptwPosition].pathway_info
    regex = re.compile(r'\(| |\)')
    ptw_idx = obj_data._reverse_idx(obj_data.pathway_id)
    M = np.empty(shape=(len(ptw_info), len(ptw_info)), dtype=np.object)
    for i in range(M.shape[0]):
        query_pID = ptw_idx[i]
        query_end2end_rxn_series = [list(filter(None, regex.split(itm))) for itm in ptw_info[query_pID][3][1]]
        query_end2end_rxn_series = [rxn for lst in query_end2end_rxn_series for rxn in lst]
        for j in np.arange(M.shape[1]):
            target_pID = ptw_idx[j]
            target_end2end_rxn_series = [list(filter(None, regex.split(itm))) for itm in ptw_info[target_pID][3][1]]
            target_end2end_rxn_series = [rxn for lst in target_end2end_rxn_series for rxn in lst]
            M[i, j] = set.intersection(set(query_end2end_rxn_series), set(target_end2end_rxn_series))
    file = fName + '.pkl'
    fileDesc = '#File Description: number of pathways x number of pathways\n'
    _saveData(data=fileDesc, fname=file, savepath=savepath,
              tag='the pathway similarity matrix', mode='w+b')
    _saveData(data=('nPathways:', str(M.shape[0])), fname=file, savepath=savepath, mode='a+b', printTag=False)
    _saveData(data=M, fname=file, savepath=savepath, mode='a+b', printTag=False)


def _aggregate_results_from_algorithms(args, obj_data, golden_dataset):
    trueECsperPtw = _loadData(fname=args.pathway_ec, loadpath=args.ospath)

    baseline = _loadPredictedData(fname="baseline.lists",
                                  loadpath=args.rspath, tag='BASELINE results')
    naive = _loadPredictedData(fname="naive.lists",
                               loadpath=args.rspath, tag='Naive results')
    minpath = _loadPredictedData(fname="minpath.lists",
                                 loadpath=args.rspath, tag='MinPath results')
    pathologic = _loadPredictedData(fname="pathologic.lists",
                                    loadpath=args.rspath, tag='PathoLogic results')
    mllg = _loadPredictedData(fname="mlLGPR_labels_en_ab_re_pe.lists",
                              loadpath=args.rspath, tag='mlLGPR-EN results')

    all_ptwys = list()
    for idx, val in baseline.items():
        all_ptwys.extend(baseline[idx][1])
        all_ptwys.extend(naive[idx][1])
        all_ptwys.extend(minpath[idx][1])
        all_ptwys.extend(pathologic[idx][1])
        all_ptwys.extend(mllg[idx][1])
    all_ptwys = list(set(all_ptwys))

    y, sample_ids = _load_y_data(fname=golden_dataset, loadpath=args.dspath, tag='y of golden dataset')

    ecoLabels = y[5]
    humanLabels = y[0]
    araLabels = y[3]
    yeastLabels = y[4]
    leishLabels = y[1]
    trypanoLabels = y[2]

    idxclasses = [5, 0, 3, 4, 1, 2, 6]  # 6 all pathways
    classes = [set(ecoLabels), set(humanLabels), set(araLabels), set(yeastLabels),
               set(leishLabels), set(trypanoLabels)]
    SixDB = [ptw for lbl in classes for ptw in list(lbl)]
    classes = [set(ecoLabels), set(humanLabels), set(araLabels), set(yeastLabels),
               set(leishLabels), set(trypanoLabels), set(SixDB)]
    classnames = ['ecocyc', 'humancyc', 'aracyc', 'yeastcyc', 'leishcyc', 'trypanocyc', 'sixdb']

    # Extract properties from MetaCyc as number of pathways
    # based on the number of enzymatic reactions they contribute
    # in the associated pathways.
    desc = "\t{0}{1}\n".format(">>  ", "Extract properties from MetaCyc as number of pathways")
    desc += "\t\t{0}\n".format("based on the number of enzymatic reactions they contribute")
    desc += "\t\t{0}\n".format("in the associated pathways.")
    desc += "\t\t\t{1:28}{0}{2:28}".format(" | ", "Number of Enzymatic Reactions", "Number of Pathways")
    print(desc)
    for num_ecs in set(np.sum(trueECsperPtw[0], axis=1)):
        print("\t\t\t{1:29}{0}{2:18}".format(" | ", num_ecs, sum(np.sum(trueECsperPtw[0], axis=1) == num_ecs)))

    for idx, val in enumerate(idxclasses):
        lst_ptwys = list()
        if val != 6:
            lst_ptwys.extend(baseline[val + 1][1])
            lst_ptwys.extend(naive[val + 1][1])
            lst_ptwys.extend(minpath[val + 1][1])
            lst_ptwys.extend(pathologic[val + 1][1])
            lst_ptwys.extend(mllg[val + 1][1])
        else:
            lst_ptwys.extend(all_ptwys)

        lst_ptwys.extend(classes[idx])
        lst_ptwys = list(set(lst_ptwys))

        df = pd.DataFrame(data=np.zeros((len(lst_ptwys), 6)), index=lst_ptwys,
                          columns=(str(classnames[idx]).upper(), 'BASELINE', 'Naive', 'MinPath', 'PathoLogic',
                                   'mlLGPR_EN'))
        df.index.names = ['Pathways']

        # Store pathways that were missed by algorithms accroding to theirs reactions count
        df_missing_ptwys = pd.DataFrame(
            columns=(str(classnames[idx]).upper(), 'BASELINE', 'Naive', 'MinPath', 'PathoLogic',
                     'mlLGPR_EN'))
        df_missing_ptwys.index.names = ['ReactionsCount']
        num_rxns_per_missing_ptwy = dict({str(classnames[idx]).upper(): Counter(), 'BASELINE': Counter(),
                                          'Naive': Counter(), 'MinPath': Counter(), 'PathoLogic': Counter(),
                                          'mlLGPR_EN': Counter()})

        for lbl in lst_ptwys:
            true_ptwy = ''
            if lbl in classes[idx]:
                true_ptwy = lbl
                df.loc[lbl, str(classnames[idx]).upper()] = 1
                n_ecs = np.sum(trueECsperPtw[0][obj_data.pathway_id[lbl], :])
                num_rxns_per_missing_ptwy[str(classnames[idx]).upper()].update({n_ecs: 1})
            if lbl in baseline[val + 1][1]:
                df.loc[lbl, 'BASELINE'] = 1
            else:
                if true_ptwy == lbl:
                    num_rxns_per_missing_ptwy['BASELINE'].update({n_ecs: 1})
            if lbl in naive[val + 1][1]:
                df.loc[lbl, 'Naive'] = 1
            else:
                if true_ptwy == lbl:
                    num_rxns_per_missing_ptwy['Naive'].update({n_ecs: 1})
            if lbl in minpath[val + 1][1]:
                df.loc[lbl, 'MinPath'] = 1
            else:
                if true_ptwy == lbl:
                    num_rxns_per_missing_ptwy['MinPath'].update({n_ecs: 1})
            if lbl in pathologic[val + 1][1]:
                df.loc[lbl, 'PathoLogic'] = 1
            else:
                if true_ptwy == lbl:
                    num_rxns_per_missing_ptwy['PathoLogic'].update({n_ecs: 1})
            if lbl in mllg[val + 1][1]:
                df.loc[lbl, 'mlLGPR_EN'] = 1
            else:
                if true_ptwy == lbl:
                    num_rxns_per_missing_ptwy['mlLGPR_EN'].update({n_ecs: 1})
        df.to_csv(path_or_buf=os.path.join(args.rspath, classnames[idx] + '_upset.csv'), sep='\t')
        lst_rxns_count = sorted(set(list(chain(*num_rxns_per_missing_ptwy.values()))))
        for rxn_count in lst_rxns_count:
            df_missing_ptwys.loc[rxn_count, str(classnames[idx]).upper()] = \
                num_rxns_per_missing_ptwy[str(classnames[idx]).upper()][rxn_count]
            df_missing_ptwys.loc[rxn_count, 'BASELINE'] = num_rxns_per_missing_ptwy['BASELINE'][rxn_count]
            df_missing_ptwys.loc[rxn_count, 'Naive'] = num_rxns_per_missing_ptwy['Naive'][rxn_count]
            df_missing_ptwys.loc[rxn_count, 'MinPath'] = num_rxns_per_missing_ptwy['MinPath'][rxn_count]
            df_missing_ptwys.loc[rxn_count, 'PathoLogic'] = num_rxns_per_missing_ptwy['PathoLogic'][rxn_count]
            df_missing_ptwys.loc[rxn_count, 'mlLGPR_EN'] = num_rxns_per_missing_ptwy['mlLGPR_EN'][rxn_count]
        df_missing_ptwys.to_csv(path_or_buf=os.path.join(args.rspath, classnames[idx] + '_missing_pathways.csv'),
                                sep='\t')


def _compute_abd_cov(e_arg, obj_data, X_file, sample_ids, n_total_components, f_name):
    print("\t>> Computing coverage and abundance information from {0} dataset...".format(X_file))
    X = _load_X_data(X_file, e_arg.dspath, properties=False, tag='X of ' + _datasetType(e_arg))
    X = X[:, :n_total_components]
    trueECsperPtw = _loadData(fname=e_arg.pathway_ec, loadpath=e_arg.ospath)
    pathwaysId = [ptw for ptw, id in obj_data.pathway_id.items()]
    pathwaysCommonName = [obj_data.processedKB['metacyc'][4].pathway_info[ptw][0][1] for ptw, id in
                          obj_data.pathway_id.items()]
    dfCoverage = pd.DataFrame(data=np.zeros((len(pathwaysId), len(sample_ids))), index=list(pathwaysId),
                              columns=tuple(sample_ids))
    dfCoverage.index.names = ['Pathways_ID']
    dfAbundance = pd.DataFrame(data=np.zeros((len(pathwaysId), len(sample_ids))), index=list(pathwaysId),
                               columns=tuple(sample_ids))
    dfAbundance.index.names = ['Pathways_ID']
    M = np.ndarray((len(pathwaysId), len(sample_ids), 2))
    for ptw, pidx in obj_data.pathway_id.items():
        print('\t\t## Progress ({0:.2f}%): processing for the {1} pathway...'.format(
            ((pidx + 1) * 100.00 / trueECsperPtw[0].shape[0]), ptw))
        tmp = np.divide(X, trueECsperPtw[0][pidx])
        tmp[tmp == np.inf] = 0
        np.nan_to_num(tmp, copy=False)
        abdECs = np.divide(np.sum(tmp, axis=1).T, np.sum(trueECsperPtw[0][pidx]))
        abdECs[abdECs == np.inf] = 0
        np.nan_to_num(abdECs, copy=False)
        dfAbundance.loc[ptw, :] = abdECs
        tmp[tmp > 0] = 1.
        covECs = np.divide(np.sum(np.multiply(tmp, trueECsperPtw[0][pidx]), axis=1), np.sum(trueECsperPtw[0][pidx]))
        covECs[covECs == np.inf] = 0
        np.nan_to_num(covECs, copy=False)
        dfCoverage.loc[ptw, :] = covECs
        M[pidx, :, 0] = dfAbundance.loc[ptw, :]
        M[pidx, :, 1] = dfCoverage.loc[ptw, :]

    dfCoverage.insert(loc=0, column='PathwaysCommonName', value=pathwaysCommonName)
    dfAbundance.insert(loc=0, column='PathwaysCommonName', value=pathwaysCommonName)
    dfCoverage.to_csv(path_or_buf=os.path.join(e_arg.rspath, f_name + '_cov.csv'), sep='\t')
    dfAbundance.to_csv(path_or_buf=os.path.join(e_arg.rspath, f_name + '_abd.csv'), sep='\t')
    _saveData(data=M, fname=f_name + '_abd_cov.pkl', savepath=e_arg.rspath, printTag=False)


def _preProcessAbdCov(objData, Xfile, yfile, covThreshold, abdThreshold, samplesThreshold, e_arg, fName):
    print("\t>> PreProcessing the coverage and abundance information from {0} dataset...".format(Xfile))
    _, sample_ids = _load_y_data(yfile, e_arg.dspath, tag='y of ' + _datasetType(e_arg))
    pathwaysId = [ptw for ptw, id in objData.pathway_id.items()]
    dfCoverage = pd.read_csv(os.path.join(e_arg.rspath, fName + '_cov.csv'), sep='\t')
    dfAbundance = pd.read_csv(os.path.join(e_arg.rspath, fName + '_abd.csv'), sep='\t')
    dfCoverage = dfCoverage.values
    dfAbundance = dfAbundance.values

    for pidx, ptw in enumerate(pathwaysId):
        print('\t\t## Progress ({0:.2f}%): processing for the {1} pathway...'.format(
            ((pidx + 1) * 100.00 / len(pathwaysId)), ptw))
        idx = np.argwhere(dfCoverage == ptw)[0][0]
        sumCov = np.sum(dfCoverage[idx, 2:])
        sumAbd = np.sum(dfAbundance[idx, 2:])
        if sumCov < covThreshold or sumAbd < abdThreshold:
            dfCoverage = np.delete(dfCoverage, idx, axis=0)
            dfAbundance = np.delete(dfAbundance, idx, axis=0)
        else:
            nonZerosSamples = np.count_nonzero(dfCoverage[idx, 2:].astype(float))
            expectedCov = np.divide(sumCov, nonZerosSamples)
            expectedAbd = np.divide(sumAbd, nonZerosSamples)
            if expectedCov < covThreshold or expectedAbd < abdThreshold or nonZerosSamples < samplesThreshold:
                dfCoverage = np.delete(dfCoverage, idx, axis=0)
                dfAbundance = np.delete(dfAbundance, idx, axis=0)

    addName = '_processed'
    nonZerosSamplesIdx = np.nonzero(dfCoverage[:, 2:])
    nonZerosRow = nonZerosSamplesIdx[0]
    nonZerosSamplesIdx = [nonZerosSamplesIdx[1][nonZerosSamplesIdx[0] == idx] for idx in
                          np.unique(nonZerosSamplesIdx[0])]
    expWeightedAbd = np.multiply(dfCoverage[:, 2:], dfAbundance[:, 2:])
    pathwaysId = dfCoverage[:, 0]
    pathwaysCommonName = dfCoverage[:, 1]

    expectedCov = np.array([np.mean(dfCoverage[idx, nonZerosSamplesIdx[idx] + 2]) for idx in np.unique(nonZerosRow)])
    stdCov = np.array([np.std(dfCoverage[idx, nonZerosSamplesIdx[idx] + 2]) for idx in np.unique(nonZerosRow)])
    dfCoverage = pd.DataFrame(data=np.c_[dfCoverage[:, 1:], expectedCov, stdCov],
                              index=dfCoverage[:, 0],
                              columns=tuple(['PathwaysCommonName'] + sample_ids + ['ExpectedCoverage', 'StdCoverage']))
    dfCoverage.index.names = ['Pathways_ID']
    dfCoverage.to_csv(path_or_buf=os.path.join(e_arg.rspath, fName + addName + '_cov.csv'), sep='\t')

    expectedAbd = np.array([np.mean(dfAbundance[idx, nonZerosSamplesIdx[idx] + 2]) for idx in np.unique(nonZerosRow)])
    stdAbd = np.array([np.std(dfAbundance[idx, nonZerosSamplesIdx[idx] + 2]) for idx in np.unique(nonZerosRow)])
    dfAbundance = pd.DataFrame(data=np.c_[dfAbundance[:, 1:], expectedAbd, stdAbd],
                               index=dfAbundance[:, 0],
                               columns=tuple(
                                   ['PathwaysCommonName'] + sample_ids + ['ExpectedAbundance', 'StdAbundance']))
    dfAbundance.index.names = ['Pathways_ID']
    dfAbundance.to_csv(path_or_buf=os.path.join(e_arg.rspath, fName + addName + '_abd.csv'), sep='\t')

    expexpWeightedAbd = np.array(
        [np.mean(expWeightedAbd[idx, nonZerosSamplesIdx[idx]]) for idx in np.unique(nonZerosRow)])
    stdexpWeightedAbd = np.array(
        [np.std(expWeightedAbd[idx, nonZerosSamplesIdx[idx]]) for idx in np.unique(nonZerosRow)])
    dfExpWeigthedAbd = pd.DataFrame(
        data=np.c_[pathwaysCommonName, expWeightedAbd, expexpWeightedAbd, stdexpWeightedAbd],
        index=pathwaysId,
        columns=tuple(['PathwaysCommonName'] + sample_ids +
                      ['ExpectedWeightedAbundance', 'StdWeightedAbundance']))
    dfExpWeigthedAbd.index.names = ['Pathways_ID']
    dfExpWeigthedAbd.to_csv(path_or_buf=os.path.join(e_arg.rspath, fName + '_weighted_abd.csv'), sep='\t')


def _analyzePathwaysPathologicVSmlLGPR(Xfile, yfile, penalizeMissSamplesThreshold, e_arg):
    print("\t>> Processing cross-validation between PathoLogic and mlLGPR train from {0} dataset...".format(Xfile))
    y, sample_ids = _load_y_data(yfile, e_arg.dspath, tag='y of ' + _datasetType(e_arg))
    dfExpWeigthedAbd = pd.read_csv(os.path.join(e_arg.rspath, 'mg_dataset_418_weighted_abd.csv'), sep='\t')
    pathwaysId = list(dfExpWeigthedAbd['Pathways_ID'])
    pathwaysCommonName = list(dfExpWeigthedAbd['PathwaysCommonName'])

    dfMutualPredPathways = pd.DataFrame(data=np.zeros((len(pathwaysId), len(sample_ids))), index=pathwaysId,
                                        columns=tuple(sample_ids))
    dfMutualPredPathways.index.names = ['Pathways_ID']

    dfUpsetPathways = pd.DataFrame(data=np.zeros((len(pathwaysId), 8)), index=pathwaysId,
                                   columns=['Total_Hyptothetical_Samples',
                                            'Total_Weighted_Abundance',
                                            'Hyptothetical_Samples_Hit_by_PathoLogic',
                                            'Hyptothetical_Samples_Missed_by_PathoLogic',
                                            'Pathway_Score_by_PathoLogic',
                                            'Hyptothetical_Samples_Hit_by_mlLGPR',
                                            'Hyptothetical_Samples_Missed_by_mlLGPR',
                                            'Pathway_Score_by_mlLGPR'])
    dfUpsetPathways.index.names = ['Pathways_ID']

    dfGutCyc = pd.read_csv(os.path.join(e_arg.dspath, 'GutCyc_Master_Table.tsv'), sep='\t')
    dfGutCyc = dfGutCyc[['MP_ID', 'Treatment']]
    treatmentIDX = list(np.unique(dfGutCyc['Treatment']))

    dfTreatments = pd.DataFrame(data=np.zeros((1, len(treatmentIDX))), columns=tuple(treatmentIDX))
    dfTreatmentPathologic = pd.DataFrame(data=np.zeros((len(pathwaysId), len(treatmentIDX))), index=pathwaysId,
                                         columns=tuple(treatmentIDX))
    dfTreatmentPathologic.index.names = ['Pathways_ID']

    dfTreatmentmlLGPR = pd.DataFrame(data=np.zeros((len(pathwaysId), len(treatmentIDX))), index=pathwaysId,
                                     columns=tuple(treatmentIDX))
    dfTreatmentmlLGPR.index.names = ['Pathways_ID']

    mllg = _loadPredictedData(fname="mlLGPR_labels_en_ab_re_pe_meta.lists", loadpath=e_arg.rspath,
                              tag='mlLGPR-EN results')
    uniquePathways = np.unique([ptw for lst_ptw in y for ptw in lst_ptw] + pathwaysId)
    print('\t\t## Number of total distinct pathways for PathoLogic: {0}'.format(len(uniquePathways)))
    uniquePathways = np.unique([ptw for lst_ptw in mllg.items() for ptw in lst_ptw[1][1]] + pathwaysId)
    print('\t\t## Number of total distinct pathways for mlLGPR:{0}'.format(len(uniquePathways)))

    dfExpWeigthedAbd = dfExpWeigthedAbd.values
    dfGutCyc = dfGutCyc.values
    pathologic = np.zeros((len(pathwaysId), len(sample_ids)))
    mllgpr = np.zeros((len(pathwaysId), len(sample_ids)))

    # 1 only for PathoLogic; 2 for mlLGPR-EN; 3 for both; 0 for the rest
    treat = False
    for pidx, ptw in enumerate(pathwaysId):
        print('\t\t## Progress ({0:.2f}%): processing for the {1} pathway...'.format(
            ((pidx + 1) * 100.00 / len(pathwaysId)), ptw))
        for sidx, sid in enumerate(sample_ids):
            idx = np.argwhere(dfGutCyc == sid)[0]
            K = dfGutCyc[idx[0], 1]
            if dfExpWeigthedAbd[pidx, sidx + 2] != 0:
                dfUpsetPathways.loc[ptw, 'Total_Hyptothetical_Samples'] += 1
            if not treat:
                dfTreatments.loc[:, K] += 1
            if ptw in y[sidx]:
                dfMutualPredPathways.loc[ptw, sid] = 1
                dfUpsetPathways.loc[ptw, 'Hyptothetical_Samples_Hit_by_PathoLogic'] += 1
                dfTreatmentPathologic.loc[ptw, K] += 1
                pathologic[pidx, sidx] = 1
            else:
                if dfExpWeigthedAbd[pidx, sidx + 2] != 0:
                    dfUpsetPathways.loc[ptw, 'Hyptothetical_Samples_Missed_by_PathoLogic'] += 1
            if ptw in mllg[sidx + 1][1]:
                dfMutualPredPathways.loc[ptw, sid] += 2
                dfUpsetPathways.loc[ptw, 'Hyptothetical_Samples_Hit_by_mlLGPR'] += 1
                dfTreatmentmlLGPR.loc[ptw, K] += 1
                mllgpr[pidx, sidx] = 1
            else:
                if dfExpWeigthedAbd[pidx, sidx + 2] != 0:
                    dfUpsetPathways.loc[ptw, 'Hyptothetical_Samples_Missed_by_mlLGPR'] += 1

        treat = True
        expPenalizedAbd = np.power(dfExpWeigthedAbd[pidx, -2], penalizeMissSamplesThreshold)
        totalWeightedAbd = np.sum(dfExpWeigthedAbd[pidx, 2:-2])
        dfUpsetPathways.loc[ptw, 'Total_Weighted_Abundance'] = totalWeightedAbd

        miss = dfUpsetPathways.loc[ptw, 'Hyptothetical_Samples_Missed_by_PathoLogic']
        probtotalAbd = np.divide(np.sum(np.multiply(
            dfExpWeigthedAbd[pidx, 2:-2], pathologic[pidx, :])) - (miss * expPenalizedAbd), totalWeightedAbd)
        probtotalAbd = (0 if probtotalAbd < 0 else probtotalAbd)
        dfUpsetPathways.loc[ptw, 'Pathway_Score_by_PathoLogic'] = probtotalAbd

        miss = dfUpsetPathways.loc[ptw, 'Hyptothetical_Samples_Missed_by_mlLGPR']
        probtotalAbd = np.divide(np.sum(np.multiply(
            dfExpWeigthedAbd[pidx, 2:-2], mllgpr[pidx, :])) - (miss * expPenalizedAbd), totalWeightedAbd)
        probtotalAbd = (0 if probtotalAbd < 0 else probtotalAbd)
        dfUpsetPathways.loc[ptw, 'Pathway_Score_by_mlLGPR'] = probtotalAbd

    # Add pathway common name column
    dfMutualPredPathways.insert(loc=0, column='Pathways_Common_Name', value=pathwaysCommonName)
    dfUpsetPathways.insert(loc=0, column='Pathways_Common_Name', value=pathwaysCommonName)
    dfTreatmentPathologic.insert(loc=0, column='Pathways_Common_Name', value=pathwaysCommonName)
    dfTreatmentmlLGPR.insert(loc=0, column='Pathways_Common_Name', value=pathwaysCommonName)

    # save all dataframes
    dfMutualPredPathways.to_csv(path_or_buf=os.path.join(e_arg.rspath, 'pathologic_mllgpr.csv'), sep='\t')
    dfUpsetPathways.to_csv(path_or_buf=os.path.join(e_arg.rspath, 'pathologic_mllgpr_upset.csv'), sep='\t')
    dfTreatments.to_csv(path_or_buf=os.path.join(e_arg.rspath, 'treatments.csv'), sep='\t')
    dfTreatmentPathologic.to_csv(path_or_buf=os.path.join(e_arg.rspath, 'pathologic_treatment.csv'), sep='\t')
    dfTreatmentmlLGPR.to_csv(path_or_buf=os.path.join(e_arg.rspath, 'mllgpr_treatment.csv'), sep='\t')


def _analyzeEffectivenessPathologicVSmlLGPR(Xfile, nPathways, e_arg):
    print("\t>> Analyzing the effectiveness of PathoLogic and mlLGPR train from {0} dataset...".format(Xfile))
    dfWeightedAbd = pd.read_csv(os.path.join(e_arg.rspath, 'mg_dataset_418_weighted_abd.csv'), sep='\t',
                                index_col='Pathways_ID').drop(
        ['PathwaysCommonName', 'ExpectedWeightedAbundance', 'StdWeightedAbundance'], axis=1)
    dfCorr = dfWeightedAbd.T.corr()
    dfCorr.to_csv(path_or_buf=os.path.join(e_arg.rspath, 'pathways_correlations.csv'), sep='\t')

    dfUpsetPathways = pd.read_csv(os.path.join(e_arg.rspath, 'pathologic_mllgpr_upset.csv'), sep='\t')
    dfUpsetPathways = list(dfUpsetPathways.values)
    dfUpsetPathways.sort(key=itemgetter(3), reverse=True)
    dfUpsetPathways = np.array(dfUpsetPathways)
    pathologic = dfUpsetPathways[:, 6]
    mllgpr = dfUpsetPathways[:, 9]
    from scipy import stats
    print('\t\t## The Kolmogorov-Smirnov statistic on PathoLogic and mlLGPR: ', stats.ks_2samp(pathologic, mllgpr))
    print('\t\t## The Kruskal-Wallis H-test on PathoLogic and mlLGPR: ', stats.kruskal(pathologic, mllgpr))
    print('\t\t## The Mann–Whitney U test on PathoLogic and mlLGPR: ', stats.mannwhitneyu(pathologic, mllgpr))
    print('\t\t## The Shapiro‐Wilks on PathoLogic', stats.shapiro(pathologic))
    print('\t\t## The Shapiro‐Wilks on mlLGPR', stats.shapiro(mllgpr))
    tx, ty = stats.obrientransform(pathologic, mllgpr)
    print('\t\t## The 1-way ANOVA test on PathoLogic and mlLGPR: ', stats.f_oneway(tx, ty))

    dfUpsetPathways = list(dfUpsetPathways)
    if nPathways < 1:
        nPathways = len(dfUpsetPathways)
    print("\t\t## Analyzing the top {0} pathways across samples by PathoLogic and mlLGPR...".format(nPathways))
    pathwaysId = [item[0] for item in dfUpsetPathways[:nPathways]]
    dfTopPathways = pd.DataFrame(data=np.zeros((nPathways, 4)), index=pathwaysId,
                                 columns=['Pathway_Common_Name',
                                          'Total_Weighted_Abundance',
                                          'Pathway_Score_by_PathoLogic',
                                          'Pathway_Score_by_mlLGPR'])
    dfTopPathways.index.names = ['Pathways_ID']

    for pidx, pid in enumerate(pathwaysId):
        dfTopPathways.loc[pid, :] = np.array(
            [dfUpsetPathways[pidx][1], dfUpsetPathways[pidx][3], dfUpsetPathways[pidx][6], dfUpsetPathways[pidx][9]])

    fname = 'top_' + str(nPathways) + '_pathoLogic_mllgpr_pathways.csv'
    dfTopPathways.to_csv(path_or_buf=os.path.join(e_arg.rspath, fname), sep='\t')


def _getCommMatrix(G):
    # commsClustering = nx.cluster.clustering(G)
    commsClustering = community.best_partition(G)
    n_nodes = nx.number_of_nodes(G)
    labels = sorted(set([label[1] for label in commsClustering.items()]))
    M = sparse.dok_matrix((n_nodes, len(labels)), dtype=np.bool)
    for node_idx, label in commsClustering.items():
        comm_idx = labels.index(label)
        M[node_idx, comm_idx] = True
    return M.tocsr(), labels


def _assignArrayToLists(X):
    byAttributeVal = defaultdict(list)
    for col_idx in np.arange(X.shape[1]):
        byAttributeVal[col_idx].append(X[:, col_idx].nonzero()[0])
    return byAttributeVal.values()


def _drawAdjacencyMatrix(G, node_order=None, partitions=None, colors=[], savename=''):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    if partitions is None:
        partitions = []
    adjacencyMatrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    # Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5))  # in inches
    pyplot.imshow(adjacencyMatrix,
                  cmap="Greys",
                  interpolation="none")

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                           len(module),  # Width
                                           len(module),  # Height
                                           facecolor="none",
                                           edgecolor=color,
                                           linewidth="1"))
            current_idx += len(module)
    fig.savefig(savename + '_adj.eps')


###***************************        Private Main Entry        ***************************###

def _eda(e_arg):
    '''
    Create objData by calling the Data class
    '''

    ##########################################################################################################
    ######################             LOADING DATA OBJECT AND DATASET                  ######################
    ##########################################################################################################

    print('*** THE DATA OBJECT IS LOCATED IN: {0:s}'.format(e_arg.dspath))
    obj_data = _loadData(fname=e_arg.objectname, loadpath=e_arg.ospath, tag='data object')
    f_object = _datasetType(e_arg)

    n_total_components = 3650
    n_total_class_labels = 2526
    n_total_evidence_features = 5120
    n_total_class_evidence_features = 80832

    item_evid_features_size = int(n_total_class_evidence_features / n_total_class_labels)
    reac_evid_features_size = n_total_evidence_features - n_total_class_labels * 2
    k_best_features = 10

    similarity_file = None
    if e_arg.adjust_by_similarity:
        print('\t>> Retreiving items similarity score matrix file from: {0:s}'.format(e_arg.pathway_similarity))
        if e_arg.similarity_type == "sw":
            similarity_file = e_arg.pathway_similarity + '_sw.pkl'
        elif e_arg.similarity_type == "chi2":
            similarity_file = e_arg.pathway_similarity + '_chi2.pkl'
        elif e_arg.similarity_type == "cos":
            similarity_file = e_arg.pathway_similarity + '_cos.pkl'
        elif e_arg.similarity_type == "rbf":
            similarity_file = e_arg.pathway_similarity + '_rbf.pkl'

    ## List of checking items
    check_baseline_mapping = False
    check_prop_ds = False
    generate_upset = False
    agg_from_algorithms = False
    check_coeff_and_correlations = False
    check_feature_selection_fisher_score = False
    check_item_similarites = False
    check_cross_validation_models = False
    check_unique_items = False
    check_similarity_graph = False

    ##########################################################################################################
    ##################                          BASELINE PREDICTION                         ##################
    ##########################################################################################################

    if check_baseline_mapping:
        print('\n*** PERFORMING BASELINE BASED PATHWAY PREDICTION...')
        fName = f_object + '_' + str(e_arg.nsample)
        Xfile = fName + '_X.pkl'
        yfile = fName + '_y.pkl'
        X = _load_X_data(Xfile, e_arg.dspath, properties=False, tag='X of ' + _datasetType(e_arg))
        y, _ = _load_y_data(yfile, e_arg.dspath, tag='y of ' + _datasetType(e_arg))

        _check_baseline_mapping(e_arg, obj_data, X, Xfile, y, n_total_components, onlyPredict=False, report=True)

    ##########################################################################################################
    ##################                    CHECK PROPERTIES OF THE DATASET                   ##################
    ##########################################################################################################

    if check_prop_ds:
        print('\n*** ANALYZING THE PROPERTIES OF THE CHOSEN DATASET...')
        fName = f_object + '_' + str(e_arg.nsample)
        Xfile = fName + '_X.pkl'
        yfile = fName + '_y.pkl'
        X = _load_X_data(Xfile, e_arg.dspath, properties=False, tag='X of ' + _datasetType(e_arg))
        y, _ = _load_y_data(yfile, e_arg.dspath, tag='y of ' + _datasetType(e_arg))

        _dataset_properties(X, y, fName, n_total_components)

    ##########################################################################################################
    ####################                CHECK COEFFICIENTS AND CORRELATIONS                ###################
    ##########################################################################################################

    if check_coeff_and_correlations:
        print('\n*** ANALYZING THE COEFFICIENTS AND CORRELATIONS OF THE LEARNED MODEL...')
        _coeffAnalysis(args=e_arg, objData=obj_data, nTotalClassLabels=n_total_class_labels,
                       nTotalComponents=n_total_components, itemEvidFeaturesSize=item_evid_features_size,
                       reacEvidFeaturesSize=reac_evid_features_size, kBestFeatures=k_best_features, computeAuc=True)

    ##########################################################################################################
    ####################            CHECK FISHER SCORE BASED FEATURE SELECTION             ###################
    ##########################################################################################################

    if check_feature_selection_fisher_score:
        print('\n*** ANALYZING THE FISHER SCORE BASED FEATURE SELECTION...')
        _featuresFisherScore(args=e_arg, objData=obj_data, fObject=f_object, nTotalComponents=n_total_components,
                             itemEvidFeaturesSize=item_evid_features_size, reacEvidFeaturesSize=reac_evid_features_size,
                             kBestFeatures=k_best_features)

    ##########################################################################################################
    ####################                      CHECK ITEM SIMILARITIES                      ###################
    ##########################################################################################################

    if check_item_similarites:
        print('\n*** ANALYZING THE SIMILARITIES AMONG PATHWAYS...')
        load_matrix = False
        if not load_matrix:
            _buildSimilarityMatrix(obj_data=obj_data, fName=e_arg.pathway_similarity, savepath=e_arg.ospath)
        else:
            itemSimFile = os.path.join(e_arg.ospath, e_arg.pathway_similarity + '.pkl')
            with open(itemSimFile, 'rb') as f_in:
                while True:
                    A = pkl.load(f_in)
                    if type(A) is np.ndarray:
                        break
            dsnames = ['ecocyc', 'humancyc', 'aracyc', 'yeastcyc', 'leishcyc', 'trypanocyc']
            for ds in dsnames:
                dfPredicted = pd.read_csv(os.path.join(e_arg.rspath, ds + '_upset.csv'), sep='\t')
                headers = dfPredicted.columns.tolist()
                for method in headers:
                    if method.lower() in dsnames or method == 'Pathways':
                        continue
                    for pidx in np.arange(dfPredicted.shape[0]):
                        if dfPredicted[pidx, ds.upper()] == 0 and dfPredicted[pidx, method] == 1:
                            ptw = dfPredicted[pidx, 0]

    ##########################################################################################################
    ##################                      GENERATE UPSET FROM DATASET                     ##################
    ##########################################################################################################

    if generate_upset:
        print('\n*** GENERATING UPSET RESULTS FROM DATASET...')
        fName = f_object + '_' + str(e_arg.nsample)
        Xfile = fName + '_X.pkl'
        yfile = fName + '_y.pkl'
        X = _load_X_data(Xfile, e_arg.dspath, properties=False, tag='X of ' + _datasetType(e_arg))
        y, sample_ids = _load_y_data(yfile, e_arg.dspath, tag='y of ' + _datasetType(e_arg))
        _generate_upset(args=e_arg, obj_data=obj_data, X=X, y=y, n_total_components=n_total_components)

    ##########################################################################################################
    ##################                  COMBINING RESULTS FROM ALL METHODS                  ##################
    ##########################################################################################################

    if agg_from_algorithms:
        print('\n*** COMBINING RESULTS FROM ALL METHODS...')
        _aggregate_results_from_algorithms(args=e_arg, obj_data=obj_data, golden_dataset="gold_dataset_ptw_ec_63_y.pkl")

    ##########################################################################################################
    ####################          CHECK RESULTS OF ITEMS FROM MODEL AND PATHOLOGIC         ###################
    ##########################################################################################################

    if check_cross_validation_models:
        print('\n*** ANALYSING RESULTS OF PATHWAYS FROM THE LEARNED MODEL AND THE PATHOLOGIC...')
        fName = f_object + '_' + str(e_arg.nsample)
        Xfile = fName + '_X.pkl'
        yfile = fName + '_y.pkl'
        y, sample_ids = _load_y_data(yfile, e_arg.dspath, tag='y of ' + _datasetType(e_arg))
        calcAbdCov = False
        preProcessAbdCov = False
        analyzePathways = False
        analyzeEffectiveness = True
        abdThreshold = 1.
        covThreshold = 0.3
        samplesThreshold = 5
        penalizeMissSamplesThreshold = 0.5

        if calcAbdCov:
            _compute_abd_cov(e_arg=e_arg, obj_data=obj_data, X_file=Xfile, sample_ids=sample_ids,
                             n_total_components=n_total_components, f_name=fName)

        if e_arg.ds_type == 'meta_ds':
            if preProcessAbdCov:
                _preProcessAbdCov(obj_data, Xfile, yfile, covThreshold=covThreshold, abdThreshold=abdThreshold,
                                  samplesThreshold=samplesThreshold, e_arg=e_arg, fName=fName)
            if analyzePathways:
                _analyzePathwaysPathologicVSmlLGPR(Xfile=Xfile, yfile=yfile,
                                                   penalizeMissSamplesThreshold=penalizeMissSamplesThreshold,
                                                   e_arg=e_arg)

            if analyzeEffectiveness:
                # for nPathways in np.arange(start=50, stop=760, step=50):
                # _analyzeEffectivenessPathologicVSmlLGPR(Xfile=Xfile, nPathways=nPathways, e_arg=e_arg)
                nPathways = 200
                _analyzeEffectivenessPathologicVSmlLGPR(Xfile=Xfile, nPathways=nPathways, e_arg=e_arg)

    ##########################################################################################################
    ####################                         CHECK UNIQUE ITEMS                        ###################
    ##########################################################################################################

    if check_unique_items:
        print('\n***  ANALYZING UNIQUE PATHWAYS...')
        dsnames = ['ecocyc', 'humancyc', 'aracyc', 'yeastcyc', 'leishcyc', 'trypanocyc']
        dfPredPathways = pd.read_csv(os.path.join(e_arg.dspath, 'sixdb_upset.csv'), sep='\t', index_col=0)
        lst_labels = dfPredPathways.index.tolist()
        dfAllPredictedUnique = pd.DataFrame(data=np.zeros((dfPredPathways.shape[0], dfPredPathways.shape[1])),
                                            index=lst_labels,
                                            columns=tuple(dsnames))
        dfAllPredictedUnique.index.names = ['Pathways']

        for ds in dsnames:
            dfPredicted = pd.read_csv(os.path.join(e_arg.rspath, ds + '_upset.csv'), sep='\t', index_col=0)
            methods = dfPredicted.columns.tolist()[1:]
            dfPredictedUnique = pd.DataFrame(data=np.zeros((dfPredPathways.shape[0], dfPredicted.shape[1] - 1)),
                                             index=lst_labels,
                                             columns=tuple(methods))
            dfPredictedUnique.index.names = ['Pathways']
            for ptwy in lst_labels:
                if dfPredPathways.loc[ptwy, ds] == 1:
                    if sum(dfPredPathways.loc[ptwy, :]) == 1:
                        dfAllPredictedUnique.loc[ptwy, ds] = 1
                        for method in methods:
                            if dfPredicted.loc[ptwy, method] == 1:
                                dfPredictedUnique.loc[ptwy, method] = 1
            dfPredictedUnique.to_csv(path_or_buf=os.path.join(e_arg.rspath, ds + '_unique.csv'), sep='\t')
        dfAllPredictedUnique.to_csv(path_or_buf=os.path.join(e_arg.rspath, 'sixdb_unique.csv'), sep='\t')

    ##########################################################################################################
    ####################         RECONSTRUCTION ERROR AND ADJACENCY MATRIX ANALYSIS        ###################
    ##########################################################################################################

    if check_similarity_graph:
        print('\n*** COMPUTE RECONSTRUCTION ERROR...')
        if similarity_file:
            ### From the static adjacency matrix
            A = _loadItemFeatures(fname=similarity_file, datasetPath=e_arg.ospath, components=False)
            np.fill_diagonal(A, 0)
            G = nx.from_numpy_matrix(A)
            commMatrix, labels = _getCommMatrix(G=G)
            nodesLabelsOrdered = [node_id for comm_idx, comm_id in enumerate(labels)
                                  for node_id in commMatrix[:, comm_idx].nonzero()[0]]
            labelsList = _assignArrayToLists(commMatrix)
            labelsList = [node for comm in labelsList for node in comm]
            _drawAdjacencyMatrix(G, nodesLabelsOrdered, [labelsList], ["blue"],
                                 savename=e_arg.pathway_similarity + '_' + e_arg.similarity_type)
            L_G = -1 * nx.normalized_laplacian_matrix(G).todense()
            L_G = np.fill_diagonal(L_G, 0)

            ### From the learned train
            clf = _loadData(fname=e_arg.model, loadpath=e_arg.mdpath)
            coef = clf.coef
            del clf
            L_coefCosine = cosine_similarity(coef)
            L_coefCorr = np.corrcoef(coef)
            del coef

            # cosine similarity
            np.nan_to_num(L_coefCosine, copy=False)
            L_coefCosine[L_coefCosine < 0.] = 0.
            np.fill_diagonal(L_coefCosine, 0)
            G = nx.from_numpy_matrix(L_coefCosine)
            commMatrix, labels = _getCommMatrix(G=G)
            labelsList = _assignArrayToLists(commMatrix)
            labelsList = [node for comm in labelsList for node in comm]
            _drawAdjacencyMatrix(G, nodesLabelsOrdered, [labelsList], ["blue"],
                                 savename=e_arg.pathway_similarity + '_cos')
            L_coefCosine = -1 * nx.normalized_laplacian_matrix(G).todense()
            L_coefCosine = np.fill_diagonal(L_coefCosine, 0)

            # correlation analysis
            np.nan_to_num(L_coefCorr, copy=False)
            L_coefCorr[L_coefCorr < 0.] = 0.
            np.fill_diagonal(L_coefCorr, 0)
            G = nx.from_numpy_matrix(L_coefCorr)
            commMatrix, labels = _getCommMatrix(G=G)
            labelsList = _assignArrayToLists(commMatrix)
            labelsList = [node for comm in labelsList for node in comm]
            _drawAdjacencyMatrix(G, nodesLabelsOrdered, [labelsList], ["blue"],
                                 savename=e_arg.pathway_similarity + '_corr')
            L_coefCorr = -1 * nx.normalized_laplacian_matrix(G).todense()
            L_coefCorr = np.fill_diagonal(L_coefCorr, 0)

            print('\t>> Reconstruction error (cosine):', np.sum(np.power(L_G - L_coefCosine, 2)) / (L_G.shape[0] * 2))
            print('\t>> Reconstruction error (correlation):', np.sum(np.power(A - L_coefCorr, 2)) / (L_G.shape[0] * 2))
        else:
            print("\t>> Adjacency matrix is not provided yet...")


def EDAMain(e_arg):
    try:
        if os.path.isdir(e_arg.ospath):
            timeref = time.time()
            _eda(e_arg)
            print('\n*** EXPLORATORY DATA ANALYSIS CONSUMED {0:f} SECONDS'.format(round(time.time() - timeref, 3)),
                  file=sys.stderr)
        else:
            print('\n*** PLEASE MAKE SURE TO PROVIDE THE CORRECT PATH FOR THE DATA OBJECT', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
