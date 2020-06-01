'''
This file trains the metagenomics inputset using multi-label
logistic regression.
'''
import os
import time
import traceback
import warnings

import numpy as np
from model.mlUtility import LoadItemFeatures, LoadYFile, ReverseIdx
from scipy.special import expit
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

try:
    import cPickle as pkl
except:
    import pickle as pkl


class mlLGPR(object):
    def __init__(self, classes, classLabelsIds, labelsComponentsFile, itemPrintFeaturesFile=None,
                 n_components=2000, scaleFeature=True, sMethod='normalizer',
                 binarizeAbundance=True, useReacEvidenceFeatures=True, useItemEvidenceFeatures=True,
                 usePossibleClassFeatures=False, useLabelComponentFeatures=False, nTotalComponents=0,
                 nTotalClassLabels=0, nTotalEvidenceFeatures=0, nTotalClassEvidenceFeatures=0, penalty='elasticnet',
                 coef_similarity_type="sw", alpha=0.0001, l1_ratio=0.65, sigma=2, fit_intercept=True, 
                 max_inner_iter=1000, nEpochs=5, nBatches=10, testInterval=2, shuffle=True, adaptive_beta=0.45,
                 threshold=0.5, learning_rate="optimal", random_state=None, n_jobs=-1):

        np.random.seed(seed=random_state)
        self.mlb = preprocessing.MultiLabelBinarizer(classes=tuple(classes))
        self.classes = classes
        self.classLabelsIds = classLabelsIds
        self.classLabelsIdx = ReverseIdx(classLabelsIds)
        self.scaleFeature = scaleFeature
        self.binarizeAbundance = binarizeAbundance
        self.nComponentsForRMethods = n_components
        sMethods = {'minmax': preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False),
                    'maxabs': preprocessing.MaxAbsScaler(copy=False),
                    'standard': preprocessing.StandardScaler(with_mean=True, with_std=True, copy=False),
                    'robust': preprocessing.RobustScaler(with_centering=True, with_scaling=True,
                                                         quantile_range=(25.0, 75.0), copy=False),
                    'normalizer': preprocessing.Normalizer(norm='l2', copy=False)}
        if sMethod.lower() not in sMethods:
            raise Exception("The scaler is not recognized. Please provide either: "
                            "minmax, maxabs, standard, robust, or normalizer.")
        else:
            self.scalerMethod = sMethods[sMethod.lower()]

        self.labelsComponentsFile = labelsComponentsFile
        self.itemPrintFeaturesFile = itemPrintFeaturesFile
        self.useReacEvidenceFeatures = useReacEvidenceFeatures
        self.usePossibleClassFeatures = usePossibleClassFeatures
        self.useLabelComponentFeatures = useLabelComponentFeatures
        self.useItemEvidenceFeatures = useItemEvidenceFeatures
        self.nTotalComponents = nTotalComponents
        self.nTotalClassLabels = nTotalClassLabels
        self.nReacEvidenceFeatures = nTotalEvidenceFeatures - self.nTotalClassLabels * 2
        self.nPossibleClassFeatures = self.nTotalClassLabels * 2
        self.nTotalClassEvidenceFeatures = nTotalClassEvidenceFeatures
        self.penalty = penalty
        self.coef_similarity_type = coef_similarity_type
        self.alpha = alpha
        self.sigma = sigma
        self.fit_intercept = fit_intercept
        self.max_inner_iter = max_inner_iter
        self.nEpochs = nEpochs
        self.nBatches = nBatches
        self.testInterval = testInterval
        self.shuffle = shuffle
        self.adaptive_beta = adaptive_beta
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.l2_ratio = 0.

        self.estimator = SGDClassifier(loss='log', penalty=self.penalty, alpha=alpha,
                                       l1_ratio=l1_ratio, fit_intercept=self.fit_intercept,
                                       max_iter=self.max_inner_iter, shuffle=True, n_jobs=self.n_jobs,
                                       random_state=self.random_state, warm_start=True, average=True)
        self.is_fit = False
        self.params = list()
        warnings.filterwarnings("ignore", category=Warning)

    def print_arguments(self):

        useitemPrintFeatures = True

        if self.itemPrintFeaturesFile is None:
            useitemPrintFeatures = False

        args = ['No. of Components for Reduction Method: {0}'.format(self.nComponentsForRMethods),
                'Scale Feature: {0}'.format(
                    self.scaleFeature), 'Scaling Method: {0}'.format(self.scalerMethod),
                'Binarize Abundance: {0}'.format(self.binarizeAbundance),
                'Use Evidence Features: {0}'.format(
                    self.useReacEvidenceFeatures),
                'Use Item Evidence Features: {0}'.format(
                    self.useItemEvidenceFeatures),
                'Use Possible Class Features: {0}'.format(
                    self.usePossibleClassFeatures),
                'Use Labels Components Features: {0}'.format(
                    self.useLabelComponentFeatures),
                'Use Item Print Features: {0}'.format(useitemPrintFeatures),
                'Number of Components: {0}'.format(self.nTotalComponents),
                'Number of Labels: {0}'.format(self.nTotalClassLabels),
                'Number of Evidence Features: {0}'.format(
                    self.nReacEvidenceFeatures),
                'Number of Possible Class Features: {0}'.format(
                    self.nPossibleClassFeatures),
                'Number of Label Evidence Features: {0}'.format(
                    self.nTotalClassEvidenceFeatures),
                'Adjust Parameter using Similarity Matrix: {0}'.format(
                    self.adjustCoef),
                'Maximum number of Iterations of the Optimization Algorithm: {0}'.format(
                    self.max_inner_iter),
                'Maximum number of Epochs: {0}'.format(self.nEpochs),
                'Number of Batches: {0}'.format(self.nBatches),
                'Adaptive Beta Hyper-Parameter: {0}'.format(
                    self.adaptive_beta),
                'A User Cut-Off Threshold: {0}'.format(self.threshold),
                'Display Interval: {0}'.format(self.testInterval),
                'Learning Rate: {0}'.format(self.learning_rate),
                'Hyperparameter to Control the Strength of Regularization: {0}'.format(
                    self.alpha),
                'Hyperparameter to Compromise between L1 and L2 Penalty: {0}'.format(
                    self.l1_ratio),
                'Hyperparameter to Compromise between L2 and Laplacian Penalty: {0}'.format(
                    self.l2_ratio),
                'Use Grid Search CV: {0}'.format(self.grid),
                'Number of CPU cores: {0}'.format(self.n_jobs)]
        args = [str(item[0] + 1) + '. ' + item[1]
                for item in zip(list(range(len(args))), args)]
        return '\n\t\t\t'.join(args)

    def _sigmoid(self, X):
        return expit(X)

    def _decision_function(self, X, classIdx):
        scores = np.dot(X, np.reshape(self.coef[classIdx], newshape=(1, self.coef[classIdx].shape[0])).T) + \
            self.intercept[classIdx]
        return scores.ravel() if scores.shape[1] == 1 else scores

    def _score(self, X, y, classIdx):
        _, y_pred = self._predict(X, classIdx)
        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred != 1] = 0
        return accuracy_score(y, y_pred)

    def _cost(self, X_file, y_file, normalize=True, forTraining=False):
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")
        y_pred = self.predict(X_file=X_file, forTraining=forTraining)
        y_true = LoadYFile(nSamples=y_pred.shape[0], y_file=y_file)
        y_true = self.mlb.fit_transform(y_true)
        if normalize:
            lloss = log_loss(y_true, y_pred)
        else:
            lloss = log_loss(y_true, y_pred, normalize=False)
        return lloss

    def _transformFeatures(self, classLabel, featureNonItem, itemEvidenceFeaturesSize, itemFeatures, labelsComponents,
                           notClassLabel=None, y_this_class=None, fit=False):
        posIDX = list()
        negIDX = list()
        X = featureNonItem[:, :self.nTotalComponents]
        if self.useReacEvidenceFeatures:
            X = np.hstack(
                (X, featureNonItem[:, X.shape[1]:X.shape[1] + self.nReacEvidenceFeatures]))
        if self.usePossibleClassFeatures:
            X = np.hstack(
                (X, featureNonItem[:, X.shape[1]:X.shape[1] + self.nPossibleClassFeatures]))
        if self.useItemEvidenceFeatures:
            idx = self.classLabelsIds[classLabel]
            X = np.hstack(
                (X, featureNonItem[:, idx:idx + itemEvidenceFeaturesSize]))
        if self.itemPrintFeaturesFile is not None:
            newShape = (X.shape[0], itemFeatures.shape[1])
            X = np.hstack((X, np.zeros(shape=newShape)))
            tmp = itemFeatures[self.classLabelsIds[classLabel]]
            tmp = np.reshape(tmp, (1, tmp.shape[0]))
            if self.scaleFeature:
                self.scalerMethod.fit_transform(tmp)
            if fit:
                posIDX = np.argwhere(np.array(y_this_class) == 1)[:, 0]
                tmp = np.tile(tmp, (len(posIDX), 1))
                X[posIDX, -itemFeatures.shape[1]:] = tmp

                # Create negative samples
                negIDX = np.argwhere(np.array(y_this_class) == 0)[:, 0]
                fLabels = np.random.choice(a=notClassLabel, size=len(negIDX))
                fIFeature = itemFeatures[[
                    self.classLabelsIds[label] for label in fLabels]]
                if self.scaleFeature:
                    self.scalerMethod.fit_transform(fIFeature)
                X[negIDX, -itemFeatures.shape[1]:] = fIFeature
            else:
                tmp = np.tile(tmp, (X.shape[0], 1))
                X = np.hstack((X, tmp))
        if self.useLabelComponentFeatures:
            newShape = (X.shape[0], labelsComponents.shape[1])
            X = np.hstack((X, np.zeros(shape=newShape)))
            tmp = labelsComponents[self.classLabelsIds[classLabel]]
            tmp = np.reshape(tmp, (1, tmp.shape[0]))
            if fit:
                if len(posIDX) == 0:
                    posIDX = np.argwhere(np.array(y_this_class) == 1)[:, 0]
                    negIDX = np.argwhere(np.array(y_this_class) == 0)[:, 0]
                tmp = np.tile(tmp, (len(posIDX), 1))
                tmp = np.int8(np.logical_and(
                    X[posIDX, : self.nTotalComponents], tmp))
                X[posIDX, -labelsComponents.shape[1]:] = tmp

                # Create negative samples
                fLabels = np.random.choice(a=notClassLabel, size=len(negIDX))
                fComponents = labelsComponents[[
                    self.classLabelsIds[label] for label in fLabels]]
                fComponents = np.int8(np.logical_and(
                    X[negIDX, : self.nTotalComponents], fComponents))
                X[negIDX, -labelsComponents.shape[1]:] = fComponents
            else:
                tmp = np.tile(tmp, (X.shape[0], 1))
                tmp = np.int8(np.logical_and(
                    X[0, : self.nTotalComponents], tmp))
                X = np.hstack((X, tmp))
        if self.scaleFeature and self.useReacEvidenceFeatures:
            startIdx = self.nTotalComponents
            lastIdx = self.nTotalComponents + \
                self.nReacEvidenceFeatures - self.nTotalClassLabels * 2
            self.scalerMethod.fit_transform(X[:, startIdx:lastIdx])
        return X

    def _fit(self, label, featureNonItem, itemFeatures, labelsComponents,
             itemEvidenceFeaturesSize):
        for classIdx, classLabel in enumerate(self.mlb.classes):
            notClassLabel = [
                c for c in self.mlb.classes if c not in classLabel]
            y = list()
            for lidx, labels in enumerate(label):
                if classLabel in labels:
                    y.append(1)
                else:
                    y.append(0)

            # If only positive or negative instances then return the function
            if len(np.unique(y)) < 2:
                continue

            X = self._transformFeatures(classLabel, featureNonItem, itemEvidenceFeaturesSize, itemFeatures,
                                        labelsComponents, notClassLabel, y, fit=True)
            if len(np.unique(y)) == 2:
                print(
                    '\t\t\t--> Building model for: {0} pathway'.format(classLabel))
                self.is_fit = True
                coef_init = np.reshape(self.coef[classIdx], newshape=(
                    1, self.coef[classIdx].shape[0]))
                intercept_init = self.intercept[classIdx]
                self.estimator.fit(X=X, y=y, coef_init=coef_init,
                                   intercept_init=intercept_init)
                self.coef[classIdx] = self.estimator.coef_[0]
                self.intercept[classIdx] = self.estimator.intercept_

    def fit(self, X_file, y_file, XdevFile=None, ydevFile=None, subSample=True, subSampleShuffle=True,
            subsampleSize=0.1, savename='mllg', savepath=''):

        oldCost = np.inf
        if self.l1_ratio == 1:
            savename = savename + '_l1_ab'
        elif self.l1_ratio == 0:
            savename = savename + '_l2_ab'
        else:
            savename = savename + '_en_ab'

        compFeaturesSize = self.nTotalComponents

        if self.useReacEvidenceFeatures != False:
            reacEvidFeaturesSize = self.nReacEvidenceFeatures
            savename = savename + '_re'
        else:
            reacEvidFeaturesSize = 0

        if self.useItemEvidenceFeatures != False:
            itemEvidFeaturesSize = int(
                self.nTotalClassEvidenceFeatures / self.nTotalClassLabels)
            savename = savename + '_pe'
        else:
            itemEvidFeaturesSize = 0

        if self.usePossibleClassFeatures != False:
            possibleItemsFeaturesSize = self.nPossibleClassFeatures
            savename = savename + '_pp'
        else:
            possibleItemsFeaturesSize = 0

        if self.itemPrintFeaturesFile is not None:
            itemFeatures = LoadItemFeatures(
                fname=self.itemPrintFeaturesFile, components=False)
            itemPrintFeaturesSize = itemFeatures.shape[1]
            savename = savename + '_pf'
        else:
            itemFeatures = None
            itemPrintFeaturesSize = 0

        if self.useLabelComponentFeatures:
            labelsComponents = LoadItemFeatures(
                fname=self.labelsComponentsFile)
            labelsComponents = labelsComponents[0]
            labelCompSize = labelsComponents.shape[1]
            savename = savename + '_pc'
        else:
            labelsComponents = None
            labelCompSize = 0

        savename = savename + '.pkl'
        fName = os.path.join(savepath, savename)
        totalUsedXSize = compFeaturesSize + reacEvidFeaturesSize + itemEvidFeaturesSize + possibleItemsFeaturesSize + \
            + labelCompSize + itemPrintFeaturesSize

        featureSizeforX = compFeaturesSize + self.nReacEvidenceFeatures + \
            self.nPossibleClassFeatures + self.nTotalClassEvidenceFeatures

        # Set hyper-paramters
        alpha = self.alpha
        lam = self.l1_ratio
        sigma = self.sigma

        if type(alpha) is not np.ndarray:
            alpha = [alpha]
        if type(lam) is not np.ndarray:
            lam = [lam]
        if type(sigma) is not np.ndarray:
            sigma = [sigma]
        self.params = [{'alpha': a, 'lam': l1, 'sigma': s} for a in alpha
                       for l1 in lam for s in sigma]
        for pidx, param in enumerate(self.params):
            params_sgd = {"alpha": param['alpha'], "l1_ratio": param['lam']}
            if self.adjustCoef:
                params_sgd = {"alpha": param['alpha']
                              * 1 / param['lam'], "l1_ratio": 0}
            self.estimator.set_params(**params_sgd)

            # Initialize coefficients
            self.coef = np.random.uniform(-1 / np.sqrt(totalUsedXSize), 1 / np.sqrt(totalUsedXSize),
                                          (len(self.mlb.classes), totalUsedXSize))
            self.intercept = np.zeros(shape=(len(self.mlb.classes), 1))

            for epoch in np.arange(self.nEpochs):
                print(
                    '\t  {0:d})- Epoch count: {0:d} (out of {1:d})...'.format(epoch + 1, self.nEpochs))
                epoch_timeref = time.time()
                with open(X_file, 'rb') as f_x_in:
                    with open(y_file, 'rb') as f_y_in:
                        while True:
                            tmp = pkl.load(f_x_in)
                            if type(tmp) is tuple and len(tmp) == 10:
                                nTrainSamples = tmp[1]
                                batchsize = int(nTrainSamples / self.nBatches)
                                print('\t\t## Number of training samples: {0:d}...'.format(
                                    nTrainSamples))
                                break
                        for batch in np.arange(self.nBatches):
                            y_file = list()
                            start_idx = 0
                            batch_timeref = time.time()
                            if batch == self.nBatches - 1:
                                batchsize = nTrainSamples - (batch * batchsize)
                            final_idx = start_idx + batchsize
                            featureMatNonItem = np.empty(
                                shape=(batchsize, featureSizeforX))
                            while start_idx < final_idx:
                                try:
                                    tmp = pkl.load(f_x_in)
                                    if type(tmp) is np.ndarray:
                                        featureMatNonItem[start_idx] = np.reshape(
                                            tmp, (tmp.shape[1],))
                                        start_idx += 1
                                    else:
                                        continue
                                except IOError:
                                    break

                            start_idx = 0
                            while start_idx < final_idx:
                                try:
                                    tmp = pkl.load(f_y_in)
                                    if type(tmp) is tuple:
                                        y_file.append(tmp[0])
                                        start_idx += 1
                                except IOError:
                                    break

                            if subSample:
                                # Try to add Nestrom based subsampling approach
                                featureMatNonItem, _, y_file, _ = train_test_split(featureMatNonItem, y_file,
                                                                                   shuffle=subSampleShuffle,
                                                                                   train_size=subsampleSize,
                                                                                   random_state=12345)
                                print('\t\t## Applied subsampled size {0:d} samples'.format(
                                    featureMatNonItem.shape[0]))
                            else:
                                print(
                                    '\t\t  ### Learning from {2:d} samples for the batch count {0:d} (out of {1:d})...'.format(
                                        batch + 1,
                                        self.nBatches,
                                        batchsize))

                            if self.binarizeAbundance:
                                preprocessing.binarize(
                                    featureMatNonItem[:, :self.nTotalComponents], copy=False)
                            self._fit(label=y_file, featureNonItem=featureMatNonItem, itemFeatures=itemFeatures,
                                      labelsComponents=labelsComponents, itemEvidenceFeaturesSize=itemEvidFeaturesSize)
                            print('\t\t\t--> Batch {0} consumed {1} seconds...'.format(batch + 1,
                                                                                       round(
                                                                                           time.time() - batch_timeref,
                                                                                           3)))
                self.is_fit = True
                print('\t\t## Epoch {0} took {1} seconds...'.format(
                    epoch + 1, round(time.time() - epoch_timeref, 3)))
                if (epoch % self.testInterval) == 0 or epoch == 0 or epoch + 1 == self.nEpochs:
                    print('\t\t## Evaluating performance...')
                    newCost = self._cost(
                        X_file=XdevFile, y_file=ydevFile, forTraining=True)
                    print(
                        '\t\t\t--> New cost: {0:.4f}; Old cost: {1:.4f}'.format(newCost, oldCost))
                    if newCost < oldCost:
                        oldCost = newCost
                        print('\t\t\t--> Storing the multi-label logistic regression train into the file: {0:s}'
                              .format(savename))
                        with open(file=fName, mode='wb') as fout:
                            pkl.dump(self, fout)
                    else:
                        print('\t\t\t--> The new cost is higher than the old cost...')

    def _predict(self, X, classIdx):
        '''
        Probability estimation for OvR logistic regression.
        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        '''
        pred = self._decision_function(X, classIdx)
        pred *= -1
        np.exp(pred, pred)
        pred += 1
        np.reciprocal(pred, pred)
        return classIdx, pred

    def predict(self, X_file, forTraining=False, applyTCriterion=False, estimateProb=False):
        '''
        Predict a list of pathways for a given set
            of features extracted from input set
        '''
        if not self.is_fit:
            raise Exception("This instance is not fitted yet. Call 'fit' with "
                            "appropriate arguments before using this method.")

        componentFeatures = self.nTotalComponents

        if self.useItemEvidenceFeatures != False:
            itemEvidenceFeaturesSize = int(
                self.nTotalClassEvidenceFeatures / self.nTotalClassLabels)
        else:
            itemEvidenceFeaturesSize = 0

        if self.itemPrintFeaturesFile is not None:
            itemFeatures = LoadItemFeatures(
                fname=self.itemPrintFeaturesFile, components=False)
        else:
            itemFeatures = None

        if self.useLabelComponentFeatures:
            labelsComponents = LoadItemFeatures(
                fname=self.labelsComponentsFile)
            labelsComponents = labelsComponents[0]
        else:
            labelsComponents = None

        featureSize = componentFeatures + self.nReacEvidenceFeatures + \
            self.nPossibleClassFeatures + self.nTotalClassEvidenceFeatures

        with open(X_file, 'rb') as f_in:
            while True:
                tmp = pkl.load(f_in)
                if type(tmp) is tuple and len(tmp) == 10:
                    nSamples = tmp[1]
                    y_pred = np.zeros((nSamples, len(self.mlb.classes)))
                    batchsize = int(nSamples / self.nBatches)
                    print(
                        '\t\t\t--> Number of samples to predict labels: {0:d}...'.format(nSamples))
                    break
            pred_idx = 0

            nBatches = self.nBatches
            if forTraining:
                nBatches = 1

            for batch in np.arange(nBatches):
                start_idx = 0
                if batch == nBatches - 1:
                    batchsize = nSamples - (batch * batchsize)
                final_idx = start_idx + batchsize
                featureNoItem = np.empty(shape=(batchsize, featureSize))
                print('\t\t## Predicting for {2:d} samples: {0:d} batch (out of {1:d})'.format(
                    batch + 1, self.nBatches, batchsize))
                while start_idx < final_idx:
                    try:
                        tmp = pkl.load(f_in)
                        if type(tmp) is np.ndarray:
                            tmp = tmp[:, :featureSize]
                            featureNoItem[start_idx] = np.reshape(
                                tmp, (tmp.shape[1],))
                            start_idx += 1
                    except Exception as e:
                        print(traceback.print_exc())
                        raise e

                if self.binarizeAbundance:
                    preprocessing.binarize(
                        featureNoItem[:, : self.nTotalComponents], copy=False)

                for classIdx, classLabel in enumerate(self.mlb.classes):
                    X = self._transformFeatures(classLabel, featureNoItem, itemEvidenceFeaturesSize,
                                                itemFeatures, labelsComponents, fit=False)
                    _, y_hat = self._predict(X=X, classIdx=classIdx)
                    y_pred[pred_idx:pred_idx + final_idx, classIdx] = y_hat
                pred_idx += final_idx

        if applyTCriterion:
            maxval = np.max(y_pred, axis=1) * self.adaptive_beta
            for sidx in np.arange(y_pred.shape[0]):
                y_pred[sidx][y_pred[sidx] >= maxval[sidx]] = 1

        if not estimateProb:
            y_pred[y_pred >= self.threshold] = 1
            y_pred[y_pred != 1] = 0

        return y_pred
