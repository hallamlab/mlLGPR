'''
==================================================================
Launch application for parsing and training the metagenomics files
==================================================================
Author = Abdurrahman Abul-Basher
Version = 1.00
License = Hallam's lab
Email = ar.basher@alumni.ubc.ca
TimeStamp: Thursday, June 02, 2020

This file is the main entry used to extract metacyc information,
and perform learning and prediction on dataset using multi-label
logistic regression train.
===================================================================
'''

import os
from argparse import ArgumentParser
try:
    import cPickle as pkl
except:
    import pickle as pkl
from utility import Path as path
from utility.Arguments import Arguments
from parsing.Parse_BioCyc import BioCycMain
from train.Train import TrainMain


def internalArgs(parse_args):
    arg = Arguments()
    arg.display_interval = parse_args.display_interval
    arg.n_jobs = parse_args.n_jobs

    ##########################################################################################################
    ##########                                  ARGUMENTS FOR PATHS                                 ##########
    ##########################################################################################################

    arg.kbpath = parse_args.kbpath
    arg.ospath = parse_args.ospath
    arg.dspath = parse_args.dspath
    arg.mdpath = parse_args.mdpath
    arg.rspath = parse_args.rspath

    ##########################################################################################################
    ##########                          ARGUMENTS FOR FILE NAMES AND MODELS                         ##########
    ##########################################################################################################

    arg.objectname = parse_args.objectname
    arg.pathwayfeature = parse_args.pathwayfeature
    arg.ecfeature = parse_args.ecfeature
    arg.pathway_ec = parse_args.pathway_ec
    arg.reaction_ec = parse_args.reaction_ec
    arg.model = parse_args.model
    arg.X_name = parse_args.X_name
    arg.y_name = parse_args.y_name
    arg.file_name = parse_args.file_name
    arg.score_file = parse_args.score_file
    arg.predict_file = parse_args.predict_file

    ##########################################################################################################
    ##########                  ARGUMENTS USED FOR CONSTRUCTING SYNTHETIC DATASET                   ##########
    ##########################################################################################################
    
    if parse_args.load_object:
        arg.save_builddata_kb = False
    else:
        arg.save_builddata_kb = True
    arg.add_noise = parse_args.add_noise
    arg.ncomponents_to_corrupt = parse_args.ncomponents_to_corrupt
    arg.lower_bound_nitem_ptw = parse_args.lower_bound_nitem_ptw
    arg.ncomponents_to_corrupt_outside = parse_args.ncomponents_to_corrupt_outside
    arg.nsample = parse_args.nSample  # metagenomic dataset =418; golden dataset =63
    arg.average_item_per_sample = parse_args.average_item_per_sample
    arg.build_golden_dataset = parse_args.build_golden_dataset
    if parse_args.build_syn_dataset:
        arg.save_dataset = True
    elif not parse_args.build_syn_dataset:
        arg.save_dataset = False

    ##########################################################################################################
    ##########                     ARGUMENTS USED FOR INPUTS METAGENOMICS DATA                      ##########
    ##########################################################################################################

    arg.metegenomics_dataset = parse_args.metegenomics_dataset
    arg.mapping = parse_args.mapping
    arg.extract_info_mg = parse_args.extract_info_mg

    ##########################################################################################################
    ##########                              ARGUMENTS USED FOR TRAINING                             ##########
    ##########################################################################################################

    if parse_args.load_prepared_dataset is True:
        arg.save_prepared_dataset = False
    elif parse_args.load_prepared_dataset is False:
        arg.save_prepared_dataset = True
    else:
        arg.save_prepared_dataset = None
    arg.all_classes = parse_args.all_classes
    arg.train_size = parse_args.train_size
    arg.val_size = parse_args.val_size

    arg.useReacEvidenceFeatures = parse_args.useReacEvidenceFeatures
    arg.useItemEvidenceFeatures = parse_args.useItemEvidenceFeatures
    arg.usePossibleClassFeatures = parse_args.usePossibleClassFeatures
    arg.useLabelComponentFeatures = parse_args.useLabelComponentFeatures

    arg.binarize = parse_args.binarize
    arg.penalty = parse_args.penalty
    arg.alpha = parse_args.alpha
    arg.l1_ratio = 1 - parse_args.l2_ratio
    arg.use_tCriterion = parse_args.use_tCriterion
    arg.adaptive_beta = parse_args.adaptive_beta
    arg.threshold = parse_args.threshold
    arg.max_inner_iter = parse_args.max_inner_iter
    arg.nEpochs = parse_args.nEpochs
    arg.nBatches = parse_args.nBatches
    arg.test_interval = parse_args.test_interval

    arg.train = parse_args.train
    arg.predict = parse_args.predict
    arg.parse_input = parse_args.parse_input
    return arg


def parseCommandLine():
    """
    Returns the arguments from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument('--display_interval', default=10, type=int,
                        help='How often to display results in # of iterations. (default value: 10)')
    parser.add_argument('--n-jobs', default=1, type=int,
                        help='Number of CPU cores to be consumed. (default value: 2)')

    ##########################################################################################################
    ##########                                  ARGUMENTS FOR PATHS                                 ##########
    ##########################################################################################################

    parser.add_argument('--kbpath', default=path.DATABASE_PATH, type=str,
                        help='The path to the MetaCyc database. The default is set to database folder outside the source code.')
    parser.add_argument('--ospath', default=path.OBJECT_PATH, type=str,
                        help='The path to the data object that contains extracted information from the MetaCyc database. The default is set to object folder outside the source code.')
    parser.add_argument('--dspath', default=path.DATASET_PATH, type=str,
                        help='The path to the dataset after the samples are processed. The default is set to dataset folder outside the source code.')
    parser.add_argument('--mdpath', default=path.MODEL_PATH, type=str,
                        help='The path to the output models. The default is set to train folder outside the source code.')
    parser.add_argument('--rspath', default=path.RESULT_PATH, type=str,
                        help='The path to the results. The default is set to result folder outside the source code.')

    ##########################################################################################################
    ##########                          ARGUMENTS FOR FILE NAMES AND MODELS                         ##########
    ##########################################################################################################

    parser.add_argument('--objectname', type=str, default='object.pkl',
                        help='The file name for the object data. (default value: "object.pkl")')
    parser.add_argument('--pathwayfeature', type=str, default='pathwayfeature.pkl',
                        help='The file name to store the featrues of pathways. (default value: "pathwayfeature.pkl")')
    parser.add_argument('--ecfeature', type=str, default='ecfeature.pkl',
                        help='The file name to store the featrues of ECs. (default value: "ecfeature.pkl")')
    parser.add_argument('--score-file', type=str, default='mlLGPR_scores',
                        help='The file name to store the scores of results. (default value: "mlLGPR_scores")')
    parser.add_argument('--predict-file', type=str, default='mlLGPR_labels',
                        help='The file name to store the predicted pathways. (default value: "mlLGPR_labels")')
    parser.add_argument('--pathway-ec', type=str, default='pathway_ec.pkl',
                        help='The file name to store Pathway to EC mapping data. (default value: "pathway_ec.pkl")')
    parser.add_argument('--reaction-ec', type=str, default='reaction_ec.pkl',
                        help='The file name to store Reaction to EC mapping data. (default value: "reaction_ec.pkl")')
    parser.add_argument('--file-name', type=str, default='synset',
                        help='The file name to store or load prepared data files. (default value: "synset")')
    parser.add_argument('--X-name', type=str, default='synset_X.pkl',
                        help='The X file name. (default value: "synset_X.pkl")')
    parser.add_argument('--y-name', type=str, default='synset_y.pkl',
                        help='The y file name. (default value: "synset_y.pkl")')
    parser.add_argument('--model', default='mlLGPR_en_ab_re_pe.pkl', type=str,
                        help='The train file name. (default value: "mlLGPR_en_ab_re_pe.pkl")')

    ##########################################################################################################
    ##########                           ARGUMENTS USED FOR PREPROCESSING                           ##########
    ##########################################################################################################

    parser.add_argument('--load-object', action='store_false', default=True,
                        help='Whether to load object data that contains MetaCyc information. (default value: True)')
    parser.add_argument('--build-syn-dataset', action='store_true', default=False,
                        help='Whether to create synthetic dataset. (default value: False)')
    parser.add_argument('--add-noise', action='store_true', default=False,
                        help='Whether to add noise in creating synthetic samples. (default value: False)')
    parser.add_argument('--ncomponents-to-corrupt', default=2, type=int,
                        help='Number of True components to be corrupted in creating synthetic samples. (default value: 2)')
    parser.add_argument('--lower-bound-nitem-ptw', default=1, type=int,
                        help='The lower bound of components in which corruption should not be proceeded for a pathway in creating synthetic samples. (default value: 1)')
    parser.add_argument('--ncomponents-to-corrupt-outside', default=2, type=int,
                        help='Number of False components to be added in creating synthetic samples. (default value: 2)')
    parser.add_argument('--nSample', default=10, type=int,
                        help='Number of synthetic samples to generate. (default value: 10)')
    parser.add_argument('--average-item-per-sample', default=500, type=int,
                        help='Average number of pathways for each sample in generating samples. (default value: 500)')
    parser.add_argument('--build-golden-dataset', action='store_true', default=False,
                        help='Whether to create golden dataset. (default value: False)')

    ##########################################################################################################
    ##########                     ARGUMENTS USED FOR INPUTS METAGENOMICS DATA                      ##########
    ##########################################################################################################

    parser.add_argument('--metegenomics-dataset', default='mg_dataset', type=str,
                        help='The file name to store the metagenomics datasets. (default value: "mg_dataset")')
    parser.add_argument('--mapping', action='store_true', default=True,
                        help='Mapping labels with functions. This is used to create data inputs for MinPath from metagenomics. ((default value: True))')
    parser.add_argument('--extract-info-mg', action='store_true', default=True,
                        help='Extract information from metagenomics datasets that are in the PathoLogic input format. ((default value: True))')

    ##########################################################################################################
    ##########                              ARGUMENTS USED FOR TRAINING                             ##########
    ##########################################################################################################

    parser.add_argument('--load-prepared-dataset', action='store_false', default=True,
                        help='Load datasets after applying stratified sampling approach. Usually, it is applied once. (default value: False)')
    parser.add_argument('--train-size', default=0.8, type=float,
                        help='The training size of type float between 0.0 and 1.0 for mlLGPR. (default value: 0.8)')
    parser.add_argument('--val-size', default=0.15, type=float,
                        help='The validation size of type float between0.0 and 1.0 for mlLGPR. (default value: 0.15)')
    parser.add_argument('--all-classes', action='store_true', default=False,
                        help='Whether to create multi-label datasets using all labels. (default value: False). Usually, it should be set once.')

    parser.add_argument('--useReacEvidenceFeatures', action='store_false', default=True,
                        help='Whether to apply evidence ec features (RE) in training. (default value: True)')
    parser.add_argument('--useItemEvidenceFeatures', action='store_false', default=True,
                        help='Whether to apply evidence pathway features (PE) in training. (default value: True)')
    parser.add_argument('--usePossibleClassFeatures', action='store_true', default=False,
                        help='Whether to apply possible pathway features (PP) in training. (default value: False)')
    parser.add_argument('--useLabelComponentFeatures', action='store_true', default=False,
                        help='Whether to apply pathway to ec common mapping features (PC) in training. (default value: False)')

    parser.add_argument('--binarize', action='store_false', default=True,
                        help='Whether binarize data (set feature values to 0 or 1). (default value: True)')
    parser.add_argument('--penalty', default='elasticnet', type=str, choices=['l1', 'l2', 'elasticnet'],
                        help='The penalty (aka regularization term) to be used. (default value: "elasticnet")')
    parser.add_argument('--alpha', default=0.0001, type=float,
                        help='Constant that multiplies the regularization term to control the amount to regularize parameters. (default value: 0.0001)')
    parser.add_argument('--l2-ratio', default=0.65, type=float,
                        help='The elasticnet mixing parameter, with 0 <= l2_ratio <= 1. l2_ratio=0 corresponds to L1 penalty, l2_ratio=1 to L2. (default value: 0.65)')
    parser.add_argument('--use-tCriterion', action='store_true', default=False,
                        help='Whether to employ adaptive mlLGPR strategy. (default value: False)')
    parser.add_argument('--adaptive-beta', default=0.45, type=float,
                        help='The adaptive beta paramter for mlLGPR. (default value: 0.45)')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='The cutoff threshold for mlLGPR. (default value: 0.5)')
    parser.add_argument('--max-inner-iter', default=100, type=float,
                        help='Similar to nEpochs if nBatches = 1 better to set to 100. (default value: 100)')
    parser.add_argument('--nEpochs', default=1, type=int,
                        help='The number of epochs to train mlLGPR. (default value: 3)')
    parser.add_argument('--nBatches', default=1, type=int,
                        help='The size of a single mini-batch for training/predicting mlLGPR. (default value: 1)')
    parser.add_argument('--test-interval', default=2, type=int,
                        help='How often to test mlLGPR\'s performance. (default value: 2)')
        
    ##########################################################################################################
    ##########                               ARGUMENTS FOR USABILITY                                ##########
    ##########################################################################################################

    parser.add_argument('--biocyc', action='store_true', default=False,
                        help='Whether to parse files from biocyc. (default value: False)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train the mlLGPR train. (default value: False)')
    parser.add_argument('--parse-input', action='store_true', default=False,
                        help='Whether to parse files from input in 0.pf format. (default value: False)')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='Whether to predict sets of pathways from inputs (in matrix format) using mlLGPR. (default value: True)')

    ##########################################################################################################
    ##########                              JOBMAN CHANNEL REPLACEMENT                              ##########
    ##########################################################################################################

    class Channel(object):
        def __init__(self, argument, train):
            self.argument = argument
            if train:
                print(
                    '*** Saving original training arguments into file: original_training_arguments.pkl')
                if not os.path.isdir(self.argument.mdpath):
                    os.mkdir(self.argument.mdpath)
                fname = os.path.join(self.argument.mdpath,
                                     'original_training_arguments.pkl')
                with open(fname, 'wb') as f_out:
                    pkl.dump(self.argument, f_out, -1)
            self.COMPLETE = 1

        def save(self):
            print(
                '\t--> Saving current training arguments into file: current_training_arguments.pkl')
            fname = os.path.join(self.argument.mdpath,
                                 'current_training_arguments.pkl')
            with open(fname, 'wb') as f_out:
                pkl.dump(self.argument, f_out, -1)

        def load(self):
            print(
                '\t--> Loading training arguments from file: current_training_arguments.pkl')
            fname = os.path.join(self.argument.mdpath,
                                 'current_training_arguments.pkl')
            with open(fname, 'rb') as f_in:
                data = pkl.load(f_in)
            return data

    # ----------------------------------------------------------

    parse_args = parser.parse_args()
    args = internalArgs(parse_args)

    if parse_args.biocyc:
        BioCycMain(b_arg=args)

    if parse_args.train or parse_args.predict:
        channel = Channel(argument=args, train=parse_args.train)
        TrainMain(t_arg=args, channel=channel)

    if not parse_args.biocyc and not parse_args.train and not parse_args.predict:
        print('\n*** NOT ACTIONS WERE CHOSEN...')


if __name__ == '__main__':
    os.system('clear')
    print(__doc__)
    # dataset types: syn_ds, meta_ds, gold_ds
    # metagenomic dataset =418; golden dataset =63
    parseCommandLine()
