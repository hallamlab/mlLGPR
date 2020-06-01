'''
==================================================================
Launch application for parsing and training the metagenomics files
==================================================================
Author = Abdurrahman Abul-Basher
Version = 1.00
License = Hallam's lab
Email = ar.basher@alumni.ubc.ca
TimeStamp: Thursday, December 28, 2019

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
from parsing.Parse_Input import InputMain
from parsing.Parse_BioCyc import BioCycMain
from train.Train import TrainMain


def internalArgs(parse_args):
    arg = Arguments()

    ##########################################################################################################
    ##########                              ARGUMENTS FOR SAVING FILES                              ##########
    ##########################################################################################################

    if parse_args.load_object:
        arg.save_builddata_kb = False
    else:
        arg.save_builddata_kb = True
    if parse_args.load_indicator:
        arg.save_indicator = False
    elif not parse_args.load_indicator:
        arg.save_indicator = True
    if parse_args.load_ptw_properties:
        arg.build_pathway_properties = False
    else:
        arg.build_pathway_properties = True
    if parse_args.load_ec_properties:
        arg.build_ec_properties = False
    else:
        arg.build_ec_properties = True

    ##########################################################################################################
    ##########                                  ARGUMENTS FOR PATHS                                 ##########
    ##########################################################################################################

    arg.kbpath = parse_args.kbpath
    arg.ospath = parse_args.ospath
    arg.inputpath = parse_args.inputpath
    arg.dspath = parse_args.dspath
    arg.mdpath = parse_args.mdpath
    arg.rspath = parse_args.rspath

    ##########################################################################################################
    ##########                          ARGUMENTS FOR FILE NAMES AND MODELS                         ##########
    ##########################################################################################################

    arg.objectname = parse_args.objectname
    arg.pathwayfeature = 'pathwayfeature.pkl'
    arg.ecfeature = 'ecfeature.pkl'
    arg.pathway_similarity = 'pathway_similarity'
    arg.syntheticfeature = 'syn_feature'
    arg.gene_go = 'gene_go.pkl'
    arg.gene_product = 'gene_product.pkl'
    arg.reaction_gene = 'reaction_gene.pkl'
    arg.reaction_ec = 'reaction_ec.pkl'
    arg.pathway_gene = 'pathway_gene.pkl'
    arg.pathway_reaction = 'pathway_reaction.pkl'
    arg.syntheticdataset_ptw_ec = 'syn_dataset_ptw_ec'
    arg.syntheticdataset_rxn_ec = 'syn_dataset_rxn_ec'
    arg.goldendataset_ptw_ec = 'gold_dataset_ptw_ec'
    arg.goldendataset_rxn_ec = 'gold_dataset_rxn_ec'
    arg.mllr = parse_args.mllr
    arg.score_file = parse_args.score_file
    arg.predict_file = parse_args.predict_file

    ##########################################################################################################
    ##########                             ARGUMENTS SHARED BY ALL FILES                            ##########
    ##########################################################################################################

    arg.display_interval = parse_args.display_interval
    arg.n_jobs = parse_args.n_jobs

    ##########################################################################################################
    ##########                  ARGUMENTS USED FOR CONSTRUCTING SYNTHETIC DATASET                   ##########
    ##########################################################################################################

    arg.add_noise = parse_args.add_noise
    arg.ncomponents_to_corrupt = parse_args.ncomponents_to_corrupt
    arg.lower_bound_nitem_ptw = parse_args.lower_bound_nitem_ptw
    arg.ncomponents_to_corrupt_outside = parse_args.ncomponents_to_corrupt_outside
    arg.use_ec = parse_args.use_ec
    arg.construct_reaction = parse_args.construct_reaction
    arg.constraint_kb = parse_args.constraint_kb
    arg.constraint_pathway = parse_args.constraint_pathway
    arg.nsample = parse_args.nSample  # metagenomic dataset =418; golden dataset =63
    arg.average_item_per_sample = parse_args.average_item_per_sample
    arg.minpath_ds = parse_args.minpath_ds
    arg.minpath_map = parse_args.minpath_map
    arg.pathologic_input = parse_args.pathologic_input
    arg.mapall = parse_args.mapall
    arg.build_synthetic_features = parse_args.build_synthetic_features
    arg.build_golden_dataset = parse_args.build_golden_dataset
    arg.build_golden_features = parse_args.build_golden_features
    if parse_args.build_syn_dataset:
        arg.save_dataset = True
    elif not parse_args.build_syn_dataset:
        arg.save_dataset = False

    ##########################################################################################################
    ##########                        ARGUMENTS USED FOR EXTRACTING FEATURES                        ##########
    ##########################################################################################################
    arg.num_pathwayfeatures = parse_args.num_pathwayfeatures
    arg.num_ecfeatures = parse_args.num_ecfeatures
    arg.num_reaction_evidence_features = parse_args.num_reaction_evidence_features
    arg.num_ec_evidence_features = parse_args.num_ec_evidence_features
    arg.num_ptw_evidence_features = parse_args.num_ptw_evidence_features

    ##########################################################################################################
    ##########                     ARGUMENTS USED FOR INPUTS METAGENOMICS DATA                      ##########
    ##########################################################################################################

    arg.metegenomics_dataset = parse_args.metegenomics_dataset
    arg.mapping = parse_args.mapping
    arg.extract_info_mg = parse_args.extract_info_mg
    arg.build_mg_features = parse_args.build_mg_features

    ##########################################################################################################
    ##########                              ARGUMENTS USED FOR TRAINING                             ##########
    ##########################################################################################################

    arg.all_classes = parse_args.all_classes
    arg.useReacEvidenceFeatures = parse_args.useReacEvidenceFeatures
    arg.useItemEvidenceFeatures = parse_args.useItemEvidenceFeatures
    arg.usePossibleClassFeatures = parse_args.usePossibleClassFeatures
    arg.useLabelComponentFeatures = parse_args.useLabelComponentFeatures
    arg.useItemfeatures = parse_args.useItemfeatures
    arg.report = parse_args.report
    arg.scale_feature = parse_args.scale_feature
    arg.binarize = parse_args.binarize
    arg.norm_op = parse_args.norm_op
    arg.grid = parse_args.grid
    arg.train_size = parse_args.train_size
    arg.val_size = parse_args.val_size
    arg.sub_sample = parse_args.sub_sample
    arg.sub_sample_size = parse_args.sub_sample_size
    arg.nEpochs = parse_args.nEpochs
    arg.nBatches = parse_args.nBatches
    arg.test_interval = parse_args.test_interval
    arg.random_state = parse_args.random_state
    arg.ds_type = parse_args.ds_type
    if parse_args.load_prepared_dataset is True:
        arg.save_prepared_dataset = False
    elif parse_args.load_prepared_dataset is False:
        arg.save_prepared_dataset = True
    else:
        arg.save_prepared_dataset = None

    arg.train = parse_args.train
    arg.evaluate = parse_args.evaluate
    arg.predict = parse_args.predict
    if parse_args.predict or parse_args.evaluate:
        if not parse_args.train:
            if os.path.exists(os.path.join(path.MODEL_PATH, parse_args.trained_model)):
                arg.model = parse_args.trained_model
            else:
                raise FileNotFoundError(
                    'Please provide a valid path to the already trained train')

    ###********************             LOGISTIC REGRESSION             ********************###
    arg.penalty = parse_args.penalty
    arg.alpha = parse_args.alpha
    arg.l1_ratio = 1 - parse_args.l2_ratio
    arg.sigma = 2
    arg.use_tCriterion = parse_args.use_tCriterion
    arg.fit_intercept = parse_args.fit_intercept
    arg.adaptive_beta = parse_args.adaptive_beta
    arg.threshold = parse_args.threshold
    arg.learning_rate = parse_args.learning_rate
    arg.shuffle = True
    arg.max_inner_iter = parse_args.max_inner_iter

    return arg


def parseCommandLine():
    """
    Returns the arguments from the command line.
    """
    parser = ArgumentParser()

    ##########################################################################################################
    ##########                             ARGUMENTS SHARED BY ALL FILES                            ##########
    ##########################################################################################################

    parser.add_argument('--display_interval', default=50, type=int,
                        help='How often to display results in # of iterations. (default value: 50)')
    parser.add_argument('--n_jobs', default=2, type=int,
                        help='Number of CPU cores to be consumed. (default value: 2)')

    ##########################################################################################################
    ##########                              ARGUMENTS FOR SAVING FILES                              ##########
    ##########################################################################################################

    parser.add_argument('--load_object', action='store_false', default=True,
                        help='Whether to load object data that contains MetaCyc information. (default value: true)')
    parser.add_argument('--load_indicator', action='store_false', default=True,
                        help='Whether to load pathway EC mapping matrix. (default value: true)')
    parser.add_argument('--load_ptw_properties', action='store_false', default=True,
                        help='Whether to load properties of pathways. (default value: true)')
    parser.add_argument('--load_ec_properties', action='store_false', default=True,
                        help='Whether to load properties of ECs. (default value: true)')

    ##########################################################################################################
    ##########                                  ARGUMENTS FOR PATHS                                 ##########
    ##########################################################################################################

    parser.add_argument('--kbpath', default=path.DATABASE_PATH, type=str,
                        help='The path to the MetaCyc database. The default is set to database folder outside the source code.')
    parser.add_argument('--ospath', default=path.OBJECT_PATH, type=str,
                        help='The path to the data object that contains extracted information from the MetaCyc database. The default is set to object folder outside the source code.')
    parser.add_argument('--inputpath', default=path.INPUT_PATH, type=str,
                        help='The path to the input data as represented by PathoLogic input format. The default is set to inputset folder outside the source code.')
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
    parser.add_argument('--mllr', type=str, default='mlLGPR',
                        help='The train name. (default value: "mlLGPR")')
    parser.add_argument('--score_file', type=str, default='mlLGPR_scores',
                        help='The file name to store the scores of results. (default value: "mlLGPR_scores")')
    parser.add_argument('--predict_file', type=str, default='mlLGPR_labels',
                        help='The file name to store the predicted pathways. (default value: "mlLGPR_labels")')

    ##########################################################################################################
    ##########                  ARGUMENTS USED FOR CONSTRUCTING SYNTHETIC DATASET                   ##########
    ##########################################################################################################

    parser.add_argument('--build_syn_dataset', action='store_true', default=False,
                        help='Whether to create synthetic dataset. (default value: false)')
    parser.add_argument('--build_synthetic_features', action='store_true', default=False,
                        help='Whether to build featrues for synthetic dataset. (default value: false)')
    parser.add_argument('--build_golden_dataset', action='store_true', default=False,
                        help='Whether to create golden dataset. (default value: false)')
    parser.add_argument('--build_golden_features', action='store_true', default=False,
                        help='Whether to build featrues for golden dataset. (default value: false)')
    parser.add_argument('--add_noise', action='store_true', default=True,
                        help='Whether to add noise in creating synthetic samples. (default value: false)')
    parser.add_argument('--ncomponents_to_corrupt', default=2, type=int,
                        help='Number of true components to be corrupted in creating synthetic samples. (default value: 2)')
    parser.add_argument('--lower_bound_nitem_ptw', default=1, type=int,
                        help='The lower bound of components in which corruption should not be proceeded for a pathway in creating synthetic samples. (default value: 1)')
    parser.add_argument('--ncomponents_to_corrupt_outside', default=2, type=int,
                        help='Number of false components to be added in creating synthetic samples. (default value: 2)')
    parser.add_argument('--use_ec', action='store_false', default=True,
                        help='Whether to use EC. (default value: true)')
    parser.add_argument('--construct_reaction', action='store_true', default=False,
                        help='Whether to construct reactions samples. (default value: false)')
    parser.add_argument('--constraint_kb', default='metacyc', type=str,
                        help='The type of database is constrained onto. (default value: "metacyc")')
    parser.add_argument('--constraint_pathway', action='store_false', default=True,
                        help='Whether to use pathways is creating synthetic samples. (default value: true)')
    parser.add_argument('--nSample', default=15000, type=int,
                        help='Number of synthetic samples to generate. (default value: 15000)')
    parser.add_argument('--average_item_per_sample', default=500, type=int,
                        help='Average number of pathways for each sample in generating samples. (default value: 500)')
    parser.add_argument('--minpath_ds', action='store_false', default=True,
                        help='Create a dataset for MinPath. ((default value: true))')
    parser.add_argument('--minpath_map', action='store_false', default=True,
                        help='Create a mapping file for  MinPath. ((default value: true))')
    parser.add_argument('--pathologic_input', action='store_false', default=True,
                        help='Create a dataset for PathoLogic. ((default value: true))')
    parser.add_argument('--mapall', action='store_false', default=True,
                        help='Referencing labels (either pathways or reactions) with functions (either gene or ec). This is used to create data for MinPath. ((default value: true))')

    ##########################################################################################################
    ##########                        ARGUMENTS USED FOR EXTRACTING FEATURES                        ##########
    ##########################################################################################################

    parser.add_argument('--num_pathwayfeatures', default=27, type=int,
                        help='The number of pathway featrues. It is not recommended to override the default value. 27(default value: 42)')
    parser.add_argument('--num_ptw_evidence_features', default=32, type=int,
                        help='The number of pathway evidence featrues. It is not recommended to override the default value. (default value: 32)')
    parser.add_argument('--num_ecfeatures', default=25, type=int,
                        help='The number of EC featrues. It is not recommended to override the default value. (default value: 27)')
    parser.add_argument('--num_ec_evidence_features', default=68, type=int,
                        help='The number of EC evidence featrues. It is not recommended to override the default value. (default value: 68)')
    parser.add_argument('--num_reaction_evidence_features', default=42, type=int,
                        help='The number of reaction evidence featrues. It is not recommended to override the default value. (default value: 42)')

    ##########################################################################################################
    ##########                     ARGUMENTS USED FOR INPUTS METAGENOMICS DATA                      ##########
    ##########################################################################################################

    parser.add_argument('--metegenomics_dataset', default='mg_dataset', type=str,
                        help='The file name to store the metagenomics datasets. (default value: "mg_dataset")')
    parser.add_argument('--mapping', action='store_true', default=True,
                        help='Mapping labels with functions. This is used to create data inputs for MinPath from metagenomics. ((default value: true))')
    parser.add_argument('--extract_info_mg', action='store_true', default=True,
                        help='Extract information from metagenomics datasets that are in the PathoLogic input format. ((default value: true))')
    parser.add_argument('--build_mg_features', action='store_true', default=True,
                        help='Whether build featrues from metagenomics datasets. ((default value: true))')

    ##########################################################################################################
    ##########                              ARGUMENTS USED FOR TRAINING                             ##########
    ##########################################################################################################

    parser.add_argument('--all_classes', action='store_true', default=False,
                        help='Whether to create multi-label datasets using all labels. (default value: false). Usually, it should be set once.')
    parser.add_argument('--useReacEvidenceFeatures', action='store_false', default=True,
                        help='Whether to apply evidence ec features (RE) in training. (default value: true)')
    parser.add_argument('--useItemEvidenceFeatures', action='store_false', default=True,
                        help='Whether to apply evidence pathway features (PE) in training. (default value: true)')
    parser.add_argument('--usePossibleClassFeatures', action='store_true', default=False,
                        help='Whether to apply possible pathway features (PP) in training. (default value: false)')
    parser.add_argument('--useLabelComponentFeatures', action='store_true', default=False,
                        help='Whether to apply pathway to ec common mapping features (PC) in training. (default value: false)')
    parser.add_argument('--useItemfeatures', action='store_true', default=False,
                        help='Whether to apply pathway print features in training. It is recommended to set it to false. (default value: false)')
    parser.add_argument('--report', action='store_true', default=False,
                        help='Whether to report the generated outputs as PGDB files from mlLGPR train. (default value: false). Not yet implemeneted.')
    parser.add_argument('--scale_feature', action='store_true', default=False,
                        help='Whether to scale features. (default value: false)')
    parser.add_argument('--binarize', action='store_false', default=True,
                        help='Whether binarize data (set feature values to 0 or 1). (default value: true)')
    parser.add_argument('--norm_op', default='normalizer', type=str, choices=['minmax', 'maxabs', 'standard',
                                                                              'robust', 'normalizer'],
                        help='Transforms features by applying an appropriate feature scaling method. Possible choices are minmax, maxabs, standard, robust, or normalizer. (default value: "normalizer")')
    parser.add_argument('--train_size', default=0.8, type=float,
                        help='The training size of type float between 0.0 and 1.0 for mlLGPR. (default value: 0.8)')
    parser.add_argument('--val_size', default=0.15, type=float,
                        help='The validation size of type float between0.0 and 1.0 for mlLGPR. (default value: 0.15)')
    parser.add_argument('--sub_sample', action='store_true', default=False,
                        help='Subsample the datasets. (default value: false)')
    parser.add_argument('--sub_sample_size', default=0.3, type=float,
                        help='The size of subsamples. (default value: 0.3)')
    parser.add_argument('--nEpochs', default=10, type=int,
                        help='The number of epochs to train mlLGPR. (default value: 10)')
    parser.add_argument('--nBatches', default=5, type=int,
                        help='The size of a single mini-batch for training mlLGPR. (default value: 5)')
    parser.add_argument('--test_interval', default=2, type=int,
                        help='How often to test mlLGPR\'s performance. (default value: 2)')
    parser.add_argument('--random_state', default=12345,
                        type=int, help='Random seed. (default value: 12345)')
    parser.add_argument('--ds_type', default='syn_ds', type=str, choices=['syn_ds', 'meta_ds', 'gold_ds'],
                        help='Dataset types: syn_ds, meta_ds, gold_ds. (default value: "syn_ds")')
    parser.add_argument('--load_prepared_dataset', action='store_false', default=True,
                        help='Load datasets after applying stratified sampling approach. Usually, it is applied once. (default value: false)')
    parser.add_argument('--trained_model', default='mlLGPR_en_ab_re_pe.pkl', type=str,
                        help='The train file name. (default value: "mlLGPR_en_ab_re_pe.pkl")')

    ###********************             LOGISTIC REGRESSION             ********************###

    parser.add_argument('--penalty', default='elasticnet', type=str, choices=['l1', 'l2', 'elasticnet'],
                        help='The penalty (aka regularization term) to be used. (default value: "elasticnet")')
    parser.add_argument('--alpha', default=0.0001, type=float,
                        help='Constant that multiplies the regularization term to control the amount to regularize parameters and in our paper it is lambda. (default value: 0.0001)')
    parser.add_argument('--l2_ratio', default=0.65, type=float,
                        help='The ellasticnet mixing parameter, with 0 <= l2_ratio <= 1. l2_ratio=0 corresponds to L1 penalty, l2_ratio=1 to L2. It could be also used as a trade-off between l2 and laplacian regularization and in our paper it is alpha. (default value: 0.65)')
    parser.add_argument('--use_tCriterion', action='store_true', default=False,
                        help='Whether to employ adaptive mlLGPR strategy. (default value: false)')
    parser.add_argument('--adaptive_beta', default=0.45, type=float,
                        help='The adaptive beta paramter for mlLGPR. (default value: 0.45)')
    parser.add_argument('--fit_intercept', action='store_false', default=True,
                        help='Whether the intercept should be estimated or not. (default value: true)')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='The cutoff threshold for mlLGPR. (default value: 0.5)')
    parser.add_argument('--learning_rate', default='optimal', type=str, choices=['constant', 'invscaling', 'adaptive',
                                                                                 'optimal'],
                        help='The learning rate schedule. (default value: "optimal")')
    parser.add_argument('--max_inner_iter', default=1500, type=float,
                        help='Similar to nEpochs if nBatches = 1 better to set to 1500. (default value: 1500)')

    ##########################################################################################################
    ##########                               ARGUMENTS FOR USABILITY                                ##########
    ##########################################################################################################

    parser.add_argument('--biocyc', action='store_true', default=False,
                        help='Whether to parse files from biocyc. (default value: false)')
    parser.add_argument('--metagenomic', action='store_true', default=False,
                        help='Whether to parse files from metagenomic data. (default value: false)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train the mlLGPR train. (default value: false)')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='Whether to predict sets of pathways from inputs using mlLGPR. (default value: true)')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate mlLGPR\'s performances. (default value: false)')

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

    if parse_args.metagenomic:
        InputMain(m_arg=args)

    if parse_args.train or parse_args.predict or parse_args.evaluate:
        channel = Channel(argument=args, train=parse_args.train)
        TrainMain(t_arg=args, channel=channel)

    if not parse_args.biocyc and not parse_args.metagenomic and not parse_args.train and not parse_args.predict and not parse_args.evaluate:
        print('\n*** NOT ACTIONS WERE CHOSEN...')


if __name__ == '__main__':
    os.system('clear')
    print(__doc__)
    # dataset types: syn_ds, meta_ds, gold_ds
    # metagenomic dataset =418; golden dataset =63
    parseCommandLine()
