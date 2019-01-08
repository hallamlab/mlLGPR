'''
==================================================================
Launch application for parsing and training the metagenomics files
==================================================================
Author = Abdurrahman Abul-Basher
Version = 1.00
License = Hallam's lab
Email = ar.basher@alumni.ubc.ca
TimeStamp: Thursday, December 28, 2018

This file is the main entry used to extract metacyc information,
and perform learning and prediction on dataset using multi-label
logistic regression train.

===================================================================
'''

import os

try:
    import cPickle as pkl
except:
    import pickle as pkl
from utility import Path as path
from utility.Arguments import Arguments
from parsing.Parse_Input import InputMain
from parsing.Parse_BioCyc import BioCycMain
from train.Train import TrainMain
from eda.EDA import EDAMain


def LaunchApplication(biocyc, metagenomic, explore, train, predict=True, evaluate=True, load_object=True,
                      load_indicator=True, load_ptw_properties=True, load_ec_properties=True,
                      build_pathway_similarities=False, build_syn_dataset=False, build_synthetic_features=False,
                      build_golden_dataset=True, build_golden_features=True, extract_info_mg=False,
                      build_mg_features=False, ds_type="syn_ds", nSample=15000, load_prepared_dataset=True,
                      alpha=0.0001, l2_ratio=0.65, sigma=2, sub_sample=False, grid=False, adjust_by_similarity=False,
                      similarity_type="sw", customFit=False, useClipping=False, trained_model="mlLGPR_en_ab_re_pe.pkl",
                      n_jobs=1, nBatches=1, nEpochs=3, report=False, kbpath=path.DATABASE_PATH, ospath=path.OBJECT_PATH,
                      inputpath=path.INPUT_PATH, dspath=path.DATASET_PATH, mdpath=path.MODEL_PATH,
                      rspath=path.RESULT_PATH):
    arg = Arguments()

    ##########################################################################################################
    ##########                              ARGUMENTS FOR SAVING FILES                              ##########
    ##########################################################################################################

    if load_object:
        arg.save_builddata_kb = False
    else:
        arg.save_builddata_kb = True

    if load_indicator:
        arg.save_indicator = False
    elif not load_indicator:
        arg.save_indicator = True

    if load_ptw_properties:
        arg.build_pathway_properties = False
    else:
        arg.build_pathway_properties = True

    if load_ec_properties:
        arg.build_ec_properties = False
    else:
        arg.build_ec_properties = True

    arg.build_pathway_similarities = build_pathway_similarities

    ##########################################################################################################
    ##########                                  ARGUMENTS FOR PATHS                                 ##########
    ##########################################################################################################

    arg.kbpath = kbpath
    arg.ospath = ospath
    arg.inputpath = inputpath
    arg.dspath = dspath
    arg.mdpath = mdpath
    arg.rspath = rspath

    ##########################################################################################################
    ##########                          ARGUMENTS FOR FILE NAMES AND MODELS                         ##########
    ##########################################################################################################

    arg.objectname = 'object.pkl'
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
    arg.pathway_ec = 'pathway_ec.pkl'
    arg.syntheticdataset_ptw_ec = 'syn_dataset_ptw_ec'
    arg.syntheticdataset_rxn_ec = 'syn_dataset_rxn_ec'
    arg.goldendataset_ptw_ec = 'gold_dataset_ptw_ec'
    arg.goldendataset_rxn_ec = 'gold_dataset_rxn_ec'
    arg.mllr = 'mlLGPR'  # 'mlLGPR.pkl'
    arg.score_file = 'mlLGPR_scores'
    arg.predict_file = 'mlLGPR_labels'

    ##########################################################################################################
    ##########                             ARGUMENTS SHARED BY ALL FILES                            ##########
    ##########################################################################################################

    arg.display_interval = 50
    arg.n_jobs = n_jobs

    ##########################################################################################################
    ##########                  ARGUMENTS USED FOR CONSTRUCTING SYNTHETIC DATASET                   ##########
    ##########################################################################################################

    arg.add_noise = False
    arg.ncomponents_to_corrupt = 2
    arg.lower_bound_nitem_ptw = 1
    arg.ncomponents_to_corrupt_outside = 2
    arg.use_ec = True
    arg.construct_reaction = False
    arg.constraint_kb = 'metacyc'
    arg.constraint_pathway = True
    arg.nsample = nSample  # metagenomic dataset =418; golden dataset =63
    arg.average_item_per_sample = 500
    arg.include_kb = True
    arg.minpath_ds = True
    arg.minpath_map = True
    arg.pathologic_input = True
    arg.mapall = True
    arg.build_synthetic_features = build_synthetic_features
    arg.build_golden_dataset = build_golden_dataset
    arg.build_golden_features = build_golden_features
    if build_syn_dataset:
        arg.save_dataset = True
    elif not build_syn_dataset:
        arg.save_dataset = False

    ##########################################################################################################
    ##########                        ARGUMENTS USED FOR EXTRACTING FEATURES                        ##########
    ##########################################################################################################

    arg.num_pathwayfeatures = 27
    arg.num_ecfeatures = 25
    arg.num_reaction_evidence_features = 42
    arg.num_ec_evidence_features = 68
    arg.num_ptw_evidence_features = 32

    ##########################################################################################################
    ##########                     ARGUMENTS USED FOR INPUTS METAGENOMICS DATA                      ##########
    ##########################################################################################################

    arg.metegenomics_dataset = 'mg_dataset'
    arg.mapping = False
    arg.extract_info_mg = extract_info_mg
    arg.build_mg_features = build_mg_features

    ##########################################################################################################
    ##########                              ARGUMENTS USED FOR TRAINING                             ##########
    ##########################################################################################################

    arg.all_classes = False

    ## Set of features used in the experiment
    arg.useReacEvidenceFeatures = True  ## Evidence ec features (RE)
    arg.useItemEvidenceFeatures = True  ## Evidence pathway features (PE)
    arg.usePossibleClassFeatures = False  ## Possible pathway features (PP)
    arg.useLabelComponentFeatures = False  ## Pathway to ec common mapping features (PC)
    arg.useItemfeatures = False  ## ALWAYS SET TO FALSE: Pathway-Print features
    arg.scale_feature = False
    arg.binarize = True
    arg.norm_op = 'normalizer'
    arg.grid = grid
    arg.train_size = 0.8
    arg.val_size = 0.15
    arg.sub_sample = sub_sample
    arg.sub_sample_size = 0.3
    arg.nEpochs = nEpochs
    arg.nBatches = nBatches
    arg.test_interval = 2
    arg.random_state = 12345
    arg.ds_type = ds_type  # dataset types: syn_ds, meta_ds, gold_ds

    if load_prepared_dataset is True:
        arg.save_prepared_dataset = False
    elif load_prepared_dataset is False:
        arg.save_prepared_dataset = True
    else:
        arg.save_prepared_dataset = None

    arg.train = train
    arg.evaluate = evaluate
    arg.predict = predict

    if predict or evaluate:
        if not train:
            if os.path.exists(os.path.join(path.MODEL_PATH, trained_model)):
                arg.model = trained_model
            else:
                raise FileNotFoundError('Please provide a valid path to the already trained train')

    ###********************             LOGISTIC REGRESSION             ********************###

    arg.loss = 'log'
    arg.penalty = 'elasticnet'
    arg.Cs = 1.e4

    ## alpha controls the amount to regularize parameters
    ## default is 0.0001 (in our paper it is lambda)
    arg.alpha = alpha

    ## The Elastic Net mixing parameter, with 0 <= l2_ratio <= 1.
    ## l2_ratio=0 corresponds to L1 penalty, l2_ratio=1 to L2.
    ## Defaults to 0.65. It could be also used as a trade-off
    ## between l2 and laplacian regularization (in our paper it is alpha)
    arg.l1_ratio = 1 - l2_ratio

    ## sigma scales the amount of laplacian norm regularization
    ## paramters. Defaults to ....
    arg.sigma = sigma

    ## The adaptive paramter for logistic regression
    arg.use_tCriterion = False
    arg.adaptive_beta = 0.45

    ## This will enforce items to have similar weight vectors
    arg.adjust_by_similarity = adjust_by_similarity
    arg.similarity_type = similarity_type
    arg.fit_intercept = True
    arg.customFit = customFit
    arg.useClipping = useClipping
    arg.threshold = 0.5
    arg.learning_rate = "optimal"
    arg.eta0 = 0.0
    arg.power_t = 0.5
    arg.shuffle = True
    arg.max_inner_iter = 1500  # aka similar to nEpochs if nBatches = 1 better to set to 1500

    ##########################################################################################################
    ##########                             ARGUMENTS USED FOR REPORTING                             ##########
    ##########################################################################################################

    arg.report = report

    ##########################################################################################################
    ##########                              JOBMAN CHANNEL REPLACEMENT                              ##########
    ##########################################################################################################

    class Channel(object):
        def __init__(self, argument, train):
            self.argument = argument
            if train:
                print('*** Saving original training arguments into file: original_training_arguments.pkl')
                if not os.path.isdir(self.argument.mdpath):
                    os.mkdir(self.argument.mdpath)
                fname = os.path.join(self.argument.mdpath, 'original_training_arguments.pkl')
                with open(fname, 'wb') as f_out:
                    pkl.dump(self.argument, f_out, -1)
            self.COMPLETE = 1

        def save(self):
            print('\t--> Saving current training arguments into file: current_training_arguments.pkl')
            fname = os.path.join(self.argument.mdpath, 'current_training_arguments.pkl')
            with open(fname, 'wb') as f_out:
                pkl.dump(self.argument, f_out, -1)

        def load(self):
            print('\t--> Loading training arguments from file: current_training_arguments.pkl')
            fname = os.path.join(self.argument.mdpath, 'current_training_arguments.pkl')
            with open(fname, 'rb') as f_in:
                data = pkl.load(f_in)
            return data

    # ----------------------------------------------------------

    if biocyc:
        BioCycMain(b_arg=arg)

    if metagenomic:
        InputMain(m_arg=arg)

    if explore:
        EDAMain(e_arg=arg)

    if train or predict or evaluate:
        channel = Channel(argument=arg, train=train)
        TrainMain(t_arg=arg, channel=channel)


if __name__ == '__main__':
    os.system('clear')
    print(__doc__)
    # dataset types: syn_ds, meta_ds, gold_ds
    # metagenomic dataset =418; golden dataset =63
    # Set nEpochs=10 or high and nBatches=1 if grid search is enabled
    LaunchApplication(biocyc=False, metagenomic=False, explore=False, train=False, predict=False, evaluate=False,
                      load_object=False, load_indicator=False, load_ptw_properties=False, load_ec_properties=False,
                      build_pathway_similarities=True, build_syn_dataset=False, build_synthetic_features=False,
                      build_golden_dataset=True, build_golden_features=True, extract_info_mg=False,
                      build_mg_features=False, ds_type="syn_ds", nSample=15000, load_prepared_dataset=True,
                      alpha=0.0001, l2_ratio=0.4, sigma=2, sub_sample=False, grid=False, adjust_by_similarity=True,
                      similarity_type="sw", customFit=False, useClipping=True,
                      trained_model="mlLGPR_sw_adjusted_l2_ab_re_pe.pkl",
                      n_jobs=5, nBatches=1, nEpochs=3)
