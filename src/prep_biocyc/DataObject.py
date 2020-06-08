'''
This file preprocesses BioCyc PGDBs and contains functions
to construct synthetic and golden datasets. Also, this module
includes constructing similarity of pathways, such as
smith-waterman algorithm.
'''

import networkx as nx
import numpy as np
import os
import re
import sys
import traceback
from collections import OrderedDict
from fuzzywuzzy import fuzz
from itertools import combinations
from multiprocessing import Pool
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import cosine_similarity, chi2_kernel

try:
    import cPickle as pkl
except:
    import pickle as pkl
from operator import itemgetter

from prep_biocyc.Enzyme import Enzyme
from prep_biocyc.Gene import Gene
from prep_biocyc.Pathway import Pathway
from prep_biocyc.Protein import Protein
from prep_biocyc.Reaction import Reaction
from feature.Feature import Feature


class DataObject(object):
    # INITIALIZATION ------------------------------------------------------------------------

    def __init__(self):
        self.lst_kbpaths = list()
        self.processedKB = OrderedDict()

        # List of ids
        self.protein_id = OrderedDict()
        self.gene_id = OrderedDict()
        self.enzyme_id = OrderedDict()
        self.reaction_id = OrderedDict()
        self.pathway_id = OrderedDict()
        self.ec_id = OrderedDict()
        self.gene_name_id = OrderedDict()
        self.go_id = OrderedDict()
        self.product_id = OrderedDict()

    # ---------------------------------------------------------------------------------------

    # BUILD DATA FROM DATABASES -------------------------------------------------------------

    def ExtractInfoFromDatabase(self, kbpath, dataFolder, constraintKB='metacyc'):
        """ Build data from a given knowledge-base path
        :type kbpath: str
        :param kbpath: The RunPathoLogic knowledge base path, where all the data folders
                            are located
        :type dataFolder: str
        :param dataFolder: The data folder under a particular knowledge base
                           that includes the files for data preprocessing
        """

        self.lst_kbpaths = [os.path.join(kbpath, folder, dataFolder) for folder in os.listdir(kbpath)
                            if not folder.startswith('.')]

        print('\t>> Building from {0} databases...'.format(len(self.lst_kbpaths)))
        for (index, kbpath) in enumerate(self.lst_kbpaths):
            if os.path.isdir(kbpath) or os.path.exists(kbpath):
                core_dbname = str(kbpath.split('/')[-2]).upper()
                self.lst_kbpaths[index] = core_dbname.lower()
                print('\t\t{0:d})- {1:s} (progress: {3:.2f}%, {0:d} out of {2:d}):'
                      .format(index + 1, core_dbname, len(self.lst_kbpaths),
                              (index + 1) * 100.00 / len(self.lst_kbpaths)))

                # Objects
                gene = Gene()
                protein = Protein()
                enzyme = Enzyme()
                reaction = Reaction()
                pathway = Pathway()

                # List of objects to be passed in processing
                ids_lst = [self.protein_id, self.go_id, self.gene_id, self.gene_name_id,
                           self.product_id, self.enzyme_id, self.reaction_id,
                           self.ec_id, self.pathway_id]

                # Process proteins
                protein.ProcessProteins(pr_id=0, go_id=1, lst_ids=ids_lst, data_path=kbpath)

                # Process genes
                gene.ProcessGenesDat(g_id=2, g_name_id=3, lst_ids=ids_lst, data_path=kbpath)
                gene.ProcessGenesCol(g_id=2, g_name_id=3, pd_id=4, lst_ids=ids_lst, data_path=kbpath)
                gene.AddProteinInfo2GeneInfo(protein_info=protein.protein_info, go_position=4, catalyzes_position=2,
                                             product_name_position=1, pd_id=4, lst_ids=ids_lst)

                # Process enzymes
                enzyme.ProcessEnzymesCol(e_id=5, lst_ids=ids_lst, data_path=kbpath)
                enzyme.ProcessEnzymaticReactionsDat(e_id=5, lst_ids=ids_lst, data_path=kbpath)

                # Process reactions
                reaction.ProcessReactions(r_id=6, lst_ids=ids_lst, data_path=kbpath)
                reaction.AddEC2Reactions(ec_id=7, lst_ids=ids_lst, data_path=kbpath)

                # Process pathways
                pathway.ProcessPathways(p_id=8, lst_ids=ids_lst, data_path=kbpath)
                pathway.ProcessPathwaysCol(p_id=8, lst_ids=ids_lst, data_path=kbpath, header=False)
                pathway.AddPathwayProperties(reactions_info=reaction.reaction_info, ec_position=3, inptw_position=4,
                                             orphn_position=5, spon_position=12)

                gene.AddPathwayGenes2GenesID(pathway_info=pathway.pathway_info, gene_id_position=12,
                                             gene_name_id_position=11, g_id=2, g_name_id=3, lst_ids=ids_lst)
                reaction.AddGenes2Reactions(genes_info=gene.gene_info, gene_name_id_position=1, reaction_position=5)
                gene.AddReactionGenes2GenesID(reaction_info=reaction.reaction_info, gene_name_id_position=13,
                                              g_name_id=3, lst_ids=ids_lst)
                datum = {core_dbname.lower(): (protein, gene, enzyme, reaction, pathway)}

                self.processedKB.update(datum)

            else:
                print('\t\t## Failed preprocessing {0} database...'.format(kbpath.split('/')[-2]),
                      file=sys.stderr)

        if constraintKB:
            print('\t>> Constraint database ids based on ...'.format(constraintKB.upper()))
            data = self.processedKB[constraintKB]

            dataInfo = data[0].protein_info
            item_lst = [item for (item, idx) in self.protein_id.items() if item in dataInfo]
            self.protein_id = OrderedDict(zip(item_lst, list(range(len(item_lst)))))

            dataInfo = data[1].gene_info
            item_lst = [item for (item, idx) in self.gene_id.items() if item in dataInfo]
            self.gene_id = OrderedDict(zip(item_lst, list(range(len(item_lst)))))

            item_lst = [items[1][1][1] for items in dataInfo.items() if items[1][1][1]]
            item_lst = np.unique(item_lst)
            self.gene_name_id = OrderedDict(zip(item_lst, list(range(len(item_lst)))))

            item_lst = [items[1][3][1] for items in dataInfo.items() if items[1][3][1] != '']
            item_lst = np.unique(item_lst)
            self.product_id = OrderedDict(zip(item_lst, list(range(len(item_lst)))))

            item_lst = [i for items in dataInfo.items() if items[1][6][1] for i in items[1][6][1]]
            item_lst = np.unique(item_lst)
            self.go_id = OrderedDict(zip(item_lst, list(range(len(item_lst)))))

            dataInfo = data[2].enzyme_info
            item_lst = [item for (item, idx) in self.enzyme_id.items() if item in dataInfo]
            self.enzyme_id = OrderedDict(zip(item_lst, list(range(len(item_lst)))))

            dataInfo = data[3].reaction_info
            item_lst = [item for (item, idx) in self.reaction_id.items() if item in dataInfo]
            self.reaction_id = OrderedDict(zip(item_lst, list(range(len(item_lst)))))

            item_lst = [i for items in dataInfo.items() if items[1][3][1] for i in items[1][3][1]]
            item_lst = np.unique(item_lst)
            self.ec_id = OrderedDict(zip(item_lst, list(range(len(item_lst)))))

            dataInfo = data[4].pathway_info
            item_lst = [item for (item, idx) in self.pathway_id.items() if item in dataInfo]
            self.pathway_id = OrderedDict(zip(item_lst, list(range(len(item_lst)))))

    # -----------------------------------------------------

    # EXTRACT INFORMATION FROM EXPERIMENTAL DATASET -------

    def ExtractInputFromMGFiles(self, colIDx, useEC, folderPath, processes=2):

        lst_ipaths = [os.path.join(folderPath, folder) for folder in os.listdir(folderPath)
                      if not folder.startswith('.')]
        lst_ipaths = [f for f in lst_ipaths if os.path.isdir(f)]

        print('\t>> Extracting input information from {0} files...'.format(len(lst_ipaths)))

        pool = Pool(processes=processes)
        results = [pool.apply_async(self._extractFeaturesFromInput,
                                    args=(lst_ipaths[idx], idx, len(lst_ipaths))) for
                   idx in range(len(lst_ipaths))]
        output = [p.get() for p in results]

        if useEC:
            col_id = self.ec_id
        else:
            col_id = self.gene_name_id

        X = np.zeros((len(output), len(colIDx)), dtype=np.int32)

        for idx, item in enumerate(output):
            for ec in item:
                if ec in col_id:
                    t = np.where(colIDx == col_id[ec])[0]
                    X[idx, t] += 1
        return X

    def _extractFeaturesFromInput(self, inputPath, idx, tInputPaths):
        '''

        :param inputPath:
        :param idx:
        :param tInputPaths:
        :return:
        '''
        if os.path.isdir(inputPath) or os.path.exists(inputPath):
            print(
                '\t\t{1:d})- Progress ({0:.2f}%): extracted input information from {1:d} samples (out of {2:d})...'.format(
                    (idx + 1) * 100.00 / tInputPaths, idx + 1, tInputPaths))

            # Preprocess inputs and outputs
            input_info = self._parseInput(inputPath=inputPath)

            inputPath = list()
            for i, item in input_info.items():
                if item[2]:
                    inputPath.append(item[2])
        else:
            print('\t>> Failed to preprocess {0} file...'.format(inputPath.split('/')[-2]),
                  file=sys.stderr)
        return inputPath

    def _parseInput(self, inputPath):
        """ Process input from a given path
        :type inputPath: str
        :param inputPath: The RunPathoLogic input path, where all the data folders
            are located
        """

        for fname in os.listdir(inputPath):
            if fname.endswith('.pf'):
                input_file = os.path.join(inputPath, fname)
                break
        if os.path.isfile(input_file):
            print('\t\t\t--> Prepossessing input file from: {0}'.format(input_file.split('/')[-1]))
            product_info = OrderedDict()
            with open(input_file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split('\t')
                        if ls:
                            if ls[0] == 'ID':
                                product_id = ' '.join(ls[1:])
                                product_name = ''
                                product_type = ''
                                product_ec = ''
                            elif ls[0] == 'PRODUCT':
                                product_name = ' '.join(ls[1:])
                            elif ls[0] == 'PRODUCT-TYPE':
                                product_type = ' '.join(ls[1:])
                            elif ls[0] == 'EC':
                                product_ec = 'EC-'
                                product_ec = product_ec + ''.join(ls[1:])
                            elif ls[0] == '//':
                                # datum is comprised of {ID: (PRODUCT, PRODUCT-TYPE, EC)}
                                datum = {product_id: (product_name, product_type, product_ec)}
                                product_info.update(datum)
            return product_info

    def ExtractOutputFromMGFiles(self, folderPath, processes=1):

        lst_opaths = [os.path.join(folderPath, folder) for folder in os.listdir(folderPath)
                      if not folder.startswith('.')]

        print('\t>> Extracting output information from {0} files...'.format(len(lst_opaths)))

        pool = Pool(processes=processes)
        results = [pool.apply_async(self._extractFeaturesFromOutput,
                                    args=(lst_opaths[idx], idx, len(lst_opaths))) for
                   idx in range(len(lst_opaths))]
        output = [p.get() for p in results]

        y = np.empty((len(output),), dtype=np.object)

        for idx, item in enumerate(output):
            y[idx] = np.unique(item)
        sample_ids = [os.path.split(opath)[-1] for opath in lst_opaths]
        return y, sample_ids

    def _extractFeaturesFromOutput(self, outputPath, idx, tInputPaths):
        '''

        :param outputPath:
        :param idx:
        :param tInputPaths:
        :return:
        '''
        if os.path.isdir(outputPath) or os.path.exists(outputPath):
            print(
                '\t\t{1:d})- Progress ({0:.2f}%): extracted output information from {1:d} samples (out of {2:d})...'.format(
                    (idx + 1) * 100.00 / tInputPaths, idx + 1, tInputPaths))
            # Preprocess outputs
            outputPath = self._parseOutput(outputPath=outputPath, idxOnly=False)

        else:
            print('\t>> Failed to preprocess {0} file...'.format(outputPath.split('/')[-2]),
                  file=sys.stderr)
        return outputPath

    def _parseOutput(self, outputPath, idxOnly=False):
        """ Process input from a given path
        :param idx:
        :type outputPath: str
        :param outputPath: The RunPathoLogic input path, where all the data folders
            are located
        """
        header = False
        ptw_txt = ''
        ptw_dat = ''
        ptw_col = ''

        for fname in os.listdir(outputPath):
            if fname.endswith('.txt'):
                ptw_txt = os.path.join(outputPath, fname)
            elif fname.endswith('.dat'):
                ptw_dat = os.path.join(outputPath, fname)
            elif fname.endswith('.col'):
                ptw_col = os.path.join(outputPath, fname)

        if os.path.isfile(ptw_txt):
            lst_pathways_idx = list()
            print('\t\t\t--> Prepossessing output file from: {0}'.format(ptw_txt.split('/')[-1]))
            with open(ptw_txt, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split('\t')
                        if ls:
                            if not header:
                                if ls[0] == 'SAMPLE':
                                    header = True
                                    ptw_id = ls.index('PWY_NAME')
                            else:
                                if ls[ptw_id]:
                                    if ls[ptw_id] in self.pathway_id:
                                        if idxOnly:
                                            lst_pathways_idx.append(self.pathway_id[ls[ptw_id]])
                                        else:
                                            lst_pathways_idx.append(ls[ptw_id])

        else:
            lst_pathways_idx = list()
            if os.path.isfile(ptw_dat):
                print('\t\t\t--> Prepossessing output file from: {0}'.format(ptw_dat.split('/')[-1]))
                with open(ptw_dat, errors='ignore') as f:
                    for text in f:
                        if not str(text).startswith('#'):
                            ls = text.strip().split()
                            if ls:
                                if ls[0] == 'UNIQUE-ID':
                                    pathway_id = ' '.join(ls[2:])
                                    if pathway_id in self.pathway_id:
                                        if pathway_id not in lst_pathways_idx:
                                            if idxOnly:
                                                lst_pathways_idx.append(self.pathway_id[pathway_id])
                                            else:
                                                lst_pathways_idx.append(pathway_id)

            if os.path.isfile(ptw_col):
                print('\t\t\t--> Prepossessing output file from: {0}'.format(ptw_col.split('/')[-1]))
                with open(ptw_col, errors='ignore') as f:
                    for text in f:
                        if not str(text).startswith('#'):
                            ls = text.strip().split('\t')
                            if ls:
                                if not header:
                                    if ls[0] == 'UNIQUE-ID':
                                        header = True
                                        for (i, item) in enumerate(ls):
                                            if item == 'UNIQUE-ID':
                                                pathway_idx = i
                                                break
                                else:
                                    pathway_id = ls[pathway_idx]
                                    if pathway_id in self.pathway_id:
                                        if pathway_id not in lst_pathways_idx:
                                            if idxOnly:
                                                lst_pathways_idx.append(self.pathway_id[pathway_id])
                                            else:
                                                lst_pathways_idx.append(pathway_id)
        return lst_pathways_idx

    # ---------------------------------------------------------------------------------------

    # EXTRACT FEATURES FROM DATA ------------------------------------------------------------

    def ExtractPathwayProperties(self, ptwECMatrix, ecIDx, nFeatures, rxnPosition=3, ptwPosition=4, db='metacyc'):
        '''

        :param ptwECMatrix:
        :param ptwPosition:
        :param db:
        :return:
        '''
        print('\t>> Extracting a set of properties of each pathway from: {0}'.format(db.upper()))
        featObj = Feature(proteinID=self.protein_id, productID=self.product_id, geneID=self.gene_id,
                          geneNameID=self.gene_name_id, goID=self.go_id, enzID=self.enzyme_id,
                          rxnID=self.reaction_id, ecID=self.ec_id, ptwID=self.pathway_id)
        ptw_info = self.processedKB[db][ptwPosition].pathway_info
        rxn_info = self.processedKB[db][rxnPosition].reaction_info
        infoList = [ptw_info] + [rxn_info]
        ptwMatrix = featObj.PathwayFeatures(infoList=infoList, ptw_ec_matrix=ptwECMatrix, ec_idx=ecIDx,
                                            nFeatures=nFeatures)
        return ptwMatrix

    def ExtractReactionProperties(self, ptwECMatrix, rxnECMatrix, pECIDx, rECIDx, nFeatures, rxnPosition=3,
                                  ptwPosition=4, db='metacyc'):
        '''

        :param ptwECMatrix:
        :param ptwPosition:
        :param db:
        :return:
        '''
        print('\t>> Extracting a set of properties of each EC from: {0}'.format(db.upper()))
        featObj = Feature(proteinID=self.protein_id, productID=self.product_id, geneID=self.gene_id,
                          geneNameID=self.gene_name_id, goID=self.go_id, enzID=self.enzyme_id,
                          rxnID=self.reaction_id, ecID=self.ec_id, ptwID=self.pathway_id)
        ptw_info = self.processedKB[db][ptwPosition].pathway_info
        rxn_info = self.processedKB[db][rxnPosition].reaction_info
        infoList = [ptw_info] + [rxn_info]
        matrixList = [ptwECMatrix] + [rxnECMatrix]
        ecMatrix = featObj.ECFeatures(infoList=infoList, matrixList=matrixList, p_ec_idx=pECIDx,
                                      r_ec_idx=rECIDx, nFeatures=nFeatures)
        return ecMatrix

    def BuildFeaturesMatrix(self, X, matrixList, colIDx, provided_lst=None, featuresList=[42, 68, 32], rxnPosition=3,
                            ptwPosition=4, displayInterval=50, kb='metacyc', constructRxn=False, XName='X.pkl',
                            savepath='.'):
        print('\t>> Building features from input data: {0}'.format(XName))
        if constructRxn:
            if provided_lst is None:
                idx_lst = self.reaction_id
            else:
                idx_lst = [(id, self.reaction_id[id]) for id in self.reaction_id.items() if id in provided_lst]
        else:
            if provided_lst is None:
                idx_lst = self.pathway_id
            else:
                idx_lst = [(id, self.pathway_id[id]) for id in self.pathway_id.items() if id in provided_lst]

        featObj = Feature(proteinID=self.protein_id, productID=self.product_id, geneID=self.gene_id,
                          geneNameID=self.gene_name_id, goID=self.go_id, enzID=self.enzyme_id,
                          rxnID=self.reaction_id, ecID=self.ec_id, ptwID=idx_lst)

        ## For now on we are concentrating on pathways only
        ptwInfo = self.processedKB[kb][ptwPosition].pathway_info
        rxnInfo = self.processedKB[kb][rxnPosition].reaction_info
        infoList = [ptwInfo] + [rxnInfo]

        fileDesc = '# This file represents the extracted features (X) with abundances...'
        self.SaveData(data=fileDesc, fname=XName, savepath=savepath, tag='the extracted features', mode='w+b')
        self.SaveData(
            data=('nTotalSamples', X.shape[0], 'nTotalComponents', X.shape[1], 'nTotalClassLabels', len(idx_lst),
                  'nEvidenceFeatures', featuresList[1] + 2 * len(idx_lst),
                  'nTotalClassEvidenceFeatures', featuresList[2] * len(idx_lst)),
            fname=XName, savepath=savepath, mode='a+b', printTag=False)

        for idx in np.arange(X.shape[0]):
            mFeatures = featObj.ECEvidenceFeatures(instance=X[idx, :], infoList=infoList,
                                                   matrixList=matrixList, col_idx=colIDx,
                                                   nFeatures=featuresList[1], pFeatures=featuresList[2])
            if idx % displayInterval == 0:
                print('\t\t\t--> Progress ({0:.2f}%): extracted features from {1:d} samples (out of {2:d})...'.format(
                    (idx + 1) * 100.00 / X.shape[0], idx + 1, X.shape[0]))
            if idx + 1 == X.shape[0]:
                print('\t\t\t--> Progress ({0:.2f}%): extracted features from {1:d} samples (out of {2:d})...'.format(
                    (idx + 1) * 100.00 / X.shape[0], idx + 1, X.shape[0]))
            tmp = np.hstack((X[idx, :].reshape(1, X.shape[1]), mFeatures))
            self.SaveData(data=tmp, fname=XName, savepath=savepath, mode='a+b', printTag=False)

    # ---------------------------------------------------------------------------------------

    # MAPPING AND CREATING MATRICES ---------------------------------------------------------

    def MapDataMatrices(self, rowdata, coldata, rowdataInfo, mapRowBasedDataID, mapColID, constrainKB, tag=''):
        '''
        This function is used for mapping any list of enzymes, reactions to reactions
        or pathways (not including superpathways). The function has following
        arguments:
        :param constrainKB:
        :type rowdata: dict
        :param rowdata: dictionary of data that a certain id is required to be mapped where
                        id is a dictionary
        :type coldata: dict
        :param coldata: dictionary of data that a certain id is required for mapping onto
                        rowdata where id is a dictionary
        :type rowdataInfo: dict
        :param rowdataInfo: dictionary of data containing full information
        :type mapColID: int
        :param mapColID: to be mapped using this key
        :type tag: str
        :param tag: to demonstrate this nugget of text onto the printing screen
        '''

        print('\t\t## Mapping {0:s}...'.format(tag))
        data_dict = OrderedDict()
        if constrainKB:
            lst_kb = [constrainKB]
        else:
            lst_kb = self.lst_kbpaths

        for kb in lst_kb:
            data = self.processedKB[kb][mapRowBasedDataID]
            data_info = eval('data' + rowdataInfo)

            for (ritem, ridx) in rowdata.items():
                if ridx not in data_dict:
                    if ritem in data_info:
                        ls = data_info[ritem][mapColID][1]
                        ls_idx = [coldata[citem] for citem in ls if citem in coldata]
                        data_dict.update({ridx: ls_idx})
        return data_dict

    # ---------------------------------------------------------------------------------------

    # CONSTRUCT INDICATOR MATRICES ----------------------------------------------------------

    def CreateIndicatorMatrix(self, rowdata, ncoldata, tag, removeZeroEntries=False):
        '''
        This function creates a binary indicator (or adjacency) matrix given a list of
        row-wise data and a list of column-wise data. The matrix is saved (or pickled)
        in a binary format. The function has following arguments:
        :param removeZeroEntries:
        :type rowdata: dict
        :param rowdata: a dictionary of data from which to construct the matrix
        :type ncoldata: int
        :param ncoldata: integer number indicating the length of the column data for
                         the matrix
        :type tag: str
        :param tag: to demonstrate this nugget of text onto the printing screen
        '''

        print('\t\t## Constructing the following sparse binary matrix: {0}'.format(tag))
        nrowdata = len(rowdata)
        matrix = np.zeros((nrowdata, ncoldata), dtype=np.int32)

        # Fill the sparse matrices
        for ridx, ritem in rowdata.items():
            for idx in ritem:
                matrix[ridx, idx] += 1

        col_idx = list(range(ncoldata))

        if removeZeroEntries:
            total = np.sum(matrix, axis=0)
            col_idx = np.nonzero(total)[0]
            zeroIdx = np.where(total == 0)[0]
            matrix = np.delete(matrix, zeroIdx, axis=1)
        return matrix, col_idx

    # ---------------------------------------------------------------------------------------

    # CONSTRUCT SYNTHETIC, MINPATH AND PATHOLOGIC CORPORA -----------------------------------

    def CorrputItemByRemovingComponents(self, X_lst, idItem, dictID, infoList, colID, nComponentsToCorrupt, exception,
                                        ptwConstraint, constructRxn):
        # Corrupting pathways that exceed some threshold by removing true ec or genes
        nComponents = len(X_lst)
        ptw_info = infoList[0]
        rxn_info = infoList[1]

        if nComponents > exception:
            if ptwConstraint and not constructRxn:
                regex = re.compile(r'\(| |\)')
                if idItem not in ptw_info:
                    pass
                text = str(ptw_info[idItem][0][1]) + ' ' + ' '.join(
                    ptw_info[idItem][1][1]) + ' ' + ' '.join(
                    ptw_info[idItem][6][1])
                text = text.lower()
                lst_rxn = [list(filter(None, regex.split(itm))) for itm in ptw_info[idItem][3][1]]
                for idx, itm in enumerate(lst_rxn):
                    itm = ' '.join(itm).replace('\"', '')
                    lst_rxn[idx] = itm.split()
                dg = nx.DiGraph()
                dg.add_nodes_from(ptw_info[idItem][4][1])
                for itm in lst_rxn:
                    if len(itm) == 1:
                        continue
                    else:
                        dg.add_edge(itm[1], itm[0])
                if 'detoxification' in text or 'degradation' in text:
                    lst_rxn = dg.in_degree()
                    lst_rxn = [n for n, d in sorted(lst_rxn, key=itemgetter(1))]
                    lst_ec = [j for itm in lst_rxn for j in rxn_info[itm][3][1] if j in ptw_info[idItem][15][1]]
                    # retain the first reaction
                    if nComponentsToCorrupt < len(lst_ec[1:]):
                        lst_rd = np.random.choice(a=lst_ec[1:], size=nComponentsToCorrupt, replace=False)
                        for r in lst_rd:
                            X_lst.remove(np.where(colID == dictID[r])[0])
                elif 'biosynthesis' in text:
                    lst_rxn = dg.out_degree()
                    lst_rxn = [n for n, d in sorted(lst_rxn, key=itemgetter(1))]
                    lst_ec = [j for itm in lst_rxn for j in rxn_info[itm][3][1] if j in ptw_info[idItem][15][1]]

                    ## retain the last two reactions
                    if nComponentsToCorrupt < len(lst_ec[2:]):
                        lst_rd = np.random.choice(a=lst_ec[2:], size=nComponentsToCorrupt, replace=False)
                        for r in lst_rd:
                            X_lst.remove(np.where(colID == dictID[r])[0])
                else:
                    if nComponentsToCorrupt < len(X_lst):
                        lst_rd = np.random.choice(a=X_lst, size=nComponentsToCorrupt, replace=False)
                        for r in lst_rd:
                            X_lst.remove(r)
            else:
                if nComponentsToCorrupt < len(X_lst):
                    lst_rd = np.random.choice(a=X_lst, size=nComponentsToCorrupt, replace=False)
                    for r in lst_rd:
                        X_lst.remove(r)
        return X_lst

    def BuildSyntheticDataset(self, rowDataMatrix, colIDx, nSamples=1000, averageNitemsPerSample=500,
                              nComponentsToCorrupt=2, exception=5, nComponentsToCorruptOutside=3, addNoise=True,
                              displayInterval=100, constraint_kb='metacyc', useEC=True, constructRxn=False,
                              ptwConstraint=True, provided_lst=None, fName=None, savepath=None):

        '''
        This function creates a binary indicator (or adjacency) matrix given a list of
        row-wise data and a list of column-wise data. The function has following
        arguments:
        :type rowDataMatrix: scipy.sparse.lil_matrix
        :param rowDataMatrix: a sparse binary indicator matrix where each row define
            a reaction (or pathway) with columns representing its associated genes
            (or enzymes)
        :type nComponentsToCorrupt: list
        :param nComponentsToCorrupt: a list of integer values indicating the number
            of genes (or enzymes) to be corrupted by removing genes (or enzymes) for each
            reaction (or pathway)
        :type nSamples_per_item: int
        :param nSamples_per_item: an integer value indicating the number of corrupted
            reaction (or pathway) per each true reaction (or pathway)
        :type exception: int
        :param exception: an integer value constraining the corruption process to only
            those reactions (or pathways) that have more then this number of genes
            (or enzymes)
        :type nComponentsToCorruptOutside: list
        :param nComponentsToCorruptOutside: a list of integer values indicating the
            number of genes (or enzymes) to be corrupted by inserting false genes
            (or enzymes) to each pathway (or reaction)
        :type nSamples_per_item_outside: int
        :param nSamples_per_item_outside: an integer value indicating the number of
            corrupted reaction (or pathway) per each true reaction (or pathway)
        :type constraint_kb: str
        :param constraint_kb: a knowledge base where the dataset is constrained upon it
        :type norm_op: str
        :param norm_op: an integer value indicating the number of corrupted
            pathway per each true pathway
        '''

        fName = fName + '_' + str(nSamples) + '_X.pkl'

        if addNoise:
            print('\t>> The following settings are used for constructing the corpora:'
                  '\n\t\t1. A list of true genes (or ec) to be corrupted per reaction '
                  '\n\t\t   (or pathway) by removing genes except those do not exceed {0} genes '
                  '\n\t\t   (or ec): {1}'
                  '\n\t\t2. Number of samples to be generated: {2}'
                  '\n\t\t3. A list of false genes (or ec) to be inserted per reaction (or pathway): {3}'
                  '\n\t\t4. Constraints using the knowledge-base: {4}'
                  '\n\t\t5. Saving the constructed dataset in (as "{5}"): {6}\n'
                  .format(exception, nComponentsToCorrupt, nSamples, nComponentsToCorruptOutside,
                          constraint_kb.upper(), fName, savepath))
        else:
            print('\t>> The following settings are used for constructing the corpora:'
                  '\n\t\t1. Add Noise: False'
                  '\n\t\t2. Number of samples to be generated: {0}'
                  '\n\t\t3. Constraints using the knowledge-base: {1}'
                  '\n\t\t4. Saving the constructed dataset in (as "{2}"): {3}\n'
                  .format(nSamples, constraint_kb.upper(), fName, savepath))
        if useEC:
            dict_id = self.ec_id
        else:
            dict_id = self.gene_name_id

        use_idx = range(len(colIDx))

        if constructRxn:
            item_idx_lst = self._reverse_idx(self.reaction_id)
            if provided_lst is None:
                idx_lst = [idx for (item, idx) in self.reaction_id.items()]
            else:
                idx_lst = [idx for (item, idx) in self.reaction_id.items() if item in provided_lst]
        else:
            item_idx_lst = self._reverse_idx(self.pathway_id)
            if provided_lst is None:
                idx_lst = [idx for (item, idx) in self.pathway_id.items()]
            else:
                idx_lst = [idx for (item, idx) in self.pathway_id.items() if item in provided_lst]

        ptwInfo = self.processedKB[constraint_kb][4].pathway_info
        rxnInfo = self.processedKB[constraint_kb][3].reaction_info
        infoList = [ptwInfo] + [rxnInfo]

        print('\t>> Constructing synthetic dataset...')
        fileDesc = '# Synthetic dataset is stored in this format: sample index, list of data components, list of labels'
        self.SaveData(data=fileDesc, fname=fName, savepath=savepath, tag='synthetic dataset', mode='w+b')

        for sidx in range(nSamples):
            nPathways = np.random.poisson(lam=averageNitemsPerSample)
            item_lst = np.random.choice(a=idx_lst, size=nPathways, replace=True)
            X = list()
            y = list()
            for idx in item_lst:
                lst_item_x = list()
                idItem = item_idx_lst[idx]
                y.append(idItem)
                tmp = np.nonzero(rowDataMatrix[idx, :])[0]
                replicate = rowDataMatrix[idx, tmp]
                for i, r in enumerate(replicate):
                    t = [tmp[i]] * r
                    lst_item_x.extend(t)

                # Choosing whether to corrupt by removing or inserting
                corruptType = np.random.choice(a=[0, 1, 2], size=1, replace=False)

                if corruptType == 1 and addNoise:
                    lst_item_x = self.CorrputItemByRemovingComponents(X_lst=lst_item_x, idItem=idItem, dictID=dict_id,
                                                                      infoList=infoList, colID=colIDx,
                                                                      nComponentsToCorrupt=nComponentsToCorrupt,
                                                                      exception=exception, ptwConstraint=ptwConstraint,
                                                                      constructRxn=constructRxn)
                elif corruptType == 2 and addNoise:
                    # Corrupting pathways by adding false ec or genes
                    lst_rd = np.random.choice(a=use_idx, size=nComponentsToCorruptOutside, replace=True)
                    for r in lst_rd:
                        lst_item_x.append(r)
                else:
                    pass
                X.extend(lst_item_x)
            if sidx % displayInterval == 0:
                print('\t\t\t--> Progress: {0:.2f}%, created {1:d} samples (out of {2:d})...'.format(
                    (sidx + 1) * 100.00 / nSamples,
                    sidx + 1, nSamples))
            if sidx + 1 == nSamples:
                print('\t\t\t--> Progress: {0:.2f}%, created {1:d} samples (out of {2:d})...'.format(
                    (sidx + 1) * 100.00 / nSamples,
                    sidx + 1, nSamples))
            self.SaveData(data=(sidx, X, y), fname=fName, savepath=savepath, mode='a+b', printTag=False)

    def BuildGoldenDataset(self, rowDataMatrix, KB_lst, displayInterval=10, constructRxn=False, constraint_kb='metacyc',
                           fName=None, savepath=None):
        KB_lst = [kb for kb in KB_lst if kb != constraint_kb]
        ds_lst = [list(combinations(KB_lst, r + 1)) for r in range(len(KB_lst))]
        ds_lst = [ds_tuple for item_lst in ds_lst for ds_tuple in item_lst]
        nSamples = len(ds_lst)
        itemInfoByKB = list()

        if constructRxn:
            item_id_lst = self.reaction_id
            for ds in KB_lst:
                itemInfoByKB.append([rxn for rxn in self.processedKB[ds][3].reaction_info if rxn in item_id_lst])
        else:
            item_id_lst = self.pathway_id
            for ds in KB_lst:
                itemInfoByKB.append([ptw for ptw in self.processedKB[ds][4].pathway_info if ptw in item_id_lst])

        print('\t>> Constructing golden dataset...')
        fName = fName + '_' + str(nSamples) + '_X.pkl'
        fileDesc = '# Golden dataset is stored in this format: sample index, list of data components, list of labels'
        self.SaveData(data=fileDesc, fname=fName, savepath=savepath, tag='golden dataset', mode='w+b')

        for sidx, item_lst in enumerate(ds_lst):
            X = list()
            y = list()
            item_lst = list(item_lst)
            for ds in item_lst:
                ptw_lst = itemInfoByKB[KB_lst.index(ds)]
                y.extend(ptw_lst)
                lst_item_x = list()
                for ptw in ptw_lst:
                    tmp = np.nonzero(rowDataMatrix[item_id_lst[ptw], :])[0]
                    replicate = rowDataMatrix[item_id_lst[ptw], tmp]
                    for i, r in enumerate(replicate):
                        t = [tmp[i]] * r
                        lst_item_x.extend(t)
                X.extend(lst_item_x)
            if sidx % displayInterval == 0:
                print('\t\t\t--> Progress: {0:.2f}%, created {1:d} samples (out of {2:d})...'.format(
                    (sidx + 1) * 100.00 / nSamples,
                    sidx + 1, nSamples))
            if sidx + 1 == nSamples:
                print('\t\t\t--> Progress: {0:.2f}%, created {1:d} samples (out of {2:d})...'.format(
                    (sidx + 1) * 100.00 / nSamples,
                    sidx + 1, nSamples))
            self.SaveData(data=(item_lst, X, y), fname=fName, savepath=savepath, mode='a+b', printTag=False)

    def FormatCuratedDataset(self, nSamples, rowDataMatrix, colIDx, useEC=True, constructRxn=False,
                             pathologicInput=False, minpathDataset=True, minpathMapFile=True, mapAll=False,
                             fName='synset.pkl', loadpath='.'):
        file = fName + '_' + str(nSamples) + '_X.pkl'
        X = np.zeros((nSamples, len(colIDx)), dtype=np.int32)
        y = np.empty((nSamples,), dtype=np.object)

        count = 0
        sample_ids = list()
        print('\t>> Constructing sparse matrix for: {0} samples...'.format(nSamples))
        print('\t\t## Loading curated dataset: {0}'.format(file))
        file = os.path.join(loadpath, file)

        with open(file, 'rb') as f_in:
            while count < nSamples:
                try:
                    tmp = pkl.load(f_in)
                    if len(tmp) == 3:
                        count += 1
                        lst_id, lst_x, lst_y = tmp
                        sidx = len(sample_ids)
                        sample_ids.append(lst_id)
                        for idx in lst_x:
                            X[sidx, idx] = X[sidx, idx] + 1
                        y[sidx] = np.unique(lst_y)
                except IOError:
                    break

        if pathologicInput:
            self.BuildPathoLogicInput(X=X, Dataset_lst_IDs=sample_ids, colIDx=colIDx, nSamples=nSamples,
                                      savepath=os.path.join(loadpath, 'ptools'))

        if minpathDataset:
            self.BuildMinPathDataset(X=X, colIDx=colIDx, useEC=useEC, nSamples=nSamples, fName=fName, savepath=loadpath)

        if minpathMapFile:
            self.MapLabelswithFunctions(rowDataMatrix=rowDataMatrix, colIDx=colIDx, y=y, nSamples=nSamples,
                                        mapAll=mapAll, useEC=useEC, constructRxn=constructRxn, fName=fName,
                                        savepath=loadpath)

        file = fName + '_' + str(nSamples) + '_Xm.pkl'
        fileDesc = '# The dataset representing a list of data components (X)...'
        self.SaveData(data=fileDesc, fname=file, savepath=loadpath, tag='the curated dataset (X)', mode='w+b')
        self.SaveData(data=X, fname=file, savepath=loadpath, mode='a+b', printTag=False)

        file = fName + '_' + str(nSamples) + '_y.pkl'
        fileDesc = '# The dataset representing a list of data components (y) with ids...'
        self.SaveData(data=fileDesc, fname=file, savepath=loadpath, tag='the curated dataset (y)', mode='w+b')
        self.SaveData(data=(y, sample_ids), fname=file, savepath=loadpath, mode='a+b', printTag=False)

        return X

    def BuildPathoLogicInput(self, X, Dataset_lst_IDs, colIDx, nSamples, savepath):
        print('\t>> Building the PathoLogic input file for: {0} samples'.format(nSamples))
        colID = self._reverse_idx(self.ec_id)
        for idx in range(X.shape[0]):
            fname = ''
            dsname = ''
            for ds in Dataset_lst_IDs[idx]:
                if not os.path.isdir(savepath):
                    os.mkdir(savepath)
                fname = 'golden_' + str(len(Dataset_lst_IDs[idx])) + '_' + str(idx)
                if len(Dataset_lst_IDs[idx]) > 1:
                    dsname += str(ds) + ' , '
                else:
                    dsname = ds

            file = '0.pf'
            spath = os.path.join(savepath, fname)

            id = 'ID\t' + fname + '\n'
            name = 'NAME\t' + str(fname) + '\n'
            type = 'TYPE\t:READ/CONTIG\n'
            annot = 'ANNOT-FILE\t' + file + '\n'
            comment = ';; DATASET\t' + dsname + '\n'
            datum = id + name + type + annot + comment
            self.SaveData(data=datum, fname='genetic-elements.dat', savepath=spath, tag='genetic elements info',
                          mode='w', wString=True)

            id = 'ID\t' + fname + '\n'
            storage = 'STORAGE\tFILE\n'
            name = 'NAME\t' + str(fname) + '\n'
            abbr_name = 'ABBREV-NAME\t' + str(fname) + '\n'
            strain = 'STRAIN\t1\n'
            rank = 'RANK\t|species|\n'
            ncbi_taxon = 'NCBI-TAXON-ID\t12908\n'

            datum = id + storage + name + abbr_name + strain + rank + ncbi_taxon
            self.SaveData(data=datum, fname='organism-params.dat', savepath=spath, tag='organism params info',
                          mode='w', wString=True)

            self.SaveData(data='', fname=file, savepath=spath, tag='data description for ' + dsname, mode='w',
                          wString=True)
            tmp = np.nonzero(X[idx, :])[0]
            replicate = X[idx, tmp]
            stbase = 0
            edbase = 1
            func = 'ORF'
            total = 0
            for i, r in enumerate(replicate):
                for rep in range(r):
                    id = 'ID\t' + str(idx) + '_' + str(total) + '\n'
                    name = 'NAME\t' + str(idx) + '_' + str(total) + '\n'
                    startbase = 'STARTBASE\t' + str(stbase) + '\n'
                    endbase = 'ENDBASE\t' + str(edbase) + '\n'
                    function = 'PRODUCT\t' + func + '\n'
                    product_type = 'PRODUCT-TYPE\tP\n'
                    tmp_ec = re.split("[.\-]+", colID[colIDx[tmp[i]]])[1:]
                    ec = list()
                    notFound = False
                    for e in tmp_ec:
                        if e.isdigit():
                            ec.append(e)
                        else:
                            notFound = True
                            ec.append('0')
                    len_ec = len(ec)
                    if len_ec < 4:
                        notFound = True
                        for e in range((4 - len_ec)):
                            ec.append('0')
                    if notFound:
                        break
                    ec = 'EC\t' + '.'.join(ec) + '\n'
                    datum = id + name + startbase + endbase + function + product_type + ec + '//\n'
                    total += 1
                    self.SaveData(data=datum, fname=file, savepath=spath, tag='data description', mode='a',
                                  wString=True, printTag=False)

    def BuildMinPathDataset(self, X, colIDx, useEC, nSamples, fName, savepath):
        print('\t>> Building the MinPath data file in (sample_(idx), EC or gene) format for: {0} samples'.format(
            nSamples))
        file = fName + '_' + str(nSamples) + '_minpath_data.txt'

        if useEC:
            colID = self.ec_id
        else:
            colID = self.gene_name_id

        colID = self._reverse_idx(colID)
        self.SaveData(data='', fname=file, savepath=savepath, tag='data description', mode='w', wString=True)
        for idx in range(X.shape[0]):
            tmp = np.nonzero(X[idx, :])[0]
            replicate = X[idx, tmp]
            for i, r in enumerate(replicate):
                sampleName = str(idx) + '\t' + colID[colIDx[tmp[i]]] + '\n'
                lst_samples = sampleName * r
                self.SaveData(data=lst_samples, fname=file, savepath=savepath, tag='data description', mode='a',
                              wString=True, printTag=False)

    def MapLabelswithFunctions(self, rowDataMatrix, colIDx, y, nSamples, mapAll=True, useEC=True, constructRxn=False,
                               fName='ptw2ec.txt', savepath='.'):
        '''

        :param rowDataMatrix:
        :param y:
        :param nSamples:
        :param mapAll:
        :param useEC:
        :param constructRxn:
        :param fName:
        :param savepath:
        :return:
        '''
        if useEC:
            col = 'ec'
            r_id = self._reverse_idx(self.ec_id)
        else:
            col = 'gene'
            r_id = self._reverse_idx(self.gene_name_id)

        col_id = {}
        for idx in colIDx:
            col_id.update({idx: r_id[idx]})

        if constructRxn:
            row_id = self.reaction_id
            row = 'reaction'
        else:
            row_id = self.pathway_id
            row = 'pathway'

        if mapAll:
            print('\t>> Referencing labels with functions...')
            file = col + '2' + row + '.txt'
            lst_y = [id for id, idx in row_id.items()]
        else:
            lst_y = [i for j in y for i in j]
            lst_y = np.unique(lst_y)
            print('\t>> Referencing labels with functions for {0} labels'.format(len(lst_y)))
            file = fName + '_' + str(nSamples) + '_minpath_mapping.txt'

        fileDesc = '# File Description: Metacyc ' + str(row) + ' and ' + str(col) + 'mapping file for ' + str(
            len(lst_y)) + ' functions\n#Pathway\tEC\n'
        self.SaveData(data=fileDesc, fname=file, savepath=savepath, tag='mapping information', mode='w', wString=True)

        for rowName in lst_y:
            idx = np.nonzero(rowDataMatrix[row_id[rowName], :])[0]
            for i in idx.tolist():
                replicate = rowDataMatrix[row_id[rowName], i]
                mapping = rowName + '\t' + str(col_id[colIDx[i]]) + '\n'
                lst_rows = mapping * replicate
                if lst_rows:
                    self.SaveData(data=lst_rows, fname=file, savepath=savepath, tag='mapping file', mode='a',
                                  wString=True, printTag=False)

    # ---------------------------------------------------------------------------------------

    # CONSTRUCT SIMILARITY MATRIX BETWEEN PATHWAYS ------------------------------------------
    def _smithWaterman(self, query, target, alignment_score: float = 1, gap_cost: float = 1) -> float:
        '''
        Reproduced from https://gist.github.com/nornagon/6326a643fc30339ece3021013ed9b48c
        :param query:
        :param target:
        :param alignment_score:
        :param gap_cost:
        :return:
        '''
        S = np.zeros((len(query) + 1, len(target) + 1))
        for i in range(1, len(query) + 1):
            for j in range(1, len(target) + 1):
                match = S[i - 1, j - 1] + (alignment_score if query[i - 1] == target[j - 1] else 0)
                delete = S[1:i, j].max() - gap_cost if i > 1 else 0
                insert = S[i, 1:j].max() - gap_cost if j > 1 else 0
                S[i, j] = max(match, delete, insert, 0)
        return S.max()

    def BuildSimilarityMatrix(self, ptwECMatrix, ptwPosition=4, kb='metacyc', fName='pathway_similarity', savepath='.'):

        print('\t>> Building pathway similarities based on cosine similarity from {0}...'.format(kb))
        M = cosine_similarity(X=ptwECMatrix)
        M = (M * 100).astype(int)
        file = fName + '_cos.pkl'
        fileDesc = '#File Description: number of pathways x number of pathways\n'
        self.SaveData(data=fileDesc, fname=file, savepath=savepath,
                      tag='the pathway similarity matrix based on on cosine similarity', mode='w+b')
        self.SaveData(data=('nPathways:', str(M.shape[0])), fname=file, savepath=savepath, mode='a+b', printTag=False)
        self.SaveData(data=M, fname=file, savepath=savepath, mode='a+b', printTag=False)

        print('\t>> Building pathway similarities based on chi-squared kernel from {0}...'.format(kb))
        M = chi2_kernel(X=ptwECMatrix)
        M = (M * 100).astype(int)
        file = fName + '_chi2.pkl'
        fileDesc = '#File Description: number of pathways x number of pathways\n'
        self.SaveData(data=fileDesc, fname=file, savepath=savepath,
                      tag='the pathway similarity matrix based on chi-squared kernel', mode='w+b')
        self.SaveData(data=('nPathways:', str(M.shape[0])), fname=file, savepath=savepath, mode='a+b', printTag=False)
        self.SaveData(data=M, fname=file, savepath=savepath, mode='a+b', printTag=False)

        print('\t>> Building pathway similarities based on gaussian kernel from {0}...'.format(kb))
        rbf = RBF(length_scale=1.)
        M = rbf(X=ptwECMatrix)
        M = (M * 100).astype(int)
        file = fName + '_rbf.pkl'
        fileDesc = '#File Description: number of pathways x number of pathways\n'
        self.SaveData(data=fileDesc, fname=file, savepath=savepath,
                      tag='the pathway similarity matrix based on gaussian kernel', mode='w+b')
        self.SaveData(data=('nPathways:', str(M.shape[0])), fname=file, savepath=savepath, mode='a+b', printTag=False)
        self.SaveData(data=M, fname=file, savepath=savepath, mode='a+b', printTag=False)

        print('\t>> Building pathway similarities based on smith-waterman algorithm from {0}...'.format(kb))
        ptwInfo = self.processedKB[kb][ptwPosition].pathway_info
        regex = re.compile(r'\(| |\)')
        ptwIDX = self._reverse_idx(self.pathway_id)
        M = np.zeros(shape=(len(ptwInfo), len(ptwInfo)))
        for i in range(M.shape[0]):
            query_pID = ptwIDX[i]
            query_ptw = ptwInfo[query_pID]
            query_text = str(query_ptw[0][1]) + ' ' + ' '.join(query_ptw[1][1]) + ' ' + ' '.join(query_ptw[6][1])
            query_text = query_text.lower()
            query_end2end_rxn_series = [list(filter(None, regex.split(itm))) for itm in ptwInfo[query_pID][3][1]]
            for idx, itm in enumerate(query_end2end_rxn_series):
                itm = ' '.join(itm).replace('\"', '')
                query_end2end_rxn_series[idx] = itm.split()
            dg = nx.DiGraph()
            dg.add_nodes_from(ptwInfo[query_pID][4][1])
            for itm in query_end2end_rxn_series:
                if len(itm) == 1:
                    continue
                else:
                    dg.add_edge(itm[1], itm[0])

            if list(nx.simple_cycles(dg)):
                query_end2end_rxn_series = [n for n, d in sorted(dg.in_degree(), key=itemgetter(1))]
            else:
                query_end2end_rxn_series = list(nx.topological_sort(dg))

            query_end2end_rxn_series = [self.reaction_id[rxn] for rxn in query_end2end_rxn_series]

            for j in np.arange(M.shape[1]):
                target_pID = ptwIDX[j]
                target_ptw = ptwInfo[target_pID]
                target_text = str(target_ptw[0][1]) + ' ' + ' '.join(target_ptw[1][1]) + ' ' + ' '.join(
                    target_ptw[6][1])
                target_text = target_text.lower()
                target_end2end_rxn_series = [list(filter(None, regex.split(itm))) for itm in ptwInfo[target_pID][3][1]]
                for idx, itm in enumerate(target_end2end_rxn_series):
                    itm = ' '.join(itm).replace('\"', '')
                    target_end2end_rxn_series[idx] = itm.split()
                dg = nx.DiGraph()
                dg.add_nodes_from(ptwInfo[target_pID][4][1])
                for itm in target_end2end_rxn_series:
                    if len(itm) == 1:
                        continue
                    else:
                        dg.add_edge(itm[1], itm[0])

                if fuzz.token_sort_ratio(query_text, target_text) > 75:
                    alignment_score = 2
                else:
                    alignment_score = 1

                if list(nx.simple_cycles(dg)):
                    target_end2end_rxn_series = [n for n, d in sorted(dg.in_degree(), key=itemgetter(1))]
                else:
                    target_end2end_rxn_series = list(nx.topological_sort(dg))

                target_end2end_rxn_series = [self.reaction_id[rxn] for rxn in target_end2end_rxn_series]
                M[i, j] = self._smithWaterman(query=query_end2end_rxn_series, target=target_end2end_rxn_series,
                                              alignment_score=alignment_score)
        file = fName + '_sw.pkl'
        fileDesc = '#File Description: number of pathways x number of pathways\n'
        self.SaveData(data=fileDesc, fname=file, savepath=savepath,
                      tag='the pathway similarity matrix based on smith waterman algorithm', mode='w+b')
        self.SaveData(data=('nPathways:', str(M.shape[0])), fname=file, savepath=savepath, mode='a+b', printTag=False)
        self.SaveData(data=M, fname=file, savepath=savepath, mode='a+b', printTag=False)

    # ---------------------------------------------------------------------------------------

    # CONVERT VALUES TO INDICES -------------------------------------------------------------

    def _reverse_idx(self, value2idx):
        idx2value = {}
        for key, value in value2idx.items():
            idx2value.update({value: key})
        return idx2value

    # ---------------------------------------------------------------------------------------

    # SAVING AND LOADING DATA ---------------------------------------------------------------

    def SaveData(self, data, fname, savepath, tag='', mode='wb', wString=False,
                 printTag=True):
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

    def LoadData(self, fname, loadpath, tag='data'):
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

    # ---------------------------------------------------------------------------------------
