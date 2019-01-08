'''
This file extracts features from metagenomics inputset. It also
requires various data objects obtained from BioCyc PGDBs to be
available as inputs to designated machine learning models.
'''

import re
from operator import itemgetter

import networkx as nx
import numpy as np
from fuzzywuzzy import process

try:
    import cPickle as pkl
except:
    import pickle as pkl

ROMAN_CONSTANTS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XX", "XXX",
                   "XL", "L", "LX", "LXX", "LXXX", "XC", "C", "CC", "CCC", "CD", "D", "DC",
                   "DCC", "DCCC", "CM", "M", "MM", "MMM"]


class Feature(object):
    def __init__(self, proteinID, productID, geneID, geneNameID, goID, enzID, rxnID, ecID, ptwID):
        self.proteinID = proteinID
        self.productID = productID
        self.geneID = geneID
        self.geneNameID = geneNameID
        self.goID = goID
        self.enzID = enzID
        self.rxnID = rxnID
        self.ecID = ecID
        self.ptwID = ptwID

    def ReverseIdx(self, value2idx):
        idx2value = {}
        for key, value in value2idx.items():
            idx2value.update({value: key})
        return idx2value

    ##########################################################################################################
    ############################           FEATURES FROM KNOWLEDGE-BASE            ###########################
    ##########################################################################################################

    def PathwayFeatures(self, infoList, ptw_ec_matrix, ec_idx, nFeatures=27, textMatch=95):
        ## Add the EC to each pathway and define kernel distance metric

        regex = re.compile(r'\(| |\)')
        mFeatures = np.zeros(shape=(len(self.ptwID), nFeatures), dtype=np.float32)

        for idx, item in infoList[0].items():
            # 0. has-orphan-reaction (boolean)
            if item[13][1] > 0:
                mFeatures[self.ptwID[idx], 0] = 1

            # 1. has-spontaneous-reaction (boolean)
            if item[14][1] > 0:
                mFeatures[self.ptwID[idx], 1] = 1

            # 2. has-single-reaction (boolean)
            # 3. num-reactions (numeric)
            # 4. multiple-reaction-pathway (boolean)
            if len(item[4][1]) == 1:
                mFeatures[self.ptwID[idx], 2] = 1
                mFeatures[self.ptwID[idx], 3] = 1
            else:
                mFeatures[self.ptwID[idx], 3] = len(item[4])
                mFeatures[self.ptwID[idx], 4] = 1

            # 5. is-subpathway (boolean)
            if len(item[9][1]) != 0:
                mFeatures[self.ptwID[idx], 5] = 1

            text = str(item[0][1]) + ' ' + ' '.join(item[1][1]) + ' ' + ' '.join(item[6][1])
            text = text.lower()

            # 6. is-energy-pathway (boolean)
            if 'energy' in text:
                mFeatures[self.ptwID[idx], 6] = 1

            # 7. is-deg-or-detox-pathway (boolean)
            # 8. is-detoxification-pathway (boolean)
            # 9. is-degradation-pathway (boolean)
            if 'detoxification' in text or 'degradation' in text:
                mFeatures[self.ptwID[idx], 7] = 1
            else:
                if 'detoxification' in text:
                    mFeatures[self.ptwID[idx], 8] = 1
                if 'degradation' in text:
                    mFeatures[self.ptwID[idx], 9] = 1

            # 10. is-biosynthesis-pathway (boolean)
            if 'biosynthesis' in text:
                mFeatures[self.ptwID[idx], 10] = 1

            # 11. is-variant (boolean)
            for t in text.split():
                if t.upper() in ROMAN_CONSTANTS:
                    mFeatures[self.ptwID[idx], 11] = 1

            # 12. num-initial-reactions (numeric)
            # 13. num-final-reactions (numeric)
            # 14. first-reaction-is-enzymatic (boolean)
            # 15. last-reaction-is-enzymatic (boolean)
            if item[3][1]:
                lst_rxn = [list(filter(None, regex.split(itm))) for itm in item[3][1]]
                dg = nx.DiGraph()
                dg.add_nodes_from(item[4][1])
                for itm in lst_rxn:
                    if len(itm) == 1:
                        continue
                    else:
                        dg.add_edge(itm[1], itm[0])

                count = 0
                for itm in dg.pred.items():
                    if len(itm[1]) == 0:
                        count += 1
                        itm = ''.join(itm[0]).replace('\"', '')
                        if infoList[1][itm][3][1]:
                            mFeatures[self.ptwID[idx], 14] = 1
                mFeatures[self.ptwID[idx], 12] = count

                count = 0
                for itm in dg.succ.items():
                    if len(itm[1]) == 0:
                        count += 1
                        itm = ''.join(itm[0]).replace('\"', '')
                        if infoList[1][itm][3][1]:
                            mFeatures[self.ptwID[idx], 15] = 1
                mFeatures[self.ptwID[idx], 13] = count

            # 16. has-unique-reactions (boolean)
            # 17. num-unique-reactions (numeric)
            if item[16][1]:
                mFeatures[self.ptwID[idx], 16] = 1
                mFeatures[self.ptwID[idx], 17] = len(item[16][1])

            # 18. num-enzymatic-reactions (numeric)
            if item[15][1]:
                mFeatures[self.ptwID[idx], 18] = len(item[15][1])

            # 19. num-unique-enzymatic-reactions (numeric)
            unique_ec_lst = list()
            for rxn in item[16][1]:
                rxn = ''.join(rxn).replace('\"', '')
                if infoList[1][rxn][3][1]:
                    e = [e for e in infoList[1][rxn][3][1]]
                    unique_ec_lst.extend(e)
            if unique_ec_lst:
                mFeatures[self.ptwID[idx], 19] = len(unique_ec_lst)

            # 20. subset-has-same-evidence (boolean)
            # 21. other-pathway-has-more-evidence (boolean)
            # 22. variant-has-more-evidence (boolean)
            lst_ptw_var = list()
            for pidx, pitem in infoList[0].items():
                if pidx != idx:
                    if set.intersection(set(item[4][1]), set(pitem[4][1])):
                        mFeatures[self.ptwID[idx], 20] = 1
                    if set(item[4][1]) <= set(pitem[4][1]):
                        mFeatures[self.ptwID[idx], 21] = 1
                    for t in text.split():
                        if t.upper() in ROMAN_CONSTANTS:
                            match = process.extract(pitem[0][1], [item[0][1]])
                            if match[0][1] > textMatch:
                                lst_ptw_var.append(pidx)
                            break
            if lst_ptw_var:
                for id in lst_ptw_var:
                    if set(item[4][1]) <= set(infoList[0][id][4][1]):
                        mFeatures[self.ptwID[idx], 22] = 1

            # 23. species-range-includes-target (boolean)
            if item[7][1]:
                mFeatures[self.ptwID[idx], 23] = 1

            # 24. taxonomic-range-includes-target (boolean)
            if item[8][1]:
                mFeatures[self.ptwID[idx], 24] = 1

            # 25. evidence-info-content-norm-all (numeric)
            m = np.sum(ptw_ec_matrix[self.ptwID[idx], :])
            total = 0
            for ec in set(item[15][1]):
                total += 1 / np.count_nonzero(ptw_ec_matrix[:, np.where(ec_idx == self.ecID[ec])[0]])
            if m != 0:
                mFeatures[self.ptwID[idx], 25] = total / m

            # 26. evidence-info-content-unnorm (numeric)
            mFeatures[self.ptwID[idx], 26] = total

            # 27. has-genes-in-directon (boolean)
            # 28. has-proximal-genes (boolean)
            # 29. fraction-genes-in-directon (numeric)
            # 30. num-genes-in-directon (numeric)
            # 31. fraction-proximal-genes (numeric)
            # 32. num-proximal-genes (numeric)
        return mFeatures

    def ECFeatures(self, infoList, matrixList, p_ec_idx, r_ec_idx, nFeatures=25, initialRxn=2, lastRxn=2):
        regex = re.compile(r'\(| |\)')
        mFeatures = np.zeros(shape=(len(self.ecID), nFeatures), dtype=np.object)
        ptwIDX = self.ReverseIdx(self.ptwID)
        rxnIDX = self.ReverseIdx(self.rxnID)

        for ec in self.ecID:
            ptw_lst = matrixList[0][:, np.where(p_ec_idx == self.ecID[ec])[0]].nonzero()[0]
            count_lst = matrixList[0][ptw_lst, np.where(p_ec_idx == self.ecID[ec])[0]]

            # 0.  num-pathways (numeric)
            mFeatures[self.ecID[ec], 0] = len(ptw_lst)

            # 1.  list-of-pathways (list)
            mFeatures[self.ecID[ec], 1] = ptw_lst

            # 2.  is-mapped-to-single-pathway (boolean)
            if len(ptw_lst) == 1:
                mFeatures[self.ecID[ec], 2] = 1

            for idx, pidx in enumerate(ptw_lst):
                ptw = infoList[0][ptwIDX[pidx]]

                # 3.  contributions-in-mapped-pathways (numeric)
                mFeatures[self.ecID[ec], 3] += count_lst[idx]

                # 4.  contributes-in-subpathway-as-inside-superpathways (boolean)
                # 5.  contributions-in-subpathway-as-inside-superpathways (numeric)
                if len(ptw[9][1]):
                    mFeatures[self.ecID[ec], 4] = 1
                    mFeatures[self.ecID[ec], 5] += len(ptw[9][1])

                text = str(ptw[0][1]) + ' ' + ' '.join(ptw[1][1]) + ' ' + ' '.join(ptw[6][1])
                text = text.lower()
                true_rxn_predecessors = [list(filter(None, regex.split(itm))) for itm in ptw[3][1]]
                for idx, itm in enumerate(true_rxn_predecessors):
                    itm = ' '.join(itm).replace('\"', '')
                    true_rxn_predecessors[idx] = itm.split()
                dg = nx.DiGraph()
                dg.add_nodes_from(ptw[4][1])
                for itm in true_rxn_predecessors:
                    if len(itm) == 1:
                        continue
                    else:
                        dg.add_edge(itm[1], itm[0])

                from operator import itemgetter
                initial_ec_lst = [n for n, d in sorted(dg.in_degree(), key=itemgetter(1))]
                initial_ec_lst = [j for itm in initial_ec_lst for j in infoList[1][itm][3][1] if
                                  j in ptw[15][1]]
                final_ec_lst = [n for n, d in sorted(dg.out_degree(), key=itemgetter(1))]
                final_ec_lst = [j for itm in final_ec_lst for j in infoList[1][itm][3][1] if
                                j in ptw[15][1]]

                # 6.  is-act-as-initial-reactions (boolean)
                # 7.  act-as-initial-reactions (numeric)
                if ec in initial_ec_lst[:initialRxn]:
                    mFeatures[self.ecID[ec], 6] = 1
                    mFeatures[self.ecID[ec], 7] += 1

                # 8.  is-act-as-final-reactions (boolean)
                # 9.  act-as-final-reactions (numeric)
                if ec in final_ec_lst[:lastRxn]:
                    mFeatures[self.ecID[ec], 8] = 1
                    mFeatures[self.ecID[ec], 9] += 1

                # 10.  is-act-as-initial-and-final-reactions (boolean)
                # 11.  act-as-initial-and-final-reactions  (numeric)
                if ec in initial_ec_lst[:initialRxn] and ec in final_ec_lst[:lastRxn]:
                    mFeatures[self.ecID[ec], 10] = 1
                    mFeatures[self.ecID[ec], 11] += 1

                # 12.  is-act-in-deg-or-detox-pathway (boolean)
                # 13.  act-in-deg-or-detox-pathway (numeric)
                if 'detoxification' in text or 'degradation' in text:
                    mFeatures[self.ecID[ec], 12] = 1
                    mFeatures[self.ecID[ec], 13] += 1

                # 14.  is-act-in-biosynthesis-pathway (boolean)
                # 15.  act-in-biosynthesis-pathway (numeric)
                if 'biosynthesis' in text:
                    mFeatures[self.ecID[ec], 14] = 1
                    mFeatures[self.ecID[ec], 15] += 1

                # 16. is-act-in-energy-pathway (boolean)
                # 17. act-in-energy-pathway (numeric)
                if 'energy' in text:
                    mFeatures[self.ecID[ec], 16] = 1
                    mFeatures[self.ecID[ec], 17] += 1

            rxn_lst = matrixList[1][:, np.where(r_ec_idx == self.ecID[ec])[0]].nonzero()[0]

            # 18.  num-reactions (numeric)
            mFeatures[self.ecID[ec], 18] = len(rxn_lst)

            # 19.  list-of-reactions (list)
            mFeatures[self.ecID[ec], 19] = rxn_lst

            # 20. is-unique-reaction (boolean)
            if len(rxn_lst) == 1:
                mFeatures[self.ecID[ec], 20] = 1

            # 21. reactions-orphaned (boolean)
            # 22. num-reactions-orphaned (numeric)
            # 23. reactions-has-species (boolean)
            # 24. reactions-has-taxonomic-range (boolean)
            for idx, ridx in enumerate(rxn_lst):
                rxn = infoList[1][rxnIDX[ridx]]
                if ec in rxn[3][1]:
                    if rxn[5][1] != False:
                        mFeatures[self.ecID[ec], 21] = 1
                        mFeatures[self.ecID[ec], 22] += 1
                    if len(rxn[8][1]):
                        mFeatures[self.ecID[ec], 23] = 1
                    if len(rxn[9][1]):
                        mFeatures[self.ecID[ec], 24] = 1
        return mFeatures

    ##########################################################################################################
    ##########################           FEATURES FROM EXPERIMENTAL DATA            ##########################
    ##########################################################################################################

    def ECEvidenceFeatures(self, instance, infoList, matrixList, col_idx, nFeatures=82, pFeatures=28, initialRxn=2,
                           lastRxn=2, threshold=0.5, beta=0.45):
        '''

        :param instance:
        :param infoList:
        :param matrixList:
        :param nFeatures:
        :param initialRxn:
        :param lastRxn:
        :param threshold:
        :return:
        '''
        oneHot = np.copy(instance)
        oneHot[oneHot > 0] = 1
        ec_idx = col_idx[np.argwhere(oneHot)[:, 0]]
        selectedList = matrixList[1][ec_idx, :]
        mFeatures = np.zeros(shape=(1, nFeatures), dtype=np.float32)
        possiblePathways = np.zeros(shape=(1, len(infoList[0])), dtype=np.float32)
        ratioPossiblePathways = np.zeros(shape=(1, len(infoList[0])), dtype=np.float32)
        pathwayFeatures = np.zeros(shape=(len(infoList[0]), pFeatures), dtype=np.float32)

        '''
        Extracting Various EC Features from Experimental Data
        '''
        # 0. fraction-total-ecs-to-distinct-ecs (numeric)
        mFeatures[0, 0] = np.sum(instance) / len(selectedList)

        # 1. fraction-total-possible-pathways-to-distinct-pathways (numeric)
        mFeatures[0, 1] = np.sum(selectedList[:, 0]) / len(np.unique(np.concatenate(selectedList[:, 1])))

        # 2. fraction-total-ecs-to-ecs-mapped-to-single-pathways (numeric)
        mFeatures[0, 2] = np.sum(selectedList[:, 2]) / np.sum(instance)

        # 3. fraction-total-ecs-mapped-to-pathways (numeric)
        mFeatures[0, 3] = np.sum(instance) / np.sum(selectedList[:, 3])

        # 4. fraction-total-distinct-ecs-contribute-in-subpathway-as-inside-superpathways (numeric)
        mFeatures[0, 4] = np.sum(selectedList[:, 4]) / np.sum(instance)

        # 5. fraction-total-ecs-contribute-in-subpathway-as-inside-superpathways (numeric)
        mFeatures[0, 5] = np.sum(selectedList[:, 5]) / np.sum(instance)

        # 6. fraction-total-distinct-ecs-act-as-initial-reactions (numeric)
        mFeatures[0, 6] = np.sum(selectedList[:, 6]) / np.sum(instance)

        # 7. fraction-total-ecs-act-as-initial-reactions (numeric)
        mFeatures[0, 7] = np.sum(selectedList[:, 7]) / np.sum(instance)

        # 8. fraction-total-distinct-ecs-act-as-final-reactions (numeric)
        mFeatures[0, 8] = np.sum(selectedList[:, 8]) / np.sum(instance)

        # 9. fraction-total-ecs-act-as-final-reactions (numeric)
        mFeatures[0, 9] = np.sum(selectedList[:, 9]) / np.sum(instance)

        # 10. fraction-total-distinct-ecs-act-as-initial-and-final-reactions (numeric)
        mFeatures[0, 10] = np.sum(np.sum(instance)) / np.sum(instance)

        # 11. fraction-total-ecs-act-as-initial-and-final-reactions  (numeric)
        mFeatures[0, 11] = np.sum(selectedList[:, 11]) / np.sum(instance)

        # 12. fraction-total-distinct-ecs-act-in-deg-or-detox-pathway (numeric)
        mFeatures[0, 12] = np.sum(selectedList[:, 12]) / np.sum(instance)

        # 13. fraction-total-ecs-act-in-deg-or-detox-pathway (numeric)
        mFeatures[0, 13] = np.sum(selectedList[:, 13]) / np.sum(instance)

        # 14. fraction-total-distinct-ec-act-in-biosynthesis-pathway (numeric)
        mFeatures[0, 14] = np.sum(selectedList[:, 14]) / np.sum(instance)

        # 15. fraction-total-ec-act-in-biosynthesis-pathway (numeric)
        mFeatures[0, 15] = np.sum(selectedList[:, 15]) / np.sum(instance)

        # 16. fraction-total-distinct-ec-act-in-energy-pathway (numeric)
        mFeatures[0, 16] = np.sum(selectedList[:, 16]) / np.sum(instance)

        # 17. fraction-total-ec-act-in-energy-pathway (numeric)
        mFeatures[0, 17] = np.sum(selectedList[:, 17]) / np.sum(instance)

        # 18. fraction-total-ecs-to-total-reactions (numeric)
        mFeatures[0, 18] = np.sum(instance) / np.sum(selectedList[:, 18])

        # 19. fraction-total-distinct-ecs-to-total-distinct-reactions (numeric)
        mFeatures[0, 19] = len(selectedList) / len(np.unique(np.concatenate(selectedList[:, 19])))

        # 20. fraction-total-ec-contribute-in-unique-reaction (numeric)
        mFeatures[0, 20] = np.sum(selectedList[:, 20]) / np.sum(instance)

        # 21. fraction-total-distinct-ec-contribute-to-reactions-has-taxonomic-range (numeric)
        mFeatures[0, 21] = np.sum(selectedList[:, 24]) / np.sum(instance)

        # 22. fraction-total-pathways-over-total-ecs (numeric)
        mFeatures[0, 22] = np.sum(selectedList[:, 0]) / np.sum(instance)

        # 23. fraction-total-pathways-over-distinct-ec (numeric)
        mFeatures[0, 23] = np.sum(selectedList[:, 0]) / len(selectedList)

        # 24. fraction-total-distinct-pathways-over-distinct-ec (numeric)
        mFeatures[0, 24] = len(np.unique(np.concatenate(selectedList[:, 1]))) / len(selectedList)

        # 25. fraction-distinct-ec-contributes-in-subpathway-over-distinct-pathways (numeric)
        mFeatures[0, 25] = np.sum(selectedList[:, 4]) / len(np.unique(np.concatenate(selectedList[:, 1])))

        # 26. fraction-ec-contributes-in-subpathway-over-total-pathways (numeric)
        mFeatures[0, 26] = np.sum(selectedList[:, 5]) / np.sum(selectedList[:, 0])

        # 27. fraction-distinct-ec-act-in-deg-or-detox-pathway-over-distinct-pathways (numeric)
        mFeatures[0, 27] = np.sum(selectedList[:, 12]) / len(np.unique(np.concatenate(selectedList[:, 1])))

        # 28. fraction-distinct-ec-act-in-deg-or-detox-pathway-over-total-pathways (numeric)
        mFeatures[0, 28] = np.sum(selectedList[:, 12]) / np.sum(selectedList[:, 0])

        # 29. fraction-ec-act-in-deg-or-detox-pathway-over-total-pathways (numeric)
        mFeatures[0, 29] = np.sum(selectedList[:, 13]) / np.sum(selectedList[:, 0])

        # 30. fraction-distinct-ec-act-in-biosynthesis-pathway-over-distinct-pathways (numeric)
        mFeatures[0, 30] = np.sum(selectedList[:, 14]) / len(np.unique(np.concatenate(selectedList[:, 1])))

        # 31. fraction-distinct-ec-act-in-biosynthesis-pathway-over-total-pathways (numeric)
        mFeatures[0, 31] = np.sum(selectedList[:, 14]) / np.sum(selectedList[:, 0])

        # 32. fraction-ec-act-in-biosynthesis-pathway-over-total-pathways (numeric)
        mFeatures[0, 32] = np.sum(selectedList[:, 15]) / np.sum(selectedList[:, 0])

        # 33. fraction-distinct-ec-act-in-energy-pathway-over-distinct-pathways (numeric)
        mFeatures[0, 33] = np.sum(selectedList[:, 16]) / len(np.unique(np.concatenate(selectedList[:, 1])))

        # 34. fraction-distinct-ec-act-in-energy-pathway-over-total-pathways (numeric)
        mFeatures[0, 34] = np.sum(selectedList[:, 16]) / np.sum(selectedList[:, 0])

        # 35. fraction-ec-act-in-energy-pathway-over-total-pathways (numeric)
        mFeatures[0, 35] = np.sum(selectedList[:, 17]) / np.sum(selectedList[:, 0])

        # 36. fraction-total-reactions-over-total-pathways (numeric)
        mFeatures[0, 36] = np.sum(selectedList[:, 18]) / np.sum(selectedList[:, 0])

        # 37. fraction-total-reactions-over-distinct-pathways (numeric)
        mFeatures[0, 37] = np.sum(selectedList[:, 18]) / len(np.unique(np.concatenate(selectedList[:, 1])))

        # 38. fraction-distinct-reaction-over-distinct-pathways (numeric)
        mFeatures[0, 38] = len(np.unique(np.concatenate(selectedList[:, 19]))) / len(
            np.unique(np.concatenate(selectedList[:, 1])))

        '''
        Extracting Pathway Features from Experimental Data
        '''

        regex = re.compile(r'\(| |\)')
        ptwIDX = self.ReverseIdx(self.ptwID)

        for pidx in np.unique(np.concatenate(selectedList[:, 1])):
            binPtw = matrixList[0][pidx]
            binPtw[binPtw > 0] = 1
            nCommonItems = (oneHot & binPtw).sum()
            if int(nCommonItems) != 0:
                ratioPossiblePathways[0, pidx] = np.divide(nCommonItems, binPtw.sum())
            pid = ptwIDX[pidx]
            ptw = infoList[0][pid]
            text = str(ptw[0][1]) + ' ' + ' '.join(ptw[1][1]) + ' ' + ' '.join(ptw[6][1])
            text = text.lower()
            true_rxn_predecessors = [list(filter(None, regex.split(itm))) for itm in infoList[0][pid][3][1]]
            for idx, itm in enumerate(true_rxn_predecessors):
                itm = ' '.join(itm).replace('\"', '')
                true_rxn_predecessors[idx] = itm.split()
            dg = nx.DiGraph()
            dg.add_nodes_from(infoList[0][pid][4][1])
            for itm in true_rxn_predecessors:
                if len(itm) == 1:
                    continue
                else:
                    dg.add_edge(itm[1], itm[0])

            true_prev_ec_lst = [n for n, d in sorted(dg.in_degree(), key=itemgetter(1))]
            true_prev_ec_lst = [self.ecID[j] for itm in true_prev_ec_lst for j in infoList[1][itm][3][1] if
                                j in infoList[0][pid][15][1]]
            true_succ_ec_lst = [n for n, d in sorted(dg.out_degree(), key=itemgetter(1))]
            true_succ_ec_lst = [self.ecID[j] for itm in true_succ_ec_lst for j in infoList[1][itm][3][1] if
                                j in infoList[0][pid][15][1]]

            unique_ec_lst = list()
            orphan_ec_lst = list()
            for rxn in ptw[16][1]:
                if infoList[1][rxn][3][1]:
                    e = [self.ecID[e] for e in infoList[1][rxn][3][1]]
                    unique_ec_lst.extend(e)
                if infoList[1][rxn][5][1] != False:
                    orphan_ec_lst.extend(e)

            '''
            Extracting Info from Knowledge Data
            '''

            sample_ec_lst = [ec for ec in true_prev_ec_lst if ec in ec_idx]
            sample_unique_ec_lst = [ec for ec in sample_ec_lst if ec in unique_ec_lst]
            if true_prev_ec_lst:
                sample_initial_ec_lst = [1 for ec in true_prev_ec_lst[:initialRxn] if ec not in sample_ec_lst]
            if true_succ_ec_lst:
                sample_final_ec_lst = [1 for ec in true_succ_ec_lst[:lastRxn] if ec not in sample_ec_lst]

            '''
            Continue Pathway Features Extracting
            '''
            # 39. ecs-in-energy-pathways-mostly-missing (numeric)
            if 'energy' in text:
                if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                    mFeatures[0, 39] += 1

            # 40. ecs-in-pathways-mostly-present (numeric)
            # 0. ecs-mostly-present-in-pathway (boolean)
            # 1. prob-ecs-mostly-present-in-pathway (numeric)
            if sample_ec_lst:
                if len(sample_ec_lst) > int(threshold * len(true_prev_ec_lst)):
                    m1 = len(sample_ec_lst) + 1 == len(ptw[15][1])
                    m2 = (len(sample_ec_lst) / len(ptw[15][1])) >= threshold
                    if m1 and m2:
                        mFeatures[0, 40] += 1
                        pathwayFeatures[pidx, 0] = 1
                        pathwayFeatures[pidx, 1] = len(sample_ec_lst) / len(ptw[15][1])

            # 41. all-initial-ecs-present-in-pathways (numeric)
            # 2. all-initial-ecs-present-in-pathway (boolean)
            # 3. prob-initial-ecs-present-in-pathway (numeric)
            if sample_initial_ec_lst:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initialRxn]):
                    mFeatures[0, 41] += 1
                    pathwayFeatures[pidx, 2] = 1
                else:
                    pathwayFeatures[pidx, 3] = len(sample_initial_ec_lst) / len(true_prev_ec_lst[:initialRxn])

            # 42. all-final-ecs-present-in-pathways (numeric)
            # 4.  all-final-ecs-present-in-pathway (boolean)
            # 5.  prob-final-ecs-present-in-pathway (numeric)
            if sample_final_ec_lst:
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:lastRxn]):
                    mFeatures[0, 42] += 1
                    pathwayFeatures[pidx, 4] = 1
                else:
                    pathwayFeatures[pidx, 5] = len(sample_final_ec_lst) / len(true_succ_ec_lst[:lastRxn])

            # 43. all-initial-and-final-ecs-present-in-pathways (numeric)
            # 6.  all-initial-and-final-ecs-present-in-pathway (boolean)
            # 7.  prob-all-initial-and-final-ecs-present-in-pathway (numeric)
            if sample_initial_ec_lst and sample_final_ec_lst:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initialRxn]) and len(sample_final_ec_lst) == len(
                        true_succ_ec_lst[:lastRxn]):
                    mFeatures[0, 43] += 1
                    pathwayFeatures[pidx, 6] = 1
                    totalPECs = len(sample_initial_ec_lst) + len(sample_final_ec_lst)
                    totalTECs = len(true_prev_ec_lst[:initialRxn]) + len(true_succ_ec_lst[:lastRxn])
                    pathwayFeatures[pidx, 7] = totalPECs / totalTECs

            # 44. all-initial-ecs-present-in-deg-or-detox-pathways (numeric)
            # 45. all-final-ecs-present-in-deg-or-detox-pathways (numeric)
            # 8. all-initial-ecs-present-in-deg-or-detox-pathway (boolean)
            # 9. prob-all-initial-ecs-present-in-deg-or-detox-pathway (numeric)
            if 'detoxification' in text or 'degradation' in text:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initialRxn]):
                    mFeatures[0, 44] += 1
                    pathwayFeatures[pidx, 8] = 1
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:lastRxn]):
                    mFeatures[0, 45] += 1
                    pathwayFeatures[pidx, 9] = 1

            # 46. all-initial-ecs-present-in-biosynthesis-pathways (numeric)
            # 47. all-final-ecs-present-in-biosynthesis-pathways (numeric)
            # 10. all-initial-ecs-present-in-biosynthesis-pathway (boolean)
            # 11. prob-all-initial-ecs-present-in-biosynthesis-pathway (numeric)
            if 'biosynthesis' in text:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initialRxn]):
                    mFeatures[0, 46] += 1
                    pathwayFeatures[pidx, 10] = 1
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:lastRxn]):
                    mFeatures[0, 47] += 1
                    pathwayFeatures[pidx, 11] = 1

            # 48. most-ecs-absent-in-pathways (numeric)
            # 12. most-ecs-absent-in-pathway (boolean)
            if len(sample_ec_lst) == 1:
                if sample_ec_lst[0] in unique_ec_lst:
                    mFeatures[0, 48] += 1
                    pathwayFeatures[0, 12] = 1

            # 49. most-ecs-absent-not-distinct-in-pathways (numeric)
            # 13. most-ecs-absent-not-distinct-in-pathway (boolean)
            if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                if unique_ec_lst:
                    mFeatures[0, 49] += 1
                    pathwayFeatures[0, 13] = 1
                    for e in unique_ec_lst:
                        if e in sample_ec_lst:
                            mFeatures[0, 49] -= 1
                            pathwayFeatures[0, 13] = 0
                            break

            # 50. one-ec-present-but-in-minority-in-pathways (numeric)
            # 14. one-ec-present-but-in-minority-in-pathway (boolen)
            if len(sample_ec_lst) == 1:
                if sample_ec_lst[0] in unique_ec_lst:
                    mFeatures[0, 50] += 1
                    pathwayFeatures[0, 14] = 1

            # 51. all-distinct-ec-present-in-pathways (numeric)
            # 52. all-ecs-present-in-pathways (numeric)
            # 53. all-distinct-ec-present-or-orphaned-in-pathways (numeric)
            # 54. all-ec-present-or-orphaned-in-pathways (numeric)

            # 15. all-distinct-ec-present-in-pathway (boolean)
            # 16. all-ecs-present-in-pathway (boolean)
            # 17. all-distinct-ec-present-or-orphaned-in-pathway (boolean)
            # 18. all-ec-present-or-orphaned-in-pathway (boolean)

            mFeatures[0, 51] += 1
            mFeatures[0, 52] += 1
            mFeatures[0, 53] += 1
            mFeatures[0, 54] += 1

            pathwayFeatures[pidx, 15] = 1
            pathwayFeatures[pidx, 16] = 1
            pathwayFeatures[pidx, 17] = 1
            pathwayFeatures[pidx, 18] = 1

            if sample_unique_ec_lst != unique_ec_lst:
                mFeatures[0, 51] -= 1
                pathwayFeatures[pidx, 15] = 0

            if sample_ec_lst != true_prev_ec_lst:
                mFeatures[0, 52] -= 1
                pathwayFeatures[pidx, 16] = 0
            u = 0
            for ec in unique_ec_lst:
                if ec in sample_unique_ec_lst or ec in orphan_ec_lst:
                    u += 1
            if u != len(unique_ec_lst):
                mFeatures[0, 53] -= 1
                pathwayFeatures[pidx, 17] = 0
            a = 0
            for ec in true_prev_ec_lst:
                if ec in sample_ec_lst or ec in orphan_ec_lst:
                    a += 1
            if a != len(true_prev_ec_lst):
                mFeatures[0, 54] -= 1
                pathwayFeatures[pidx, 18] = 0

            # 55. majority-of-ecs-absent-in-pathways (numeric)
            # 56. majority-of-ecs-present-in-pathways (numeric)
            # 19. majority-of-ecs-absent-in-pathway (boolean)
            # 20. majority-of-ecs-present-in-pathway (boolean)
            if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                mFeatures[0, 55] += 1
                pathwayFeatures[pidx, 19] = 1
            else:
                mFeatures[0, 56] += 1
                pathwayFeatures[pidx, 20] = 1

            # 57. majority-of-distinct-ecs-present-in-pathways (numeric)
            # 21. majority-of-distinct-ecs-present-in-pathway (boolean)
            if len(sample_unique_ec_lst) > int(threshold * len(unique_ec_lst)):
                mFeatures[0, 57] += 1
                pathwayFeatures[pidx, 21] = 1

            # 58. majority-of-reactions-present-distinct-in-pathways (numeric)
            # 22. majority-of-reactions-present-distinct-in-pathway (boolean)
            if len(sample_ec_lst) > int(threshold * len(true_prev_ec_lst)):
                mFeatures[0, 58] += 1
                pathwayFeatures[pidx, 22] = 1
                for ec in sample_ec_lst:
                    if ec not in unique_ec_lst:
                        mFeatures[0, 58] -= 1
                        pathwayFeatures[pidx, 22] = 0
                        break

            # 59.  missing-at-most-one-ec-in-pathways (numeric)
            # 23.  missing-at-most-one-ec-in-pathway (boolean)
            if len(sample_ec_lst) + 1 == len(true_prev_ec_lst):
                ec = set.difference(set(sample_ec_lst), set(true_prev_ec_lst))
                if ec not in orphan_ec_lst:
                    mFeatures[0, 59] += 1
                    pathwayFeatures[pidx, 23] = 1

            # 60.  has-distinct-ecs-present-in-pathways (numeric)
            # 24.  has-distinct-ecs-present-in-pathway (boolean)
            if sample_unique_ec_lst:
                mFeatures[0, 60] += 1
                pathwayFeatures[pidx, 24] = 1

            # 62.  fraction-reactions-present-or-orphaned-distinct-in-pathways (numeric)
            # 26.  fraction-reactions-present-or-orphaned-distinct-in-pathway (numeric)
            sample_rxn_unique_orphand = set.union(set(sample_unique_ec_lst), set(orphan_ec_lst))
            true_ec_orphand = set.union(set(true_prev_ec_lst), set(orphan_ec_lst))
            mFeatures[0, 61] += len(sample_rxn_unique_orphand) / len(true_ec_orphand)
            pathwayFeatures[pidx, 25] = len(sample_rxn_unique_orphand) / len(true_ec_orphand)

            # 61.  fraction-distinct-ecs-present-or-orphaned-in-pathways (numeric)
            # 25.  fraction-distinct-ecs-present-or-orphaned-in-pathway (numeric)
            if len(unique_ec_lst):
                true_ec_unique_orphand = set.union(set(unique_ec_lst), set(orphan_ec_lst))
                mFeatures[0, 62] += len(sample_rxn_unique_orphand) / len(true_ec_unique_orphand)
                pathwayFeatures[pidx, 26] = len(sample_rxn_unique_orphand) / len(true_ec_unique_orphand)

            # 63.  fraction-reactions-present-or-orphaned-in-pathways (numeric)
            # 27.  fraction-reactions-present-or-orphaned-in-pathway (numeric)
            sample_rxn_orphand = set.union(set(sample_ec_lst), set(orphan_ec_lst))
            mFeatures[0, 63] += len(sample_rxn_orphand) / len(true_ec_orphand)
            pathwayFeatures[pidx, 27] = len(sample_rxn_orphand) / len(true_ec_orphand)

            # 64.  num-distinct-reactions-present-or-orphaned-in-pathways (numeric)
            # 28.  num-distinct-reactions-present-or-orphaned-in-pathway (numeric)
            mFeatures[0, 64] += len(sample_rxn_unique_orphand)
            pathwayFeatures[pidx, 28] = len(sample_rxn_unique_orphand)

            # 65.  num-reactions-present-or-orphaned-in-pathways (numeric)
            # 29.  num-reactions-present-or-orphaned-in-pathway (numeric)
            mFeatures[0, 65] += len(sample_rxn_orphand)
            pathwayFeatures[pidx, 29] = len(sample_rxn_orphand)

            # 66.  evidence-info-content-norm-present-in-pathways (numeric)
            # 67.  evidence-info-content-present-in-pathways (numeric)
            # 30.  evidence-info-content-norm-present-in-pathway (numeric)
            # 31.  evidence-info-content-present-in-pathway (numeric)
            total = 0
            for ec in set(sample_ec_lst):
                total += 1 / np.count_nonzero(matrixList[0][:, np.where(col_idx == ec)[0]])
            if sample_ec_lst:
                mFeatures[0, 66] += total / len(true_prev_ec_lst)
                mFeatures[0, 67] += total
                pathwayFeatures[pidx, 30] = total / len(true_prev_ec_lst)
                pathwayFeatures[pidx, 31] = total

        mFeatures[0, 39:] = mFeatures[0, 39:] / pathwayFeatures.shape[0]
        pathwayFeatures = pathwayFeatures.reshape(1, pathwayFeatures.shape[0] * pathwayFeatures.shape[1])
        # 68.  possible-pathways-present (boolean)
        maxval = np.max(ratioPossiblePathways) * beta
        possiblePathways[ratioPossiblePathways >= maxval] = 1
        possiblePathways[ratioPossiblePathways >= threshold] = 1
        # 69.  prob-possible-pathways-present (numeric)
        ratioPossiblePathways = ratioPossiblePathways
        return np.hstack((mFeatures, possiblePathways, ratioPossiblePathways, pathwayFeatures))

    def ReactionEvidenceFeatures(self, instance, infoList, ptw_ec_matrix, nFeatures=42, initialRxn=2, lastRxn=2,
                                 threshold=0.5):
        mFeatures = np.zeros(shape=(len(self.ptwID), nFeatures), dtype=np.float32)
        regex = re.compile(r'\(| |\)')

        for id in self.ptwID:
            ptw = infoList[0][id]
            text = str(ptw[0][1]) + ' ' + ' '.join(ptw[1][1]) + ' ' + ' '.join(ptw[6][1])
            text = text.lower()
            true_rxn_predecessors = [list(filter(None, regex.split(itm))) for itm in infoList[0][id][3][1]]
            for idx, itm in enumerate(true_rxn_predecessors):
                itm = ' '.join(itm).replace('\"', '')
                true_rxn_predecessors[idx] = itm.split()
            dg = nx.DiGraph()
            dg.add_nodes_from(infoList[0][id][4][1])
            for itm in true_rxn_predecessors:
                if len(itm) == 1:
                    continue
                else:
                    dg.add_edge(itm[1], itm[0])
            true_prev_ec_lst = sorted(dg.in_degree(), key=dg.in_degree().get)
            true_prev_ec_lst = [j for itm in true_prev_ec_lst for j in infoList[1][itm][3][1] if
                                j in infoList[0][id][15][1]]
            true_succ_ec_lst = sorted(dg.out_degree(), key=dg.out_degree().get)
            true_succ_ec_lst = [j for itm in true_succ_ec_lst for j in infoList[1][itm][3][1] if
                                j in infoList[0][id][15][1]]

            unique_ec_lst = list()
            orphan_ec_lst = list()
            for rxn in ptw[16][1]:
                if infoList[1][rxn][3][1]:
                    e = [e for e in infoList[1][rxn][3][1]]
                    unique_ec_lst.extend(e)
                if infoList[1][rxn][5][1] != False:
                    orphan_ec_lst.extend(e)

            #######################################################################################################
            ################################ Extracting Info from Experimental Data ###############################
            #######################################################################################################

            sample_ec_lst = [ec for ec in ptw[15][1] if instance[:, self.ecID[ec]] != 0]
            sample_unique_ec_lst = [ec for ec in sample_ec_lst if ec in unique_ec_lst]
            sample_orphan_ec_lst = [ec for ec in sample_ec_lst if ec in orphan_ec_lst]
            if true_prev_ec_lst:
                sample_initial_ec_lst = [1 for ec in true_prev_ec_lst[:initialRxn] if ec not in sample_ec_lst]
            if true_succ_ec_lst:
                sample_final_ec_lst = [1 for ec in true_succ_ec_lst[:lastRxn] if ec not in sample_ec_lst]

            #######################################################################################################
            ############################## Extracting Features from Experimental Data #############################
            #######################################################################################################

            # 0. energy-pathway-mostly-missing (boolean)
            if 'energy' in text:
                if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                    mFeatures[self.ptwID[id], 0] = 1

            # 1. mostly-present (boolean)
            if len(sample_ec_lst) > int(threshold * len(true_prev_ec_lst)):
                m1 = len(sample_ec_lst) + 1 == len(ptw[15][1])
                m2 = (len(sample_ec_lst) / len(ptw[15][1])) >= threshold
                if m1 and m2:
                    mFeatures[self.ptwID[id], 1] = 1

            # 2.   some-initial-reactions-present (boolean)
            if sample_initial_ec_lst:
                mFeatures[self.ptwID[id], 2] = 1

            # 3.  all-initial-reactions-present (boolean)
            if sample_initial_ec_lst:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initialRxn]):
                    mFeatures[self.ptwID[id], 3] = 1

            # 4.   some-final-reactions-present (boolean)
            if sample_final_ec_lst:
                mFeatures[self.ptwID[id], 4] = 1

            # 5.   all-final-reactions-present (boolean)
            if sample_final_ec_lst:
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:lastRxn]):
                    mFeatures[self.ptwID[id], 5] = 1

            # 6.   some-initial-and-final-reactions-present (boolean)
            if sample_initial_ec_lst and sample_final_ec_lst:
                mFeatures[self.ptwID[id], 6] = 1

            # 7.   all-initial-and-final-reactions-present  (boolean)
            if sample_initial_ec_lst and sample_final_ec_lst:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initialRxn]) and len(sample_final_ec_lst) == len(
                        true_succ_ec_lst[:lastRxn]):
                    mFeatures[self.ptwID[id], 7] = 1

            # 8.   deg-or-detox-pathway-all-initial-reactions-present (boolean)
            # 9.   deg-or-detox-pathway-all-final-reactions-present (boolean)
            if 'detoxification' in text or 'degradation' in text:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initialRxn]):
                    mFeatures[self.ptwID[id], 8] = 1
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:lastRxn]):
                    mFeatures[self.ptwID[id], 9] = 1

            # 10.   biosynthesis-pathway-all-initial-reactions-present (boolean)
            # 11.   biosynthesis-pathway-all-final-reactions-present (boolean)
            if 'biosynthesis' in text:
                if len(sample_initial_ec_lst) == len(true_prev_ec_lst[:initialRxn]):
                    mFeatures[self.ptwID[id], 10] = 1
                if len(sample_final_ec_lst) == len(true_succ_ec_lst[:lastRxn]):
                    mFeatures[self.ptwID[id], 11] = 1

            # 12.  mostly-absent (boolean)
            if len(sample_ec_lst) == 0:
                mFeatures[self.ptwID[id], 12] = 1
            elif len(sample_ec_lst) == 1:
                if sample_ec_lst[0] in unique_ec_lst:
                    mFeatures[self.ptwID[id], 12] = 1

            # 13.  mostly-absent-not-unique (boolean)
            if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                if unique_ec_lst:
                    mFeatures[self.ptwID[id], 13] = 1
                    for e in unique_ec_lst:
                        if e in sample_ec_lst:
                            mFeatures[self.ptwID[id], 13] = 0
                            break

            # 14.  one-reaction-present-but-in-minority (boolean)
            if len(sample_ec_lst) == 1:
                if sample_ec_lst[0] in unique_ec_lst:
                    mFeatures[self.ptwID[id], 14] = 1

            # 15.  every-unique-reaction-present (boolean)
            # 16.  every-reaction-present (boolean)
            # 17.  every-unique-reaction-present-or-orphaned (boolean)
            # 18.  every-reaction-present-or-orphaned (boolean)
            if sample_ec_lst:
                mFeatures[self.ptwID[id], 15] = 1
                mFeatures[self.ptwID[id], 16] = 1
                mFeatures[self.ptwID[id], 17] = 1
                mFeatures[self.ptwID[id], 18] = 1
                if sample_ec_lst != unique_ec_lst:
                    mFeatures[self.ptwID[id], 15] = 0
                if sample_ec_lst != true_prev_ec_lst:
                    mFeatures[self.ptwID[id], 16] = 0
                u = 0
                for ec in unique_ec_lst:
                    if ec in sample_unique_ec_lst or ec in orphan_ec_lst:
                        u += 1
                if u != len(unique_ec_lst):
                    mFeatures[self.ptwID[id], 17] = 0
                a = 0
                for ec in true_prev_ec_lst:
                    if ec in sample_ec_lst or ec in orphan_ec_lst:
                        a += 1
                if a != len(true_prev_ec_lst):
                    mFeatures[self.ptwID[id], 18] = 0

            # 19.  majority-of-reactions-absent (boolean)
            # 20.  majority-of-reactions-present (boolean)
            if len(sample_ec_lst) < int(threshold * len(true_prev_ec_lst)):
                mFeatures[self.ptwID[id], 19] = 1
            else:
                mFeatures[self.ptwID[id], 20] = 1

            # 21.  majority-of-unique-reactions-present (boolean)
            if len(sample_unique_ec_lst) > int(threshold * len(unique_ec_lst)):
                mFeatures[self.ptwID[id], 21] = 1

            # 22.  majority-of-reactions-present-unique (boolean)
            if len(sample_ec_lst) > int(threshold * len(true_prev_ec_lst)):
                mFeatures[self.ptwID[id], 22] = 1
                for ec in sample_ec_lst:
                    if ec not in unique_ec_lst:
                        mFeatures[self.ptwID[id], 22] = 0
                        break

            # 23.  missing-at-most-one-reaction (boolean)
            if len(sample_ec_lst) + 1 == len(true_prev_ec_lst):
                ec = set.difference(set(sample_ec_lst), set(true_prev_ec_lst))
                if ec not in orphan_ec_lst:
                    mFeatures[self.ptwID[id], 23] = 1

            # 24.  has-unique-reactions-present (boolean)
            if sample_ec_lst:
                if sample_unique_ec_lst:
                    mFeatures[self.ptwID[id], 24] = 1

            # 25.  has-reactions-present (boolean)
            if sample_ec_lst:
                mFeatures[self.ptwID[id], 25] = 1

            # 26.  fraction-final-reactions-present (numeric)
            if len(true_prev_ec_lst):
                mFeatures[self.ptwID[id], 26] = len(sample_final_ec_lst) / len(true_prev_ec_lst)

            # 27.  fraction-initial-reactions-present (numeric)
            if len(true_prev_ec_lst):
                mFeatures[self.ptwID[id], 27] = len(sample_initial_ec_lst) / len(true_prev_ec_lst)

            # 28.  num-final-reactions-present (numeric)
            mFeatures[self.ptwID[id], 28] = len(sample_final_ec_lst)

            # 29.  num-initial-reactions-present (numeric)
            mFeatures[self.ptwID[id], 29] = len(sample_initial_ec_lst)

            # 30.  fraction-unique-reactions-present-or-orphaned (numeric)
            sample_rxn_unique_orphand = set.union(set(sample_unique_ec_lst), set(orphan_ec_lst))
            if len(true_prev_ec_lst):
                mFeatures[self.ptwID[id], 30] = len(sample_rxn_unique_orphand) / len(true_prev_ec_lst)

            # 31.  fraction-reactions-present-or-orphaned-unique (numeric)
            if len(unique_ec_lst):
                mFeatures[self.ptwID[id], 31] = len(sample_rxn_unique_orphand) / len(unique_ec_lst)

            # 32.  fraction-reactions-present-or-orphaned (numeric)
            sample_rxn_orphand = set.union(set(sample_ec_lst), set(orphan_ec_lst))
            if len(true_prev_ec_lst):
                mFeatures[self.ptwID[id], 32] = len(sample_rxn_orphand) / len(true_prev_ec_lst)

            # 33.  fraction-unique-reactions-present (numeric)
            if len(true_prev_ec_lst):
                mFeatures[self.ptwID[id], 33] = len(sample_unique_ec_lst) / len(true_prev_ec_lst)

            # 34.  fraction-reactions-present-unique (numeric)
            if len(unique_ec_lst):
                mFeatures[self.ptwID[id], 34] = len(sample_unique_ec_lst) / len(unique_ec_lst)

            # 35.  fraction-reactions-present (numeric)
            if len(true_prev_ec_lst):
                mFeatures[self.ptwID[id], 35] = len(sample_ec_lst) / len(true_prev_ec_lst)

            # 36.  num-unique-reactions-present-or-orphaned (numeric)
            mFeatures[self.ptwID[id], 36] = len(sample_rxn_unique_orphand)

            # 37.  num-reactions-present-or-orphaned (numeric)
            mFeatures[self.ptwID[id], 37] = len(sample_rxn_orphand)

            # 38.  num-unique-reactions-present (numeric)
            mFeatures[self.ptwID[id], 38] = len(sample_unique_ec_lst)

            # 39.  num-reactions-present (numeric)
            mFeatures[self.ptwID[id], 39] = len(sample_ec_lst)

            # 40.  evidence-info-content-norm-present (numeric)
            total = 0
            for ec in set(sample_ec_lst):
                total += 1 / np.count_nonzero(ptw_ec_matrix[:, self.ecID[ec]])
            if sample_ec_lst:
                mFeatures[self.ptwID[id], 40] = total / len(sample_ec_lst)

            # 41.  evidence-info-content-unnorm-present (numeric)
            mFeatures[self.ptwID[id], 41] = total

        return mFeatures
