'''
This file preprocesses two pathway files: pathways.dat and
pathways.col from BioCyc PGDBs to an appropriate format that
is could be used as inputs to designated machine learning
models.
'''

import os
import os.path
from collections import OrderedDict


class Pathway(object):
    def __init__(self, ptwFname='pathways.dat', ptwFnamewithGenes='pathways.col'):
        """ Initialization
        :param ptwFnameLinks:
        :type ptwFname: str
        :param ptwFname: file name for the pathway
        """
        self.pathway_fname = ptwFname
        self.pathway_genes_fname = ptwFnamewithGenes
        self.superpathways = list()
        self.pathway_info = OrderedDict()

    def ProcessPathways(self, p_id, lst_ids, data_path):
        file = os.path.join(data_path, self.pathway_fname)
        if os.path.isfile(file):
            print('\t\t\t--> Prepossessing pathways from: {0}'.format(file.split(os.sep)[-1]))
            with open(file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split()
                        if ls:
                            if ls[0] == 'UNIQUE-ID':
                                pathway_id = ' '.join(ls[2:])
                                pathway_name = ''
                                lst_pathway_types = list()
                                lst_pathways_links = list()
                                lst_predecessors = list()
                                lst_reactions = list()
                                lst_reactions_layout = list()
                                lst_synonyms = list()
                                lst_species = list()
                                lst_taxonomic_range = list()
                                lst_sup_pathways = list()
                                lst_sub_pathways = list()
                            elif ls[0] == 'COMMON-NAME':
                                pathway_name = ' '.join(ls[2:])
                            elif ls[0] == 'TYPES':
                                lst_pathway_types.append(' '.join(ls[2:]))
                            elif ls[0] == 'PATHWAY-LINKS':
                                lst_pathways_links.append(' '.join(ls[2:]))
                            elif ls[0] == 'PREDECESSORS':
                                lst_predecessors.append(' '.join(ls[2:]))
                            elif ls[0] == 'REACTION-LIST':
                                lst_reactions.append(' '.join(ls[2:]))
                            elif ls[0] == 'REACTION-LAYOUT':
                                lst_reactions_layout.append(' '.join(ls[2:]))
                            elif ls[0] == 'SYNONYMS':
                                lst_synonyms.append(' '.join(ls[2:]))
                            elif ls[0] == 'SPECIES':
                                lst_species.append(' '.join(ls[2:]))
                            elif ls[0] == 'TAXONOMIC-RANGE':
                                lst_taxonomic_range.append(' '.join(ls[2:]))
                            elif ls[0] == 'SUPER-PATHWAYS':
                                lst_sup_pathways.append(' '.join(ls[2:]))
                            elif ls[0] == 'SUB-PATHWAYS':
                                lst_sub_pathways.append(' '.join(ls[2:]))
                            elif ls[0] == '//':
                                if 'Super-Pathways' in lst_pathway_types:
                                    self.superpathways.append(pathway_id)
                                    continue
                                if pathway_id not in lst_ids[p_id]:
                                    lst_ids[p_id].update({pathway_id: len(lst_ids[p_id])})

                                if pathway_id not in self.pathway_info:
                                    # datum is comprised of {UNIQUE-ID: (COMMON-NAME, TYPES, PATHWAY-LINKS, PREDECESSORS,
                                    # REACTION-LIST, REACTION-LAYOUT, SYNONYMS, SPECIES, TAXONOMIC-RANGE, SUPER-PATHWAYS,
                                    # SUB-PATHWAYS)}
                                    datum = {pathway_id: (['COMMON-NAME', pathway_name],
                                                          ['TYPES', lst_pathway_types],
                                                          ['PATHWAY-LINKS', lst_pathways_links],
                                                          ['PREDECESSORS', lst_predecessors],
                                                          ['REACTION-LIST', lst_reactions],
                                                          ['REACTION-LAYOUT', lst_reactions_layout],
                                                          ['SYNONYMS', lst_synonyms],
                                                          ['SPECIES', lst_species],
                                                          ['TAXONOMIC-RANGE', lst_taxonomic_range],
                                                          ['SUPER-PATHWAYS', lst_sup_pathways],
                                                          ['SUB-PATHWAYS', lst_sub_pathways])}
                                    self.pathway_info.update(datum)

    def ProcessPathwaysCol(self, p_id, lst_ids, data_path, header=False):
        file = os.path.join(data_path, self.pathway_genes_fname)
        if os.path.isfile(file):
            print('\t\t\t--> Prepossessing pathways from: {0}'.format(file.split(os.sep)[-1]))
            with open(file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split('\t')
                        if ls:
                            if not header:
                                if ls[0] == 'UNIQUE-ID':
                                    header = True
                                    lst_g_name_idx = list()
                                    lst_g_idx = list()
                                    for (i, item) in enumerate(ls):
                                        if item == 'UNIQUE-ID':
                                            pathway_idx = i
                                        elif item == 'NAME':
                                            pathway_name_idx = i
                                        elif item == 'GENE-NAME':
                                            lst_g_name_idx.append(i)
                                        elif item == 'GENE-ID':
                                            lst_g_idx.append(i)
                            else:
                                if ls[pathway_idx] in self.superpathways:
                                    continue
                                if ls[pathway_idx] not in lst_ids[p_id]:
                                    lst_ids[p_id].update({ls[pathway_idx]: len(lst_ids[p_id])})

                                if ls[pathway_idx] in self.pathway_info:
                                    (pathway_name, lst_pathway_types, lst_pathways_links, lst_predecessors,
                                     lst_reactions, lst_reactions_layout, lst_synonyms, lst_species,
                                     lst_taxonomic_range, lst_sup_pathways, lst_sub_pathways) = self.pathway_info[
                                        ls[pathway_idx]]

                                lst_genes_names = ls[pathway_name_idx + 1: lst_g_name_idx[-1] + 1]
                                lst_genes_ids = ls[lst_g_name_idx[-1] + 1:]
                                lst_genes_names = [gn for gn in lst_genes_names if gn]
                                lst_genes_ids = [gid for gid in lst_genes_ids if gid]

                                # datum is comprised of {UNIQUE-ID: (COMMON-NAME, TYPES, PATHWAY-LINKS, PREDECESSORS,
                                # REACTION-LIST, REACTION-LAYOUT, SYNONYMS, SPECIES, TAXONOMIC-RANGE, SUPER-PATHWAYS,
                                # SUB-PATHWAYS, GENES-NAME, GENES-ID)}
                                datum = {ls[pathway_idx]: (pathway_name,
                                                           lst_pathway_types,
                                                           lst_pathways_links,
                                                           lst_predecessors,
                                                           lst_reactions,
                                                           lst_reactions_layout,
                                                           lst_synonyms,
                                                           lst_species,
                                                           lst_taxonomic_range,
                                                           lst_sup_pathways,
                                                           lst_sub_pathways,
                                                           ['GENES-NAME', lst_genes_names],
                                                           ['GENES-ID', lst_genes_ids])}
                                self.pathway_info.update(datum)

    def AddPathwayProperties(self, reactions_info, ec_position, inptw_position, orphn_position, spon_position):
        print('\t\t\t--> Include enzymatic reactions to pathways')
        for (p_id, p_item) in self.pathway_info.items():
            (pathway_name, lst_pathway_types, lst_pathways_links, lst_predecessors, lst_reactions,
             lst_reactions_layout, lst_synonyms, lst_species, lst_taxonomic_range, lst_sup_pathways, lst_sub_pathways,
             lst_genes_names, lst_genes_ids) = p_item
            spontaneous = 0
            orphans = 0
            lst_ec = list()
            lst_unique_rxns = list()
            for r_id in lst_reactions[1]:
                if r_id in reactions_info:
                    ec_lst = reactions_info[r_id][ec_position][1]
                    for ec in ec_lst:
                        lst_ec.append(ec)
                    if reactions_info[r_id][spon_position][1]:
                        spontaneous += 1
                    if reactions_info[r_id][orphn_position][1]:
                        orphans += 1
                    if len(reactions_info[r_id][inptw_position][1]) == 1:
                        if p_id in reactions_info[r_id][inptw_position][1]:
                            lst_unique_rxns.append(r_id)

            # datum is comprised of {UNIQUE-ID: (COMMON-NAME, TYPES, PATHWAY-LINKS, PREDECESSORS,
            # REACTION-LIST, REACTION-LAYOUT, SPECIES, TAXONOMIC-RANGE, SUPER-PATHWAYS, SUB-PATHWAYS,
            # GENES-NAME, GENES-ID, ORPHANS, SPONTANEOUS, EC, UNIQUE REACTIONS)}
            datum = {p_id: (pathway_name,
                            lst_pathway_types,
                            lst_pathways_links,
                            lst_predecessors,
                            lst_reactions,
                            lst_reactions_layout,
                            lst_synonyms,
                            lst_species,
                            lst_taxonomic_range,
                            lst_sup_pathways,
                            lst_sub_pathways,
                            lst_genes_names,
                            lst_genes_ids,
                            ['ORPHANS', orphans],
                            ['SPONTANEOUS', spontaneous],
                            ['EC', lst_ec],
                            ['UNIQUE REACTIONS', lst_unique_rxns])}
            self.pathway_info.update(datum)
