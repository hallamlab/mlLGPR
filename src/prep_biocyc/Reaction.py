'''
This file preprocesses two reaction files: reactions.dat and
reaction-links.dat from BioCyc PGDBs to an appropriate format
that is could be used as inputs to designated machine learning
models.
'''

import os
import os.path
from collections import OrderedDict


class Reaction(object):
    def __init__(self, fname_reaction='reactions.dat', fname_ec_reaction='reaction-links.dat'):
        """ Initialization
        :type fname_reaction: str
        :param fname_reaction: file name for the reaction
        """
        self.reaction_fname = fname_reaction
        self.ec_reaction_fname = fname_ec_reaction
        self.reaction_info = OrderedDict()

    def ProcessReactions(self, r_id, lst_ids, data_path):
        """ process reactions data
        :type data_path: str
        :param data_path: file name for the reaction
        """
        reaction_file = os.path.join(data_path, self.reaction_fname)
        if os.path.isfile(reaction_file):
            print('\t\t\t--> Prepossessing reactions database from: {0}'.format(reaction_file.split('/')[-1]))
            with open(reaction_file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split()
                        if ls:
                            if ls[0] == 'UNIQUE-ID':
                                reaction_id = ' '.join(ls[2:])
                                reaction_name = ''
                                reaction_direction = ''
                                lst_reaction_types = list()
                                lst_reaction = list()
                                lst_enzymatic_reactions = list()
                                lst_ec = list()
                                lst_species = list()
                                spontaneous = False
                                lst_synonyms = list()
                                systematic_name = ''
                                lst_taxonomic_range = list()
                                lst_pathway = list()
                                orphan = False
                            elif ls[0] == 'COMMON-NAME':
                                reaction_name = ' '.join(ls[2:])
                            elif ls[0] == 'TYPES':
                                lst_reaction_types.append(' '.join(ls[2:]))
                            elif ls[0] == 'ENZYMATIC-REACTION':
                                lst_enzymatic_reactions.append(' '.join(ls[2:]))
                            elif ls[0] == 'EC-NUMBER':
                                ec = ' '.join(ls[2:])
                                ec = ''.join(ec.split('|'))
                                lst_ec.append(ec)
                            elif ls[0] == 'IN-PATHWAY':
                                lst_pathway.append(' '.join(ls[2:]))
                            elif ls[0] == 'ORPHAN?':
                                if ls[2].split(':')[1].lower() != 'no':
                                    orphan = True
                            elif ls[0] == 'REACTION-LIST':
                                lst_reaction.append(' '.join(ls[2:]))
                            elif ls[0] == 'REACTION-DIRECTION':
                                reaction_direction = ' '.join(ls[2:])
                            elif ls[0] == 'SPECIES':
                                lst_species.append(' '.join(ls[2:]))
                            elif ls[0] == 'TAXONOMIC-RANGE':
                                lst_taxonomic_range.append(' '.join(ls[2:]))
                            elif ls[0] == 'SYNONYMS':
                                lst_synonyms.append(' '.join(ls[2:]))
                            elif ls[0] == 'SYSTEMATIC-NAME':
                                systematic_name = ' '.join(ls[2:])
                            elif ls[0] == 'SPONTANEOUS?':
                                if ' '.join(ls[2:]).lower() == 't':
                                    spontaneous = True
                            elif ls[0] == '//':
                                if reaction_id not in lst_ids[r_id]:
                                    lst_ids[r_id].update({reaction_id: len(lst_ids[r_id])})
                                if reaction_id not in self.reaction_info:
                                    # datum is comprised of {UNIQUE-ID: (COMMON-NAME, TYPES, ENZYMATIC-REACTION,
                                    # EC-NUMBER, IN-PATHWAY, ORPHAN?, REACTION-LIST, REACTION-DIRECTION, SPECIES,
                                    # TAXONOMIC-RANGE, SYNONYMS, SYSTEMATIC-NAME, SPONTANEOUS?, GENES-NAME)}
                                    datum = {reaction_id: (['COMMON-NAME', reaction_name],
                                                           ['TYPES', lst_reaction_types],
                                                           ['ENZYMATIC-REACTION', lst_enzymatic_reactions],
                                                           ['EC-NUMBER', lst_ec],
                                                           ['IN-PATHWAY', lst_pathway],
                                                           ['ORPHAN', orphan],
                                                           ['REACTION-LIST', lst_reaction],
                                                           ['REACTION-DIRECTION', reaction_direction],
                                                           ['SPECIES', lst_species],
                                                           ['TAXONOMIC-RANGE', lst_taxonomic_range],
                                                           ['SYNONYMS', lst_synonyms],
                                                           ['SYSTEMATIC-NAME', systematic_name],
                                                           ['SPONTANEOUS', spontaneous],
                                                           ['GENES-NAME', []])}
                                    self.reaction_info.update(datum)

    def AddEC2Reactions(self, ec_id, lst_ids, data_path):
        """ process reactions data
        :type data_path: str
        :param data_path: file name for the reaction
        """
        ec_reaction_file = os.path.join(data_path, self.ec_reaction_fname)
        reaction_ec = OrderedDict()

        if os.path.isfile(ec_reaction_file):
            print('\t\t\t--> Adding ec number to reactions from: {0}'.format(ec_reaction_file.split('/')[-1]))
            with open(ec_reaction_file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split('\t')
                        if ls:
                            reaction_ec.update({ls[0]: ls[1:]})
                            for ec in ls[1:]:
                                ec = ''.join(ec.split('|'))
                                if ec not in lst_ids[ec_id]:
                                    lst_ids[ec_id].update({ec: len(lst_ids[ec_id])})

        for (r_id, r_item) in self.reaction_info.items():
            # r_item is comprised of (COMMON-NAME, TYPES, ENZYMATIC-REACTION, EC-NUMBER, IN-PATHWAY, ORPHAN?, REACTION-LIST,
            # REACTION-DIRECTION, SPECIES, TAXONOMIC-RANGE, SYNONYMS, SYSTEMATIC-NAME, SPONTANEOUS?, GENES-NAME)
            (reaction_name, lst_reaction_types, lst_enzymatic_reactions, lst_ec, lst_pathway, orphan, lst_reaction,
             reaction_direction, lst_species, lst_taxonomic_range, lst_synonyms, systematic_name, spontaneous,
             lst_gene_name_id) = r_item
            ec_lst = list()
            for ec in lst_ec[1]:
                ec = ''.join(ec.split('|'))
                ec_lst.append(ec)
                if ec not in lst_ids[ec_id]:
                    lst_ids[ec_id].update({ec: len(lst_ids[ec_id])})
            lst_ec[1] = ec_lst

            if r_id in reaction_ec:
                ec_lst = reaction_ec[r_id]
                if ec_lst:
                    for ec in ec_lst:
                        if ec not in lst_ec[1]:
                            lst_ec[1].append(ec)
            # datum is comprised of {UNIQUE-ID: (COMMON-NAME, TYPES, ENZYMATIC-REACTION,
            # EC-NUMBER, IN-PATHWAY, ORPHAN?, REACTION-LIST, REACTION-DIRECTION, SPECIES,
            # TAXONOMIC-RANGE, SYNONYMS, SYSTEMATIC-NAME, SPONTANEOUS?, GENES-NAME)}
            datum = {r_id: (reaction_name,
                            lst_reaction_types,
                            lst_enzymatic_reactions,
                            lst_ec,
                            lst_pathway,
                            orphan,
                            lst_reaction,
                            reaction_direction,
                            lst_species,
                            lst_taxonomic_range,
                            lst_synonyms,
                            systematic_name,
                            spontaneous,
                            lst_gene_name_id)}
            self.reaction_info.update(datum)

    def AddGenes2Reactions(self, genes_info, gene_name_id_position, reaction_position):
        print('\t\t\t--> Adding genes names id to reactions')
        tmp_enzrxns_genes = OrderedDict()
        for (g_id, g_item) in genes_info.items():
            lst_reaction = g_item[reaction_position][1]
            if lst_reaction:
                for r in lst_reaction:
                    if r not in tmp_enzrxns_genes:
                        tmp_enzrxns_genes.update({r: [g_item[gene_name_id_position][1]]})
                    elif r in tmp_enzrxns_genes:
                        if g_item[gene_name_id_position][1] not in tmp_enzrxns_genes[r]:
                            tmp_enzrxns_genes[r].extend([g_item[gene_name_id_position][1]])

        for (r_id, r_item) in self.reaction_info.items():
            # r_item is comprised of (COMMON-NAME, TYPES, ENZYMATIC-REACTION, EC-NUMBER, IN-PATHWAY, ORPHAN?, REACTION-LIST,
            # REACTION-DIRECTION, SPECIES, TAXONOMIC-RANGE, SYNONYMS, SYSTEMATIC-NAME, SPONTANEOUS?, GENES-NAME)
            (reaction_name, lst_reaction_types, lst_enzymatic_reactions, lst_ec, lst_pathway, orphan, lst_reaction,
             reaction_direction, lst_species, lst_taxonomic_range, lst_synonyms, systematic_name, spontaneous,
             lst_gene_name_id) = r_item
            if lst_enzymatic_reactions[1]:
                for r in lst_enzymatic_reactions[1]:
                    if r in tmp_enzrxns_genes:
                        for g in tmp_enzrxns_genes[r]:
                            if g == " ":
                                continue
                            if g not in lst_gene_name_id[1]:
                                lst_gene_name_id[1].extend([g])

            # datum is comprised of {UNIQUE-ID: (COMMON-NAME, TYPES, ENZYMATIC-REACTION,
            # EC-NUMBER, IN-PATHWAY, ORPHAN?, REACTION-LIST, REACTION-DIRECTION, SPECIES,
            # TAXONOMIC-RANGE, SYNONYMS, SYSTEMATIC-NAME, SPONTANEOUS?, GENES-NAME)}
            datum = {r_id: (reaction_name,
                            lst_reaction_types,
                            lst_enzymatic_reactions,
                            lst_ec,
                            lst_pathway,
                            orphan,
                            lst_reaction,
                            reaction_direction,
                            lst_species,
                            lst_taxonomic_range,
                            lst_synonyms,
                            systematic_name,
                            spontaneous,
                            lst_gene_name_id)}
            self.reaction_info.update(datum)
