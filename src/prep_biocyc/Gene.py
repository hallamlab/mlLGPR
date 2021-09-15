'''
This file preprocesses two gene files: genes.dat and genes.col
from BioCyc PGDBs to an appropriate format that is could be used
as inputs to designated machine learning models.
'''

import os
import os.path
from collections import OrderedDict


class Gene(object):
    def __init__(self, fname_gene_col='genes.col', fname_gene_dat='genes.dat'):
        """ Initialization
        :type fname_gene_col: str
        :param fname_gene_col: file name for the genes.col, containing id for enzymatic-reactions
        :type fname_gene_dat: str
        :param fname_gene_dat: file name for the genes.dat, containing id for enzymatic-reactions
        """
        self.gene_col_fname = fname_gene_col
        self.gene_dat_fname = fname_gene_dat
        self.gene_info = OrderedDict()

    def ProcessGenesDat(self, g_id, g_name_id, lst_ids, data_path):
        gene_file = os.path.join(data_path, self.gene_dat_fname)
        if os.path.isfile(gene_file):
            print('\t\t\t--> Prepossessing genes database from: '
                  '{0}'.format(gene_file.split(os.sep)[-1]))
            with open(gene_file, errors='ignore') as f:
                for text in f:
                    if not str(text).strip().startswith('#'):
                        ls = text.strip().split()
                        if ls:
                            if ls[0] == 'UNIQUE-ID':
                                gene_id = ' '.join(ls[2:])
                                lst_gene_types = list()
                                gene_name_id = ''
                                gene_product = ''
                            elif ls[0] == 'TYPES':
                                lst_gene_types.append(' '.join(ls[2:]))
                            elif ls[0] == 'COMMON-NAME':
                                gene_name_id = ''.join(ls[2:])
                                if gene_name_id:
                                    if gene_name_id not in lst_ids[g_name_id]:
                                        lst_ids[g_name_id].update({gene_name_id: len(lst_ids[g_name_id])})
                            elif ls[0] == 'PRODUCT':
                                gene_product = ''.join(ls[2].split('|'))
                            elif ls[0] == '//':
                                if gene_id not in lst_ids[g_id]:
                                    lst_ids[g_id].update({gene_id: len(lst_ids[g_id])})

                                if gene_id not in self.gene_info:
                                    # datum is comprised of {UNIQUE-ID: (TYPES, NAME, PRODUCT, PRODUCT-NAME, SWISS-PROT-ID,
                                    # REACTION-LIST, GO-TERMS)}
                                    datum = {gene_id: (['TYPES', lst_gene_types],
                                                       ['NAME', gene_name_id],
                                                       ['PRODUCT', gene_product],
                                                       ['PRODUCT-NAME', ''],
                                                       ['SWISS-PROT-ID', ''],
                                                       ['REACTION-LIST', []],
                                                       ['GO-TERMS', []])}
                                    self.gene_info.update(datum)

    def ProcessGenesCol(self, g_id, g_name_id, pd_id, lst_ids, data_path, header=False):
        gene_file = os.path.join(data_path, self.gene_col_fname)
        if os.path.isfile(gene_file):
            print('\t\t\t--> Prepossessing genes database from: {0}'
                  .format(gene_file.split(os.sep)[-1]))
            with open(gene_file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.split('\t')
                        if ls:
                            if not header:
                                if ls[0] == 'UNIQUE-ID':
                                    header = True
                                    gene_id = ls.index('UNIQUE-ID')
                                    gene_name_id = ls.index('NAME')
                                    gene_product_name = ls.index('PRODUCT-NAME')
                                    gene_swiss_prot_id = ls.index('SWISS-PROT-ID')
                            else:
                                if ls[gene_id] not in lst_ids[g_id]:
                                    lst_ids[g_id].update({ls[gene_id]: len(lst_ids[g_id])})

                                if ls[gene_name_id]:
                                    if ls[gene_name_id] not in lst_ids[g_name_id]:
                                        lst_ids[g_name_id].update({ls[gene_name_id]: len(lst_ids[g_name_id])})

                                if ls[gene_product_name]:
                                    if ls[gene_product_name] not in lst_ids[pd_id]:
                                        lst_ids[pd_id].update({ls[gene_product_name]: len(lst_ids[pd_id])})

                                if ls[gene_id] not in self.gene_info:
                                    # datum is comprised of {UNIQUE-ID: (TYPES, NAME, PRODUCT, PRODUCT-NAME, SWISS-PROT-ID,
                                    # REACTION-LIST, GO-TERMS)}
                                    datum = {ls[gene_id]: (['TYPES', []],
                                                           ['NAME', ls[gene_name_id]],
                                                           ['PRODUCT', ''],
                                                           ['PRODUCT-NAME', ls[gene_product_name]],
                                                           ['SWISS-PROT-ID', ls[gene_swiss_prot_id]],
                                                           ['REACTION-LIST', []],
                                                           ['GO-TERMS', []])}
                                    self.gene_info.update(datum)
                                else:
                                    (t_gene_types, t_gene_name_id, t_gene_product, t_gene_product_name,
                                     t_gene_swiss_prot_id, lst_catalyzes, lst_go) = self.gene_info[ls[gene_id]]
                                    if not t_gene_name_id[1]:
                                        t_gene_name_id[1] = ls[gene_name_id]
                                    # datum is comprised of {UNIQUE-ID: (TYPES, NAME, PRODUCT, PRODUCT-NAME, SWISS-PROT-ID,
                                    # REACTION-LIST, GO-TERMS)}
                                    datum = {ls[gene_id]: (t_gene_types,
                                                           t_gene_name_id,
                                                           t_gene_product,
                                                           ['PRODUCT-NAME', ls[gene_product_name]],
                                                           ['SWISS-PROT-ID', ls[gene_swiss_prot_id]],
                                                           lst_catalyzes,
                                                           lst_go)}
                                    self.gene_info.update(datum)

    def AddProteinInfo2GeneInfo(self, protein_info, go_position, catalyzes_position, product_name_position, pd_id,
                                lst_ids):
        print('\t\t\t--> Adding proteins information to genes')
        for (gid, g_item) in self.gene_info.items():
            (t_gene_types, t_gene_name_id, t_gene_product, t_gene_product_name,
             t_gene_swiss_prot_id, lst_catalyzes, lst_go) = g_item
            if t_gene_product[1] in protein_info:
                if protein_info[t_gene_product[1]][catalyzes_position][1]:
                    lst_catalyzes[1].extend(protein_info[t_gene_product[1]][catalyzes_position][1])
                if protein_info[t_gene_product[1]][go_position][1]:
                    lst_go[1].extend(protein_info[t_gene_product[1]][go_position][1])
                if not t_gene_product_name[1]:
                    product_name = protein_info[t_gene_product[1]][product_name_position][1]
                    if product_name not in lst_ids[pd_id] and product_name != '':
                        lst_ids[pd_id].update({product_name: len(lst_ids[pd_id])})
                else:
                    product_name = t_gene_product_name[1]
                datum = {gid: (t_gene_types,
                               t_gene_name_id,
                               t_gene_product,
                               ['PRODUCT-NAME', product_name],
                               t_gene_swiss_prot_id,
                               lst_catalyzes,
                               lst_go)}
                self.gene_info.update(datum)

    def AddPathwayGenes2GenesID(self, pathway_info, gene_id_position, gene_name_id_position, g_id, g_name_id, lst_ids):
        print('\t\t\t--> Adding additional genes to gene id and gene name id from pathways genes')
        for (p_id, p_item) in pathway_info.items():
            if p_item[gene_id_position][1]:
                for g in p_item[gene_id_position][1]:
                    if g not in lst_ids[g_id]:
                        lst_ids[g_id].update({g: len(lst_ids[g_id])})
                        self.gene_info.update({g: (['TYPES', []],
                                                   ['NAME', ''],
                                                   ['PRODUCT', ''],
                                                   ['PRODUCT-NAME', ''],
                                                   ['SWISS-PROT-ID', ''],
                                                   ['REACTION-LIST', []],
                                                   ['GO-TERMS', []])})
                for g in p_item[gene_name_id_position][1]:
                    if g not in lst_ids[g_name_id]:
                        lst_ids[g_name_id].update({g: len(lst_ids[g_name_id])})

    def AddReactionGenes2GenesID(self, reaction_info, gene_name_id_position, g_name_id, lst_ids):
        print('\t\t\t--> Adding additional genes to gene name id from reactions genes')
        for (r_id, r_item) in reaction_info.items():
            if r_item[gene_name_id_position][1]:
                for g in r_item[gene_name_id_position][1]:
                    if g not in lst_ids[g_name_id]:
                        lst_ids[g_name_id].update({g: len(lst_ids[g_name_id])})
