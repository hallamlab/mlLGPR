'''
This file preprocesses proteins.dat file from BioCyc PGDBs
to an appropriate format that is could be used as inputs
to designated machine learning models.
'''

import os
import os.path
from collections import OrderedDict


class Protein(object):
    def __init__(self, fname_protein='proteins.dat'):
        """ Initialization
        :type fname_reaction: str
        :param fname_reaction: file name for the proteins
        """
        self.fname_protein = fname_protein
        self.protein_info = OrderedDict()

    def ProcessProteins(self, pr_id, go_id, lst_ids, data_path):
        protein_file = os.path.join(data_path, self.fname_protein)
        if os.path.isfile(protein_file):
            print('\t\t\t--> Prepossessing proteins database from: {0}'.format(protein_file.split(os.sep)[-1]))
            with open(protein_file, errors='ignore') as f:
                for text in f:
                    if not str(text).startswith('#'):
                        ls = text.strip().split()
                        if ls:
                            if ls[0] == 'UNIQUE-ID':
                                protein_id = ' '.join(ls[2:])
                                protein_name = ''
                                species = ''
                                lst_protein_types = list()
                                lst_catalysts = list()
                                lst_gene = list()
                                lst_go_terms = list()
                            elif ls[0] == 'TYPES':
                                lst_protein_types.append(' '.join(ls[2:]))
                            elif ls[0] == 'COMMON-NAME':
                                protein_name = ' '.join(ls[2:])
                            elif ls[0] == 'CATALYZES':
                                lst_catalysts.append(' '.join(ls[2:]))
                            elif ls[0] == 'GENE':
                                lst_gene.append(' '.join(ls[2:]))
                            elif ls[0] == 'GO-TERMS':
                                go = ''.join(ls[2].split('|'))
                                lst_go_terms.append(go)
                                if go not in lst_ids[go_id]:
                                    lst_ids[go_id].update({go: len(lst_ids[go_id])})
                            elif ls[0] == 'SPECIES':
                                species = ' '.join(ls[2:])
                            elif ls[0] == '//':
                                if protein_id not in lst_ids[pr_id]:
                                    lst_ids[pr_id].update({protein_id: len(lst_ids[pr_id])})
                                if protein_id not in self.protein_info:
                                    # datum is comprised of {UNIQUE-ID: (TYPES, COMMON-NAME, CATALYZES, GENE, GO-TERMS, SPECIES)}
                                    datum = {protein_id: (['TYPES', lst_protein_types],
                                                          ['COMMON-NAME', protein_name],
                                                          ['CATALYZES', lst_catalysts],
                                                          ['GENE', lst_gene],
                                                          ['GO-TERMS', lst_go_terms],
                                                          ['SPECIES', species])}
                                    self.protein_info.update(datum)
