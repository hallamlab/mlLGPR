'''
This file parse features that is stored in FeaturesList.txt file.
'''

import os.path
import sys
import traceback


def ExtractFeaturesNames(path=os.getcwd(), fname='FeaturesList.txt', printFeats=False, tag='a list of features name'):
    try:
        print('\t\t## Loading {0:s} from: {1:s}'.format(tag, fname))
        file = os.path.join(path, fname)
        dictFeatures = dict()
        with open(file, 'r') as f_in:
            lst_features = list()
            for data in f_in:
                if not data.startswith('#'):
                    if printFeats:
                        print(data.strip())
                    if len(lst_features) != 0:
                        dictFeatures.update({featureName.split('.')[1].strip(): lst_features})
                    featureName = data.strip()
                    lst_features = list()
                elif data.startswith('#'):
                    feature = data.split('#')[1]
                    if printFeats:
                        print('\item \\textbf{' + feature.split('.')[1].strip() + '}')
                    lst_features.append(feature.split('.')[1].split(' ')[1].strip())
        dictFeatures.update({featureName.split('.')[1].strip(): lst_features})
        return dictFeatures
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(file), file=sys.stderr)
        print(traceback.print_exc())
        raise e


if __name__ == '__main__':
    os.system('clear')
    print(__doc__)
    dictFeatures = ExtractFeaturesNames()
    print(dictFeatures)
