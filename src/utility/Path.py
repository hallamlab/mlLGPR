import os.path

DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-2])

LOG_PATH = os.path.join(DIRECTORY_PATH, 'Log')

FUNCTIONAL_PATH = os.path.join(REPO_PATH, 'database', 'functional')
METACYC_PATH = os.path.join(FUNCTIONAL_PATH, 'metacyc-v5-2011-10-21')

DATABASE_PATH = os.path.join(REPO_PATH, 'database', 'biocyc-flatfiles')
OBJECT_PATH = os.path.join(REPO_PATH, 'objectset')
DATASET_PATH = os.path.join(REPO_PATH, 'dataset')
INPUT_PATH = os.path.join(REPO_PATH, 'inputset')
RESULT_PATH = os.path.join(REPO_PATH, 'result')
MODEL_PATH = os.path.join(REPO_PATH, 'model')
