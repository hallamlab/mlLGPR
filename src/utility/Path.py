import os.path

DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-2])

LOG_PATH = os.path.join(DIRECTORY_PATH, 'Log')
DATABASE_PATH = os.path.join(REPO_PATH, 'MetaPaIn', 'database/biocyc-flatfiles')
OBJECT_PATH = os.path.join(REPO_PATH, 'MetaPaIn', 'objectset')
DATASET_PATH = os.path.join(REPO_PATH, 'MetaPaIn', 'dataset')
INPUT_PATH = os.path.join(REPO_PATH, 'MetaPaIn', 'inputset')
RESULT_PATH = os.path.join(REPO_PATH, 'MetaPaIn', 'result')
MODEL_PATH = os.path.join(REPO_PATH, 'MetaPaIn', 'model')
