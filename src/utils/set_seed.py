import random
import os
import numpy as np


def seed_everything(seed: int = 42):
    '''
    결과의 재현성을 위해 seed 값을 설정하는 함수
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
