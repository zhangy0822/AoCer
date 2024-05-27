import sys
sys.path.append('./')
import random
import numpy as np

import torch

from train.trainer_txt_img_matching import TextImageMatchingTrainer
from utils.utils import *
from utils.config import get_test_config
from utils.vocab import Vocabulary
from datasets.vg import vg

def test_model(config):
    test_db = vg(config, 'test')
    if not config.use_ensemble:
        trainer = TextImageMatchingTrainer(config)
        trainer.test(test_db)
    else:
        trainer1 = TextImageMatchingTrainer(config)
        result1 = trainer1.test(test_db)
        data_len1 = result1[1]
        config.BFAN = 'prob'
        trainer2 = TextImageMatchingTrainer(config)
        result2 = trainer2.test(test_db)
        data_len2 = result2[1]
        assert data_len2 == data_len1, 'Different data lengths'
        trainer2.evaluate_ensemble((result1[0] + result2[0])/2, data_len2)


if __name__ == '__main__':
    config, unparsed = get_test_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    test_model(config)