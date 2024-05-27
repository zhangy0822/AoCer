import sys
sys.path.append('./')
import numpy as np
import torch

from train.trainer_ppo_coherence import PPOTrainer
from utils.config import get_test_config
from utils.vocab import Vocabulary
result = []

def test_model(config):
    trainer = PPOTrainer(config, 'test')
    return trainer.test()

if __name__ == '__main__':
    config, unparsed = get_test_config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.seed)
    action, query = [10,5,3], [1,2,4]


    for q in query:
        for a in action:
            config.test_turns, config.ppo_num_actions = q, a
            re = test_model(config)
            title = "q" + str(q) + " " + "a" + str(a) + "\n"
            with open('metrices_record_gcn_2_likelihood.txt', 'a+') as w:
                w.write("\n")
                w.write(title)
                w.write(re)
                