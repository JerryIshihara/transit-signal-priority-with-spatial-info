import os
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-env", "--env", help="DQN Environment")
args = parser.parse_args()

from model import CNN, DQN
from environment import CartPoleEnv
from environment import AimsunEnv

import torch
import torch.nn.functional as F
import torch.optim as optim

CONFIG = {
    'REPLAY': 100000,
    'BATCH_SIZE': 256,
    'GAMMA': 0.999,
    'EPS_START': 0.9,
    'EPS_END': 0.05,
    'EPS_DECAY': 200,
    'TARGET_UPDATE': 10,
    'DEVICE': 'cpu', # 'cuda:0'
    'lr': 1e-3,
}

AIMSUN_CONFIG = {
    'ACTION_LOG': '',
    'REWARD_LOG': '',
    'ACTION_SPACE': '' # TODO: Change action space'
    'HISTORY': '' # number of channerls
}

if __name__ == "__main__":
    loss = F.smooth_l1_loss
    optimizer = optim.Adam
    
    if args.env.upper() == 'CARTPOLE':
        ENV = CartPoleEnv()
        ENV.reset()
        n_actions = ENV.action_space
        CONFIG['ACTION_SPACE'] = n_actions
        init_screen = ENV.get_screen()
        _, _, screen_height, screen_width = init_screen.shape
        cnn = CNN(screen_height, screen_width, n_actions)
        DQN = DQN(CONFIG, cnn, loss, optimizer)
        ENV.train_cartpole(DQN, episode=200)
        
    if args.env.upper() == 'AIMSUN':
        os._exit(0)
        
        ENV = AimsunEnv(AIMSUN_CONFIG)
        n_actions = ENV.action_space
        CONFIG['ACTION_SPACE'] = n_actions
        
        # TODO: fill the variables
        channels, screen_height, screen_width, n_actions = None, None, None, None
        start_rep, end_rep = None, None
        
        cnn = CNN(screen_height, screen_width, n_actions, channels)
        DQN = DQN(CONFIG, cnn, loss, optimizer)
        ENV.train_aimsun(DQN, start_rep, end_rep)
        
