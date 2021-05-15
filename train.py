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
    'REPLAY':        100000,
    'BATCH_SIZE':    256,
    'GAMMA':         0.999,
    'EPS_START':     0.9,
    'EPS_END':       0.05,
    'EPS_DECAY':     200,
    'TARGET_UPDATE': 10,
    'DEVICE':        'cpu', # 'cuda:0'
    'lr':            1e-3,
}

AIMSUN_CONFIG = {
    'IPC_ADDR':      'localhost',
    'IPC_PORT':      23000,
    'INSTANCE_PATH': 'C:\\Program Files\\Aimsun\\Aimsun Next 8.3\\Aimsun Next.exe',
    'ANG_PATH'     : 'C:\\Users\\siwei\\Documents\\Developer\\aimsun\\finchTSPs_3 intx_west_Subnetwork 1171379.ang',
    'ACTION_LOG':    'C:\\Users\\siwei\\Documents\\Developer\\transit-signal-priority-with-spatial-info\\environment\\action.npy',
    'REWARD_LOG':    '',
    'ACTION_SPACE':  '', # TODO: Change action space'
    'HISTORY':       '' # number of channels
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
        # os._exit(0)
        
        ENV = AimsunEnv(AIMSUN_CONFIG)
        n_actions = ENV.action_space
        CONFIG['ACTION_SPACE'] = n_actions
        
        # TODO: fill the variables
        channels, screen_height, screen_width, n_actions = 8, 8, 179, 1
        start_rep, end_rep = 1180681, 1180701
        
        cnn = CNN(screen_height, screen_width, n_actions, channels)
        DQN = DQN(CONFIG, cnn, loss, optimizer)
        ENV.train_aimsun(DQN, start_rep, end_rep)
        
