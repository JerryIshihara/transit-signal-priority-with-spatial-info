"""AimsunEnv
"""
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from config import *
import csv
import numpy as np
from uuid import uuid4


REWARD_INPUT_LEN = 2
STATE_INPUT_LEN = 17

class AimsunEnv:
    """Aimsun Next environment
    
    Attributes
    ----------
    num_step : int
        total time steps simulated
    reward_flag : int
        check if received reward is at the new time step
    state_flag : int
        check if received state is at the new time step
    """
    
    def __init__(self, config):
        """Initialize Aimsun Next environment object
        """
        self.ACTION_LOG = config['ACTION_LOG']
        self.REWARD_LOG = config['REWARD_LOG']
        self.action_space = config['ACTION_SPACE']
        self.reward_flag = 0
        self.state_flag = 0
        self.num_step = 0
        self.num_channels = config['HISTORY']
        
    def _get_reward(self):
        """Receive, log and return the new reward
        
        Returns
        -------
        float
            newly received reward
        """
        # receive from REWARD_LOG
        is_read = False
        while not is_read:
            try:
                f = open(self.REWARD_LOG, "r")
                data = f.read()
                f.close()
                data = data.split()
                if len(data) != REWARD_INPUT_LEN: continue
                reward, new_flag = float(data[0]), int(data[1])
                if new_flag != self.reward_flag:
                    is_read = True
                    self.reward_flag = new_flag
            except:
                continue
        return reward

    def _write_action(self, index):
        """write the newly received action to Aimsun
        
        Parameters
        ----------
        index : int
            the index of the new action
        """
        is_written = False
        while not is_written:
            try:
                f = open(self.ACTION_LOG, "w+")
                f.write("{} {}".format(index, uuid4().int))
                f.close()
                is_written = True
            except:
                continue

    def _get_state(self, replication):
        """Read the state of the replication.
        - Get DQN number of input channels: self.num_channels
        the returned state size is (num_channels, height, width)
        - wait the npy file until it has enough input channel and updated
        
        Returns
        -------
        numpy array with size (num_channels, height, width)
        """
        pass
        # while True:
        #     break
        # return S_

    def step(self, action_index):
        """Apply the write the action to Aimsun and wait for the new
        state and reward
        
        Parameters
        ----------
        action_index : int
            the index of the action space
        
        Returns
        -------
        list, float, bool
            new state, new reward, and simulation finish
        """
        done = None
        self._write_action(action_index)
        reward = self._get_reward()
        # print log
        return reward, done

    # def reset(self):
    #     """Reset the Aimsun environment and receive the first state
    #     """
    #     print('Reset Aimsun Environment')
    #     return self._get_state()

    def exclude(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        if len(self.check_in_time) > 10: self.check_in_time.pop(0)
        if len(self.check_in_time) <= 2: return True
        if self.check_in_time[-1] < self.check_in_time[-2]:
            return True
        return False
    
    def train_aimsun(self, DQN, start_rep, end_rep):
        num_episodes = episode
        for i_episode in range(start_rep, end_rep + 1):
            # Initialize the environment and state
            self.reset()
            state = self._get_state(i_episode)
            for t in count():
                # Select and perform an action
                action = DQN.select_action(state)
                reward, done = self.step(action.item())
                reward = torch.tensor([reward], device=DQN.device)

                # Observe new state
                # last_screen = current_screen
                # current_screen = self.get_screen()
                if not done:
                    next_state = self._get_state(i_episode)
                else:
                    next_state = None

                # Store the transition in memory
                DQN.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                DQN.optimize_model()
                if done:
                    break
            # Update the target network, copying all weights and biases in DQN
            TARGET_UPDATE = DQN.config['TARGET_UPDATE']
            if i_episode % TARGET_UPDATE == 0:
                DQN.update()

        print('Complete')
        self.close()