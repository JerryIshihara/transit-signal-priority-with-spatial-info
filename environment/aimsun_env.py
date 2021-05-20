"""AimsunEnv
"""
from uuid import uuid4
import numpy as np
import csv
# from config import *
import os
import sys
import inspect
import socket
from subprocess import Popen, PIPE

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


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
        self.IPC_ADDR      = config['IPC_ADDR']
        self.IPC_PORT      = config['IPC_PORT']
        self.INSTANCE_PATH = config['INSTANCE_PATH']
        self.ANG_PATH      = config['ANG_PATH']
        self.ACTION_LOG    = config['ACTION_LOG']
        self.REWARD_LOG    = config['REWARD_LOG']
        self.action_space  = config['ACTION_SPACE']
        self.reward_flag   = 0
        self.state_flag    = 0
        self.num_step      = 0
        self.num_channels  = config['HISTORY']

    def _get_reward(self, conn):
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
                if len(data) != REWARD_INPUT_LEN:
                    continue
                reward, new_flag = float(data[0]), int(data[1])
                if new_flag != self.reward_flag:
                    is_read = True
                    self.reward_flag = new_flag
            except:
                continue
        return reward

    def _write_action(self, conn, index):
        """write the newly received action to Aimsun

        Parameters
        ----------
        index : int
            the index of the new action
        """
        is_written = False
        while not is_written:
            try:
                with open(self.ACTION_LOG, "w+") as f:
                    f.write("{} {}".format(index, uuid4().int))
                    is_written = True
                conn.send(b'WRITE_ACTION')
            except:
                continue

    def _get_state(self, conn, replication):
        """Read the state of the replication.
        - Get DQN number of input channels: self.num_channels
        the returned state size is (num_channels, height, width)
        - wait the npy file until it has enough input channel and updated

        Returns
        -------
        numpy array with size (num_channels, height, width)
        """
        conn.send(b"GET_STATE")

        data = conn.recv(1024).decode("utf-8")
        if(len(data) < 10 or data[:10] != 'DATA_READY'):
            print("ERROR")
        else:
            time = data[10:]
            feature = np.load('realtime_state.npy')
            print("Time: " + time)
        pass
        # while True:
        #     break
        # return S_

    def step(self, conn, action_index):
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
        self._write_action(conn, action_index)
        reward = self._get_reward(conn)
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
        if len(self.check_in_time) > 10:
            self.check_in_time.pop(0)
        if len(self.check_in_time) <= 2:
            return True
        if self.check_in_time[-1] < self.check_in_time[-2]:
            return True
        return False

    def start_aimsun_instance(self, s, rep):
        process = Popen(['"aconsole.exe"', '-v', '-log', 
                    '-project', self.ANG_PATH,
                    '-cmd', 'execute', '-target', str(rep)], executable=self.INSTANCE_PATH)

        print("Waiting for aimsun instance to connect...")
        conn, addr = s.accept()
        print('[Info] Connected by', addr)
        conn.send(b'SYN')
        data = conn.recv(1024).decode("utf-8")
        if data != "SYN":
            print("[ERROR] Handshake Failed.")
            return False, -1 
        else:
            print("[Info] Aimsun Instance connected.")
            return True, conn

    def train_aimsun(self, DQN, start_rep, end_rep):
        # create a socket for IPC
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.IPC_ADDR, self.IPC_PORT))
            s.listen(10)
            print('[Info] Aimsun Manager Ready. Waiting For Aimsun Instance...')

            # num_episodes = episode
            for i_episode in range(start_rep, end_rep + 1):
                # Initialize the environment and state
                status, conn = self.start_aimsun_instance(s, i_episode)
                if not status:
                    continue

                self.reset()
                state = self._get_state(conn, i_episode)

                for t in count():
                    # Select and perform an action
                    action = DQN.select_action(state)
                    reward, done = self.step(conn, action.item())
                    reward = torch.tensor([reward], device=DQN.device)

                    # Observe new state
                    # last_screen = current_screen
                    # current_screen = self.get_screen()
                    if not done:
                        next_state = self._get_state(conn, i_episode)
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
