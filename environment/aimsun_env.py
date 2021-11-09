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
# from _thread import start_new_thread
import threading
import torch
import time

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

        self.rep_end       = False
        self.PATH          = config['DQN_MODEL']


    def reward_recv_thread(self, conn):
        data = conn.recv(1024).decode("utf-8")
        print("[DQN]RECV: " + str(data))
        if str(data) == "FIN":
            self.rep_end = True

        if not self.rep_end:
            self.reward = float(data[9:])

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
                conn.send(b'WRITE_ACTION:{}'.format("EXTEND" if index == 0 else "NOTHING"))
            except:
                continue

    def _get_state(self, conn):
        """Read the state of the replication.
        - Get DQN number of input channels: self.num_channels
        the returned state size is (num_channels, height, width)
        - wait the npy file until it has enough input channel and updated

        Returns
        -------
        numpy array with size (num_channels, height, width)
        """
        print('[DQN] Send GET_STATE')
        conn.send(b'GET_STATE')
        data = conn.recv(1024).decode("utf-8")
        if(data == 'FIN'):
            print("FIN Received")
            self.rep_end = True
        elif(data[:10] != 'DATA_READY'):
            print("ERROR")
            raise Exception
        # time = data[10:]
        state = np.load('environment/realtime_state.npy')
        return state
            # print(state.shape)
            # print("Time: " + time)

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
        # t = start_new_thread(self.reward_recv_thread, (conn,))

        t = threading.Thread(target=self.reward_recv_thread, args=(conn,))
        t.start()

        self._write_action(conn, action_index)
        print("[DQN]Wait for reward")
        t.join()

        print("[DQN]Get reward")
        # print log
        if not self.rep_end:
            return self.reward, done
        else:
            done = True
            self.rep_end = False
            return 0, done

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

    def train_aimsun(self, DQN, start_rep, end_rep, AIMSUNU_MODEL_PATH, ACONSOLE_PATH):
        # create a socket for IPC
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #     s.bind((self.IPC_ADDR, self.IPC_PORT))
        #     s.listen(10)
        #     print('[Info] Aimsun Manager Ready. Waiting For Aimsun Instance...')
        HOST = 'localhost'
        PORT = 23000
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(10)
        

        # get state
        # conn.send(b'GET_STATE')
        # data = conn.recv(1024).decode("utf-8")
        # if(data == 'FIN'):
        #     print("FIN Received")
        #     return
        # elif(data[:10] != 'DATA_READY'):
        #     print("ERROR")
        #     return
        # else:
        #     time = data[10:]
        #     state = np.load('realtime_state.npy')
        #     print(state.shape)
        #     print("Time: " + time)

        # num_episodes = episode
        for i_episode in range(start_rep, end_rep + 1):
            print('[Info] Aimsun Manager Ready. Waiting For Aimsun Instance...')
            process = Popen(['"aconsole.exe"', '-v', '-log',
                                '-project', AIMSUNU_MODEL_PATH,
                                '-cmd', 'execute', '-target', str(i_episode)], executable=ACONSOLE_PATH)
            conn, addr = s.accept()
            print('Connected by', addr)
            sync_message = 'SYN' + str(i_episode)
            conn.send(bytes(sync_message, 'utf-8'))
            data = conn.recv(1024).decode("utf-8")
            if data != "SYN":
                print("[ERROR] Handshake Failed.")
            else:
                print("[Info] Aimsun Instance connected.")

            # Initialize the environment and state
            state = self._get_state(conn)
            while True:
                # Select and perform an action
                action = DQN.select_action(torch.from_numpy(state[None, :]))
                # TODO: fetching reward
                reward, done = self.step(conn, action.item())
                reward = torch.tensor([reward], device=DQN.device)
                # Observe new state
                if not done:
                    next_state = self._get_state(conn)
                else:
                    next_state = None

                # Store the transition in memory
                DQN.memory.push(state, action, next_state, reward)
                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the target network)
                DQN.optimize_model()
                if done:
                    time.sleep(10)
                    break
            # Update the target network, copying all weights and biases in DQN
            TARGET_UPDATE = DQN.config['TARGET_UPDATE']
            if i_episode % TARGET_UPDATE == 0:
                DQN.update()
                torch.save(DQN.target_net.state_dict, self.PATH)

        print('Complete')
        self.close()
