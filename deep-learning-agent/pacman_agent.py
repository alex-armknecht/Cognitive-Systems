'''
Pacman Agent employing a PacNet trained in another module to
navigate perilous ghostly pellet mazes.
'''

import time
import random
import numpy as np
import torch
from os.path import exists
from torch import nn
from pathfinder import *
from queue import Queue
from constants import *
from reinforcement_trainer import *
from maze_problem import *
from itertools import chain

class PacmanAgent:
    '''
    Deep learning Pacman agent that employs PacNet DQNs.
    '''

    def __init__(self, maze):
        """
        Initializes the PacmanAgent with any attributes needed to make decisions;
        for the deep-learning implementation, this includes initializing the
        policy DQN (+ target DQN, ReplayMemory, and optimizer if training) and
        any other 
        :maze: The maze on which this agent is to operate. Must be the same maze
        structure as the one on which this agent's model was trained.
        """
        self.policy_net =  PacNet(maze)
        if exists(Constants.PARAM_PATH) : self.policy_net.load_state_dict(torch.load(Constants.PARAM_PATH))
        self.target_net =  PacNet(maze)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.replay_mem = ReplayMemory(Constants.MEM_SIZE) 
        if exists(Constants.MEM_PATH) : self.replay_mem.load()
        self.steps_done = 0
        self.maze = maze
        self.optimizer = torch.optim.Adam(self.policy_net.parameters()) 
        return

    def choose_action(self, perception, legal_actions):
        """
        Returns an action from the options in Constants.MOVES based on the agent's
        perception (the current maze) and legal actions available. If training,
        must manage the explore vs. exploit dilemma through some form of ASR.
        :perception: The current maze state in which to act
        :legal_actions: Map of legal actions to their next agent states
        :return: Action choice from the set of legal_actions
        """
        legal_actions = dict(legal_actions)
        vectorized = torch.from_numpy(ReplayMemory.vectorize_maze(perception)).float().unsqueeze(0)
        sample = random.random()
        if not Constants.TRAINING or sample > Constants.EPS_GREEDY:
            Q_vals = self.policy_net(vectorized).tolist()[0] + []
            Q_vals_dict = {"U": Q_vals[0], "D": Q_vals[1], "L": Q_vals[2], "R": Q_vals[3]}
            legal_dict = {key: value for key, value in Q_vals_dict.items() if key in legal_actions}
            return max(legal_dict, key=legal_dict.get)

        else:
            return random.choice(list(legal_actions.keys()))


    
    def get_reward(self, state, action, next_state):
        '''
        The reward function that determines the numerical desirability of the
        given transition from state -> next_state with the chosen action.
        :state: state at which the transition begun
        :action: the action the agent chose from state
        :next_state: the state at which the agent began its next turn
        :returns: R(s, a, s') for the given transition
        '''
        past_maze = MazeProblem(state) 
        current_maze = MazeProblem(next_state) 
        if not current_maze.get_win_state() is None : return 500 
        if len(past_maze.get_pellets()) > len(current_maze.get_pellets()) : return 30
        if current_maze.get_death_state() : return -500
        if current_maze.get_timeout_state() : return -300  
        return -2
    
    def give_transition(self, state, action, next_state, is_terminal):
        '''
        Called by the Environment after both Pacman and ghosts have moved on a
        given turn, supplying the transition that was observed, which can then
        be added to the training agent's memory and the model optimized. Also
        responsible for periodically updating the target network.
        [!] If not training, this method should do nothing.
        :state: state at which the transition begun
        :action: the action the agent chose from state
        :next_state: the state at which the agent began its next turn
        :is_terminal: whether or not next_state is a terminal state
        '''
        if not Constants.TRAINING: return
        reward = self.get_reward(state, action, next_state)
        self.replay_mem.push(ReplayMemory.vectorize_maze(state), ReplayMemory.move_vec_to_index(ReplayMemory.vectorize_move(action)), 
                             ReplayMemory.vectorize_maze(next_state), reward, is_terminal)
        self.optimize_model()
        if self.steps_done == Constants.TARGET_UPDATE:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.steps_done = 0
        self.steps_done += 1 
        return
    
    def give_terminal(self):
        '''
        Called by the Environment upon reaching any of the terminal states:
          - Winning (eating all of the pellets)
          - Dying (getting eaten by a ghost)
          - Timing out (taking more than Constants.MAX_MOVES number of turns)
        Useful for cleaning up fields, saving weights and memories to disk if
        desired, etc.
        [!] If not training, this method should do nothing.
        '''
        if not Constants.TRAINING: return
        self.replay_mem.save()
        torch.save(self.policy_net.state_dict(), Constants.PARAM_PATH) 
        return
    
    def optimize_model(self):
        '''
        Primary workhorse for training the policy DQN. Samples a mini-batch of
        episodes from the ReplayMemory and then takes a step of the optimizer
        to train the DQN weights.
        [!] If not training OR fewer episodes than Constants.BATCH_SIZE have
        been recorded, this method should do nothing.
        🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦🦦
        '''
        if not Constants.TRAINING or len(self.replay_mem) < Constants.BATCH_SIZE: return
        episodes = self.replay_mem.sample(Constants.BATCH_SIZE)
    
        batch = Episode(*zip(*episodes))
        terminal_batch = torch.tensor(batch.is_terminal)
        next_state= torch.tensor(batch.next_state)
        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)
        non_final_mask = torch.tensor(tuple(map(lambda t: t == False, terminal_batch)), dtype=torch.bool) 
        non_final_next_states = torch.stack([s for s,t in zip(next_state, terminal_batch) if t == False]).float()
        action_batch = torch.unsqueeze(action_batch, dim=-1)
        terminal_batch = torch.unsqueeze(terminal_batch, dim=-1)
        reward_batch = torch.unsqueeze(reward_batch, dim=-1)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(Constants.BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values.unsqueeze(1) * Constants.GAMMA) + reward_batch.float()

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return
