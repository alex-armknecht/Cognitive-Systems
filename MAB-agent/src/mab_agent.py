'''
  mab_agent.py
  
  Agent specifications implementing Action Selection Rules.
'''

import numpy as np
import random
import math

# ----------------------------------------------------------------
# MAB Agent Superclasses
# ----------------------------------------------------------------

class MAB_Agent:
    '''
    MAB Agent superclass designed to abstract common components
    between individual bandit players (below)
    ---- These are needed for everything else ----

    K = len(P_R)
    K = how many ads you have to pick from 

    P_R = likelihood someone will click on that ad
    P_R = np.array(
  # A =  0     1     2     3
     [0.50, 0.60, 0.40, 0.30]
)
    '''
    
    def __init__ (self, K):
        self.history = [[1,2] for i in range(K)]
        self.K = K
        self.iterations = 0
        
    def give_feedback (self, a_t, r_t):
        '''
        Provides the action a_t and reward r_t chosen and received
        in the most recent trial, allowing the agent to update its
        history
        [!] Called by the simulations after your agent's choose
            method is called
        '''
        if r_t : 
            self.history[a_t][0] = self.history[a_t][0] + 1 
        self.history[a_t][1] = self.history[a_t][1] + 1 


    
    def clear_history(self):
        '''
        IMPORTANT: Resets your agent's history between simulations.
        No information is allowed to transfer between each of the N
        repetitions
        [!] Called by the simulations after a Monte Carlo repetition
        '''  
        self.history = [[1,2] for i in range(self.K)]
        self.iterations = 0

    def make_greedy_choice(self): 
        best_Q_val = [(0,0)] 
        for i in range(0, len(self.history)):
            q_val = self.history[i][0] / self.history[i][1]
            if best_Q_val[0][1] < q_val  :
                best_Q_val = [(i, q_val)] 
            elif best_Q_val[0][1] == q_val :
                best_Q_val.append((i, q_val))
        return random.choice(best_Q_val)[0]



# ----------------------------------------------------------------
# MAB Agent Subclasses
# ----------------------------------------------------------------

class Greedy_Agent(MAB_Agent):
    '''
    greedy_Agent = inheritance = it has the attributions and methods of MAB_Agent

    Greedy bandit player that, at every trial, selects the
    arm (ad) with the presently-highest sampled Q value (number of clicks per ad shown)
    '''
    
    def __init__ (self, K):
        MAB_Agent.__init__(self, K)
    
    def choose (self, *args):
        return Greedy_Agent.make_greedy_choice(self)

        


class Epsilon_Greedy_Agent(MAB_Agent):
    '''
    Exploratory bandit player that makes the greedy choice with
    probability 1-epsilon, and chooses randomly with probability
    epsilon
    '''
    
    def __init__ (self, K, epsilon):
        MAB_Agent.__init__(self, K)
        self.epsilon = epsilon
        
    def choose (self, *args):
        greedy_choice = Epsilon_Greedy_Agent.make_greedy_choice(self)         
        coinflip = random.uniform(0.0,1.0)
        if coinflip >= self.epsilon: return greedy_choice 
        return np.random.choice(list(range(self.K))) 
        


class Epsilon_First_Agent(MAB_Agent):
    '''
    Exploratory bandit player that takes the first epsilon*T
    trials to randomly explore, and thereafter chooses greedily
    '''
    
    def __init__ (self, K, epsilon, T):
        MAB_Agent.__init__(self, K)
        self.T = T
        self.epsilon = epsilon
        
    def choose (self, *args):
        self.iterations += 1
        if self.iterations < self.T * self.epsilon:
            return np.random.choice(list(range(self.K)))
        return Epsilon_First_Agent.make_greedy_choice(self)  


class Epsilon_Decreasing_Agent(MAB_Agent):
    '''
    Exploratory bandit player that acts like epsilon-greedy but
    with a decreasing value of epsilon over time
    '''
    
    def __init__ (self, K):
        MAB_Agent.__init__(self, K)
        self.epsilon = 1
        
    def choose (self, *args):
        greedy_choice = Epsilon_Decreasing_Agent.make_greedy_choice(self)         
        coinflip = random.uniform(0.0,1.0)
        if coinflip >= self.epsilon: 
            self.epsilon -= 0.005 
            return greedy_choice
        self.epsilon -= 0.005
        return np.random.choice(list(range(self.K))) 
    
    def clear_history(self):
        self.epsilon = 1
        self.history = [[1,2] for i in range(self.K)]
        self.iterations = 0


class TS_Agent(MAB_Agent):
    '''
    Thompson Sampling bandit player that self-adjusts exploration
    vs. exploitation by sampling arm qualities from successes
    summarized by a corresponding beta distribution

    Encoding wins and losses: 
    numpy.random.beta(wins, losses) = part of solution 

    win = interaction; self.history[i][0]

    loss = non-interaction with a view (view - interactions)
    loss = self.history[i][1] - self.history[i][0]


   *** numpy.random.beta(self.history[i][0], self.history[i][1] - self.history[i][0]) **
   - ^^ this is the samplin g
   ^^ all we need to do is collect all the information from the sample then have it pick 
   - since [i] is for a specific arm we need to make it adaptable for all arms 


    '''
    
    def __init__ (self, K):
        MAB_Agent.__init__(self, K)
    
    def choose (self, *args):
        ts_History = []
        for i in range(0, len(self.history)):
            ts_History.append(np.random.beta(self.history[i][0], self.history[i][1] - self.history[i][0]))
        return np.argmax(ts_History)
    
    
class Custom_Agent(MAB_Agent):
    '''
    Custom agent that manages the explore vs. exploit dilemma via
    your own strategy, or by implementing a strategy you discovered
    that is not amongst those above!
    '''
    def __init__ (self, K):
        MAB_Agent.__init__(self, K)
    
    def choose (self, *args):
        ca_history = []
        self.iterations += 1
        c = .5  
        log = float(math.log(int(self.iterations)))
        
        for i in range(0, self.K):
            q = self.history[i][0]/self.history[i][1]
            ca_history.append(q+ c*math.sqrt( log / self.history[i][1]))
        return np.argmax(ca_history)

