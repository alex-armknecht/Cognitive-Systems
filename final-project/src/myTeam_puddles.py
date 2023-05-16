# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import json

##################
# Game Constants #
##################

# Set TRAINING to True while agents are learning, False if in deployment
# [!] Submit your final team with this set to False!
TRAINING = False
EPS_GREEDY = 0.1
DISCOUNT = 0.8 #double check
ALPHA = 0.01

# Name of weights / any agent parameters that should persist between
# games. Should be loaded at the start of any game, training or otherwise
# [!] Replace MY_TEAM with your team name
OFFENSIVE_WEIGHT_PATH = 'weights_puddles_off.json'
DEFENSIVE_WEIGHT_PATH = 'weights_puddles_def.json'

# Any other constants used for your training (learning rate, discount, etc.)
# should be specified here
# [!] TODO

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    Notes:
    need to add: something to determine either win or die, numpelseaten association with returning home,
    prioritization of moving back where we came from when returning home = 1.  
    '''
    self.weights = util.Counter()
    self.last_state = None
    self.last_action = None
    self.invader_loc = None
    self.line_of_scrim = None
    self.width = gameState.getWalls().width
    self.height = gameState.getWalls().height

    if self.red:
      self.invader_loc = (13, 7)
      self.line_of_scrim = 16
      self.debugDraw((self.line_of_scrim, self.height//2 ), [1,0,0])
    else:
      self.invader_loc = (17, 7)
      self.line_of_scrim = 15
      self.debugDraw((self.line_of_scrim, self.height//2 ), [1,0,0])
    
    defense = isinstance(self, DefensiveAgent)
    if defense:
      with open(DEFENSIVE_WEIGHT_PATH) as json_file:
        weightDict = json.load(json_file)
        if weightDict:
          self.weights = weightDict
    else:
      with open(OFFENSIVE_WEIGHT_PATH) as json_file:
        weightDict = json.load(json_file)
        if weightDict:
          self.weights = weightDict
    


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    if self.last_state != None or self.last_action != None:
      self.update(self.last_state, self.last_action, gameState, self.getReward(self.last_state, self.last_action, gameState))
      self.last_state = gameState

    sample = random.random()
    best_actions = []
    actions_and_q_vals = {}
    if (sample > EPS_GREEDY) or (not TRAINING):
      for action in actions:
        actions_and_q_vals[action] = self.getQValue(gameState, action)
      max_val = max(actions_and_q_vals.values())
      for action, q_val in actions_and_q_vals.items():
        if q_val == max_val:
          best_actions.append(action)
      best_action = random.choice(best_actions)
      self.last_state = gameState
      self.last_action = best_action
      return best_action
    else:
      best_action = random.choice(actions)
      self.last_state = gameState
      self.last_action = best_action
      return best_action

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos): return successor.generateSuccessor(self.index, action)
    else: return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """

    features = util.Counter()
    features['eatsPellet'] = 0
    features['closerToPel'] = 0
    features['headedHome'] = 0
    features['movedTowardsGhost'] = 0

    successor = self.getSuccessor(gameState, action)
    location = successor.getAgentState(self.index)
    prevLoc = gameState.getAgentPosition(self.index)
    myPos = location.getPosition()
    food_locs_len = len(self.getFood(gameState).asList())
    food_locs_len_next = len(self.getFood(successor).asList())

    if food_locs_len_next < food_locs_len:
        features['eatsPellet'] = 1

    pellet_dist = self.findClosestPel(gameState)
    successor_pel_dist = self.findClosestPel(successor)
    if successor_pel_dist < pellet_dist:
      features['closerToPel'] = 1

    if gameState.getAgentState(self.index).numCarrying >=4:
      home_dist = self.returnHome(gameState)
      next_home_dist = self.returnHome(successor)
      if next_home_dist < home_dist:
        features['headedHome'] = 1
      
    enemies_gameState = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
    ghosts_gameState = [a for a in enemies_gameState if not a.isPacman and a.getPosition()
                    != None]
    enemies = [successor.getAgentState(i) 
                   for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition()
                    != None and a.getPosition()[0] != self.line_of_scrim] 
    if (len(ghosts_gameState) > 0) and (len(ghosts) > 0): 
      dist_to_ghost = self.getMazeDistance(prevLoc, ghosts_gameState[0].getPosition())
      next_dist_to_ghost = self.getMazeDistance(myPos, ghosts[0].getPosition())
      if (next_dist_to_ghost < dist_to_ghost) :
        features['movedTowardsGhost'] = 1
    return features

  def getReward(self, state, action, nextState):
    nextStateLocation = nextState.getAgentPosition(self.index)
    features = self.getFeatures(state, action)
    sumRewards = 0
    if features['eatsPellet'] == 1 :
      sumRewards += 0.17
    if features['closerToPel'] == 1 : 
      sumRewards += 0.04
    if features['headedHome'] == 1:
      sumRewards += 0.57
    if features['movedTowardsGhost'] == 1 :
      sumRewards -= 0.77
    if self.didIDie(nextState, state) :
      sumRewards -= 1

    return sumRewards

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return self.weights

  def getQValue(self, gameState, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    feats = self.getFeatures(gameState, action)
    return feats * self.weights

  def getHighestQVal(self, gameState) :
    actions = gameState.getLegalActions(self.index)
    q_vals = []
    for action in actions:
      q_vals.append(self.getQValue(gameState, action))
    return max(q_vals)


  
  def update(self, state, action, nextState, reward):
    """
        Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    features = self.getFeatures(state, action)
    q_val = self.getQValue(state, action)
    action_qvals = util.Counter()
    future_qval = self.getHighestQVal(nextState)
    difference = (reward + DISCOUNT * future_qval) - q_val
    for state_action, feature_val in features.items():
      new_weight = ALPHA * difference * feature_val
      self.weights[state_action] += new_weight
    defense = isinstance(self, DefensiveAgent)
    self.printWeights(defense)
    
      
  def printWeights(self, defense):
    fileName = ""
    if defense:
      fileName = "weights_puddles_def.json"
    else:
      fileName = "weights_puddles_off.json"
    out_file = open(fileName, "w")
    json.dump(self.weights, out_file)
    out_file.close()
    
  def findClosestPel(self, state) :
    foodFound = self.getFood(state).asList()
    min_distance = 1000
    for loc in foodFound:
      distance = self.getMazeDistance(state.getAgentPosition(self.index), loc) 
      if self.getMazeDistance(state.getAgentPosition(self.index), loc) < min_distance:
        min_distance = distance
    return min_distance
  
  def returnHome(self, state):
    goalX = 16
    min_distance = 1000
    if state.isOnRedTeam(self.index):
      goalX = 14 
    for i in range(1, 14):
      legal = state.getWalls().asList(False)
      if (goalX, i) in legal :
        distance = self.getMazeDistance(state.getAgentPosition(self.index), (goalX,i)) 
        if self.getMazeDistance(state.getAgentPosition(self.index), (goalX,i)) < min_distance:
          min_distance = distance

    return min_distance
  
  def didIDie(self, state, pastState):
    IAmPac = pastState.getAgentState(self.index).isPacman
    curPos = state.getAgentState(self.index).getPosition()
    if IAmPac and curPos == self.start: return True
    return False
  

class DefensiveAgent(DummyAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def chooseAction(self, gameState):
      """
      Picks among actions randomly.
      """
      actions = gameState.getLegalActions(self.index)
      actions.remove(Directions.STOP)
      if self.last_state != None or self.last_action != None:
        self.update(self.last_state, self.last_action, gameState, self.getReward(self.last_state, self.last_action, gameState))
        self.last_state = gameState
      best_actions = []
      actions_and_q_vals = {}
      for action in actions:
        actions_and_q_vals[action] = self.getQValue(gameState, action)
      max_val = max(actions_and_q_vals.values())
      for action, q_val in actions_and_q_vals.items():
        if q_val == max_val:
          best_actions.append(action)
      best_action = random.choice(best_actions)
      self.last_state = gameState
      self.last_action = best_action
      return best_action


    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        location = successor.getAgentState(self.index)
        prevLoc = gameState.getAgentPosition(self.index)
        myPos = location.getPosition()

        features['eatsEnemy'] = 0
        features['closerToPel'] = 0

        features['onDefense'] = 1
        if location.isPacman:
            features['onDefense'] = 0

        enemies_gameState = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
        invaders_gameState = [a for a in enemies_gameState if a.isPacman and a.getPosition()
                    != None]
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None and a.getPosition()[0] != self.line_of_scrim]
        features['numInvaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (len(invaders_gameState) > 0) and (len(invaders) > 0):
          stateGhostDist = self.getMazeDistance(prevLoc, invaders_gameState[0].getPosition())
          nextStateGhostDist = self.getMazeDistance(myPos, invaders[0].getPosition())
          if stateGhostDist == 1 and ( nextStateGhostDist > 2 or invaders[0].getPosition() == None) : 
            features['eatsEnemy'] = 1
            self.recenter()
        
        our_side_food_next  = self.getFoodYouAreDefending(successor).asList()
        our_side_food = self.getFoodYouAreDefending(gameState).asList()
        if len(our_side_food_next) < len(our_side_food) :
          pellet_eaten = set(our_side_food) - set(our_side_food_next)
          self.invader_loc = list(pellet_eaten)[0]

        if len(invaders) > 1:
          self.invader_loc = invaders[0].getPosition()

        if self.invader_loc != None:
          dist_to_invader = self.getMazeDistance(prevLoc, self.invader_loc)
          next_dist_to_invader = self.getMazeDistance(myPos, self.invader_loc)
          if next_dist_to_invader < dist_to_invader:
            features['closerToInvader'] = 1

        return features
    
    def recenter(self):
      if self.red:
        self.invader_loc = (13, 7)
      else:
        self.invader_loc = (17, 7)

    def getReward(self, state, action, nextState):
      successor = self.getSuccessor(state, action)
      features = self.getFeatures(state, action)
      sumRewards = 0
      
      if features['onDefense'] == 1 : 
        sumRewards += 0.02
      if features['eatsEnemy'] == 1 :
        sumRewards += 0.2
      if features['closerToInvader'] == 1:
        sumRewards += 0.05

      if state.getAgentPosition(self.index)[0] >= self.line_of_scrim and self.red:
        sumRewards -= 1
      elif state.getAgentPosition(self.index)[0] <= self.line_of_scrim and not self.red:
        sumRewards -= 1
      
      return sumRewards



