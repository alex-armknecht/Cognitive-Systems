'''
Simulation / Maze constants important for the Pacman problem

[!] IMPORTANT:
  - YOU MUST NOT TOUCH THIS FILE AT ALL, NO EDITS OR ADDITIONS!
    Any changes will be overwritten during testing
  - If you need additional constants shared between your files,
    make your own damn module
'''

import torch

class Constants:

    # Simulation constants
    # -------------------------------------------------------------------
    # Number of games to run in reinforcement_trainer.py
    N_SIMS = 8000
    # Max number of moves available to Pacman per game
    MAX_MOVES = 200
    # How long the game will pause between moves. Use 0 for ASAP games
    TICK_LEN = 0 # in ms
    # Random move propensity of ghosts -- higher = easier
    GHOST_EPSILON = 0.1
    # Whether the debug output will print during each game
    DEBUG = False
    # Whether the game will be rendered in the terminal with action choices
    VERBOSE = False
    # Whether the game GUI will launch to show you the game state
    GUI = False
    # Whether or not the models will be training and Pacman taking exploratory
    # moves. When false, takes only greedy moves.
    TRAINING = True
    
    # Training Constants
    # -------------------------------------------------------------------
    # Number of episodes to sample from the Replay Buffer
    BATCH_SIZE = 32
    # Discount Factor (remember: Gamma = Grandma loves her discounts)
    GAMMA = 0.95
    # Exploration rate for epsilon-greedy agent (can be changed to other ASRs)
    EPS_GREEDY = 0.1
    # Number of episodes between when the target network's weights are set to
    # the policy net's current weights
    TARGET_UPDATE = 100
    # Episode capacity of the ReplayMemory
    MEM_SIZE = 10000
    # Path of weights that will be saved for the policy network when prompted
    PARAM_PATH = "./dat/params_ghosty.pth"
    # Path to which the ReplayMemory will be saved and loaded when prompted
    MEM_PATH = "./dat/mem_ghosty.pkl"

    # Movement constants + location modifiers
    # -------------------------------------------------------------------
    MOVES = ["U", "D", "L", "R"]
    MOVE_DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

    # Maze content constants
    # -------------------------------------------------------------------
    WALL_BLOCK = "X"
    GHOST_BLOCK = "G"
    PELLET_BLOCK = "O"
    SAFE_BLOCK = "."
    PLR_BLOCK = "P"
    DEATH_BLOCK = "D"
    WIN_BLOCK = "W"
    TIMEOUT_BLOCK = "T"
    ENTITIES = [WALL_BLOCK, PELLET_BLOCK, PLR_BLOCK, GHOST_BLOCK]
    
    # Used to determine whether GPU acceleration is available or not
    # -------------------------------------------------------------------
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Active Maze Environment
    # -------------------------------------------------------------------
    # Maze 1: Train and save to params_pellets.pth
    MAZE = ["XXXXXXXXX",
            "X..O...PX",
            "X.......X",
            "X..XXXO.X",
            "XO.....OX",
            "X.......X",
            "XXXXXXXXX"]
    
    # Maze 2: Train and save to params_ghosty.pth
    MAZE = ["XXXXXXXXX",
           "X..O...PX",
           "X.......X",
           "X..XXXO.X",
           "XO.....OX",
           "X......GX",
           "XXXXXXXXX"]
    