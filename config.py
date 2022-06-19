"""AGENT SETTINGS"""
# The hyperparameter for the environment: Number of agents.
N_AGENTS = 2
# The pipeline agent
AGENT_TYPE = 'ActorCriticCentral'
# Alpha learning rate for the critic
ALPHA = 0.5
# Learning rate for the actor
BETA = 0.3
# Final temperature parameter
TAU = 5.0
# The weighting for the moving average
ZETA = 0.01
# The episodes with temperature greater than TAU
EXPLORE_EPISODES = 3000
# Applies temperature parameter from 100-TAU
EXPLORE = True

"""ENVIRONMENT SETTINGS"""
# Only for consensus learners
CONSENSUS_MATRIX_TYPE = 'metropolis'
# Maximum number of edges.
CONSENSUS_MAX_EDGES = 0
# Scales rewards by a random coefficient
RANDOMIZE_REWARD_COEFFICIENTS = True

"""TRAINING AND EVALUATION SETTINGS"""
# The total number of training episodes.
EPISODES = 7500
# Stop training at every checkpoint_interval
CHECKPOINT_INTERVAL = 500
# Evaluate checkpoint for checkpoint evaluations
CHECKPOINT_EVALUATIONS = 32
# Training_cycle = train for checkpoint interval + checkpoint_evaluation
TRAINING_CYCLES = EPISODES // CHECKPOINT_INTERVAL
# The path that the experiments will be saved at.
BASE_PATH = 'data/05_duo_randomized_coefficents'
# The number of pipeline workers
N_WORKERS = 10
# Seed for individual runs, e.g, `python central.py`
SEED = 1
# Seed increment for rollouts
ROLLOUT_SEED_INC = 1000
# Those are the training random seeds, i.e., `.\pipeline`
PIPELINE_SEEDS = [
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76
]
