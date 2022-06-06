"""Configuration"""
ALPHA = 0.05  # ALPHA:
BETA = 0.03  # BETA:
TAU = 5.0   # Final TAU
ZETA = 0.01
EXPLORE_EPISODES = 2500
EPISODES = 7500
EXPLORE = True  # WHETER OR NOT WE USE EXPLORATION
CONSENSUS_MATRIX_TYPE = 'laplacian'

SEED = 1
BASE_PATH = 'data/04_duo_consensus_learners/02_laplacian'

N_WORKERS = 6
N_AGENTS = 2
AGENT_TYPE = 'ActorCriticConsensus'

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
