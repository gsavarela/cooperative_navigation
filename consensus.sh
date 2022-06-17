#!/bin/sh
# Tests consensus but set different CONSENSUS_MAX_EDGES
sed -i "s/^AGENT_TYPE = .*/AGENT_TYPE = 'ActorCriticConsensus'/g" config.py
sed -i "s/^N_AGENTS = .*/N_AGENTS = 2/g" config.py
sed -i "s/^EXPLORE_EPISODES = .*/EXPLORE_EPISODES = 3000/g" config.py
sed -i "s/^EPISODES = .*/EPISODES = 7500/g" config.py
for n in 0 1 
do
  # Replace agent
  dir="data\/00_duo_consensus\/0${n}_edges/0${n}" &&
  sed -i "s/^CONSENSUS_MAX_EDGES = .*/CONSENSUS_MAX_EDGES = $n/g" config.py &&
  sed -i "s/^BASE_PATH = .*/BASE_PATH = '$dir'/g" config.py  &&
  ./pipeline.py
done

sed -i "s/^N_AGENTS = .*/N_AGENTS = 3/g" config.py
sed -i "s/^EXPLORE_EPISODES = .*/EXPLORE_EPISODES = 10000/g" config.py
sed -i "s/^EPISODES = .*/EPISODES = 30000/g" config.py
for n in 0 1 2 3 
do
  # Replace agent
  dir="data\/01_trio_consensus\/0${n}_edges/0${n}" &&
  sed -i "s/^CONSENSUS_MAX_EDGES = .*/CONSENSUS_MAX_EDGES = $n/g" config.py &&
  sed -i "s/^BASE_PATH = .*/BASE_PATH = '$dir'/g" config.py && 
  ./pipeline.py
done
