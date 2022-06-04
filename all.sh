#!/bin/sh
# for-in loop to execute pipeline on all agents
for n in Central Distributed Independent Consensus
do
  # Replace agent
  sed -i "s/AGENT_TYPE = .*/AGENT_TYPE = 'ActorCritic$n'/g" config.py && ./pipeline.py
done
