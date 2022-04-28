# Cooperative Navigation Task

The Multi-agent reinforcement learning on [Lowe, et al. 2017](https://arxiv.org/pdf/1706.02275.pdf).

# Core Principles

- Muh (minimalism):
    - Less frameworks as possible.
    - No. Reinforcement learning is not deep reinforcement learning.
    - The python should approximate a papers' pseudocode.
- Follow [numpydoc guidelines](https://numpydoc.readthedocs.io/en/latest/format.html)
- When in doubt follow UNIX: 
    - Make it easy to write, test, and run programs.
    - Interactive use instead of batch processing.
    - Economy and elegance of design due to size constraints ("salvation through suffering").

# Requirements
- python==3.5.4
- gym==0.10.5
- numpy==1.14.5

# Getting Started

1. Install [multiagent-particles-env](https://github.com/openai/multiagent-particle-envs).
```
> git clone https://github.com/openai/multiagent-particle-envs.git
```

2. Install [cooperative_navigation](https://github.com/gsavarela/cooperative_navigation.git)
```
> git clone https://github.com/gsavarela/cooperative_navigation.git
```
3. Run:
```
> pip install -r requirements.txt.
```
4. Invoke from the project's root directory:
```
>./pipeline.py
```

5. Change parameters on pipeline to your heart's content.


