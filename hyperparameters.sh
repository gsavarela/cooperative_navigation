#!/bin/sh

sed -i "s/TAU = .*/TAU = 3.0/g" config.py
sed -i "s/BASE_PATH = .*/BASE_PATH = 'data\/00_duo_hyperparameter_search\/00_tau03'/g" config.py
git add config.py && git commit -m 'Set tau temperature parameter to 3.0'
./pipeline.py

sed -i "s/TAU = .*/TAU = 5.0/g" config.py
sed -i "s/BASE_PATH = .*/BASE_PATH = 'data\/00_duo_hyperparameter_search\/01_tau05'/g" config.py
git add config.py && git commit -m 'Set tau temperature parameter to 5.0'
./pipeline.py

sed -i "s/TAU = .*/TAU = 10.0/g" config.py
sed -i "s/BASE_PATH = .*/BASE_PATH = 'data\/00_duo_hyperparameter_search\/02_tau10'/g" config.py
git add config.py && git commit -m 'Set tau temperature parameter to 10.0'
./pipeline.py

sed -i "s/TAU = .*/TAU = 5.0/g" config.py
git add config.py && git commit -m 'Revert tau temperature parameter to 5.0'

sed -i "s/ALPHA = .*/ALPHA = 0.05/g" config.py
sed -i "s/BETA = .*/BETA = 0.01/g" config.py
sed -i "s/BASE_PATH = .*/BASE_PATH = 'data\/00_duo_hyperparameter_search\/03_alpha005_beta001'/g" config.py
git add config.py && git commit -m 'Set alpha = 0.05 and beta = 0.01'
./pipeline.py

sed -i "s/ALPHA = .*/ALPHA = 0.1/g" config.py
sed -i "s/BETA = .*/BETA = 0.05/g" config.py
sed -i "s/BASE_PATH = .*/BASE_PATH = 'data\/00_duo_hyperparameter_search\/04_alpha010_beta005'/g" config.py
git add config.py && git commit -m 'Set alpha = 0.1 and beta = 0.05'
./pipeline.py

sed -i "s/ALPHA = .*/ALPHA = 0.9/g" config.py
sed -i "s/BETA = .*/BETA = 0.5/g" config.py
sed -i "s/BASE_PATH = .*/BASE_PATH = 'data\/00_duo_hyperparameter_search\/05_alpha090_beta050'/g" config.py
git add config.py && git commit -m 'Set alpha = 0.9 and beta = 0.5'
./pipeline.py

sed -i "s/ALPHA = .*/ALPHA = 0.5/g" config.py
sed -i "s/BETA = .*/BETA = 0.3/g" config.py
git add config.py && git commit -m 'Revert alpha = 0.5 and beta = 0.3'

sed -i "s/ZETA = .*/ZETA = 0.001/g" config.py
sed -i "s/BASE_PATH = .*/BASE_PATH = 'data\/00_duo_hyperparameter_search\/06_zeta0001'/g" config.py
git add config.py && git commit -m 'Set zeta = 0.001 the weighting of the global average'
./pipeline.py

sed -i "s/ZETA = .*/ZETA = 0.01/g" config.py
sed -i "s/BASE_PATH = .*/BASE_PATH = 'data\/00_duo_hyperparameter_search\/07_zeta001'/g" config.py
git add config.py && git commit -m 'Set zeta = 0.001 the weighting of the global average'
./pipeline.py

sed -i "s/ZETA = .*/ZETA = 0.05/g" config.py
sed -i "s/BASE_PATH = .*/BASE_PATH = 'data\/00_duo_hyperparameter_search\/08_zeta005'/g" config.py
git add config.py && git commit -m 'Set zeta = 0.05 the weighting of the global average'
./pipeline.py

sed -i "s/ZETA = .*/ZETA = 0.10/g" config.py
git add config.py && git commit -m 'Revert zeta = 0.1 the weighting of the global average'
