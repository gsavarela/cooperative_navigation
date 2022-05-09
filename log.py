"""Logging functions to help understating whats going on.

when we need to check for overflow.
"""
from typing import List
import config
from pathlib import Path

import numpy as np
import pandas as pd

from common import Array, PlayerActions


def critic(
    deltas: List[float],
    rewards: List[float],
    mus: List[float],
    dxs: List[float],
    omegas0: List[float],
    omegas1: List[float],
    xs: List[float],
) -> None:
    path = Path(config.BASE_PATH) / "{0:02d}".format(config.SEED)
    # Create individual dataframes
    deltas_df = pd.DataFrame(np.vstack(deltas), columns=["deltas"])
    rewards_df = pd.DataFrame(np.vstack(rewards), columns=["rewards"])
    mus_df = pd.DataFrame(np.vstack(mus), columns=["mus"])
    dxs_df = pd.DataFrame(
        np.vstack(dxs), columns=["dx_a", "dy_a", "dv_x", "dv_y", "dx_r", "dy_r"]
    )
    omegas0_df = pd.DataFrame(
        np.vstack(omegas0), columns=["w0_1", "w0_2", "w0_3", "w0_4", "w0_5", "w0_6"]
    )
    omegas1_df = pd.DataFrame(
        np.vstack(omegas1), columns=["w1_1", "w1_2", "w1_3", "w1_4", "w1_5", "w1_6"]
    )
    xs_df = pd.DataFrame(
        np.vstack(xs), columns=["x_a", "y_a", "v_x", "v_y", "x_r", "y_r"]
    )

    # 1. delta dataframe
    dataframes = [deltas_df, rewards_df, mus_df, dxs_df, omegas0_df]
    df = pd.concat(dataframes, axis=1)
    df.to_csv(path / "deltas-seed{0:02d}.csv".format(config.SEED))

    # 2. mu dataframe
    dataframes = [mus_df, deltas_df]
    df = pd.concat(dataframes, axis=1)
    df.to_csv(path / "mus-seed{0:02d}.csv".format(config.SEED))

    # 3. omegas dataframe
    dataframes = [omegas1_df, omegas0_df, deltas_df, xs_df]
    df = pd.concat(dataframes, axis=1)
    df.to_csv(path / "omegas-seed{0:02d}.csv".format(config.SEED))


def actor(
    taus: List[Array],
    pis: List[Array],
    pis_tau: List[Array],
) -> None:
    path = Path(config.BASE_PATH) / "{0:02d}".format(config.SEED)
    # Create individual dataframes
    taus_df = pd.DataFrame(data=np.vstack(taus), columns=["taus"])
    columns = [str(PlayerActions(act).name) for act in range(5)]
    action_columns = ["ACTUAL_%s" % col for col in columns]
    pis_df = pd.DataFrame(
        data=np.vstack(pis),
        columns=action_columns,
    )
    tau_columns = ["TEMP_%s" % col for col in columns]
    pis_tau_df = pd.DataFrame(data=np.vstack(pis_tau), columns=tau_columns)
    dataframes = [taus_df, pis_df, pis_tau_df]
    df = pd.concat(dataframes, axis=1)
    df.to_csv(path / "pi-seed{0:02d}.csv".format(config.SEED))


def traces(
    x0: List[Array],
    actions: List[int],
    x1: List[Array],
    vs: List[float],
) -> None:

    path = Path(config.BASE_PATH) / "{0:02d}".format(config.SEED)
    actions = [str(PlayerActions(act).name) for act in actions]

    # individual dataframes
    x0_df = pd.DataFrame(
        data=np.vstack(x0), columns=["x0_1", "x0_2", "x0_3", "x0_4", "x0_5", "x0_6"]
    )
    actions_df = pd.DataFrame(data=actions, columns=["actions"])
    x1_df = pd.DataFrame(
        data=np.vstack(x1), columns=["x1_1", "x1_2", "x1_3", "x1_4", "x1_5", "x1_6"]
    )

    vs_df = pd.DataFrame(data=np.vstack(vs), columns=["v(x)"])

    # delta dataframe
    dataframes = [x0_df, actions_df, x1_df, vs_df]
    df = pd.concat(dataframes, axis=1)
    df.to_csv(path / "traces-seed{0:02d}.csv".format(config.SEED))
