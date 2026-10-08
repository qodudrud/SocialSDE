import copy

import argparse

import os.path as path

import numpy as np
import pandas as pd

import torch
from LBN.model.train_swag import train_epoch, evaluate



# ---------- Argument parser ----------

def type_or_none(base_type):
    """
    A helper function to create an argparse type that can parse a given base type or 'None' (case-insensitive) to None.
    Args:
        base_type: The base type to parse (e.g., int, float, str).
    Returns:
        A function that can be used as the 'type' argument in argparse that will parse the base type or 'None'.
    """
    def parse_value(value):
        if value.lower() == 'none':
            return None
        try:
            return base_type(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value: '{value}'. Expected {base_type.__name__} or 'None'."
            )
    return parse_value


def build_parser(
    ) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Langevin Bayesian Neural networks (LBN)"
    )
    # Basic settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use for computation (default: 'cuda')",
    )
    parser.add_argument(
        "--no-cuda", 
        action="store_true", 
        default=False, 
        help="disables CUDA training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--Langevin-type",
        type=str,
        default='OLE',
        help="Langevin type: 'OLE' or 'ULE' (default: 'OLE')"
    )
    parser.add_argument(
        "--model-type", 
        type=str, 
        default='drift', 
        help="Model type: 'drift' or 'diff' (default: 'drift')"
    )
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        metavar="DATA-PATH",
        help="Data path (default: None)",
    )
    parser.add_argument(
        "--train", 
        type=int, 
        default=1, 
        help="Train or not (default: 1)"
    )
    parser.add_argument(
        "--save-path",
        default="./checkpoint",
        type=str,
        metavar="PATH",
        help="path to save result (default: ./checkpoint)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        type=str,
        help="path to save output files (default: None)",
    )
    parser.add_argument(
        "--load-path",
        default=None,
        type=str,
        metavar="LOAD",
        help="load model or not (default: None, no loading)",
    )

    parser.add_argument(
        "--n-folds",
        default=10,
        type=int,
        help="number of folds for cross-validation (default: 10)",
    )
    parser.add_argument(
        "--record-interval",
        type=int,
        default=5,
        metavar="N",
        help="the interval for recording training status (default: 5)",
    )

    # Optimization settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--valid-batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="valid batch size (default: 1024)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="WD",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-drift",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-diff",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--epochs-drift",
        type=int,
        default=500,
        metavar="N",
        help="number of epochs to train (default: 500)",
    )
    parser.add_argument(
        "--epochs-diff",
        type=int,
        default=500,
        metavar="N",
        help="number of epochs to train (default: 500)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default='cosine',
        help="lr scheduler (default: 'cosine', others not implemented yet)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="number of warmup epochs for cosine scheduler (default: 0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="N",
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--cv-idx",
        type=int,
        default=0,
        help="index of cross-validation fold (default: 0)",
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="max norm for gradient clipping (default: 0.0, no clipping)",
    )

    # Network architecture
    parser.add_argument(
        "--n-layer",
        type=int,
        default=5,
        metavar="N",
        help="number of layers (default: 5)",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=256,
        metavar="N",
        help="number of hidden neuron (default: 256)",
    )
    parser.add_argument(
        "--use-LN",
        type=int,
        default=0,
        help="whether to use layer normalization in the model (default: 0, not use LN; 1, use LN)"
    )

    # Other settings
    parser.add_argument(
        "--time-step",
        type=float,
        default=1e-2,
        help="Time step size (default: 1e-2)",
    )
    parser.add_argument(
        "--include-time",
        type=int,
        default=0,
        help="whether to include time as input feature (default: 0-False, 1-True)"
    )
    parser.add_argument(
        "--time-reg-lambda",
        type=float,
        default=0.0,
        help="the regularization strength for time derivative (default: 0.0, no regularization)"
    )

    parser.add_argument(
        "--ensure-psd",
        action="store_true",
        default=True,
        help="whether to ensure the output diffusion matrix is PSD (default: True)"
    )

    # SWAG settings
    parser.add_argument(
        "--save-swag",
        type=int,
        default=1,
        help="whether to save SWAG after training (default: 1)"
    )
    parser.add_argument(
        "--lr-swag",
        type=float,
        default=1e-3,
        help="the learning rate for SWAG (default: 1e-3)"
    )
    parser.add_argument(
        "--snap-every",
        type=int,
        default=3,
        help="the frequency of SWAG snapshots (default: 3)"
    )


    # Task specific settings
    # For synthetic data, we can specify the sampling interval, total duration, and total transitions to control the difficulty of the learning task.
    parser.add_argument(
        '--sampling-interval',
        type=int,
        default=1,
        help="the sampling interval for the synthetic data (default: 1)"
    )
    parser.add_argument(
        '--total-duration',
        type=int,
        default=100,
        help="the total duration of the synthetic data trajectories (default: 100)"
    )
    parser.add_argument(
        '--total-transitions',
        type=int,
        default=int(1e5),
        help="the total number of transitions for the synthetic data trajectories (default: 1e5)"
    )
    parser.add_argument(
        '--protocol',
        type=str,
        default='A',
        help="the protocol for synthetic data generation: 'A' (fixed sampling interval, varying total duration), 'B' (fixed total duration & number of transitions, varying sampling interval) (default: 'A')"
    )

    # For language data
    parser.add_argument(
        '--domain-indicator',
        type=type_or_none(int),
        default=None,
        help="the domain indicator for the language data, either -1 (emo) or 1 (rat) (default: None, not used)"
    )
    parser.add_argument(
        '--n-language-value-ids',
        type=int,
        default=1,
        help="the number of language value ids to use as input features for the language data (default: 1; 'z_score' only. 2; 'z_score_fic', 'z_score_excl_fic')"
    )

    # For real world data, we can specify the start and end year to control the time range of the data used for training and evaluation.
    parser.add_argument(
        '--start-year',
        type=type_or_none(int),
        default=None,
        help="the start year for the data (default: None, use all years)"
    )
    parser.add_argument(
        '--end-year',
        type=type_or_none(int),
        default=None,
        help="the end year for the data (default: None, use all years)"
    )

    return parser



# ---------- Dataframe utils ----------

def bootstrap_global_metric(
    res_df,
    value_col,
    year_col="year",
    cluster_col="id",
    start_year=None,
    end_year=None,
    B=10_000,
    min_n=8,
    seed=42,
    cumulative=False,
    require_complete_cohort=True,
):
    """Cluster-bootstrap time-specific means and optional cumulative sums."""

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Prepare cluster × time matrix
    # ------------------------------------------------------------
    df = (
        res_df[[cluster_col, year_col, value_col]]
        .dropna()
        .copy()
    )

    if df.empty:
        raise ValueError(f"No valid observations for '{value_col}'.")

    df[year_col] = df[year_col].astype(int)

    start_year = df[year_col].min() if start_year is None else start_year
    end_year = df[year_col].max() if end_year is None else end_year

    df = df.loc[
        df[year_col].between(start_year, end_year)
    ]

    df = (
        df.groupby(
            [cluster_col, year_col],
            observed=True,
        )[value_col]
        .mean()
        .reset_index()
    )

    years = np.sort(df[year_col].unique())

    pivot = (
        df.pivot(
            index=cluster_col,
            columns=year_col,
            values=value_col,
        )
        .reindex(columns=years)
    )

    # ------------------------------------------------------------
    # Cluster-bootstrap column means
    # ------------------------------------------------------------
    def bootstrap_means(data):
        values = data.to_numpy(dtype=float)

        if np.isinf(values).any():
            raise ValueError(
                f"'{value_col}' contains infinite values."
            )

        observed = ~np.isnan(values)
        values_filled = np.where(observed, values, 0.0)

        n_cluster = len(data)

        weights = rng.multinomial(
            n_cluster,
            np.full(n_cluster, 1 / n_cluster),
            size=B,
        )

        numerator = weights @ values_filled
        denominator = weights @ observed.astype(float)

        return np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, np.nan),
            where=denominator > 0,
        )

    # ------------------------------------------------------------
    # Local metric
    # ------------------------------------------------------------
    local_obs = pivot.mean(axis=0).to_numpy()
    local_n = pivot.notna().sum(axis=0).to_numpy()

    valid = local_n >= min_n
    local_obs[~valid] = np.nan

    boot_local = bootstrap_means(pivot)
    boot_local[:, ~valid] = np.nan

    result = {
        "years": years,
        "local_obs": local_obs,
        "local_n": local_n,
        "boot_local": boot_local,
        "local_lo": np.nanpercentile(boot_local, 2.5, axis=0),
        "local_hi": np.nanpercentile(boot_local, 97.5, axis=0),
        "all_ids": pivot.index.to_numpy(),
        "n_all_ids": len(pivot),
        "value_col": value_col,
        "B": B,
        "min_n": min_n,
        "cumulative": cumulative,
        "bootstrap_type": "cluster",
    }

    # ------------------------------------------------------------
    # Optional cumulative metric
    # ------------------------------------------------------------
    if not cumulative:
        return result

    if require_complete_cohort:
        cohort_ids = pivot.index[
            pivot.notna().all(axis=1)
        ]
        cohort_name = "complete_fixed_cohort"
    else:
        cohort_ids = pivot.index[
            pivot.iloc[:, 0].notna()
        ]
        cohort_name = "observed_at_start"

    if len(cohort_ids) == 0:
        raise ValueError(
            "No clusters satisfy the cumulative cohort condition."
        )

    cohort = pivot.loc[cohort_ids]
    cumulative_local_obs = cohort.mean(axis=0).to_numpy()

    if np.isnan(cumulative_local_obs).any():
        raise ValueError(
            "The cumulative cohort has missing time-specific means. "
            "Use require_complete_cohort=True."
        )

    boot_cumulative_local = bootstrap_means(cohort)

    if np.isnan(boot_cumulative_local).any():
        raise ValueError(
            "Some cumulative bootstrap replicates contain missing means."
        )

    cumulative_obs = np.cumsum(cumulative_local_obs)
    boot_cumulative = np.cumsum(
        boot_cumulative_local,
        axis=1,
    )

    result.update({
        "cumulative_local_obs": cumulative_local_obs,
        "cumulative_obs": cumulative_obs,
        "cumulative_n": cohort.notna().sum(axis=0).to_numpy(),
        "boot_cumulative_local": boot_cumulative_local,
        "boot_cumulative": boot_cumulative,
        "cumulative_lo": np.percentile(
            boot_cumulative, 2.5, axis=0
        ),
        "cumulative_hi": np.percentile(
            boot_cumulative, 97.5, axis=0
        ),
        "cumulative_ids": cohort_ids.to_numpy(),
        "n_cumulative_ids": len(cohort_ids),
        "cumulative_cohort": cohort_name,
    })

    return result


def masked_trajs_to_dataframe(
    trajs,
    mask,
    rel_cols,
    base_ids,
    sim_rep,
    initial_times,
    dt=100,
    base_id_col="base_id",
    sim_id_col="sim_id",
    sim_rep_col="sim_rep",
    step_col="step",
    time_col="year",
):
    """
    Convert masked generated trajectories to a long-format dataframe.

    Parameters
    ----------
    trajs : torch.Tensor
        Generated trajectories, shape [N_total, T, D].

    mask : torch.BoolTensor
        Boolean mask, shape [N_total, T].

    rel_cols : list[str]
        State-variable column names. Length must equal D.

    base_ids : list-like
        Base trajectory IDs, length N_total.

    sim_rep : torch.Tensor or array-like
        Simulation replicate index, length N_total.

    initial_times : torch.Tensor or array-like
        Initial absolute time for each trajectory, length N_total.

    dt : float
        Time increment per simulation step. For Seshat year scale, dt=100.

    Returns
    -------
    pandas.DataFrame
        Columns:
            base_id, sim_id, sim_rep, step, year, rel_cols...
    """
    if trajs.shape[:2] != mask.shape:
        raise ValueError(
            f"trajs.shape[:2]={trajs.shape[:2]} and mask.shape={mask.shape} do not match."
        )
    if trajs.ndim != 3:
        raise ValueError(f"trajs must have shape [N_total, T, D], got {trajs.shape}.")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor.")

    N_total, _, D = trajs.shape
    if len(rel_cols) != D:
        raise ValueError(f"len(rel_cols)={len(rel_cols)} but D={D}.")

    base_ids = np.asarray(base_ids, dtype=object)
    # Convert sim_rep and initial_times to numpy arrays if they are torch tensors
    if isinstance(sim_rep, torch.Tensor):
        sim_rep = sim_rep.detach().cpu().numpy()
    else:
        sim_rep = np.asarray(sim_rep)
    if isinstance(initial_times, torch.Tensor):
        initial_times = initial_times.detach().cpu().numpy()
    else:
        initial_times = np.asarray(initial_times)

    # Validate lengths of base_ids, sim_rep, and initial_times
    if len(base_ids) != N_total:
        raise ValueError(f"len(base_ids)={len(base_ids)} but N_total={N_total}.")
    if len(sim_rep) != N_total:
        raise ValueError(f"len(sim_rep)={len(sim_rep)} but N_total={N_total}.")
    if len(initial_times) != N_total:
        raise ValueError(f"len(initial_times)={len(initial_times)} but N_total={N_total}.")

    traj_idx, step_idx = mask.nonzero(as_tuple=True)

    traj_idx_np = traj_idx.detach().cpu().numpy()
    step_idx_np = step_idx.detach().cpu().numpy()

    states = trajs[traj_idx, step_idx].detach().cpu().numpy()

    row_base_ids = base_ids[traj_idx_np]
    row_sim_rep = sim_rep[traj_idx_np]
    row_year = initial_times[traj_idx_np].astype(float) + step_idx_np.astype(float) * float(dt)

    row_sim_ids = np.asarray(
        [f"{bid}__sim{rep}" for bid, rep in zip(row_base_ids, row_sim_rep)],
        dtype=object,
    )

    data = {
        base_id_col: row_base_ids,
        sim_id_col: row_sim_ids,
        sim_rep_col: row_sim_rep,
        step_col: step_idx_np,
        time_col: row_year,
    }

    for j, col in enumerate(rel_cols):
        data[col] = states[:, j]

    return pd.DataFrame(data)


def build_obs_mask_from_dataframe(
    data_frame,
    id_col,
    time_col,
    base_ids=None,
    dt=None,
    n_steps=None
    ) -> dict:
    """
    Build an observation mask from a long-format dataframe.

    The returned mask indicates which simulation steps correspond to observed
    data points for each trajectory.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        Long-format dataframe containing trajectory IDs and observation times.

    id_col : str
        Column identifying each trajectory.

    time_col : str
        Column containing either:
            - simulation step indices, if dt is None, or
            - raw observation times, if dt is provided.
    
    base_ids : list-like or None
        The trajectory IDs in the same order as initial_points.
        If provided, obs_mask rows follow this order exactly.
        If None, rows follow the order in which IDs appear in the dataframe.

    dt : float or None
        If None:
            time_col is assumed to already contain relative simulation step indices
            such as 0, 1, 3, 7, ...
        If provided:
            time_col is treated as raw time, and step indices are computed as
            round((time - first_time_of_trajectory) / dt).

    n_steps : int or None
        If None, the number of steps is inferred from the data.
        If provided, it overrides the inferred number of steps.

    Returns
    -------
    dict
        {
            'obs_mask': torch.BoolTensor of shape [N_traj, T],
            'obs_indices_list': list of torch.LongTensor,
            'id_order': list of trajectory IDs,
            'n_steps': int
        }
    """
    required_cols = [id_col, time_col]
    missing_cols = [c for c in required_cols if c not in data_frame.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data_frame: {missing_cols}")

    if dt is not None:
        dt = float(dt)
        if dt <= 0:
            raise ValueError("dt must be positive if provided.")

    df = data_frame[required_cols].dropna().copy()

    if base_ids is None:
        base_ids = list(df[id_col].drop_duplicates())
    else:
        base_ids = list(base_ids)
    if len(base_ids) == 0:
        raise ValueError("base_ids is empty.")

    obs_indices_list = []
    max_idx = -1

    for traj_id in base_ids:
        g = df[df[id_col] == traj_id].sort_values(time_col)

        times = g[time_col].to_numpy(dtype=float)

        if len(times) == 0:
            raise ValueError(f"Trajectory {traj_id!r} has no valid time values.")

        if dt is None:
            # time_col is already assumed to be a simulation step index.
            rel_steps = times
        else:
            # time_col is raw time. Use the first observation as simulation step 0.
            rel_steps = (times - times[0]) / dt

        obs_idx = np.rint(rel_steps).astype(np.int64)

        # Check that times lie on the simulation grid.
        if not np.allclose(rel_steps, obs_idx, atol=1e-6):
            raise ValueError(
                f"Trajectory {traj_id}: time values are not aligned to the simulation grid. "
                f"rel_steps={rel_steps}, rounded={obs_idx}"
            )

        if obs_idx.min() < 0:
            raise ValueError(
                f"Trajectory {traj_id}: negative observation step found. "
                f"obs_idx={obs_idx}"
            )

        # Duplicate indices usually indicate duplicate observations or an inconsistent time grid.
        if len(np.unique(obs_idx)) != len(obs_idx):
            raise ValueError(
                f"Trajectory {traj_id}: duplicate observation step indices found. "
                f"obs_idx={obs_idx}"
            )

        obs_idx_t = torch.as_tensor(obs_idx, dtype=torch.long)
        obs_indices_list.append(obs_idx_t)

        max_idx = max(max_idx, int(obs_idx.max()))

    if max_idx < 0:
        raise ValueError("No valid observation times found.")

    n_traj = len(base_ids)
    if n_steps is None:
        n_steps = max_idx + 1

    obs_mask = torch.zeros(n_traj, n_steps, dtype=torch.bool)

    for i, obs_idx_t in enumerate(obs_indices_list):
        obs_mask[i, obs_idx_t] = True

    return {
        "obs_mask": obs_mask,
        "obs_indices_list": obs_indices_list,
        "id_order": base_ids,
        "n_steps": n_steps,
    }
    


def repeat_inputs(
    initial_points,
    n_reps,
    metadata=None,
    ) -> dict:
    """
    Repeat initial points and per-trajectory metadata for simulation.

    Ordering is rep-major, consistent with:
        initial_points.repeat(n_reps, ...)

    For N base trajectories and R repetitions, the repeated order is:
        rep 0: base 0, base 1, ..., base N-1
        rep 1: base 0, base 1, ..., base N-1
        ...
        rep R-1: base 0, base 1, ..., base N-1

    Parameters
    ----------
    initial_points : torch.Tensor
        Tensor of shape [N, ...].

    n_reps : int
        Number of repeated simulations per initial point.

    metadata : dict or None
        Per-trajectory metadata. Each value must have length/shape[0] == N.
        Supported types:
            - torch.Tensor with shape [N, ...]
            - np.ndarray with shape [N, ...]
            - list/tuple with len == N
        Each metadata entry is repeated in the same rep-major order as initial_points.
    Returns
    -------
    res : dict
        Dictionary containing repeated initial points and repeated metadata.
        - res['initial_points']: Tensor of shape [N * n_reps, ...].
        - res['sim_rep']: Tensor of shape [N * n_reps], with values in [0, n_reps-1] indicating the repetition index for each row.
    """
    if metadata is None:
        metadata = {}

    if not isinstance(initial_points, torch.Tensor):
        raise TypeError("initial_points must be a torch.Tensor.")

    n_reps = int(n_reps)
    if n_reps <= 0:
        raise ValueError("n_reps must be positive.")

    n_base = initial_points.shape[0]
    device = initial_points.device

    res = {}

    # Repeat initial points along the first dimension.
    repeat_shape = (n_reps,) + (1,) * (initial_points.ndim - 1)
    res["initial_points"] = initial_points.repeat(*repeat_shape)
    res["sim_rep"] = torch.arange(n_reps, device=device).repeat_interleave(n_base)

    for k, v in metadata.items():
        if isinstance(v, torch.Tensor):
            if v.shape[0] != n_base:
                raise ValueError(
                    f"metadata['{k}'] has shape[0]={v.shape[0]}, "
                    f"but expected {n_base}."
                )

            repeat_shape = (n_reps,) + (1,) * (v.ndim - 1)
            res[k] = v.repeat(*repeat_shape)

        elif isinstance(v, np.ndarray):
            if v.shape[0] != n_base:
                raise ValueError(
                    f"metadata['{k}'] has shape[0]={v.shape[0]}, "
                    f"but expected {n_base}."
                )

            tile_shape = (n_reps,) + (1,) * (v.ndim - 1)
            res[k] = np.tile(v, tile_shape)

        elif isinstance(v, (list, tuple)):
            if len(v) != n_base:
                raise ValueError(
                    f"metadata['{k}'] has len={len(v)}, "
                    f"but expected {n_base}."
                )

            res[k] = [
                copy.deepcopy(item)
                for _ in range(n_reps)
                for item in v
            ]

        else:
            raise TypeError(
                f"metadata['{k}'] has unsupported type {type(v)}. "
                "Use torch.Tensor, np.ndarray, list, or tuple with first dimension/length N."
            )

    return res



# ---------- Irregular-time ACF/VACF with confidence intervals ----------
def acf_with_ci_irregular(
    data_frame,
    id_col,
    time_col,
    value_cols,
    K=20,
    method='ACF',
    tau_step=None,
    match_tol=None,
    min_pairs=2,
    min_trajectories=2,
    nan_for_insufficient=True,
    dtype=torch.float64,
    device=None,
    z=1.9599639846,  # z-score for 95% confidence interval
    eps=1e-18,
    verbose=False,
):
    """
    Irregular-time ACF/VACF from a raw dataframe.

    This function uses trajectory-level aggregation:
        1. Compute ACF/VACF separately for each trajectory.
        2. Average trajectory-level correlations across trajectories.
        3. Compute confidence intervals from trajectory-to-trajectory variability.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        Raw long-format dataframe.

    id_col : str
        Column identifying each trajectory, e.g. 'NGA', 'country', 'iso3'.

    time_col : str
        Column containing observation times.
        Due to numerical precision, ensure that time values are not too large or too close together relative to tau_step and match_tol.

    value_cols : list[str] or str
        Columns whose ACF/VACF should be computed.

    K : int
        Maximum lag index. Returns K+1 lag points including lag 0.

    method : {'ACF', 'VACF'}
        - 'ACF'  : compute autocorrelation of value_cols.
        - 'VACF' : treat value_cols as states, first construct finite-difference
                   velocities, then compute ACF of velocities.

    tau_step : float or None
        Lag spacing. If None, uses median positive time difference after preprocessing.

    match_tol : float or None
        Tolerance for matching t_i + tau_k to the nearest observed time.
        If None, uses 0.5 * tau_step.

    min_pairs : int
        Minimum number of matched pairs within a trajectory required to define
        that trajectory's ACF at a given lag.

    min_trajectories : int
        Minimum number of contributing trajectories required to define CI.
        The ACF itself is returned as long as at least one trajectory contributes,
        unless nan_for_insufficient is modified manually.

    nan_for_insufficient : bool
        If True:
            - acf is NaN where no trajectory contributes.
            - CI is NaN where fewer than min_trajectories contribute.

    Returns
    -------
    dict
        Keys below contain the returned tensors.

    acf : torch.Tensor, shape [K+1, D]
        Mean trajectory-level ACF/VACF.

    se : torch.Tensor, shape [K+1, D]
        Standard error across contributing trajectories.

    ci_lo : torch.Tensor, shape [K+1, D]
        Lower confidence interval.

    ci_hi : torch.Tensor, shape [K+1, D]
        Upper confidence interval.

    cnt_traj : torch.Tensor, shape [K+1, D]
        Number of trajectories contributing to each lag and dimension.

    cnt_pairs : torch.Tensor, shape [K+1, D]
        Total number of matched pairs contributing to each lag and dimension.

    tau_grid : torch.Tensor, shape [K+1]
        Lag times.
    """
    method = method.upper()
    if method not in ('ACF', 'VACF'):
        raise ValueError("method must be either 'ACF' or 'VACF'.")

    if isinstance(value_cols, str):
        value_cols = [value_cols]
    value_cols = list(value_cols)

    required_cols = [id_col, time_col] + value_cols
    missing_cols = [c for c in required_cols if c not in data_frame.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data_frame: {missing_cols}")

    if device is None:
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 1. Convert dataframe into a list of trajectory tensors.
    # ------------------------------------------------------------------
    trajectories = []
    all_positive_dts = []

    df = data_frame[required_cols].copy()
    df = df.dropna(subset=required_cols)

    for traj_id, g in df.groupby(id_col, sort=False):
        g = g.sort_values(time_col)

        # Duplicate times are not meaningful for nearest-time matching.
        # Average value columns for duplicate times and keep one entry per time point.
        if g[time_col].duplicated().any():
            if verbose:
                print(f"Duplicate time points found in trajectory {traj_id}.")
                print(f"Averaging values for duplicate times and keeping one entry per time point.")
            g = g.groupby(time_col, as_index=False)[value_cols].mean()            

        if len(g) < 2:
            continue

        times_np = g[time_col].to_numpy(dtype=float)
        values_np = g[value_cols].to_numpy(dtype=float)

        times = torch.as_tensor(times_np, dtype=dtype, device=device)
        values = torch.as_tensor(values_np, dtype=dtype, device=device)
        dt = times[1:] - times[:-1]

        if method == 'ACF':
            proc_times = times
            proc_values = values

        else:  # method == 'VACF'
            dt = times[1:] - times[:-1]               # [T-1]
            dx = values[1:] - values[:-1]             # [T-1, D]

            proc_values = dx / dt.reshape(-1, 1)      # [T-1, D]
            proc_times = times[:-1]

            if proc_values.shape[0] < 2:
                continue

        proc_dt = proc_times[1:] - proc_times[:-1]
        positive_dt = proc_dt[proc_dt > 0]
        if positive_dt.numel() > 0:
            all_positive_dts.append(positive_dt)

        trajectories.append((traj_id, proc_times, proc_values))

    if len(trajectories) == 0:
        raise ValueError("No valid trajectories were found after preprocessing.")

    # ------------------------------------------------------------------
    # 2. Choose tau_step and match_tol.
    # ------------------------------------------------------------------
    if tau_step is None:
        if len(all_positive_dts) == 0:
            raise ValueError("Cannot infer tau_step because no positive time differences were found.")

        all_positive_dts = torch.cat(all_positive_dts)
        tau_step = torch.median(all_positive_dts).item()

        if verbose:
            print(f"Using default tau_step = median positive dt = {tau_step:.6g}")

    tau_step = float(tau_step)
    if tau_step <= 0:
        raise ValueError("tau_step must be positive.")

    if match_tol is None:
        match_tol = 0.5 * tau_step

    match_tol = float(match_tol)
    if match_tol < 0:
        raise ValueError("match_tol must be non-negative.")

    D = len(value_cols)
    tau_grid = torch.arange(K + 1, dtype=dtype, device=device) * tau_step

    # ------------------------------------------------------------------
    # 3. Accumulate trajectory-level rho values.
    # ------------------------------------------------------------------
    sum_r = torch.zeros(K + 1, D, dtype=dtype, device=device)
    sumsq_r = torch.zeros(K + 1, D, dtype=dtype, device=device)

    cnt_traj = torch.zeros(K + 1, D, dtype=torch.long, device=device)
    cnt_pairs = torch.zeros(K + 1, D, dtype=torch.long, device=device)

    for traj_id, t, x in trajectories:
        T = x.shape[0]
        if T < 2:
            continue

        mu = x.mean(dim=0)
        var = ((x - mu) ** 2).mean(dim=0)

        valid_dim = var > eps
        if not torch.any(valid_dim):
            if verbose:
                print(f"Skipping trajectory {traj_id}: all dimensions have near-zero variance.")
            continue

        rho_s = torch.full((K + 1, D), float("nan"), dtype=dtype, device=device)
        pair_s = torch.zeros(K + 1, D, dtype=torch.long, device=device)

        # k = 0 is exactly 1 for dimensions with nonzero variance.
        rho_s[0, valid_dim] = 1.0
        pair_s[0, valid_dim] = T

        ii = torch.arange(T, device=device)

        for k in range(1, K + 1):
            tau = tau_grid[k]
            targets = t + tau

            # searchsorted gives the first index j with t[j] >= target.
            j = torch.searchsorted(t, targets)

            # Nearest neighbor candidate: j or j-1.
            j0 = torch.clamp(j, 0, T - 1)
            j1 = torch.clamp(j - 1, 0, T - 1)

            dt0 = torch.abs(t[j0] - targets)
            dt1 = torch.abs(t[j1] - targets)

            use0 = dt0 <= dt1
            jj = torch.where(use0, j0, j1)

            # Must be a future observation and close enough to target time.
            ok = (jj > ii) & (torch.abs(t[jj] - targets) <= match_tol)

            n_pairs = int(ok.sum().item())
            if n_pairs < min_pairs:
                continue

            x1 = x[ok]
            x2 = x[jj[ok]]

            cov = ((x1 - mu) * (x2 - mu)).mean(dim=0)
            rho = cov / var.clamp_min(eps)

            finite_dim = torch.isfinite(rho) & valid_dim
            rho_s[k, finite_dim] = rho[finite_dim]
            pair_s[k, finite_dim] = n_pairs

        # Add this trajectory's rho to global trajectory-level aggregates.
        finite = torch.isfinite(rho_s)

        sum_r[finite] += rho_s[finite]
        sumsq_r[finite] += rho_s[finite] ** 2
        cnt_traj[finite] += 1
        cnt_pairs += pair_s

    # ------------------------------------------------------------------
    # 4. Compute mean and trajectory-level confidence intervals.
    # ------------------------------------------------------------------
    cnt_f = cnt_traj.to(dtype)

    denom = torch.clamp(cnt_f, min=1.0)
    acf = sum_r / denom

    # Sample variance across trajectories.
    var_num = sumsq_r - (sum_r ** 2) / denom
    var_den = torch.clamp(cnt_f - 1.0, min=1.0)

    var_traj = (var_num / var_den).clamp_min(0.0)
    se = torch.sqrt(var_traj / denom)

    ci_lo = acf - z * se
    ci_hi = acf + z * se

    if nan_for_insufficient:
        no_acf = cnt_traj == 0
        no_ci = cnt_traj < int(min_trajectories)

        acf = acf.masked_fill(no_acf, float("nan"))
        ci_lo = ci_lo.masked_fill(no_ci, float("nan"))
        ci_hi = ci_hi.masked_fill(no_ci, float("nan"))

    return {'acf': acf, 'se': se, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'cnt_traj': cnt_traj, 'cnt_pairs': cnt_pairs, 'tau_grid': tau_grid}


# ---------- Checkpoint utils ----------
def save_checkpoint(state, save_path, epoch, is_best=False, model_type='drift'):
    if is_best:
        torch.save(state, path.join(save_path, "model_best_%s.pth.tar" %(model_type)))
    else:
        torch.save(state, path.join(save_path, "checkpoint_%s_%d.pth.tar" %(model_type, epoch)))


# ---------- SWAG utils ----------
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def collect_swag_snapshots(
    model, train_loader, valid_loader, model_type, device = None,
    # training hyperparameters
    time_step=1e-2, time_reg_lambda=0.0, grad_clip=0.0, include_time=False, 
    # SWAG hyperparameters
    lr_swag=1e-4, momentum=0.9, weight_decay=5e-4,
    n_snaps=50, burn_in_epochs=5, snap_every=3,
    diff_tol=5e-2, start_from_swa_state=None
    ):
    """
        Collect SWAG snapshots with SGD for Bayesian model averaging.

        Args:
            model: the neural network model
            train_loader: data loader for training data
            valid_loader: data loader for validation data
            model_type: type of model ('drift' or 'diff')
            device: torch device
            lr_swag: SGD learning rate used during SWAG collection
            momentum: SGD momentum
            weight_decay: L2 weight decay for parameters (except biases/norms)
            n_snaps: number of snapshot to collect
            burn_in_epochs: number of warmup epochs before taking snapshots
            snap_every: snapshot after the first post-burn-in epoch, then every N epochs
            diff_tol: relative weight drift tolerance vs. initial weights to trigger LR decay
            start_from_swa_state: optional state_dict to initialize model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 0
    m, m2, t = None, None, 0  # Running parameter mean, second moment, and snapshot count.
    k_std = 2.5 

    model.to(device)
    if start_from_swa_state is not None:
        model.load_state_dict(start_from_swa_state)

    valid_loss = evaluate(model, valid_loader, model_type, device=device)
    print(f"SWAG collection start, Initial valid Loss: {valid_loss:.6f}")

    model.train()

    # Split parameters into those with/without weight decay
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
            
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    # SGD without a scheduler; snapshot checks below can reduce the learning rate.
    optim = torch.optim.SGD(param_groups, lr=lr_swag, momentum=momentum, nesterov=True)

    snaps = []
    w0 = vectorize(model) # initial weights for relative distance

    # Burn-in (warmup) before collecting snapshots
    for _ in range(burn_in_epochs):
        train_epoch(model, 
                    train_loader, 
                    optim, 
                    model_type, 
                    device=device, 
                    time_step=time_step, 
                    time_reg_lambda=time_reg_lambda, 
                    grad_clip=grad_clip, 
                    include_time=include_time, 
                    )



    min_lr = lr_swag * 0.01
    # Run training and periodically record snapshots
    for epoch in range(n_snaps * snap_every):
        _ = train_epoch(model, 
                        train_loader, 
                        optim, 
                        model_type, 
                        device=device, 
                        time_step=time_step, 
                        time_reg_lambda=time_reg_lambda, 
                        grad_clip=grad_clip, 
                        include_time=include_time, 
                        )


        if epoch % snap_every == 0:
            valid_loss = evaluate(model, valid_loader, model_type, device=device)
            print(f"SWAG collection epoch {epoch+1}/{n_snaps * snap_every}, Valid Loss: {valid_loss:.6f}")

            snaps.append(vectorize(model))
            wt = snaps[-1].to(w0.device, dtype=w0.dtype)

            # Running estimates of mean and variance
            t += 1
            if m is None:
                m = wt.clone()
                m2 = wt.pow(2)
            else:
                delta = wt - m
                m  = m  + delta / t
                m2 = m2 + (wt*wt - m2) / t

            if t >= 10: # wait a few snaps before using variance-based LR reduction
                var = (m2 - m*m).clamp_min(1e-8)
                std = var.sqrt()
                z_rms = ((wt - m) / (std + 1e-8)).pow(2).mean().sqrt()
                if z_rms.item() > k_std:
                    print(f"  > snapshot z-norm {z_rms.item():.2f} > {k_std}, reducing LR {optim.param_groups[0]['lr']:.4e}")
                    for g in optim.param_groups:
                        g["lr"] = max(g["lr"] * 0.5, min_lr)

            if diff_tol is not None and (t % 5) == 0: # check weight drift every few snaps
                rel = torch.norm(wt - w0) / (torch.norm(w0) + 1e-12)
                if rel.item() > diff_tol:
                    # reduce LR to avoid drifting out of basin
                    print(f"  > weight Drift {rel.item():.4f} > {diff_tol}, reducing LR {optim.param_groups[0]['lr']:.4e}")
                    for g in optim.param_groups:
                        g["lr"] = max(g["lr"] * 0.5, min_lr)
    return snaps


@torch.no_grad()
def vectorize(model: torch.nn.Module) -> torch.Tensor:
    # Grab parameters as a flat CPU vector (no aliasing with model params)
    with torch.no_grad():
        v = parameters_to_vector([p.detach().clone() for p in model.parameters()])
        return v.cpu()


@torch.no_grad()
def load_vector(model: torch.nn.Module, vec: torch.Tensor):
    # Write a sampled parameter vector back into the model (device/dtype-safe)
    p0 = next(model.parameters())
    vector_to_parameters(vec.to(device=p0.device, dtype=p0.dtype), model.parameters())


# SWAG stats from snapshots of parameters
def build_swag_stats(snapshots: list[torch.Tensor], rank_k: int = 20, eps: float = 1e-12):
    """
        snapshots: list of flattened parameter vectors collected over the SWA phase (CPU tensors).
        rank_k:    how many recent deviations to keep for low-rank covariance.
    """
    theta = torch.stack(snapshots, 0).float()   # [T, P] on CPU
    mu    = theta.mean(0)                       # SWA mean
    second= (theta * theta).mean(0)
    diag  = (second - mu * mu).clamp_min(eps)   # diagonal variance

    K = min(rank_k, theta.size(0))

    # Use LAST K deviations for trajectory covariance
    S = (theta[-K:] - mu).t().contiguous() if K > 0 else torch.empty(mu.numel(), 0)

    return mu, diag, S  # all on CPU

# Draw a SWAG sample 
def sample_swag(mu: torch.Tensor, diag: torch.Tensor, S: torch.Tensor, scale: float = 1.0, device=None, dtype=None):
    """
        Samples w ~ N(mu, scale * (diag + (1/(K-1)) S S^T)) where S contains K zero-mean deviations.
    """
    if device is None: device = mu.device
    if dtype  is None: dtype  = mu.dtype

    mu   = mu.to(device=device, dtype=dtype)
    diag = diag.to(device=device, dtype=dtype)
    S    = S.to(device=device, dtype=dtype)

    z1 = torch.randn_like(mu)
    w  = mu + (scale * diag).sqrt() * z1

    if S.numel() > 0:
        K = S.shape[1]
        z2 = torch.randn(K, device=device, dtype=dtype)
        # Cov_lowrank ≈ (1/(K-1)) S S^T  ⇒ multiply by z2 / sqrt(K-1)
        w = w + S @ (z2 / (K - 1)**0.5) * (scale ** 0.5)

    return w
