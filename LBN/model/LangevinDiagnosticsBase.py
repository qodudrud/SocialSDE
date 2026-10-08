import numpy as np
import pandas as pd
from scipy.stats import chi2
import math
from tqdm import tqdm

import torch
from torch.linalg import cholesky as torch_cholesky


def add_exogeneity_flags(
    df,
    thr=4.0,
    s_threshold=None,
    dim=None,
    id_col='id',
    time_col='year',
):
    out = df.copy()
    out = out.sort_values([id_col, time_col]).copy()

    # ------------------------------------------------------------
    # Recompute exogeneity scores from d2, if s_threshold is given
    # ------------------------------------------------------------
    if s_threshold is not None:
        if dim is None:
            raise ValueError(
                "dim must be provided when s_threshold is given, "
                "because scores are recomputed from chi-square tail probabilities."
            )

        score_map = {
            's_step': 'd2_step',
            's_bridge': 'd2_bridge',
            's_endpoint': 'd2_endpoint',
        }

        missing = [
            d2_col for d2_col in score_map.values()
            if d2_col not in out.columns
        ]
        if missing:
            raise ValueError(
                "Cannot recompute exogeneity scores because the following "
                f"distance columns are missing: {missing}. "
                "Build the dataframe with the d2 columns included."
            )

        for s_col, d2_col in score_map.items():
            d2 = pd.to_numeric(out[d2_col], errors='coerce')

            valid = d2.notna() & np.isfinite(d2)

            s = pd.Series(np.nan, index=out.index, dtype=float)

            p = chi2.sf(
                np.maximum(d2.loc[valid].to_numpy(dtype=float), 0.0),
                df=dim,
            )
            p = np.clip(p, 1e-300, 1.0)

            s.loc[valid] = -np.log10(p)

            # Apply the NEW clipping threshold
            s.loc[valid] = np.minimum(
                s.loc[valid].to_numpy(),
                float(s_threshold),
            )

            out[s_col] = s

    # ------------------------------------------------------------
    # Exogeneity flags
    # ------------------------------------------------------------
    if 's_step' in out.columns:
        valid = out['s_step'].notna()

        out['ex_flag_step'] = pd.Series(
            pd.NA, index=out.index, dtype='boolean'
        )
        out.loc[valid, 'ex_flag_step'] = (
            out.loc[valid, 's_step'] >= thr
        )

    if 's_bridge' in out.columns:
        valid = out['s_bridge'].notna()

        out['ex_flag_bridge'] = pd.Series(
            pd.NA, index=out.index, dtype='boolean'
        )
        out.loc[valid, 'ex_flag_bridge'] = (
            out.loc[valid, 's_bridge'] >= thr
        )

    if 's_endpoint' in out.columns:
        valid = out['s_endpoint'].notna()

        out['ex_flag_endpoint'] = pd.Series(
            pd.NA, index=out.index, dtype='boolean'
        )
        out.loc[valid, 'ex_flag_endpoint'] = (
            out.loc[valid, 's_endpoint'] >= thr
        )

    # ------------------------------------------------------------
    # Perturbation type
    # ------------------------------------------------------------
    if all(
        col in out.columns
        for col in ['s_step', 's_bridge', 's_endpoint']
    ):
        step = out['ex_flag_step']
        bridge = out['ex_flag_bridge']
        endpoint = out['ex_flag_endpoint']

        out['perturb_type'] = pd.Series(
            'none', index=out.index, dtype='string'
        )

        complete = (
            step.notna()
            & bridge.notna()
            & endpoint.notna()
        )
        out.loc[~complete, 'perturb_type'] = pd.NA

        # Localized: step=True, bridge=False, endpoint=False
        # Recovered:  step=True, bridge=True,  endpoint=False
        # Persistent: step=True, bridge=False, endpoint=True
        # Off-path:   step=True, bridge=True,  endpoint=True
        c_type_flag = complete & step & ~bridge & ~endpoint
        r_type_flag = complete & step & bridge & ~endpoint
        p_type_flag = complete & step & ~bridge & endpoint
        o_type_flag = complete & step & bridge & endpoint

        out.loc[c_type_flag, 'perturb_type'] = 'localized'
        out.loc[r_type_flag, 'perturb_type'] = 'recovered'
        out.loc[p_type_flag, 'perturb_type'] = 'persistent'
        out.loc[o_type_flag, 'perturb_type'] = 'offpath'

    return out


def build_res_dataframe(
    xs,
    dxs,
    dts,
    series_slices,
    valid_id,
    valid_yrs,
    state_cols=None,
    id2name=None,
    step_s_res=None,
    bridge_s_res=None,
    end_s_res=None,
    irrev_res=None,
    year_threshold=None,
    *,
    state_start=0,
    cond_cols=None,
    s_threshold=6.0,
    id_col="id",
    name_col="name",
    irrev_col="irrev_step",
    include_distance=False
):
    """
    Build a compact state-level dataframe for exogeneity analysis.

    Row definition
    --------------
    Each row corresponds to a state, not a transition.

    For a trajectory with T transitions:
        xs[0], xs[1], ..., xs[T-1]
    this function also reconstructs the terminal state:
        xs[T] = xs[T-1] + dxs[T-1]

    Score alignment
    ---------------
    For state x_j:

    - s_step:
        incoming transition surprisal x_{j-1} -> x_j.
        Initial state has NaN.
        Terminal state receives the final transition's s_step.

    - s_bridge:
        bridge surprisal for central observed state x_j.
        Defined only for interior states j = 1, ..., T-1.

    - s_endpoint:
        endpoint surprisal for x_{j-1} -> x_{j+1}.
        Defined only for interior states j = 1, ..., T-1.

    - irrev_step:
        incoming transition irreversibility for x_{j-1} -> x_j.
        Alignment is the same as s_step.

    Gap columns
    -----------
    - gap_years:
        incoming gap length for x_{j-1} -> x_j.
        Initial state has NaN.

    - bridge_gap_max_years:
        max of the previous and next gaps used by bridge/endpoint.
        Defined only for interior states.
    - long_gap:
        True when the incoming gap_years exceeds year_threshold; the
        bridge_gap_max_years column is not used for this flag.

    Parameters
    ----------
    s_threshold:
        If not None, clip s_step, s_bridge, and s_endpoint at this value.
        Irreversibility is not clipped.
    """
    def _to_numpy(a):
        if hasattr(a, "detach"):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def _is_nested_list_like(a):
        if len(a) == 0:
            return False
        first = a[0]
        if isinstance(first, (str, bytes)):
            return False
        return isinstance(first, (list, tuple, np.ndarray))

    def _get_flat_metric(res, key, N, *, name=None):
        """
        Return a flat metric array of length N.

        Accepted input forms:
        1. None
           -> all NaN
        2. dict
           -> use res[key]
        3. flat array-like
           -> use directly

        This assumes the metric is aligned with flattened transition indices, i.e. the same index system as xs, dxs, dts.
        """
        metric_name = name or key

        if res is None:
            return np.full(N, np.nan, dtype=float)

        if isinstance(res, dict):
            if key not in res:
                return np.full(N, np.nan, dtype=float)

            arr = res[key]
        else:
            arr = res

        arr = _to_numpy(arr).astype(float).reshape(-1)

        if arr.shape[0] != N:
            raise ValueError(
                f"{metric_name} has length {arr.shape[0]}, "
                f"but expected length {N}."
            )

        return arr

    def _clip_s(x):
        if s_threshold is None or not np.isfinite(x):
            return x
        return min(float(x), float(s_threshold))

    def _lookup_name(id2name, entity_id):
        if id2name is None:
            return None

        key = entity_id.item() if hasattr(entity_id, "item") else entity_id
        if hasattr(id2name, "get"):
            name = id2name.get(key, None)
            if name is None:
                name = id2name.get(str(key), None)
            return name
        try:
            return id2name[key]
        except Exception:
            return None

    def _get_series_id(valid_id, series_id, sl):
        ids = list(valid_id)

        # trajectory-level ids
        if len(ids) == len(series_slices) and not _is_nested_list_like(ids):
            return ids[series_id]

        # flattened ids
        if len(ids) == N:
            return ids[sl.start]

        raise ValueError(
            "valid_id should be either trajectory-level ids "
            "or flattened ids with length N."
        )

    def _get_series_years(valid_yrs, series_id, sl, dts_np):
        yrs_obj = list(valid_yrs)
        L = sl.stop - sl.start  # number of transitions in this trajectory

        # trajectory-level year arrays
        if len(yrs_obj) == len(series_slices) and _is_nested_list_like(yrs_obj):
            yrs = np.asarray(yrs_obj[series_id], dtype=float)

            if len(yrs) >= L + 1:
                # Already includes terminal state year.
                return yrs[:L + 1]

            if len(yrs) == L:
                # xs-aligned years only. Reconstruct terminal year from dts.
                terminal_year = yrs[-1] + dts_np[sl.stop - 1] * dt_to_years
                return np.concatenate([yrs, [terminal_year]])

            raise ValueError(
                f"valid_yrs[{series_id}] has length {len(yrs)}, "
                f"but expected either {L} or {L + 1}."
            )

        # flattened xs-aligned years
        if len(yrs_obj) == N:
            yrs = np.asarray(yrs_obj[sl.start:sl.stop], dtype=float)
            terminal_year = yrs[-1] + dts_np[sl.stop - 1] * dt_to_years
            return np.concatenate([yrs, [terminal_year]])

        raise ValueError(
            "valid_yrs should be either trajectory-level year arrays "
            "or flattened years with length N."
        )

    xs_np = _to_numpy(xs)
    dxs_np = _to_numpy(dxs)
    dts_np = _to_numpy(dts).reshape(-1)

    min_yrs = np.min([np.min(np.abs(np.diff(yrs))) for yrs in valid_yrs if len(yrs) > 1])
    min_dts = np.min(np.abs(dts_np))
    dt_to_years = min_yrs / min_dts if min_dts > 0 else 1.0

    if xs_np.ndim != 2:
        raise ValueError("xs should have shape [N, input_dim].")
    if dxs_np.ndim != 2:
        raise ValueError("dxs should have shape [N, dim].")

    N = xs_np.shape[0]
    dim = dxs_np.shape[1]

    if dxs_np.shape[0] != N or dts_np.shape[0] != N:
        raise ValueError("xs, dxs, and dts should have the same first dimension.")

    if state_cols is None:
        state_cols = [f"state_{i}" for i in range(dim)]
    if len(state_cols) != dim:
        raise ValueError(
            f"len(state_cols)={len(state_cols)} does not match dxs dim={dim}."
        )
    
    n_cond = state_start
    if cond_cols is None:
        cond_cols = [f"cond_{i}" for i in range(n_cond)]

    if len(cond_cols) != n_cond:
        raise ValueError(
            f"len(cond_cols)={len(cond_cols)} does not match "
            f"the number of conditioning columns, state_start={n_cond}."
        )

    state_end = state_start + dim
    if state_end > xs_np.shape[1]:
        raise ValueError("state_start + dim exceeds xs.shape[1].")

    s_step_raw = _get_flat_metric(step_s_res, "s_step", N)
    s_bridge_raw = _get_flat_metric(bridge_s_res, "s_bridge", N)
    s_endpoint_raw = _get_flat_metric(end_s_res, "s_endpoint", N)
    if include_distance:
        z2_step_raw = step_s_res["z2_step"]
        d2_step_raw = _get_flat_metric(step_s_res, "d2_step", N)
        d2_bridge_raw = _get_flat_metric(bridge_s_res, "d2_bridge", N)
        d2_endpoint_raw = _get_flat_metric(end_s_res, "d2_endpoint", N)

    irrev_raw = _get_flat_metric(
        irrev_res,
        irrev_col,
        N,
        name=irrev_col,
    )

    rows = []

    for series_id, sl in enumerate(series_slices):
        L = sl.stop - sl.start
        if L <= 0:
            continue

        nga_id = _get_series_id(valid_id, series_id, sl)
        nga_name = _lookup_name(id2name, nga_id)
        yrs = _get_series_years(valid_yrs, series_id, sl, dts_np)

        # Observed start states for all transitions.
        conditions = xs_np[sl.start:sl.stop, :state_start].copy()
        states = xs_np[sl.start:sl.stop, state_start:state_end].copy()

        # Reconstruct terminal state from the final transition.
        terminal_state = states[-1] + dxs_np[sl.stop - 1]
        states_full = np.vstack([states, terminal_state[None, :]])

        # The terminal state retains the condition of the final transition.
        if n_cond > 0:
            conditions_full = np.vstack([
                conditions,
                conditions[-1][None, :],
            ])
        else:
            conditions_full = np.empty((L + 1, 0))

        # Now local state index j runs from 0 to L.
        for j in range(L + 1):
            is_initial = (j == 0)
            is_terminal = (j == L)

            # Original flattened index for this state, if it exists in xs.
            # Terminal state has no corresponding xs row.
            k_state = None if is_terminal else sl.start + j

            # Incoming transition index, if it exists.
            k_in = None if is_initial else sl.start + j - 1

            row = {
                "series_id": series_id,
                id_col: nga_id,
                "year": yrs[j],
                "s_step": np.nan if k_in is None else _clip_s(s_step_raw[k_in]),
                "s_bridge": np.nan,
                "s_endpoint": np.nan,
                irrev_col: np.nan if k_in is None else float(irrev_raw[k_in]),
                "gap_years": (
                    np.nan if k_in is None
                    else float(abs(dts_np[k_in]) * dt_to_years)
                ),
                "bridge_gap_max_years": np.nan,
                "long_gap": False,
                "is_terminal": is_terminal,
            }

            if id2name is not None:
                row[name_col] = nga_name

            for c, val in zip(cond_cols, conditions_full[j]):
                row[c] = val

            for c, val in zip(state_cols, states_full[j]):
                row[c] = val

            # Bridge / endpoint are defined only for interior observed states:
            # j = 1, ..., L-1, corresponding to original index k_state.
            if (not is_initial) and (not is_terminal):
                k = k_state

                row["s_bridge"] = _clip_s(s_bridge_raw[k])
                row["s_endpoint"] = _clip_s(s_endpoint_raw[k])

                gap_prev = abs(dts_np[k - 1]) * dt_to_years
                gap_next = abs(dts_np[k]) * dt_to_years
                row["bridge_gap_max_years"] = float(max(gap_prev, gap_next))

            # Long-gap flag:
            gap_for_flag = row["gap_years"]
            

            row["long_gap"] = (
                False if year_threshold is None or not np.isfinite(gap_for_flag)
                else gap_for_flag > year_threshold
            )

            if include_distance:
                row['z2_step'] = np.nan if k_in is None else z2_step_raw[k_in]
                row['d2_step'] = np.nan if k_in is None else float(d2_step_raw[k_in])
                row['d2_bridge'] = np.nan
                row['d2_endpoint'] = np.nan

                if (not is_initial) and (not is_terminal):
                    row["d2_bridge"] = d2_bridge_raw[k_state]
                    row["d2_endpoint"] = d2_endpoint_raw[k_state]

            rows.append(row)

    df = pd.DataFrame(rows)

    base_cols = [
        "series_id",
        id_col,
    ]

    if id2name is not None:
        base_cols.append(name_col)

    base_cols += [
        "year",
        *cond_cols,
        *state_cols,
        "s_step",
        "s_bridge",
        "s_endpoint",
        irrev_col,
        "gap_years",
        "bridge_gap_max_years",
        "long_gap",
    ]
    if include_distance:
        base_cols += [
            "z2_step",
            "d2_step",
            "d2_bridge",
            "d2_endpoint",
        ]

    return df[base_cols]


class LangevinDiagnosticsBase:
    def __init__(self, args, device=None):
        # Store common geometry, time scale, and device information.
        self.dim = args.dim
        self.input_dim = args.input_dim
        self.time_step = args.time_step

        self.n_cond_onehot = getattr(args, 'n_cond_onehot', 0)
        self.include_time = getattr(args, 'include_time', False)
        self.time_reg_lambda = getattr(args, 'time_reg_lambda', 0.0)

        # Select the device used by diagnostics; subclasses handle their models.
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    # ==== Helper functions for building model inputs and extracting state slices ====
    def _state_slice(self):
        return slice(self.n_cond_onehot, self.n_cond_onehot + self.dim)

    def _replace_state_in_input_batch(self, base_inputs, x_states):
        """
        Build full model inputs by copying base_inputs and replacing only
        the physical-state block with candidate states.

        Args:
            base_inputs: [B, input_dim]
            x_states:    [B, dim]
        Returns:
            cand:        [B, input_dim]
        """
        cand = base_inputs.clone()
        cand[:, self._state_slice()] = x_states
        return cand

    def _as_tensor(self, name, value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                raise TypeError(
                    f"{name} cannot be converted because it has "
                    "dtype=object."
                )
            return torch.as_tensor(value)
        raise TypeError(
            f"{name} must be a torch.Tensor or numpy.ndarray, "
            f"but received {type(value).__name__}."
        )

    def _prepare_transition_inputs(
        self,
        xs,
        dxs,
        dts,
        *,
        check_finite=True,
    ):
        """
        Validate and standardize transition-level inputs.

        Parameters
        ----------
        xs : torch.Tensor or numpy.ndarray, shape [N, input_dim]
            Full model inputs at the beginning of each transition.

        dxs : torch.Tensor or numpy.ndarray, shape [N, dim]
            Physical-state increments.

        dts : torch.Tensor or numpy.ndarray, shape [N] or [N, 1]
            Transition intervals.

        check_finite : bool
            If True, reject NaN and infinite values.

        Returns
        -------
        xs : torch.Tensor, shape [N, input_dim]
            Tensor moved to self.device.

        dxs : torch.Tensor, shape [N, dim]
            Tensor moved to self.device and cast to xs.dtype.

        dts : torch.Tensor, shape [N]
            One-dimensional tensor moved to self.device and cast to xs.dtype.
        """
        # --------------------------------------------------------------
        # 1. Type validation
        # --------------------------------------------------------------
        xs = self._as_tensor("xs", xs)
        dxs = self._as_tensor("dxs", dxs)
        dts = self._as_tensor("dts", dts)

        # --------------------------------------------------------------
        # 2. Shape validation
        # --------------------------------------------------------------
        if xs.ndim != 2 or xs.shape[1] != self.input_dim:
            raise ValueError(
                f"xs must have shape [N, {self.input_dim}], "
                f"but received {tuple(xs.shape)}."
            )

        if dxs.ndim != 2 or dxs.shape[1] != self.dim:
            raise ValueError(
                f"dxs must have shape [N, {self.dim}], "
                f"but received {tuple(dxs.shape)}."
            )

        if dts.ndim == 1:
            pass
        elif dts.ndim == 2 and dts.shape[1] == 1:
            dts = dts[:, 0]
        else:
            raise ValueError(
                "dts must have shape [N] or [N, 1], "
                f"but received {tuple(dts.shape)}."
            )

        N = xs.shape[0]
        if N == 0:
            raise ValueError(
                "Transition inputs must contain at least one transition."
            )

        if dxs.shape[0] != N or dts.shape[0] != N:
            raise ValueError(
                "xs, dxs, and dts must contain the same number of "
                "transitions, but received "
                f"N_xs={N}, N_dxs={dxs.shape[0]}, "
                f"and N_dts={dts.shape[0]}."
            )

        if not torch.is_floating_point(xs):
            raise TypeError(
                "xs must have a floating-point dtype, "
                f"but received dtype={xs.dtype}."
            )

        # --------------------------------------------------------------
        # 3. Device and dtype normalization
        # --------------------------------------------------------------
        xs = xs.to(self.device)
        dxs = dxs.to(
            device=self.device,
            dtype=xs.dtype,
        )
        dts = dts.to(
            device=self.device,
            dtype=xs.dtype,
        )

        # --------------------------------------------------------------
        # 4. Value validation
        # --------------------------------------------------------------
        if check_finite:
            for name, tensor in (
                ("xs", xs),
                ("dxs", dxs),
                ("dts", dts),
            ):
                nonfinite = ~torch.isfinite(tensor)

                if nonfinite.any().item():
                    bad_rows = torch.where(
                        nonfinite.reshape(N, -1).any(dim=1)
                    )[0].detach().cpu().tolist()

                    raise ValueError(
                        f"{name} contains non-finite values. "
                        f"Invalid row indices: {bad_rows[:10]}."
                    )

        nonpositive_dt = dts <= 0

        if nonpositive_dt.any().item():
            bad_idx = torch.where(
                nonpositive_dt
            )[0].detach().cpu().tolist()

            raise ValueError(
                "All transition intervals must be positive. "
                f"Invalid dts found at indices {bad_idx[:10]}."
            )

        return xs, dxs, dts

    # ==== Core methods to be implemented by subclasses: field evaluation ====
    def _diffusion_is_constant(self):
        # Subclasses can override this method to indicate that the diffusion is constant.
        return False

    def eval_field_mean_var(self, pts, n_MC=20, model_type='drift', store_MC_outputs=False, return_cpu=False):
        raise NotImplementedError("eval_field_mean_var must be implemented in subclasses of LangevinDiagnosticsBase.")

    def _eval_field_with_grad(self, pts, model_type='drift'):
        raise NotImplementedError("_eval_field_with_grad must be implemented in subclasses of LangevinDiagnosticsBase.")

    def _correct_diffusion_for_finite_dt(self, D_raw, F, eps=1e-6):
        """
        Convert raw second-moment diffusion estimate
            D_raw ≈ E[Δx Δx^T | x] / (2 dt)
        to covariance diffusion
            D_cov ≈ Cov[Δx | x] / (2 dt).
        This correction uses the common dt = self.time_step, not individual
        transition intervals, and optionally adds eps to the diagonal.
        """
        dt = float(self.time_step)

        C = 0.5 * torch.bmm(F.unsqueeze(-1), F.unsqueeze(1)) * dt
        D_cov = D_raw - C
        D_cov = 0.5 * (D_cov + D_cov.transpose(-1, -2))

        if eps is not None:
            I = torch.eye(self.dim, device=D_cov.device, dtype=D_cov.dtype).unsqueeze(0)
            D_cov = D_cov + eps * I

        return D_cov


    def return_drift_diff_estimate(self, xs, series_slices=None, n_MC=20, return_lists=False, return_cpu=False, adjust_for_drift=False):
        """
        Return averaged drift and diffusion estimates at the supplied inputs.

        Args:
            xs: Full model inputs of shape (N, input_dim).
            series_slices: Optional slices for per-trajectory outputs.
            n_MC: Number of SWAG samples per fold for LBN evaluation.
            return_lists: Add per-trajectory lists when series_slices is provided.
            return_cpu: Detach and move estimates to CPU when True.
            adjust_for_drift: Apply the finite-dt correction using self.time_step.
        Returns:
            Dictionary with drift_est (N, dim) and diff_est (N, dim, dim).
            With return_lists=True and series_slices provided, also includes
            drift_est_list and diff_est_list.
        """
        F_est = self.eval_field_mean_var(xs, n_MC=n_MC, model_type='drift', return_cpu=False)[0]
        D_est = self.eval_field_mean_var(xs, n_MC=n_MC, model_type='diff', return_cpu=False)[0]
        if adjust_for_drift:
            D_est = self._correct_diffusion_for_finite_dt(D_est, F_est)

        if return_cpu:
            F_est = F_est.detach().cpu()
            D_est = D_est.detach().cpu()

        res = {
            "drift_est": F_est,
            "diff_est": D_est
        }
        if return_lists:
            if series_slices is None:
                print("Warning: series_slices is None, returning full drift and diffusion estimates without slicing into trajectories.")
            else:
                F_est_list = []
                D_est_list = []
                for sl in series_slices:
                    F_est_list.append(F_est[sl.start:sl.stop])
                    D_est_list.append(D_est[sl.start:sl.stop])
                res["drift_est_list"] = F_est_list
                res["diff_est_list"] = D_est_list
        return res


    def eval_current_fields(
        self,
        pts,
        adjust_for_drift=False,
        return_cpu=False,
    ):
        """
        Evaluate the drift, diffusion, diffusion divergence, and current-form drift at arbitrary model-input points.

        Computes
            F(x, t),
            D(x, t),
            (div_x D)_i = sum_j partial D_ij / partial x_j,
            F_current = F - div_x D.

        Spatial derivatives are taken only with respect to the physical-state coordinates. 
        Conditioning variables and time coordinates are held fixed.

        Parameters
        ----------
        pts : torch.Tensor, shape [N, input_dim]
            Full model inputs at which the fields are evaluated.

            If include_time=True, pts must already contain the desired time coordinate, such as the midpoint time or a selected plotting time.

        adjust_for_drift : bool
            If True, apply the existing finite-dt diffusion correction before calculating div_x D.

        return_cpu : bool
            If True, return detached CPU tensors. Otherwise, return detached tensors on the current computational device.

        Returns
        -------
        dict
            F : torch.Tensor, shape [N, dim]
                Ito/Kramers-Moyal drift.

            D : torch.Tensor, shape [N, dim, dim]
                Symmetrized diffusion tensor.

            div_D : torch.Tensor, shape [N, dim]
                Spatial divergence of the diffusion tensor.

            current_drift : torch.Tensor, shape [N, dim]
                Current-form drift F - div_x D.

        """
        if pts.ndim != 2 or pts.shape[-1] != self.input_dim:
            raise ValueError(
                f"pts must have shape [N, {self.input_dim}], "
                f"but received {tuple(pts.shape)}."
            )

        # Make the complete input tensor the differentiation variable.
        pts_eval = (
            pts.to(self.device)
            .detach()
            .clone()
            .requires_grad_(True)
        )

        N = pts_eval.shape[0]
        state_sl = self._state_slice()
        state_start = state_sl.start

        with torch.enable_grad():
            F = self._eval_field_with_grad(
                pts_eval,
                model_type="drift",
            )

            D = self._eval_field_with_grad(
                pts_eval,
                model_type="diff",
            )

            expected_F_shape = (N, self.dim)
            expected_D_shape = (N, self.dim, self.dim)

            if tuple(F.shape) != expected_F_shape:
                raise RuntimeError(
                    f"Expected drift shape {expected_F_shape}, "
                    f"but received {tuple(F.shape)}."
                )

            if tuple(D.shape) != expected_D_shape:
                raise RuntimeError(
                    f"Expected diffusion shape {expected_D_shape}, "
                    f"but received {tuple(D.shape)}."
                )

            if adjust_for_drift:
                D = self._correct_diffusion_for_finite_dt(
                    D,
                    F,
                )

            # Suppress small numerical asymmetries.
            D = 0.5 * (
                D + D.transpose(-1, -2)
            )

            # ----------------------------------------------------------
            # Calculate the spatial diffusion divergence
            #
            # (div_x D)_i = sum_j partial D_ij / partial x_j
            # ----------------------------------------------------------
            if not D.requires_grad:
                if self._diffusion_is_constant():
                    # For state-independent diffusion, div_x D = 0 exactly.
                    div_D = torch.zeros_like(F)

                else:
                    raise RuntimeError(
                        "The diffusion output does not retain an autograd graph, "
                        "although the model reports state-dependent diffusion."
                    )
            else:
                div_D_components = []

                n_grad_calls = self.dim * self.dim
                grad_call = 0

                for i in range(self.dim):
                    div_i = torch.zeros(
                        N,
                        device=pts_eval.device,
                        dtype=pts_eval.dtype,
                    )

                    for j in range(self.dim):
                        grad_call += 1

                        grad_Dij = torch.autograd.grad(
                            outputs=D[:, i, j].sum(),
                            inputs=pts_eval,
                            retain_graph=(grad_call < n_grad_calls),
                            create_graph=False,
                            allow_unused=False,
                        )[0]  # [N, input_dim]

                        # Physical coordinate x_j is located at
                        # input column state_start + j.
                        state_col = state_start + j

                        div_i = div_i + grad_Dij[:, state_col]

                    div_D_components.append(div_i)

                div_D = torch.stack(
                    div_D_components,
                    dim=-1,
                )  # [N, dim]

            current_drift = F - div_D

        # The autograd graph is no longer needed after div_D has been computed.
        results = {
            "F": F.detach(),
            "D": D.detach(),
            "div_D": div_D.detach(),
            "current_drift": current_drift.detach(),
        }

        if return_cpu:
            results = {
                key: value.cpu()
                for key, value in results.items()
            }

        return results


    # ===== Irreversibility calculation =====
    def calculate_irreversibility_log_ratio(
        self,
        xs,
        dxs,
        dts,
        series_slices,
        n_MC=20,
        return_lists=True,
        seed=42,
        adjust_for_drift=False,
    ):
        """
        Calculate local transition irreversibility using the direct forward/reverse local-Gaussian transition-density ratio.

        For each observed transition
            x_k -> x_{k+1},
            x_{k+1} = x_k + dx_k,
        define
            sigma_k
                = log q_dt(x_{k+1} | x_k)
                - log q_dt(x_k | x_{k+1}),
        where
            q_dt(x_{k+1} | x_k)
                = N(
                    x_{k+1};
                    x_k + F(x_k) dt,
                    2 D(x_k) dt
                ),
            q_dt(x_k | x_{k+1})
                = N(
                    x_k;
                    x_{k+1} + F(x_{k+1}) dt,
                    2 D(x_{k+1}) dt
                ).

        Thus, sigma_k > 0 means that the observed transition is more probable in its forward direction than in reverse under the inferred dynamics.

        Parameters
        ----------
        xs : torch.Tensor, shape [N, input_dim]
            Model inputs at the beginning of each transition.

        dxs : torch.Tensor, shape [N, dim]
            Observed state increments.

        dts : torch.Tensor, shape [N] or [N, 1]
            Positive transition intervals.

        series_slices : list[slice]
            Slices identifying individual trajectories in the flattened arrays.

        n_MC : int
            Number of SWAG samples per fold.

        return_lists : bool
            If True, also return trajectory-wise arrays.

        seed : int
            Random seed used for SWAG sampling.

        adjust_for_drift : bool
            If True, apply _correct_diffusion_for_finite_dt using the common
            self.time_step, not transition-specific dts. The helper also adds
            its default diagonal regularizer.

        Notes
        -----
        Transition log probabilities use a fixed internal Cholesky jitter of
        1e-8. There is no adaptive-jitter retry or public jitter argument.

        Returns
        -------
        results : dict
            Contains:
                - irrev_step:
                    Flattened local irreversibility values.
                - logp_forward:
                    Flattened forward log transition densities.
                - logp_reverse:
                    Flattened reverse log transition densities.
                - irrev_step_list, logp_forward_list, logp_reverse_list:
                    Trajectory-wise arrays, if return_lists=True.

            All logarithms are natural logarithms, so irreversibility is measured
            in nats.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        xs, dxs, dts = self._prepare_transition_inputs(
            xs,
            dxs,
            dts,
        )
        N = xs.shape[0]

        state_sl = self._state_slice()
        # --------------------------------------------------------------
        # 1. Construct the endpoint input for every observed transition
        # --------------------------------------------------------------
        x_next = xs.clone()
        x_next[:, state_sl] = x_next[:, state_sl] + dxs

        if self.include_time:
            # The final input coordinate is assumed to be time.
            x_next[:, -1] = x_next[:, -1] + dts

        # --------------------------------------------------------------
        # 2. Evaluate F and D at both endpoints
        # Concatenation evaluates both endpoints with the same sampled model weights.
        # --------------------------------------------------------------
        both_inputs = torch.cat([xs, x_next], dim=0)

        field_res = self.return_drift_diff_estimate(
            both_inputs,
            n_MC=n_MC,
            adjust_for_drift=adjust_for_drift,
            return_cpu=False,
        )

        F_all = field_res["drift_est"]
        D_all = field_res["diff_est"]

        F_start = F_all[:N]
        F_end = F_all[N:]

        D_start = D_all[:N]
        D_end = D_all[N:]

        # Explicit symmetrization suppresses small numerical asymmetries.
        D_start = 0.5 * (D_start + D_start.transpose(-1, -2))
        D_end = 0.5 * (D_end + D_end.transpose(-1, -2))

        # --------------------------------------------------------------
        # 4. multivariate-Gaussian log probability
        # --------------------------------------------------------------
        def _transition_log_prob(residual, diffusion, dt, jitter=1e-8):
            eye = torch.eye(
                self.dim,
                device=diffusion.device,
                dtype=diffusion.dtype,
            ).unsqueeze(0)

            chol = torch.linalg.cholesky(diffusion + jitter * eye)

            # L z = r, so ||z||^2 = r^T D^{-1} r.
            whitened = torch.linalg.solve_triangular(
                chol,
                residual.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)

            quad_D = torch.sum(whitened**2, dim=-1)

            logdet_D = 2.0 * torch.log(
                torch.diagonal(chol, dim1=-2, dim2=-1)
            ).sum(dim=-1)

            logdet_cov = (
                logdet_D
                + self.dim * torch.log(2.0 * dt)
            )

            mahalanobis = quad_D / (2.0 * dt)

            return -0.5 * (
                self.dim * math.log(2.0 * math.pi)
                + logdet_cov
                + mahalanobis
            )

        # Forward: dx ~ N(F(x_k) dt, 2 D(x_k) dt)
        residual_forward = dxs - F_start * dts[:, None]

        # Reverse: -dx ~ N(F(x_{k+1}) dt, 2 D(x_{k+1}) dt)
        residual_reverse = -dxs - F_end * dts[:, None]

        logp_forward = _transition_log_prob(
            residual=residual_forward,
            diffusion=D_start,
            dt=dts,
        )

        logp_reverse = _transition_log_prob(
            residual=residual_reverse,
            diffusion=D_end,
            dt=dts,
        )

        irrev_step = logp_forward - logp_reverse

        # --------------------------------------------------------------
        # 5. Convert to NumPy and reconstruct trajectory-wise outputs
        # --------------------------------------------------------------
        irrev_np = irrev_step.detach().cpu().numpy()
        logp_forward_np = logp_forward.detach().cpu().numpy()
        logp_reverse_np = logp_reverse.detach().cpu().numpy()

        results = {
            "irrev_step": irrev_np,
            "logp_forward": logp_forward_np,
            "logp_reverse": logp_reverse_np,
        }

        if return_lists:
            results.update({
                "irrev_step_list": [
                    irrev_np[sl.start:sl.stop]
                    for sl in series_slices
                ],
                "logp_forward_list": [
                    logp_forward_np[sl.start:sl.stop]
                    for sl in series_slices
                ],
                "logp_reverse_list": [
                    logp_reverse_np[sl.start:sl.stop]
                    for sl in series_slices
                ],
            })

        return results


    # ===== Continuous-time Stratonovich path irreversibility =====
    def calculate_irreversibility(
        self,
        xs,
        dxs,
        dts,
        series_slices,
        return_lists=True,
        seed=42,
        adjust_for_drift=False,
        eps=1e-8,
        return_components=False,
    ):
        """
        Approximate continuous-time conditional path irreversibility using

            Sigma[Gamma]
                = integral g(x, t)^T o dx,

        where
            g(x, t)
                = D(x, t)^(-1)
                [F(x, t) - div_x D(x, t)],
        and
            (div_x D)_i
                = sum_j partial D_ij(x, t) / partial x_j.

        Each observed increment is evaluated using the Stratonovich spacetime-midpoint rule:

            sigma_k
                = g(
                    (x_k + x_{k+1}) / 2,
                    (t_k + t_{k+1}) / 2
                )^T
                (x_{k+1} - x_k).

        When include_time=False, this reduces to the time-homogeneous form.

        The returned quantity is the conditional dynamical path-ratio contribution. 

        Parameters
        ----------
        xs : torch.Tensor, shape [N, input_dim]
            Full model inputs at the starting point of each transition.

        dxs : torch.Tensor, shape [N, dim]
            Observed state increments.

        dts : torch.Tensor, shape [N] or [N, 1]
            Positive transition intervals.

        series_slices : list[slice]
            Slices identifying individual trajectories.

        return_lists : bool
            If True, return trajectory-wise irreversibility arrays.

        seed : int
            Random seed.

        adjust_for_drift : bool
            If True, apply the existing finite-dt correction to D before
            calculating div_x D.

        eps : float
            Small diagonal regularizer added before solving with D.

        return_components : bool
            If True, also return div_x D, the effective path drift,
            and the path-force field g.

        Returns
        -------
        dict
            irrev_step:
                Flattened local path-irreversibility contributions.

            irrev_step_list:
                Per-trajectory arrays, if return_lists=True.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        xs, dxs, dts = self._prepare_transition_inputs(
            xs,
            dxs,
            dts,
        )
        N = xs.shape[0]

        state_sl = self._state_slice()

        # --------------------------------------------------------------
        # 1. Construct the spacetime midpoint of every transition
        # --------------------------------------------------------------

        # Physical-state midpoint:
        # x_mid = x_k + 0.5 * dx_k
        # This tensor must require gradients because div_x D is calculated by differentiating the diffusion tensor with respect to x_mid.
        x_mid_states = (
            xs[:, state_sl] + 0.5 * dxs
        ).detach().clone().requires_grad_(True)

        # Start from the original model inputs.
        base_mid_inputs = xs.detach().clone()
        if self.include_time:
            # The final model-input coordinate is time.
            # t_mid = t_k + 0.5 * dt_k
            base_mid_inputs[:, -1] = (
                base_mid_inputs[:, -1] + 0.5 * dts
            )

        # Replace only the physical-state block with x_mid_states.
        # Conditional one-hot variables and other context variables remain unchanged.
        x_mid_inputs = self._replace_state_in_input_batch(
            base_mid_inputs,
            x_mid_states,
        )

        # --------------------------------------------------------------
        # 2. Evaluate F(x_mid, t_mid), D(x_mid, t_mid), div_x D(x_mid, t_mid), and the effective current drift F - div_x D
        # --------------------------------------------------------------
        field_res = self.eval_current_fields(
            pts=x_mid_inputs,
            adjust_for_drift=adjust_for_drift,
            return_cpu=False,
        )

        F_mid = field_res["F"]
        D_mid = field_res["D"]
        div_D = field_res["div_D"]
        current_drift = field_res["current_drift"]

        # --------------------------------------------------------------
        # 4. Calculate g = D^{-1}(F - div_x D)
        # --------------------------------------------------------------
        with torch.no_grad():
            D_num = D_mid.detach()
            current_drift_num = current_drift.detach()

            eye = torch.eye(
                self.dim,
                device=self.device,
                dtype=xs.dtype,
            ).unsqueeze(0)

            D_reg = D_num + eps * eye

            irrev_field = torch.linalg.solve(
                D_reg,
                current_drift_num.unsqueeze(-1),
            ).squeeze(-1)

            # Stratonovich spacetime-midpoint contribution:
            #
            # sigma_k = g(x_mid, t_mid)^T dx_k
            irrev_step = torch.einsum(
                "ni,ni->n",
                irrev_field,
                dxs,
            )

        # --------------------------------------------------------------
        # 5. Package outputs
        # --------------------------------------------------------------
        irrev_np = irrev_step.cpu().numpy()

        results = {
            "irrev_step": irrev_np,
        }

        if return_lists:
            results["irrev_step_list"] = [
                irrev_np[sl.start:sl.stop]
                for sl in series_slices
            ]

        if return_components:
            results.update({
                "div_diffusion": div_D.detach().cpu().numpy(),
                "current_drift": (
                    current_drift.detach().cpu().numpy()
                ),
                "irrev_field": irrev_field.cpu().numpy(),
                "midpoint_state": (
                    x_mid_states.detach().cpu().numpy()
                ),
            })

            if self.include_time:
                results["midpoint_time"] = (
                    base_mid_inputs[:, -1].detach().cpu().numpy()
                )

        return results

    # ==== Noise calculation and local one-step surprisal estimation =====
    def calculate_noises(self, xs, dxs, dts, series_slices, n_MC=20, seed=42, adjust_for_drift=False, return_lists=False):
        """
            Calculate the noise terms for the given trajectories based on the estimated drift and diffusion.
            Args:
                xs: tensor of shape (N, input_dim) representing the states of the trajectories
                dxs: tensor of shape (N, dim) representing the increments of the trajectories
                dts: tensor of shape (N,) representing the time steps for each increment
                series_slices: list of slices representing the indices of each trajectory in the flattened xs and dxs tensors
                n_MC: number of Monte Carlo samples to use for estimating mean and variance (default: 20)
                seed: random seed for reproducibility (default: 42)
                adjust_for_drift: whether to adjust for drift in noise calculation (default: False)
            Returns:
                Dictionary with noises: a detached CPU tensor of shape (N, dim).
                If return_lists=True, also includes per-trajectory noises_list.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        np.random.seed(seed)
        torch.manual_seed(seed)
        
        xs, dxs, dts = self._prepare_transition_inputs(
            xs,
            dxs,
            dts,
        )

        res = self.return_drift_diff_estimate(xs, series_slices=series_slices, n_MC=n_MC, adjust_for_drift=adjust_for_drift)
        F_est, D_est = res["drift_est"], res["diff_est"]
        try:
            B_est = torch_cholesky(2 * D_est)
        except RuntimeError:
            eps = 1e-6
            I = torch.eye(self.dim, device=D_est.device, dtype=D_est.dtype).expand(D_est.shape[0], -1, -1)
            B_est = torch_cholesky(2 * D_est + eps * I)            

        dts = dts.unsqueeze(-1) if dts.ndim == 1 else dts
        v  = (dxs - F_est * dts).to(B_est.dtype)

        # whitening: ε = B^{-1} v  (= inv(cholesky(2D)) @ residual)
        noises = torch.linalg.solve_triangular(B_est, v.unsqueeze(-1), upper=False).squeeze(-1)
        noises = noises / torch.sqrt(dts)  # scale by sqrt(dt) to get the noise term in the SDE

        result = {"noises": noises.detach().cpu()}
        if return_lists:
            noises_list = []
            for sl in series_slices:
                noises_list.append(noises[sl.start:sl.stop].detach().cpu())

            result["noises_list"] = noises_list

        return result
    
    
    def _step_probabilities_from_noise(self, noise, series_slices, return_lists=True):
        """
        Calculate chi-square tail probabilities from standardized noise.
        Args:
            noise: input noise (N, D)
            series_slices: list of slices representing the indices of each trajectory in the flattened noise tensor
            return_lists: if True, add trajectory-wise lists for each output (default: True)
        Returns:
            Dictionary containing:
                - z2_step: element-wise squared standardized noise
                - d2_step: squared Euclidean norm of the standardized noise
                - p_step: chi-square survival probability at d2_step
                - s_step: -log10(p_step)
        """
        z = np.asarray(noise, dtype=float)
        assert z.ndim == 2, "noise must be (N, D)"
        dim = z.shape[1]

        z2 = z**2  # element-wise square
        d2 = np.sum(z2, axis=1)  # ||z||^2
        p_step = chi2.sf(d2, df=dim)  # P(ChiSquare_dim >= ||z||^2)

        p_step = np.clip(p_step, 1e-300, 1.0)
        s_step = -np.log10(p_step)

        result = {"z2_step": z2, "d2_step": d2, "p_step": p_step, "s_step": s_step}

        if return_lists:
            per_traj_z2 = []
            per_traj_d2 = []
            per_traj_p = []
            per_traj_s = []

            for sl in series_slices:
                idx_range = np.arange(sl.start, sl.stop, dtype=int)

                if len(idx_range) == 0:
                    per_traj_z2.append(np.empty((0, dim)))
                    per_traj_d2.append(np.empty((0,)))
                    per_traj_p.append(np.empty((0,)))
                    per_traj_s.append(np.empty((0,)))
                else:
                    per_traj_z2.append(result["z2_step"][idx_range])
                    per_traj_d2.append(result["d2_step"][idx_range])
                    per_traj_p.append(result["p_step"][idx_range])
                    per_traj_s.append(result["s_step"][idx_range])

            result.update({
                "z2_step_list": per_traj_z2,
                "d2_step_list": per_traj_d2,
                "p_step_list": per_traj_p,
                "s_step_list": per_traj_s,
            })

        return result


    def calculate_step_surprisal(self, xs, dxs, dts, series_slices, n_MC=20, seed=42, return_lists=True, adjust_for_drift=False):
        result = self.calculate_noises(xs, dxs, dts, series_slices, n_MC=n_MC, seed=seed, adjust_for_drift=adjust_for_drift, return_lists=return_lists)
        noises = result["noises"]
        return self._step_probabilities_from_noise(noises, series_slices, return_lists=return_lists)
    

    # ===== Two-step optimization for estimating the most likely trajectory between two observed states =====
    def _two_step_terms_from_precomputed(
        self,
        x_states,
        m_f,
        Sigma_f,
        x_next_states,
        dt_next,
        base_mid_inputs,
        eps=1e-6,
        adjust_for_drift=False,
    ):
        """
        Batched two-step path/action terms.

        For each batch element, evaluate the cost of a candidate intermediate
        state z connecting x_start -> z -> x_end under the local Gaussian
        SDE approximation.

        For each batch element i:
            q1 = (x_i - m_f_i)^T Sigma_f_i^{-1} (x_i - m_f_i)
            q2 = r_i^T Sigma_b_i^{-1} r_i
            r_i = x_{+,i} - x_i - F(x_i) dt_{+,i}
            Sigma_b_i = 2 D(x_i) dt_{+,i}

            and 
            Q_i = q1 + q2
            Phi_i(x_i) = 0.5 * q1 + 0.5 q2 + 0.5 log det Sigma_b_i

            Q is the standardized two-step noise/action cost.
            Phi is the unnormalized negative log-density objective used for
            MAP bridge optimization, up to constants independent of z.

        Args:
            x_states:       [B, dim], optimization variables
            m_f:            [B, dim]
            Sigma_f:        [B, dim, dim]
            x_next_states:  [B, dim]
            dt_next:        [B]
            base_mid_inputs:[B, input_dim]
            adjust_for_drift: whether to adjust for drift in diffusion calculation (default: False)
        Returns:
            Dictionary with Q_each (q1 + q2) and phi_each
            (0.5 * (q1 + q2 + log det Sigma_b_i)), each of shape [B].
        """
        B = x_states.shape[0]
        device = x_states.device
        dtype = x_states.dtype

        I = torch.eye(self.dim, device=device, dtype=dtype).unsqueeze(0)  # [1, dim, dim]

        # ---- prior term ----
        Sigma_f = 0.5 * (Sigma_f + Sigma_f.transpose(-1, -2)) + eps * I
        D_f = (x_states - m_f).unsqueeze(-1)  # [B, dim, 1]

        sol_f = torch.linalg.solve(Sigma_f, D_f)  # [B, dim, 1]
        prior_quad = 0.5 * torch.bmm(D_f.transpose(1, 2), sol_f).squeeze(-1).squeeze(-1)  # 0.5 * q1, [B]

        # ---- likelihood term ----
        x_cand_full = self._replace_state_in_input_batch(base_mid_inputs, x_states)  # [B, input_dim]

        F_x = self._eval_field_with_grad(x_cand_full, model_type='drift')  # [B, dim]
        D_x = self._eval_field_with_grad(x_cand_full, model_type='diff')   # [B, dim, dim]
        if adjust_for_drift:
            D_x = self._correct_diffusion_for_finite_dt(D_x, F_x)

        Sigma_b = 2.0 * D_x * dt_next[:, None, None]
        Sigma_b = 0.5 * (Sigma_b + Sigma_b.transpose(-1, -2)) + eps * I

        resid = (x_next_states - x_states - F_x * dt_next[:, None]).unsqueeze(-1)  # [B, dim, 1]

        sol_b = torch.linalg.solve(Sigma_b, resid)
        like_quad = 0.5 * torch.bmm(resid.transpose(1, 2), sol_b).squeeze(-1).squeeze(-1)  # 0.5 * q2, [B]

        sign, logabsdet = torch.linalg.slogdet(Sigma_b)
        if torch.any(sign <= 0).item():
            raise RuntimeError("Some Sigma_b matrices are not SPD in batched bridge likelihood.")

        logdet_term = 0.5 * logabsdet  # [B]

        Q_each = 2.0 * prior_quad + 2.0 * like_quad
        phi_each = prior_quad + like_quad + logdet_term
        return {"Q_each": Q_each, "phi_each": phi_each}


    def _build_bridge_valid_indices(self, series_slices):
        """
        Build flattened valid indices for bridge surprisal.

        Valid i are:
            s.start + 1, ..., s.stop - 1

        because x_- = xs[i-1] is needed, while x_+ is always defined as
            x_+ = xs[i][state_slice] + dxs[i].
        """
        valid_indices = []
        for s in series_slices:
            if s.start + 1 < s.stop:
                valid_indices.extend(range(s.start + 1, s.stop))
        return valid_indices
    

    def _bridge_gauss_newton_cov_batch(
        self,
        mu_states,
        Sigma_f,
        x_next_states,
        dt_next,
        base_mid_inputs,
        eps=1e-6,
        adjust_for_drift=False
    ):
        """
        Batched Gauss-Newton covariance.

        For each batch element i:
            r_i(x_i) = x_{+,i} - x_i - F(x_i) dt_i
            J_r_i    = -I - J_F_i dt_i

            H_i = Sigma_f_i^{-1} + J_r_i^T Sigma_b_i^{-1} J_r_i
            Sigma_i = H_i^{-1}

        Args:
            mu_states:       [B, dim]
            Sigma_f:         [B, dim, dim]
            x_next_states:   [B, dim]
            dt_next:         [B]
            base_mid_inputs: [B, input_dim]
        Returns:
            Sigma_t: [B, dim, dim]
            H_t:     [B, dim, dim]
        """
        B = mu_states.shape[0]
        device = mu_states.device
        dtype = mu_states.dtype
        Ddim = self.dim

        I = torch.eye(Ddim, device=device, dtype=dtype)
        I_b = I.unsqueeze(0).expand(B, -1, -1)

        # We need gradients wrt mu_states only for J_F.
        z = mu_states.detach().clone().requires_grad_(True)
        x_cand_full = self._replace_state_in_input_batch(base_mid_inputs, z)

        F_z = self._eval_field_with_grad(
            x_cand_full, model_type='drift'
        )  # [B, dim]

        # Build batched drift Jacobian J_F: [B, dim, dim]
        J_F_cols = []
        for j in range(Ddim):
            grad_j = torch.autograd.grad(
                F_z[:, j].sum(),
                z,
                retain_graph=(j < Ddim - 1),
                create_graph=False,
                allow_unused=False,
            )[0]  # [B, dim]
            J_F_cols.append(grad_j)

        # J_F[b, j, k] = d F_j / d x_k
        J_F = torch.stack(J_F_cols, dim=1)  # [B, dim, dim]

        # Local diffusion metric at the mode. No derivative of D is needed for GN.
        with torch.no_grad():
            x_cand_full_ng = self._replace_state_in_input_batch(base_mid_inputs, mu_states.detach())
            D_z = self._eval_field_with_grad(
                x_cand_full_ng, model_type='diff'
            )  # [B, dim, dim]
            if adjust_for_drift:
                D_z = self._correct_diffusion_for_finite_dt(D_z, F_z.detach())  # [B, dim, dim]

        Sigma_b = 2.0 * D_z * dt_next[:, None, None]
        Sigma_b = 0.5 * (Sigma_b + Sigma_b.transpose(-1, -2)) + eps * I_b

        Sigma_f = 0.5 * (Sigma_f + Sigma_f.transpose(-1, -2)) + eps * I_b

        # J_r = -I - dt * J_F
        J_r = -I_b - dt_next[:, None, None] * J_F  # [B, dim, dim]

        Sigma_f_inv = torch.linalg.inv(Sigma_f)
        H_t = Sigma_f_inv + J_r.transpose(1, 2) @ torch.linalg.solve(Sigma_b, J_r)

        H_t = 0.5 * (H_t + H_t.transpose(-1, -2)) + eps * I_b
        Sigma_t = torch.linalg.inv(H_t)
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-1, -2))

        return Sigma_t, H_t

    def _bridge_exact_hvp_score_batch(
        self,
        mu_bridge,
        x_mid_obs,
        m_f,
        Sigma_f,
        x_next_states,
        dt_next,
        base_mid_inputs,
        eps=1e-6,
        adjust_for_drift=False,
    ):
        """
        Compute Q = delta^T H_phi(mu_bridge) delta
        using a Hessian-vector product.
        """
        if mu_bridge.ndim != 2 or mu_bridge.shape[1] != self.dim:
            raise ValueError(
                f"mu_bridge must have shape [B, {self.dim}], "
                f"got {tuple(mu_bridge.shape)}."
            )

        if x_mid_obs.shape != mu_bridge.shape:
            raise ValueError(
                "x_mid_obs and mu_bridge must have identical shapes."
            )

        with torch.enable_grad():
            x_mode = (
                mu_bridge.detach()
                .clone()
                .requires_grad_(True)
            )

            delta = (
                x_mid_obs.detach()
                - x_mode.detach()
            )

            terms = self._two_step_terms_from_precomputed(
                x_states=x_mode,
                m_f=m_f.detach(),
                Sigma_f=Sigma_f.detach(),
                x_next_states=x_next_states.detach(),
                dt_next=dt_next.detach(),
                base_mid_inputs=base_mid_inputs.detach(),
                eps=eps,
                adjust_for_drift=adjust_for_drift,
            )

            phi_each = terms["phi_each"]

            if not torch.isfinite(phi_each).all():
                raise RuntimeError(
                    "Non-finite phi encountered during bridge HVP."
                )

            grad_phi = torch.autograd.grad(
                outputs=phi_each,
                inputs=x_mode,
                grad_outputs=torch.ones_like(phi_each),
                create_graph=True,
                allow_unused=False,
            )[0]

            H_delta = torch.autograd.grad(
                outputs=grad_phi,
                inputs=x_mode,
                grad_outputs=delta,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]

            d2 = torch.sum(
                delta * H_delta,
                dim=-1,
            )

            grad_norm = torch.linalg.vector_norm(
                grad_phi.detach(),
                dim=-1,
            )

        return d2.detach(), grad_norm

    # ==== Bridge-based exogeneity score estimation for intermediate states between two observed states ====
    def calculate_bridge_surprisal(
        self,
        xs,
        dxs,
        dts,
        series_slices,
        max_iter=50,
        lr=5e-2,
        loss_tol=1e-4,
        init_mode='obs',
        bridge_batch_size=1024,
        seed=42,
        return_lists=True,
        Q_method='hvp',
        eps=1e-6,
        adjust_for_drift=False,
        verbose=True
    ):
        """
        Batched bridge-based surprisal for interior observed states.

        Collect valid bridge points across trajectories, then optimize the
        Laplace mode in fixed-size mini-batches.

        For each valid index i:
            x_-       = xs[i-1]
            x_obs     = xs[i]
            x_+       = xs[i][state_slice] + dxs[i]
            dt_-      = dts[i-1]
            dt_+      = dts[i]

        Posterior:
            p(X_i | x_-, x_+) ∝ p(X_i | x_-) p(x_+ | X_i)

        Mode:
            mu_i = argmin_x Phi_i(x)

        Score selected by Q_method:
            'hvp': Hessian quadratic form at the optimized bridge mode.
            'gauss_newton': quadratic form using Gauss-Newton precision.
            'phi_diff': twice the objective difference between observation and mode.
        The statistic is calibrated against a chi-square reference. Covariance
        matrices are used internally for 'gauss_newton' but are not returned.

        Args:
            xs: [N, input_dim]
            dxs: [N, dim]
            dts: [N] or [N,1]
            series_slices: list of slices
            max_iter: Adam iterations for batched mode search
            lr: Adam learning rate
            init_mode: 'mf' or 'obs'. Default is 'obs'.
            bridge_batch_size: number of bridge points per optimization batch
            seed: random seed
            return_lists: whether to return trajectory-wise lists
            Q_method: 'hvp', 'gauss_newton' or 'phi_diff' (default: 'hvp')
            eps: diagonal jitter for covariance matrices
            adjust_for_drift: whether to adjust for drift in diffusion calculation (default: False)
        Returns:
            result dict:
                mu_bridge:      [N, dim]
                d2_bridge:      [N]
                p_bridge:       [N]
                s_bridge:       [N]
                valid_bridge:   [N], structural validity mask
                valid_bridge_indices: flattened interior-state indices
                Q_method: chosen scoring method
                optionally trajectory-wise lists
            Floating-point arrays use NaN outside valid indices. Tail scores
            also remain NaN for non-finite or negative statistics.
        """
        assert init_mode in ['mf', 'obs'], "init_mode must be 'mf' or 'obs' (default: 'obs')"
        
        valid_Q_methods = ['hvp', 'gauss_newton', 'phi_diff']
        if Q_method not in valid_Q_methods:
            raise ValueError(
                f"Q_method must be one of {sorted(valid_Q_methods)}, "
                f"but received {Q_method!r}."
            )

        np.random.seed(seed)
        torch.manual_seed(seed)
        
        xs, dxs, dts = self._prepare_transition_inputs(
            xs,
            dxs,
            dts,
        )
        N = xs.shape[0]

        state_sl = self._state_slice()

        # ------------------------------------------------------------------
        # 1. Build valid bridge indices
        # ------------------------------------------------------------------
        valid_indices = self._build_bridge_valid_indices(series_slices)

        if len(valid_indices) == 0:
            raise ValueError("No valid bridge indices were found. Check series_slices and include_last_step.")

        valid_idx = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
        M = valid_idx.numel()

        # ------------------------------------------------------------------
        # 2. Build all bridge contexts
        # ------------------------------------------------------------------
        x_prev_full_all = xs[valid_idx - 1].detach()
        x_mid_full_all = xs[valid_idx].detach()
        x_mid_obs_all = xs[valid_idx][:, state_sl].detach()
        x_next_all = (xs[valid_idx][:, state_sl] + dxs[valid_idx]).detach()

        dt_prev_all = dts[valid_idx - 1].detach()
        dt_next_all = dts[valid_idx].detach()

        # ------------------------------------------------------------------
        # 3. Precompute prior terms in batch
        # ------------------------------------------------------------------
        with torch.no_grad():
            F_prev_all = self._eval_field_with_grad(
                x_prev_full_all,
                model_type='drift',
            )  # [M, dim]

            D_prev_all = self._eval_field_with_grad(
                x_prev_full_all,
                model_type='diff',
            )  # [M, dim, dim]
            if adjust_for_drift:
                D_prev_all = self._correct_diffusion_for_finite_dt(D_prev_all, F_prev_all)  # [M, dim, dim]

            x_prev_state_all = x_prev_full_all[:, state_sl]
            m_f_all = x_prev_state_all + F_prev_all * dt_prev_all[:, None]
            Sigma_f_all = 2.0 * D_prev_all * dt_prev_all[:, None, None]

            I_all = torch.eye(self.dim, device=self.device, dtype=Sigma_f_all.dtype).unsqueeze(0)
            Sigma_f_all = 0.5 * (Sigma_f_all + Sigma_f_all.transpose(-1, -2)) + eps * I_all

            m_f_all = m_f_all.detach()
            Sigma_f_all = Sigma_f_all.detach()

        # ------------------------------------------------------------------
        # 4. Batched mode optimization
        # ------------------------------------------------------------------
        mu_all = torch.empty((M, self.dim), device=self.device, dtype=xs.dtype)

        for start in tqdm(range(0, M, bridge_batch_size), desc="Batched bridge mode optimization"):
            end = min(start + bridge_batch_size, M)

            m_f_b = m_f_all[start:end]
            Sigma_f_b = Sigma_f_all[start:end]
            x_next_b = x_next_all[start:end]
            dt_next_b = dt_next_all[start:end]
            x_mid_full_b = x_mid_full_all[start:end]
            x_mid_obs_b = x_mid_obs_all[start:end]

            if init_mode == 'mf':
                x0_b = m_f_b.detach().clone()
            else:
                x0_b = x_mid_obs_b.detach().clone()

            with torch.enable_grad():
                x_var = x0_b.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([x_var], lr=lr)

                prev_loss = None
    
                loss_val_list = []
                for it in range(max_iter):
                    optimizer.zero_grad()

                    phi_each = self._two_step_terms_from_precomputed(
                        x_states=x_var,
                        m_f=m_f_b,
                        Sigma_f=Sigma_f_b,
                        x_next_states=x_next_b,
                        dt_next=dt_next_b,
                        base_mid_inputs=x_mid_full_b,
                        eps=eps,
                        adjust_for_drift=adjust_for_drift,
                    )['phi_each']  # [B]

                    loss = phi_each.mean()
                    loss_val = loss.detach()

                    loss_val_list.append(loss_val.item())

                    if prev_loss is not None:
                        rel_change = torch.abs(prev_loss - loss_val) / (torch.abs(prev_loss) + 1e-12)

                        if rel_change.item() < loss_tol:
                            break

                    loss.backward()
                    optimizer.step()

                    prev_loss = loss_val

                if verbose:
                    print(f"Stopping at iteration {it} with loss {loss_val.item():.6f} and initial_loss {loss_val_list[0]:.6f}")

                mu_all[start:end] = x_var.detach()

        # ------------------------------------------------------------------
        # 5. Batched covariance and tail probability
        # ------------------------------------------------------------------
        mu_flat_t = torch.full((N, self.dim), float('nan'), device=self.device, dtype=xs.dtype)
        d2_flat_t = torch.full((N,), float('nan'), device=self.device, dtype=xs.dtype)
        valid_flat_t = torch.zeros((N,), device=self.device, dtype=torch.bool)

        for start in range(0, M, bridge_batch_size):
            end = min(start + bridge_batch_size, M)

            idx_b = valid_idx[start:end]              # [B]
            mu_b = mu_all[start:end]                  # [B, dim]
            Sigma_f_b = Sigma_f_all[start:end]         # [B, dim, dim]
            x_next_b = x_next_all[start:end]           # [B, dim]
            dt_next_b = dt_next_all[start:end]         # [B]
            x_mid_obs_b = x_mid_obs_all[start:end]     # [B, dim]
            x_mid_full_b = x_mid_full_all[start:end]   # [B, input_dim]


            d2_b = None
            if Q_method == 'hvp':
                d2_b, bridge_grad_norm_b = (
                    self._bridge_exact_hvp_score_batch(
                        mu_bridge=mu_b,
                        x_mid_obs=x_mid_obs_b,
                        m_f=m_f_all[start:end],
                        Sigma_f=Sigma_f_b,
                        x_next_states=x_next_b,
                        dt_next=dt_next_b,
                        base_mid_inputs=x_mid_full_b,
                        eps=eps,
                        adjust_for_drift=adjust_for_drift,
                    )
                )

            elif Q_method == 'gauss_newton':
                # Mahalanobis bridge statistic based on the local
                # Gauss-Newton covariance at the optimized bridge mode.
                Sigma_brg_b, H_b = self._bridge_gauss_newton_cov_batch(
                    mu_states=mu_b,
                    Sigma_f=Sigma_f_b,
                    x_next_states=x_next_b,
                    dt_next=dt_next_b,
                    base_mid_inputs=x_mid_full_b,
                    eps=eps,
                    adjust_for_drift=adjust_for_drift,
                )

                delta_b = (x_mid_obs_b - mu_b).unsqueeze(-1)  # [B, dim, 1]

                d2_b = torch.bmm(
                    delta_b.transpose(1, 2),
                    torch.bmm(H_b, delta_b),
                ).squeeze(-1).squeeze(-1)

                d2_b = torch.clamp(d2_b, min=0.0)

            elif Q_method == 'phi_diff':
                # Energy at the optimized bridge mode.
                with torch.no_grad():
                    phi_mu = self._two_step_terms_from_precomputed(
                        x_states=mu_b,
                        m_f=m_f_all[start:end],
                        Sigma_f=Sigma_f_b,
                        x_next_states=x_next_b,
                        dt_next=dt_next_b,
                        base_mid_inputs=x_mid_full_b,
                        eps=eps,
                        adjust_for_drift=adjust_for_drift,
                    )["phi_each"]

                    # Energy at the observed middle state.
                    phi_obs = self._two_step_terms_from_precomputed(
                        x_states=x_mid_obs_b,
                        m_f=m_f_all[start:end],
                        Sigma_f=Sigma_f_b,
                        x_next_states=x_next_b,
                        dt_next=dt_next_b,
                        base_mid_inputs=x_mid_full_b,
                        eps=eps,
                        adjust_for_drift=adjust_for_drift,
                    )["phi_each"]

                    # Likelihood-ratio / energy-gap bridge statistic.
                    d2_b = 2.0 * (phi_obs - phi_mu)
                    d2_b = torch.clamp(d2_b, min=0.0)

            mu_flat_t[idx_b] = mu_b.detach()
            d2_flat_t[idx_b] = d2_b.detach()
            valid_flat_t[idx_b] = True

        mu_flat = mu_flat_t.detach().cpu().numpy()
        d2_flat = d2_flat_t.detach().cpu().numpy()
        valid_flat = valid_flat_t.detach().cpu().numpy()

        p_flat = np.full(N, np.nan, dtype=float)
        s_flat = np.full(N, np.nan, dtype=float)

        # Calibrate only structurally valid bridges with finite, nonnegative
        # statistics from the selected Q_method.
        score_valid = (
            valid_flat
            & np.isfinite(d2_flat)
            & (d2_flat >= 0.0)
        )

        valid_d2 = d2_flat[score_valid]

        # For dim=1, chi2 survival with df=1 equals the two-sided Gaussian tail.
        p_valid = chi2.sf(valid_d2, df=self.dim)
        p_valid = np.clip(p_valid, 1e-300, 1.0)

        p_flat[score_valid] = p_valid
        s_flat[score_valid] = -np.log10(p_valid)

        # ------------------------------------------------------------------
        # 6. Reconstruct per-trajectory lists using series_slices
        # ------------------------------------------------------------------
        result = {
            "mu_bridge": mu_flat,
            "d2_bridge": d2_flat,
            "p_bridge": p_flat,
            "s_bridge": s_flat,
            "valid_bridge": valid_flat,
            "valid_bridge_indices": np.array(valid_indices, dtype=int),
            "Q_method": Q_method,
        }

        if return_lists:
            per_traj_mu = []
            per_traj_d2 = []
            per_traj_p = []
            per_traj_s = []

            for s in series_slices:
                end = s.stop
                idx_range = np.arange(s.start, end, dtype=int)

                if len(idx_range) == 0:
                    per_traj_mu.append(np.empty((0, self.dim)))
                    per_traj_d2.append(np.empty((0,)))
                    per_traj_p.append(np.empty((0,)))
                    per_traj_s.append(np.empty((0,)))
                else:
                    per_traj_mu.append(result["mu_bridge"][idx_range])
                    per_traj_d2.append(result["d2_bridge"][idx_range])
                    per_traj_p.append(result["p_bridge"][idx_range])
                    per_traj_s.append(result["s_bridge"][idx_range])

            result.update({
                "mu_bridge_list": per_traj_mu,
                "d2_bridge_list": per_traj_d2,
                "p_bridge_list": per_traj_p,
                "s_bridge_list": per_traj_s,
            })

        return result


    def calculate_endpoint_surprisal(
        self,
        xs,
        dxs,
        dts,
        series_slices,
        n_process_samples=1024,
        endpoint_batch_size=1024,
        mc_chunk_size=64,
        seed=42,
        return_lists=True,
        eps=1e-6,
        adjust_for_drift=False,
        verbose=True,
    ):
        """
        Calculate the endpoint surprisal using Monte Carlo moment propagation.

        For each valid two-step segment i:

            x_-       = xs[i - 1]
            x_t       = latent intermediate state
            x_+       = xs[i][state_slice] + dxs[i]
            dt_-      = dts[i - 1]
            dt_+      = dts[i]

        The first-step predictive distribution is approximated as

            Z | x_-
                ~ N(m_f, Sigma_f),

            m_f
                = x_- + F(x_-) dt_-,

            Sigma_f
                = 2 D(x_-) dt_-.

        Draw intermediate-state samples

            Z^(m) ~ N(m_f, Sigma_f),

        and propagate each sample through the second local-Gaussian transition:

            X_+ | Z^(m)
                ~ N(g^(m), Sigma_+^(m)),

            g^(m)
                = Z^(m) + F(Z^(m)) dt_+,

            Sigma_+^(m)
                = 2 D(Z^(m)) dt_+.

        The endpoint distribution is then approximated by a moment-matched
        Gaussian using the laws of total expectation and total covariance:

            mu_endpoint
                = E[g(Z)],

            Sigma_endpoint
                = Cov[g(Z)] + E[Sigma_+(Z)].

        The endpoint Mahalanobis statistic is

            Q_endpoint
                = (x_+ - mu_endpoint)^T
                Sigma_endpoint^{-1}
                (x_+ - mu_endpoint),

        with the Gaussian-reference tail probability

            p_endpoint
                = P(ChiSquare_dim >= Q_endpoint).

        Notes
        -----
        - n_process_samples controls Monte Carlo propagation of process noise.
        It is distinct from any SWAG/model-posterior Monte Carlo parameter.
        - Drift and diffusion are evaluated in no-gradient mode because no
        optimization or field derivative is needed.
        - mc_chunk_size controls the number of process samples evaluated at
        once for each endpoint batch, limiting GPU memory use.
        - The resulting p-value remains based on a moment-matched Gaussian
        approximation. It is not a fully empirical Monte Carlo tail
        probability.

        Parameters
        ----------
        xs : torch.Tensor or numpy.ndarray, shape [N, input_dim]
            Full model inputs at the beginning of each observed transition.

        dxs : torch.Tensor or numpy.ndarray, shape [N, dim]
            Observed physical-state increments.

        dts : torch.Tensor or numpy.ndarray, shape [N] or [N, 1]
            Positive transition intervals.

        series_slices : list[slice]
            Slices identifying individual trajectories.

        n_process_samples : int
            Number of first-step intermediate-state samples per two-step
            segment. Must be at least 2.

        endpoint_batch_size : int
            Number of two-step segments processed together.

        mc_chunk_size : int
            Number of Monte Carlo samples per segment evaluated together.
            Smaller values reduce peak memory consumption.

        seed : int
            Random seed for intermediate-state sampling.

        return_lists : bool
            If True, return trajectory-wise arrays in addition to flattened
            arrays.

        eps : float
            Minimum covariance eigenvalue and numerical regularization.

        adjust_for_drift : bool
            If True, apply the existing finite-dt diffusion correction.

        verbose : bool
            If True, display a progress bar.

        Returns
        -------
        dict
            mu_endpoint:
                Moment-matched endpoint means, shape [N, dim].

            Sigma_endpoint:
                Moment-matched endpoint covariances, shape [N, dim, dim].

            d2_endpoint:
                Endpoint Mahalanobis statistics, shape [N].

            p_endpoint:
                Chi-square-calibrated tail probabilities, shape [N].

            s_endpoint:
                -log10(p_endpoint), shape [N].

            valid_endpoint:
                Boolean mask identifying valid two-step segments.

            valid_endpoint_indices:
                Flattened valid indices.

            n_process_samples:
                Number of process samples used.

            If return_lists=True, trajectory-wise versions are also returned.
        """
        if n_process_samples < 2:
            raise ValueError(
                "n_process_samples must be at least 2 so that the "
                "intermediate-state covariance can be estimated."
            )

        if endpoint_batch_size <= 0:
            raise ValueError("endpoint_batch_size must be positive.")

        if mc_chunk_size <= 0:
            raise ValueError("mc_chunk_size must be positive.")

        np.random.seed(seed)
        torch.manual_seed(seed)

        xs, dxs, dts = self._prepare_transition_inputs(
            xs,
            dxs,
            dts,
        )

        N = xs.shape[0]
        state_sl = self._state_slice()

        # --------------------------------------------------------------
        # 1. Build valid two-step indices
        # --------------------------------------------------------------
        valid_indices = self._build_bridge_valid_indices(series_slices)

        if len(valid_indices) == 0:
            raise ValueError("No valid endpoint indices were found.")

        valid_idx = torch.tensor(
            valid_indices,
            device=self.device,
            dtype=torch.long,
        )

        M = valid_idx.numel()

        # --------------------------------------------------------------
        # 2. Construct all two-step contexts
        # --------------------------------------------------------------
        x_prev_full_all = xs[valid_idx - 1].detach()

        # Full inputs at the intermediate time. The physical-state block
        # will later be replaced by sampled intermediate states.
        x_mid_full_all = xs[valid_idx].detach()

        x_next_all = (
            xs[valid_idx][:, state_sl]
            + dxs[valid_idx]
        ).detach()

        dt_prev_all = dts[valid_idx - 1].detach()
        dt_next_all = dts[valid_idx].detach()

        # --------------------------------------------------------------
        # 3. First-step predictive Gaussian
        #
        # Z | x_- ~ N(m_f, Sigma_f)
        # --------------------------------------------------------------
        with torch.no_grad():
            F_prev_all = self._eval_field_with_grad(
                x_prev_full_all,
                model_type="drift",
            )

            D_prev_all = self._eval_field_with_grad(
                x_prev_full_all,
                model_type="diff",
            )

            if adjust_for_drift:
                D_prev_all = self._correct_diffusion_for_finite_dt(
                    D_prev_all,
                    F_prev_all,
                )

            D_prev_all = 0.5 * (
                D_prev_all
                + D_prev_all.transpose(-1, -2)
            )

            x_prev_state_all = x_prev_full_all[:, state_sl]

            m_f_all = (
                x_prev_state_all
                + F_prev_all * dt_prev_all[:, None]
            )

            Sigma_f_all = (
                2.0
                * D_prev_all
                * dt_prev_all[:, None, None]
            )

            I_all = torch.eye(
                self.dim,
                device=self.device,
                dtype=xs.dtype,
            ).unsqueeze(0)

            Sigma_f_all = 0.5 * (
                Sigma_f_all
                + Sigma_f_all.transpose(-1, -2)
            )

            # Ensure positive definiteness before sampling.
            min_eig_f = torch.linalg.eigvalsh(
                Sigma_f_all
            )[:, 0]

            shift_f = torch.clamp(
                eps - min_eig_f,
                min=0.0,
            )

            Sigma_f_all = (
                Sigma_f_all
                + shift_f[:, None, None] * I_all
            )

            m_f_all = m_f_all.detach()
            Sigma_f_all = Sigma_f_all.detach()

        # --------------------------------------------------------------
        # 4. Allocate valid-segment results
        # --------------------------------------------------------------
        mu_all = torch.empty(
            (M, self.dim),
            device=self.device,
            dtype=xs.dtype,
        )

        Sigma_all = torch.empty(
            (M, self.dim, self.dim),
            device=self.device,
            dtype=xs.dtype,
        )

        Q_all = torch.empty(
            (M,),
            device=self.device,
            dtype=xs.dtype,
        )

        batch_iterator = range(
            0,
            M,
            endpoint_batch_size,
        )

        if verbose:
            batch_iterator = tqdm(
                batch_iterator,
                desc="MC endpoint moment propagation",
            )

        # --------------------------------------------------------------
        # 5. Propagate first-step uncertainty through the second step
        # --------------------------------------------------------------
        with torch.no_grad():
            for start in batch_iterator:
                end = min(
                    start + endpoint_batch_size,
                    M,
                )

                B = end - start

                m_f_b = m_f_all[start:end]
                Sigma_f_b = Sigma_f_all[start:end]
                x_next_b = x_next_all[start:end]
                dt_next_b = dt_next_all[start:end]
                x_mid_full_b = x_mid_full_all[start:end]

                I_b = torch.eye(
                    self.dim,
                    device=self.device,
                    dtype=xs.dtype,
                ).unsqueeze(0).expand(B, -1, -1)

                # Cholesky factor for sampling:
                #
                # Z = m_f + L epsilon,
                # epsilon ~ N(0, I).
                chol_f_b = torch.linalg.cholesky(
                    Sigma_f_b
                )

                # Streaming accumulators avoid storing every propagated
                # sample simultaneously.
                sum_g = torch.zeros(
                    (B, self.dim),
                    device=self.device,
                    dtype=xs.dtype,
                )

                sum_gg = torch.zeros(
                    (B, self.dim, self.dim),
                    device=self.device,
                    dtype=xs.dtype,
                )

                sum_cond_cov = torch.zeros(
                    (B, self.dim, self.dim),
                    device=self.device,
                    dtype=xs.dtype,
                )

                n_done = 0

                while n_done < n_process_samples:
                    n_chunk = min(
                        mc_chunk_size,
                        n_process_samples - n_done,
                    )

                    # Standard-normal samples:
                    # shape [B, n_chunk, dim]
                    noise = torch.randn(
                        (B, n_chunk, self.dim),
                        device=self.device,
                        dtype=xs.dtype,
                    )

                    # For row-vector samples:
                    # noise @ L^T = L noise in column notation.
                    z_samples = (
                        m_f_b[:, None, :]
                        + torch.matmul(
                            noise,
                            chol_f_b.transpose(1, 2),
                        )
                    )

                    # Repeat the non-state model inputs for each process
                    # sample and replace only the physical-state block.
                    base_inputs = (
                        x_mid_full_b[:, None, :]
                        .expand(
                            B,
                            n_chunk,
                            self.input_dim,
                        )
                        .reshape(
                            B * n_chunk,
                            self.input_dim,
                        )
                    )

                    z_flat = z_samples.reshape(
                        B * n_chunk,
                        self.dim,
                    )

                    z_full = self._replace_state_in_input_batch(
                        base_inputs,
                        z_flat,
                    )

                    F_z_flat = self._eval_field_with_grad(
                        z_full,
                        model_type="drift",
                    )

                    D_z_flat = self._eval_field_with_grad(
                        z_full,
                        model_type="diff",
                    )

                    if adjust_for_drift:
                        D_z_flat = self._correct_diffusion_for_finite_dt(
                            D_z_flat,
                            F_z_flat,
                        )

                    F_z = F_z_flat.reshape(
                        B,
                        n_chunk,
                        self.dim,
                    )

                    D_z = D_z_flat.reshape(
                        B,
                        n_chunk,
                        self.dim,
                        self.dim,
                    )

                    D_z = 0.5 * (
                        D_z
                        + D_z.transpose(-1, -2)
                    )

                    # Conditional endpoint mean:
                    #
                    # g(z) = z + F(z) dt_+
                    g_samples = (
                        z_samples
                        + F_z * dt_next_b[:, None, None]
                    )

                    # Conditional endpoint covariance:
                    #
                    # Sigma_+(z) = 2 D(z) dt_+
                    cond_cov_samples = (
                        2.0
                        * D_z
                        * dt_next_b[:, None, None, None]
                    )

                    # Accumulate E[g], E[g g^T], and E[Sigma_+(Z)].
                    sum_g = sum_g + g_samples.sum(dim=1)

                    sum_gg = sum_gg + torch.einsum(
                        "bsi,bsj->bij",
                        g_samples,
                        g_samples,
                    )

                    sum_cond_cov = (
                        sum_cond_cov
                        + cond_cov_samples.sum(dim=1)
                    )

                    n_done += n_chunk

                n_mc = float(n_process_samples)

                # ------------------------------------------------------
                # 6. Laws of total expectation and total covariance
                # ------------------------------------------------------
                mu_b = sum_g / n_mc

                # Unbiased MC estimate of Cov[g(Z)]:
                #
                # sum_m (g_m - mean)(g_m - mean)^T / (M - 1)
                mean_outer_b = torch.bmm(
                    mu_b.unsqueeze(-1),
                    mu_b.unsqueeze(-2),
                )

                cov_mean_b = (
                    sum_gg
                    - n_mc * mean_outer_b
                ) / float(n_process_samples - 1)

                mean_cond_cov_b = (
                    sum_cond_cov / n_mc
                )

                Sigma_end_b = (
                    cov_mean_b
                    + mean_cond_cov_b
                )

                Sigma_end_b = 0.5 * (
                    Sigma_end_b
                    + Sigma_end_b.transpose(-1, -2)
                )

                # Enforce a minimum eigenvalue before inversion.
                min_eig_end = torch.linalg.eigvalsh(
                    Sigma_end_b
                )[:, 0]

                shift_end = torch.clamp(
                    eps - min_eig_end,
                    min=0.0,
                )

                Sigma_end_b = (
                    Sigma_end_b
                    + shift_end[:, None, None] * I_b
                )

                # ------------------------------------------------------
                # 7. Endpoint Mahalanobis statistic
                # ------------------------------------------------------
                residual_b = (
                    x_next_b - mu_b
                ).unsqueeze(-1)

                solved_b = torch.linalg.solve(
                    Sigma_end_b,
                    residual_b,
                )

                Q_b = torch.bmm(
                    residual_b.transpose(1, 2),
                    solved_b,
                ).squeeze(-1).squeeze(-1)

                Q_b = torch.clamp(
                    Q_b,
                    min=0.0,
                )

                mu_all[start:end] = mu_b
                Sigma_all[start:end] = Sigma_end_b
                Q_all[start:end] = Q_b

        # --------------------------------------------------------------
        # 8. Scatter valid results back to flattened state alignment
        # --------------------------------------------------------------
        mu_flat_t = torch.full(
            (N, self.dim),
            float("nan"),
            device=self.device,
            dtype=xs.dtype,
        )

        Sigma_flat_t = torch.full(
            (N, self.dim, self.dim),
            float("nan"),
            device=self.device,
            dtype=xs.dtype,
        )

        Q_flat_t = torch.full(
            (N,),
            float("nan"),
            device=self.device,
            dtype=xs.dtype,
        )

        valid_flat_t = torch.zeros(
            (N,),
            device=self.device,
            dtype=torch.bool,
        )

        mu_flat_t[valid_idx] = mu_all
        Sigma_flat_t[valid_idx] = Sigma_all
        Q_flat_t[valid_idx] = Q_all
        valid_flat_t[valid_idx] = True

        mu_flat = mu_flat_t.detach().cpu().numpy()
        Sigma_flat = Sigma_flat_t.detach().cpu().numpy()
        Q_flat = Q_flat_t.detach().cpu().numpy()
        valid_flat = valid_flat_t.detach().cpu().numpy()

        # --------------------------------------------------------------
        # 9. Gaussian-reference tail probability
        # --------------------------------------------------------------
        p_flat = np.full(
            N,
            np.nan,
            dtype=float,
        )

        s_flat = np.full(
            N,
            np.nan,
            dtype=float,
        )

        valid_Q = Q_flat[valid_flat]

        p_valid = chi2.sf(
            valid_Q,
            df=self.dim,
        )

        p_valid = np.clip(
            p_valid,
            1e-300,
            1.0,
        )

        p_flat[valid_flat] = p_valid
        s_flat[valid_flat] = -np.log10(p_valid)

        # --------------------------------------------------------------
        # 10. Package flattened outputs
        # --------------------------------------------------------------
        result = {
            "mu_endpoint": mu_flat,
            "Sigma_endpoint": Sigma_flat,
            "d2_endpoint": Q_flat,
            "p_endpoint": p_flat,
            "s_endpoint": s_flat,
            "valid_endpoint": valid_flat,
            "valid_endpoint_indices": np.asarray(
                valid_indices,
                dtype=int,
            ),
            "n_process_samples": int(n_process_samples),
        }

        # --------------------------------------------------------------
        # 11. Reconstruct per-trajectory outputs
        # --------------------------------------------------------------
        if return_lists:
            per_traj_mu = []
            per_traj_Sigma = []
            per_traj_Q = []
            per_traj_p = []
            per_traj_s = []

            for sl in series_slices:
                idx_range = np.arange(
                    sl.start,
                    sl.stop,
                    dtype=int,
                )

                if len(idx_range) == 0:
                    per_traj_mu.append(
                        np.empty((0, self.dim))
                    )

                    per_traj_Sigma.append(
                        np.empty((0, self.dim, self.dim))
                    )

                    per_traj_Q.append(
                        np.empty((0,))
                    )

                    per_traj_p.append(
                        np.empty((0,))
                    )

                    per_traj_s.append(
                        np.empty((0,))
                    )
                else:
                    per_traj_mu.append(
                        result["mu_endpoint"][idx_range]
                    )

                    per_traj_Sigma.append(
                        result["Sigma_endpoint"][idx_range]
                    )

                    per_traj_Q.append(
                        result["d2_endpoint"][idx_range]
                    )

                    per_traj_p.append(
                        result["p_endpoint"][idx_range]
                    )

                    per_traj_s.append(
                        result["s_endpoint"][idx_range]
                    )

            result.update({
                "mu_endpoint_list": per_traj_mu,
                "Sigma_endpoint_list": per_traj_Sigma,
                "d2_endpoint_list": per_traj_Q,
                "p_endpoint_list": per_traj_p,
                "s_endpoint_list": per_traj_s,
            })

        return result
