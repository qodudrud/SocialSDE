import numpy as np
import math
from tqdm import tqdm

import torch


from model.net import make_model
from model.LangevinDiagnosticsBase import LangevinDiagnosticsBase
from ..misc.utils import load_vector, sample_swag, collect_swag_snapshots, build_swag_stats


class Langevin_from_LBN(LangevinDiagnosticsBase):
    def __init__(self, args, drift_models=None, diff_models=None, device=None):
        super().__init__(args, device=device)

        # Create the drift and diffusion models and save the provided states for later use.
        self.drift_model = make_model(args, 'drift')
        self.diff_model = make_model(args, 'diff')

        self.drift_models = drift_models
        self.diff_models = diff_models
        if drift_models is not None:
            print(f"Initialized Langevin_from_LBN with {len(drift_models)} drift model states.")
        if diff_models is not None:
            print(f"Initialized Langevin_from_LBN with {len(diff_models)} diffusion model states.")

        self.swag = {'drift': False, 'diff': False}
        if drift_models is not None and isinstance(drift_models[0], dict) and 'model_mu' in drift_models[0]:
            print("Drift models appear to be SWAG states.")
            self.swag['drift'] = True
        if diff_models is not None and isinstance(diff_models[0], dict) and 'model_mu' in diff_models[0]:
            print("Diffusion models appear to be SWAG states.")
            self.swag['diff'] = True

        self.uct_abs_threshold = {'drift': None, 'diff': None}
        self.uct_rel_threshold = {'drift': None, 'diff': None}

        # Use the drift model count, falling back to the diffusion model count.
        assert self.drift_models is not None or self.diff_models is not None, "At least one of drift_models or diff_models must be provided"
        self.n_folds = len(drift_models) if drift_models is not None else len(diff_models)

        # Move the models to the specified device for computation.
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.drift_model.to(self.device)
        self.diff_model.to(self.device)


    def set_device(self, device):
        """
        Set the device for computation and move the models to that device.
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.drift_model.to(self.device)
        self.diff_model.to(self.device)


    # ===== SWAG sampling helper function =====
    def prepare_swag_models(self,
                            train_loader, 
                            valid_loader, 
                            model_type='drift',
                            # training hyperparameters
                            grad_clip=0.0,
                            # SWAG hyperparameters
                            lr_swag=1e-4, momentum=0.9, weight_decay=5e-4,
                            n_snaps=50, burn_in_epochs=5, snap_every=3,
                            diff_tol=5e-2
                            ):
        """
        Prepare SWAG models by loading the best model states and collecting SWAG snapshots to build the SWAG statistics.
        Args:
            train_loader: DataLoader for the training dataset
            valid_loader: DataLoader for the validation dataset
            model_type: 'drift', 'diff', or 'both' to specify which model to prepare for SWAG (default: 'drift')
        """
        assert model_type in ['drift', 'diff', 'both'], "model_type must be 'drift', 'diff', or 'both'"
        if model_type == 'both':
            print("Preparing SWAG models for both drift and diffusion...")
            self.prepare_swag_models(train_loader, valid_loader, model_type='drift', grad_clip=grad_clip,
                                     lr_swag=lr_swag, momentum=momentum, weight_decay=weight_decay, 
                                     n_snaps=n_snaps, burn_in_epochs=burn_in_epochs, snap_every=snap_every, 
                                     diff_tol=diff_tol)
            self.prepare_swag_models(train_loader, valid_loader, model_type='diff', grad_clip=grad_clip,
                                     lr_swag=lr_swag, momentum=momentum, weight_decay=weight_decay, 
                                     n_snaps=n_snaps, burn_in_epochs=burn_in_epochs, snap_every=snap_every, 
                                     diff_tol=diff_tol)
            return
        print(f"Preparing SWAG models for {model_type}...")
        swag_models = []
        states = self.drift_models if model_type == 'drift' else self.diff_models
        for s in states:
            model = self.drift_model if model_type == 'drift' else self.diff_model
            model.load_state_dict(s)
            
            snaps = collect_swag_snapshots(model, 
                                           train_loader, 
                                           valid_loader,
                                           model_type,
                                           device=self.device,
                                           # training hyperparameters for snapshot collection
                                           time_step=self.time_step, time_reg_lambda=self.time_reg_lambda, grad_clip=grad_clip, include_time=self.include_time,
                                           # SWAG hyperparameters for snapshot collection
                                           lr_swag=lr_swag, momentum=momentum, weight_decay=weight_decay, 
                                           n_snaps=n_snaps, burn_in_epochs=burn_in_epochs, snap_every=snap_every, 
                                           diff_tol=diff_tol)
            model_mu, model_diag, model_S = build_swag_stats(snaps, rank_k=20)
            swag_state = {'model_mu': model_mu, 'model_diag': model_diag, 'model_S': model_S}
            swag_models.append(swag_state)
        
        if model_type == 'drift':
            self.drift_models = swag_models
        elif model_type == 'diff':
            self.diff_models = swag_models
        print(f"Prepared SWAG models for {model_type} with {len(swag_models)} folds.")


    
    def set_uncertainty_threshold(self, xs, q=0.99, model_type='drift', eps=1e-8, n_MC=20):
        """
        Set empirical uncertainty thresholds from reference input states.

        Args:
            xs: tensor of shape (N, input_dim).
                These should usually be observed/training states.
            q: quantile value for determining the threshold.
            model_type: 'drift', 'diff', or 'both'.
            eps: small constant for relative uncertainty.
            n_MC: number of SWAG samples per fold.
        """
        if model_type == 'both':
            self.set_uncertainty_threshold(xs, q=q, model_type='drift')
            self.set_uncertainty_threshold(xs, q=q, model_type='diff')
            return
        
        mean_output, var_output = self.eval_field_mean_var(xs, n_MC=n_MC, model_type=model_type, return_cpu=True)  # [N, dim] or [N, dim, dim]
        mean_output = mean_output.flatten(start_dim=1)  # [N, dim*dim] for diffusion or [N, dim] for drift
        var_output = var_output.flatten(start_dim=1).clamp_min(0.0)  # [N, dim*dim] for diffusion or [N, dim] for drift

        uct_abs = torch.sqrt(torch.sum(var_output, dim=-1))  # [N]
        uct_rel = uct_abs / (torch.norm(mean_output, dim=-1) + eps)  # relative uncertainty, [N]

        self.uct_abs_threshold[model_type] = torch.quantile(uct_abs, q)
        self.uct_rel_threshold[model_type] = torch.quantile(uct_rel, q)
        print(f"Set {model_type} uncertainty thresholds: absolute {q}-quantile = {torch.quantile(uct_abs, q):.4f}, relative {q}-quantile = {torch.quantile(uct_rel, q):.4f}")
        

    # ===== SWAG inference (mean/var over samples) =====
    def eval_field_mean_var(self, pts, n_MC=20, model_type='drift', store_MC_outputs=False, return_cpu=False):
        """
        Evaluate output mean and variance across SWAG samples or saved model states.
        Args:
            pts: input points (N, input_dim)
            n_MC: number of SWAG samples per fold (default: 20)
            model_type: 'drift' or 'diff' to specify which model to use for prediction (default: 'drift')
            store_MC_outputs: if True, store outputs from each MC sample for debugging/analysis (default: False)
            return_cpu: if True, return mean and variance on CPU (default: False)
        Returns:
            mean: mean (N, dim) if model_type=='drift' else (N, dim, dim) 
            var: variance (N, dim) if model_type=='drift' else (N, dim, dim)
            outs_MC: additionally returned in SWAG mode with store_MC_outputs=True;
                shape (n_folds, n_MC, N, dim) or (n_folds, n_MC, N, dim, dim).
        """
        assert model_type in ['drift', 'diff'], "model_type must be 'drift' or 'diff'"

        model = self.drift_model if model_type == 'drift' else self.diff_model
        models = self.drift_models if model_type == 'drift' else self.diff_models
        if models is None:
            raise ValueError(f"No models provided for model_type '{model_type}'")

        pts = pts.to(self.device)

        N = pts.size(0)
        out_shape = (N, self.dim) if model_type == 'drift' else (N, self.dim, self.dim)
        
        count = 0
        if store_MC_outputs:
            outs_MC = []  # to store outputs from each MC sample for debugging/analysis
        else:
            sum_out = torch.zeros(out_shape, device=self.device)
            sum_out_sq = torch.zeros_like(sum_out)

        with torch.inference_mode():
            model.eval()

            # Loop over each fold's models and perform Monte Carlo sampling to estimate mean and variance of the model output.
            for st in models:
                if not self.swag[model_type]:
                    model.load_state_dict(st)
                    outs = model(pts)

                    sum_out += outs  # accumulate sum for mean calculation
                    sum_out_sq += outs * outs  # accumulate sum of squares for variance calculation
                    if store_MC_outputs:
                        outs_MC.append(outs.detach().cpu() if return_cpu else outs.detach())  # store output for debugging/analysis, [N, dim] or [N, dim, dim]
                    count += 1

                else:
                    # Sample from the SWAG distribution and compute model outputs for each sample to estimate mean and variance.
                    MC_list = []
                    for _ in range(n_MC):
                        mu, diag, S = st['model_mu'], st['model_diag'], st['model_S']

                        w = sample_swag(mu, diag, S, scale=1.0,
                                        device=self.device, dtype=next(model.parameters()).dtype)
                        load_vector(model, w)   

                        outs = model(pts)                    # [N, dim] on device
                        if store_MC_outputs:
                            MC_list.append(outs.detach().cpu() if return_cpu else outs.detach())  # store output for debugging/analysis
                        else:
                            sum_out += outs  # accumulate sum for mean calculation
                            sum_out_sq += outs * outs  # accumulate sum of squares for variance calculation
                        count += 1

                    if store_MC_outputs:
                        outs_MC.append(torch.stack(MC_list, dim=0))  # [n_MC, N, dim] or [n_MC, N, dim, dim]

        if store_MC_outputs:
            outs_MC = torch.stack(outs_MC, dim=0)  # [n_folds, n_MC, N, dim] or [n_folds, n_MC, N, dim, dim]
            mean = outs_MC.mean(dim=(0, 1))  # [N, dim] or [N, dim, dim]
            var = outs_MC.var(dim=(0, 1), unbiased=True)  # [N, dim] or [N, dim, dim]
        else:
            mean = sum_out / count
            var = (sum_out_sq - sum_out*sum_out/count) / (count - 1)
        
        if return_cpu:
            mean = mean.detach().cpu()
            var = var.detach().cpu()

        if store_MC_outputs:
            return mean, var, outs_MC  # also return all MC outputs for debugging/analysis
        else:
            return mean, var

    
    def _eval_field_with_grad(self, pts, model_type='drift'):
        """
        Differentiable prediction using SWAG mean weights only w.r.t. input points, not model parameters.
        We average predictions over fold mean weights (model_mu), without MC sampling,
        so gradients w.r.t. input points are available.

        Args:
            pts: [N, input_dim]
            model_type: 'drift' or 'diff'
        Returns:
            drift -> [N, dim]
            diff  -> [N, dim, dim]
        """
        assert model_type in ['drift', 'diff']
        model = self.drift_model if model_type == 'drift' else self.diff_model
        swag_states = self.drift_models if model_type == 'drift' else self.diff_models
        
        if swag_states is None:
            raise ValueError(f"No SWAG states provided for model_type='{model_type}'")

        model.eval()
        pts = pts.to(self.device)

        # We need gradients w.r.t. pts, not model parameters.
        for p in model.parameters():
            p.requires_grad_(False)

        out_sum = None
        for st in swag_states:
            mu = st['model_mu']
            load_vector(model, mu)
            out = model(pts)
            out_sum = out if out_sum is None else out_sum + out

        return out_sum / len(swag_states)


    def _build_state_update(self, state_update, cond_vec, n_trajs, dt):
        """
        Build a full-input increment with zero increments for conditioning features
        and an optional time increment dt.
        Args:
            state_update: tensor of shape (n_trajs, dim) representing the drift or diffusion term to be added to the state
            cond_vec: tensor of shape (n_trajs, n_cond_onehot) representing the conditional one-hot vector for each trajectory (if n_cond_onehot > 0)
            n_trajs: number of trajectories being simulated
            dt: time step for simulation (used if include_time is True)
        Returns:
            state_update: full-input increment of shape (n_trajs, input_dim)
        """
        if self.n_cond_onehot > 0:
            zero_cond = torch.zeros_like(cond_vec)
            state_update = torch.cat([zero_cond, state_update], dim=1)

        if self.include_time:
            time_term = torch.full((n_trajs, 1), dt, device=self.device)  # [N_traj, 1]
            state_update = torch.cat([state_update, time_term], dim=1)

        return state_update


    # ===== Trajectory simulation from the learned LBN model =====
    def simulation_from_model(
        self,
        initial_points,
        n_steps,
        seed=0,
        adjust_for_drift=False,
        uncertainty_aware=False,
        use_relative_uncertainty=False,
        n_MC=20,
        jitter=1e-6,
        return_info=False,
    ):
        """
        Simulate trajectories from the learned LBN model.

        Args:
            initial_points: tensor of shape (n_trajs, input_dim).
                If include_time is True, the last component is the time coordinate.
            n_steps: number of stored states, including the initial state; n_steps - 1 updates.
            seed: random seed for reproducibility.
            adjust_for_drift: if True, subtract the finite-time drift contribution
                from diffusion. This is an optional finite-time second-moment correction.
            uncertainty_aware: if True, stop each trajectory when epistemic uncertainty
                is too high or when Cholesky decomposition fails.
            use_relative_uncertainty: if True, relative uncertainty is also used as a
                hard stopping criterion. Recommended default is False.
            n_MC: number of SWAG samples per fold used in eval_field_mean_var.
            jitter: diagonal jitter added before Cholesky decomposition.
            return_info: if True, return masks and stopping diagnostics.

        Returns:
            If return_info is False:
                trajs: tensor of shape (n_trajs, n_steps, input_dim).
            If return_info is True:
                dict containing trajs, alive_mask, stop_time, stop_reason,
                and uncertainty histories.
        """
        assert initial_points.dim() == 2, "initial_points should have shape (n_trajs, input_dim)"
        assert self.input_dim == initial_points.shape[1], "initial_points should have shape (n_trajs, input_dim)"
        assert self.drift_models is not None and self.diff_models is not None, \
            "simulation_from_model requires both drift_models and diff_models."

        if uncertainty_aware:
            assert hasattr(self, "uct_abs_threshold"), \
                "Run set_uncertainty_threshold(..., model_type='both') before using uncertainty_aware=True."
            assert "drift" in self.uct_abs_threshold and "diff" in self.uct_abs_threshold, \
                "Both drift and diffusion absolute uncertainty thresholds are required."

            if use_relative_uncertainty:
                assert hasattr(self, "uct_rel_threshold"), \
                    "Relative uncertainty thresholds are missing."
                assert "drift" in self.uct_rel_threshold and "diff" in self.uct_rel_threshold, \
                    "Both drift and diffusion relative uncertainty thresholds are required."

        np.random.seed(seed)
        torch.manual_seed(seed)

        n_trajs = initial_points.shape[0]
        dt = float(self.time_step)
        sqrt_dt = math.sqrt(dt)

        x = initial_points.clone().to(self.device)
        dtype = x.dtype

        cond_vec = x[:, :self.n_cond_onehot] if self.n_cond_onehot > 0 else None

        trajs = torch.full(
            (n_trajs, n_steps, self.input_dim),
            float("nan"),
            device=self.device,
            dtype=dtype,
        )
        trajs[:, 0, :] = x

        alive = torch.ones(n_trajs, dtype=torch.bool, device=self.device)
        alive_mask = torch.zeros(n_trajs, n_steps, dtype=torch.bool, device=self.device)
        alive_mask[:, 0] = True

        # stop_time = last valid index.
        # If a trajectory reaches the end, stop_time remains n_steps - 1.
        stop_time = torch.full(
            (n_trajs,),
            n_steps - 1,
            dtype=torch.long,
            device=self.device,
        )

        # stop_reason codes:
        # 0 = not stopped
        # 1 = high drift absolute uncertainty
        # 2 = high diffusion absolute uncertainty
        # 3 = high drift relative uncertainty
        # 4 = high diffusion relative uncertainty
        # 5 = Cholesky failure
        # 6 = non-finite update
        stop_reason = torch.zeros(n_trajs, dtype=torch.long, device=self.device)

        if return_info:
            drift_abs_hist = torch.full((n_trajs, n_steps), float("nan"), device=self.device, dtype=dtype)
            diff_abs_hist = torch.full((n_trajs, n_steps), float("nan"), device=self.device, dtype=dtype)
            drift_rel_hist = torch.full((n_trajs, n_steps), float("nan"), device=self.device, dtype=dtype)
            diff_rel_hist = torch.full((n_trajs, n_steps), float("nan"), device=self.device, dtype=dtype)
        else:
            drift_abs_hist = diff_abs_hist = drift_rel_hist = diff_rel_hist = None

        eye = torch.eye(self.dim, device=self.device, dtype=dtype).unsqueeze(0)

        print(f"Generating {n_trajs} trajectories of length {n_steps} with dt={dt}")
        if uncertainty_aware:
            print("Uncertainty-aware simulation is ON.")

        for it in tqdm(range(1, n_steps), desc="Simulating Trajectories"):
            if not torch.any(alive):
                break

            alive_idx = torch.where(alive)[0]
            x_alive = x[alive_idx]

            drift_mean, drift_var = self.eval_field_mean_var(x_alive, n_MC=n_MC, model_type="drift", return_cpu=False)
            diff_mean, diff_var = self.eval_field_mean_var(x_alive, n_MC=n_MC, model_type="diff", return_cpu=False)

            drift_mean = drift_mean.to(dtype=dtype)
            diff_mean = diff_mean.to(dtype=dtype)
            drift_var = drift_var.to(dtype=dtype).clamp_min(0.0)
            diff_var = diff_var.to(dtype=dtype).clamp_min(0.0)

            # ===== Compute scalar epistemic uncertainty scores =====
            drift_mean_flat = drift_mean.flatten(start_dim=1)
            diff_mean_flat = diff_mean.flatten(start_dim=1)
            drift_var_flat = drift_var.flatten(start_dim=1)
            diff_var_flat = diff_var.flatten(start_dim=1)

            drift_abs = torch.sqrt(torch.sum(drift_var_flat, dim=-1))
            diff_abs = torch.sqrt(torch.sum(diff_var_flat, dim=-1))

            drift_rel = drift_abs / (torch.norm(drift_mean_flat, dim=-1) + 1e-8)
            diff_rel = diff_abs / (torch.norm(diff_mean_flat, dim=-1) + 1e-8)

            if return_info:
                drift_abs_hist[alive_idx, it - 1] = drift_abs
                diff_abs_hist[alive_idx, it - 1] = diff_abs
                drift_rel_hist[alive_idx, it - 1] = drift_rel
                diff_rel_hist[alive_idx, it - 1] = diff_rel

            # ===== Stop trajectories before update if uncertainty is too high =====
            stop_now = torch.zeros(len(alive_idx), dtype=torch.bool, device=self.device)
            reason_now = torch.zeros(len(alive_idx), dtype=torch.long, device=self.device)

            if uncertainty_aware:
                drift_abs_thr = torch.as_tensor(self.uct_abs_threshold["drift"], device=self.device, dtype=dtype)
                diff_abs_thr = torch.as_tensor(self.uct_abs_threshold["diff"], device=self.device, dtype=dtype)

                bad = drift_abs > drift_abs_thr
                reason_now = torch.where(
                    bad & (reason_now == 0),
                    torch.tensor(1, device=self.device),
                    reason_now,
                )
                stop_now |= bad

                bad = diff_abs > diff_abs_thr
                reason_now = torch.where(
                    bad & (reason_now == 0),
                    torch.tensor(2, device=self.device),
                    reason_now,
                )
                stop_now |= bad

                if use_relative_uncertainty:
                    drift_rel_thr = torch.as_tensor(self.uct_rel_threshold["drift"], device=self.device, dtype=dtype)
                    diff_rel_thr = torch.as_tensor(self.uct_rel_threshold["diff"], device=self.device, dtype=dtype)

                    bad = drift_rel > drift_rel_thr
                    reason_now = torch.where(
                        bad & (reason_now == 0),
                        torch.tensor(3, device=self.device),
                        reason_now,
                    )
                    stop_now |= bad

                    bad = diff_rel > diff_rel_thr
                    reason_now = torch.where(
                        bad & (reason_now == 0),
                        torch.tensor(4, device=self.device),
                        reason_now,
                    )
                    stop_now |= bad

            if torch.any(stop_now):
                stopped_idx = alive_idx[stop_now]
                alive[stopped_idx] = False
                stop_time[stopped_idx] = it - 1
                stop_reason[stopped_idx] = reason_now[stop_now]

            keep = ~stop_now
            if not torch.any(keep):
                continue

            alive_idx = alive_idx[keep]
            x_alive = x_alive[keep]
            drift_mean = drift_mean[keep]
            diff_mean = diff_mean[keep]

            # ===== Optional finite-time correction =====
            if adjust_for_drift:
                diff_mean = diff_mean - 0.5 * torch.bmm(
                    drift_mean.unsqueeze(-1),
                    drift_mean.unsqueeze(1),
                ) * dt

            # Cholesky with small diagonal jitter.
            diff_for_chol = diff_mean + jitter * eye.expand(diff_mean.shape[0], -1, -1)

            L, chol_info = torch.linalg.cholesky_ex(diff_for_chol)
            chol_bad = chol_info > 0

            # Cholesky failure means the inferred diffusion is not simulation-valid here.
            # In uncertainty-aware mode, these trajectories are stopped.
            # In non-uncertainty-aware mode, we still stop them to avoid forced SPD projection.
            if torch.any(chol_bad):
                bad_idx = alive_idx[chol_bad]
                alive[bad_idx] = False
                stop_time[bad_idx] = it - 1
                stop_reason[bad_idx] = 5

                keep_chol = ~chol_bad
                if not torch.any(keep_chol):
                    continue

                alive_idx = alive_idx[keep_chol]
                x_alive = x_alive[keep_chol]
                drift_mean = drift_mean[keep_chol]
                L = L[keep_chol]

            # ===== Euler-Maruyama update =====
            noise = torch.randn(len(alive_idx), self.dim, device=self.device, dtype=dtype)
            stochastic_term = torch.bmm(L, noise.unsqueeze(-1)).squeeze(-1)

            state_update = drift_mean * dt + math.sqrt(2.0) * stochastic_term * sqrt_dt

            cond_alive = cond_vec[alive_idx] if cond_vec is not None else None
            full_update = self._build_state_update(
                state_update,
                cond_alive,
                len(alive_idx),
                dt,
            )

            x_next = x_alive + full_update

            finite = torch.isfinite(x_next).all(dim=-1)

            if torch.any(~finite):
                bad_idx = alive_idx[~finite]
                alive[bad_idx] = False
                stop_time[bad_idx] = it - 1
                stop_reason[bad_idx] = 6

            if torch.any(finite):
                good_idx = alive_idx[finite]
                x[good_idx] = x_next[finite]
                trajs[good_idx, it, :] = x_next[finite]
                alive_mask[good_idx, it] = True

        if return_info:
            return {
                "trajs": trajs.cpu(),
                "alive_mask": alive_mask.cpu(),
                "stop_time": stop_time.cpu(),
                "stop_reason": stop_reason.cpu(),
                "stop_reason_codes": {
                    0: "not_stopped",
                    1: "high_drift_abs_uncertainty",
                    2: "high_diff_abs_uncertainty",
                    3: "high_drift_rel_uncertainty",
                    4: "high_diff_rel_uncertainty",
                    5: "cholesky_failure",
                    6: "nonfinite_update",
                },
                "drift_abs_uncertainty": drift_abs_hist.cpu(),
                "diff_abs_uncertainty": diff_abs_hist.cpu(),
                "drift_rel_uncertainty": drift_rel_hist.cpu(),
                "diff_rel_uncertainty": diff_rel_hist.cpu(),
            }

        return trajs.cpu()
