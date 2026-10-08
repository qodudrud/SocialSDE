"""Stochastic force inference compatible with ``LangevinDiagnosticsBase``.

The implementation follows Frishman & Ronceray, Phys. Rev. X 10, 021009
(2020), and optionally applies the PASTIS information criterion of Gerardos &
Ronceray, Phys. Rev. Lett. 135, 167401 (2025), to the drift library.

The fitted object exposes the same field-evaluation hooks as the existing LBN
and function-backed Langevin classes, so the diagnostics implemented in
``LangevinDiagnosticsBase`` can be used without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
import math
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch

from model.LangevinDiagnosticsBase import LangevinDiagnosticsBase


Tensor = torch.Tensor


def _weighted_mean_and_scale(x: Tensor, weights: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    weights = weights / weights.sum()
    mean = torch.sum(weights[:, None] * x, dim=0)
    var = torch.sum(weights[:, None] * (x - mean).square(), dim=0)
    return mean, torch.sqrt(var.clamp_min(eps))


class ConstantBasis:
    """A single constant scalar basis function."""

    def __init__(self, n_features: int):
        self.n_features = int(n_features)
        self.names = ["1"]

    def fit(self, x: Tensor, weights: Optional[Tensor] = None) -> "ConstantBasis":
        self._validate(x)
        return self

    def _validate(self, x: Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected basis input [N, {self.n_features}], got {tuple(x.shape)}."
            )

    def __call__(self, x: Tensor) -> Tensor:
        self._validate(x)
        return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)

    def jacobian(self, x: Tensor) -> Tensor:
        self._validate(x)
        return torch.zeros((x.shape[0], 1, self.n_features), device=x.device, dtype=x.dtype)


class PolynomialBasis:
    """Scalar monomials up to a chosen total degree.

    Inputs are standardized using a weighted sample mean and standard
    deviation by default. The reported coefficients therefore refer to the
    standardized monomials, while evaluation transparently accepts the original
    coordinates.
    """

    def __init__(
        self,
        n_features: int,
        degree: int = 2,
        *,
        include_bias: bool = True,
        interaction_only: bool = False,
        standardize: bool = True,
        feature_names: Optional[Sequence[str]] = None,
        scale_eps: float = 1e-12,
    ):
        if n_features < 1:
            raise ValueError("n_features must be positive.")
        if degree < 0:
            raise ValueError("degree must be non-negative.")
        self.n_features = int(n_features)
        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)
        self.standardize = bool(standardize)
        self.scale_eps = float(scale_eps)
        self.feature_names = list(feature_names or [f"x{i}" for i in range(n_features)])
        if len(self.feature_names) != self.n_features:
            raise ValueError("feature_names must have length n_features.")

        exponents: list[list[int]] = []
        if self.include_bias:
            exponents.append([0] * self.n_features)
        for total_degree in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(self.n_features), total_degree):
                exponent = [0] * self.n_features
                for j in combo:
                    exponent[j] += 1
                if self.interaction_only and max(exponent) > 1:
                    continue
                exponents.append(exponent)
        if not exponents:
            raise ValueError("The requested polynomial library is empty.")

        self._exponents_list = exponents
        self.names = [self._format_name(e) for e in exponents]
        self.center_: Optional[Tensor] = None
        self.scale_: Optional[Tensor] = None

    def _format_name(self, exponent: Sequence[int]) -> str:
        factors = []
        for name, power in zip(self.feature_names, exponent):
            if power == 1:
                factors.append(name)
            elif power > 1:
                factors.append(f"{name}^{power}")
        return "*".join(factors) if factors else "1"

    def _validate(self, x: Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected basis input [N, {self.n_features}], got {tuple(x.shape)}."
            )

    def fit(self, x: Tensor, weights: Optional[Tensor] = None) -> "PolynomialBasis":
        self._validate(x)
        if weights is None:
            weights = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        if self.standardize:
            self.center_, self.scale_ = _weighted_mean_and_scale(x, weights, self.scale_eps)
        else:
            self.center_ = torch.zeros(self.n_features, device=x.device, dtype=x.dtype)
            self.scale_ = torch.ones(self.n_features, device=x.device, dtype=x.dtype)
        return self

    def _normalization(self, x: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(x)
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Call basis.fit(...) before evaluating the basis.")
        center = self.center_.to(device=x.device, dtype=x.dtype)
        scale = self.scale_.to(device=x.device, dtype=x.dtype)
        return (x - center) / scale, scale

    def _exponents(self, x: Tensor) -> Tensor:
        return torch.as_tensor(self._exponents_list, device=x.device, dtype=torch.long)

    def __call__(self, x: Tensor) -> Tensor:
        z, _ = self._normalization(x)
        exponent = self._exponents(x)
        return torch.prod(z[:, None, :].pow(exponent[None, :, :]), dim=-1)

    def jacobian(self, x: Tensor) -> Tensor:
        z, scale = self._normalization(x)
        exponent = self._exponents(x)
        derivatives = []
        for j in range(self.n_features):
            reduced = exponent.clone()
            reduced[:, j] = torch.clamp(reduced[:, j] - 1, min=0)
            derivative = torch.prod(z[:, None, :].pow(reduced[None, :, :]), dim=-1)
            derivative = derivative * exponent[None, :, j].to(x.dtype) / scale[j]
            derivatives.append(derivative)
        return torch.stack(derivatives, dim=-1)


class CallableBasis:
    """Wrap a differentiable callable as a scalar basis library.

    ``func(x)`` must return ``[N, n_basis]``. If no analytic Jacobian callable
    is supplied, the Jacobian is obtained with PyTorch autograd.
    """

    def __init__(
        self,
        n_features: int,
        func: Callable[[Tensor], Tensor],
        names: Sequence[str],
        jacobian_func: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        self.n_features = int(n_features)
        self.func = func
        self.names = list(names)
        self.jacobian_func = jacobian_func
        if not self.names:
            raise ValueError("names must contain at least one basis-function name.")

    def fit(self, x: Tensor, weights: Optional[Tensor] = None) -> "CallableBasis":
        self._validate_input(x)
        self._validate_output(self.func(x), x.shape[0])
        return self

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected basis input [N, {self.n_features}], got {tuple(x.shape)}."
            )

    def _validate_output(self, y: Tensor, n: int) -> None:
        if not torch.is_tensor(y) or y.shape != (n, len(self.names)):
            raise ValueError(
                f"Basis callable must return [{n}, {len(self.names)}], got {getattr(y, 'shape', None)}."
            )

    def __call__(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        y = self.func(x)
        self._validate_output(y, x.shape[0])
        return y

    def jacobian(self, x: Tensor) -> Tensor:
        if self.jacobian_func is not None:
            jac = self.jacobian_func(x)
            expected = (x.shape[0], len(self.names), self.n_features)
            if jac.shape != expected:
                raise ValueError(f"Jacobian callable must return {expected}, got {tuple(jac.shape)}.")
            return jac

        x_grad = x.detach().clone().requires_grad_(True)
        values = self(x_grad)
        columns = []
        for k in range(values.shape[1]):
            grad = torch.autograd.grad(
                values[:, k].sum(), x_grad, retain_graph=True, create_graph=False
            )[0]
            columns.append(grad)
        return torch.stack(columns, dim=1)


@dataclass(frozen=True)
class _SelectionFit:
    active: tuple[int, ...]
    coefficients: Tensor
    information: float
    criterion: float


class Langevin_from_SFI(LangevinDiagnosticsBase):
    """SFI drift/diffusion estimator with the diagnostics-base interface.

    Parameters
    ----------
    basis:
        Scalar basis library for the drift. It is replicated along each output
        coordinate, matching the vector basis used by PASTIS. By default a
        standardized polynomial library is built from ``args.sfi_drift_degree``
        (default 2).
    diffusion_basis:
        Scalar library for the diffusion tensor. By default it is constant when
        args.sfi_diffusion_degree is 0; a positive degree builds a polynomial
        library. An explicit polynomial or callable library can also be supplied.
    feature_indices:
        Columns of the full ``xs`` tensor supplied to the drift basis. The
        default is the physical state block defined by ``_state_slice()``.
    drift_estimator:
        ``"ito"`` implements PRX Eq. (7). ``"stratonovich"`` implements the
        measurement-noise-robust conversion in PRX Eq. (14).
    diffusion_estimator:
        ``"standard"`` implements PRX Eq. (12), ``"measurement_noise"``
        implements Eqs. (13)/(G13), ``"large_dt"`` implements PASTIS Eq. (D2)'s
        three-point diffusion estimate, and ``"residual"`` subtracts the fitted
        drift before estimating the increment covariance.
    selection:
        ``None`` fits the complete drift library. ``"pastis"``, ``"aic"``, or
        ``"bic"`` performs vector-term selection. Selection currently requires
        the ideal-data Itô estimator; corrected sparse likelihoods are not mixed
        silently with the ideal-data criterion.
    """

    _DRIFT_ESTIMATORS = {"ito", "stratonovich"}
    _DIFFUSION_ESTIMATORS = {"standard", "measurement_noise", "large_dt", "residual"}
    _SELECTION_CRITERIA = {None, "pastis", "aic", "bic"}

    def __init__(
        self,
        args,
        basis=None,
        diffusion_basis=None,
        *,
        feature_indices: Optional[Sequence[int]] = None,
        diffusion_feature_indices: Optional[Sequence[int]] = None,
        drift_estimator: str = "ito",
        diffusion_estimator: str = "standard",
        selection: Optional[str] = None,
        pastis_p: float = 1e-3,
        n_random_starts: int = 4,
        ridge: float = 1e-10,
        rcond: float = 1e-10,
        min_diffusion: float = 1e-8,
        pair_dt_rtol: float = 1e-5,
        seed: int = 0,
        device=None,
        fit_dtype: torch.dtype = torch.float64,
    ):
        super().__init__(args, device=device)
        self.feature_indices = self._resolve_feature_indices(feature_indices)
        self.diffusion_feature_indices = list(
            self.feature_indices if diffusion_feature_indices is None else diffusion_feature_indices
        )

        drift_degree = int(getattr(args, "sfi_drift_degree", 2))
        diffusion_degree = int(getattr(args, "sfi_diffusion_degree", 0))

        # Drift basis
        if basis is None:
            self.basis = PolynomialBasis(
                len(self.feature_indices),
                degree=drift_degree,
            )
        else:
            self.basis = basis

        # Diffusion basis
        if diffusion_basis is None:
            if diffusion_degree == 0:
                self.diffusion_basis = ConstantBasis(
                    len(self.diffusion_feature_indices)
                )
            else:
                self.diffusion_basis = PolynomialBasis(
                    len(self.diffusion_feature_indices),
                    degree=diffusion_degree,
                )
        else:
            self.diffusion_basis = diffusion_basis

            
        self._validate_basis_dimension(self.basis, self.feature_indices, "basis")
        self._validate_basis_dimension(
            self.diffusion_basis, self.diffusion_feature_indices, "diffusion_basis"
        )

        if drift_estimator not in self._DRIFT_ESTIMATORS:
            raise ValueError(f"drift_estimator must be one of {sorted(self._DRIFT_ESTIMATORS)}.")
        if diffusion_estimator not in self._DIFFUSION_ESTIMATORS:
            raise ValueError(
                f"diffusion_estimator must be one of {sorted(self._DIFFUSION_ESTIMATORS)}."
            )
        if selection not in self._SELECTION_CRITERIA:
            raise ValueError("selection must be None, 'pastis', 'aic', or 'bic'.")
        if selection is not None and drift_estimator != "ito":
            raise ValueError(
                "Sparse selection currently requires drift_estimator='ito'. "
                "Fit the complete basis for the PRX Eq. (14) correction."
            )
        if not (0 < pastis_p <= 1):
            raise ValueError("pastis_p must lie in (0, 1].")

        self.drift_estimator = drift_estimator
        self.diffusion_estimator = diffusion_estimator
        self.selection = selection
        self.pastis_p = float(pastis_p)
        self.n_random_starts = int(n_random_starts)
        self.ridge = float(ridge)
        self.rcond = float(rcond)
        self.min_diffusion = float(min_diffusion)
        self.pair_dt_rtol = float(pair_dt_rtol)
        self.seed = int(seed)
        self.fit_dtype = fit_dtype

        self.is_fitted = False
        self.drift_coefficients_: Optional[Tensor] = None
        self.diffusion_coefficients_: Optional[Tensor] = None
        self.mean_diffusion_: Optional[Tensor] = None
        self.gram_: Optional[Tensor] = None
        self.diffusion_gram_: Optional[Tensor] = None
        self.total_time_: Optional[float] = None
        self.information_: Optional[float] = None
        self.criterion_: Optional[float] = None
        self.selected_terms_: list[str] = []
        self.selected_indices_: tuple[int, ...] = ()
        self.relative_error_estimate_: float = math.inf

    def _diffusion_is_constant(self):
        return isinstance(self.diffusion_basis, ConstantBasis)

    def _resolve_feature_indices(self, feature_indices: Optional[Sequence[int]]) -> list[int]:
        if feature_indices is None:
            feature_indices = range(self._state_slice().start, self._state_slice().stop)
        result = [int(i) for i in feature_indices]
        if not result or len(set(result)) != len(result):
            raise ValueError("feature_indices must be a non-empty sequence without duplicates.")
        if min(result) < 0 or max(result) >= self.input_dim:
            raise ValueError("feature_indices contains a column outside [0, input_dim).")
        return result

    def _validate_basis_dimension(self, basis, indices: Sequence[int], label: str) -> None:
        if not hasattr(basis, "n_features") or int(basis.n_features) != len(indices):
            raise ValueError(f"{label}.n_features must equal the number of selected input columns.")
        if not hasattr(basis, "names") or len(basis.names) == 0:
            raise ValueError(f"{label} must expose a non-empty names sequence.")

    def _as_fit_tensor(self, x) -> Tensor:
        return torch.as_tensor(x, device=self.device, dtype=self.fit_dtype)

    def _as_eval_tensor(self, x) -> Tensor:
        if torch.is_tensor(x):
            if not x.is_floating_point():
                x = x.float()
            return x.to(device=self.device)
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    @staticmethod
    def _basis_input(x: Tensor, indices: Sequence[int]) -> Tensor:
        index = torch.as_tensor(indices, device=x.device, dtype=torch.long)
        return torch.index_select(x, dim=1, index=index)

    def _validate_fit_inputs(
        self, xs, dxs, dts, series_slices
    ) -> tuple[Tensor, Tensor, Tensor, list[slice]]:
        xs = self._as_fit_tensor(xs)
        dxs = self._as_fit_tensor(dxs)
        dts = self._as_fit_tensor(dts).reshape(-1)
        if xs.ndim != 2 or xs.shape[1] != self.input_dim:
            raise ValueError(f"xs must have shape [N, {self.input_dim}].")
        if dxs.ndim != 2 or dxs.shape[1] != self.dim:
            raise ValueError(f"dxs must have shape [N, {self.dim}].")
        if not (xs.shape[0] == dxs.shape[0] == dts.shape[0]):
            raise ValueError("xs, dxs, and dts must have the same first dimension.")
        if xs.shape[0] == 0:
            raise ValueError("At least one transition is required.")
        if not torch.isfinite(xs).all() or not torch.isfinite(dxs).all():
            raise ValueError("xs and dxs must contain only finite values.")
        if not torch.isfinite(dts).all() or torch.any(dts <= 0):
            raise ValueError("All dts must be finite and strictly positive.")

        if series_slices is None:
            series_slices = [slice(0, xs.shape[0])]
        slices = []
        for sl in series_slices:
            start = 0 if sl.start is None else int(sl.start)
            stop = xs.shape[0] if sl.stop is None else int(sl.stop)
            if not (0 <= start <= stop <= xs.shape[0]):
                raise ValueError(f"Invalid trajectory slice {sl}.")
            slices.append(slice(start, stop))
        return xs, dxs, dts, slices

    def _next_inputs(self, xs: Tensor, dxs: Tensor, dts: Tensor) -> Tensor:
        result = xs.clone()
        result[:, self._state_slice()] = result[:, self._state_slice()] + dxs
        if self.include_time:
            result[:, -1] = result[:, -1] + dts
        return result

    def _pair_mask(self, n: int, slices: Sequence[slice]) -> Tensor:
        mask = torch.zeros(n, device=self.device, dtype=torch.bool)
        for sl in slices:
            if sl.stop - sl.start >= 2:
                mask[sl.start + 1 : sl.stop] = True
        return mask

    def _local_diffusion(
        self,
        dxs: Tensor,
        dts: Tensor,
        slices: Sequence[slice],
        *,
        residual: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        increments = dxs if residual is None else residual
        n = dxs.shape[0]
        outer = torch.einsum("ni,nj->nij", increments, increments)

        if self.diffusion_estimator in {"standard", "residual"}:
            return outer / (2.0 * dts[:, None, None]), torch.ones(
                n, device=self.device, dtype=torch.bool
            )

        mask = self._pair_mask(n, slices)
        if not mask.any():
            raise ValueError("The selected diffusion correction requires two adjacent transitions.")
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        previous = indices - 1
        dt_now = dts[indices]
        dt_previous = dts[previous]
        relative_gap = torch.abs(dt_now - dt_previous) / torch.maximum(dt_now, dt_previous)
        if torch.any(relative_gap > self.pair_dt_rtol):
            raise ValueError(
                f"diffusion_estimator='{self.diffusion_estimator}' assumes equal adjacent dts. "
                "Use 'standard' or 'residual' for irregular sampling."
            )

        local = torch.full(
            (n, self.dim, self.dim), float("nan"), device=self.device, dtype=self.fit_dtype
        )
        current_dx = dxs[indices]
        previous_dx = dxs[previous]
        if self.diffusion_estimator == "large_dt":
            delta = current_dx - previous_dx
            local[indices] = torch.einsum("ni,nj->nij", delta, delta) / (
                4.0 * dt_now[:, None, None]
            )
        else:
            # PRX Eq. (13), written explicitly in symmetric tensor form as G13.
            cc = torch.einsum("ni,nj->nij", current_dx, current_dx)
            pp = torch.einsum("ni,nj->nij", previous_dx, previous_dx)
            cp = torch.einsum("ni,nj->nij", current_dx, previous_dx)
            pc = cp.transpose(-1, -2)
            local[indices] = (cc + pp + 2.0 * cp + 2.0 * pc) / (
                4.0 * dt_now[:, None, None]
            )
        return local, mask

    def _project_psd(self, matrix: Tensor) -> Tensor:
        matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        eigenvalues = eigenvalues.clamp_min(self.min_diffusion)
        return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)

    def _weighted_mean_diffusion(self, local: Tensor, dts: Tensor, mask: Tensor) -> Tensor:
        weights = dts[mask]
        mean = torch.einsum("n,nij->ij", weights, local[mask]) / weights.sum()
        return self._project_psd(mean)

    def _solve(self, gram: Tensor, rhs: Tensor) -> Tensor:
        gram = 0.5 * (gram + gram.transpose(-1, -2))
        if self.ridge > 0:
            scale = torch.trace(gram).abs() / max(gram.shape[0], 1)
            gram = gram + self.ridge * scale.clamp_min(1.0) * torch.eye(
                gram.shape[0], device=gram.device, dtype=gram.dtype
            )
        return torch.linalg.pinv(gram, rtol=self.rcond, hermitian=True) @ rhs

    def _scalar_gram(self, phi: Tensor, dts: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = torch.ones(phi.shape[0], device=phi.device, dtype=torch.bool)
        weights = dts[mask]
        total_time = weights.sum()
        return (phi[mask].T * weights) @ phi[mask] / total_time

    def _criterion_penalty(self, n_terms: int, n_library: int, total_time: float) -> float:
        if self.selection is None:
            return 0.0
        if self.selection == "pastis":
            return n_terms * math.log(n_library / self.pastis_p)
        if self.selection == "aic":
            return float(n_terms)
        return 0.5 * n_terms * math.log(max(total_time, 1.0))

    def _select_drift(
        self, gram_scalar: Tensor, projected_velocity: Tensor, diffusion: Tensor, total_time: float
    ) -> _SelectionFit:
        n_basis = gram_scalar.shape[0]
        n_library = self.dim * n_basis
        diffusion_inv = torch.linalg.inv(diffusion)
        gram_vector = torch.kron(diffusion_inv.contiguous(), gram_scalar.contiguous())
        score_vector = (projected_velocity @ diffusion_inv).T.reshape(-1)
        cache: dict[tuple[int, ...], _SelectionFit] = {}

        def evaluate(active_iter: Iterable[int]) -> _SelectionFit:
            active = tuple(sorted(set(int(i) for i in active_iter)))
            if active in cache:
                return cache[active]
            coefficients = torch.zeros(n_library, device=self.device, dtype=self.fit_dtype)
            if active:
                index = torch.as_tensor(active, device=self.device, dtype=torch.long)
                sub_gram = gram_vector[index][:, index]
                sub_score = score_vector[index]
                fitted = self._solve(sub_gram, sub_score)
                coefficients[index] = fitted
                information = max(0.0, float((0.25 * total_time * fitted.dot(sub_score)).item()))
            else:
                information = 0.0
            criterion = information - self._criterion_penalty(
                len(active), n_library, total_time
            )
            result = _SelectionFit(active, coefficients, information, criterion)
            cache[active] = result
            return result

        def climb(start: Iterable[int]) -> _SelectionFit:
            current = evaluate(start)
            while True:
                active_set = set(current.active)
                best = current
                for candidate in range(n_library):
                    toggled = active_set.symmetric_difference({candidate})
                    trial = evaluate(toggled)
                    if trial.criterion > best.criterion + 1e-12:
                        best = trial
                if best.active == current.active:
                    return current
                current = best

        rng = np.random.default_rng(self.seed)
        starts: list[Iterable[int]] = [(), range(n_library)]
        for _ in range(max(0, self.n_random_starts)):
            size = int(rng.integers(0, n_library + 1))
            starts.append(rng.choice(n_library, size=size, replace=False).tolist())
        fits = [climb(start) for start in starts]
        return max(fits, key=lambda fit: fit.criterion)

    def _basis_state_jacobian(self, xs: Tensor) -> Tensor:
        z = self._basis_input(xs, self.feature_indices)
        jacobian = self.basis.jacobian(z)
        state_columns = list(range(self._state_slice().start, self._state_slice().stop))
        column_to_local = {column: j for j, column in enumerate(self.feature_indices)}
        state_jacobian = torch.zeros(
            (xs.shape[0], len(self.basis.names), self.dim),
            device=xs.device,
            dtype=xs.dtype,
        )
        for state_dimension, full_column in enumerate(state_columns):
            if full_column in column_to_local:
                state_jacobian[:, :, state_dimension] = jacobian[
                    :, :, column_to_local[full_column]
                ]
        return state_jacobian

    def _fit_drift(
        self,
        xs: Tensor,
        dxs: Tensor,
        dts: Tensor,
        phi: Tensor,
        local_diffusion: Tensor,
        local_mask: Tensor,
        diffusion: Tensor,
    ) -> tuple[Tensor, float, float, tuple[int, ...]]:
        total_time = float(dts.sum().item())
        gram = self._scalar_gram(phi, dts)

        if self.drift_estimator == "ito":
            projected_velocity = phi.T @ dxs / dts.sum()
        else:
            fit_mask = local_mask
            x_next = self._next_inputs(xs, dxs, dts)
            midpoint = 0.5 * (xs + x_next)
            phi_midpoint = self.basis(self._basis_input(midpoint, self.feature_indices))
            jacobian = self._basis_state_jacobian(xs)
            normalizer = dts[fit_mask].sum()
            velocity = phi_midpoint[fit_mask].T @ dxs[fit_mask] / normalizer
            correction = torch.einsum(
                "n,nuv,nkv->ku",
                dts[fit_mask],
                local_diffusion[fit_mask],
                jacobian[fit_mask],
            ) / normalizer
            projected_velocity = velocity - correction

        if self.selection is None:
            coefficients = self._solve(gram, projected_velocity)
            diffusion_inv = torch.linalg.inv(diffusion)
            information_tensor = torch.einsum(
                "ku,kl,lv,uv->", coefficients, gram, coefficients, diffusion_inv
            )
            information = max(0.0, 0.25 * total_time * float(information_tensor.item()))
            active = tuple(range(self.dim * phi.shape[1]))
            criterion = information
        else:
            selection_fit = self._select_drift(
                gram, projected_velocity, diffusion, total_time
            )
            coefficients = selection_fit.coefficients.reshape(self.dim, phi.shape[1]).T
            information = selection_fit.information
            criterion = selection_fit.criterion
            active = selection_fit.active
        self.gram_ = gram.detach().clone()
        return coefficients, information, criterion, active

    def _fit_diffusion_projection(
        self, xs: Tensor, dts: Tensor, local: Tensor, mask: Tensor
    ) -> Tensor:
        z = self._basis_input(xs, self.diffusion_feature_indices)
        psi = self.diffusion_basis(z)
        gram = self._scalar_gram(psi, dts, mask)
        weights = dts[mask]
        total_time = weights.sum()
        target = torch.einsum(
            "n,nk,nij->kij", weights, psi[mask], local[mask]
        ) / total_time
        coefficients = self._solve(gram, target.reshape(target.shape[0], -1)).reshape(
            target.shape
        )
        self.diffusion_gram_ = gram.detach().clone()
        return 0.5 * (coefficients + coefficients.transpose(-1, -2))

    def fit(self, xs, dxs, dts, series_slices=None) -> "Langevin_from_SFI":
        """Fit drift and diffusion projections from flattened transitions."""
        xs, dxs, dts, slices = self._validate_fit_inputs(xs, dxs, dts, series_slices)
        self.basis.fit(self._basis_input(xs, self.feature_indices), weights=dts)
        self.diffusion_basis.fit(
            self._basis_input(xs, self.diffusion_feature_indices), weights=dts
        )
        phi = self.basis(self._basis_input(xs, self.feature_indices))

        local, local_mask = self._local_diffusion(dxs, dts, slices)
        mean_diffusion = self._weighted_mean_diffusion(local, dts, local_mask)
        coefficients, information, criterion, active = self._fit_drift(
            xs, dxs, dts, phi, local, local_mask, mean_diffusion
        )

        if self.diffusion_estimator == "residual":
            fitted_drift = phi @ coefficients
            residual = dxs - fitted_drift * dts[:, None]
            local, local_mask = self._local_diffusion(
                dxs, dts, slices, residual=residual
            )
            mean_diffusion = self._weighted_mean_diffusion(local, dts, local_mask)
            coefficients, information, criterion, active = self._fit_drift(
                xs, dxs, dts, phi, local, local_mask, mean_diffusion
            )

        diffusion_coefficients = self._fit_diffusion_projection(
            xs, dts, local, local_mask
        )

        self.drift_coefficients_ = coefficients.detach().clone()
        self.diffusion_coefficients_ = diffusion_coefficients.detach().clone()
        self.mean_diffusion_ = mean_diffusion.detach().clone()
        self.total_time_ = float(dts.sum().item())
        self.information_ = float(information)
        self.criterion_ = float(criterion)
        self.selected_indices_ = active
        n_basis = len(self.basis.names)
        self.selected_terms_ = [
            f"F[{index // n_basis}] <- {self.basis.names[index % n_basis]}"
            for index in active
        ]
        n_parameters = len(active)
        self.relative_error_estimate_ = (
            n_parameters / (2.0 * self.information_)
            if self.information_ > 0
            else math.inf
        )
        self.is_fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit(xs, dxs, dts, series_slices) before field evaluation.")

    def _eval_drift(self, pts: Tensor) -> Tensor:
        self._require_fitted()
        phi = self.basis(self._basis_input(pts, self.feature_indices))
        coefficients = self.drift_coefficients_.to(device=pts.device, dtype=pts.dtype)
        return phi @ coefficients

    def _eval_diffusion(self, pts: Tensor) -> Tensor:
        self._require_fitted()
        psi = self.diffusion_basis(
            self._basis_input(pts, self.diffusion_feature_indices)
        )
        coefficients = self.diffusion_coefficients_.to(device=pts.device, dtype=pts.dtype)
        diffusion = torch.einsum("nk,kij->nij", psi, coefficients)
        return self._project_psd(diffusion)

    def _drift_prediction_variance(self, pts: Tensor) -> Tensor:
        phi = self.basis(self._basis_input(pts, self.feature_indices))
        n_basis = phi.shape[1]
        result = torch.zeros((pts.shape[0], self.dim), device=pts.device, dtype=pts.dtype)
        diffusion_diag = torch.diagonal(
            self.mean_diffusion_.to(device=pts.device, dtype=pts.dtype)
        )
        gram = self.gram_.to(device=pts.device, dtype=pts.dtype)
        total_time = max(float(self.total_time_), torch.finfo(pts.dtype).eps)

        for mu in range(self.dim):
            active_scalar = sorted(
                index % n_basis
                for index in self.selected_indices_
                if index // n_basis == mu
            )
            if not active_scalar:
                continue
            index = torch.as_tensor(active_scalar, device=pts.device, dtype=torch.long)
            sub_gram = gram[index][:, index]
            sub_inv = torch.linalg.pinv(sub_gram, rtol=self.rcond, hermitian=True)
            sub_phi = phi[:, index]
            leverage = torch.einsum("nk,kl,nl->n", sub_phi, sub_inv, sub_phi)
            result[:, mu] = (2.0 * diffusion_diag[mu] / total_time) * leverage.clamp_min(0)
        return result

    def eval_field_mean_var(
        self,
        pts,
        n_MC: int = 20,
        model_type: str = "drift",
        store_MC_outputs: bool = False,
        return_cpu: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del n_MC, store_MC_outputs
        pts = self._as_eval_tensor(pts)
        if pts.ndim != 2 or pts.shape[1] != self.input_dim:
            raise ValueError(f"pts must have shape [N, {self.input_dim}].")
        if model_type == "drift":
            mean = self._eval_drift(pts)
            variance = self._drift_prediction_variance(pts)
        elif model_type == "diff":
            mean = self._eval_diffusion(pts)
            # The PRX diffusion error is a projection-level bound, not an
            # elementwise posterior variance; returning zeros avoids pretending
            # otherwise while preserving the diagnostics-base contract.
            variance = torch.zeros_like(mean)
        else:
            raise ValueError("model_type must be either 'drift' or 'diff'.")
        if return_cpu:
            return mean.detach().cpu(), variance.detach().cpu()
        return mean, variance

    def _eval_field_with_grad(self, pts, model_type: str = "drift") -> Tensor:
        pts = self._as_eval_tensor(pts)
        if model_type == "drift":
            return self._eval_drift(pts)
        if model_type == "diff":
            return self._eval_diffusion(pts)
        raise ValueError("model_type must be either 'drift' or 'diff'.")

    def eval_physical_force(self, pts, *, return_cpu: bool = False) -> Tensor:
        """Evaluate ``F = Phi - div(D)`` for state-dependent diffusion (PRX Eq. 15)."""
        pts = self._as_eval_tensor(pts)
        if not pts.requires_grad:
            pts = pts.detach().clone().requires_grad_(True)
        drift = self._eval_drift(pts)
        diffusion = self._eval_diffusion(pts)
        divergence = torch.zeros_like(drift)
        if diffusion.requires_grad:
            for mu in range(self.dim):
                for nu in range(self.dim):
                    gradient = torch.autograd.grad(
                        diffusion[:, mu, nu].sum(),
                        pts,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    if gradient is not None:
                        divergence[:, mu] = divergence[:, mu] + gradient[
                            :, self._state_slice().start + nu
                        ]
        force = drift - divergence
        return force.detach().cpu() if return_cpu else force

    def fit_summary(self) -> dict:
        """Return data-only SFI/PASTIS diagnostics for the fitted projection."""
        self._require_fitted()
        return {
            "drift_estimator": self.drift_estimator,
            "diffusion_estimator": self.diffusion_estimator,
            "selection": self.selection,
            "total_time": self.total_time_,
            "information_nats": self.information_,
            "capacity_nats_per_time": self.information_ / self.total_time_,
            "criterion": self.criterion_,
            "n_library_terms": self.dim * len(self.basis.names),
            "n_selected_terms": len(self.selected_indices_),
            "selected_terms": list(self.selected_terms_),
            "relative_squared_coefficient_error_estimate": self.relative_error_estimate_,
            "mean_diffusion": self.mean_diffusion_.detach().cpu(),
        }


__all__ = [
    "CallableBasis",
    "ConstantBasis",
    "Langevin_from_SFI",
    "PolynomialBasis",
]
