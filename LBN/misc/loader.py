import pickle
from pathlib import Path


import numpy as np
import torch

# data handling
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence
import pandas as pd



def load_dataset(data_path: str):
    """
    Load a dataset and infer its content label from the filename.

    Args:
        data_path: Input path. A filename containing 'synthetic' is read as
            pickle; all other files are read as CSV.
    Returns:
        (data, data_content): The loaded object and one of 'synthetic',
        'DIG', 'seshat', 'ngram', or 'unknown'.
    """
    data_path = Path(data_path)
    file_name = data_path.name
    file_name_lower = file_name.lower()

    data_content = None
    if 'synthetic' in file_name_lower:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        data_content = 'synthetic'
        print("Loading synthetic data from:", data_path)

    else:
        data = pd.read_csv(data_path)
        if 'dig' in file_name_lower:
            data_content = 'DIG'
        elif 'seshat' in file_name_lower:
            data_content = 'seshat'
        elif 'ngram' in file_name_lower:
            data_content = 'ngram'
        else:
            data_content = 'unknown'
        print("Loading data from:", data_path, "with content type:", data_content)

    return data, data_content


def dataset_to_langevin_data(
        data_content: str, 
        data_df: pd.DataFrame, 
        args
    ):
    """
    Processes latent data to generate tensors for time-series models.

    Args:
        data_content: The type of data, e.g., 'dig', 'seshat', 'ngram', or 'synthetic'.
        data_df: A panel DataFrame, or a trajectory array for synthetic data.
        args: An object containing the following attributes:
            - time_step: Scaling factor for time intervals.
            - include_time: Whether to include time as a feature in the output tensors.
            - start_year: The start year for filtering the data (optional).
            - end_year: The end year for filtering the data (optional).
    Returns:
        A dictionary containing:
            - xs: Tensor of shape (N, input_dim); optional conditioning, physical states, then optional time.
            - dxs: Tensor of shape (N, D) representing the state changes.
            - dts: Tensor of shape (N, 1) representing the time intervals.
            - series_slices: List of slice objects indicating the start and end indices for each group.
            - valid_ids: List of unique identifiers corresponding to each group.
            - valid_times: List of np.ndarray containing time points for each group.
    """
    # Extract parameters from args with default values
    time_step = getattr(args, "time_step", 0.01)
    include_time = getattr(args, "include_time", False)
    start_year = getattr(args, "start_year", None)
    end_year = getattr(args, "end_year", None)
    
    data_content = data_content.lower()  # Normalize data_content to lowercase for consistency

    # Call the appropriate processing function based on data_content
    if data_content == 'synthetic':
        return synthetic_to_langevin_data(data_df, obs_time_step=time_step)
    
    elif data_content == 'dig':
        return panel_to_langevin_data(
            data_df, 
            id_col='iso3',
            state_cols=['DMC_PC1', 'INQ_PC1', 'GDP_PC1'],
            time_col='year',
            time_unit = 1.0,
            time_step = time_step, 
            timestep_limit=10,
            include_time = include_time,
            start_year = start_year, 
            end_year = end_year
        )
    elif data_content == 'seshat':
        return panel_to_langevin_data(
            data_df, 
            id_col='nga_id',
            state_cols=['Scale_PC1', 'Comp_w'],
            time_col='year',
            time_unit = 0.01,   # Scale years to centuries for Seshat data
            time_step = time_step, 
            timestep_limit=5,
            include_time = include_time,
            start_year = start_year, 
            end_year = end_year
        )
    elif data_content in ['ngram', 'language']:
        domain_indicator = getattr(args, "domain_indicator", None)
        df = data_df.copy()

        if domain_indicator is None:
            df = df[df['emo2rat_indicator'].isin([1, -1])].reset_index(drop=True)
            condition_col = 'emo2rat_indicator'
            condition_name_map = {1: 'rationality', -1: 'emotionality'}
        elif domain_indicator in [1, -1]:
            df = df[df['emo2rat_indicator'] == domain_indicator].reset_index(drop=True)
            condition_col = None
            condition_name_map = None
        else:
            raise ValueError("domain_indicator should be 1 (rationality), -1 (emotionality), or None.")
        
        return panel_to_langevin_data(
            df, 
            id_col='word',
            state_cols=['z_score'],
            time_col='year',
            time_unit = 1.0,
            time_step = time_step, 
            include_time = include_time,
            start_year = start_year, 
            end_year = end_year,
            condition_col = condition_col,
            condition_name_map = condition_name_map
        )
    
    else:
        try:
            condition_col=getattr(args, "condition_col", None)
            condition_name_map=getattr(args, "condition_name_map", None)
            return panel_to_langevin_data(
                data_df, 
                id_col=args.id_col,
                state_cols=args.state_cols,
                time_col=args.time_col,
                time_unit = args.time_unit,
                time_step = time_step, 
                include_time = include_time,
                start_year = start_year, 
                end_year = end_year,
                condition_col = condition_col,
                condition_name_map = condition_name_map
            )
        
        except Exception as e:
            raise ValueError(
                f"Error processing data_content '{data_content}'. Please ensure that the required arguments are provided in 'args'."
                f"\n Error: {e}"
            )



def synthetic_to_langevin_data(
        trajs: Union[np.ndarray, torch.Tensor], 
        obs_time_step: float = 0.01, 
        Langevin_type: str = 'OLE',
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
    """
    Convert synthetic trajectories into flattened OLE training transitions.

    Args:
        trajs: Array or tensor with shape (n_trajs, n_steps, n_dims).
        obs_time_step: Time interval between consecutive observations.
        Langevin_type: Only 'OLE' is implemented; other values raise
            NotImplementedError.
        dtype: Output tensor dtype.
        device: Output tensor device.
    Returns:
        A dictionary with xs and dxs of shape (N, n_dims), dts of shape
        (N, 1), series_slices, trajectory-index valid_ids, and per-trajectory
        valid_times for transition start points, where N = n_trajs * (n_steps - 1).
    """
    if Langevin_type != 'OLE':
        raise NotImplementedError("Only 'OLE' is currently implemented.")

    if isinstance(trajs, torch.Tensor):
        trajs_t = trajs.detach().to(device=device, dtype=dtype)
    else:
        trajs_t = torch.as_tensor(trajs, dtype=dtype, device=device)
    if trajs_t.ndim != 3:
        raise ValueError("trajs must have shape (n_trajs, n_steps, n_dims)")

    n_trajs, n_steps, n_dims = trajs_t.shape
    if n_steps < 2:
        raise ValueError("Each trajectory must have at least 2 time steps.")
    
    n_pairs = n_steps - 1

    series_slices = [
        slice(i * n_pairs, (i + 1) * n_pairs, None)
        for i in range(n_trajs)
    ]

    xs = trajs_t[:, :-1, :].reshape(-1, n_dims)
    dxs = torch.diff(trajs_t, dim=1).reshape(-1, n_dims)
    dts = torch.full(
        (n_trajs * n_pairs, 1),
        float(obs_time_step),
        dtype=dtype,
        device=device,
    )

    return {
        'xs': xs, 
        'dxs': dxs, 
        'dts': dts, 
        'series_slices': series_slices, 
        'valid_ids': list(range(n_trajs)),  # valid_id: simply use trajectory indices as valid IDs
        'valid_times': [np.arange(n_pairs) * obs_time_step for _ in range(n_trajs)]  # valid_yrs: time points for each trajectory
    }


def panel_to_langevin_data(
    panel_df: pd.DataFrame,
    id_col: str,
    state_cols: List[str],
    time_col: str,
    time_unit: float = 1.0,
    time_step: float = 0.01,
    include_time: bool = False,
    timestep_limit: float = 10.0,
    start_year: Optional[float] = None,
    end_year: Optional[float] = None,
    condition_col: Optional[str] = None,
    condition_name_map: Optional[Dict[Any, str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Converts a panel DataFrame into tensors suitable for Langevin model training.

    Args:
        panel_df: Input DataFrame containing panel data.
        id_col: Column name for unique identifiers (e.g., country codes).
        state_cols: List of column names representing the state variables.
        time_col: Column name for the time variable (e.g., year).
        time_unit: Factor applied to raw time values before model-time scaling.
            For Seshat, if raw time is measured in years but the desired unit is centuries, use 1/100.
        time_step: Scaling factor for time intervals.
        include_time: Whether to include time as a feature in the output tensors.
        timestep_limit: Maximum gap in raw-time units multiplied by time_unit.
        Filtering compares model_dt with timestep_limit * time_step (plus tolerance).
        start_year: Optional start year for filtering the data.
        end_year: Optional end year for filtering the data.
        condition_col: Optional column name for the condition variable.
        condition_name_map: Optional dictionary mapping condition values to names for one-hot encoded features.
    Returns:
        A dictionary containing:
            - xs: Tensor of shape (N, input_dim); optional conditioning, D physical states, then optional time.
            - dxs: Tensor of shape (N, D) representing the state changes.
            - dts: Tensor of shape (N, 1) representing the time intervals.
            - series_slices: List of slice objects indicating the start and end indices for each group.
            - valid_ids: List of unique identifiers corresponding to each group.
            - valid_times: List of np.ndarray containing time points for each group.
    """
    if id_col not in panel_df.columns:
        raise ValueError(f"ID_col: '{id_col}' column not found in the DataFrame.")
    if time_col not in panel_df.columns:
        raise ValueError(f"Time_col: '{time_col}' column not found in the DataFrame.")
    if not all(col in panel_df.columns for col in state_cols):
        missing_cols = [col for col in state_cols if col not in panel_df.columns]
        raise ValueError(f"State_cols: Missing columns in the DataFrame: {missing_cols}")
    if timestep_limit is None:
        timestep_limit = float('inf')
        if verbose:
            print("No timestep_limit provided, using infinity.")
    if condition_col is not None:
        if condition_col not in panel_df.columns:
            raise ValueError(f"Condition_col: '{condition_col}' column not found in the DataFrame.")
        if panel_df[condition_col].isna().any():
            raise ValueError(f"condition_col '{condition_col}' contains NaN values.")
            
        n_cond_per_id = panel_df.groupby(id_col)[condition_col].nunique(dropna=False)
        bad_ids = n_cond_per_id[n_cond_per_id > 1]
        if len(bad_ids) > 0:
            raise ValueError(
                f"Each {id_col} must have only one {condition_col}, "
                f"but found multiple conditions for: {bad_ids.index[:10].tolist()}"
            )
    
    df = panel_df.copy()

    # ==== Preprocessing ====
    feature_cols = state_cols.copy()

    # Filter by start_year and end_year if provided
    if start_year is not None:
        df = df[df[time_col] >= start_year]
        print(f"Filtered data to include {time_col} from {start_year} onwards. Remaining data points: {len(df)}")
    if end_year is not None:
        df = df[df[time_col] <= end_year]
        print(f"Filtered data to include {time_col} up to {end_year}. Remaining data points: {len(df)}")
    df = df.sort_values(by=[id_col, time_col], ascending=[True, True]).reset_index(drop=True)

    # Compute differences (next time point - current time point) for state variables and time variable
    for col in state_cols:
        df["d" + col] = df.groupby(id_col)[col].shift(-1) - df[col]
    df["model_time"] = df[time_col] * time_unit * time_step   # Scale time column by time_unit and time_step
    df["model_dt"] = df.groupby(id_col)["model_time"].shift(-1) - df["model_time"]

    if condition_col is not None:
        if condition_col not in df.columns:
            raise ValueError(f"Condition_col: '{condition_col}' column not found in the DataFrame.")
        unique_conditions = sorted(df[condition_col].dropna().unique(), key=str)

        condition_features = []
        for cond in unique_conditions:
            cond_name = f"is_{cond}"
            if condition_name_map is not None:
                cond_name = f"is_{condition_name_map.get(cond, cond)}"
            df[cond_name] = (df[condition_col] == cond).astype(float)
            condition_features.append(cond_name)
            
        feature_cols = condition_features + state_cols

    # Select relevant columns
    d_state_cols = ["d" + col for col in state_cols]
    rel_cols = [id_col, time_col, "model_time", "model_dt"] + feature_cols + d_state_cols
    df = df[rel_cols].copy()

    # Make mask for time differences (dt) to filter out large time gaps
    model_dts = df["model_dt"].values
    mask = mask = (
        np.isfinite(model_dts)
        & (model_dts > 0)
        & (model_dts <= (timestep_limit * time_step + 1e-6))
    )
    
    # Remove NaN values resulting from differencing (last data point of each group)
    df = df[mask].dropna().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No valid transitions remain after filtering.")

    # --- Report data statistics ---
    if verbose:
        counts = df.groupby(id_col).size()
        min_id = counts.idxmin(); max_id = counts.idxmax()
        print(f"Min length: {counts[min_id]} ({id_col} = {min_id})")
        print(f"Max length: {counts[max_id]} ({id_col} = {max_id})")
        print(f"Average length: {counts.mean():.2f}")

    # Group by id_col to create slice information
    series_slices = []
    valid_ids = []
    valid_times = []
    for id, g in df.groupby(id_col, sort=False):
        sl_idx = g.index.to_numpy()
        if len(sl_idx) == 0:
            continue

        start = int(sl_idx[0])          
        end = int(sl_idx[-1]) + 1

        series_slices.append(slice(start, end, None))
        valid_ids.append(id)
        valid_times.append(g[time_col].to_numpy())

    if include_time:
        times = df["model_time"].values.reshape(-1, 1)
        
        # Center model-time coordinate for numerical stability.
        times = (times - times.mean())
        df['time_feature'] = times  # Store centered model time; time scaling was applied above.

        xs = torch.tensor(df[feature_cols + ['time_feature']].values,
                          dtype=torch.float32)
    else:
        xs = torch.tensor(df[feature_cols].values,
                          dtype=torch.float32)

    dxs = torch.tensor(df[d_state_cols].values, dtype=torch.float32)
    dts = torch.tensor((df["model_dt"].values).reshape(-1, 1), dtype=torch.float32)

    return {
        'xs': xs, 
        'dxs': dxs, 
        'dts': dts, 
        'series_slices': series_slices, 
        'valid_ids': valid_ids,     
        'valid_times': valid_times
    }


def make_folds_kfold(
        series_slices, 
        K=8, 
        seed=100
    ):
    """
        Shuffle transitions within each time series and split them into K chunks.
        Series with fewer than K transitions may be absent from some validation folds.
        This is a within-series split; entities and chronological blocks are not held out.
        Args:
            series_slices (List[slice]): List of slices indicating the start and end indices of each time series in the dataset.
            K (int): Number of folds for cross-validation.
            seed (int): Random seed for reproducibility.
        Returns:
            List[Dict]: A list of length K, where each element is a dictionary with 'train_idx' and 'valid_idx' keys containing the respective indices.
    """
    assert K > 1, "K must be at least 2."

    N = max(s.stop for s in series_slices)
    rng = np.random.default_rng(seed)

    per_series = [rng.permutation(np.arange(s.start, s.stop, dtype=np.int32)) for s in series_slices]

    per_chunks = [np.array_split(arr, K) for arr in per_series]
    for c in per_chunks: rng.shuffle(c)

    folds = []
    for k in range(K):
        valid_parts = [chunks[k] for chunks in per_chunks if chunks[k].size > 0]
        valid_idx = np.concatenate(valid_parts).astype(np.int32, copy=False) if valid_parts else np.empty(0, np.int32)

        mask = np.ones(N, dtype=bool)
        if valid_idx.size > 0:
            mask[valid_idx] = False
        train_idx = np.flatnonzero(mask).astype(np.int32, copy=False)

        folds.append({"train_idx": train_idx, "valid_idx": np.sort(valid_idx)})  # 정렬은 선택
    return folds


def _np2tensor(data: Optional[Union[np.ndarray, torch.Tensor]]) -> Optional[torch.Tensor]:
    """Helper function to safely convert NumPy array to PyTorch tensor."""
    if data is None or isinstance(data, torch.Tensor):
        return data
    return torch.from_numpy(data)


class TrajectoryDataset(Dataset):
    def __init__(self,
                 Langevin_type: str,
                 model_type: str,
                 xs: Union[np.ndarray, torch.Tensor],
                 dts: Union[np.ndarray, torch.Tensor],
                 dxs: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 vs: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 dvs: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 indices: Optional[Sequence[int]] = None,
                 dtype: torch.dtype = torch.float32,
                 ):
        """
        Args:
            Langevin_type: The type of Langevin equation, either 'OLE' or 'ULE'.
            model_type: The target of the model, either 'drift' or 'diff'.
            xs: State positions.
            dts: Time intervals.
            dxs: (OLE only) Change in state positions.
            vs: (ULE only) State velocities.
            dvs: (ULE only) Change in state velocities.
            indices: Optional list of indices to select a subset of the data.
            dtype: The desired data type for the output tensors.
        """
        assert Langevin_type in ['OLE', 'ULE'], "Langevin_type must be 'OLE' or 'ULE'"
        
        self.Langevin_type = Langevin_type
        self.model_type = model_type
        self.dtype = dtype

        # --- Input Validation ---
        if self.Langevin_type == 'OLE':
            if dxs is None:
                raise ValueError("`dxs` must be provided for `Langevin_type='OLE'`.")
            assert len(xs) == len(dxs) == len(dts), "For OLE, lengths of xs, dxs, and dts must be the same."
        elif self.Langevin_type == 'ULE':
            if vs is None or dvs is None:
                raise ValueError("`vs` and `dvs` must be provided for `Langevin_type='ULE'`.")
            assert len(xs) == len(vs) == len(dvs) == len(dts), "For ULE, lengths of xs, vs, dvs, and dts must match."
        else:
            raise ValueError(f"Invalid `Langevin_type`: {Langevin_type}. Must be 'OLE' or 'ULE'.")

        # --- Data Conversion and Indexing ---
        xs, dts, dxs, vs, dvs = map(_np2tensor, [xs, dts, dxs, vs, dvs])

        self.indices = indices if indices is not None else list(range(len(xs)))

        # --- Define Model State (inputs) and Delta Term (for targets) ---
        if self.Langevin_type == 'OLE':
            # For OLE, retain the full input, including any conditioning and time features.
            self.states = xs[self.indices]
            delta_term = dxs[self.indices]
        else: # ULE
            # For ULE, the state is the concatenation of position and velocity
            self.states = torch.cat([xs[self.indices], vs[self.indices]], dim=-1)
            delta_term = dvs[self.indices]

        # --- Define Model Target (ys) ---
        if 'drift' in self.model_type:
            # The target is the change in state.
            self.ys = delta_term
        elif self.model_type == 'diff':
            # The target estimates D * dt; train_epoch multiplies the predicted D by dt.
            # OLE: D = 0.5 * <ΔxΔx^T>/Δt
            # ULE: D = (3/2) * 0.5 * <ΔvΔv^T>/Δt = 0.75 * <ΔvΔv^T>/Δt
            # (..., D) -> (..., D*D)  outer product flatten
            prefactor = 0.5 if self.Langevin_type == 'OLE' else 0.75
            self.ys = prefactor * (delta_term.unsqueeze(-1) @ delta_term.unsqueeze(-2)).flatten(start_dim=-2)

        self.dts = dts[self.indices]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a tuple containing the model inputs and targets.
        
        Returns:
            - x: The state vector (input to the model).
            - y: The target vector (drift or diffusion).
            - dt: The time interval.
        """
        x = self.states[idx].to(dtype=self.dtype)  
        y = self.ys[idx].to(dtype=self.dtype)  
        dt = self.dts[idx].to(dtype=self.dtype)  
        return x, y, dt
