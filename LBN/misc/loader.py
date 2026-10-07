import os
import json
import copy
import pickle
from pathlib import Path


import numpy as np
import torch

# data handling
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Tuple, Optional, Dict, Any, Union, Literal, Sequence
import pandas as pd



def load_dataset(data_path: str):
    """
        Loads latent data from a file.

        Args:
            data_path: Path to the file containing latent data.
        Returns:
            latent_data: The loaded latent variable data.
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


# def latent2data(data_content, data_df, args):
#     """
#     Processes latent data to generate tensors for time-series models.

#     Args:
#         data_content: The type of data, e.g., 'DIG' or 'wikiart'.
#         data_df: The input data DataFrame.
#         Langevin_type: Type of model, either 'OLE' or 'ULE'.
#         time_step: Scaling factor for time intervals.
#         model_type: Required if Langevin_type is 'ULE'. Can be 'drift' or 'diff'.
#     Returns:
#         A tuple containing concatenated tensors and series slices.
#         For OLE: (xs, dxs, dts, series_slices)
#         For ULE: (xs, vs, dvs, dts, series_slices)
#     """
#     if data_content == 'DIG':
#         return DIGpca2data(data_df, time_step=args.time_step, include_time = args.include_time,
#                             start_year = args.start_year, end_year = args.end_year)
#     elif data_content == 'seshat':
#         return Seshat2data(data_df, time_step=args.time_step, include_time=args.include_time)
#     elif data_content == 'ngram':
#         return Ngram2data(data_df, time_step=args.time_step, value_ids=value_ids,
#                              include_time=args.include_time, domain_indicator=args.domain_indicator)
#     elif data_content == 'synthetic':
#         return synthetic2data(data_df, obs_time_step=args.time_step, Langevin_type=args.Langevin_type)
#     else:
#         raise ValueError("Unsupported data_content type. Choose from 'DIG', 'seshat', 'ngram', or 'synthetic'.")

def dataset_to_langevin_data(
        data_content: str, 
        data_df: pd.DataFrame, 
        args
    ):
    """
    Processes latent data to generate tensors for time-series models.

    Args:
        data_content: The type of data, e.g., 'dig', 'seshat', 'ngram', or 'synthetic'.
        data_df: The input data DataFrame.
        args: An object containing the following attributes:
            - time_step: Scaling factor for time intervals.
            - include_time: Whether to include time as a feature in the output tensors.
            - start_year: The start year for filtering the data (optional).
            - end_year: The end year for filtering the data (optional).
    Returns:
        A dictionary containing:
            - xs: Tensor of shape (N, D) representing the states (features).
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
    Processes synthetic data to generate tensors for OLE or ULE time-series models.

    Args:
        trajs: Input trajectory array with shape (n_trajs, n_steps, n_dims).
        obs_time_step: Time interval between consecutive observations.
        Langevin_type: Type of model, either 'OLE' or 'ULE'.

    Returns:
        For OLE:
            xs, dxs, dts, series_slices, None, None
        For ULE:
            xs, vs, dvs, dts, series_slices, None, None (not implemented yet)
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
        timestep_limit: Maximum allowed time difference between consecutive points.
        start_year: Optional start year for filtering the data.
        end_year: Optional end year for filtering the data.
        condition_col: Optional column name for the condition variable.
        condition_name_map: Optional dictionary mapping condition values to names for one-hot encoded features.
    Returns:
        A dictionary containing:
            - xs: Tensor of shape (N, D) representing the states.
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

    # Group by 'iso3' to create slice information
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
        df['time_feature'] = times  # scale time feature by time_unit and time_step for better numerical stability

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


# def DIGpca2data(latent_df, time_step=0.01, include_time=False, timestep_limit=10, 
#                 start_year=None, end_year=None):
#     """
#     Processes latent data to generate tensors for OLE or ULE time-series models.

#     Args:
#         latent_df: The input data frame (pandas DataFrame).
#         time_step: Scaling factor for time intervals.
#         include_time: Whether to include time as a feature.
#         timestep_limit: Maximum allowed time step for filtering data.
#         start_year: The start year for filtering data (default: None, use all years).
#         end_year: The end year for filtering data (default: None, use all years).

#     Returns:
#         A tuple containing concatenated tensors and series slices.
#         For OLE: (xs, dxs, dts, series_slices)
#         For ULE: (xs, vs, dvs, dts, series_slices)
#     """

#     var_cols = ['DMC_PC1', 'INQ_PC1', 'GDP_PC1']
#     dvar_cols = ["d"+col for col in var_cols]

#     df = latent_df.sort_values(by=['iso3', 'year'], ascending=[True, True]).reset_index(drop=True)
#     for col in var_cols:
#         df["d"+col] = df.groupby('iso3')[col].diff().shift(-1)
    
#     df["dyrs"] = df.groupby('iso3')["year"].diff().shift(-1)
    
#     # Select relevant columns
#     rel_cols = ['iso3', "year", "dyrs"] + var_cols + dvar_cols
#     df = df[rel_cols]

#     # --- mask  ---
#     raw_dts = df["dyrs"].values * time_step
#     mask = raw_dts <= (timestep_limit * time_step) + 1e-6
    
#     # Remove NaN values resulting from differencing (last data point of each group)
#     df = df[mask].dropna().reset_index(drop=True)

#     # Filter by start_year and end_year if provided
#     if start_year is not None:
#         df = df[df["year"] >= start_year]
#         print(f"Filtered data to include years from {start_year} onwards. Remaining data points: {len(df)}")
#     if end_year is not None:
#         df = df[df["year"] <= end_year]
#         print(f"Filtered data to include years up to {end_year}. Remaining data points: {len(df)}")
#     df = df.reset_index(drop=True)

#     # --- Report data statistics ---
#     counts = df.groupby('iso3').size()
#     min_iso3 = counts.idxmin(); max_iso3 = counts.idxmax()
#     print(f"Min length: {counts[min_iso3]} (iso3 = {min_iso3})")
#     print(f"Max length: {counts[max_iso3]} (iso3 = {max_iso3})")
#     print(f"Average length: {counts.mean():.2f}")

#     # Group by 'iso3' to create slice information
#     series_slices = []
#     valid_countries = []
#     valid_yrs = []
#     for iso3, g in df.groupby('iso3', sort=False):
#         idx = g.index.to_numpy()
#         if len(idx) == 0:
#             continue

#         start = int(idx[0])          
#         end = int(idx[-1]) + 1

#         series_slices.append(slice(start, end, None))
#         valid_countries.append(iso3)
#         valid_yrs.append(g["year"].to_numpy())

#     if include_time:
#         yrs = df['year'].values.reshape(-1, 1)
        
#         yrs = (yrs - yrs.mean())
#         df['time_feature'] = yrs * time_step  # scale time feature by time_step for better numerical stability

#         xs = torch.tensor(df[var_cols + ['time_feature']].values,
#                           dtype=torch.float32)
#     else:
#         xs = torch.tensor(df[var_cols].values,
#                           dtype=torch.float32)

#     dxs = torch.tensor(df[dvar_cols].values, dtype=torch.float32)
    
#     # Apply time_step scaling to time intervals (dt)
#     dts = torch.tensor((df["dyrs"].values * time_step).reshape(-1, 1),
#                        dtype=torch.float32)

#     return (
#         xs,            # features (State)
#         dxs,           # feature differences (dState)
#         dts,           # time differences (dt)
#         series_slices, # slice(start, end, None) for each country series
#         valid_countries, # list of valid countries (iso3 codes) aligned with series_slices
#         valid_yrs 
#     )



# def Seshat2data(latent_data, time_step=0.01, include_time=False, key_id='nga_id', 
#                  value_ids=['Scale_PC1', 'Comp_w'], timestep_limit=5):
#     """
#     Processes Seshat PCA data to generate tensors for time-series models.
#     """
#     if timestep_limit is None:
#         timestep_limit = float('inf')

#     df = latent_data.sort_values([key_id, "year"])

#     # Calculate differences (next time point - current time point)
#     # Compute differences within each group (PolityID) and map future changes to the current row using shift(-1)
#     df["d"+value_ids[0]] = df.groupby(key_id)[value_ids[0]].diff().shift(-1)
#     df["d"+value_ids[1]] = df.groupby(key_id)[value_ids[1]].diff().shift(-1)
#     df["dyrs"] = df.groupby(key_id)["year"].diff().shift(-1)/100 # convert to centuries for better scaling

#     # Select relevant columns
#     df = df[[key_id, "year", value_ids[0], value_ids[1], "d"+value_ids[0], "d"+value_ids[1], "dyrs"]]

#     # --- mask  ---
#     raw_dts = df["dyrs"].values * time_step
#     mask = raw_dts <= (timestep_limit * time_step)
    
#     # Remove NaN values resulting from differencing (last data point of each group)
#     df = df[mask].dropna().reset_index(drop=True)

#     series_slices = []
#     valid_ngas = []
#     valid_yrs = []

#     # Group by nga_id to create slice information
#     for nga_id, g in df.groupby(key_id, sort=False):
#         idx = g.index.to_numpy()
#         if len(idx) == 0:
#             continue

#         start = int(idx[0])          
#         end = int(idx[-1]) + 1

#         series_slices.append(slice(start, end, None))
#         valid_ngas.append(nga_id)
#         valid_yrs.append(g["year"].to_numpy())

#     if include_time:
#         yrs = df['year'].values.reshape(-1, 1)
        
#         yrs = (yrs - yrs.mean()) / yrs.std()
#         df['time_feature'] = yrs

#         xs = torch.tensor(df[[value_ids[0], value_ids[1], "time_feature"]].values,
#                           dtype=torch.float32)
#     else:
#         xs = torch.tensor(df[[value_ids[0], value_ids[1]]].values,
#                           dtype=torch.float32)

#     dxs = torch.tensor(df[["d"+value_ids[0], "d"+value_ids[1]]].values, dtype=torch.float32)
    
#     # Apply time_step scaling to time intervals (dt)
#     dts = torch.tensor((df["dyrs"].values * time_step).reshape(-1, 1),
#                        dtype=torch.float32)
    
#     return (
#         xs,            # features (State)
#         dxs,           # feature differences (dState)
#         dts,           # time differences (dt)
#         series_slices, # slice info
#         valid_ngas,    
#         valid_yrs 
#     )



# def Ngram2data(latent_df, time_step=0.01, value_ids = ['z_score'], domain_indicator = None,
#                   include_time=True, timestep_limit=5, start_year=None, end_year=None):
#     """
#     Processes data to generate tensors for OLE or ULE time-series models.

#     Args:
#         latent_df: The input data pandas dataframe.
#         time_step: Scaling factor for time intervals.
#         value_ids: The list of column names in the dataframe that contains the feature values to be used.
#         include_time: Whether to include time as a feature in the output tensors.
#         timestep_limit: Maximum allowed time difference between consecutive points.
#         start_year: The starting year for filtering the data.
#         end_year: The ending year for filtering the data.

#     Returns:
#         (xs, dxs, dts, series_slices, valid_words, valid_yrs)
#         xs: features
#         dxs: feature differences
#         dts: time differences
#         series_slices: list of slice(start, end, None) for each word series
#         valid_words: list of word names aligned with series_slices
#         valid_yrs: list of np.ndarray of years for each series
#     """
#     df = latent_df.sort_values(by=['word', 'Year'])

#     # Domain filtering based on emo2rat_indicator
#     # if domain_indicator is None, keep all data; if 1, keep only positive domain (rationality); if -1, keep only negative domain (emotionality)
#     if domain_indicator is None:
#         print("No domain filtering applied, using both rationality and emotionality data...")
#         df = df[df['emo2rat_indicator'].isin([1, -1])].reset_index(drop=True)
#     else:
#         print(f"Filtering data for domain_indicator = {domain_indicator}...")
#         if domain_indicator == 1:
#             df = df[df['emo2rat_indicator'] == 1].reset_index(drop=True)
#         elif domain_indicator == -1:
#             df = df[df['emo2rat_indicator'] == -1].reset_index(drop=True)
#         else:
#             raise ValueError("domain_indicator should be 1 (positive), -1 (negative), or None (both)")

#     df['is_rationality'] = (df['emo2rat_indicator'] == 1).astype(float)
#     df['is_emotionality'] = (df['emo2rat_indicator'] == -1).astype(float)

#     # Calculate differences
#     for value_id in value_ids:
#         df["d"+value_id] = df.groupby('word')[value_id].diff().shift(-1)
#     df["dyrs"] = df.groupby('word')["Year"].diff().shift(-1)

#     # Select relevant columns
#     df = df[['wordID', 'word', "Year"] + value_ids + ['emo2rat_indicator', "is_rationality", "is_emotionality"] +
#              ["d"+value_id for value_id in value_ids] + ["dyrs"]]

#     # --- mask  ---
#     raw_dts = df["dyrs"].values * time_step
#     mask = raw_dts <= (timestep_limit * time_step)
    
#     # Remove NaN values resulting from differencing (last data point of each group)
#     df = df[mask].dropna().reset_index(drop=True)

#     series_slices = []
#     valid_words = []
#     valid_yrs = []

#     # Group by 'word' to create slice information
#     for word, g in df.groupby('word', sort=False):
#         idx = g.index.to_numpy()
#         if len(idx) == 0:
#             continue

#         start = int(idx[0])          
#         end = int(idx[-1]) + 1

#         series_slices.append(slice(start, end, None))
#         valid_words.append(word)
#         valid_yrs.append(g["Year"].to_numpy())

#     if include_time:
#         yrs = df['Year'].values.reshape(-1, 1)
        
#         yrs = (yrs - yrs.mean())
#         df['time_feature'] = yrs * time_step  # scale time feature by time_step for better numerical stability


#     # Construct feature columns based on domain_indicator and include_time
#     feature_cols = []
    
#     if domain_indicator is None:
#         feature_cols.extend(['is_rationality', 'is_emotionality'])

#     feature_cols.extend(value_ids)
#     if include_time:
#         feature_cols.append('time_feature')
        
#     xs = torch.tensor(df[feature_cols].values, dtype=torch.float32)

#     dxs = torch.tensor(df[["d"+value_id for value_id in value_ids]].values, dtype=torch.float32)
#     # Apply time_step scaling to time intervals (dt)
#     dts = torch.tensor((df["dyrs"].values * time_step).reshape(-1, 1),
#                        dtype=torch.float32)
    
#     return (
#         xs,            # features (State = [domain one-hot + value + optional time])
#         dxs,           # feature differences (dState)
#         dts,           # time differences (dt)
#         series_slices, # slice (start, end, None) for each word series
#         valid_words,    
#         valid_yrs 
#     )



def make_folds_kfold(
        series_slices, 
        K=8, 
        seed=100
    ):
    """
        Stratified within-series K-fold split, ensuring that each fold contains transitions from every time series.
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
        # assert model_type in ['drift', 'diff'], "model_type must be 'drift' or 'diff'"
        
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
            # For OLE, the state is just the position
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
            # The target is the diffusion matrix, estimated from the outer product of the change in state
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
