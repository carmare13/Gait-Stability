
import pandas as pd
import numpy as np
from scipy.signal import decimate

def downsample_df(df, rate, cols=None, zero_phase=True):
    """
    Downsample a pandas DataFrame by an integer factor `rate`.

    Args:
        df      (pd.DataFrame): time-series indexed by time (or integer index).
        rate    (int): factor by which to decimate (keep 1 sample every `rate`).
        cols    (List[str], optional): which columns to downsample. If None, all.
        zero_phase (bool): if True, uses zero-phase filtering (via `filtfilt` 
                           internally in `decimate`), to avoid phase shift.

    Returns:
        pd.DataFrame: downsampled DataFrame, with the same index (kept) or reset.
    """
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=cols)

    to_ds = cols if cols is not None else df.columns.tolist()
        
    out = {c: decimate(df[c].values, rate, ftype='iir', zero_phase=zero_phase) for c in to_ds}
    
    # build new DataFrame; index will be compressed accordingly
    # if your index is numeric time and you want to keep it, you can do:
    if hasattr(df.index, 'values'):
        new_index = df.index.values[::rate][: len(out[to_ds[0]])]
    else:
        new_index = range(len(out[to_ds[0]]))
    
    result = pd.DataFrame(out, index=new_index)
    
    # if there are other columns you didn’t downsample, copy them via slicing
    if cols is not None and set(cols) != set(df.columns):
        extra_cols = df.columns.difference(cols)
        extras = df.loc[:, extra_cols].iloc[::rate].reset_index(drop=True)
        extras.index = result.index  
        result = pd.concat([result, extras], axis=1)
    
    return result


