from typing import Callable
import inspect
import pandas as pd
import numpy as np

def apply(func: Callable, df: pd.DataFrame, **kwargs) -> pd.Series:
    assert set(kwargs).issubset(inspect.signature(func).parameters)
    vargs = {k: v for k, v in kwargs.items() if v in df.columns}
    normalargs = {k: v for k, v in kwargs.items() if k not in vargs}
    fv = np.vectorize(func, excluded=normalargs.keys())
    vargs2 = {k: df[v] for k, v in vargs.items()}
    allargs = {**vargs2, **normalargs}
    output = fv(**allargs)
    return output
