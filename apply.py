from typing import Callable
import inspect
import pandas as pd
import numpy as np

def apply(func: Callable, df: pd.DataFrame, **kwargs) -> pd.Series:

    # ensure that all arguments are actually accepted by the function 
    params_used = set(kwargs)
    params_allowed = set(inspect.signature(func).parameters)
    assert params_used.issubset(params_allowed)

    # identify the parameters whose arguments are column names in the DataFrame (which are to be vectorized)
    params_with_colname_args = [param for param, arg in kwargs.items() if arg in df.columns]

    # update the arguments that were column names to instead contain the columns themselves
    function_inputs = {param: df[arg] if param in params_with_colname_args else arg for param, arg in kwargs.items()}

    # identify parameters that do not have column names as arguments (which are not to be vectorized)
    non_colname_params = set(kwargs).difference(params_with_colname_args)
    vectorized_func = np.vectorize(func, excluded=non_colname_params)
    return vectorized_func(**function_inputs)
