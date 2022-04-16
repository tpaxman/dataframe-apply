from typing import Callable
import inspect
import pandas as pd
import numpy as np

def apply(func: Callable, df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Applies a vectorized version of a given function (func) to a dataframe (df)
    where the function inputs are described by a mapping of parameters to
    argument values (kwargs)
    """

    # ensure that all arguments are actually accepted by the function 
    params_used = set(kwargs)
    params_allowed = set(inspect.signature(func).parameters)
    assert params_used.issubset(params_allowed)

    # identify the parameters whose arguments are column names in the DataFrame (which are to be vectorized)
    params_with_colname_args = [param for param, arg in kwargs.items() if arg in df.columns]

    # vectorize the function, excluding any parameters not referring to column names
    non_colname_params = set(kwargs).difference(params_with_colname_args)
    vectorized_func = np.vectorize(func, excluded=non_colname_params)

    # update the arguments containing column names to contain the column data and run them in the vectorized function
    function_inputs = {param: df[arg] if param in params_with_colname_args else arg for param, arg in kwargs.items()}
    outputs_list = vectorized_func(**function_inputs)
    return outputs_list
