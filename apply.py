from typing import Callable
import inspect
import pandas as pd
import numpy as np

def apply(func: Callable, df: pd.DataFrame, **kwargs) -> pd.Series:

    # ensure that all arguments are actually accepted by the function 
    params_used = set(kwargs)
    params_allowed = set(inspect.signature(func).parameters)
    assert params_used.issubset(params_allowed)

    # update parameter arguments as the actual column whenever a column name is given
    cleaned_args = {} 
    exclude_from_vectorization = []
    for param, arg in kwargs.items():
        if arg in df.columns:
            # if an input argument matches a column name assume the value is the column in the table
            cleaned_args[param] = df[arg]
        else:
            # otherwise just take the input argument at face value
            cleaned_args[param] = arg
            exclude_from_vectorization.append(arg)

    # vectorize the function and exclude the "constant" input values
    vectorized_func = np.vectorize(func, excluded=exclude_from_vectorization)

    # run the vectorized function on the DataFrame columns provided
    return vectorized_func(**cleaned_args)
