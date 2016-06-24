"""A miscellaneous collection of utility functions."""

def _assign_value_if_none(value, default):
    if value is None:
        return default
    else:
        return value
        
