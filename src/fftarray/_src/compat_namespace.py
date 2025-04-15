import array_api_compat

def get_compat_namespace(xp):
    """
        Wraps a namespace in a compatibility wrapper if necessary.
    """
    return get_array_compat_namespace(xp.asarray(1))

def get_array_compat_namespace(x):
    """
        Get the Array API compatible namespace of x.
        This is basically a single array version of `array_api_compat.array_namespace`.
        But it has a special case for torch because `array_api_compat.array_namespace`
        is currently incompatible with `torch.compile`.
    """
    # Special case torch array.
    # As of pytorch 2.6.0 and array_api_compat 1.11.0
    # torch.compile is not compatible with `array_api_compat.array_namespace`.
    if array_api_compat.is_torch_array(x):
        import array_api_compat.torch as torch_namespace
        return torch_namespace

    return array_api_compat.array_namespace(x)
