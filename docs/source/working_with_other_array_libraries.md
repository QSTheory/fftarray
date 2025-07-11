---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: dev
  language: python
  name: python3
---

# Working with other array libraries

By default FFTArray uses NumPy which is great for compatibility with other libraries and available everywhere.
However, in order to achieve higher performance or get access to specific features it can be helpful to use another array library.
FTArray is built upon the [Python Array API Standard](https://data-apis.org/array-api/latest/) to enable this use case.

All {py:mod}`Array creation functions<fftarray._src.creation_functions>` try to infer the array library from their inputs if possible and also have an optional `xp` argument.
The `xp` argument accepts a namespace like `numpy` or `jax.numpy` and automatically wraps it with the [`array-api-compat` library](https://data-apis.org/array-api-compat/) to make it compatible to the Python Array API standard.

## Default array library
If a function cannot determine via its arguments which array library to use it falls back to NumPy.
This default fallback can be changed via the methods in {py:mod}`Setting defaults<fftarray._src.defaults>`.

## Sharp bits

### Array conversion between libraries

The Array API standard only defines limited rules about converting arrays from one array library to another.
The {py:func}`values <fftarray.Array.values>` method has an otpional `xp` argument which converts the values into another Array API namespace.
This is always done via NumPy and only supports explicitly supported libraries.
See {py:func}`values <fftarray.Array.values>` for more details.

### Default dtype and type promotions
The default `dtype` and `dtype` promotion rules can differ between different libraries since the standard only defines a minimum set of rules: [Array API Data Types](https://data-apis.org/array-api/latest/API_specification/data_types.html),  [Array API Type Promotion Rules](https://data-apis.org/array-api/latest/API_specification/type_promotion.html).

FFTArray does not add any custom behavior on top of those rules but simply passes through the behavior of the used array library.
Therefore `dtype`s in the same code can change when the used array library is changed.

### Binary operations with an array library scalar as the first and a fftarray.Array as the second operand

As of v2024.12 the Array API standard does not mandate that implementations return `NotImplemented` on binary operations.
Some array libraries do not return `NotImplemented` but raise an error on binary operations with an array library scalar as the first and an `fftarray.Array` as the second operand.
In this case FFTArray never gets the chance to implement the binary operation and this does not work.
Since NumPy and JAX both return `NotImplemented` in these cases this works with these two libraries.
