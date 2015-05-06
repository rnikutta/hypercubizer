# padarray

Pad &amp; unpad a ragged (2D) array with any value. Work-around of
Numpy [bug 2190](https://github.com/numpy/numpy/issues/2190), which
for instance prevents storing of float64 ragged arrays in HDF5
files. This is also tracked as [bug
433](https://github.com/h5py/h5py/issues/433) in
[h5py](http://www.h5py.org/).

Please see the docstrings in [padarray.py](./padarray.py) for usage.
