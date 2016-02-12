hypercubizer
============

**Author:** Robert Nikutta

**Version:** 2016-02-12

**License:** BSD 3-clause, please see [LICENSE](./LICENSE) file

**Synopsis:**
Convert an entire directory (possibly with sub-directories) of
template-based model text files (i.e. each file is one realization of
a model, with the model results stored as columns in the files) to a
number of N-dimensional hypercubes. The values of the model parameters
that generated a file must be contained in the file names.

Each column to be read from the files will result in one hypercube
computed. The shape and dimensionality of the cubes will be determined
from the number of parameters of the model that generated the text
files.

**Example:**
All you need to provide is the root directory containing the text
files, a pattern for the file names, an which columns you want to be
read from the files (and turned into hypercubes, like this:

```python
rootdir = '/home/foo/superproject'
pattern = '[density](0.01)_[gravity](123)_[temperature](1e4)K.dat'
columns = (1,2)
```

Here `pattern` assumes that the model file names are
e.g. `density0.01_gravity123_temperature1e4K.dat`. Note how you tell
the tool the parameter names by enclosing them with square brackets,
e.g. `[density]`, and their numerical values with round brackets,
e.g. `(0.01)`. The numerical values in the given `pattern` are of
course irrelevant, only the structure of the file name matters. The
column numbers to be read from the files are of course counted
pythonically (i.e. first column in the file is `0`, etc.)

Now just run the show:

```python
H = Hypercubes(rootdir,pattern,columns)
```

From here, everything is automatic. Let the code scan `rootdir`,
determine the dimensionality (number n of parameters), ready all files
and convert all desired M columns into M n-dimensional hypercubes.

**Saving the hypercubes:**
Once this is done you can (and should) then store the computed
hypercubes into a single and convenient HDF5 file. The storage code is
provided (see docstring).

*TODO: show example.*

**Optional args:**
Optionally you can also provide the names for the read columns. If you
do, the hypercubes will carry these names. If you don't, some generic
names will be assigned.

```python
colnames = ('fluxA','fluxB')
```

Another option is to store a one-dimensional x-column. In astronomical
context, often the models output is stored as a function of
wavelength. So telling [hypercubizer] which column holds is the
wavelengths, it will store it alongside the hypercubes. This is
super-convenient if you want to do N-dimensional interpolation of your
entire model database. (Such interpolation code is also available from
the same author, here: [ndiminterpolation](../ndiminterpolation/).)

```python
xcol = 0
xcolname = 'wave_micron'
```

Run with the optional args:

```python
H = Hypercubes(rootdir,pattern,columns,colnames,xcol=xcol,xcolname=xcolname)
```

Check the docstrings for more.
