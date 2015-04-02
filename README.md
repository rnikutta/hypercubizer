# hypercubizer #

Convert an entire directory (possibly with sub-directories) of template-based model text files (i.e. each file is one realization of a model, with the model results stored as columns in the files) to a number of N-dimensional hypercubes. The values of the model parameters that generated a file must be contained in the file names (see 'Example').

Each column to be read from the files will result in one hypercube computed. The shape and dimensionality of the cubes will be determined from the number of parameters of the model that generated the text files.

All you need to provide is the root directory containing the text files, a pattern for the file names (of course the file names must contain the numerical values of the model for that particular file, and the columns to be read from the files. Everything else is automatic. You can (and should) then store the computed hypercubes into a single and convenient HDF5 file. The storage code is provided.

In addition (but this is optional) you can supply the names for the read columns, they will be stored alongside the hypercubes. Another option is to store a one-dimensional x-column. In astronomical context, often the models output is stored as a function of wavelength. So telling [hypercubizer] which column holds is the wavelengths, it will store it alongside the hypercubes. This is super-convenient if you want to do N-dimensional interpolation of your entire model database. (Such interpolation code is also available from the same author, here: [link-to-willitfit].

Example:
tbw

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Who do I talk to? ###

* Robert Nikutta