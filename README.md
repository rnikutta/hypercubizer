# hypercubizer #

Convert an entire directory (possibly with sub-directories) of template-based model text files (i.e. each file is one realization of a model, with the values stored as columns in the files) to a number of N-dimensional hypercubes.

Each column to be read from the files will result in one hypercube computed. The shape and dimensionality of the cubes will be determined from the number of parameters of the model that generated the text files.

All you need to provide is the root directory containing the text files, and a pattern for the file names (of course the file names should contain the numerical values of the model for that particular file. Everything else is automatic. You can (and should) then store the computed hypercubes into a single and convenient HDF5 file. The storage code is provided.

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Who do I talk to? ###

* Robert Nikutta
