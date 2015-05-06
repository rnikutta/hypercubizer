"""Find template-based model files, read columns, convert them to n-dim hypercubes, store as HDF5.
"""

import os
import re
import warnings
import numpy as N
import h5py
#import sys
#sys.path.append('./externals')
from externals.padarray import padarray

__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = "20150505"

# TODO: add simple logging

class Hypercubes:

    def __init__(self,rootdir,pattern,cols,colnames=None,xcol=None,xcolname=None,paramnames=None):

        """Load columns from regex-matched files in rootdir and convert them
        to hypercubes.

        The computed hypercubes and other data can then be stored in
        an HDF5 file (inside a dedicated group). The results of other
        conversion runs (for instance on other model file sets) can be
        stored into the same HDF5 file, but providing a different (not
        yet existing) group name.

        See 'Example' below for a quick how-to.

        Parameters:
        -----------
        rootdir : str
            Path to the root directory holding the files to be
            scanned.

        pattern : str
            An example filename such as those that will be matched
            with the pattern determined here. Only the basename of
            'pattern' will be considered, i.e. no superordinate
            directories.

            See the docstring of get_pattern_and_paramnames() on how
            you should format this example file name, or under
            'Example' below.

        cols : seq of integers
            (Pythonic) indices of the columns to be read from every
            matched file. There will be as many hypercubes as columns
            you specify. Example: cols=(4,5,8).

        colnames : {None, seq of strings}
            If not None, these are the names of the columns, each
            corresponding to one element in 'hypercubes'. Exactly as
            many column names must be provided as there are members in
            'hypercubes'. If None, generic column names will be
            assigned (col00, col01, etc.)

        paramnames : {None, seq}
            None, or sequence of strings (length=number of
            parameters).

            If 'paramnames' is a sequence of strings, they will be
            used as the parameter names. In this case the sequence
            must be exactly as long as the number of values extracted
            from 'pattern'.

            If 'pattern' contains square brackets (i.e. if the file
            names contain the parameter names), and 'paramnames' is
            not a sequence of strings, then the parameter names
            extracted from the pattern will be used.

            If 'pattern' contains no square brackets (i.e. no
            parameter names to be be extracted from the file names),
            and 'paramnames' is not a sequence of strings, then
            generic names will be created for the parameter names
            ('param00','param01',etc.)

        Example:
        --------
        Say your many files are in /home/foo/superproject
        (possibly in subdirectories). Say also that all files are
        named like this:

          /home/foo/superproject/dir1/dir12/density0.01_gravity123_temperature1e4K.dat

        You want to match the parameter names density, gravity,
        temperature, and their numerical values, in all files. Let's
        assume each file is structured like this:

          # column 0     1        2        3
          # wavelength   fluxA    fluxB    priceofgas
          # micron       W/m^2    W/m^2    $/gallon
          1.0e-2         1.3e-14  1.8e-15  2.51
          1.0e-1         5.3e-14  2.3e-15  2.78
          1.0e-0         2.0e-13  1.2e-14  2.13
          ...
        
        Finally, let's say you want the two flux columns to be
        converted to hypercubes, and you don't care about the price of
        gas. You would also like the wavelength column to be stored
        separately as an x-column. Simply do:

          rootdir = '/home/foo/superproject'
          pattern = '[density](0.01)_[gravity](123)_[temperature](1e4)K.dat'
          columns = (1,2)
          colnames = ('fluxA','fluxB')
          H = Hypercubes(rootdir,pattern,columns,colnames,xcol=0,xcolname='wave_micron')

        That's it! This will scan for all files in rootdir, keep only
        those that match the example file name pattern, extract the
        parameter values (in round brackets () in 'pattern'), extract
        the parameter names (in square brackets []), load columns 1
        and 2 from all matched files, and convert them to
        3-dimensional hypercubes (we have 3 matched parameters). It
        will also load the x-column 'wavelength' (column 0).

        Once this is done, you can store everything in a convenient
        HDF5 file:

          HC('somestoragefile.hdf5',groupname='mymodels')

        The groupname is optional, and if you don't provide it, the
        basename of rootdir will be chosen (i.e. here 'superproject').

        If you want another set of files (possibly with a completely
        different regex pattern) to be stored as another group in the
        same HDF5 file, just do the Hypercubes() step above, i.e.

          # ... preparation ...
          HC2 = Hypercubes(rootdir2,pattern2,columns2,colnames2,xcol=0,xcolname='something')

        and then store via:

          HC('somestoragefile.hdf5',groupname='foobar')  # note same HDF5 file name

        To access the data in the HDF5 file:

          import h5py
          h = h5py.File('somestoragefile.hdf5','r')
          h.items()
            [(u'mymodels', <HDF5 group "/mymodels" (5 members)>),
             (u'foobar', <HDF5 group "/foobar" (5 members)>),
          h['mymodels'].items()
            etc.

        """

        # regex pattern, number of values, extracted parameter names (if any)
        self.pattern, self.nparams, self.paramnames = get_pattern_and_paramnames(pattern)

        # parameter names
        if isinstance(paramnames,(list,tuple)):
            assert ( len(set(paramnames)) == self.nparams ), "The number of provided 'paramnames' (%d provided) must be equal to the number of matched values in 'pattern' (%d), and they must be unique." % (len(paramnames),self.nparams)
            assert ( all([isinstance(e,str) for e in paramnames]) ), "Not all members in 'paramnames' seem to be strings."
            self.paramnames = paramnames   # this overrides even the names extracted from 'pattern' (if any)
        elif paramnames is None:
            if self.paramnames == []:
                self.paramnames = ['param%02d' % j for j in xrange(self.nparams)]
        else:
            raise Exception, "'paramnames' must either be None (and then 'pattern' must contain square brackets to indicate the parameter names to be extracted), or a list of strings."

        self.rootdir = rootdir
        self.cols = cols              # from every matched file load column cols
        self.colnames = colnames      # optional: give names to the columns that will be converted; if None, will be named sequentially (['col0','col1',...]).
        self.xcol = xcol              # optional: designate a single column as the independent variable (useful for n-dim interpolation later)
        self.xcolname = xcolname      # optional name for xcol. Default is 'xcol'.

        # get a list of all files under rootdir
        files = get_files_in_dir(self.rootdir,verbose=False,returnsorted=True)

        # match all files to pattern, keep only the matching ones, and a list of matched numerical values
        self.matched_files, self.matched_values = match_pattern_to_strings(files,pattern=self.pattern,op=os.path.basename,progress=True)

        # turn the list of all matched values (per file) into lists of unique values (per parameter)
        theta_strings, self.theta, self.hypercubeshape = get_uniques(self.matched_values,returnnumerical=True)

        self.x, self.hypercubes = self.convert_columns_to_hypercube(self.matched_files,self.matched_values)

        # As described in issue #5 on bitbucket, we need to work
        # around a numpy and/or h5py bug. That's why we nan-pad
        # self.theta, and store as a regular 2-d array (self.theta.pad)
        self.theta = padarray.PadArray(self.theta)  # has members .pad (2-d array) and .unpad (list of 1-d arrays)

        self.sanity()


    def __call__(self,storefile,groupname=None):

        """Store to a group in an HDF5 file, if you like. You should like."""

        self.storefile = storefile
        self.groupname = groupname

        self.store()


    def sanity(self):
        
        assert (len(set([h.shape for h in self.hypercubes])) == 1), "Not all cubes in 'hypercubes' have the same shape."

        if self.x is not None:
            assert (self.x.size == self.hypercubes[0].shape[-1]), "xcol was given; the length of x must be equal to the size of each hypercube's last axis."

        if self.colnames is not None:
            assert (len(self.colnames) == len(self.hypercubes)), "The number of column names given in 'colnames' must be equal to the number of dimensions in each hypercube (plus one, of xcol was also given)."
        else:
            self.colnames = ['col' + '%02d' % j for j in xrange(len(self.hypercubes))]  # generic column names, if none provided


    def store(self):

        """The actual storage method."""

        if self.groupname is None:
            try:
                self.groupname = os.path.basename(os.path.normpath(self.rootdir))
            except:
                raise Exception

        print "Storing all data to file '%s', under group name '%s'" % (self.storefile,self.groupname)
        self.S = Storage(self.storefile)
        self.S(self.groupname, self.hypercubes, self.theta.pad, self.colnames, self.paramnames, xcol=self.x, xcolname=self.xcolname)


    def convert_columns_to_hypercube(self,files,values):

        """Load specified columns from all pattern-matched files, and store
        them in a list of n-dimensional hypercubes, each properly shaped
        according to the matched parameter values.

        Given the set of unique parameter values (for every
        parameter), and the set of parameter values matched for a
        given file, we can determine the n-dimensional position the
        the current parameter values, i.e. where in the hypercube to
        put data read from that file.

        The indices of columns to read from every matched file are
        taken from self.cols.

        We have to do these somewhat complicated antics because we can
        not rely on the correct sorting order of the files (wrt to the
        parameter values they represent). If we could, we would simply
        concatenate the numbers from all files, and then reshape them
        to the n-dimensional hypercube using the known unique
        parameter values sets.

        Parameters
        ----------
        files : list
            List of path to all files that have been pattern-matched
            for the parameter values matched in their file names.

        values : list
            List of parameter values match in each file in 'files'.

        """

        # check in the first files how many rows there are to read
        data = N.loadtxt(files[0],usecols=(self.cols[0],),unpack=True)

        # add the number of rows to the hypercube shape
        self.hypercubeshape.append(data.size)

        # load the x-column, if user has requested it
        if self.xcol is not None:
            x = N.loadtxt(files[0],usecols=(self.xcol,),unpack=True)
        else:
            x = None

        # prepare a list of n-dimensional hypercubes to hold the
        # re-shaped column data (one hypercube per column read)
        ys = [N.zeros(shape=self.hypercubeshape) for j in xrange(len(self.cols))] # careful not to reference the same physical array n times

        nvalues = float(len(values))

        # LOOP OVER GOOD FILES
        for ivalue,value in enumerate(values):

            f = files[ivalue]

            if ivalue % 100 == 0:
                print "Working on model %d of %d, %.3f%%" % (ivalue+1,nvalues,100*(ivalue+1)/nvalues)

            # find 5-dim location in the fluxes hypercube for this particular model
            pos = [N.argwhere(self.theta[j]==float(e)).item() for j,e in enumerate(value)]
            pos.append(Ellipsis)  # this adds as many dimensions as necessary

            # ndim=2 ensures same dimensionality of array, even if only one column was read
            data = N.loadtxt(f,usecols=self.cols,ndmin=2)

            # store the read columns (iy) into the appropriate  hypercubes, in the determined location 'pos'.
            for iy in xrange(len(ys)):
                ys[iy][pos] = data[:,iy]

        return x, ys


class Storage:

    def __init__(self,fname):

        """Store data to an HDF5 file.

        Store a list of hypercubes, an x-column, theta arrays, and
        column names, to an HDF5 file.

        Parameters
        ----------
        fname : str
            Path to HDF5 output file. If file exists, we'll try to
            open it in append mode. Otherwise we'll try to create the
            file.

        Usage
        -----
        Instantiate the Storage class by providing the path to the
        HDF5 file. Please read the docstrings of __call__() for the
        storage syntax.

        """

        self.fname = fname


    def __call__(self,groupname,hypercubes,theta,colnames,paramnames,xcol=None,xcolname='xcol'):

        """Perform the storage to HDF5 file.

        A call opens the hdfile, creates a group, stores all data in
        to the group, and closes the file. A new call with the same
        HDF5 file name can store another set of data in a different
        group, but in the same file.

        Parameters
        ----------
        groupname : str
            All data will be stored under this group name. Examples:
               group='models'
               group='libraryXYZ'
               group='/librayA/sublibraryB7'
    
            This group must not exist yet in the output file.

        hypercubes : list
            List of n-dimensional arrays to store in the HDF5
            file. All must have the same shape.

        theta : array
            Variable-length array of 1-d arrays, each containing the
            unique and sorted values of a single model parameter
            (corresponding to one of the axes of the hypercubes). E.g.
               
                theta = array( array(1,2,3), array(0.1,0.4,0.9,1.8), array(-3,1.3,123.) )

        colnames : {seq of strings, None}
            If not None, these are the names of the columns, each
            corresponding to one element in 'hypercubes'. Exactly as
            many column names must be provided as there are members in
            'hypercubes'. If None, generic column names will be
            assigned (col00, col01, etc.)

        paramnames : {seq of strings, None}
            If not None, these are the names of the matched
            parameters, each corresponding to one axis of a
            hypercube. Exactly as many parameter names must be
            provided as there are axes in a hypercube. If None,
            parameter names will be param0, param1, etc.

        xcol : {1-d array, None}
            If not None, this is the dependent variable, and will be
            stored as well. Useful for n-dim interpolation of the
            hypercubes. If None, no x-array will be stored in the HDF5
            file.

        xcolname : str
            Only used if xcol is not None. xcol will be saved under
            this name (default is 'xcol').

        """

        self.groupname = groupname
        self.hypercubes = hypercubes
        self.theta = theta
        self.colnames = colnames
        self.xcol = xcol
        self.xcolname = xcolname
        self.paramnames = paramnames

        self.open_file()     # try to open HDF5 file in append mode (or create new if it doesn't exist yet)
        self.create_group()  # if opened successfully, test of group already exists
        self.store()         # store everything
        self.close_file()

        print "All data stored and file %s closed properly." % self.fname


    def open_file(self):

        try:
            self.hout = h5py.File(self.fname,'a')
        except:
            print "Problem either creating or opening output file %s in 'append' mode." % self.fname
            raise Exception


    def close_file(self):

        try:
            self.hout.close()
        except:
            print "Problem closing HDF5 file %s. Was it open?" % self.fname
            raise Exception


    def create_group(self):

        try:
            self.group = self.hout.create_group(self.groupname)
        except ValueError:
            raise Exception, "Group %s already exists. Chose a different group name, or delete existing group first." % self.groupname


    def store(self): #,fname,xcol=None,axesnames=None,group=None

        # store hypercubes
        for j,cube in enumerate(self.hypercubes):
            name = self.colnames[j]
            print "Storing hypercube %s..." % name
            dset = self.group.create_dataset(name,data=cube,dtype='float32')  # explicitly 4-bit to save storage and RAM
                
        # store theta
        print "Storing theta..."
        dset = self.group.create_dataset('theta',data=self.theta,dtype='float64')  # 2-d nan-padded array

#numpy bug [2190]        dflt = h5py.special_dtype(vlen=N.dtype('float64'))
#numpy bug [2190]        print "Storing theta..."
#numpy bug [2190]
#numpy bug [2190]
#numpy bug [2190]
#numpy bug [2190]#        dset = self.group.create_dataset('theta',data=self.theta, dtype=dflt)
#numpy bug [2190]
#numpy bug [2190]        # need to store the ragged array 'theta' member-by-member, b/c it fails in one go, i.e.
#numpy bug [2190]        #   dset = self.group.create_dataset('theta',data=self.theta, dtype=dflt)
#numpy bug [2190]        # presumably due to a bug in h5py.
#numpy bug [2190]        dset = self.group.create_dataset('theta', (len(self.theta),), dtype=dflt)
#numpy bug [2190]        print "len(self.theta), len(dset), dset.shape, dset.size = ", len(self.theta), len(dset), dset.shape, dset.size
#numpy bug [2190]        for j,e in enumerate(self.theta):
#numpy bug [2190]            
#numpy bug [2190]            print "j, e, type(e), e.dtype: ", j, e, type(e), e.dtype
#numpy bug [2190]            print "dset[j] :", dset[j]
#numpy bug [2190]            dset[j] = e[...]
#numpy bug [2190]            print "dset[j] :", dset[j]
#numpy bug [2190]            print 

        # store xcol, if provided
        if self.xcol is not None:
            print "Storing xcol as '%s'..." % self.xcolname
            dset = self.group.create_dataset(self.xcolname,data=self.xcol,dtype='float64')  # 8-bit, since this is often wavelength (need accuracy)

        # store colnames as a list, to know their order
        dset = self.group.create_dataset('colnames',data=self.colnames)

        # store parameter names as a list
        dset = self.group.create_dataset('paramnames',data=self.paramnames)


#class PadArray:
#
#    """Class for padding a ragged array.
#
#    This is limited to the simple case of a list of 1-d arrays of
#    variable lengths; convert them through padding into a 2-d array of
#    shape (len(list),max([e.size for e in list])). The values of all
#    1-d arrays are written left-bounded into the rows of the array,
#    with their right-hand sides padded with padval (default: nan) up
#    until the max length of all of the 1-d arrays.
#
#    The other direction also works, i.e. providing a 2-d array and
#    returning a list of 1-d arrays, with the specified padval removed
#    from them.
#
#    Parameters:
#    -----------
#    inp : seq
#        List or tuple of 1-d arrays of numbers (or things that are
#        float()-able.
#
#    padval : {float, nan}
#        Pad value to be used. Default is nan. The value can be
#        anything that can be converted by float(), e.g. '3', or 1e7,
#        etc.
#
#    Example:
#    --------
#    # pad a list of 1-d arrays
#    inp = [array([1,2]),array([1,2,3]),array([1,2,3,4,5])]
#    pa = PadArray(inp)
#    pa.unpad
#      Out:   [array([1,2]),array([1,2,3]),array([1,2,3,4,5])]
#    pa.pad
#      Out:  array([[  1.,   2.,  nan,  nan,  nan],
#                   [  1.,   2.,   3.,  nan,  nan],
#                   [  1.,   2.,   3.,   4.,   5.]])
#
#    # unpad a 2-d array
#    inp = array( [1.,2.,-1.,-1.,-1.],
#                 [1.,2.,3., -1.,-1.],
#                 [1.,2.,3.,  4., 5.] )
#    pa = PadArray(inp,padval=-1)
#    pa.pad
#      Out:  array( [1.,2.,-1.,-1.,-1.],
#                   [1.,2.,3., -1.,-1.],
#                   [1.,2.,3.,  4., 5.] )
#    pa.unpad
#      Out:  [array([ 1.,  2.]), array([ 1.,  2.,  3.]), array([ 1.,  2.,  3.,  4.,  5.])]
#
#    """
#
#    def __init__(self,inp,padval=N.nan):
#
#        self.inp = inp
#        self.padval = padval
#        self._setup()
#        self._convert()
#
#
#    def _setup(self):
#
#        try:
#            self.padval = float(self.padval)
#        except ValueError:
#            raise Exception, "padval is not convertible to a floating-point number."
#
#        if isinstance(self.inp,(list,tuple)):
#            # TODO: check if all members of inp can be safely converted to 1-d numerical arrays
#            self.inpmode = 'unpadded'
#            self.unpad = self.inp
#            self.nrow = len(self.unpad)
#            self.ncol = max([e.size for e in self.unpad])
#
#        elif isinstance(self.inp,(N.ndarray)):
#            if self.inp.ndim == 2:
#                self.inpmode = 'padded'
#                self.pad = self.inp
#                self.nrow, self.ncol = self.inp.shape
#            else:
#                raise Exception, "input appears to be an array, but is not 2-d."
#
#        else:
#            raise Exception, "input is neither a sequence of 1-d arrays, nor a 2-d array."
#
#
#    def _convert(self):
#
#        if self.inpmode == 'unpadded':
#            self._pad()
#        elif self.inpmode == 'padded':
#            self._unpad()
#
#
#    def _pad(self):
#
#        self.pad = N.ones((self.nrow,self.ncol))*self.padval
#
#        for j,e in enumerate(self.unpad):
#            self.pad[j,:e.size] = e
#
#
#    def _unpad(self):
#
#        self.unpad = []
#
#        for j in xrange(self.nrow):
#            aux = self.pad[j,:]
#
#            if N.isnan(self.padval):
#                aux = aux[~N.isnan(aux)]
#            else:
#                aux = aux[aux!=self.padval]
#                
#            self.unpad.append(aux)
            
       

# STAND-ALONE FUNCTIONS
# They could be useful outside of the scope of hypercubizer.

def get_pattern_and_paramnames(examplefile):

    """Extract regex matchig pattern and (optinally) the parameter names
    from a single example of a file name to be matched.

    Assuming that the file names to be matched contain some numerical
    parameter values, this function will determine the regex pattern
    needed to extract these parameter values from all similarily named
    files.

    Optionally, if the file names to be matched also contain parameter
    names, these names can also be extracted here.

    Parameters:
    -----------
    examplefile : str
        The filename such as those that will be matched with the
        pattern determined here. Only the basename of 'examplefile'
        will be considered, i.e. no superordinate directories.

    Usage/examples:
    ---------------
    Enclose the numerical values of the parameters that are encoded in
    the file name with round brackets (), and optionally the parameter
    names in square brackets []. E.g., if the example file name is

      'density0.01_gravity123_temperature1e4K.dat'
    
    and there are three numerical parameters, you should supply this:

      example='density(0.01)_gravity(123)_temperature(1e4)K.dat'

    Note that what is in parentheses must be convertible to a floating
    point number.

    If the parameter names are density, gravity and temperature, and
    you wish them to be returned as well, supply this:

      example='[density](0.01)_[gravity](123)_[temperature](1e4)K.dat'

    You don't have to convert all the numbers in a file name to
    matched parameters. E.g. you could just supply:

      example='[density](0.01)_gravity123_[temperature](1e4)K.dat'

    and this would only return the parameter names
    ['density','temperature'], and a regex pattern that matches the
    two numerical values in round brackets.

    Finally, note that if you ask to extract the parameter names (the
    [] brackets), the number of [] and () pairs must be equal (i.e. as
    many parameter names as parameter numerical values).

    Returns:
    --------
    pattern, paramernames : (str,list)

        The regex matching pattern needed to extract the parameter
        names and (optinally) the parameter names from files named in the same fashion as 'examplefile'

    """

#    example = '[tauinf](0.10)[nu](0.3)_[ageb](10.00)_[fmol](.10)_[tesc](.005)_[rcm](6.14)_[rc](1.92)_beta1.8.0'  # infer param names and pattern
#    example2 = 'tauinf(0.10)nu(0.3)_ageb(10.00)_fmol(.10)_tesc(.005)_rcm(6.14)_rc(1.92)_beta1.8.0'               # infer only pattern; parnames either user-set, or generic

    square_pattern = re.compile('\[(.+?)\]')   # look for square brackets; if present, they contain the parameter names

    number = '(.+?)'   # this doesn't check for digits, but it also matches things like '1e7'; we can perform float()-abilty tests after matching
    round_pattern = re.compile(r"\("+number+r"\)")   

    # try to extract parameter values from the supplied example pattern
    values = round_pattern.findall(examplefile)
    if values == []:
        raise Exception, "No numerical parameter values found in 'examplefile'. Make sure to enclose the numerical value is round brackets, e.g. (0.127)."
    else:
        # check if all found values can be converted to floats
        try:
            dummy = N.array(values,dtype=N.float64)
        except ValueError:
            print "Not all parameter values in 'examplefile' (those enclosed in round brackets ()) could be converted to floating-point numbers. Please check."

    # try to extract parameter names
    paramnames = square_pattern.findall(examplefile)
    if paramnames == []:
        # no automatic param names; either user supplies them separately, or we'll do generic ones (par0, par1, etc.)
        print "No automatic parameter names could be determined from the supplied example file pattern."
    else:
        assert (len(values) == len(paramnames)), "Number of round () and square brackets [] in 'examplefile' must be equal (same number of parameter names and parameter values)."
    
    # Generate the actual search pattern: replace all instances of a
    # number in round brackets with the actual 'number' regex pattern
    pattern = re.sub(round_pattern.pattern, number, examplefile)

    # Remove square brackets from the pattern (if any); we don't need
    # them in the regex search pattern
    pattern = pattern.translate(None,'[]')

    return pattern, len(values), paramnames  # caution: paramnames can be None; check externally


def get_uniques(values,returnnumerical=True):

    """Compute unique set of values from a list of lists.

    Parameters
    ----------
    values : list

        List of sequences, e.g.

            values=[('0','0.1','-3'),('0.0'.'1','-2'),...],

        where each entry is a sequence of the matched parameter values
        (as strings) matched per file (from the file names, useing the
        pattern; see class Hypercubes).

    returnnumerical : bool
       If True (default), also return the unique sequences of values
       per parameter in a numerical (float) form.

    Returns
    -------

    uniques : sequence
       The unique sequences of parameter values (per parameter) as
       strings.

    uniques_float : sequence
       The float version of uniques. Optional (only if
       returnnumerical=True).

    shape : list
       A sequence of the lengths of all uniques. This is the N-dim
       array shape used to form N-dim hypercubes ob read columns (see
       class Hypercubes)

    Example
    -------

    values = [ (0,0.1,-3), (0,0.1,-2), (0,0.2,-3), (0,0.2,-2),
               (1,0.1,-3), (1,0.1,-2), (1,0.2,-3), (1,0.2,-2) ]

    uniques, uniques_float, shape = get_uniques(values, returnnumerical=True)
    print uniqes
      [('0','1'),('0.1','0.2'),('-3','-2')]
    print uniqes_float
      [(0.,1.),(0.1,0.2),(-3.,-2.)]
    print shape
      (2,2,2)

    """

    # conversions, dimensions
    vals = N.array(values)  # strings
    uniques = [N.unique(vals[:,j]) for j in range(vals.shape[1])]
    shape = [u.size for u in uniques]

    if returnnumerical == True:
#        uniques_float = [e.astype('float64') for e in uniques]
        uniques_float = [ N.sort(e.astype('float64')) for e in uniques ]
        return uniques, uniques_float, shape
    else:
        warnings.warn("CAUTION! 'uniques' is not necessarily sorted numerically! Give returnnumerical=True, and use the uniques_float returned value.")
        return uniques, shape


def get_files_in_dir(root,verbose=False,returnsorted=True):

    """Recursive walk down from root dir, return list of files (end nodes).

    Parameters
    ----------
    root : str
        Path to the root directory holding the files to be scanned.

    verbose : bool
        If True reports on progress.

    returnsorted : bool
        If True sorts the returned list of files found in root.

    Returns
    -------

    """

    files = []
    sep = os.path.sep

    for dirname, subdirlist, filelist in os.walk(root):
        if verbose == True: print "Entering dir ", dirname
        
        for fname in filelist:
            if verbose == True: print "Adding file ", fname

            files.append(dirname+sep+fname)

    if returnsorted == True:
        files = sorted(files)

    return files


def match_pattern_to_strings(strings,\
                             pattern='tauinf(.+)nu(.+)_ageb(.+)_fmol(.+)_tesc(.+)_rcm(.+)_rc(.+)_beta.+\.0',\
                             op=os.path.basename,\
                             progress=True,patterntype='noname'):

    """Match a regex pattern to a list of strings.

    Match a regular expression 'pattern' to all members of list
    'strings'. If an optional operation 'op' is specified (defaults to
    None), perform op(member) on each member of 'strings' before
    pattern-matching.

    Parameters
    ----------
    strings : list
        List of strings. Each string will be matched against 'pattern'

    pattern : str
        Regex pattern, with parentheses standing for groups to be
        matched.

    op : func reference, optional
        op(elem) is applied to each element in strings before regex
        pattern matching is performed.  Example: op=os.path.basename()

    progress : bool
       If True, progress messages are printed on stdout. (every 100
       processed strings).

    Returns
    -------
    matched_strings : list
        List of strings from the provided list that were actually
        successfully matched against 'pattern'.

    matched_values : list
        List of tuples. Each tuple contains the matched groups of the
        i-th element in matched_strings.

    """

    rg = re.compile(pattern)

    nmatches = rg.groups   # number of expected name matches in each scanned model file name

    matched_strings = []
    matched_values = []

    if progress == True:
        n = len(strings)
        progress_step = int(n)/100   # integeger division op! (no remainder)

    # Try to match each file name to the pattern. Extract parameter
    # values from matching file names. Ignore file names that don't
    # match the pattern.
    for j,s in enumerate(strings):
        
        if progress == True:
            if j % progress_step == 0:
                print "Working on file %d of %d" % (j,n)

        if op is not None:
            try:
                arg = op(s)  # operation 'op' is given by the user. For instance, this could be os.path.basename(s)
            except:
                raise Exception, "Applying op to argument %d, '%s', failed." % (j,s)
        else:
            arg = s

        try:
            matches = rg.search(arg).groups()
            matched_values.append(matches)
            matched_strings.append(s)
        except AttributeError:
            pass

    return matched_strings, matched_values
