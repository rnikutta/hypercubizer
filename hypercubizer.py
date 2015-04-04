"""Find template based model files, read columns, convert them to n-dim hypercubes, store as HDF5.
"""

import os
import re
import numpy as N
import h5py

__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = "20150404"


# TODO: add simple logging


class Hypercubes:

    def __init__(self,rootdir,pattern,cols,colnames=None,xcol=None,xcolname=None,paramnames=None):

        """Init loads all data from rootdir and converts to hypercubes.

        To actually store to HDF5 file, use __call__() function.
        """

        self.rootdir = rootdir        # root directory where to look for files...
        self.pattern = pattern        # that regexp-match the pattern
        self.cols = cols              # from every matched file load column cols
        self.colnames = colnames      # optional: give names to the columns that will be converted; if None, will be named sequentially (['col0','col1',...]).
        self.xcol = xcol              # optional: designate a single column as the independent variable (useful for n-dim interpolation later)
        self.xcolname = xcolname      # optional name for xcol. Default is 'xcol'.
        self.paramnames = paramnames  # optional: names of the matched parameters.

        files = get_files_in_dir(self.rootdir,verbose=False,returnsorted=True)

        self.matched_files, self.matched_values = match_pattern_to_strings(files,pattern=self.pattern,op=os.path.basename,progress=True)

        theta_strings, self.theta, self.hypercubeshape = get_uniques(self.matched_values,returnnumerical=True)
        self.nparams = self.theta.size

        self.x, self.hypercubes = self.convert_columns_to_hypercube(self.matched_files,self.matched_values)


    def __call__(self,storefile,groupname=None):

        """Store to a group in an HDF5 file, if you like. You should like."""

        self.storefile = storefile
        self.groupname = groupname

        self.store()


    def store(self):

        """The actual storage method."""

        if self.groupname is None:
            try:
                self.groupname = os.path.basename(os.path.normpath(self.rootdir))
            except:
                raise Exception

        print "Storing all data to file '%s', under group name '%s'" % (self.storefile,self.groupname)
        self.S = Storage(self.storefile)
        self.S(self.groupname,self.hypercubes,self.theta,colnames=self.colnames,xcol=self.x,xcolname=self.xcolname)


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


    def __call__(self,groupname,hypercubes,theta,colnames=None,xcol=None,xcolname='xcol',paramnames=None):

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

        colnames : sequence of strings, or None
            If not None, these are the names of the columns, each
            corresponding to one element in 'hypercubes'. Exactly as
            many column names must be provided as there are members in
            'hypercubes'. If None, column names will be col0, col1,
            etc.

        xcol : 1-d array, or None
            If not None, this is the dependent variable, and will be
            stored as well. Useful for n-dim interpolation of the
            hypercubes. If None, no x-array will be stored in the HDF5
            file.

        xcolname : str
            Only used if xcol is not None. xcol will be saved under
            this name (default is 'xcol').

        paramnames : sequence of strings, or None
            If not None, these are the names of the matched
            parameters, each corresponding to one axis of a
            hypercube. Exactly as many parameter names must be
            provided as there are axes in a hypercube. If None,
            parameter names will be param0, param1, etc.

        """

        self.groupname = groupname
        self.hypercubes = hypercubes
        self.theta = theta
        self.colnames = colnames
        self.xcol = xcol
        self.xcolname = xcolname
        self.parnames = parnames

        self.sanity()        # sanity checks

        self.open_file()     # try to open HDF5 file in append mode (or create new if it doesn't exist yet)
        self.create_group()  # if opened successfully, test of group already exists
        self.store()         # store everything
        self.close_file()

        print "All data stored and file %s closed properly." % self.fname


    def sanity(self):
        
        # all hypercubes equal shape
        assert (len(set([h.shape for h in self.hypercubes])) == 1), "Not all cubes in 'hypercubes' have the same shape."

        # if xcol is given, its length must be equal to the size of each hypercube's last axis
        if self.xcol is not None:
            assert (self.xcol.size == self.hypercubes[0].shape[-1]), "xcol was given; its length must be equal to the size of each hypercube's last axis."

        # number of colnames (if provided) must be consistent with number of columns given
        if self.colnames is not None:
            assert (len(self.colnames) == len(self.hypercubes)), "The number of column names given in 'colnames' must be equal to the number of dimensions in each hypercube (plus one, of xcol was also given)."
        else:
            self.colnames = ['col' + '%03d' % j for j in xrange(len(self.hypercubes))]

        if self.paramnames is not None:
            assert ( len(self.paramnames) == len(self.hypercubes) ),  "The number of supplied parameter must equal the number of hypercubes."
        else:
            self.paramnames = ['param' + '%03d' % j for j in xrange(len(self.hypercubes))]
            

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
            print "Storing column %s..." % name
            dset = self.group.create_dataset(name,data=cube,dtype='float32')  # explicitly 4-bit to save storage and RAM
                
        # store theta
        dflt = h5py.special_dtype(vlen=N.dtype('float64'))
        print "Storing theta..."
        dset = self.group.create_dataset('theta',data=self.theta, dtype=dflt)

        # store xcol, if provided
        if self.xcol is not None:
            print "Storing xcol as '%s'..." % self.xcolname
            dset = self.group.create_dataset(self.xcolname,data=self.xcol,dtype='float64')  # 8-bit, since this is often wavelength (need accuracy)

        # store colnames as a list, to know their order
        dset = self.group.create_dataset('colnames',data=self.colnames)

        

# STAND-ALONE FUNCTIONS.
# They could be useful outside of the scope of hypercubizer.

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
        uniques_float = N.array([e.astype('float64') for e in uniques])
        return uniques, uniques_float, shape

    else:
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
                             progress=True):

    """Match a regexp pattern to a list of strings.

    Match a regular expression 'pattern' to all members of list
    'strings'. If an optional operation 'op' is specified (defaults to
    None), perform op(member) on each member of 'strings' before
    pattern-matching.

    Parameters
    ----------
    strings : list
        List of strings. Each string will be matched against 'pattern'

    pattern : str
        Regexp pattern, with parentheses standing for groups to be
        matched.

    op : func reference, optional
        op(elem) is applied to each element in strings before regexp
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
    

    # pattern to look for in file names
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
