import os


class Hypercubes:

    def __init__(self,rootdir,pattern,cols,xcol=None,colnames=None):

        self.rootdir = rootdir   # root directory where to look for files...
        self.pattern = pattern   # that regexp-match the pattern
        self.cols = cols         # from every matched file load column cols
        self.xcol = xcol         # optional: designate a single column as the independent variable (useful for n-dim interpolation later)
        self.colnames = colnames # optional: give names to the columns that will be converted; if not None and if xcol is not None, then the first name in colnames must be the name of the xcol

        files = get_files_in_dir(self.rootdir,verbose=True,returnsorted=True)

        matched_files, matched_values = match_pattern_to_strings(files,pattern=self.pattern,op=os.path.basename,progress=True)

        theta_strings, self.theta, self.hypercubeshape = get_uniqes(matched_values,returnnumerical=True)







def convert_columns_to_hypercube(files,values,cols=None):

    """Load fluxes from all pattern-matched files, and store them in a
    n-dimensional hypercube, properly shaped according to the matched
    parameter values.

    Given the set of unique parameter values (for every parameter),
    and the set of parameter values for a given file, we can determine
    the n-dimensional position the the current parameter values,
    i.e. where in the hypercube to put that file.

    We have to do these somewhat complicated antics because we can not
    rely on the correct sorting order of the files (wrt to the
    parameter values they represent). If we could, we could simply
    concatenate the numbers from all files, and then reshape them to
    the n-dimensional hypercube using the known unique parameter
    values sets.

    Parameters:
    -----------

    files : list
        List of path to all files that have been pattern-matched for
        the parameter values matched in their file names.

    values : list
        List of parameter values match in each file in 'files'.

    cols : int tuple
        Indices of the columns to load from each file. Their order
        will be obeyed. As many hypercubes will be returned as there
        are column indices specified here.

 matters. 


wavelength and flux columns in 'files'.

    """

    # conversions, dimensions
#    vals = N.array(values)  # strings
#    uniques = [N.unique(vals[:,j]) for j in range(vals.shape[1])]
#
#    valsa = convert_gridparams(vals,mappings=None)
#    uniquesa = [N.unique(valsa[:,j]) for j in range(valsa.shape[1])]
#    shape = [u.size for u in uniques]
    
    theta, theta_float, shape = get_uniques(values,returnnumerical=True)


#################

    # GET SHAPE FROM FIRST FILE
    data = N.loadtxt(files[0],usecols=cols,unpack=False)
    print "data.shape = ", data.shape

    print "shape = ", shape
    shape.append(data.shape[0])   # length of the columns to be read
    print "shape = ", shape
#    x = data[:,cols[0]]
#    nx = x.size
#    shape.append(nx)   # lengths of wavelength array

    # list of n-dimensional hypercubes to hold the re-shaped column data
#    ys = [N.zeros(shape=shape) for j in xrange(len(cols)-1)]
    ys = [N.zeros(shape=shape)]*len(cols)
    print "len(ys), ys[0].shape = ", len(ys), ys[0].shape

    # 2nd pass: get fluxes
    c = 0  # silly counter

    nvalues = float(len(values))


    # LOOP OVER GOOD FILES
    for ivalue,value in enumerate(values):

        f = files[ivalue]

        c += 1
        if c % 100 == 0:
            print "Working on model %d of %d, %.2f" % (c,nvalues,c/nvalues)


        # find 5-dim location in the fluxes hypercube for this particular model
        pos = [N.argwhere(theta[j]==e).item() for j,e in enumerate(value)]
        pos.append(Ellipsis)  # this adds as many dimensions as necessary (here two: for i and wavelength)


        data = N.loadtxt(f,usecols=cols)
#        x = data[:,cols[0]]
#        y = data[:,cols[1:]]

        for iy in xrange(data.shape[1]):
            ys[iy][pos] = data[:,iy].T


    return ys
















############
def get_uniques(values,returnnumerical=True):

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

    """Recursive walk down from root dir, return list of files (end nodes)."""


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
    
    import os
    import re

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
                arg = op(s)  # e.g. os.path.basename(f)
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




class Storage:

    """Parameters:
    -----------

    fname : str
        Path to HDF5 output file. If file exists, we'll try opening it
        in append mode. Otherwise we'll try to create the file.

    hypercubes : list
        List of n-dimensional arrays to store in the HDF5 file. All
        must have the same shape.

    theta : array
        Variable-length array of 1-d arrays, each containing the
        unique and sorted values of a single model parameter
        (corresponding to one of the axes of the hypercubes). E.g.
           
            theta = array( array(1,2,3), array(0.1,0.4,0.9,1.8), array(-3,1.3,123.) )

    group : str
        All data will be stored under this group name. Examples:
           group='models'
           group='libraryXYZ'
           group='/librayA/sublibraryB7'
    
        This group must not exist yet in the output file.

    colnames : sequence (optional)
        If given (i.e. not None), these are the names of the columns,
        each corresponding to one element in 'hypercubes'. Exactly as
        many column names must be provided as there are members in
        'hypercubes'.

    xcol : 1-d array (optional)
        If not None, this is the dependent variable. Useful for n-dim
        interpolation of the hypercubes.

    xcolname : list (optional)
        Only used if xcol is not None. xcol will be saved under this
        name (else the default 'xcol' will be used).

    """

    # TODO: add simple logging


#    def __init__(self,fname,hypercubes,theta,groupname,colnames=None,xcol=None,xcolname='xcol'):
    def __init__(self,fname,verbose=1):

        self.fname = fname



    def __call__(self,groupname,hypercubes,theta,colnames=None,xcol=None,xcolname='xcol'):

        self.groupname = groupname
        self.hypercubes = hypercubes
        self.theta = theta
        self.colnames = colnames
        self.xcol = xcol
        self.xcolname = xcolname

        self.sanity()        # sanity checks

        self.open_file()     # try to open HDF5 file in append mode (or create new if it doesn't exist yet)
        self.create_group()  # if opened successfully, test of group already exists
        self.store()         # store everything
        self.close_file()




    def open_file(self):

        try:
            self.hout = h5py.File(self.fname,'a')
        except:
            raise Exceptions, "Output file %s either can't be created or can't be opened in 'append' mode." % self.fname


    def close_file(self):

        try:
            self.hout.close()
        except:
            raise Exception, "Problems closing HDF5 file %s. Was it open?" % self.fname


    def create_group(self):

        try:
            self.group = self.hout.create_group(self.groupname)
        except ValueError:
            raise Exception, "Group %s already exists. Chose a different group name, or delete existing group first." % self.groupname




    def sanity(self):
        
        # all hypercubes equal shape
        assert (len(set([h.shape for h in self.hypercubes])) == 1), "Not all cubes in 'hypercubes' have the same shape."

        # if xcol is given, its length must be equal to the size of each hypercube's last axis
        if self.xcol is not None:
            assert (self.xcol.size == self.hypercubes[0].shape[-1]), "xcol was given; its length must be equal to the size of each hypercube's last axis."

        # number of colnames (if provided) must be consistent with number of columns given
        if self.colnames is not None:
            targetlength = hypercubes[0].ndim

            if self.xcol is not none:
                targetlength += 1

            assert (len(self.colnames) == targetlength), "The number of column names given in 'colnames' must be equal to the number of dimensions in each hypercube (plus one, of xcol was also given)."

        else:
            self.colnames = ['col' + '%03d' % j for j in xrange(len(self.hypercubes))]


    def store(self): #,fname,xcol=None,axesnames=None,group=None

        # store hypercubes
        for j,cube in enumerate(self.hypercubes):
            dset = self.group.create_dataset(name,data=cube)
                
        # store theta
        dflt = h5py.special_dtype(vlen=N.dtype('float64'))
        dset = self.group.create_dataset('theta',data=uniquesa, dtype=dflt)

        # store xcol, if provided
        if self.xcol is not None:
            dset = self.group.create_dataset(self.xcolname,data=self.xcol)

        # store colnames as a list, to know their order
        dset = self.group.create_dataset('colnames',data=self.colnames)

        



#    if savefile is not None:
#        print "Saving arrays to file", savefile
# 
#       
#        h = h5py.File(savefile,'w')
#        dummy = h.create_dataset('age',data=x)
#
#        dummy = h.create_dataset('theta',data=uniquesa, dtype=dflt)
#
##        dummy = h.create_dataset('theta',data=N.array(uniquesa))
#        for iy,y_ in enumerate(ys):
#            name_ = colnames[1:][iy]
#            dummy = h.create_dataset(name_,data=y_)
#
#
#        h.close()

