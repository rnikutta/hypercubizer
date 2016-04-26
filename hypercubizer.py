"""Find template-based model files, read columns, convert them to n-dim hypercubes, store as HDF5.
"""

import os, sys, re, warnings
import numpy as N
import h5py
from externals.padarray import padarray
import pyfits
import filefuncs

__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = "20160426"

# TODO: add simple logging

class Hypercubes:

    def __init__(self,rootdir,pattern,hypercubenames=None,func='asciitable',**kwargs):

        mykwargs = kwargs
        mykwargs['hypercubenames'] = hypercubenames
        
        self.hypercubenames = hypercubenames
        self.func = getattr(filefuncs,func)
        self.funcname = self.func.func_name
        
        # regex pattern, number of values, extracted parameter names (if any)
        self.pattern, self.Nparam, self.paramnames = get_pattern_and_paramnames(pattern)

        self.rootdir = rootdir

        # get a list of all files under rootdir
        files = get_files_in_dir(self.rootdir,verbose=False,returnsorted=True)
        
        # match all files to pattern, keep only the matching ones, and a list of matched numerical values
        self.matched_files, self.matched_values = match_pattern_to_strings(files,pattern=self.pattern,op=os.path.basename,progress=True)

        # turn the list of all matched values (per file) into lists of unique values (per parameter)
        theta_strings, self.theta, self.hypercubeshape = get_uniques(self.matched_values,returnnumerical=True)

        # return hypercubes, but also update: theta, paramnames
        self.hypercubes = self.convert(self.matched_files,self.matched_values,**mykwargs)

        # As described in issue #5 on bitbucket, we need to work
        # around a numpy and/or h5py bug. That's why we nan-pad
        # self.theta, and store as a regular 2-d array (self.theta.pad)
        self.theta = padarray.PadArray(self.theta)  # has members .pad (2-d array) and .unpad (list of 1-d arrays)

        self.sanity()


    def sanity(self):
        
        assert (len(set([h.shape for h in self.hypercubes])) == 1), "Not all cubes in 'hypercubes' have the same shape."

        if self.hypercubenames is not None:
            assert (len(self.hypercubenames) == self.Nhypercubes),\
                "The number of hypercube names given in 'hypercubenames' (%d) must be equal to the number of hypercubes (%d)." % (len(self.hypercubenames),self.Nhypercubes)
        else:
            self.hypercubenames = ['hc' + '%02d' % j for j in xrange(self.Nhypercubes)]  # generic hypercube names, if none provided


    def convert(self,files,values,**kwargs):

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
        datasets, axnames, axvals, hypercubenames = self.func(files[0],**kwargs)  # kwargs are: cols, e.g.: cols=(0,(1,2,3)), xcol are the values of the column given by kwarg 'xcol'

        self.hypercubenames = hypercubenames
        
        # how many cubes?
        self.Nhypercubes = len(datasets)

        # one dataset has hoe many dimensions?
        ndim = datasets[0].ndim

        self.axnames = axnames
        self.axvals = axvals
        
        if self.axnames is None:
            self.axnames = ['ax%02d' % j for j in xrange(ndim)]

        if self.axvals is None:
            self.axvals = [N.arange(axissize) for axissize in datasets[0].shape] # this explictly assumes that all datasets are of same shape!
            
        # extend hypercube shape by whatever the returned shape of datasets is
        self.hypercubeshape = self.hypercubeshape + list(datasets[0].shape)


        # prepare a list of n-dimensional hypercubes to hold the
        # re-shaped column data (one hypercube per column read)
        hypercubes = [N.zeros(shape=self.hypercubeshape,dtype=N.float32) for j in xrange(self.Nhypercubes)] # careful not to reference the same physical array n times

        nvalues = float(len(values))

        # LOOP OVER GOOD FILES
        print "Converting matched models to hypercubes"
        for ivalue,value in enumerate(values):

            f = files[ivalue]

            progressbar(ivalue,nvalues,"Working on model")

            # find 5-dim location in the fluxes hypercube for this particular model
            pos = [N.argwhere(self.theta[j]==float(e)).item() for j,e in enumerate(value)]
            pos.append(Ellipsis)  # this adds as many dimensions as necessary

            datasets, axnames, axvals, hypercubenames = self.func(f,**kwargs)  # kwargs are: cols, e.g.: cols=(0,(1,2,3))
            
            # store the read datasets (iy) into the appropriate hypercubes, in the determined N-dim index 'pos'.
            for iy in xrange(len(hypercubes)):
                hypercubes[iy][pos] = datasets[iy]


        # extend list of parameter names by the axes of a single dataset
        self.paramnames = self.paramnames + list(self.axnames)
        
        if isinstance(self.axvals,list):
            self.theta = self.theta + self.axvals
        elif isinstance(self.axvals,N.ndarray):
            self.theta.append(self.axvals)

        self.Nparam += len(self.axnames)
                      
        return hypercubes


    def store2hdf(self,filename=None,groupname=None): # e.g. filename='foo.hdf', groupname='imgdata'

        """Store attribiutes of self to an hdf5 file.

        Parameters:
        -----------
        filename: None or str
            If None, a default string name will be computed as the
            tail of self.rootdir (plus hdf5 extension). If not None,
            filename is the name of the hdf5 output file. If filename
            does not end with '.hdf5', this suffix will be added.

        groupname : None or str
            The name of the top-level group in the hdf5 file to write
            the output to. If None, the root group will simply be '/'.

        Examples:
        ---------

        # After computing H...
        H.store2hdf(filename='foo.hdf5',grouname='mynicemodels')

        """
        
        def store_attrs(groupname,obj,attrs):

            group = hdfout.require_group(groupname)
            
            for attr in attrs:
                print "    Storing attribute: ", attr

                value = getattr(obj,attr)

                # force reduced data size (4-byte 32-bit floats) for floating point arrays with more than one dimension
                if isinstance(value,N.ndarray) and value.ndim > 1:
                    dtype = N.float32
                else:
                    dtype = None

                # create dataset
                try:
                    dataset = group.create_dataset(attr,data=value,compression='gzip',compression_opts=9,dtype=dtype)
                except TypeError:
                    dataset = group.create_dataset(attr,data=value,dtype=dtype)

                        
        # default hdf5 file name
        if filename is None:
            filename = os.path.split(self.rootdir)[-1]

        # default top-level group name
        if groupname is None:
            groupname = '/'

        # construct the hdf5 file name (path and file name)
        hdffile = os.path.join(self.rootdir,filename)
        if not hdffile.endswith('.hdf5'):
            hdffile = hdffile + '.hdf5'

        print "Storing results to hdf5 file"
        print "File: %s" % hdffile

        # open file for appending; creates it if file doesn't exist yet
        hdfout = h5py.File(hdffile,'a')

        
        # store all common metadata
        attrs = ('pattern','rootdir','Nhypercubes','Nparam','hypercubenames','hypercubeshape','paramnames','funcname')
        store_attrs(groupname,self,attrs)

        # store theta
        obj = DictToObject({'theta':self.theta.pad})
        attrs = ('theta',)
        store_attrs(groupname,obj,attrs)

        
        # make a temporary object instance to hold the hypercubes
        obj = DictToObject( dict(zip((self.hypercubenames),(self.hypercubes))) )

        # store all hypercubes to a sub-group (called 'hypercubes') of the top-level group
        group = groupname + '/hypercubes'
        attrs = self.hypercubenames
        store_attrs(group,obj,attrs)

        print "Closing hdf5 file."
        hdfout.close()

        
class DictToObject:

    """Simple class to hold the contents of a dictionary as members of an object instance.

    Example:
    --------

    dict = {'foo' : 7, 'bar' : ['orange','yellow']}
    obj = DictToObject(dict)
    obj.foo
      7
    obj.bar
      ['orange','yellow']

    """
    
    def __init__(self,dic=None):
        if dic is not None:
            try:
                for k,v in dic.items():
                    setattr(self,k,v)
            except:
                raise



#TMP#numpy bug [2190]        dflt = h5py.special_dtype(vlen=N.dtype('float64'))
#TMP#numpy bug [2190]        print "Storing theta..."
#TMP#numpy bug [2190]
#TMP#numpy bug [2190]
#TMP#numpy bug [2190]
#TMP#numpy bug [2190]#        dset = self.group.create_dataset('theta',data=self.theta, dtype=dflt)
#TMP#numpy bug [2190]
#TMP#numpy bug [2190]        # need to store the ragged array 'theta' member-by-member, b/c it fails in one go, i.e.
#TMP#numpy bug [2190]        #   dset = self.group.create_dataset('theta',data=self.theta, dtype=dflt)
#TMP#numpy bug [2190]        # presumably due to a bug in h5py.
#TMP#numpy bug [2190]        dset = self.group.create_dataset('theta', (len(self.theta),), dtype=dflt)
#TMP#numpy bug [2190]        print "len(self.theta), len(dset), dset.shape, dset.size = ", len(self.theta), len(dset), dset.shape, dset.size
#TMP#numpy bug [2190]        for j,e in enumerate(self.theta):
#TMP#numpy bug [2190]            
#TMP#numpy bug [2190]            print "j, e, type(e), e.dtype: ", j, e, type(e), e.dtype
#TMP#numpy bug [2190]            print "dset[j] :", dset[j]
#TMP#numpy bug [2190]            dset[j] = e[...]
#TMP#numpy bug [2190]            print "dset[j] :", dset[j]
#TMP#numpy bug [2190]            print 

       

# STAND-ALONE FUNCTIONS
# They could be useful outside of the scope of hypercubizer.

def progressbar(i,n,prefix="",suffix=""):
    
    sys.stdout.write("\r%s %d of %d (%.1f percent) %s" % (prefix,i+1,n,((i+1)*100/float(n)),suffix))
    sys.stdout.flush()
                
    if i == n-1:
        print "\n"


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
    print "Matching file names to pattern"
    for j,s in enumerate(strings):
        
        if progress == True:
            progressbar(j,n,"Working on model")
                
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
