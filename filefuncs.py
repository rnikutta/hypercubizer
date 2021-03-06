"""Built-in file reading functions."""

import numpy as N
import astropy.io.fits as pyfits
import pandas as pd

__author__ = "Robert Nikutta <robert.nikutta@gmail.com>"
__version__ = "20180921"

def asciitable(filename,cols=(1,),xcol=None,xcolname=None,hypercubenames=None,**kwargs):

    """Read a column-based data file (ASCII).

    This version reads all colums at once, then splits into one-column
    datasets.

    Parameters:
    -----------
    filename : str
        Path to the ascii file to be read.

    cols : {int, tuple of ints}
        Column numbers (counting from 0) to be read. Every column will
        be converted into a separate hypercube.

    xcol : {int, None}
        If not None, this is a column number that will be read
        independently of 'cols', and the resulting 1d array of values
        will be returned to teh caller, and will be added as an extra
        dimension to the hypercubes. (E.g. the 0-th column might be
        wavelength and the next one is a flux to be read using 'cols
        '; then the resulting flux hypercube will have shape
        (Npar1,Npar2,...,Nwave). If None, None will be returned.

    xcolname : {str, None}
        The value of xcolname will be passed-through to the called.

    hypercubenames : {None, list of strings}
        If not None, then currently only length-one list. This a
        pass-through arg, which will be returned back to
        caller. Allows for user to pass own value for
        'hypercubenames'.


    TODO: allow ranges of columns to be given as (3-5), etc.

    """

    print(filename)
    # if single column index given, convert it to a lenght-one tuple
    if isinstance(cols,int):
        cols = (cols,)

    if isinstance(xcolname,str):
        xcolname = (xcolname,)

    # add xcol to the list of cols, if xcol not None
    if isinstance(xcol,int):
        print("adding xcol", xcol)
        cols = sorted(list(cols) + [xcol])

    # fast-read all requested cols, including xcol (if any) in the mix
    datasets = N.atleast_2d( pd.read_csv(filename,engine='c',delim_whitespace=True,header=None,comment='#',usecols=cols).values.T )

    # if xcol was given, separate the xcol values from the other columns
    if isinstance(xcol,int):
        sel = N.array([True if c != xcol else False for c in cols])
        #print "sel", sel
        xcol = datasets[~sel,:]
        datasets = datasets[sel,:]

    return datasets, xcolname, xcol, hypercubenames
#    return datasets, axnames, axvals, hypercubenames


def fitsfile_clumpy(filename,ext=None,header=True,**kwargs):

    """Read a (CLUMPY) fits file.

    Parameters:
    -----------

    filename : str
        Path to the CLUMPY .fits file to be read.

    ext : {int,str}
        The FITS extension to be read. Either as EXTVER, specifying
        the HDU by an integer, or as EXTNAME, giving the HDU by name.

        For CLUMPY FITS files:
           0 or 'imgdata' is the image cube
    `      1 or 'clddata' is the projected map of cloud number per ine-of-sight.

    header : bool
        If True, the HDU header will also be read. Not used currently.

    """

    if 'hypercubenames' in kwargs:
        ext = kwargs['hypercubenames'][0]
        
    assert (isinstance(ext,(int,str))),\
        "'ext' must be either integer or a string, specifying the FITS extension by number or by name, respectively."
    
    dataset, header = pyfits.getdata(filename,ext,header=header)  # dataset.shape is (Nwave,Nypix,Nxpix) for 3D, and (Nypix,Nxpix) for 2D.

    x = N.arange(float(header['NAXIS1']))
    y = N.arange(float(header['NAXIS2']))

#    x = range(header['NAXIS1'])
#    y = range(header['NAXIS2'])

    if dataset.ndim == 2:
        axes = None
        axnames = ['x','y']
        axvals = [x,y]

    elif dataset.ndim == 3:
        axes = (0,2,1)
        wave = N.array([v for k,v in header.items() if k.startswith('LAMB')])
        axnames = ['wave','x','y']
        axvals = [wave,x,y]

    dataset = N.transpose(dataset,axes=axes)  # now it's (Nwave,Nxpix,Nypix) for 3D, and (Nxpix,Nypix) for 2D.

    datasets = [dataset]  # has to be a list for function 'convert'
    hypercubenames = kwargs['hypercubenames']
    
    return datasets, axnames, axvals, hypercubenames
