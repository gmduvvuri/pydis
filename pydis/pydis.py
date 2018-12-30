# -*- coding: utf-8 -*-
"""
pyDIS: A simple one dimensional spectra reduction and analysis package

Created with the Apache Point Observatory (APO) 3.5-m telescope's
Dual Imaging Spectrograph (DIS) in mind. YMMV

e.g. DIS specifics:
- have BLUE/RED channels
- hand-code in that the RED channel wavelength is backwards
- dispersion along the X, spatial along the Y axis

"""

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Cursor
import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel
from scipy.optimize import curve_fit
import scipy.signal
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import SmoothBivariateSpline
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
import warnings
import pdb #JSP - debugging
# import datetime
# from matplotlib.widgets import SpanSelector

__all__ = ['OpenImg', 'ap_trace', 'ap_extract', 'HeNeAr_fit', 'mapwavelength',
           'biascombine', 'flatcombine', 'line_trace', 'find_peaks',
           'lines_to_surface', 'normalize', 'DefFluxCal', 'ApplyFluxCal',
           'AirmassCor']


def _mag2flux(wave, mag, zeropt=48.60):
    '''
    Convert magnitudes to flux units. This is important for dealing with standards
    and files from IRAF, which are stored in AB mag units. To be clear, this converts
    to "PHOTFLAM" units in IRAF-speak. Assumes the common flux zeropoint used in IRAF

    Parameters
    ----------
    wave : 1d numpy array
        The wavelength of the data points
    mag : 1d numpy array
        The magnitudes of the data
    zeropt : float, optional
        Conversion factor for mag->flux. (Default is 48.60)

    Returns
    -------
    Flux values!
    '''

    c = 2.99792458e18 # speed of light, in A/s
    flux = 10.0**( (mag + zeropt) / (-2.5) )
    return flux * (c / wave**2.0)


def _gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function, for internal use only

    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def _WriteSpec(spec, wfinal, ffinal, efinal, trace):
    # write file with the trace (y positions)
    tout = open(spec+'.trace','w')
    tout.write('#  This file contains the x,y coordinates of the trace \n')
    for k in range(len(trace)):
        tout.write(str(k)+', '+str(trace[k]) + '\n')
    tout.close()

    # write the final spectrum out
    fout = open(spec+'.spec','w')
    fout.write('#  This file contains the final extracted (wavelength,flux,err) data \n')
    for k in range(len(wfinal)):
        fout.write(str(wfinal[k]) + '  ' + str(ffinal[k]) + '  ' + str(efinal[k]) + '\n')
    fout.close()
    return


def _CheckMono(wave):
    '''
    Check if the wavelength array is monotonically increasing. Return a
    warning if not. NOTE: because RED/BLUE wavelength direction is flipped
    it has to check both increasing and decreasing. It must satisfy one!

    Method adopted from here:
    http://stackoverflow.com/a/4983359/4842871
    '''

    # increasing
    up = all(x<y for x, y in zip(wave, wave[1:]))

    # decreasing
    dn = all(x>y for x, y in zip(wave, wave[1:]))

    if (up is False) and (dn is False):
        print("WARNING: Wavelength array is not monotonically increasing!")

    return


class OpenImg(object):
    """
    A simple wrapper for astropy.io.fits (pyfits) to open and extract
    the data we want from images and headers.

    Parameters
    ----------
    file : string
        The path to the image to open
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is True)

    Returns
    -------
    image object
    """
    # maybe there should be an option to feed a dictionary in with the relevant keywords incase for different spectrographs, they don't match the expectations here
    
    def __init__(self, file, trim=True):
        self.file = file
        self.trim = trim

        hdu = fits.open(file)
        if trim is True:
            self.datasec = hdu[0].header['DATASEC'][1:-1].replace(':',',').split(',')
            d = list(map(int, self.datasec))
            self.data = hdu[0].data[d[2]-1:d[3],d[0]-1:d[1]]
        else:
            self.data = hdu[0].data

        if 'AIRMASS' in hdu[0].header:
            self.airmass = hdu[0].header['AIRMASS']
        elif 'ZD' in hdu[0].header:
            # try using the Zenith Distance (assume in degrees)
            ZD = hdu[0].header['ZD'] / 180.0 * np.pi
            self.airmass = 1.0/np.cos(ZD) # approximate airmass
        else:
            self.airmass = 1.0

        # compute the approximate wavelength solution
        try:
            self.disp_approx = hdu[0].header['DISPDW']
            self.wcen_approx = hdu[0].header['DISPWC']
            # the red chip wavelength is backwards (DIS specific)
            clr = hdu[0].header['DETECTOR']
            if (clr.lower()=='red'):
                sign = -1.0
            else:
                sign = 1.0
            self.wavelength = (np.arange(self.data.shape[1]) -
                               (self.data.shape[1])/2.0) * \
                              self.disp_approx * sign + self.wcen_approx
        except KeyError:
            # if these keywords aren't in the header, just return pixel #
            self.wavelength = np.arange(self.data.shape[1])

        try:
            self.gain = hdu[0].header['GAIN']  # gain in electrons/ ADU
        except KeyError:
            self.gain = 1  # gain in electrons/ ADU
            print("Warning: Missing 'GAIN' keyword in header, check input, setting to unity")

        try:
            self.rdnoise = hdu[0].header['RDNOISE']  # readnoise in electrons/pix
        except KeyError:
            self.rdnoise = 0  # readnoise in electrons/pix
            print("Warning: Missing 'RDNOISE' keyword in header, check input, setting to zero")

        self.exptime = hdu[0].header['EXPTIME']
        
        try:
            self.dateobs = hdu[0].header['DATE-OBS']
        except KeyError:
            print("Missing DATE-OBS in header...")
            self.dateobs = ""

        try:
            self.timesys = hdu[0].header['TIMESYS'].lower()
        except KeyError:
            self.timesys = 'utc'
            print("Unknown time system, assuming UTC")

        hdu.close(closed=True)

        # return raw, exptime, airmass, wapprox, gain, rdnoise


def biascombine(biaslist, output='BIAS.fits', trim=True, silent=True):
    """
    Combine the bias frames in to a master bias image. Currently only
    supports median combine.

    Parameters
    ----------
    biaslist : str
        Path to file containing list of bias images.
    output: str, optional
        Name of the master bias image to write. (Default is "BIAS.fits")
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is True)
    silent : bool, optional
        If False, print details about the biascombine. (Default is True)

    Returns
    -------
    bias : 2-d array
        The median combined master bias image
    """

    # assume biaslist is a simple text file with image names
    # e.g. ls flat.00*b.fits > bflat.lis
    files = np.genfromtxt(biaslist,dtype=np.str)

    if silent is False:
        print('biascombine: combining ' + str(len(files)) + ' files in ' + biaslist)

    for i in range(0,len(files)):
        hdu_i = fits.open(files[i])

        if trim is False:
            im_i = hdu_i[0].data
        if trim is True:
            datasec = hdu_i[0].header['DATASEC'][1:-1].replace(':',',').split(',')
            d = list(map(int, datasec))
            im_i = hdu_i[0].data[d[2]-1:d[3],d[0]-1:d[1]]

        # create image stack
        if (i==0):
            all_data = im_i
        elif (i>0):
            all_data = np.dstack( (all_data, im_i) )
        hdu_i.close(closed=True)

    # do median across whole stack
    bias = np.nanmedian(all_data, axis=2)

    # write output to disk for later use
    hduOut = fits.PrimaryHDU(bias)
    hduOut.writeto(output, overwrite=True)
    return bias


def overscanbias(img, cols=(1,), rows=(1,)):
    '''
    Generate a bias frame based on overscan region.
    Can work with rows or columns, pass either kwarg the limits:

    >>> bias = overscanbias(imagedata, cols=(1024,1050))  # doctest: +SKIP

    '''
    bias = np.zeros_like(img)
    if len(cols) > 1:
        bcol = np.nanmean(img[:, cols[0]:cols[1]], axis=0)
        for j in range(img.shape()[1]):
            img[j,:] = bcol

    elif len(rows) > 1:
        brow = np.nanmean(img[rows[0]:rows[1], :], axis=1)
        for j in range(img.shape()[0]):
            img[j,:] = brow

    else:
        print('OVERSCANBIAS ERROR: need to pass either cols=(a,b) or rows=(a,b),')
        print('setting bias = zero as result!')

    return bias


def flatcombine(flatlist, bias, output='FLAT.fits', trim=True, mode='spline',
                display=True, flat_poly=5, response=True, Saxis=1,badmask=None):
    """
    Combine the flat frames in to a master flat image. Subtracts the
    master bias image first from each flat image. Currently only
    supports median combining the images.

    Parameters
    ----------
    flatlist : str
        Path to file containing list of flat images.
    bias : str or 2-d array
        Either the path to the master bias image (str) or
        the output from 2-d array output from biascombine
    output : str, optional
        Name of the master flat image to write. (Default is "FLAT.fits")
    response : bool, optional
        If set to True, first combines the median image stack along the
        spatial (Y) direction, then fits polynomial to 1D curve, then
        divides each row in flat by this structure. This nominally divides
        out the spectrum of the flat field lamp. (Default is True)
    mode: str, optional
        Default is 'spline'; but can also be 'poly' -- defines the method of fitting used to determine the 1d flat curve when response = True. If 'spline', the 'flat_poly' keyword is not used. 'spline' uses the UnivariateSpline method of scipy.interpolate with ext=0, k=2 ,s=0.001
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is True)
    display : bool, optional
        Set to True to show 1d flat, and final flat (Default is False)
    flat_poly : int, optional
        Polynomial order to fit 1d flat curve with. Only used if
        response is set to True. (Default is 5)
    Saxis : int, optional
        Set which axis the spatial dimension is along. 1 = Y axis, 0 = X.
        (Default is 1)
    badmask: str or 2d-array, optional
        Used to mask out bad pixels in generation of flat, requires pre-existing badpixel mask

    Returns
    -------
    flat : 2-d array
        The median combined master flat
    """
    # read the bias in, BUT we don't know if it's the numpy array or file name
    if isinstance(bias, str):
        # read in file if a string
        bias_im = fits.open(bias)[0].data
    else:
        # assume is proper array from biascombine function
        bias_im = bias

    # check for a badpixel mask,  True (1) for bad pixels False (0) for good pixels
    if badmask is not None:
        if isinstance(badmask, str):
        # read in file if a string
            badpix = fits.open(badmask)[0].data
        else:
            # assume it is proper array
            badpix = badmask

    # assume flatlist is a simple text file with image names
    # e.g. ls flat.00*b.fits > bflat.lis
    files = np.genfromtxt(flatlist,dtype=np.str)

    for i in range(0,len(files)):
        hdu_i = fits.open(files[i])
        if trim is False:
            im_i = hdu_i[0].data - bias_im
        elif trim is True:
            datasec = hdu_i[0].header['DATASEC'][1:-1].replace(':',',').split(',')
            d = list(map(int, datasec))
            im_i = hdu_i[0].data[d[2]-1:d[3],d[0]-1:d[1]] - bias_im

        if badmask is not None:
            im_i[badpix.astype('bool')] = np.nan   #use NaN to mask out bad pixels, need to make sure input mask is the right shape, trimmed or not

        # check for bad regions (not illuminated) in the spatial direction
        ycomp = np.nansum(im_i,axis=Saxis) # compress to spatial axis only
        illum_thresh = 0.8 # value compressed data must reach to be used for flat normalization
        ok = np.where( (ycomp>= np.nanmedian(ycomp)*illum_thresh) )

        # assume a median scaling for each flat to account for possible different exposure times
        if (i==0):
            all_data = im_i / np.nanmedian(im_i[ok,:])
        elif (i>0):
            all_data = np.dstack( (all_data, im_i / np.nanmedian(im_i[ok,:])) )
        hdu_i.close(closed=True)

    # do median across whole stack of flat images
    flat_stack = np.nanmedian(all_data, axis=2)
    #pdb.set_trace()
    # define the wavelength axis
    Waxis = 0
    # add a switch in case the spatial/wavelength axis is swapped
    if Saxis is 0:
        Waxis = 1

    if response is True:
        xdata = np.arange(all_data.shape[1]) # x pixels

        # median along spatial axis, smooth w/ 5pixel boxcar, take log of flux; NaNs ignored in convolve
        flat_1d = np.log10(convolve(np.nanmedian(flat_stack,axis=Waxis), Box1DKernel(5)))

        if mode=='spline':
            spl = UnivariateSpline(xdata, flat_1d, ext=0, k=2 ,s=0.001)
            flat_curve = 10.0**spl(xdata)
        elif mode=='poly':
            # fit log flux with polynomial
            flat_fit = np.polyfit(xdata, flat_1d, flat_poly)
            # get rid of log
            flat_curve = 10.0**np.polyval(flat_fit, xdata)

        if display is True:
            plt.figure()
            plt.plot(10.0**flat_1d)
            plt.plot(xdata, flat_curve,'r')
            plt.show()

        # divide median stacked flat by this RESPONSE curve
        flat = np.zeros_like(flat_stack)

        if Saxis is 1:
            for i in range(flat_stack.shape[Waxis]):
                flat[i,:] = flat_stack[i,:] / flat_curve
        else:
            for i in range(flat_stack.shape[Waxis]):
                flat[:,i] = flat_stack[:,i] / flat_curve
    else:
        flat = flat_stack

    if display is True:
        plt.figure()
        plt.imshow(flat, origin='lower',aspect='auto')
        plt.colorbar()
        plt.show()

    # write output to disk for later use; add fits entries for masks to allow loading flat and mask w/o running flatcombine again
    hduOut = fits.PrimaryHDU(flat)
    ilumfmask = fits.ImageHDU(ok[0],name="FlatMask")

    # place holder for actual bad pixel mask -- 1/True in mask is invalid data/pixel -- use to replace bad data with NaNs, need flat to already exist to match shape
    if badmask is None:
        badpix = np.zeros_like(flat)

    badout = fits.ImageHDU(badpix,name="BadMask")

    hduL = fits.HDUList([hduOut,ilumfmask,badout])    # use HDUList as container for fits image + masks
    hduL.writeto(output, overwrite=True)

    return flat ,ok[0], badpix


def ap_trace(img, fmask=(1,), nsteps=20, interac=False,
             recenter=False, prevtrace=(0,), bigbox=15,
             Saxis=1, display=False,mode='spline',degree=3):
    """
    Trace the spectrum aperture in an image

    Assumes wavelength axis is along the X, spatial axis along the Y.
    Chops image up in bins along the wavelength direction, fits a Gaussian
    within each bin to determine the spatial center of the trace. Finally,
    draws a cubic spline through the bins to up-sample the trace.

    Parameters
    ----------
    img : 2d numpy array
        This is the image, stored as a normal numpy array. Can be read in
        using astropy.io.fits like so:

        >>> hdu = fits.open('file.fits')  # doctest: +SKIP
        >>> img = hdu[0].data  # doctest: +SKIP

    nsteps : int, optional
        Keyword, number of bins in X direction to chop image into. Use
        fewer bins if ap_trace is having difficulty, such as with faint
        targets (default is 50, minimum is 4)
    fmask : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine.
    interac : bool, optional
        Set to True to have user click on the y-coord peak. (Default is
        False)
    recenter : bool, optional
        Set to True to use previous trace, but allow small shift in
        position. Currently only allows linear shift (Default is False)
    bigbox : float, optional
        The number of sigma away from the main aperture to allow to trace
    display : bool, optional
        If set to true display the trace over-plotted on the image
    Saxis : int, optional
        Set which axis the spatial dimension is along. 1 = Y axis, 0 = X.
        (Default is 1)

    Returns
    -------
    my : array
        The spatial (Y) positions of the trace, interpolated over the
        entire wavelength (X) axis
    """

    # define the wavelength axis
    Waxis = 0
    # add a switch in case the spatial/wavelength axis is swapped
    if Saxis is 0:
        Waxis = 1

    print('Tracing Aperture using nsteps='+str(nsteps))
    # the valid y-range of the chip
    if (len(fmask)>1):
        ydata = np.arange(img.shape[Waxis])[fmask]
    else:
        ydata = np.arange(img.shape[Waxis])

    # need at least 4 samples along the trace. sometimes can get away with very few
    if (nsteps<4):
        nsteps = 4

    # median smooth to crudely remove cosmic rays
    img_sm = scipy.signal.medfilt2d(img, kernel_size=(5,5))

    #--- Pick the strongest source, good if only 1 obj on slit
    ztot = np.nansum(img_sm,axis=Saxis)[ydata]
    yi = np.arange(img.shape[Waxis])[ydata]
    peak_y = yi[np.nanargmax(ztot)]
    peak_guess = [np.nanmax(ztot), np.nanmedian(ztot), peak_y, 2.]

    #-- allow interactive mode, if mult obj on slit
    if interac is True and recenter is False:
        class InteracTrace(object):
            def __init__(self):
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111)
                self.ax.plot(yi, ztot)
                plt.ylabel('Counts (Image summed in X direction)')
                plt.xlabel('Y Pixel')
                plt.title('Click on object!')

                self.cursor = Cursor(self.ax, useblit=False, horizOn=False,
                                     color='red', linewidth=1 )
                self.connect = self.fig.canvas.mpl_connect
                self.disconnect = self.fig.canvas.mpl_disconnect
                self.ClickID = self.connect('button_press_event', self.__onclick__)

                return

            def __onclick__(self,click):
                if self.fig.canvas.manager.toolbar._active is None:
                    self.xpoint = click.xdata
                    self.ypoint = click.ydata
                    self.disconnect(self.ClickID) # disconnect from event
                    self.cursor.disconnect_events()
                    self.cursor._update()
                    plt.close() # close window when clicked
                    return self.xpoint, self.ypoint
                else:
                    pass
    
        if matplotlib.is_interactive():
            print("Momentarily turning off matplotlib interactive mode...")
            plt.ioff()
            mplwasinteract = True
        else:
            mplwasinteract = False

        theclick = InteracTrace()  #interactive bits don't work properly as is withn matplotlib interactive is on
        plt.show()

        if mplwasinteract:
            plt.ion()

        xcl = theclick.xpoint
        # ycl = theclick.ypoint

        peak_guess[2] = xcl

    #-- use middle of previous trace as starting guess
    if (recenter is True) and (len(prevtrace)>10):
        peak_guess[2] = np.nanmedian(prevtrace)

    #-- fit a Gaussian to peak
    popt_tot, pcov = curve_fit(_gaus, yi[np.isfinite(ztot)], ztot[np.isfinite(ztot)], p0=peak_guess)
    #-- only allow data within a box around this peak
    ydata2 = ydata[np.where((ydata>=popt_tot[2] - popt_tot[3]*bigbox) &
                            (ydata<=popt_tot[2] + popt_tot[3]*bigbox))]

    yi = np.arange(img.shape[Waxis])[ydata2]
    # define the X-bin edges
    xbins = np.linspace(0, img.shape[Saxis], nsteps).astype('int')
    ybins = np.zeros_like(xbins)

    for i in range(0,len(xbins)-1):
        #-- fit gaussian w/i each window
        if Saxis is 1:
            zi = np.nanmean(img_sm[ydata2, xbins[i]:xbins[i+1]],axis=Saxis)
        else:
            zi = np.nanmean(img_sm[xbins[i]:xbins[i+1], ydata2],axis=Saxis)

        pguess = [np.nanmax(zi), np.nanmedian(zi), yi[np.nanargmax(zi)], 2.]
        popt,pcov = curve_fit(_gaus, yi[np.isfinite(zi)], zi[np.isfinite(zi)], p0=pguess)

        # if gaussian fits off chip, then use chip-integrated answer
        if (popt[2] <= min(ydata)+25) or (popt[2] >= max(ydata)-25):
            ybins[i] = popt_tot[2]
            popt = popt_tot
        else:
            ybins[i] = popt[2]

        # update the box it can search over, in case a big bend in the order
        # ydata2 = ydata[np.where((ydata>= popt[2] - popt[3]*bigbox) &
        #                         (ydata<= popt[2] + popt[3]*bigbox))]

    # recenter the bin positions, trim the unused bin off in Y
    mxbins = (xbins[:-1]+xbins[1:]) / 2.
    mybins = ybins[:-1]

    mx = np.arange(0, img.shape[Saxis])

    if mode=='spline':
        # run a cubic spline thru the bins
        ap_spl = UnivariateSpline(mxbins, mybins, ext=0, k=3, s=0)
        # interpolate the spline to 1 position per column
        my = ap_spl(mx)
    elif mode=='poly':
        trcoeffs = np.polyfit(mxbins,mybins,degree)
        my = np.polyval(trcoeffs,mx)


    if display is True:
        plt.figure()
        plt.imshow(np.log10(img),origin='lower',aspect='auto',cmap=cm.Greys_r)
        plt.autoscale(False)
        plt.plot(mx,my,'b',lw=1)
        # plt.plot(mx,my+popt_tot[3]*bigbox,'y')
        # plt.plot(mx,my-popt_tot[3]*bigbox,'y')
        plt.show()

    print("> Trace gaussian width = "+str(popt_tot[3])+' pixels')
    return my


def line_trace(img, pcent, wcent, fmask=(1,), maxbend=10, display=False,rowcenter=None):
    '''
    Trace the lines of constant wavelength along the spatial dimension.

    To be run after peaks found in the HeNeAr lamp. Usually run internally
    to HeNeAr_fit()

    Method works by tracing up and down from the image center (slice) along
    each HeNeAr line by 1 pixel, fitting a gaussian to find the center.

    Parameters
    ----------
    img : 2d float
        the HeNeAr data
    pcent : float array
        the pixel center along the image slice of each HeNeAr line to trace
    wcent : float array
        the identified wavelength that corresponds to each peak's pixel center (pcent)
    fmask : float array, optional
        the illumination section to trace trace over in spatial dimension
    maxbend : int, optional
        How big of a width (in pixel units) to allow the bend in the HeNeAr
        line to search over (Default is 10). Probably doesn't need to be
        modified much.
    display : bool, optional
        should we display plot after? (Default is False)

    Returns
    -------
    xcent, ycent, wcent
    These are the arrays of X pixel (wavelength dimension), Y pixel
    (spatial dimension), and corresponding wavelengths of each HeNeAr line.
    '''
    xcent_big = []
    ycent_big = []
    wcent_big = []

    # the valid y-range of the chip
    if (len(fmask)>1):
        ydata = np.arange(img.shape[0])[fmask]
    else:
        ydata = np.arange(img.shape[0])

    ybuf = 10
    # split the chip in to 2 parts, above and below the center
    
    if rowcenter is None:
        ydata1 = ydata[np.where((ydata>=img.shape[0]/2) &
                            (ydata<img.shape[0]-ybuf))]
        ydata2 = ydata[np.where((ydata<img.shape[0]/2) &
                            (ydata>ybuf))][::-1]
    else:
        ydata1 = ydata[np.where((ydata>=rowcenter) &
                                (ydata<img.shape[0]-ybuf))]
        ydata2 = ydata[np.where((ydata<rowcenter) &
                                (ydata>ybuf))][::-1]


    # plt.figure()
    # plt.plot(img[img.shape[0]/2,:])
    # plt.scatter(pcent, pcent*0.+np.mean(img))
    # plt.show()

    img_med = np.nanmedian(img)
    # loop over every HeNeAr peak that had a good fit

    for i in range(len(pcent)):
        xline = np.arange(int(pcent[i])-maxbend,int(pcent[i])+maxbend)

        # above center line (where fit was done)
        for j in ydata1:
            yline = img[j-ybuf:j+ybuf, int(pcent[i])-maxbend:int(pcent[i])+maxbend].sum(axis=0)
            # fit gaussian, assume center at 0, width of 2
            if j==ydata1[0]:
                cguess = pcent[i] # xline[np.argmax(yline)]

            pguess = [np.nanmax(yline), img_med, cguess, 2.]
            try:
                popt,pcov = curve_fit(_gaus, xline, yline, p0=pguess)

                if popt[2]>0 and popt[2]<img.shape[1]:
                    cguess = popt[2] # update center pixel

                    xcent_big = np.append(xcent_big, popt[2])
                    ycent_big = np.append(ycent_big, j)
                    wcent_big = np.append(wcent_big, wcent[i])
            except RuntimeError:
                popt = pguess

        # below center line, from middle down
        for j in ydata2:
            yline = img[j-ybuf:j+ybuf, int(pcent[i])-maxbend:int(pcent[i])+maxbend].sum(axis=0)
            # fit gaussian, assume center at 0, width of 2
            if j==ydata2[0]:
                cguess = pcent[i] # xline[np.argmax(yline)]

            pguess = [np.nanmax(yline), img_med, cguess, 2.]
            try:
                popt,pcov = curve_fit(_gaus, xline, yline, p0=pguess)

                if popt[2]>0 and popt[2]<img.shape[1]:
                    cguess = popt[2] # update center pixel

                    xcent_big = np.append(xcent_big, popt[2])
                    ycent_big = np.append(ycent_big, j)
                    wcent_big = np.append(wcent_big, wcent[i])
            except RuntimeError:
                popt = pguess


    if display is True:
        plt.figure()
        plt.imshow(np.log10(img), origin = 'lower',aspect='auto',cmap=cm.Greys_r)
        plt.colorbar()
        plt.scatter(xcent_big,ycent_big,marker='|',c='r')
        plt.show()

    return xcent_big, ycent_big, wcent_big


def find_peaks(wave, flux, pwidth=10, pthreshold=97, minsep=1):
    '''
    Given a slice thru a HeNeAr image, find the significant peaks.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Wavelength
    flux : `~numpy.ndarray`
        Flux
    pwidth : float
        the number of pixels around the "peak" to fit over
    pthreshold : float
        Peak threshold
    minsep : float
        Minimum separation

    Returns
    -------
    Peak Pixels, Peak Wavelengths
    '''
    # sort data, cut top x% of flux data as peak threshold
    flux_thresh = np.percentile(flux, pthreshold)

    # find flux above threshold
    high = np.where((flux >= flux_thresh))

    # find  individual peaks (separated by > 1 pixel)
    pk = high[0][1:][ ( (high[0][1:]-high[0][:-1]) > minsep ) ]

    # offset from start/end of array by at least same # of pixels
    pk = pk[pk > pwidth]
    pk = pk[pk < (len(flux) - pwidth)]

    # print('Found '+str(len(pk))+' peaks in HeNeAr to fit Gaussians to')

    pcent_pix = np.zeros_like(pk,dtype='float')
    wcent_pix = np.zeros_like(pk,dtype='float') # wtemp[pk]
    # for each peak, fit a gaussian to find center
    for i in range(len(pk)):
        xi = wave[pk[i] - pwidth:pk[i] + pwidth]
        yi = flux[pk[i] - pwidth:pk[i] + pwidth]

        pguess = (np.nanmax(yi), np.nanmedian(flux), float(np.nanargmax(yi)), 2.)
        try:
            popt,pcov = curve_fit(_gaus, np.arange(len(xi),dtype='float'), yi,
                                  p0=pguess)

            # the gaussian center of the line in pixel units
            pcent_pix[i] = (pk[i]-pwidth) + popt[2]
            # and the peak in wavelength units
            wcent_pix[i] = xi[np.nanargmax(yi)]

        except RuntimeError:
            pcent_pix[i] = float('nan')
            wcent_pix[i] = float('nan')

    wcent_pix, ss = np.unique(wcent_pix, return_index=True)
    pcent_pix = pcent_pix[ss]
    okcent = np.where((np.isfinite(pcent_pix)))
    return pcent_pix[okcent], wcent_pix[okcent]


def lines_to_surface(img, xcent, ycent, wcent,
                     mode='spline2d', fit_order=2, display=False):
    '''
    Turn traced arc lines into a wavelength solution across the entire chip

    Requires inputs from line_trace(). Outputs are a 2d wavelength solution

    Parameters
    ----------

    img : 2d array
        the HeNeAr data
    xcent : 1d array
        the X (spatial) pixel positions of the HeNeAr lines
    ycent : 1d array
        the Y (wavelength) pixel positions of the HeNeAr lines
    wcent : 1d array
        the wavelength values of the HeNeAr lines
    mode : str, {'poly', 'spline', 'spline2d'}
        what mode of interpolation to use to go from traces along the
        HeNeAr lines to a wavelength value for every (x,y) pixel?
        Options include (1) poly: along 1-pixel wide slices in the spatial
        dimension, fit a polynomial between the HeNeAr lines. Uses fit_order;
        (2) spline: along 1-pixel wide slices in the spatial dimension,
        fit a quadratic spline; (3) spline2d: fit a full 2d surface using a
        cubic spline. This is the best option, in principle.

    Returns
    -------
    the 2d wavelenth solution. Output depends on mode parameter.
    '''

    xsz = img.shape[1]

    #  fit the wavelength solution for the entire chip w/ a 2d spline
    if (mode=='spline2d'):
        xfitd = 5 # the spline dimension in the wavelength space
        print('Fitting Spline2d - NOTE: this mode doesnt work well')
        wfit = SmoothBivariateSpline(xcent, ycent, wcent, kx=xfitd, ky=3,
                                     bbox=[0,img.shape[1],0,img.shape[0]], s=0)

    #elif mode=='poly2d':
    ## using 2d polyfit
        # wfit = polyfit2d(xcent_big, ycent_big, wcent_big, order=3)

    elif mode=='spline':
        wfit = np.zeros_like(img)
        xpix = np.arange(xsz)

        for i in np.arange(ycent.min(), ycent.max()):
            x = np.where((ycent == i))

            x_u, ind_u = np.unique(xcent[x], return_index=True)

            # this smoothing parameter is absurd...
            spl = UnivariateSpline(x_u, wcent[x][ind_u], ext=0, k=3, s=5e7)

            if display is True:
                plt.figure()
                plt.scatter(xcent[x][ind_u], wcent[x][ind_u])
                plt.plot(xpix, spl(xpix))
                plt.show()

            wfit[int(i),:] = spl(xpix)

    elif mode=='poly':
        wfit = np.zeros_like(img)
        xpix = np.arange(xsz)

        for i in np.arange(ycent.min(), ycent.max()):
            x = np.where((ycent == i))
            coeff = np.polyfit(xcent[x], wcent[x], fit_order)
            wfit[int(i),:] = np.polyval(coeff, xpix)
    return wfit


def ap_extract(img, trace, apwidth=8, skysep=3, skywidth=7, skydeg=0
               ,gain=1.68,rdnoise=4.9,optimal=True,rectified=False):
    """
    1. Extract the spectrum using the trace. Simply add up all the flux
    around the aperture within a specified +/- width.

    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An major simplification at present. To be changed!

    2. Fits a polynomial to the sky at each column

    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An important simplification.

    3. Computes the uncertainty in each pixel

    Parameters
    ----------
    img : 2d numpy array
        This is the image, stored as a normal numpy array. Should be bias and flat corrected image, not divided by exposure time. Can be read in
        using astropy.io.fits like so:

        >>> hdu = fits.open('file.fits') # doctest: +SKIP
        >>> img = hdu[0].data # doctest: +SKIP

    trace : 1-d array
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from ap_trace
    apwidth : int, optional
        The width along the Y axis on either side of the trace to extract.
        Note: a fixed width is used along the whole trace.
        (default is 8 pixels)
    skysep : int, optional
        The separation in pixels from the aperture to the sky window.
        (Default is 3)
    skywidth : int, optional
        The width in pixels of the sky windows on either side of the
        aperture. (Default is 7)
    skydeg : int, optional
        The polynomial order to fit between the sky windows.
        (Default is 0)
    rectified: boolean, optional
        Used to change behavior of trace. If false, the aperature is extracted at each column centered on peak defined by trace. 
        If True, the median of the trace is used as center of window with given aperture sizes -- this is reasonable for traces with only slight bend ~ a couple pixes across the dector. Use of rectified helps with optimal extraction to prevent jumps in the estimated profile and weights when trace peak jumps by full pixel value.
    optimal: boolean, optional
        Use to allow optimal extraction for variance weighted aperature summation in spectral extraction, following Horne 1986. If false the spectrum is simply the summed flux in the aperture.

    Returns
    -------
    sumspec0 : 1-d array
        The summed flux at each column about the trace after sky subtraction
    stdspec0 : 1-d array
        1d error spectrum of extracted aperture
    skyspec0 : 1-d array
        Estimate of the sky spectrum along the trace
    """

    #Follow Horne 1986 for optimal extraction

    if rdnoise <= 0:
        rdnoise = 1
        print("Enforcing minimum readnoise of 1 electron")

    #note that if trace is of integer type then so will sumspec and other output which will give wrong results
    sumspec0 = np.zeros(len(trace))
    varspec0 = np.zeros(len(trace))
    skyspec0 = np.zeros(len(trace))

    if optimal:
        imgstrip = np.zeros((2*apwidth + 1, len(trace)))
        varstrip = np.zeros((2*apwidth + 1, len(trace)))
        skyvar =   np.zeros((2*apwidth + 1, len(trace))) #to keep track of sky contribution in variance to pass to optimal extraction

    if rectified:
        itrace = np.repeat(np.round(np.median(trace)),len(trace)).astype('int')
    else:
        itrace = np.round(trace).astype('int')

    varimg = (rdnoise/gain)**2 + np.abs(img.copy())/gain  # for the variance img; in data units
    skyimg = np.zeros_like(img)   #holds the 2d sky estimate

    # first determine the sky image, and aperature summed spectra
    for i in range(0,len(itrace)):
        #-- first do the aperture flux
        # juuuust in case the trace gets too close to the edge; what about the sky aperture in this case??
        widthup = apwidth
        widthdn = apwidth
        if (itrace[i]+widthup > img.shape[0]):
            widthup = img.shape[0]-itrace[i] - 1
        if (itrace[i]-widthdn < 0):
            widthdn = itrace[i] - 1

        #y defines the sky pixels
        y = np.append(np.arange(itrace[i]-apwidth-skysep-skywidth, itrace[i]-apwidth-skysep),
              np.arange(itrace[i]+apwidth+skysep+1, itrace[i]+apwidth+skysep+skywidth+1))

        z = img[y,i]
        if (skydeg>0):
            
            if len(np.where(np.isfinite(z))[0]) > skydeg + 1:
                # based on fitting with outlier removal from astropy --- API subject to change...
                polymod = models.Polynomial1D(skydeg)
                sigmafitting = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),sigma_clip,niter=4,sigma=2.5)
                #defaults to 3 iterations, 3 sigma clipping; maybe makes these optional keywords to ap_extract
                
                # fit a polynomial to the sky in this column
                skyfit, polyout = sigmafitting(polymod,y,z)  #can also add weights to this fitting, leaving it off for now
                
                ap = np.arange(itrace[i]-apwidth, itrace[i]+apwidth+1) # define the aperture in this column
                skyimg[itrace[i]-widthdn:itrace[i]+widthup+1,i] = polyout(ap) # evaluate the polynomial across the aperture for sky estimate
                
                #add contribution to variance image from background subtraction, estimate for fitted background variance based on optimal extraction documentation in iraf -- approximate
                varimg[itrace[i]-widthdn:itrace[i]+widthup+1,i] = varimg[itrace[i]-widthdn:itrace[i]+widthup+1,i] + polyout(ap)/(gain * (len(z)-1))
            
                if optimal:
                    skyvar[:,i] = polyout(ap)/(gain * (len(z)-1))
            else:
                skyimg[itrace[i]-widthdn:itrace[i]+widthup+1,i] = np.nan
                print("Warning: empty data array in column {}, skipping sky subtraction here".format(i))
                if optimal:
                    skyvar[:,i] = np.nan
    
        elif (skydeg==0):
            skyimg[itrace[i]-widthdn:itrace[i]+widthup+1,i] = np.nanmean(z)
            # more sky pixels should yield smaller sky error... # the gain factors are to follow poisson statistics then convert back to data units
            varimg[itrace[i]-widthdn:itrace[i]+widthup+1,i] = varimg[itrace[i]-widthdn:itrace[i]+widthup+1,i] + np.var(gain*z)/(gain**2 * len(z))

            if optimal:
                skyvar[:,i] = np.var(gain*z)/(gain**2 * len(z))

        elif (skydeg<0):
            skyimg[itrace[i]-widthdn:itrace[i]+widthup+1,i] = 0
            print("No background subtraction is being applied")

        #resulting variance img within aperature is variance estimate for Flux - Background; depart from Horne 1986 to include background subtraction in variance estimate for summed 1d spectrum

        sumspec0[i] = (img[itrace[i]-widthdn:itrace[i]+widthup+1,i] - skyimg[itrace[i]-widthdn:itrace[i]+widthup+1,i]).sum()
        varspec0[i] = varimg[itrace[i]-widthdn:itrace[i]+widthup+1,i].sum()
        skyspec0[i] = np.median(skyimg[itrace[i]-widthdn:itrace[i]+widthup+1,i])*(widthup+widthdn + 1)  # just to get idea for what the sky background is

        if optimal:
            imgstrip[:,i] = img[itrace[i]-widthdn:itrace[i]+widthup+1,i] - skyimg[itrace[i]-widthdn:itrace[i]+widthup+1,i]
            varstrip[:,i] = varimg[itrace[i]-widthdn:itrace[i]+widthup+1,i]

    if optimal:
        # should have keywords passed from call to ap_extract
        prof2d,spec1d,errspec = OptExtProfile(sumspec0,imgstrip,varstrip,skyvar,rdnoise,gain)
        out = (spec1d,errspec,skyspec0)
    else:
        out = (sumspec0,np.sqrt(varspec0),skyspec0)
    
    return out





def OptExtProfile(spec1d,img0,varimg0,skyvar,rdnoise,gain,degree=3,display=False):
    """
    1. Determine the spatial profile for each column to weight the aperture summation to produce 1D spectrum following optimal extraction of Horne 1986
    
    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An important simplification.
    
    2. Profile weighting is determined for each row with low order polynomial fit

    3. Computes the uncertainty in each pixel
    
    Parameters
    ----------
    spec1d: 1d numpy array
    This is the initial estimate of the object spectrum as a simple summation of counts within the aperature, in data units
    
    img: 2d numpy array
    This is the image of the data stored as a normal numpy array. Only pass the aperature portion of the data, is background subtracted, in data units.
    
    varimg : 2d numpy array
    This is the image of the variance of the data stored as a normal numpy array. Only pass the aperature portion of the data, includes contrbitution from background estimate, in data units
    
    skyvar, 2d numpy array
    This is an image, same shape as 'img' of the contribution to the variance image from the estimates of the background
    
    degree: scalar integer >= 0
    Defines the polynomial degree of the dispersion axis row by row fit to determine profile as function of column pixel, should be at least 3 for decent results, especially if trace is slightly curved
    
    display: boolean
    Default is False, set to true to show 2d image of the profile
    
    Returns
    -------
    prof : 2-d numpy array
    The resulting profile used to determine the weights along the spatial direction for the summation
    spectrum : 1-d numpy array
    1d output spectrum from optimal extraction
    err : 1-d numpy array
    1d ouput error spectrum
    """
    
    print("Using Optimal Extraction")
    
    img = img0.copy()
    varimg = varimg0.copy()

    nr,nc= img.shape  #maybe have an axis keyword if the dispersion axis is in other direction, for now assume rows are spatial
    spec = spec1d.copy()
    
    pix = np.arange(nc)
    prof = np.zeros((nr,nc) ) # initialize variable with profile
    
    oldmask = np.zeros_like(img).astype('bool')
    l = 0
    iteratecrit = True
    while iteratecrit:
        spec_profile = img / np.tile(spec,(nr,1)) # divide spectra by initial estimate to normalize out the spectral features
        var_profile = varimg / np.tile(np.power(spec,2),(nr,1))
        
        # loop over each row
        for i in range(nr):
            #median filter of spectral profile as initial smoothing -- replaced with sigma clipping for profile
            #temprow = scipy.signal.medfilt(spec_profile[i,:],(9,))

            temprow = spec_profile[i,:]
            nanmask = np.isfinite(temprow)

            polymod = models.Polynomial1D(degree)
            sigmafit = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),sigma_clip,niter=3,sigma=3)
            
            _ , polyout = sigmafit(polymod,pix[nanmask],temprow[nanmask],weights=1/var_profile[i,nanmask])  #uses weights
            
            prof[i,:] = polyout(pix)

            bottom = (prof[i,:] < 0).nonzero()[0]
            prof[i,bottom] = 0.  # enforce positivity
        
        prof = prof/ np.tile(np.sum(prof,axis=0),(nr,1)) # normalizes the weights
    
        #update variance image:
        varimg = (rdnoise/gain)**2 + (spec*prof/gain)  + skyvar
        badmask = np.power(img - np.tile(spec,(nr,1))*prof,2)  > 25*varimg   # 5 sigma overall clipping mask

        badmask = badmask | oldmask

        prof[badmask] = 0  #omit
        spec = np.sum(img*prof/varimg,axis=0) / np.sum(np.power(prof,2)/varimg,axis=0)   #this is the variance weighted summation 'optimal'
        variance =  np.sum(prof,axis=0) / (np.sum(np.power(prof,2)/varimg,axis=0))
        l += 1
        #print(l)
        if (l > 0) & np.all(badmask == oldmask):
            iteratecrit = False
        oldmask = badmask.copy()

    if display:
        fig, ax = plt.subplots(2,1)
        profmap = ax[0].imshow(prof)
        plt.colorbar(profmap,orientation='horizontal',ax=ax[0])
        ax[1].plot(np.median(prof,axis=1))
        ax[1].set_title("Median Profile")

    return prof,spec,np.sqrt(variance)





def HeNeAr_fit(calimage, linelist='apohenear.dat', interac=True,
               trim=True, fmask=(1,), display=False,outputlines=None,
               tol=10, fit_order=2, previous='',mode='poly',rowcenter=None,
               second_pass=True):
    """
    Determine the wavelength solution to be used for the science images.
    Can be done either automatically (buyer beware) or manually. Both the
    manual and auto modes use a "slice" through the chip center to learn
    the wavelengths of specific HeNeAr lines. Emulates the IDENTIFY
    function in IRAF.

    If the automatic mode is selected (interac=False), program tries to
    first find significant peaks in the "slice", then uses a brute-force
    guess scheme based on the grating information in the header. While
    easy, your mileage may vary with this method.

    If the interactive mode is selected (interac=True), you click on
    features in the "slice" and identify their wavelengths.

    Parameters
    ----------
    calimage : str
        Path to the HeNeAr calibration image
    linelist : str, optional
        The linelist file to use in the resources/linelists/ directory.
        Only used in automatic mode. (Default is apohenear.dat)
    interac : bool, optional
        Should the HeNeAr identification be done interactively (manually)?
        (Default is True)
    trim : bool, optional
        Trim the image using the DATASEC keyword in the header, assuming
        has format of [0:1024,0:512] (Default is True)
    fmask : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine.
    display : bool, optional
    tol : int, optional
        When in automatic mode, the tolerance in pixel units between
        linelist entries and estimated wavelengths for the first few
        lines matched... use carefully. (Default is 10)
    mode : str, optional
        What type of function to use to fit the entire 2D wavelength
        solution? Options include (poly, spline2d). (Default is poly)
    fit_order : int, optional
        The polynomial order to use to interpolate between identified
        peaks in the HeNeAr (Default is 2)
    previous : string, optional
        name of file containing previously identified peaks. Still has to
        do the fitting.
    outputlines: string, optional
        for setting name and location of output identified linelist, 
        defaults to input calimage + '.lines' in working directory
    rowcenter: int, optional
        for determining where to do the initial slice for the central 
        wavelength solution fitting, defaults to detector shape/2

    Returns
    -------
    wfit : bivariate spline object or 2d polynomial
        The wavelength solution at every pixel. Output type depends on the
        mode keyword above (poly is recommended)
    """

    print('Running HeNeAr_fit function on file '+calimage)

    # set output filename for linelist
    if outputlines is None:
        outputlines = calimage+'.lines'
    else:
        pass

    # silence the polyfit warnings
    warnings.simplefilter('ignore', np.RankWarning)

    hdu = fits.open(calimage)  # hdu data is dtype uint16; need to enforce float to prevent bugs later down stream
    if trim is False:
        img = hdu[0].data.astype('float')
    if trim is True:
        datasec = hdu[0].header['DATASEC'][1:-1].replace(':',',').split(',')
        d = list(map(int, datasec))
        img = hdu[0].data[d[2]-1:d[3],d[0]-1:d[1]].astype('float')

    # this approach will be very DIS specific
    try:
        disp_approx = hdu[0].header['DISPDW']
        wcen_approx = hdu[0].header['DISPWC']

    except KeyError:
        disp_approx = 0.
        wcen_approx = 1.

    try:
        # the red chip wavelength is backwards (DIS specific)
        clr = hdu[0].header['DETECTOR']
        if (clr.lower() == 'red'):
            sign = -1.0
        else:
            sign = 1.0
    except KeyError:
        sign = 1.0

    hdu.close(closed=True)

    #-- this is how I *want* to do this. Need to header values later though...
    # img, _, _, wtemp = OpenImg(calimage, trim=trim)


    # take a slice thru the data (+/- 10 pixels) in center row of chip or use rowcenter to define middle for slice
    if rowcenter is None:
        slice = img[int(img.shape[0]/2-10):int(img.shape[0]/2+10),:].sum(axis=0)
    else:
        slice = img[int(rowcenter-10):int(rowcenter+10),:].sum(axis=0)

    # use the header info to do rough solution (linear guess)
    wtemp = (np.arange(len(slice))-len(slice)/2) * disp_approx * sign + wcen_approx


    ######   IDENTIFY   (auto and interac modes)
    # = = = = = = = = = = = = = = = =
    #-- automatic mode
    if (interac is False) and (len(previous)==0):
        print("Doing automatic wavelength calibration on HeNeAr.")
        print("Note, this is not very robust. Suggest you re-run with interac=True")
        # find the linelist of choice

        linelists_dir = os.path.dirname(os.path.realpath(__file__))+ '/resources/linelists/'
        # if (len(linelist)==0):
        #     linelist = os.path.join(linelists_dir, linelist)

        # import the linelist
        linewave = np.genfromtxt(os.path.join(linelists_dir, linelist), dtype='float',
                              skip_header=1,usecols=(0,),unpack=True)


        pcent_pix, wcent_pix = find_peaks(wtemp, slice, pwidth=10, pthreshold=97)

    #   loop thru each peak, from center outwards. a greedy solution
    #   find nearest list line. if not line within tolerance, then skip peak
        pcent = []
        wcent = []

        # find center-most lines, sort by dist from center pixels
        ss = np.argsort(np.abs(wcent_pix-wcen_approx))

        #coeff = [0.0, 0.0, disp_approx*sign, wcen_approx]
        coeff = np.append(np.zeros(fit_order-1),(disp_approx*sign, wcen_approx))

        for i in range(len(pcent_pix)):
            xx = pcent_pix-len(slice)/2
            #wcent_pix = coeff[3] + xx * coeff[2] + coeff[1] * (xx*xx) + coeff[0] * (xx*xx*xx)
            wcent_pix = np.polyval(coeff, xx)

            if display is True:
                plt.figure()
                plt.plot(wtemp, slice, 'b')
                plt.scatter(linewave,np.ones_like(linewave)*np.nanmax(slice),marker='o',c='cyan')
                plt.scatter(wcent_pix,np.ones_like(wcent_pix)*np.nanmax(slice)/2.,marker='*',c='green')
                plt.scatter(wcent_pix[ss[i]],np.nanmax(slice)/2., marker='o',c='orange')

            # if there is a match w/i the linear tolerance
            if (min((np.abs(wcent_pix[ss][i] - linewave))) < tol):
                # add corresponding pixel and *actual* wavelength to output vectors
                pcent = np.append(pcent,pcent_pix[ss[i]])
                wcent = np.append(wcent, linewave[np.argmin(np.abs(wcent_pix[ss[i]] - linewave))] )

                if display is True:
                    plt.scatter(wcent,np.ones_like(wcent)*np.nanmax(slice),marker='o',c='red')

                if (len(pcent)>fit_order):
                    coeff = np.polyfit(pcent-len(slice)/2, wcent, fit_order)

            if display is True:
                plt.xlim((min(wtemp),max(wtemp)))
                plt.show()

        lout = open(outputlines, 'w')
        lout.write("# This file contains the HeNeAr lines identified [auto] Columns: (pixel, wavelength) \n")
        for l in range(len(pcent)):
            lout.write(str(pcent[l]) + ', ' + str(wcent[l])+'\n')
        lout.close()

        # the end result is the vector "coeff" has the wavelength solution for "slice"
        # update the "wtemp" vector that goes with "slice" (fluxes)
        wtemp = np.polyval(coeff, (np.arange(len(slice))-len(slice)/2))


    # = = = = = = = = = = = = = = = =
    #-- manual (interactive) mode
    elif (interac is True):
        if (len(previous)==0):
            print('')
            print('Using INTERACTIVE HeNeAr_fit mode:')
            print('1) Click on HeNeAr lines in plot window')
            print('2) Enter corresponding wavelength in terminal and press <return>')
            print('   If mis-click or unsure, just press leave blank and press <return>')
            print('3) To delete an entry, click on label, enter "d" in terminal, press <return>')
            print('4) Close plot window when finished')

            xraw = np.arange(len(slice))
            class InteracWave(object):
                # http://stackoverflow.com/questions/21688420/callbacks-for-graphical-mouse-input-how-to-refresh-graphics-how-to-tell-matpl
                def __init__(self):
                    self.fig = plt.figure()
                    self.ax = self.fig.add_subplot(111)
                    self.ax.plot(wtemp, slice, 'b')
                    plt.xlabel('Wavelength')
                    plt.ylabel('Counts')

                    self.pcent = [] # the pixel centers of the identified lines
                    self.wcent = [] # the labeled wavelengths of the lines
                    self.ixlib = [] # library of click points

                    self.cursor = Cursor(self.ax, useblit=False,horizOn=False,
                                         color='red', linewidth=1 )
                    self.connect = self.fig.canvas.mpl_connect
                    self.disconnect = self.fig.canvas.mpl_disconnect
                    self.clickCid = self.connect("button_press_event",self.OnClick)

                def OnClick(self, event):
                    # only do stuff if toolbar not being used
                    # NOTE: this subject to change API, so if breaks, this probably why
                    # http://stackoverflow.com/questions/20711148/ignore-matplotlib-cursor-widget-when-toolbar-widget-selected
                    if self.fig.canvas.manager.toolbar._active is None:
                        ix = event.xdata

                        # if the click is in good space, proceed
                        if (ix is not None) and (ix > np.nanmin(slice)) and (ix < np.nanmax(slice)):
                            # disable button event connection
                            self.disconnect(self.clickCid)

                            # disconnect cursor, and remove from plot
                            self.cursor.disconnect_events()
                            self.cursor._update()

                            # get points nearby to the click
                            nearby = np.where((wtemp > ix-10*disp_approx) &
                                              (wtemp < ix+10*disp_approx) )

                            # find if click is too close to an existing click (overlap), doesn't remove previously plotted red circle
                            kill = None
                            if len(self.pcent)>0:
                                for k in range(len(self.pcent)):
                                    if np.abs(self.ixlib[k]-ix)<tol:
                                        kill_d = input('> WARNING: Click too close to existing point. To delete existing point, enter "d"\n')
                                        if kill_d=='d':
                                            kill = k
                                if kill is not None:
                                    del(self.pcent[kill])
                                    del(self.wcent[kill])
                                    del(self.ixlib[kill])


                            # If there are enough valid points to possibly fit a peak too...
                            if (len(nearby[0]) > 4) and (kill is None):
                                imax = np.nanargmax(slice[nearby])

                                pguess = (np.nanmax(slice[nearby]), np.nanmedian(slice), xraw[nearby][imax], 2.)
                                try:
                                    popt,pcov = curve_fit(_gaus, xraw[nearby], slice[nearby], p0=pguess)
                                    self.ax.plot(wtemp[int(popt[2])], popt[0], 'r|')
                                except ValueError:
                                    print('> WARNING: Bad data near this click, cannot centroid line with Gaussian. I suggest you skip this one')
                                    popt = pguess
                                except RuntimeError:
                                    print('> WARNING: Gaussian centroid on line could not converge. I suggest you skip this one')
                                    popt = pguess

                                # using raw_input sucks b/c doesn't raise terminal, but works for now
                                try:
                                    number=float(input('> Enter Wavelength: '))
                                    self.pcent.append(popt[2])
                                    self.wcent.append(number)
                                    self.ixlib.append((ix))
                                    self.ax.plot(wtemp[int(popt[2])], popt[0], 'ro')
                                    print('  Saving '+str(number))
                                except ValueError:
                                    print("> Warning: Not a valid wavelength float!")

                            elif (kill is None):
                                print('> Error: No valid data near click!')

                            # reconnect to cursor and button event
                            self.clickCid = self.connect("button_press_event",self.OnClick)
                            self.cursor = Cursor(self.ax, useblit=False,horizOn=False,
                                             color='red', linewidth=1 )
                    else:
                        pass

            #need to turn of interactive mode for InteracWave to proceed properely
            if matplotlib.is_interactive():
                print("Momentarily turning off matplotlib interactive mode...")
                plt.ioff()
                mplwasinteract = True
            else:
                mplwasinteract = False
            
            # run the interactive program
            # Note if other plot windows are open prior to beginning this InteracWave they will also need to be closed to proceed
            wavefit = InteracWave()
            plt.show() #activate the display - GO!
            
            if mplwasinteract:
                plt.ion()
            
            # how I would LIKE to do this interactively:
            # inside the interac mode, do a split panel, live-updated with
            # the wavelength solution, and where user can edit the fit_order

            # how I WILL do it instead
            # a crude while loop here, just to get things moving

            # after interactive fitting done, get results fit peaks
            pcent = np.array(wavefit.pcent,dtype='float')
            wcent = np.array(wavefit.wcent, dtype='float')
            #pdb.set_trace()
            print('> You have identified '+str(len(pcent))+' lines')
            lout = open(outputlines, 'w')
            lout.write("# This file contains the HeNeAr lines identified [manual] Columns: (pixel, wavelength) \n")
            for l in range(len(pcent)):
                lout.write(str(pcent[l]) + ', ' + str(wcent[l])+'\n')
            lout.close()


        if (len(previous)>0):
            pcent, wcent = np.genfromtxt(previous, dtype='float',
                                      unpack=True, skip_header=1,delimiter=',')


        #---  FIT SMOOTH FUNCTION ---

        # fit polynomial thru the peak wavelengths
        # xpix = (np.arange(len(slice))-len(slice)/2)
        # coeff = np.polyfit(pcent-len(slice)/2, wcent, fit_order)
        xpix = np.arange(len(slice))
        coeff = np.polyfit(pcent, wcent, fit_order)
        wtemp = np.polyval(coeff, xpix)

        done = str(fit_order)
        while (done != 'd'):
            fit_order = int(done)
            coeff = np.polyfit(pcent, wcent, fit_order)
            wtemp = np.polyval(coeff, xpix)

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(pcent, wcent, 'bo')
            ax1.plot(xpix, wtemp, 'r')

            ax2.plot(pcent, wcent - np.polyval(coeff, pcent),'ro')
            residual = np.std(wcent - np.polyval(coeff, pcent))
            ax2.set_xlabel('pixel')
            ax1.set_ylabel('wavelength')
            ax2.set_ylabel('residual')
            ax1.set_title('fit_order = '+str(fit_order))
            ax2.set_title("residual RMS = {0:6.4f}".format(residual))

            # ylabel('wavelength')

            print(" ")
            print('> How does this look?  Enter "d" to be done (accept), ')
            print('  or a number to change the polynomial order and re-fit')
            print('> Currently fit_order = '+str(fit_order))
            print(" ")

            plt.show()

            _CheckMono(wtemp)

            print(' ')
            done = str(input('ENTER: "d" (done) or a # (poly order): '))


    # = = = = = = = = = = = = = = = = = =
    # now rough wavelength is found, either via interactive or auto mode!

    #-- SECOND PASS
    if second_pass is True:
        linelists_dir = os.path.dirname(os.path.realpath(__file__))+ '/resources/linelists/'
        hireslinelist = 'henear.dat'
        linewave2 = np.genfromtxt(os.path.join(linelists_dir, hireslinelist), dtype='float',
                               skip_header=1, usecols=(0,), unpack=True)

        tol2 = tol # / 2.0

        pcent_pix2, wcent_pix2 = find_peaks(wtemp, slice, pwidth=10, pthreshold=80)

        pcent2 = []
        wcent2 = []
        # sort from center wavelength out
        ss = np.argsort(np.abs(wcent_pix2-wcen_approx))

        # coeff should already be set by manual or interac mode above
        # coeff = np.append(np.zeros(fit_order-1),(disp_approx*sign, wcen_approx))
        for i in range(len(pcent_pix2)):
            xx = pcent_pix2-len(slice)/2
            wcent_pix2 = np.polyval(coeff, xx)

            if (min((np.abs(wcent_pix2[ss][i] - linewave2))) < tol2):
                # add corresponding pixel and *actual* wavelength to output vectors
                pcent2 = np.append(pcent2, pcent_pix2[ss[i]])
                wcent2 = np.append(wcent2, linewave2[np.argmin(np.abs(wcent_pix2[ss[i]] - linewave2))] )

            #-- update in real time. maybe not good for 2nd pass
            # if (len(pcent2)>fit_order):
            #     coeff = np.polyfit(pcent2-len(slice)/2, wcent2, fit_order)

            if display is True:
                plt.figure()
                plt.plot(wtemp, slice, 'b')
                plt.scatter(linewave2,np.ones_like(linewave2)*np.nanmax(slice),
                            marker='o',c='cyan')
                plt.scatter(wcent_pix2,np.ones_like(wcent_pix2)*np.nanmax(slice)/2.,
                            marker='*',c='green')
                plt.scatter(wcent_pix2[ss[i]],np.nanmax(slice)/2.,
                            marker='o',c='orange')
                plt.text(np.nanmin(wcent_pix2), np.nanmax(slice)*0.95, hireslinelist)
                plt.text(np.nanmin(wcent_pix2), np.nanmax(slice)/2.*1.1, 'detected lines')

                plt.scatter(wcent2,np.ones_like(wcent2)*np.nanmax(slice)*1.05,marker='o',c='red')
                plt.text(np.nanmin(wcent_pix2), np.nanmax(slice)*1.1, 'matched lines')

                plt.ylim((np.nanmin(slice), np.nanmax(slice)*1.2))
                plt.xlim((min(wtemp),max(wtemp)))
                plt.show()
        wtemp = np.polyval(coeff, (np.arange(len(slice))-len(slice)/2))

        lout = open(outputlines+'2', 'w')
        lout.write("# This file contains the HeNeAr lines identified [2nd pass] Columns: (pixel, wavelength) \n")
        for l in range(len(pcent2)):
            lout.write(str(pcent2[l]) + ', ' + str(wcent2[l])+'\n')
        lout.close()

        xpix = np.arange(len(slice))
        coeff = np.polyfit(pcent2, wcent2, fit_order)
        wtemp = np.polyval(coeff, xpix)


        #---  FIT SMOOTH FUNCTION ---
        if interac is True:
            done = str(fit_order)
            while (done != 'd'):
                fit_order = int(done)
                coeff = np.polyfit(pcent2, wcent2, fit_order)
                wtemp = np.polyval(coeff, xpix)

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                ax1.plot(pcent2, wcent2, 'bo')
                ax1.plot(xpix, wtemp, 'r')

                ax2.plot(pcent2, wcent2 - np.polyval(coeff, pcent2),'ro')
                residual = np.std(wcent2 - np.polyval(coeff, pcent2))
                ax2.set_xlabel('pixel')
                ax1.set_ylabel('wavelength')
                ax2.set_ylabel('residual')
                ax1.set_title('2nd pass, fit_order = '+str(fit_order))
                ax2.set_title("residual RMS = {0:6.4f}".format(residual))
                # ylabel('wavelength')

                print(" ")
                print('> How does this look?  Enter "d" to be done (accept), ')
                print('  or a number to change the polynomial order and re-fit')
                print('> Currently fit_order = '+str(fit_order))
                print(" ")

                plt.show()

                _CheckMono(wtemp)

                print(' ')
                done = str(input('ENTER: "d" (done) or a # (poly order): '))

    #-- trace the peaks vertically --
    xcent_big, ycent_big, wcent_big = line_trace(img, pcent, wcent, rowcenter=rowcenter,
                                                 fmask=fmask, display=display)

    #-- turn these vertical traces in to a whole chip wavelength solution
    wfit = lines_to_surface(img, xcent_big, ycent_big, wcent_big,
                            mode=mode, fit_order=fit_order)

    return wfit


def mapwavelength(trace, wavemap, mode='spline2d'):
    """
    Compute the wavelength along the center of the trace, to be run after
    the HeNeAr_fit routine.

    Parameters
    ----------
    trace : 1-d array
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from ap_trace
    wavemap : bivariate spline object or image-like wavelength map
        The wavelength evaluated at every pixel, output from HeNeAr_fit
        Type depends on mode parameter.
    mode : str, optional
        Which mode was used to generate the 2D wavelength solution in
        HeNeAr_fit(), and specifically in lines_to_surface()?
        Options include: poly, spline, spline2d (Default is 'spline2d')

    Returns
    -------
    trace_wave : 1d array
        The wavelength vector evaluated at each position along the trace
    """
    # use the wavemap from the HeNeAr_fit routine to determine the wavelength along the trace
    if mode=='spline2d':
        trace_wave = wavemap.ev(np.arange(len(trace)), trace)

    elif mode=='poly' or mode=='spline':
        trace_wave = np.zeros_like(trace)
        for i in range(len(trace)):
            trace_wave[i] = np.interp(trace[i], range(wavemap.shape[0]), wavemap[:,i])

    ## using 2d polyfit
    # trace_wave = polyval2d(np.arange(len(trace)), trace, wavemap)
    return trace_wave


def normalize(wave, flux, mode='poly', order=5):
    '''
    Return a flattened, normalized spectrum. A model spectrum is made of
    the continuum by fitting either a polynomial or spline to the data,
    and then the data is normalized with the equation:

    >>> norm = (flux - model) / model  # doctest: +SKIP

    Parameters
    ----------
    wave : 1-d array
        The object's wavelength array
    flux : 1-d array
        The object's flux array
    mode : str, optional
        Decides which mode should be used to flatten the spectrum.
        Options are 'poly' (Default), 'spline', 'interac'.
    order : int, optional
        The polynomial order to use for mode='poly'. (Default is 3)

    Returns
    -------
    Flux normalized spectrum at same wavelength points as the input
    '''

    if (mode != 'interac') and (mode != 'spline') and (mode != 'poly'):
        mode = 'poly'
        print("WARNING: invalid mode set in normalize. Changing to 'poly'")

    if mode=='interac':
        print('interac mode not built yet. sorry...')
        mode = 'poly'

    if mode=='poly':
        fit = np.polyfit(wave, flux, order)
        model = np.polyval(fit, wave)

    if mode=='spline':
        spl = UnivariateSpline(wave, flux, ext=0, k=2 ,s=0.0025)
        model = spl(wave)

    norm = (flux - model) / model
    return norm


def AirmassCor(obj_wave, obj_flux, airmass, airmass_file='apoextinct.dat'):
    """
    Correct the spectrum based on the airmass

    Parameters
    ----------
    obj_wave : 1-d array
        The 1-d wavelength array of the spectrum
    obj_flux : 1-d or 2-d array
        The 1-d or 2-d flux array of the spectrum
    airmass : float
        The value of the airmass, not the header keyword.
    airmass_file : str, {'apoextinct.dat', 'ctioextinct.dat', 'kpnoextinct.dat', 'ormextinct.dat'}
        The name of the airmass extinction file. This routine assumes
        the file is stored in the resources/extinction/ subdirectory.
        Available files are (Default is apoextinct.dat)

    Returns
    -------
    The flux array
    """
    # read in the airmass extinction curve
    extinction_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'resources/extinction')
    if len(airmass_file)==0:
        air_wave, air_cor = np.genfromtxt(os.path.join(extinction_dir, airmass_file),
                                       unpack=True,ski_header=2)
    else:
        print('> Loading airmass library file: '+airmass_file)
        # print('  Note: first 2 rows are skipped, assuming header')
        air_wave, air_cor = np.genfromtxt(os.path.join(extinction_dir, airmass_file),
                                       unpack=True,skip_header=2)
    # air_cor in units of mag/airmass
    airmass_ext = 10.0**(0.4 * airmass *
                         np.interp(obj_wave, air_wave, air_cor))
    # arimas_ext is broadcast to obj_flux if it is a 2-d array
    return obj_flux * airmass_ext


def DefFluxCal(obj_wave, obj_flux, stdstar='', mode='spline', polydeg=9,
               display=False):
    """

    Parameters
    ----------
    obj_wave : 1-d array
        The 1-d wavelength array of the spectrum

    obj_flux : 1-d array
        The 1-d flux array of the spectrum

    stdstar : str
        Name of the standard star file to use for flux calibration. You
        must give the subdirectory and file name, for example:

        >>> sensfunc = DefFluxCal(wave, flux, mode='spline', stdstar='spec50cal/feige34.dat')  # doctest: +SKIP

        If no standard is set, or an invalid standard is selected, will
        return array of 1's and a warning. A list of all available
        subdirectories and objects is available on the wiki, or look in
        pydis/resources/onedstds/

    mode : str, optional
        either "linear", "spline", or "poly" (Default is spline)

    polydeg : float, optional
        set the order of the polynomial to fit through (Default is 9)

    display : bool, optional
        If True, plot the down-sampled sensfunc and fit to screen (Default
        is False)

    Returns
    -------
    sensfunc : 1-d array
        The sensitivity function for the standard star

    """
    stdstar2 = stdstar.lower()
    std_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'resources', 'onedstds')

    if os.path.isfile(os.path.join(std_dir, stdstar2)):
        std_wave, std_mag, std_wth = np.genfromtxt(os.path.join(std_dir, stdstar2),
                                                skip_header=1, unpack=True)
        # standard star spectrum is stored in magnitude units
        std_flux = _mag2flux(std_wave, std_mag)

        # Automatically exclude these obnoxious lines...
        balmer = np.array([6563, 4861, 4341], dtype='float')

        # down-sample (ds) the observed flux to the standard's bins
        obj_flux_ds = []
        obj_wave_ds = []
        std_flux_ds = []
        for i in range(len(std_wave)):
            rng = np.where((obj_wave >= std_wave[i] - std_wth[i] / 2.0) &
                           (obj_wave < std_wave[i] + std_wth[i] / 2.0))
            IsH = np.where((balmer >= std_wave[i] - std_wth[i] / 2.0) &
                           (balmer < std_wave[i] + std_wth[i] / 2.0))

            # does this bin contain observed spectra, and no Balmer line?
            if (len(rng[0]) > 1) and (len(IsH[0]) == 0):
                # obj_flux_ds.append(np.sum(obj_flux[rng]) / std_wth[i])
                obj_flux_ds.append( np.nanmean(obj_flux[rng]) )
                obj_wave_ds.append(std_wave[i])
                std_flux_ds.append(std_flux[i])


        # the ratio between the standard star flux and observed flux
        # has units like erg / counts
        ratio = np.abs(np.array(std_flux_ds, dtype='float') /
                       np.array(obj_flux_ds, dtype='float'))


        # interp calibration (sensfunc) on to object's wave grid
        # can use 3 types of interpolations: linear, cubic spline, polynomial

        # if invalid mode selected, make it spline
        if mode not in ('linear', 'spline', 'poly'):
            mode = 'spline'
            print("WARNING: invalid mode set in DefFluxCal. Changing to spline")

        # actually fit the log of this sensfunc ratio
        # since IRAF does the 2.5*log(ratio), everything in mag units!
        LogSensfunc = np.log10(ratio)

        # interpolate back on to observed wavelength grid
        if mode=='linear':
            sensfunc2 = np.interp(obj_wave, obj_wave_ds, LogSensfunc)
        elif mode=='spline':
            spl = UnivariateSpline(obj_wave_ds, LogSensfunc, ext=0, k=2 ,s=0.0025)
            sensfunc2 = spl(obj_wave)
        elif mode=='poly':
            fit = np.polyfit(obj_wave_ds, LogSensfunc, polydeg)
            sensfunc2 = np.polyval(fit, obj_wave)

        if display is True:
            plt.figure()
            plt.plot(std_wave, std_flux, 'r', alpha=0.5, label='standard flux')
            plt.xlabel('Wavelength')
            plt.ylabel('Standard Star Flux')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(obj_wave, obj_flux, 'k', label='observed counts')
            plt.plot(obj_wave_ds, obj_flux_ds, 'bo',
                    label='downsample observed')
            plt.xlabel('Wavelength')
            plt.ylabel('Observed Counts/S')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(obj_wave_ds, LogSensfunc, 'ko', label='sensfunc')
            plt.plot(obj_wave, sensfunc2, label='interpolated sensfunc')
            plt.xlabel('Wavelength')
            plt.ylabel('log Sensfunc')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(obj_wave, obj_flux*(10**sensfunc2),'k',
                        label='applied sensfunc')
            plt.plot(std_wave, std_flux, 'ro', alpha=0.5, label='standard flux')
            plt.xlabel('Wavelength')
            plt.ylabel('Standard Star Flux')
            plt.legend()
            plt.show()
    else:
        sensfunc2 = np.zeros_like(obj_wave)
        print('ERROR: in DefFluxCal no valid standard star file found at ')
        print(os.path.join(std_dir, stdstar2))

    return 10**sensfunc2


def ApplyFluxCal(obj_wave, obj_flux, obj_err, cal_wave, sensfunc):
    # the sensfunc should already be BASICALLY at the same wavelenths as the targets
    # BUT, just in case, we linearly resample it:

    # ensure input array is sorted!
    ss = np.argsort(cal_wave)

    sensfunc2 = np.interp(obj_wave, cal_wave[ss], sensfunc[ss])

    # then simply apply re-sampled sensfunc to target flux
    return obj_flux * sensfunc2, obj_err * sensfunc2
