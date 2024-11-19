from __future__ import print_function, division, absolute_import

# Std. lib imports
from sys import stdout
from math import pi

# Non-std. lib imports
from numpy import array, loadtxt
from input_reader import InputReader, SUPPRESS, ReaderError, \
                         range_check, abs_file_path

HZ2WAVENUM = 1 / ( 100 * 2.99792458E8 ) # Hz to cm^{-1} conversion

__all__ = ['read_input', 'ReaderError']


def read_input(input_file):
    '''Defines what to expect from the input file and then
    reads it in.'''

    # Creates an input reader instance
    reader = InputReader(default=SUPPRESS)

    # Rate parameter, either rate or lifetime, not both
    rate = reader.add_mutually_exclusive_group(required=True)
    # The units are s, ns, ps, or fs.  The default is ps.
    rate.add_line_key('lifetime', type=float,
                      glob={'len' : '?',
                            'type' : ('ps', 'fs', 'ns', 's'),
                            'default' : 'ps'})
    rate.add_line_key('rate', type=float,
                      glob={'len' : '?',
                            'type' : ('thz', 'phz', 'ghz', 'hz'),
                            'default' : 'thz'})

    # The range of the X-axis
    reader.add_line_key('xlim', type=[int, int], default=(1900, 2000))
    reader.add_boolean_key('reverse', action=True, default=False)

    # Read in the raw data.  
    reader.add_line_key('raw', type=[], glob={'len':'*', 'join':True, },
                               default=None, case=True)

    # Read in the peak data.  The wavenumber and height is required.
    # The Lorentzian and Gaussian widths are defaulted to 10 if not given.
    floatkw = {'type' : float, 'default' : 10.0}
    reader.add_line_key('peak', required=True, repeat=True, type=[float,float],
                                keywords={'g':floatkw, 'l':floatkw,
                                          'num' : {'type':int,'default':-1}})

    # Read the exchange information.
    reader.add_line_key('exchange', repeat=True, type=[int, int],
                                    glob={'type' : float,
                                          'default' : 1.0,
                                          'len' : '?'})
    reader.add_boolean_key('nosym', action=False, default=True,
                           dest='symmetric_exchange')

    # Actually read the input file
    args = reader.read_input(input_file)

    # Make sure the filename was given correctly and read in data
    if args.raw:
        args.add('rawName', args.raw)
        args.raw = loadtxt(abs_file_path(args.raw))

    # Make the output file path absolute if given
    args.data = abs_file_path(args.data) if 'data' in args else ''

    if 'save_plot_script' in args:
        args.save_plot_script = abs_file_path(args.save_plot_script)
    else:
        args.save_plot_script = ''

    # Adjust the input rate or lifetime to wavenumbers
    if 'lifetime' in args:
        convert = { 'ps' : 1E-12, 'ns' : 1E-9, 'fs' : 1E-15, 's' : 1 }
        args.add('k', 1 / ( convert[args.lifetime[1]] * args.lifetime[0] ))
    else:
        convert = { 'thz' : 1E12, 'ghz' : 1E9, 'phz' : 1E15, 'hz' : 1 }
        args.add('k', convert[args.rate[1]] * args.rate[0])
    args.k *= HZ2WAVENUM / ( 2 * pi )

    # Parse the vibrational input
    num, vib, Gamma_Lorentz, Gamma_Gauss, heights, rel_rates, num_given = (
                                                    [], [], [], [], [], [], [])
    for peak in args.peak:
        # Vibration #
        num.append(peak[2]['num'])
        num_given.append(False if peak[2]['num'] < 0 else True)
        # Angular frequency
        vib.append(peak[0])
        # Relative peak heights
        heights.append(peak[1])
        # Default Gaussian or Lorentzian width or relative rate
        Gamma_Lorentz.append(peak[2]['l'])
        Gamma_Gauss.append(peak[2]['g'])

    # Either all or none of the numbers must be given explicitly
    if not (all(num_given) or not any(num_given)):
        raise ReaderError('All or none of the peaks must '
                          'be given numbers explicitly')
    # If the numbers were give, make sure there are no duplicates
    if all(num_given):
        if len(num) != len(set(num)):
            raise ReaderError('Duplicate peaks cannot be given')
    # If none were given, number automatically
    else:
        num = range(1, len(num)+1, 1)

    args.add('num', array(num))
    args.add('vib', array(vib))
    args.add('heights', array(heights))
    args.add('Gamma_Lorentz', array(Gamma_Lorentz))
    args.add('Gamma_Gauss', array(Gamma_Gauss))

    # Set up the exchanges
    # Make sure the each exchange number appears in num.
    num = set(num)
    ex = []
    rates = []
    string = 'Requested peak {0} in exchange does not exist'
    if 'exchange' in args:
        for exchange in args.exchange:
            p1 = exchange[0]
            if p1 not in num:
                raise ReaderError(string.format(p1))
            p2 = exchange[1]
            if p2 not in num:
                raise ReaderError(string.format(p2))
            if p1 == p2 and args.symmetric_exchange:
                raise ReaderError('Self exchange is not allowed')
            rate = exchange[2]
            # Offset the peak number by one to match python indicies
            ex.append([p1-1, p2-1])
            rates.append(rate)
    else:
        ex = []
        rates = []
    args.add('exchanges', array(ex, dtype=int))
    args.add('exchange_rates', array(rates))

    # Make sure the xlimits are ascending
    try:
        range_check(args.xlim[0], args.xlim[1])
    except ValueError:
        raise ReaderError('In xrange, the low value must '
                          'less than the high value')

    return args



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 21:21:39 2024

@author: benjaminlear
"""

# Std. lib imports
from sys import exit
from math import sqrt, log, pi
import numpy as np

# Non-std lib imports
from scipy.linalg import eig, inv
from scipy.special import wofz
from numpy import array, argsort, diag, dot, eye, zeros

class SpectrumError(Exception):
    '''An exception for making the spectrum'''
    pass

# A few constants
SQRT2LOG2_2 = sqrt(2 * log(2)) * 2
INVSQRT2LOG2_2 = 1 / SQRT2LOG2_2
SQRT2 = sqrt(2)
SQRT2PI = sqrt(2 * pi)

def clip(xy, xlimits):
    '''Clips according to x-values.'''
    xy = xy[np.where(xy[:,0] > xlimits[0])]
    return xy[np.where(xy[:,0] < xlimits[1])]


def normalize(y):
    '''Normalizes a data set.  Clips according to x-values if given.''' 
    # Set baseline to zero.
    y = y - y.min()
    # Normalize
    return y / y.max()



def plot(args, x, y):
    '''Plots the normalized data.'''
    try:
        from matplotlib.pyplot import rc, plot, xlim, ylim, \
                                      xlabel, ylabel, show
    except ImportError:
        #print('It appears that you are missing matplotlib.', file=stderr)
        #print('You should install it using your favorite method', file=stderr)
        return 1
    
    # 14 point font size
    rc('font', **{'size': 14})

    # Plot the data and set the data window
    if args.raw is not None:
        plot(x, y, 'b-', args.raw[:,0], args.raw[:,1], 'g-', lw=1.5)
    else:
        plot(x, y, 'b-', lw=1.5)
    if args.reverse:
        xlim(args.xlim[1], args.xlim[0])
    else:
        xlim(args.xlim[0], args.xlim[1])
    ylim(-0.05, 1.1)
    xlabel('Wavenumbers')
    ylabel('Intensity (Normalized)')
    show()
    return 0


def ZMat(npeaks, peak_exchanges, relative_rates, symmetric):
    '''Construct the Z matrix.  Symmetry can be enforced or not.'''

    Z = zeros((npeaks, npeaks)) # make a square matrix of dimension == number of peaks

    if symmetric:
        # Place the relative exchange rates symmetrically in Z
        for index, rate in zip(peak_exchanges, relative_rates):
            Z[index[0],index[1]] = rate
            Z[index[1],index[0]] = rate

        # The diagonals of Z must be 1 minus the sum
        # of the off diagonals for that row
        sums = zeros(npeaks)
        for i in range(npeaks):
            sums[i] = sum(Z[i,:])
            Z[i,i]  = 1 - sums[i]

        # Now, if any of the sums are greater than 1, normalize
        if any(sums > 1):
            Z /= sums.max()
    else:
        # Place the relative exchange rates in Z
        for index, rate in zip(peak_exchanges, relative_rates):
            Z[index[0],index[1]] = rate
        
    return Z

def ZMat_web(first_peaks, second_peaks, forward_rates, reverse_rates):
    '''Construct the Z matrix.  Symmetry can be enforced or not.'''

    Z = zeros((len(first_peaks), len(second_peaks))) # make a square matrix of dimension == number of peaks

    for first, second, forward, back in zip(first_peaks, second_peaks, forward_rates, reverse_rates):
        Z[first][second] = forward
        Z[second][first] = back
        
    return Z


def spectrum(Z, k, vib, Gamma_Lorentz, Gamma_Gauss, heights, omega):
    '''This routine contains the code that drives the actual calculation
    of the intensities.
    '''
    npeaks = len(vib)
    N = range(npeaks)

    # Multiply Z-I by k to get K
    K = k * ( Z - eye(npeaks) )

    ############################
    # Find S, S^{-1}, and Lambda
    ############################

    # Construct the A matrix from K, the vibrational frequencies,
    # and the Lorentzian HWHM
    A = diag(-1j * vib + 0.5 * Gamma_Lorentz) - K
    # Lambda is the eigenvalues of A, S is the eigenvectors
    Lambda, S = eig(A)
    # Since the eigens are unordered, order by
    # the imaginary part of Lambda
    indx = argsort(abs(Lambda.imag))
    S, Sinv, Lambda = S[:,indx], inv(S[:,indx]), Lambda[indx]

    #################################
    # Use S and S^{-1} to find Gprime
    #################################

    # Convert Gamma_Gauss to sigma
    sigma = Gamma_Gauss * INVSQRT2LOG2_2

    # Construct the G matrix from sigma,
    # then use S and S^{-1} to get Gprime
    G = diag(sigma**(-2))
    # Off-diagonals are zero
    Gprime = array(diag(dot(dot(Sinv, G), S)), dtype=complex).real

    ##########################################
    # Construct an array of the new parameters
    ##########################################

    h = [height(j, heights, S, Sinv) for j in N]
    peaks = [-x.imag for x in Lambda]
    HWHM  = [x.real for x in Lambda]
    try:
        sigmas = [1 / sqrt(x) for x in Gprime]
    except ValueError:
        # I'm not sure this is a problem anymore, but this happened at some
        # stage of development
        raise SpectrumError('The input parameters for this system are '
                            'not physical.\nTry increasing the Gaussian '
                            'line widths')
    # Also create the modified input parameters for return
    GL = [2 * x for x in HWHM]
    GG = [SQRT2LOG2_2 * x for x in sigmas]
    new_params = peaks, GL, GG, [x.real for x in h]

    ################################################
    # Use these new values to calculate the spectrum
    ################################################

    # Return the sum of voigt profiles for each peak,
    # along with the new parameters
    return (array([voigt(omega, j, h, peaks, HWHM, sigmas) for j in N]).sum(0),
            new_params)


def voigt(freq, j, height, vib, HWHM, sigma):
    '''Return a Voigt line shape over a given domain about a given vib'''

    # Define what to pass to the complex error function
    z = ( freq - vib[j] + 1j*HWHM[j] ) / ( SQRT2 * sigma[j] )
    # The Voigt is the real part of the complex error function with some
    # scaling factors.  It is multiplied by the height here.
    return ( height[j].conjugate() * wofz(z) ).real / ( SQRT2PI * sigma[j] )


def height(j, heights, S, Sinv):
    '''Return the modified peak height'''
    N = range(len(heights))
    return sum([heights[a] * S[a,j] * Sinv[j,ap] for a in N for ap in N])


def run_non_interactive(file):
    '''Driver to calculate the spectra non-interactively
    (i.e. from the command line).
    '''

    # Read in the input file that is given
    try:
        args = read_input(file) # <-- just manually given them...
        print(args)
        #for a in args:
            #print(f"{a}")
    except (OSError, IOError) as e:
        print("something happened in reading file")
        pass

    # Generate the Z matrix
    Z = ZMat(
             len(args.num), 
             args.exchanges, 
             args.exchange_rates,
             args.symmetric_exchange
             )
    
    print("\n first the ZMat function")
    print(f"args.num = {args.num}")
    print(f"args.exchanges = {args.exchanges}")
    print(f"args.exchange_rates = {args.exchange_rates}")
    print(f"args.symmetric_exchange = {args.symmetric_exchange}")
    

    # Generate the frequency domain
    omega = np.arange(args.xlim[0]-10, args.xlim[1]+10, 0.5) # <-- this will be np.arrange()

    # Calculate the spectrum
    try:
        I_omega, new_params = spectrum(Z,
                                       args.k,
                                       args.vib,
                                       args.Gamma_Lorentz,
                                       args.Gamma_Gauss,
                                       args.heights,
                                       omega
                                      )
        
        print("\n Now for the spectrum function")
        print(f"args.k = {args.k}")
        print(f"args.vib = {args.vib}")
        print(f"args.Gamma_Lorentz = {args.Gamma_Lorentz}")
        print(f"args.Gamma_Gauss = {args.Gamma_Gauss}")
        print(f"args.heights = {args.heights}")
        #print(f"omega = {omega}")



    except SpectrumError as se:
        print("something went wrong with calculating the spectrum")
        return 1

    # Make a tuple of the old parameters
    old_params = (args.vib,
                  args.Gamma_Lorentz,
                  args.Gamma_Gauss,
                  args.heights)

    # Normalize the generated data
    I_omega = normalize(I_omega)
    # Repeat for the raw data if given.  Clip according to the xlimits
    if args.raw is not None:
        args.raw = clip(args.raw, args.xlim)
        args.raw[:,1] = normalize(args.raw[:,1])

    # Plot the data

    return plot(args, omega, I_omega)

run_non_interactive("/Users/benjaminlear/Documents/GitHub/rapid/web assembly/template.inp")