#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 08:03:00 2024

@author: benjaminlear
"""
import math
import numpy as np
from scipy.linalg import eig, inv
from scipy.special import wofz
from plotly.subplots import make_subplots


# A few constants
SQRT2LOG2_2 = math.sqrt(2 * math.log(2)) * 2  #natural logarithm
INVSQRT2LOG2_2 = 1 / SQRT2LOG2_2
SQRT2 = math.sqrt(2)
SQRT2PI = math.sqrt(2 * math.pi)
HZ2WAVENUM = 1 / ( 100 * 2.99792458E8 ) # Hz to cm^{-1} conversion



def ZMat_web(first_peaks, second_peaks, forward_rates, reverse_rates):
    '''Construct the Z matrix.  Symmetry can be enforced or not.'''

    Z = np.zeros((len(first_peaks), len(second_peaks))) # make a square matrix of dimension == number of peaks

    for first, second, forward, back in zip(first_peaks, second_peaks, forward_rates, reverse_rates):
        Z[first][second] = forward
        Z[second][first] = back
        
    return Z

def height(j, heights, S, Sinv):
    '''Return the modified peak height'''
    N = range(len(heights))
    return sum([heights[a] * S[a,j] * Sinv[j,ap] for a in N for ap in N])


def voigt(freq, j, height, vib, HWHM, sigma):
    '''Return a Voigt line shape over a given domain about a given vib'''

    # Define what to pass to the complex error function
    z = ( freq - vib[j] + 1j*HWHM[j] ) / ( SQRT2 * sigma[j] )
    # The Voigt is the real part of the complex error function with some
    # scaling factors.  It is multiplied by the height here.
    return ( height[j].conjugate() * wofz(z) ).real / ( SQRT2PI * sigma[j] )

def spectrum_web(Z, k, vib, Gamma_Lorentz, Gamma_Gauss, heights, omega):
    '''This routine contains the code that drives the actual calculation
    of the intensities.
    '''
    npeaks = len(vib)
    N = range(npeaks)

    # Multiply Z-I by k to get K
    K = k * ( Z - np.eye(npeaks) )

    ############################
    # Find S, S^{-1}, and Lambda
    ############################

    # Construct the A matrix from K, the vibrational frequencies,
    # and the Lorentzian HWHM
    A = np.diag(-1j * vib + 0.5 * Gamma_Lorentz) - K
    # Lambda is the eigenvalues of A, S is the eigenvectors
    Lambda, S = eig(A)
    # Since the eigens are unordered, order by
    # the imaginary part of Lambda
    indx = np.argsort(abs(Lambda.imag))
    S, Sinv, Lambda = S[:,indx], inv(S[:,indx]), Lambda[indx]

    #################################
    # Use S and S^{-1} to find Gprime
    #################################

    # Convert Gamma_Gauss to sigma
    sigma = Gamma_Gauss * INVSQRT2LOG2_2

    # Construct the G matrix from sigma,
    # then use S and S^{-1} to get Gprime
    G = np.diag(sigma**(-2))
    # Off-diagonals are zero
    Gprime = np.array(np.diag(np.dot(np.dot(Sinv, G), S)), dtype=complex).real

    ##########################################
    # Construct an array of the new parameters
    ##########################################

    h = [height(j, heights, S, Sinv) for j in N]
    peaks = [-x.imag for x in Lambda]
    HWHM  = [x.real for x in Lambda]
    try:
        sigmas = [1 / math.sqrt(x) for x in Gprime]
    except ValueError:
        # I'm not sure this is a problem anymore, but this happened at some
        # stage of development
        print('The input parameters for this system are '
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
    return (np.array([voigt(omega, j, h, peaks, HWHM, sigmas) for j in N]).sum(0),
            new_params)


def web_sim_driver (spectrum_vals, # a list of values to be use din making the spectrum: k, positions, lorentzians, gaussians, heights,
                Z_vals, # a list of values to be used in making the Z-matrix: first_peaks, second_peaks, forward_rates, reverse_rates
                xlims = ["auto", "auto"], # a list of two objects
                exp_spectrum = None, # list containing x and y values for the experimental spectrum
                plot_exp = True, # the default is to show the experimental spectrum, if it exists. 
                plot_stopped = True, # the default is to see the spectrum, without exchange
                plot_old = False, # the default is to not see the last iteration
                ):
    
    #unpack the spectrum_vals and the Z_vals
    k, positions, gaussians, lorentzians, heights = spectrum_vals
    first_peaks, second_peaks, forward_rates, reverse_rates = Z_vals
    
    lorentzians = np.array(lorentzians)
    gaussians = np.array(gaussians)
    
    #first, build the sim_lims
    max_width = max(0.5346*lorentzians + np.sqrt(0.2166*lorentzians**2 + gaussians**2))
    sim_lims = [
        min(positions) - 6*max_width,
        max(positions) + 6*max_width
        ]
    
    
    if xlims[0] == "auto": # then calculate them...
        xlims[0] = sim_lims[0]
    
    if xlims[1] == "auto":
        xlims[1] = sim_lims[1]
        
        
        xlims = [1900, 2050] # eventually, base this off the lorentzian and gaussian parameters
    sim_x = np.linspace(xlims[0], xlims[1], 2000)
    
    # then build the sim_limits
    
    # get the Z-matrix
    Z = ZMat_web(first_peaks, second_peaks, forward_rates, reverse_rates)
    
    
    
    # make our plot
    rapid_plot = make_subplots()
    
    # add the experimental data trace
    if exp_spectrum in globals() and plot_exp == True:
        add_exp_trace
        rapid_plot.add_scatter(x = exp_spectrum[0], y = exp_spectrum[1], mode = "lines", lines = dict(color = "grey", width = 3),)
    
    # make and add the "stopped exchange" trace
    if plot_stopped == True:
        rapid_plot.add_scatter(x = sim_x, y = spectrum_web(Z, 
                                                           0, 
                                                           positions, 
                                                           lorentzians, 
                                                           gaussians, 
                                                           heights, 
                                                           sim_x),
                                                       mode = "lines",
                                                       lines = dict(color = "blue", width = 1),
                                                    )
    
    # make and add the old simulated data <-- can get this from the plot on the webpage??
    if old_k in globals() and plot_old == True:
        rapid_plot.add_scatter(x = sim_x, y = spectrum_web(Z, 
                                                       old_k, 
                                                       old_positions, 
                                                       old_lorentzians, 
                                                       old_gaussians, 
                                                       old_heights, 
                                                       sim_x),
                                                   mode = "lines",
                                                   lines = dict(color = "pink", width = 1),
                                                )
    
    # make the new simulated trace
    rapid_plot.add_scatter(x = sim_x, y = spectrum_web(Z, 
                                                       k, 
                                                       positions, 
                                                       lorentzians, 
                                                       gaussians, 
                                                       heights, 
                                                       sim_x),
                                                   mode = "lines",
                                                   lines = dict(color = "red", width = 1),
                                                )
    
    # now format this plot
    
    rapid_plot.update_xaxes(range = [xmin, xmax], title = "wavenumbers")
    
    rapid_plot.update_layout(template = "simple_white")
    
    return rapid_plot

