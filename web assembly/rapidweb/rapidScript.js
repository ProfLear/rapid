let pyodideReadyPromise = loadPyodide();

window.onload = async function() {
    // Initialize plotly with placeholder data
    let pyodide = await loadPyodideAndPackages();

    pyodide.runPythonAsync(`
import plotly.graph_objects as go

rapid_plot = go.Figure()
for c, w, n in zip(
                    ["grey", "blue", "pink", "red"], 
                    [3,1,1,1],
                    ["experimental", "stopped", "last", "simulated"]
                    ):
    rapid_plot.add_scatter(x = [-1, -2], y = [-1, -1], # hide the lines initially
                    name = n,
                    mode = "lines", 
                    line = dict(color = c, width = w)
                    #showlegend = False,
                        )

rapid_plot.update_xaxes(range = [0,1], title = "wavenumber")
rapid_plot.update_yaxes(range = [0,1], title = "intensity")
rapid_plot.update_layout(template = "simple_white")

# And now show this plot
rapid_json = rapid_plot.to_json()
rapid_json # This will be passed back via the JavaScript
    `).then((rapid_json) => {
        const plotlyChart = document.getElementById('plotlyChart');
        Plotly.react(plotlyChart, JSON.parse(rapid_json));
    });
};

document.getElementById('addRow').addEventListener('click', addRow);
document.getElementById('expData').addEventListener('change', plotExpData); // Ensure event listener is properly set up
document.getElementById('simulate').addEventListener('click', simulate);
document.getElementById('symmetric').addEventListener('change', updateSymmetricState);

let rowCount = 0;

async function loadPyodideAndPackages() {
    let pyodide = await pyodideReadyPromise;
    await pyodide.loadPackage('numpy');
    await pyodide.loadPackage('micropip');
    let micropip = pyodide.pyimport('micropip');
    await micropip.install(['plotly', 'lmfit']); // Both plotly and lmfit are needed
    return pyodide;
}

function addRow() {
    rowCount++;
    const row = document.createElement('div');
    row.classList.add('input-row');
    row.dataset.rowNumber = rowCount;
    row.innerHTML = `
        <label class="row-number">${rowCount}</label>
        <input type="text" placeholder="Position" class="position input-box">
        <input type="text" placeholder="Intensity" class="intensity input-box">
        <input type="text" placeholder="Lorentzian" class="lorentzian input-box">
        <input type="text" placeholder="Gaussian" class="gaussian input-box">
        <span class="remove-row">x</span>
    `;
    document.getElementById('inputRows').appendChild(row);
    row.querySelector('.remove-row').addEventListener('click', () => removeRow(row));
    updateExchangeBlock();
}

function removeRow(row) {
    document.getElementById('inputRows').removeChild(row);
    updateRowNumbers();
    updateExchangeBlock();
}

function updateRowNumbers() {
    const rows = document.querySelectorAll('.input-row');
    rowCount = 0;
    rows.forEach((row, index) => {
        rowCount++;
        row.dataset.rowNumber = rowCount;
        row.querySelector('.row-number').textContent = rowCount;
    });
}

function updateExchangeBlock() {
    const rows = document.querySelectorAll('.input-row');
    const exchangeContainer = document.getElementById('exchangeContainer');
    exchangeContainer.innerHTML = ''; // Clear existing content

    const rowNumbers = Array.from(rows).map(row => row.dataset.rowNumber);
    const pairs = getUniquePairs(rowNumbers);

    pairs.forEach(pair => {
        const exchangeEntry = document.createElement('div');
        exchangeEntry.classList.add('exchange-entry');
        exchangeEntry.innerHTML = `
            <div>relative rate: <input type="text" value="1" class="forward_rate" data-pair="${pair.join('-')}"></div>
            <div class="exchange-arrow">
                <span class="first_peak">${pair[0]}</span>
                <span class="exchange-arrow-symbol">⇌</span>
                <span class="second_peak">${pair[1]}</span>
            </div>
            <div>relative rate: <input type="text" value="1" class="backward_rate" data-pair="${pair.join('-')}" disabled></div>
        `;
        exchangeContainer.appendChild(exchangeEntry);
        addBufferLine(exchangeContainer);
    });

    updateSymmetricState();
    attachSymmetricHandlers();
}

function addBufferLine(container) {
    const bufferLine = document.createElement('div');
    bufferLine.style.height = '1px';
    container.appendChild(bufferLine);
}

function getUniquePairs(array) {
    const pairs = [];
    for (let i = 0; i < array.length; i++) {
        for (let j = i + 1; j < array.length; j++) {
            pairs.push([array[i], array[j]]);
        }
    }
    return pairs;
}

function updateSymmetricState() {
    const symmetric = document.getElementById('symmetric').checked;
    const backwardRates = document.querySelectorAll('.backward_rate');
    backwardRates.forEach(rate => {
        rate.disabled = symmetric;
        if (symmetric) {
            rate.value = document.querySelector(`.forward_rate[data-pair="${rate.dataset.pair}"]`).value;
        }
    });
}

function attachSymmetricHandlers() {
    const symmetric = document.getElementById('symmetric').checked;
    const forwardRates = document.querySelectorAll('.forward_rate');
    if (symmetric) {
        forwardRates.forEach(rate => {
            rate.addEventListener('input', syncRates);
        });
    } else {
        forwardRates.forEach(rate => {
            rate.removeEventListener('input', syncRates);
        });
    }
}

function syncRates(event) {
    const pair = event.target.dataset.pair;
    const value = event.target.value;
    const backwardRate = document.querySelector(`.backward_rate[data-pair="${pair}"]`);
    if (document.getElementById('symmetric').checked) {
        backwardRate.value = value;
    }
}

async function plotExpData() {
    console.log("Function plotExpData called."); // Debug log to ensure the function is called
    let pyodide = await loadPyodideAndPackages();

    // Get the JSON representation of the Plotly chart
    const plt = document.getElementById('plotlyChart');

    // Serialize the Plotly chart to JSON
    const plotlyChartJSON = JSON.stringify({
        data: plt.data,
        layout: plt.layout,
    });
    // Replace `true` with `True` and `false` with `False` to be Python compatible
    const plotlyChartJSON_TF = plotlyChartJSON.replace(/true/g, 'True').replace(/false/g, 'False');

    const fileInput = document.getElementById("expData");
    console.log("Checking if file is selected."); // Debug log

    if (fileInput.files.length > 0) {
        console.log("File selected."); // Debug log
        const file = fileInput.files[0];
        const reader = new FileReader();

        reader.onload = async function(e) {
            console.log("FileReader onload event triggered."); // Debug log
            const exp_spectrum = e.target.result;
            console.log("File content loaded."); // Debug log

            // Set the file content and chart JSON in Pyodide globals
            await pyodide.globals.set("spectrum_file_content", exp_spectrum);
            await pyodide.globals.set("plotlyChartJSON_TF", plotlyChartJSON_TF);

            const plotData = await pyodide.runPythonAsync(`
import numpy as np
import plotly.graph_objects as go
from io import StringIO

# obtain and load the spectrum data
spectrum_file_content = globals()['spectrum_file_content']
exp_spectrum = np.genfromtxt(StringIO(spectrum_file_content), delimiter=',', skip_header=2, usecols=[0, 1], unpack=True)

# obtain the currently plotly figure
plotly_json_dict = ${plotlyChartJSON_TF}
rapid_plot = go.Figure(plotly_json_dict)


rapid_plot.data[0].x = exp_spectrum[0]
rapid_plot.data[0].y = exp_spectrum[1]

# format the axes correctly...
xmins = [] # these are the current minimum x-values
xmaxs = []
ymins = []
ymaxs = []
for i in range(4):
    if min(rapid_plot.data[i].x) > 0: # only get positive values, since this means it is not the starting values
        xmins.append(min(rapid_plot.data[i].x))
        ymins.append(min(rapid_plot.data[i].y))
        xmaxs.append(max(rapid_plot.data[i].x))
        ymaxs.append(max(rapid_plot.data[i].y))

print(xmins)
print(xmaxs)

print(ymins)
print(ymaxs)

y_range = max(ymaxs) - min(ymins)
rapid_plot.update_xaxes(range = [
                                max(xmins), min(xmaxs) # taking the max and min this way, means we will get the simulated range, if we have it.
                                ])

rapid_plot.update_yaxes(range = [
                                min(ymins), max(ymaxs) + y_range*0.05
                                ])     

rapid_json = rapid_plot.to_json()
rapid_json
            `).then((rapid_json) => {
                //console.log("Plot data received from Python:", rapid_json); // Debug log to show received plot data
                const plotlyChart = document.getElementById('plotlyChart');
                Plotly.react(plotlyChart, JSON.parse(rapid_json));
            });
        };

        reader.onerror = function() {
            console.error("FileReader error:", reader.error); // Debug log for errors
        };

        reader.onabort = function() {
            console.warn("FileReader aborted."); // Debug log for aborts
        };

        console.log("Reading file as text."); // Debug log
        reader.readAsText(file);
    } else {
        console.log("No file selected."); // Debug log
    }
}



async function simulate() {
    let pyodide = await loadPyodideAndPackages();

    const positions = JSON.stringify([...document.querySelectorAll('.position')].map(input => parseFloat(input.value)));
    const intensities = JSON.stringify([...document.querySelectorAll('.intensity')].map(input => parseFloat(input.value)));
    const lorentzians = JSON.stringify([...document.querySelectorAll('.lorentzian')].map(input => parseFloat(input.value)));
    const gaussians = JSON.stringify([...document.querySelectorAll('.gaussian')].map(input => parseFloat(input.value)));
    
    const base_rate = document.querySelector('.base_rate').value;
    const forward_rates = JSON.stringify([...document.querySelectorAll('.forward_rate')].map(input => parseFloat(input.value)));
    const reverse_rates = JSON.stringify([...document.querySelectorAll('.backward_rate')].map(input => parseFloat(input.value)));

    const first_peaks = JSON.stringify([...document.querySelectorAll('.first_peak')].map(peak => parseInt(peak.textContent)));
    const second_peaks = JSON.stringify([...document.querySelectorAll('.second_peak')].map(peak => parseInt(peak.textContent)));

    // Get the JSON representation of the Plotly chart
    const plt = document.getElementById('plotlyChart');

    // Serialize the Plotly chart to JSON
    const plotlyChartJSON = JSON.stringify({
        data: plt.data,
        layout: plt.layout,
    });
    // Replace `true` with `True` and `false` with `False` to be Python compatible
    const plotlyChartJSON_TF = plotlyChartJSON.replace(/true/g, 'True').replace(/false/g, 'False');
    console.log('positions:', positions);
    console.log('intensities:', intensities);
    console.log('lorentzians:', lorentzians);
    console.log('gaussians:', gaussians);

    console.log('base_rate:', base_rate);
    console.log('forward_rates:', forward_rates);
    console.log('reverse_rates:', reverse_rates);
    console.log('first_peaks:', first_peaks);
    console.log('second_peaks:', second_peaks);

    await pyodide.runPythonAsync(`

#python code pasted here
import plotly.io as pio
import plotly.graph_objects as go
import math
import numpy as np
from scipy.linalg import eig, inv
from scipy.special import wofz



# A few constants
SQRT2LOG2_2 = math.sqrt(2 * math.log(2)) * 2  #natural logarithm
INVSQRT2LOG2_2 = 1 / SQRT2LOG2_2
SQRT2 = math.sqrt(2)
SQRT2PI = math.sqrt(2 * math.pi)
HZ2WAVENUM = 1 / ( 100 * 2.99792458E8 ) # Hz to cm^{-1} conversion



def ZMat_web(first_peaks, second_peaks, forward_rates, reverse_rates):
    '''Construct the Z matrix.  Symmetry can be enforced or not.'''

    if len(first_peaks) != len(second_peaks):
        print("There is a problem with the number of your peaks.")
        
    npeaks = max(max(first_peaks), max(second_peaks)) # find the largest numbered peak
    Z = np.zeros(
        (npeaks, npeaks)
        ) # make a square matrix of dimension == number of peaks

    for first, second, forward, back in zip(first_peaks, second_peaks, forward_rates, reverse_rates):
        Z[first-1][second-1] = forward  # the -1 is to account for index starting at 0
        Z[second-1][first-1] = back # the -1 is to account for index starting at 0
    
    # The diagonals of Z must be 1 minus the sum
        # of the off diagonals for that row
        sums = np.zeros(npeaks)
        for i in range(npeaks):
            sums[i] = sum(Z[i,:])
            Z[i,i]  = 1 - sums[i]

        # Now, if any of the sums are greater than 1, normalize
        if any(sums > 1):
            Z /= sums.max()    
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

def spectrum_web(k, # this is the base rate
                 peak_vals, # values for the peaks
                 exchange_vals, # values for the exchange
                 omega, # the x-values along which we will simulate the spectrum
                 ):
    
    # unpack the variables...
    vib, Gamma_Lorentz, Gamma_Gauss, heights = peak_vals
    first_peaks, second_peaks, forward_rates, reverse_rates = exchange_vals
    
    '''This routine contains the code that drives the actual calculation
    of the intensities.
    '''
    
    Z = ZMat_web(first_peaks, second_peaks, forward_rates, reverse_rates)
    
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
        print("The input parameters for this system are not physical. Try increasing the Gaussian line widths")
    
    # Also create the modified input parameters for return
    GL = [2 * x for x in HWHM]
    GG = [SQRT2LOG2_2 * x for x in sigmas]
    new_params = peaks, GL, GG, [x.real for x in h] # in case we want to use these later

    ################################################
    # Use these new values to calculate the spectrum
    ################################################

    return np.array([voigt(omega, j, h, peaks, HWHM, sigmas) for j in N]).sum(0), new_params

#
# now, start the processing...
#


# get the plotly figure and convert it to my desired figure object
plotly_json_dict = ${plotlyChartJSON_TF}
rapid_plot = go.Figure(plotly_json_dict)


k = ${base_rate}*1e12 * HZ2WAVENUM / ( 2 * math.pi ) # bring in k and convert to wavenumbers

peak_values = [
    np.array(${positions}),
    np.array(${lorentzians}),
    np.array(${gaussians}),
    np.array(${intensities}),
    ]

exchange_values = [
    np.array(${first_peaks}),
    np.array(${second_peaks}),
    np.array(${forward_rates}),
    np.array(${reverse_rates}),
    ]


# find the limits for the simulated data out how wide we need to simulate data
max_width = max(0.5346*peak_values[1] + np.sqrt(0.2166*peak_values[1]**2 + peak_values[2]**2))
sim_lims = [
    min(peak_values[0]) - 6*max_width,
    max(peak_values[0]) + 6*max_width
    ]

# determine the number of points we need to simulate
if len(rapid_plot.data[0].x) > 2000:
    n_sim_points = len(rapid_plot.data[0].x) # then set this equal to the number of points that the experimental spectrum has
else: 
    n_sim_points = 2000
   
# make x-values to use for the simulation
sim_x = np.linspace(sim_lims[0], sim_lims[1], n_sim_points)


# figure out what we are going to plot <-- eventually read from the html
plot_old = True
plot_stopped = True

# then go through and update each trace, if we need to...     

# if we want to see the last simulation, then that is fine
if plot_old == True:
    # get the data from the original trace and add it to the "old trace"
    rapid_plot.data[2].x = rapid_plot.data[3].x
    rapid_plot.data[2].y = rapid_plot.data[3].y

if plot_stopped == True:
    #simulate this
    stopped_spectrum, stopped_params = spectrum_web(0, peak_values, exchange_values, sim_x) # pass with no rate
    rapid_plot.data[1].x = sim_x
    rapid_plot.data[1].y =  stopped_spectrum 

# also, simulate the actual thing, since we know we want that. 
sim_spectrum, sim_params = spectrum_web(k, peak_values, exchange_values, sim_x) 
rapid_plot.data[3].x = sim_x
rapid_plot.data[3].y = sim_spectrum

# format the axes correctly...
xmins = [] # these are the current minimum x-values
xmaxs = []
ymins = []
ymaxs = []
for i in range(4):
    if min(rapid_plot.data[i].x) > 0: # only get positive values, since this means it is not the starting values
        xmins.append(min(rapid_plot.data[i].x))
        ymins.append(min(rapid_plot.data[i].y))
        xmaxs.append(max(rapid_plot.data[i].x))
        ymaxs.append(max(rapid_plot.data[i].y))

print(xmins)
print(xmaxs)

print(ymins)
print(ymaxs)

y_range = max(ymaxs) - min(ymins)
rapid_plot.update_xaxes(range = [
                                max(xmins), min(xmaxs) # taking the max and min this way, means we will get the simulated range, if we have it.
                                ])

rapid_plot.update_yaxes(range = [
                                min(ymins), max(ymaxs) + y_range*0.05
                                ])   

rapid_json = rapid_plot.to_json()
rapid_json`).then((rapid_json) => {
        const plotlyChart = document.getElementById('plotlyChart');
        Plotly.react(plotlyChart, JSON.parse(rapid_json));
    });
}

// Add the first two rows by default
addRow();
addRow();
