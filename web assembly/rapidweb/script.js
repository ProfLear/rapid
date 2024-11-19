let pyodideReadyPromise = loadPyodide();


document.getElementById('addRow').addEventListener('click', addRow);
document.getElementById('simulate').addEventListener('click', simulate);
document.getElementById('symmetric').addEventListener('change', updateSymmetricState);

let rowCount = 0;



async function loadPyodideAndPackages() {
    let pyodide = await pyodideReadyPromise;
    await pyodide.loadPackage('numpy');
    await pyodide.loadPackage('micropip');
    let micropip = pyodide.pyimport('micropip');
    await micropip.install(['plotly', 'lmfit']); // both plotly and lmfit are needed
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
            <div><input type="text" placeholder="Rate" class="forward_rate" data-pair="${pair.join('-')}"></div>
            <div class="exchange-arrow">
                <span class ="first_peak">${pair[0]}</span>
                <span class="exchange-arrow-symbol">⇌</span>
                <span class ="second_peak">${pair[1]}</span>
            </div>
            <div><input type="text" placeholder="Rate" class="backward_rate" data-pair="${pair.join('-')}" disabled></div>
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
    const bottomRates = document.querySelectorAll('.bottom-rate');
    bottomRates.forEach(rate => {
        rate.disabled = symmetric;
        if (symmetric) {
            rate.value = document.querySelector(`.top-rate[data-pair="${rate.dataset.pair}"]`).value;
        }
    });
}

function attachSymmetricHandlers() {
    const symmetric = document.getElementById('symmetric').checked;
    const topRates = document.querySelectorAll('.top-rate');
    if (symmetric) {
        topRates.forEach(rate => {
            rate.addEventListener('input', syncRates);
        });
    } else {
        topRates.forEach(rate => {
            rate.removeEventListener('input', syncRates);
        });
    }
}

function syncRates(event) {
    const pair = event.target.dataset.pair;
    const value = event.target.value;
    const bottomRate = document.querySelector(`.bottom-rate[data-pair="${pair}"]`);
    if (document.getElementById('symmetric').checked) {
        bottomRate.value = value;
    }
}

async function simulate() {
    let pyodide = await loadPyodideAndPackages();

    const positions = [...document.querySelectorAll('.position')].map(input => parseFloat(input.value));
    const intensities = [...document.querySelectorAll('.intensity')].map(input => parseFloat(input.value));
    const lorentzians = [...document.querySelectorAll('.lorentzian')].map(input => parseFloat(input.value));
    const gaussians = [...document.querySelectorAll('.gaussian')].map(input => parseFloat(input.value));
    
    const forward_rates = [...document.querySelectorAll('.forward_rate')].map(input => parseFloat(input.value));
    const reverse_rates = [...document.querySelectorAll('.reverse_rate')].map(input => parseFloat(input.value));

    const first_peaks = [...document.querySelectorAll('.first_peak')].map(input => parseInt(input.value));
    const second_peaks = [...document.querySelectorAll('.second_peaks')].map(input => parseInt(input.value));

    console.log('Positions:', positions);
    console.log('Intensities:', intensities);
    console.log('Lorentzians:', lorentzians);
    console.log('Gaussians:', gaussians);


    await pyodide.runPythonAsync(`

k = ${base_rate} * HZ2WAVENUM / ( 2 * pi ) # bring in k and convert to wavenumbers
positions = ${positions} # get values from js
lorentzians = ${lorentzians}
gaussians = ${gaussians}
heights = ${intensities}


# figure out what we are going to plot
plot_old = False
plot_stopped = True
plot_exp == True

# get all the information we are going to need from the webpage
xmin = 1900
xmin = 2050




# make the blank plot
from plotly.subplots import make_subplots
placeholder_points = ["Nan", "Nan"]

rapid_plot = make_subplots()

for i in range(4): # this will add 4 traces to the plot
    rapid_plot.add_scatter(x = placeholder_points, y = placeholder_points)

# now we can just update the ones we need...








#run the program to simulate, plot, etc
rapid_plot = web_driver()



# and now show this plot
rapid_json = rapid_plot.to_json()
rapid_json # this will be passed back via the javascript


# now save the old parameters 
old_k = k 
old_positions = positions # in units of wavenumbers
old_intensities = intensities
old_lorentzians = lorentzians
old_gaussians = gaussians


    `).then((fig_json) => {
        const plotlyChart = document.getElementById('plotlyChart');
        Plotly.react(plotlyChart, JSON.parse(rapid_json));
    });
}

// Add the first two rows by default
addRow();
addRow();

// Initialize Plotly plot with placeholder data
Plotly.newPlot('plotlyChart', [{
    x: [0, 1],
    y: [0, 1]
}], {
    title: 'Placeholder Plot'
});