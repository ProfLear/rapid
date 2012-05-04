from __future__ import print_function
from sys import stderr, stdout
try:
    from input_reader import ReaderError
except ImportError:
    print("Cannot find input_reader module", file=stderr)
    print("Find it at github.com/SethMMorton/input_reader", file=stderr)
    return 1
from output import write, save_script, plot
from .common import spectrum, SpectrumError

def run_non_interactive(input_file):
    '''Driver to calculate the spectra non-interactively
    (i.e. from the command line).
    '''

    # Read in the input file that is given
    try:
        args = read_input(input_file)
    except (OSError, IOError) as e:
        print(str(e), file=stderr) # An error occured when locating the file
        return 1
    except ReaderError as r:
        print(str(r), file=stderr) # An error occured when reading the file
        return 1

    # Calculate the spectrum
    try:
        I_omega, omega, new_params = spectrum(args.exchanges,
                                              args.exchange_rates,
                                              args.k,
                                              args.vib,
                                              args.Gamma_Lorentz
                                              args.Gamma_Gaussian,
                                              args.heights
                                             )
    except SpectrumError as se:
        print(str(se), file=stderr)
        return 1

    # Make a tuple of the old parameters
    old_params = (args.vib,
                  args.Gamma_Lorentzian,
                  args.Gamma_Gaussian,
                  args.heights)

    # Plot the data or write to file
    if 'data' in args:
        return write(omega, I_omega, args.data)
    elif args.parameters:
        return numerics(old_params, new_params, stdout)
    elif args.save_plot_script is not None:
        return save_script(omega, I_omega, old_params, new_params,
                           args.save_plot_script)
    else:
        return plot(args, omega, I_omega)
