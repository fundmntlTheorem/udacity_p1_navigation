from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
from cycler import cycler

def setup_environment(cycle_type=None):
    '''
        Call the normal things I call to get the matplotlib settings I like.
    '''
	
    if cycle_type=="monochrome":
        # set mono chrome cycler
        plt.rcParams['axes.prop_cycle'] = (cycler('color', ['k']) * cycler('marker', ['', '.']) * cycler('linestyle', ['-', '--', ':', '-.']))
    elif cycle_type=="grayscale":
        plt.rcParams['axes.prop_cycle'] = (cycler('color', ['k', '0.75', '0.50', '0.25']))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    set_matplotlib_formats('png')
    SMALL_SIZE = 6
    MEDIUM_SIZE = 7
    BIGGER_SIZE = 10

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 5
    params = {
        'figure.figsize': (3, 2),
        # controls default text sizes
        'font.size': SMALL_SIZE,
        # fontsize of the axes title
        'axes.titlesize': SMALL_SIZE,
        # fontsize of the x and y labels
        'axes.labelsize': SMALL_SIZE,
        'xtick.labelsize': SMALL_SIZE,
        'ytick.labelsize': SMALL_SIZE,
        'legend.fontsize': SMALL_SIZE,
        'figure.titlesize': BIGGER_SIZE
    }
    plt.rcParams.update(params)
    return colors