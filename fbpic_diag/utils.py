import numpy as np 
import sys
def divergence(px=None, py=None, pz=None):

    """
    Function that computes the bunch divergence,
    either along a planar slice (div = px/pz) or the
    total divergence as sqrt((px**2+py**2)/pz**2)

    Parameters
    --------
    px: np.array
        Transverse momentum along the first direction
    py: np.array
        Transverse momentum along the second direction.
        If None it distinguish between the planar divergence and
        the solid one
    pz: np.array
        Longitudinal momentum

    Returns
    --------
    div: np.array
        Divergence
    """
    if py is not None:
        div = np.arctan(np.sqrt((px**2+py**2))/pz)
    else:
        div = np.arctan(px/pz)

    return div

def central_average(x, w):

    """
    Function to calculate the second order momentum of quantity x
    with w-distribution.
    **Parameters**
    x: 1darrays of particles' phase space coord
    w: ndarray of particles' weights
    **Returns**
    average: float
    """
    x_mean = np.ma.average(x, weights=w)
    sigma_x2 = np.ma.average((x-x_mean)**2, weights=w)
    average = np.sqrt(sigma_x2)
    return average

def covar(x, ux, w):

    """
    Function to calculate covariance of x, ux variables.
    **Parameters**
    x, ux: two 1darrays of  phase-space coords
    w: ndarray of particles' weights
    **Returns**
    covariance: float
    """
    x_mean = np.ma.average(x, weights=w)
    ux_mean = np.ma.average(ux, weights=w)
    covariance = np.ma.average((x-x_mean)*(ux-ux_mean), weights=w)
    return covariance

def emittance(x, ux, w):

    """
    Function to calculate emittance of a bunch.
    **Parameters**
    x, ux: two 1darrays of  phase-space coords
    w: ndarray of particles' weights
    **Returns**
    emittance: float
    """
    sigma_x = central_average(x, w)
    sigma_ux = central_average(ux, w)
    covariance = covar(x, ux, w)

    emit = np.sqrt(sigma_x**2*sigma_ux**2-covariance**2)

    return emit

def twiss(x, px, pz, w, type):

    """
    Function to calulate the Courant-Snyder parameters
    of the bunch
    **Parameters**
    x: np.array
        The space coords of particle.
    px, pz: np.arrays
        The transverse and longitudinal momenta of particles
        to calculate the planar slice slope corresponding to 'x'
    w: np.array
        Weights of particles
    type: str
        'alpha', 'beta', or 'gamma' to select the desired twiss
    **Returns**
    tw: float
        Twiss parameter specified
    """
    slope = divergence(px=px, pz=pz)
    emit = emittance(x, slope, w)
    inv_emit = 1/emit
    if type == 'alpha':
        covariance = covar(x, slope, w)
        tw = covariance*(-inv_emit)
    elif type == 'beta':
        sigma_x = central_average(x, w)
        tw = sigma_x**2*inv_emit
    elif type == 'gamma':
        sigma_slope = central_average(slope, w)
        tw = sigma_slope**2*inv_emit

    return tw

def mean(x, w):
    mean = np.ma.average(x, weights=w)
    return mean

def energy_spread(gamma, w, kind='rms'):

    """
    Function to calculate energy spread of bunch's energy spectra
    **Parameters**
        gamma: float, array
            An array of normalized energy values
        w: float, array
            An array of weights
    """
    match kind:
        case 'rms':
            mean = mean(gamma, w)
            dev = central_average(gamma, w)
        case 'mad':
            mean = weighted_median(gamma, w)
            dev = median_absolute_deviation(gamma, w)*1.4826
    sigma = dev/mean
    return sigma

def weighted_median(x,w=None):

    """
    Inspired from "weightedstats" by Jack Peterson: minor changes
    using numpy functions to speed up
    **Parameters**
        x: iterable
            Array of values from which build a distribution and 
            calculate median
        w: iterable or None
            Array of weights; if None return standard median
    """
    if all((isinstance(tmp,np.ndarray) for tmp in (x,w))):
        pass
    else:
        x,w = map(np.array,(x,w))
    if w is None:
        return np.ma.median(x)
    if any(w > 0):
        sorted_w = w[np.ma.argsort(x)]
        sorted_x = np.ma.sort(x)
        midpoint = 0.5 * np.ma.sum(sorted_w)
        if any(w > midpoint):
            return (x[np.ma.argmax(w)])[0]
        cumulative_weight = np.ma.cumsum(sorted_w)
        below_midpoint_index = np.ma.where(cumulative_weight <= midpoint)[0][-1]
        if np.ma.abs(cumulative_weight[below_midpoint_index] - midpoint) < sys.float_info.epsilon:
            return np.ma.mean(sorted_x[below_midpoint_index:below_midpoint_index+2])
        return sorted_x[below_midpoint_index+1]
    
def median_absolute_deviation(x, w):
    median = weighted_median(x, w)
    mad = weighted_median(np.ma.abs(x-median),w)
    return mad