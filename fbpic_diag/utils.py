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

def mean(x, w):
    x = np.ma.masked_invalid(x)
    w = np.ma.masked_invalid(w)
    m = np.ma.average(x, weights=w)
    return m

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
    if w is None:
        x = np.ma.masked_invalid(x)
        return np.ma.median(x)
    else:
        x,w = map(np.ma.masked_invalid,(x,w))
    if any(w > 0):
        sorted_w = w[np.ma.argsort(x)]
        sorted_x = np.ma.sort(x)
        midpoint = 0.5 * np.ma.sum(sorted_w)
        if any(w > midpoint):
            return [x[np.ma.argmax(w)]][0]
        cumulative_weight = np.ma.cumsum(sorted_w)
        below_midpoint_indexes = np.ma.where(cumulative_weight <= midpoint)[0]
        if below_midpoint_indexes.size == 0:
            return np.nan
        else:
            below_midpoint_index = below_midpoint_indexes[-1]
        if np.ma.abs(cumulative_weight[below_midpoint_index] - midpoint) < sys.float_info.epsilon:
            return np.ma.mean(sorted_x[below_midpoint_index:below_midpoint_index+2])
        return sorted_x[below_midpoint_index+1]
    else:
        return np.nan
    
def central_average(x, w, kind):

    """
    Function to calculate the second order momentum of quantity x
    with w-distribution.
    **Parameters**
    x: 1darrays of particles' phase space coord
    w: ndarray of particles' weights
    **Returns**
    average: float
    """
    match kind:
        case 'rms':
            x_mean = mean(x,w)
            average = np.sqrt(mean((x-x_mean)**2, w))
        case 'mad':
            x_mean = weighted_median(x,w)
            average = weighted_median(abs(x-x_mean),w)*1.4826
    return average

def covar(x, ux, w, kind):

    """
    Function to calculate covariance of x, ux variables.
    **Parameters**
    x, ux: two 1darrays of  phase-space coords
    w: ndarray of particles' weights
    **Returns**
    covariance: float
    """
    match kind:
        case 'rms':
            x_mean = mean(x, w)
            ux_mean = mean(ux, w)
            covariance = mean((x-x_mean)*(ux-ux_mean), w)
        case 'mad':
            x_mean = weighted_median(x,w)
            ux_mean = weighted_median(ux,w)
            covariance = weighted_median((x-x_mean)*(ux-ux_mean),w)    
    return covariance

def emittance(x, ux, w,kind):

    """
    Function to calculate emittance of a bunch.
    **Parameters**
    x, ux: two 1darrays of  phase-space coords
    w: ndarray of particles' weights
    **Returns**
    emittance: float
    """
    sigma_x = central_average(x, w, kind)
    sigma_ux = central_average(ux, w, kind)
    covariance = covar(x, ux, w, kind)

    emit = np.sqrt(sigma_x**2*sigma_ux**2-covariance**2)

    return emit

def twiss(x, px, pz, w, type, kind):

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
    emit = emittance(x, slope, w,kind)
    inv_emit = 1/emit
    if type == 'alpha':
        covariance = covar(x, slope, w, kind)
        tw = covariance*(-inv_emit)
    elif type == 'beta':
        sigma_x = central_average(x, w, kind)
        tw = sigma_x**2*inv_emit
    elif type == 'gamma':
        sigma_slope = central_average(slope, w, kind)
        tw = sigma_slope**2*inv_emit

    return tw

def energy_spread(gamma, w, kind):

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
            Mean = mean(gamma,w)
        case 'mad':
            Mean = weighted_median(gamma, w)
    dev = central_average(gamma, w, kind)
    sigma = dev/Mean
    return sigma

def if_not_div(components):
    if 'div_' in components[0] or 'div_' in components[1]:
        return False
    else:
        return True
def where_div(components):
    if 'div_' in components[0] and 'div_' not in components[1]:
        return 0
    elif 'div_' in components[1] and 'div_' not in components[0]:
        return 1
    else:
        return 'both'
def which_div(components,where):
    dictio=dict()
    if where == 'both':
        for i,div in enumerate(components):
            dictio[i] = div.split('_')[1]
    else:
        dictio[where] = components[where].split('_')[1]
    return dictio
