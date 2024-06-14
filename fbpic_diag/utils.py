import numpy as np 

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

def mean(x, w, energy=False):

    mean = np.ma.average(x, weights=w)
    if energy:
        mean *= 0.511
    return mean

def energy_spread(gamma, w):

    """
    Function to calculate energy spread of bunch's energy spectra
    **Parameters**
        gamma: float, array
            An array of normalized energy values
        w: float, array
            An array of weights
    """
    mean = np.ma.average(gamma, weights=w)
    average = central_average(gamma, w)
    sigma = average/mean
    return sigma