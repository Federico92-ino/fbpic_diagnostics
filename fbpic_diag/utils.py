import numpy as np
from scipy.constants import pi, m_e, c, e
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
    
def dev(x, w, kind):

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
            deviation = np.sqrt(mean((x-x_mean)**2, w))
        case 'mad':
            x_mean = weighted_median(x,w)
            deviation = weighted_median(abs(x-x_mean),w)*1.4826
    return deviation

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
            dev_x,dev_ux = [dev(v,w,'mad') for v in [x,ux]]
            z_x = x/dev_x
            z_ux = ux/dev_ux
            dev_p = dev((z_x+z_ux),w,'mad')
            dev_m = dev((z_x-z_ux),w,'mad')
            cor = (dev_p**2-dev_m**2)/(dev_p**2+dev_m**2)
            covariance = cor*dev_x*dev_ux
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
    sigma_x = dev(x, w, kind)
    sigma_ux = dev(ux, w, kind)
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
        sigma_x = dev(x, w, kind)
        tw = sigma_x**2*inv_emit
    elif type == 'gamma':
        sigma_slope = dev(slope, w, kind)
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
    deviation = dev(gamma, w, kind)
    sigma = deviation/Mean
    return sigma

def CartField(Fr,Ft,theta,coord):
    if coord == 'x':
        F = Fr*np.cos(theta)-Ft*np.sin(theta)
    elif coord == 'y':
        F = Fr*np.sin(theta)+Ft*np.cos(theta)
    return F

def length_um(factor=1):
    match factor:
        case 1:
            return '[m]'
        case 1e2:
            return '[cm]'
        case 1e3:
            return '[mm]'
        case 1e6:
            return r'[$\mu$m]'
        case 1e9:
            return '[nm]'
        case _:
            return ''
def coord_label(coord,norm):
    if coord in ['x','y','z','ux','uy','uz','gamma']:
        match coord:
            case 'x':
                return 'x '+length_um(norm)
            case 'y':
                return 'y '+length_um(norm)
            case 'z':
                return 'z '+length_um(norm)
            case 'ux':
                return r'$u_x$ [$m_e c$]'
            case 'uy':
                return r'$u_y$ [$m_e c$]'
            case 'uz':
                return r'$u_z$ [$m_e c$]'
            case 'gamma':
                if norm == m_e*c**2/e:
                    return r'$\mathcal{E}$ [eV]'
                elif norm == m_e*c**2/e*1e-3 or norm == 511:
                    return r'$\mathcal{E}$ [keV]'
                elif norm == m_e*c**2/e*1e-6 or norm == 0.511:
                    return r'$\mathcal{E}$ [MeV]'
                elif norm == m_e*c**2/e*1e-9 or norm == 0.000511:
                    return r'$\mathcal{E}$ [GeV]'
                else:
                    return r'$\gamma$'
    elif 'div' in coord:
        Dir = coord.split('_')[-1]
        match norm:
            case 1:
                return f'$\\theta_{{{Dir}}}$'
            case 1e3:
                return f'$\\theta_{{{Dir}}}$ [mrad]'
            case 1e6:
                return f'$\\theta_{{{Dir}}}$ [$\\mu$rad]'
            case 1e9:
                return f'$\\theta_{{{Dir}}}$ [nrad]'
            case _:
                return f'$\\theta_{{{Dir}}}$'
    elif 'beta' in coord:
        Dir = coord.split('_')[-1]
        return f'$\\beta_{{{Dir}}}$'