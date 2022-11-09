"""
Set of functions by FA

"""
# Import section
import numpy as np
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
import json
from scipy.constants import e, m_e, c, pi

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
        div = np.sqrt((px**2+py**2)/pz**2)
    else:
        div = px/pz

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


class Diag(object):

    """
    A class to handle diagnostics of a plasma simulation;
    pass the path of hd5f files
    """
    plt.ion()

    def __init__(self, path):
        self.ts = OpenPMDTimeSeries(path+'/diags/hdf5')
        self.params = json.load(open(path+'/params.json'))
        self.iterations = self.ts.iterations
        self.t = self.ts.t
        self.avail_fields = self.ts.avail_fields
        self.avail_geom = self.ts.avail_geom
        self.avail_species = self.ts.avail_species
        self.avail_record_components = self.ts.avail_record_components
        self.avail_bunch_prop = ['ph_emit_n', 'tr_emit', 'beam_size',
                                 'momenta_spread', 'divergence', 'solid_div', 'charge', 'mean_energy',
                                 'en_spread', 'tw_alpha', 'tw_beta', 'tw_gamma']

    def __normalize__(self, field_name, coord, N):
        if N is None:
            n_e = self.params['n_e']
            omega0 = self.params['omega0']
            omegap = self.params['omegap']

            if field_name == 'rho':
                N = -e*n_e
            elif field_name == 'J':
                N = e*n_e*c
            elif field_name == 'E':
                if coord == 'z':
                    N = m_e*omegap*c/e
                elif coord in ['x', 'y', 'r', 't']:
                    N = m_e*omega0*c/e
            elif field_name == 'B':
                N = m_e*omega0/e
            elif field_name == 'phi':
                N = m_e*c**2/e
            elif field_name == 'force':
                N = m_e*omegap*c
        return N

    def __comoving_selection__(self, i, time, select):
        v_w = self.params['v_window']
        selection = select.copy()
        selection['z'] = [select['z'][0]+(v_w*(i-time))*1e6,
                          select['z'][1]+(v_w*(i-time))*1e6]
        return selection

    def __potential__(self, iteration, theta=0, m='all'):
        """
        Method to integrate electrostatic potential from longitudinal field Ez.

        **Parameters**
            iteration: int
                The same as usual
            theta, m:
                Same parameters of .get_field() method.
                Same defaults (0, 'all')
        """
        Ez, info_e = self.ts.get_field('E', coord='z', iteration=iteration, theta=theta, m=m)
        phi = np.zeros_like(Ez)
        max = Ez.shape[1]
        for i in range(max-2, -1, -1):
            phi[:, i] = np.trapz(Ez[:, i:i+2], dx=info_e.dz) + phi[:, i+1]
        return phi, info_e

    def __force__(self, coord, iteration, theta=0, m='all'):
        """
        Method to calculate transverse components of force .

        **Parameters**
            coord: str
                Which component 'x' or 'y'
            iteration: int
                The same as usual
            theta, m:
                Same parameters of .get_field() method.
                Same defaults (0, 'all')
        """

        if coord == 'x':
            E, info_e = self.ts.get_field('E', 'x', iteration=iteration, m=m, theta=theta)
            B, info_b = self.ts.get_field('B', 'y', iteration=iteration, m=m, theta=theta)
            del info_b
            F = e*(E - c*B)
        elif coord == 'y':
            E, info_e = self.ts.get_field('E', 'y', iteration=iteration, m=m, theta=theta)
            B, info_b = self.ts.get_field('B', 'x', iteration=iteration, m=m, theta=theta)
            del info_b
            F = e*(E + c*B)
        elif coord == 'r':
            E, info_e = self.ts.get_field('E', 'r', iteration=iteration, m=m, theta=theta)
            B, info_b = self.ts.get_field('B', 't', iteration=iteration, m=m, theta=theta)
            del info_b
            F = e*(E - c*B)
        elif coord == 't':
            E, info_e = self.ts.get_field('E', 't', iteration=iteration, m=m, theta=theta)
            B, info_b = self.ts.get_field('B', 'r', iteration=iteration, m=m, theta=theta)
            del info_b
            F = e*(E + c*B)
        else:
            raise ValueError("You must specify a force component in \n"
                             "\t\a 'x', 'y', 'r' or 't' direction for 'coord'")
        return F, info_e

    def slice_emit(self, N, select=None, species=None, iteration=None,
                    plot=False, components=['x','ux'], mask=0., trans_space='x',
                    z0=0., norms=[1.,1.], **kwargs):
        """
        Function to calculate slice emittances of a 'N sliced' bunch

        **Parameters**

            N: int
                Number of slices
            select: dict or ParticleTracker object, optional
                - If `select` is a dictionary:
                then it lists a set of rules to select the particles, of the form
                'x' : [-4., 10.]   (Particles having x between -4 and 10 meters)
                'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
                'uz' : [5., None]  (Particles with uz above 5 mc)
                - If `select` is a ParticleTracker object:
                then it returns particles that have been selected at another
                iteration ; see the docstring of `ParticleTracker` for more info.
            species: string
                A string indicating the name of the species
                This is optional if there is only one species
            iteration : int
                The iteration at which to obtain the data
                Default is first iteration
            plot: bool
                If 'True' returns the plot. Default is 'False'.
            components: list of str
                The components of phase-space to plot. Default is 'x','ux'.
            mask: float
                A value to mask undesired points in plot.
            trans_space: str
                Transverse phase space to consider in calculation: 'x' or 'y'; default is 'x'
            z0: float
                If 'z' is in 'components' the z axis is transformed to z+z0.
                Default is z0=0; to be set in meters.
            norms: list of floats
                A list of two float constants to multiply the values of 'components' for normalization;
                consider that positions are in meters.
                Default is [1.,1.].
            **kwargs
                Parameters of .pcolormesh method.

        **Returns**

            S_prop: dictionary
                Properties of each slice:
                ph_emit_n, beam_size, momenta_spread and mean position of each slice
            dz: float
                Longitudinal slices' thickness.
        """
        if trans_space == 'y':
            A = 'y'
            B = 'uy'
        else:
            A = 'x'
            B = 'ux'
        x, ux, z, w = self.ts.get_particle([A, B, 'z', 'w'], select=select, iteration=iteration, species=species)
        dz = (z.max() - z.min())/N

        s_emit = np.zeros(N)
        s_sigma_x = np.zeros(N)
        s_sigma_ux = np.zeros(N)
        Z = np.zeros(N)

        a = z.argsort()
        x = x[a]
        ux = ux[a]
        w = w[a]
        z.sort()

        for n in range(N):
            inds = np.where((z >= z.min()+n*dz) &
                            (z <= z.min()+(n+1)*dz))[0]

            Z[n] = mean(z[inds], w[inds])

            s_emit[n] = emittance(x[inds], ux[inds], w[inds])
            s_sigma_x[n] = central_average(x[inds], w[inds])
            s_sigma_ux[n] = central_average(ux[inds], w[inds])

        S_prop = {'s_emit': s_emit, 's_sigma_x': s_sigma_x,
                  's_sigma_ux': s_sigma_ux, 'z': Z}
        if plot:
            cmap = ['Reds', 'Blues', 'Greens']
            bins = 1000
            density = True
            alpha = 1

            if 'cmap' in kwargs:
                cmap = kwargs['cmap']
                del kwargs['cmap']

            if 'bins' in kwargs:
                bins = kwargs['bins']
                del kwargs['bins']

            if 'density' in kwargs:
                density = kwargs['density']
                del kwargs['density']

            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
                del kwargs['alpha']

            if 'div_x' in components:
                if components.index('div_x') == 0:
                    px, pz, comp2, weight = \
                        self.ts.get_particle(['ux', 'uz', components[1], 'w'], iteration=iteration,
                                             select=select, species=species)
                    comp1 = divergence(px=px, pz=pz)
                else:
                    px, pz, comp1, weight = \
                        self.ts.get_particle(['ux', 'uz', components[0], 'w'], iteration=iteration,
                                             select=select, species=species)
                    comp2 = divergence(px=px, pz=pz)
            elif 'div_y' in components:
                if components.index('div_y') == 0:
                    px, pz, comp2, weight = \
                        self.ts.get_particle(['uy', 'uz', components[1],'w'], iteration=iteration,
                                             select=select, species=species)
                    comp1 = divergence(px=px, pz=pz)
                else:
                    px, pz, comp1, weight = \
                        self.ts.get_particle(['uy', 'uz', components[0],'w'], iteration=iteration,
                                             select=select, species=species)
                    comp2 = divergence(px=px, pz=pz)
            elif 'div2' in components:
                if components.index('div2') == 0:
                    px, py, pz, comp2, weight = \
                        self.ts.get_particle(['ux', 'uy', 'uz', components[1],'w'], iteration=iteration,
                                             select=select, species=species)
                    comp1 = divergence(px=px, py=py, pz=pz)
                else:
                    px, py, pz, comp1, weight = \
                        self.ts.get_particle(['ux', 'uy', 'uz', components[0],'w'], iteration=iteration,
                                             select=select, species=species)
                    comp2 = divergence(px=px, py=py, pz=pz)
            else:
                comp1, comp2, weight = \
                    self.ts.get_particle([components[0], components[1], 'w'],
                                         iteration=iteration, select=select,
                                         species=species)

            if 'z' in components and z0:
                if components.index('z') == 0:
                    comp1 += z0
                else:
                    comp2 += z0
            comp1 = comp1[a]
            comp2 = comp2[a]
            weight = weight[a]
            for n in range(-1, 2):
                inds = np.where((z >= mean(z,w)+n*central_average(z,w)-dz/2) &
                                (z <= mean(z,w)+n*central_average(z,w)+dz/2))
                X = comp1[inds]
                UX = comp2[inds]
                weight = w[inds]
                H, xedge, yedge = \
                    np.histogram2d(X, UX,
                                   bins=bins, weights=weight,
                                   density=density)
                H = H.T
                X, Y = np.meshgrid(xedge, yedge)
                H = np.ma.masked_where(H <= mask, H)
                plt.pcolormesh(X*norms[0], Y*norms[1], H, cmap=cmap[n+1], alpha=alpha,**kwargs)
        return S_prop, dz

    def slice_analysis(self, dz, n_slice, prop, trans_space='x', species=None, select=None, norm_z=1.):
        """
        Function to calculate 'prop' evolution of 'n_slice' slices of width 'dz'

        **Parameters**
            dz: float
                Width, in meters, of each slice
            n_slice: list of floats
                A list of floats indicating which slices are considered respect the
                z_mean position of selected bunch, in units of sigma_z,; z_mean is
                calculated for each iteration, e.g:
                    - for n_slice ranging from m to n, this function make computation for slices
                      centered  in (z_mean + m*sigma_z), ..., (z_mean + n*sigma_z), calculating
                      z_mean and sigma_z every iteration
            prop: str
                This sets which property to be calculated
                You can choose from the following list:
                    - ph_emit_n (normalized phase emittance)
                    - tr_emit (trace emittance)
                    - beam_size
                    - momenta_spread
                    - divergence
                    - solid_div
                    - charge
                    - mean_energy
                    - en_spread (energy spread)
                    - tw_alpha, tw_beta, tw_gamma (Twiss parameters)
            trans_space: str
                'x' or 'y' transverse phase space; default is 'x'
            select: dict or ParticleTracker object, optional
              - If `select` is a dictionary:
              then it lists a set of rules to select the particles, of the form
              'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
              'x' : [-4., 10.]   (Particles having x between -4 and 10 meters)
              'uz' : [5., None]  (Particles with uz above 5 mc)
              - If `select` is a ParticleTracker object:
              then it returns particles that have been selected at another
              iteration ; see the docstring of `ParticleTracker` for more info.
            species: string
                A string indicating the name of the species
                This is optional if there is only one species
            norm_z: float
                Constant to multiply z-axis for normalization; set in meters
        **Returns**
            Z: ndarray
                Array of shape (len(n_slice),len(iterations)), each raw corresponding to
                slices 
            a: ndarray 
                Array of prop values, each raw corrisponding to slices
        """         
        if prop not in self.avail_bunch_prop:
            p = '\n -'.join(self.avail_bunch_prop)
            raise ValueError(str(prop) + " is not an available property.\n"
                             "Available properties are:\n -{:s}\nTry again".format(p))
        A, B = 'x', 'ux'
        if trans_space == 'y':
            A = 'y'
            B = 'uy'
        a=np.zeros((len(n_slice),len(self.iterations)))
        Z=np.zeros_like(a)
        ptcl_percent = self.params['subsampling_fraction']
        for i,t in enumerate(self.iterations):
            z, w = self.ts.get_particle(['z','w'], select=select, iteration=t, species=species)
            z_mean = mean(z,w)
            sigma_z = central_average(z,w)
            for j,n in enumerate(n_slice):
                if select is None:
                    selection = {'z':[z_mean+n*sigma_z-dz/2,z_mean+n*sigma_z+dz/2]}
                else:
                    selection = select.copy()
                    selection['z'] = [z_mean+n*sigma_z-dz/2,z_mean+n*sigma_z+dz/2]
                if prop == 'beam_size':
                    x = self.ts.get_particle([A], species=species, select=selection, iteration=t)[0]
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    a[j,i] = central_average(x,W)
                    Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'momenta_spread':
                    x = self.ts.get_particle([B], species=species, select=selection, iteration=t)[0]
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    a[j,i] = central_average(x,W)
                    Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'divergence':
                    ux, uz = self.ts.get_particle([B,'uz'], species=species, select=selection, iteration=t)
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    slope = divergence(px=ux,pz=uz)
                    a[j,i] = central_average(slope,W)
                    Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'solid_div':
                    ux, uy, uz = self.ts.get_particle(['x','uy','uz'], species=species, select=selection, iteration=t)
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    slope = divergence(px=ux,py=uy,pz=uz)
                    a[j,i] = central_average(slope,W)
                    Z[j,i] = z_mean+n*sigma_z                
                    continue
                if prop == 'ph_emit_n':
                    x, ux = self.ts.get_particle([A,B], species=species, select=selection, iteration=t)
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    a[j,i] = emittance(x, ux, W)
                    Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'mean_energy':
                    gamma = self.ts.get_particle(['gamma'], species=species, select=selection, iteration=t)[0]
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    a[j,i] = mean(gamma, W, energy=True)
                    Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'en_spread':
                    gamma = self.ts.get_particle(['gamma'], species=species, select=selection, iteration=t)[0]
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    a[j,i] = energy_spread(gamma,W)
                    Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'charge':
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    if W.sum() == 0:
                        a[j,i] = np.nan
                        Z[j,i] = z_mean+n*sigma_z
                        pass
                    else:
                        a[j,i] = e*W.sum()/ptcl_percent
                        Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'tr_emit':
                    x, ux, uz = self.ts.get_particle([A,B,'uz'], iteration=t, select=selection, species=species)
                    slope = divergence(px=ux, pz=uz)
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    a[j,i] = emittance(x, slope, W)
                    Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'tw_alpha':
                    x, ux, uz = self.ts.get_particle([A,B,'uz'], iteration=t, select=selection, species=species)
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    a[j,i] = twiss(x, ux, uz, W, 'alpha')
                    Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'tw_beta':
                    x, ux, uz = self.ts.get_particle([A,B,'uz'], iteration=t, select=selection, species=species)
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    a[j,i] = twiss(x, ux, uz, W, 'beta')
                    Z[j,i] = z_mean+n*sigma_z
                    continue
                if prop == 'tw_gamma':
                    x, ux, uz = self.ts.get_particle([A,B,'uz'], iteration=t, select=selection, species=species)
                    inds = np.where((z>=z_mean+n*sigma_z-dz/2) & (z<=z_mean+n*sigma_z+dz/2))[0]
                    W = w[inds]
                    a[j,i] = twiss(x, ux, uz, W, 'gamma')
                    Z[j,i] = z_mean+n*sigma_z
                    continue
        return Z*norm_z, a

    def lineout(self, field_name, iteration,
                coord=None, theta=0, m='all',
                normalize=False, A0=None, slicing='z',
                on_axis=None, z0=0., norm_z=1., **kwargs):
        """
        Method to get a lineout plot of passed field_name

        **Parameters**

            field_name: string
                    Field to plot
            iteration: int
                    The same as usual
            coord, theta, m: same parameters of .get_field() method.
                    Same defaults (None, 0, 'all')
            normalize: bool, optional
                    If normalize=True this 'turns on' the normalization.
                    Default is 'False'.
            A0: float, optional
                    If normalize=True this allows to set the normilizing costant.
                    Default is 'None: in this case normalization is set to
                    usual units, e.g:
                    - e*n_e for charge density 'rho'; this returns normalized density
                    - m_e*c*omega_0/e for transverse 'E'
                    - m_e*c*omega_p/e for longitudinal 'E'
            slicing: str, optional
                    This sets the slicing along the chosen direction ('z' or 'r').
                    Default is 'z'.
            on_axis: float, in meters
                    Coord in meters of slicing line along the chosen direction.
                    Default is 'r' = '0.' or 'z' = mid of the z-axis
            z0: float, optional
                    Transforms z coords into z+z0 coords; to be set in meters.
                    Deafult is z0=0.
            norm_z: float
                    Constant to multiply x-axis for normalization to be set in meters^-1,
                    or changing the order of magnitude (e.g. multiply for 1.e6 to set microns).
            **kwargs: keywords to pass to .pyplot.plot() function
        """
        if field_name == 'phi':
            E, info_e = self.__potential__(iteration, theta=theta, m=m)
        elif field_name == 'force':
            E, info_e = self.__force__(coord, iteration, theta, m)
        else:
            E, info_e = self.ts.get_field(field=field_name, coord=coord,
                                          iteration=iteration, theta=theta, m=m)
        if slicing == 'z':
            if on_axis is None:
                on_axis = 0.
            N = self.params['Nr'] + int(on_axis*1.e-6/info_e.dr)
            E = E[N, :]
            z = info_e.z
            if z0:
                z = info_e.z+z0
        else:
            if on_axis is None:
                on_axis = info_e.z[int(self.params['Nz']/2)]
            N = int(self.params['Nz']/2) + int((on_axis-info_e.z[int(self.params['Nz']/2)])/info_e.dz)
            E = E[:, N]
            z = info_e.r

        E0 = 1
        if normalize:
            E0 = self.__normalize__(field_name, coord, A0)

        plt.plot(z*norm_z, E/E0, **kwargs)

    def map(self, field_name, iteration,
            coord=None, theta=0, m='all', normalize=False, A0=None, 
            z0=0., norms=[1.,1.], **kwargs):
        """
        Method to get a 2D-map of passed field_name

        **Parameters**

            field_name: string
                    Field to plot
            iteration: int
                    The same as usual
            coord, theta, m: same parameters of .get_field() method.
                         Same defaults (None, 0, 'all')
            normalize: bool, optional;
                    If normalize=True this 'turns on' the normalization.
                    Default is 'False'.
            A0: float, optional;
                    If normalize=True this allows to set the normalizing
                    constant.
                    Default is 'None: in this case normalization is set to
                    usual units, e.g:
                    - e*n_e for charge density 'rho'; this returns normalized
                      density
                    - m_e*c*omega_0/e for transverse 'E'
                    - m_e*c*omega_p/e for longitudinal 'E'
            z0: float, optional
                    Transforms z coords into z+z0 coords; to be set in meters.
                    Deafult is z0=0.
            norms: list of floats
                    A list of two float constants to multiply the values
                    of both axis for normalization or magnitude changings; 
                    norms[0] for z-axis, norms[1] for r-axis.
                    Set in meters^-1; default is [1.,1.].
            **kwargs: keywords to pass to .Axes.imshow() method
        """
        if field_name == 'phi':
            E, info_e = self.__potential__(iteration, theta=theta, m=m)
        elif field_name == 'force':
            E, info_e = self.__force__(coord, iteration, theta, m)
        else:
            E, info_e = self.ts.get_field(field=field_name, coord=coord,
                                          iteration=iteration, theta=theta, m=m)

        E0 = 1
        if normalize:
            E0 = self.__normalize__(field_name, coord, A0)

        origin = 'lower'
        if 'origin' in kwargs:
            origin = kwargs['origin']
            del kwargs['origin']
        extent = info_e.imshow_extent.copy()
        if z0:
            extent[0:2]+=z0
        extent[0:2]*=norms[0]
        extent[2:4]*=norms[1]
        plt.imshow(E/E0, extent=extent,
                  origin=origin, **kwargs)

    def transverse_map(self, field_name, iteration, coord=None,
            m='all', normalize=False, A0=None,
            z_pos=None, swap_axis=False, norms=[1.,1.], **kwargs):
        """
        Method to get a 2D-transverse map of passed field_name
        in x-y  or y-x plane

        **Parameters**

            field_name: string
                    Field to plot
            coord, m: same parameters of .get_field() method.
                         Same defaults (None, 0, 'all')
            iteration: int
                    The same as usual
            normalize: bool, optional;
                    If normalize=True this 'turns on' the normalization.
                    Default is 'False'.
            A0: float, optional;
                    If normalize=True this allows to set the normalizing
                    constant.
                    Default is 'None: in this case normalization is set to
                    usual units, e.g:
                    - e*n_e for charge density 'rho'; this returns normalized
                      density
                    - m_e*c*omega_0/e for transverse 'E'
                    - m_e*c*omega_p/e for longitudinal 'E'
            z_pos: float, optional
                    Choose the actual z-position where to slice the considered field_name;
                    to be set in meters. Default is the first slice.
            swap_axis: bool
                    Whether to plot in x-y or y-x plane with inverted y-axis; default is x-y (False)
            norms: list of floats
                    A list of two float constants to multiply the values
                    of both axis for normalization or magnitude changings; 
                    norms[0] for z-axis, norms[1] for r-axis.
                    Set in meters^-1; default is [1.,1.].
            **kwargs: keywords to pass to .pcolormesh() method
        """
        Nr = self.params['Nr']
        test_field = self.avail_fields[0]
        if self.ts.fields_metadata[test_field]['type'] == 'vector':
            test_coord =  'x'
        else:
            test_coord = None
        info = self.ts.get_field(test_field,test_coord,iteration=iteration)[1]
        dz = info.dz
        dr = info.dr
        if z_pos == None:
            z_pos=info.zmin
        if z_pos < info.zmin or z_pos > info.zmax:
            raise ValueError('Ehi, watch out!\n'
                              'z_pos = {:f}  cannot be less than {:f}'
                              'or greater than {:f} meters'.format(z_pos,info.zmin,info.zmax))
        nz = int((z_pos-info.zmin)/dz+0.5)
        theta = np.linspace(0,2*pi*(1+1/Nr),Nr+1)
        r = np.insert(info.r[Nr:],0,0.)
        field = np.zeros([Nr,Nr])

        for i,T in enumerate(theta[:-1]):
            if field_name == 'phi':
                E = self.__potential__(iteration, theta=T, m=m)[0]
                field[:,i] = E[Nr:,nz].copy()
            elif field_name == 'force':
                E = self.__force__(coord, iteration, theta=T, m=m)[0]
                field[:,i] = E[Nr:,nz].copy()
            else:
                E = self.ts.get_field(field=field_name, coord=coord,
                                          iteration=iteration, theta=T, m=m)[0]
                field[:,i] = E[Nr:,nz].copy()
        del E

        E0 = 1
        if normalize:
            E0 = self.__normalize__(field_name, coord, A0)
        field /= E0
        Theta, R = np.meshgrid(theta,r)
        X, Y = R*np.cos(Theta), R*np.sin(Theta)
        X*=norms[0]
        Y*=norms[1]
        if swap_axis:
            plt.pcolormesh(Y,X,field,**kwargs)
            ax=plt.gca()
            ax.invert_xaxis()
        else:
            plt.pcolormesh(X,Y,field,**kwargs)

    def bunch_properties_evolution(self, select, properties, species=None, trans_space='x',
                                    zeta_coord=False, time=0., t_lim=False, plot_over=False,
                                    norm_z=1, Norm=1., **kwargs):
        """
        Method to select a bunch and to plot the evolution of
        its characteristics along propagation length

        **Parameters**

            select: dict or ParticleTracker object, optional
              - If `select` is a dictionary:
              then it lists a set of rules to select the particles, of the form
              'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
              'x' : [-4., 10.]   (Particles having x between -4 and 10 meters)
              'uz' : [5., None]  (Particles with uz above 5 mc)
              - If `select` is a ParticleTracker object:
              then it returns particles that have been selected at another
              iteration ; see the docstring of `ParticleTracker` for more info.
              - If 'select' contains 'z' and 'zeta_coord'='True':
              selection is made in co-moving frame
            properties: list of str
                This sets which properties will be plotted in order you set the list.
                You can choose from the following list:
                    - ph_emit_n (normalized phase emittance)
                    - tr_emit (trace emittance)
                    - beam_size 
                    - momenta_spread
                    - divergence
                    - solid_div
                    - charge
                    - mean_energy
                    - en_spread (energy spread)
                    - tw_alpha, tw_beta, tw_gamma (Twiss parameters)
            species: string
                A string indicating the name of the species
                This is optional if there is only one species
            trans_space: str
                'x' or 'y' transverse phase space; default is 'x'
            zeta_coord: bool
                If 'True'and 'z' in 'select', the 'z' selection
                is done in co-moving frame
            time: float
                Specify the time (s) at which refers the 'z' selection
                when activated the co-moving frame.
                Default is the first iteration.
            t_lim: list of two floats 
                Set the time window you want to plot
                Use self.t values as limits (s); default is all time_series
            plot_over: bool
                If you want to plot all properties in the same graph
                Default is 'False'
            norm_z: float
                Multiplying constant to normalize z-axis; set in meters
            Norm: float
                Multiplying constant to set properties normalization
            **kwargs: keyword to pass to .pyplot.plot()

        """
        ptcl_percent = self.params['subsampling_fraction']
        if not t_lim:
            t_lim = [self.t.min(), self.t.max()]
        inds = np.where((self.t >= t_lim[0]) & (self.t <= t_lim[1]))
        t = self.t[inds]
        a = np.zeros_like(t)
        Z = np.zeros_like(t)
        if trans_space == 'y':
            A = 'y'
            B = 'uy'
        else:
            A = 'x'
            B = 'ux'
        for p in properties:
            if p not in self.avail_bunch_prop:
                prop = '\n -'.join(self.avail_bunch_prop)
                print(str(p) + " is not an available property.\n"
                                 "Available properties are:\n -{:s}\nTry again".format(prop))
            else:
                if zeta_coord and ('z' in select):
                    
                    if time != 0.:
                        time = time
                    else:
                        time = self.t[0]
                                        
                    for k, i in enumerate(t):
                        selection = self.__comoving_selection__(i, time, select)
                        z, w = self.ts.get_particle(['z', 'w'],
                                                 t=i, select=selection,
                                                 species=species)
                        Z[k] = mean(z,w)
                        if p == 'ph_emit_n':
                            x, ux = self.ts.get_particle([A, B],t=i,
                                    select=selection,species=species)                        
                            a[k] = emittance(x, ux, w)
                            continue
                        if p == 'beam_size':
                            x = self.ts.get_particle([A], t=i, select=selection, species=species)[0]
                            a[k] = central_average(x, w)
                            continue
                        if p == 'momenta_spread':
                            ux = self.ts.get_particle([B], t=i, select=selection, species=species)[0]
                            a[k] = central_average(ux, w)
                            continue
                        if p == 'divergence':
                            ux, uz = self.ts.get_particle([B,'uz'], t=i, select=selection, species=species)
                            slope = divergence(px=ux,pz=uz)
                            a[k] = central_average(slope,w)
                            continue
                        if p == 'solid_div':
                            ux, uy, uz = self.ts.get_particle(['ux','uy','uz'], t=i, select=selection, species=species)
                            solid = divergence(ux,uy,uz)
                            a[k] = central_average(solid,w)
                            continue
                        if p == 'charge':
                            if w.sum() == 0:
                                pass
                            else:
                                a[k] = e*w.sum()/ptcl_percent
                            continue
                        if p == 'mean_energy':
                            gamma = self.ts.get_particle(['gamma'], t=i, select=selection, species=species)[0]
                            a[k] = mean(gamma,w,energy=True)
                            continue
                        if p == 'en_spread':
                            gamma = self.ts.get_particle(['gamma'], t=i, select=selection, species=species)[0]
                            a[k] = energy_spread(gamma, w)
                            continue
                        if p == 'tr_emit':
                            x, ux, uz = self.ts.get_particle([A,B,'uz'], t=i, select=selection, species=species)
                            slope = divergence(px=ux, pz=uz)
                            a[k] = emittance(x, slope, w)
                            continue
                        if p == 'tw_alpha':
                            x, ux, uz = self.ts.get_particle([A,B,'uz'], t=i, select=selection, species=species)
                            a[k] = twiss(x, ux, uz, w, 'alpha')
                            continue
                        if p == 'tw_beta':
                            x, ux, uz = self.ts.get_particle([A,B,'uz'], t=i, select=selection, species=species)
                            a[k] = twiss(x, ux, uz, w, 'beta')
                            continue
                        if p == 'tw_gamma':
                            x, ux, uz = self.ts.get_particle([A,B,'uz'], t=i, select=selection, species=species)
                            a[k] = twiss(x, ux, uz, w, 'gamma')
                            continue
                    if plot_over and (len(properties) == 1):
                        plt.plot(Z*norm_z, a*Norm, **kwargs)
                    else:        
                        plt.figure()
                        plt.title(p)
                        plt.plot(Z*norm_z, a*Norm, **kwargs)

                else:
                    for k, i in enumerate(t):
                        selection = select
                        z, w = self.ts.get_particle(['z', 'w'],
                         t=i, select=selection,
                         species=species)
                        Z[k] = mean(z,w)
                        if p == 'ph_emit_n':
                            x, ux = self.ts.get_particle([A, B],t=i,
                                    select=selection,species=species)                        
                            a[k] = emittance(x, ux, w)
                            continue
                        if p == 'beam_size':
                            x = self.ts.get_particle([A], t=i, select=selection, species=species)[0]
                            a[k] = central_average(x, w)
                            continue
                        if p == 'momenta_spread':
                            ux = self.ts.get_particle([B], t=i, select=selection, species=species)[0]
                            a[k] = central_average(ux, w)
                            continue
                        if p == 'divergence':
                            ux, uz = self.ts.get_particle([B,'uz'], t=i, select=selection, species=species)
                            slope = divergence(px=ux,pz=uz)
                            a[k] = central_average(slope,w)
                            continue
                        if p == 'solid_div':
                            ux, uy, uz = self.ts.get_particle(['ux','uy','uz'], t=i, select=selection, species=species)
                            solid = divergence(ux,uy,uz)
                            a[k] = central_average(solid,w)
                            continue
                        if p == 'charge':
                            if w.sum() == 0:
                                pass
                            else:
                                a[k] = e*w.sum()/ptcl_percent
                            continue
                        if p == 'mean_energy':
                            gamma = self.ts.get_particle(['gamma'], t=i, select=selection, species=species)[0]
                            a[k] = mean(gamma,w,energy=True)
                            continue
                        if p == 'en_spread':
                            gamma = self.ts.get_particle(['gamma'], t=i, select=selection, species=species)[0]
                            a[k] = energy_spread(gamma, w)
                            continue
                        if p == 'tr_emit':
                            x, ux, uz = self.ts.get_particle([A,B,'uz'], t=i, select=selection, species=species)
                            slope = divergence(px=ux, pz=uz)
                            a[k] = emittance(x, slope, w)
                            continue
                        if p == 'tw_alpha':
                            x, ux, uz = self.ts.get_particle([A,B,'uz'], t=i, select=selection, species=species)
                            a[k] = twiss(x, ux, uz, w,'alpha')
                            continue
                        if p == 'tw_beta':
                            x, ux, uz = self.ts.get_particle([A,B,'uz'], t=i, select=selection, species=species)
                            a[k] = twiss(x, ux, uz, w,'beta')
                            continue
                        if p == 'tw_gamma':
                            x, ux, uz = self.ts.get_particle([A,B,'uz'], t=i, select=selection, species=species)
                            a[k] = twiss(x, ux, uz, w,'gamma')
                            continue

                    if plot_over and (len(properties) == 1):
                        plt.plot(Z*norm_z, a*Norm, **kwargs)
                    else:        
                        plt.figure()
                        plt.title(p)
                        plt.plot(Z*norm_z, a*Norm,**kwargs)

    def spectrum(self, component, iteration, select=None, species=None,
                 output=False, energy=False, charge=False, Z=1, **kwargs):
        """
        Method to easily get an energy spectrum of 'selected' particles

        **Parameters**

            
            component: str
                Choose a component in .avail_recorded_components
                to do the weighted distibution of that quantity
            iteration: int
                Which iteration we need
            select: dictionary or ParticleTracker instance
                Particle selector
            species: str, optional
                Default is the first available species
            output: bool, optional, default: 'False'
                If 'True' returns the values of histogram and bins
                edges; length of bins array is nbins+1 
                (lefts edges and right edge of the last bin).
            energy: bool, optional
                If 'True' this sets the x-axis on energy(MeV),
                otherwise x-axis has dimensionless gamma values.
                Default is 'False'.
            charge: bool, optional
                If True this sets the y-axis on dQ/dcomp values.
                Default is False, that means setting y-axis on dN/dcomp values
            Z: int
                The atomic number of ion; default is 1.
            **kwargs: keyword to pass to .hist() method; in kwargs['text_pos'] you can also
                    set the position of text inset in 'figure' frame [(0.,1.),(0.,1.)].
                    Default is [0.7,0.7].

        **Returns**

            values, bins: np.arrays
                If 'output' is True, arrays with bins values and bins edges.

        """
        in_ptcl_percent = 1/self.params['subsampling_fraction']
        pos = [0.7, 0.7]
        bins = 300
        comp, w = self.ts.get_particle([component, 'w'], iteration=iteration,
                                        species=species, select=select)
        tot_charge = w.sum()*Z*e*in_ptcl_percent
        q = 1/(Z*e)

        if charge:
            q = 1

        if 'density' in kwargs:
            del kwargs['density']

        if 'weights' in kwargs:
            del kwargs['weights']

        if 'text_pos' in kwargs:
            pos = kwargs['text_pos']
            del kwargs['text_pos']

        if 'bins' in kwargs:
            bins = kwargs['bins']
            del kwargs['bins']

        if (component=='gamma' and energy):
            a = 0.511        
            es = energy_spread(comp, w)
            me = mean(comp, w, energy=True)
        else:
            a = 1
            es = central_average(comp, w)
            me = mean(comp, w)

        values, bins = np.histogram(a*comp, bins=bins, weights=w, density=True)
        values, bins, patches = plt.hist(bins[:-1], bins, weights=values*q*tot_charge, **kwargs)
        del patches
        
        fig = plt.gcf()
        if fig.texts:
            fig.texts[0].remove()
        if (component=='gamma' and energy):
            plt.figtext(pos[0], pos[1], "Total charge is {:.1e} C\n"
                                        "Mean energy is {:.2f} MeV\n"
                                        "Energy spread is {:3.1f} %".format(tot_charge, me, es*100))
        else:
            plt.figtext(pos[0], pos[1], "Total charge is {:.1e}C\n"
                                        "Mean is {:.2f}\n"
                                        "Standar deviation is {:.1f}".format(tot_charge,me,es))
        if output:
            return values, bins

    def phase_space_hist(self, species, iteration, components=['z','uz'],
                         select=None, z0=0., norms=[1.,1.], charge=False,
                         mask=0., Z=1., **kwargs):
        """
        Method that plots a 2D histogram of the particles phase space.

        Parameters:
        ----------
            species: str
                Select the particle specie among the available ones
                (Check them out in avail_species)
            iteration: int
                Selected iteration
            components: list, str
                List of phase space components of the phase space plot
                (Check the available components in avail_record_components)
                You can plot also the 'divergence' method outputs:
                    -'div_x' is for div along planar slice x-z
                    -'div_y' is for div along planar slice y-z
                    -'div2' is for total div
            select: dict
                Particle selector
            z0: float
                If 'z' is in 'components' the z axis is transformed to z+z0.
                Default is z0=0; to be set in meters.
            norms: list of floats
                A list of two float constants to multiply the values
                of 'components' for normalization; consider that positions are in meters.
                Default is [1.,1.].
            charge: bool, optional
                If True, sets the values of histogram to dQ/dc1dc2, otherwise dN/dc1dc2.
                Default is False.
            mask: float, optional
                A float in [0.,1.] to mask all bins with a value <= mask*max(hist_values).
            Z: float, optional
                Atomic number of the ion, default is 1.
            **kwargs:
                keywords passing to plt.pcolormesh(); in **kwargs can also be set 'density'
                to pass to np.histogram2d(); if True it plots the particle (charge, whenever
                'charge'=True) density in the 2D-phase space, if False the histogram values
                count for number of particles in bins (i.e. hist_values.sum() = weight.sum()).
                Default is True.
        """
        cmap = 'Reds'
        bins = 1000
        density = True
        alpha = 1
        q = 1

        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
            del kwargs['cmap']

        if 'bins' in kwargs:
            bins = kwargs['bins']
            del kwargs['bins']

        if 'density' in kwargs:
            density = kwargs['density']
            del kwargs['density']

        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
            del kwargs['alpha']

        if charge:
            q = Z*e

        if 'div_x' in components:
            if components.index('div_x') == 0:
                px, pz, comp2, weight = \
                    self.ts.get_particle(['ux', 'uz', components[1],'w'],iteration=iteration,
                                         select=select,species=species)
                comp1 = divergence(px=px, pz=pz)
            else:
                px, pz, comp1, weight = \
                    self.ts.get_particle(['ux', 'uz', components[0],'w'],iteration=iteration,
                                         select=select,species=species)
                comp2 = divergence(px=px, pz=pz)
        elif 'div_y' in components:
            if components.index('div_y') == 0:
                px, pz, comp2, weight = \
                    self.ts.get_particle(['uy', 'uz', components[1],'w'],iteration=iteration,
                                         select=select,species=species)
                comp1 = divergence(px=px, pz=pz)
            else:
                px, pz, comp1, weight = \
                    self.ts.get_particle(['uy', 'uz', components[0],'w'],iteration=iteration,
                                         select=select,species=species)
                comp2 = divergence(px=px, pz=pz)
        elif 'div2' in components:
            if components.index('div2') == 0:
                px, py, pz, comp2, weight = \
                    self.ts.get_particle(['ux', 'uy', 'uz', components[1],'w'],iteration=iteration,
                                         select=select,species=species)
                comp1 = divergence(px=px, py=py, pz=pz)
            else:
                px, py, pz, comp1, weight = \
                    self.ts.get_particle(['ux', 'uy', 'uz', components[0],'w'],iteration=iteration,
                                         select=select,species=species)
                comp2 = divergence(px=px, py=py, pz=pz)
        else:
            comp1, comp2, weight = \
                self.ts.get_particle([components[0], components[1], 'w'],
                                     iteration=iteration, select=select,
                                     species=species)

        if 'z' in components and z0:
            if components.index('z') == 0:
                comp1 += z0
            else:
                comp2 += z0

        if mask > 1.:
            raise ValueError("mask = {:f} can't be greater than 1.".format(mask))

        H, xedge, yedge = \
            np.histogram2d(comp1*norms[0], comp2*norms[1],
                           bins=bins, weights=weight,
                           density=density)
        H = H.T*q*weight.sum()
        H = np.ma.masked_less_equal(H,mask*H.max())
        plt.pcolormesh(xedge, yedge, H, cmap=cmap, alpha=alpha,**kwargs)
