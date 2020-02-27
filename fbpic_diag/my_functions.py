"""
Set of functions by FA

"""
# Import section
import numpy as np
import matplotlib.pyplot as plt
from opmd_viewer import OpenPMDTimeSeries
import json
from scipy.constants import e, m_e, c


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

def beam_size(x, ux, w):
    """
    Function to calculate size of a bunch.
    **Parameters**
    x, ux: two 1darrays of  phase-space coords
    w: ndarray of particles' weights
    **Returns**
    beam size: float
    """
    x_mean = np.ma.average(x, weights=w)
    sigma_x2 = np.ma.average((x-x_mean)**2, weights=w)
    beam_size = np.sqrt(sigma_x2)
    return beam_size

def momenta_spread(x, ux, w):
    """
    Function to calculate momenta spread of a bunch.
    **Parameters**
    x, ux: two 1darrays of  phase-space coords
    w: ndarray of particles' weights
    **Returns**
    momenta spread: float
    """    
    ux_mean = np.ma.average(ux, weights=w)
    sigma_ux2 = np.ma.average((ux-ux_mean)**2, weights=w)
    momenta = np.sqrt(sigma_ux2)
    return momenta

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
    sigma_x = beam_size(x, ux, w)
    sigma_ux = momenta_spread(x, ux, w)
    covariance = covar(x, ux, w)
    
    emit = np.sqrt(sigma_x**2*sigma_ux**2-covariance**2)

    return emit

def twiss(x, px, pz, w):
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
    **Returns**
    alpha, beta, gamma: float
        Twiss parameters
    """
    slope = divergence(px=px, pz=pz)
    emit = emittance(x, slope, w)
    sigma_x = beam_size(x, slope, w)
    sigma_slope = momenta_spread(x, slope, w)
    covariance = covar(x, slope, w)
    inv_emit=1/emit
    
    alpha = covariance*(-inv_emit)
    beta = sigma_x**2*inv_emit
    gamma = sigma_slope**2*inv_emit

    return alpha, beta, gamma

def mean_energy(gamma, w):
    mean = np.ma.average(gamma, weights=w)
    return mean*0.511

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
    mean2 = np.ma.average(gamma**2, weights=w)
    sigma2 = (mean2 - mean**2)/mean**2
    return np.sqrt(sigma2)


class Diag(object):

    """
    A class to handle diagnostics of a plasma simulation;
    pass the path of hd5f files
    """
    plt.ion()

    def __init__(self, path):
        self.ts = OpenPMDTimeSeries(path)
        self.params = json.load(open('/params.json'))
        self.iterations = self.ts.iterations
        self.t = self.ts.t
        self.avail_fields = self.ts.avail_fields
        self.avail_geom = self.ts.avail_geom
        self.avail_species = self.ts.avail_species
        self.avail_record_components = self.ts.avail_record_components
        self.avail_bunch_prop = ['ph_emit', 'ph_emit_n', 'tr_emit', 'tr_emit_n', 'beam_size',
                                 'momenta_spread', 'charge', 'mean_energy', 'en_spread', 'tw_alpha',
                                 'tw_beta', 'tw_gamma']

    def __normalize__(self, field_name, coord, N):
        if N is None:
            n_e = self.params['n_e']
            omega0 = self.params['omega0']
            omegap = self.params['omegap']

            if field_name == 'rho':
                N = -e*n_e
            elif field_name == 'J':
                N = -e*n_e*c
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
                N = m_e*omega0*c
        return N

    def __comoving_selection__(self, i, time, select):
        v_w = self.params['v_window']
        selection = select.copy()
        selection['z'] = [select['z'][0]+(v_w*(i-time))*1e6,
                          select['z'][1]+(v_w*(i-time))*1e6]
        return selection


    def slice_emit(self, N, plot=False, comp1='x', comp2='ux', mask=0., **kwargs):
        """
        Function to calculate slice emittances of a 'N sliced' bunch

        **Parameters**

            N: int, number of slices
            plot: bool
                If 'True' returns the plot. Default is 'False'.
            comp1, comp2: str
                The components of phase-space to plot. Default is 'x','ux'.
            mask: float
                A value to mask undesired points in plot. 
            **kwargs
                Same parameters of .get_particle(), except 'var_list'.

        **Returns**

            S_prop: dictionary
                Properties of each slice:
                emittance, size, momenta spread and mean position of each slice
            dz: float
                Longitudinal slices' thickness.
        """
        if 'var_list' in kwargs:
            raise Exception("You don't need to pass 'var_list' argument!\n" 
                            "Try again with just the others kwargs\n"
                            "of .get_particle()")

        x, ux, z, w= self.ts.get_particle(['x', 'ux', 'z', 'w'], **kwargs)
        dz = (z.max()-z.min())/N

        s_emit = np.zeros(N)
        s_sigma_x2 = np.zeros(N)
        s_sigma_ux2 = np.zeros(N)
        Z = np.zeros(N)

        a = z.argsort()
        x = x[a]
        ux = ux[a]
        w = w[a]
        z.sort()

        for n in range(N):
            inds = np.where((z >= z.min()+n*dz) &
                            (z <= z.min()+(n+1)*dz))[0]

            Z[n] = (z[inds].mean())

            s_emit[n] = emittance(x[inds], ux[inds], w[inds])
            s_sigma_x2[n] = beam_size(x[inds], ux[inds], w[inds])
            s_sigma_ux2[n] = momenta_spread(x[inds], ux[inds], w[inds])

        S_prop = {'s_emit': s_emit, 's_sigma_x2': s_sigma_x2,
                  's_sigma_ux2': s_sigma_ux2, 'z': Z}
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

            x, ux, w = self.ts.get_particle([comp1, comp2, 'w'], **kwargs)
            for n in range(-1, 2):
                inds = np.where((z >= z.mean()+n*np.sqrt(z.var())-dz/2) &
                                (z <= z.mean()+n*np.sqrt(z.var())+dz/2))
                X = x[inds]
                UX = ux[inds]
                weight = w[inds]
                H, xedge, yedge = \
                    np.histogram2d(X, UX,
                                   bins=bins, weights=weight,
                                   density=density)
                H = H.T
                X, Y = np.meshgrid(xedge, yedge)
                H = np.ma.masked_where(H <= mask, H)
                plt.pcolormesh(X, Y, H, cmap=cmap[n+1], alpha=alpha,**kwargs)
        return S_prop, dz

    def potential(self, iteration, theta=0, m='all'):
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

    def force(self, coord, iteration, theta=0, m='all'):
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
        else:
            raise ValueError("You must specify a force component in \n"
                             "\t\a 'x' or 'y' direction for 'coord'")
        return F, info_e

    def lineout(self, field_name, iteration,
                coord=None, theta=0, m='all',
                normalize=False, A0=None, slicing='z',
                on_axis=None, zeta_coord=False, **kwargs):
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
            on_axis: float, in microns
                    Coord in microns of slicing line along the chosen direction.
                    Default is 'r' = '0.' or 'z' = mid of the z-axis
            zeta_coord: bool, optional
                    If 'True' transforms z coords into z-v_w*t coords;
                    v_w is moving window velocity. Default is 'False'
            **kwargs: keywords to pass to .pyplot.plot() function

        """
        if field_name == 'phi':
            E, info_e = self.potential(iteration, theta=theta, m=m)
        elif field_name == 'force':
            E, info_e = self.force(coord, iteration, theta, m)
        else:
            E, info_e = self.ts.get_field(field=field_name, coord=coord,
                                          iteration=iteration, theta=theta, m=m)
        if slicing == 'z':
            if on_axis == None:
                on_axis = 0.
            N = self.params['Nr'] + int(on_axis*1.e-6/info_e.dr)
            E = E[N, :]
            z = info_e.z*1.e6
            if zeta_coord:
                v_w = self.params['v_window']
                t = self.ts.current_t
                z = (info_e.z-v_w*t)*1.e6
        else:
            if on_axis == None:
                on_axis = info_e.z[int(self.params['Nz']/2)]*1.e6
            N = int(self.params['Nz']/2) + int((on_axis*1.e-6-info_e.z[int(self.params['Nz']/2)])/info_e.dz)
            E = E[:, N]
            z = info_e.r*1.e6
        E0 = 1

        if normalize:
            E0 = self.__normalize__(field_name, coord, A0)

        plt.plot(z, E/E0, **kwargs)

    def map(self, field_name, iteration,
            coord=None, theta=0, m='all', normalize=False, N=None, zeta_coord=False, **kwargs):
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
            N: float, optional;
                    If normalize=True this allows to set the normalizing
                    constant.
                    Default is 'None: in this case normalization is set to
                    usual units, e.g:
                    - e*n_e for charge density 'rho'; this returns normalized
                      density
                    - m_e*c*omega_0/e for transverse 'E'
                    - m_e*c*omega_p/e for longitudinal 'E'
            zeta_coord: bool;
                    If 'True' sets the co-moving frame
            **kwargs: keywords to pass to .Axes.imshow() method
        **Return**

            ax: a matplotlib.axes.Axes instance

        """
        if field_name == 'phi':
            E, info_e = self.potential(iteration, theta=theta, m=m)
        elif field_name == 'force':
            E, info_e = self.force(coord, iteration, theta, m)
        else:
            E, info_e = self.ts.get_field(field=field_name, coord=coord,
                                          iteration=iteration, theta=theta, m=m)

        E0 = 1
        if normalize:
            E0 = self.__normalize__(field_name, coord, N)

        fig, ax = plt.subplots(1, 1)
        origin = 'low'
        if 'origin' in kwargs:
            origin = kwargs['origin']
            del kwargs['origin']
        extent = info_e.imshow_extent.copy()
        if zeta_coord:
            extent[0:2]-=c*self.ts.current_t
        ax.imshow(E/E0, extent=extent*1.e6,
                  origin=origin, **kwargs)
        fig.colorbar(ax.get_images()[0], ax=ax, use_gridspec=True)

        return fig, ax

    def bunch_properties_evolution(self, select, properties, species='electrons',
                                    zeta_coord=False, time=0., t_lim=False, plot_over=False,**kwargs):
        """
        Method to select a bunch and to plot the evolution of
        its characteristics along propagation length

        **Parameters**

            select: dict or ParticleTracker object, optional
              - If `select` is a dictionary:
              then it lists a set of rules to select the particles, of the form
              'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
              'x' : [-4., 10.]   (Particles having x between -4 and 10 microns)
              'uz' : [5., None]  (Particles with uz above 5 mc)
              - If `select` is a ParticleTracker object:
              then it returns particles that have been selected at another
              iteration ; see the docstring of `ParticleTracker` for more info.
              - If 'select' contains 'z' and 'zeta_coord'='True':
              selection is made in co-moving frame
            properties: list of str
                This sets which properties will be plotted in order you set the list.
                You can choose from the following list:
                    - ph_emit (phase emittance)
                    - ph_emit_n
                    - tr_emit (trace emittance)
                    - tr_emit_n
                    - beam_size 
                    - momenta_spread
                    - charge
                    - mean_energy
                    - en_spread (energy spread)
                    - tw_alpha, tw_beta, tw_gamma (Twiss parameters)
            species: string
                A string indicating the name of the species
                This is optional if there is only one species;
                default is 'electrons'.
            zeta_coord: bool
                If 'True' the 'z' selection is done in co-moving frame
            time: float
                Specify the time (s) at which refers the 'z' selection.
                Default is the first iteration, i.e. time = 0.0 s
            output: bool
                If 'True' returns a dict of five np.arrays which
                contains bunch_properties values.            
            **kwargs: keyword to pass to .pyplot.plot()

        **Returns**

            prop: dictionary, if 'output' is 'True'
                  A dict of bunch's properties values:
                  emittance, beam size,
                momenta spread, energy spread and beam charge
            fig, ax: Figure, Axes
                To handle the plot output

        """
        ptcl_percent = self.params['subsampling_fraction']
        if not t_lim:
            t_lim = [self.t.min(), self.t.max()]
        inds = np.where((self.t >= t_lim[0]) & (self.t <= t_lim[1]))
        t = self.t[inds]
        z = c*t*1.e6  # in microns
        a = np.zeros_like(t)

        for n in range(len(properties)):
            if properties[n] not in self.avail_bunch_prop:
                prop = '\n -'.join(self.avail_bunch_prop)
                raise ValueError("One or more property is not available. "
                                 "Available properties are:\n -{:s}\nTry again".format(prop))
            else:
                if zeta_coord and ('z' in select):
                    if time != 0.:
                        time = time
                    else:
                        time = self.t[0]                
                    for k, i in enumerate(t):
                        selection = self.__comoving_selection__(i, time, select)
                        x, ux, uz, gamma, w = \
                            self.ts.get_particle(['x', 'ux', 'uz', 'gamma', 'w'],
                                                 t=i, select=selection,
                                                 species=species)
                        if properties[n] == 'ph_emit':
                            a[k] = emittance(x, ux, w)*m_e*c
                        if properties[n] == 'ph_emit_n':
                            a[k] = emittance(x, ux, w)
                        if properties[n] == 'beam_size':
                            a[k] = beam_size(x, ux, w)
                        if properties[n] == 'momenta_spread':
                            a[k] = momenta_spread(x, ux, w)
                        if properties[n] == 'charge':
                            a[k] = e*w.sum()/ptcl_percent
                        if properties[n] == 'mean_energy':
                            a[k] = mean_energy(gamma,w)
                        if properties[n] == 'en_spread':
                            a[k] = energy_spread(gamma, w)
                        if properties[n] == 'tr_emit':
                            slope = divergence(px=ux, pz=uz)
                            a[k] = emittance(x, slope, w)
                        if properties[n] == 'tr_emit_n':
                            slope = divergence(px=ux, pz=uz)
                            mean_uz = np.ma.average(uz, weights=w)
                            a[k] = mean_uz*emittance(x, slope, w)
                        if properties[n] == 'tw_alpha':
                            a[k] = twiss(x, ux, uz, w)[0]
                        if properties[n] == 'tw_beta':
                            a[k] = twiss(x, ux, uz, w)[1]
                        if properties[n] == 'tw_gamma':
                            a[k] = twiss(x, ux, uz, w)[2]
                    if plot_over and (len(properties) == 1):
                        plt.plot(z, a, **kwargs)
                    else:        
                        plt.figure()
                        plt.title(properties[n])
                        plt.plot(z, a)
                        plt.xlim(left=z.min())

                else:
                    for k, i in enumerate(t):
                        x, ux, uz, gamma, w = \
                            self.ts.get_particle(['x', 'ux', 'uz', 'gamma', 'w'],
                                                 t=i, select=select,
                                                 species=species)
                        if properties[n] == 'ph_emit_n':
                            a[k] = emittance(x, ux, w)
                        if properties[n] == 'ph_emit':
                            a[k] = m_e*c*emittance(x, ux, w)
                        if properties[n] == 'beam_size':
                            a[k] = beam_size(x, ux, w)
                        if properties[n] == 'momenta_spread':
                            a[k] = momenta_spread(x, ux, w)
                        if properties[n] == 'charge':
                            a[k] = e*w.sum()/ptcl_percent
                        if properties[n] == 'mean_energy':
                            a[k] = mean_energy(gamma, w)
                        if properties[n] == 'en_spread':
                            a[k] = energy_spread(gamma, w)
                        if properties[n] == 'tr_emit':
                            slope = divergence(px=ux, pz=uz)
                            a[k] = emittance(x, slope, w)
                        if properties[n] == 'tr_emit_n':
                            slope = divergence(px=ux, pz=uz)
                            mean_uz = np.ma.average(uz, weights=w)
                            a[k] = mean_uz*emittance(x, slope, w)
                        if properties[n] == 'tw_alpha':
                            a[k] = twiss(x, ux, uz, w)[0]
                        if properties[n] == 'tw_beta':
                            a[k] = twiss(x, ux, uz, w)[1]
                        if properties[n] == 'tw_gamma':
                            a[k] = twiss(x, ux, uz, w)[2]
                    if plot_over and (len(properties) == 1):
                        plt.plot(z, a, **kwargs)
                    else:        
                        plt.figure()
                        plt.title(properties[n])
                        plt.plot(z, a)
                        plt.xlim(left=z.min())

    def spectrum(self, iteration, select=None, species='electrons',
                 energy=False, charge=False, Z=1, **kwargs):
        """
        Method to easily get an energy spectrum of 'selected' particles

        **Parameters**

            iteration: int
                Which iteration we need
            select: dictionary or ParticleTracker instance
                Particle selector
            species: str, optional
                Default is 'electrons'
            energy: bool, optional
                If 'True' this sets the x-axis on energy(MeV),
                otherwise x-axis has dimensionless gamma values.
                Default is 'False'.
            charge: bool, optional
                If True this sets the y-axis on dQ/dE values,
                multiplying the weights for electron charge.
                Default is False, that means setting y-axis on dN/dE values
            Z: int
                The atomic number of ion; default is 1.
            **kwargs: keyword to pass to .hist() method; in kwargs you can also
                    set the position of text inset in 'figure' frame [(0.,1.),(0.,1.)].
                    Default is [0.7,0.7].

        **Returns**

            ax: axes.Axes object to handle

        """
        in_ptcl_percent = 1/self.params['subsampling_fraction']
        a = 1
        if energy:
            a = 0.511
        q = 1
        if charge and Z:
            q = Z*e
        gamma, w = self.ts.get_particle(['gamma', 'w'], iteration=iteration,
                                        species=species, select=select)
        es = energy_spread(gamma, w)
        me = mean_energy(gamma, w)
        tot_charge = w.sum()*e*in_ptcl_percent

        pos = [0.7, 0.7]
        if 'text_pos' in kwargs:
            pos = kwargs['text_pos']
            del kwargs['text_pos']

        plt.hist(a*gamma, weights=q*in_ptcl_percent*w, **kwargs)
        fm = plt.get_current_fig_manager()
        num = fm.num
        fig = plt.figure(num)
        if fig.texts:
            fig.texts[0].remove()
        plt.figtext(pos[0], pos[1], "Total charge is {:.1e} C\n"
                                    "Mean energy is {:.2f} MeV\n"
                                    "Energy spread is {:3.1f} %".format(tot_charge, me, es*100))

    def phase_space_hist(self, species, iteration, components=['z','uz'],
                         select=None, zeta_coord=False,
                         mask=0., **kwargs):
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
                -'div1' is for div along planar slice x-z
                -'div2' is for total div 
        select: dict
            Particle selector
        zeta_coord: bool
            If 'True' this sets the z values in co-moving frame
        mask: float, optional
            A float in [0,1] to exclude particles with <='mask' normalized
            weights values.
        """
        cmap = 'Reds'
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

        if 'div1' in components:
            if components.index('div1') == 0:
                px, pz, comp2, weight = \
                    self.ts.get_particle(['ux', 'uz', components[1],'w'],iteration=iteration,
                                         select=select,species=species)
                comp1 = divergence(px=px, pz=pz)
            else:
                px, pz, comp1, weight = \
                    self.ts.get_particle(['ux', 'uz', components[0],'w'],iteration=iteration,
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

        if 'z' in components and zeta_coord:
            t = self.ts.current_t
            if components.index('z') == 0:
                comp1 -= c*t*1.e6
            else:
                comp2 -= c*t*1.e6


        H, xedge, yedge = \
            np.histogram2d(comp1, comp2,
                           bins=bins, weights=weight,
                           density=density)
        H = H.T
        X, Y = np.meshgrid(xedge, yedge)
        H = np.ma.masked_where(H <= mask, H)
        plt.pcolormesh(X, Y, H, cmap=cmap, alpha=alpha,**kwargs)
