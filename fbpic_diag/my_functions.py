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


def emittance_l(x, ux, w):
    """
    Function to calculate bunches' normalized longitudinal emittance;
    the result is given in mm*mrad.
    **Parameters**
    x, ux: two 1darrays of  phase-space coords
    w: ndarray of particles' weights
    **Returns**
    emittance, beam size and momenta spread
    """
    x_mean = np.ma.average(x, weights=w)
    ux_mean = np.ma.average(ux, weights=w)
    sigma_x2 = np.ma.average((x-x_mean)**2, weights=w)
    sigma_ux2 = np.ma.average((ux-ux_mean)**2, weights=w)
    sigma_xux2 = (np.ma.average((x-x_mean)*(ux-ux_mean), weights=w))**2
    emit = np.sqrt(sigma_x2*sigma_ux2-sigma_xux2)
    return emit, np.sqrt(sigma_x2), np.sqrt(sigma_ux2)


def emittance_t(x, ux, w):
    """
    Function to calculate bunches'normalized transverse emittance;
    the result is given in mm*mrad.
    **Parameters**
    x, ux: two ndarrays of  phase-space coords
    w: ndarray of particles' weights
    **Returns**
    emittance, beam size, momenta spread
    """
    sigma_x2 = np.ma.average(x**2, weights=w)
    sigma_ux2 = np.ma.average(ux**2, weights=w)
    sigma_xux = np.ma.average(x*ux, weights=w)
    emit = np.sqrt(sigma_x2*sigma_ux2 - sigma_xux**2)
    return emit, np.sqrt(sigma_x2), np.sqrt(sigma_ux2)


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
    def __init__(self, path):
        self.ts = OpenPMDTimeSeries(path)
        self.params = json.load(open('params.json'))
        self.iterations = self.ts.iterations
        self.t = self.ts.t
        self.avail_fields = self.ts.avail_fields
        self.avail_geom = self.ts.avail_geom
        self.avail_species = self.ts.avail_species
        self.avail_record_components = self.ts.avail_record_components

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
        return N

    def read_properties(self, var_list, **kwargs):

        """
        Function to convert a OpenPMDTimeSeries.get_particle() output in a dict

        **Parameters**

            var_list:  list of strings of quantities to get
            **kwargs: the same parameters of .get_particle() method,
                    passed as keywords

        """
        dictionary = dict()

        for key in var_list:
            dictionary[key] = self.ts.get_particle([key], **kwargs)[0]
        return dictionary

    def slice_emit(self, N, **kwargs):
        """
        Function to calculate slice emittances of a 'N sliced' bunch

        **Parameters**

            N: int, number of slices
            **kwargs: same parameters of .get_particle(), except 'var_list'

        **Returns**

            S_prop: dictionary of properties of each slice:
                    emittance, size, momenta spread and phase_space
            Ph_space: dictionary
                    'x'-'ux'-'z' values of three slices taken at
                    z_mean, (z_mean-z_dev) and (z_mean+z_dev)
            dz: longitudinal slices' thickness

        Note: here indexing of dict_keys labels over slices

        """
        if 'var_list' in kwargs:
            raise Exception("You don't need to pass 'var_list' argument!\n \
                             Try again with just the others kwargs\
                             of .get_particle()")

        dictionary = self.read_properties(['x', 'ux', 'z', 'w'], **kwargs)
        dz = (dictionary['z'].max()-dictionary['z'].min())/N

        s_emit = list()
        s_sigma_x2 = list()
        s_sigma_ux2 = list()
        Z = list()
        X = list()
        UX = list()
        ZZ = list()

        a = dictionary['z'].argsort()
        x = dictionary['x'][a]
        ux = dictionary['ux'][a]
        w = dictionary['w'][a]
        dictionary['z'].sort()
        z = dictionary['z']

        for n in range(N):
            inds = np.where((z >= z.min()+n*dz) &
                            (z <= z.min()+(n+1)*dz))[0]

            Z.append(z[inds].mean())

            s_prop = emittance_t(x[inds], ux[inds], w[inds])
            s_emit.append(s_prop[0])
            s_sigma_x2.append(s_prop[1])
            s_sigma_ux2.append(s_prop[2])

        S_prop = {'s_emit': s_emit, 's_sigma_x2': s_sigma_x2,
                  's_sigma_ux2': s_sigma_ux2, 'z': Z}

        for n in range(-1, 1):
            inds = np.where((z >= z.mean()+np.sqrt((n*z.var()))-dz/2) &
                            (z <= z.mean()+np.sqrt((n*z.var()))+dz/2))
            X.append(x[inds])
            UX.append(ux[inds])
            ZZ.append(z[inds])

        Ph_space = {'x': X, 'ux': UX, 'z': ZZ}

        return S_prop, Ph_space, dz

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
        for i in range(max-2,-1,-1):
            phi[:,i] = np.trapz(Ez[:,i:i+2],dx=info_e.dz) + phi[:,i+1]
        return phi, info_e

    def lineout(self, field_name, iteration,
                coord=None, theta=0, m='all', normalize=False, A0=None, slicing='z', on_axis=None, zeta_coord=False, **kwargs):
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
        else:
            E, info_e = self.ts.get_field(field=field_name, coord=coord,
                                          iteration=iteration, theta=theta, m=m)
        if slicing == 'z':
            if on_axis == None:
                on_axis = 0.
            N = self.params['Nr'] + int(on_axis*1.e-6/info_e.dr)
            E = E[N,:]
            z = info_e.z*1.e6
            if zeta_coord:
                v_w = self.params['v_window']
                t = self.ts.current_t
                z = (info_e.z-v_w*t)*1.e6
        else:
            if on_axis == None:
                on_axis = info_e.z[int(self.params['Nz']/2)]*1.e6
            N = int(self.params['Nz']/2) + int((on_axis*1.e-6-info_e.z[int(self.params['Nz']/2)])/info_e.dz)
            E = E[:,N]
            z = info_e.r*1.e6
        E0 = 1

        if normalize:
            E0 = self.__normalize__(field_name, coord, A0)

        plt.plot(z, E/E0, **kwargs)

    def map(self, field_name, iteration,
            coord=None, theta=0, m='all', normalize=False, N=None, **kwargs):
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
            **kwargs: keywords to pass to .Axes.imshow() method
        **Return**

            ax: a matplotlib.axes.Axes instance

        """
        if field_name == 'phi':
            E, info_e = self.potential(iteration,theta=theta,m=m)
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
        ax.imshow(E/E0, extent=info_e.imshow_extent*1.e6,
                  origin=origin, **kwargs)
        fig.colorbar(ax.get_images()[0], ax=ax, use_gridspec=True)

        return fig, ax

    def bunch_properties_evolution(self, select, species='electrons',
                                   output=False, zeta_coord=False, time=0.,
                                   **kwargs):
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
              - If 'select' contains 'z' list:
              selection is made in co-moving frame
            species: string
                A string indicating the name of the species
                This is optional if there is only one species;
                default is 'electrons'.
            output: bool
                If 'True' returns a dict of five np.arrays which
                contains bunch_properties values.
            zeta_coord: bool
                If 'True' the 'z' selection is done in co-moving frame
            time: float
                Specify the time (s) at which refers the 'z' selection.
                Default is the first iteration, i.e. time = 0.0 s
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
        emit = list()
        sigma_x2 = list()
        sigma_ux2 = list()
        es = list()
        charge = list()
        z = c*self.t*1.e6  # in microns

        if zeta_coord and ('z' in select):
            if time != 0.:
                time = time
            else:
                time = self.t[0]
            for k, i in enumerate(self.iterations):
                v_w = self.params['v_window']
                selection = select.copy()
                selection['z'] = [select['z'][0]+(v_w*(self.t[k]-time))*1e6,
                                  select['z'][1]+(v_w*(self.t[k]-time))*1e6]
                x, ux, gamma, w = \
                    self.ts.get_particle(['x', 'ux', 'gamma', 'w'],
                                         iteration=i, select=selection,
                                         species=species)
                l, m, n = emittance_t(x, ux, w)
                o = energy_spread(gamma, w)
                emit.append(l)
                sigma_x2.append(m)
                sigma_ux2.append(n)
                es.append(o)
                charge.append(w.sum()*e/ptcl_percent)
        else:
            for i in self.iterations:
                x, ux, gamma, w = \
                    self.ts.get_particle(['x', 'ux', 'gamma', 'w'],
                                         iteration=i, select=select,
                                         species=species)
                l, m, n = emittance_t(x, ux, w)
                o = energy_spread(gamma, w)
                emit.append(l)
                sigma_x2.append(m)
                sigma_ux2.append(n)
                es.append(o)
                charge.append(w.sum()*e/ptcl_percent)

        fig, ax = plt.subplots(2, 3, figsize=(10, 10))

        ax[0, 0].plot(z, emit, **kwargs)
        ax[0, 0].set_xlim(left=z.min())
        ax[0, 0].set_title('emit')

        ax[0, 1].plot(z, sigma_x2, **kwargs)
        ax[0, 1].set_xlim(left=z.min())
        ax[0, 1].set_title('beam size')

        ax[0, 2].plot(z, sigma_ux2, **kwargs)
        ax[0, 2].set_xlim(left=z.min())
        ax[0, 2].set_title('momenta spread')

        ax[1, 0].plot(z, es, **kwargs)
        ax[1, 0].set_xlim(left=z.min())
        ax[1, 0].set_title('energy spread')

        ax[1, 1].plot(z, charge, **kwargs)
        ax[1, 1].set_title('charge')

        ax[1, 2].remove()
        plt.tight_layout()

        if output:
            emit = np.array(emit)
            sigma_x2 = np.array(sigma_x2)
            sigma_ux2 = np.array(sigma_ux2)
            es = np.array(es)
            charge = np.array(charge)

            prop = {'emittance': emit, 'sigma_x2': sigma_x2,
                    'sigma_ux2': sigma_ux2, 'energy_spread': es,
                    'charge': charge}
            return prop, fig, ax
        else:
            return fig, ax

    def spectrum(self, iteration, select=None, species='electrons',
                 energy=False, charge=False, **kwargs):
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
            **kwargs: keyword to pass to .hist() method

        **Returns**

            ax: axes.Axes object to handle

        """
        in_ptcl_percent = 1/self.params['subsampling_fraction']
        a = 1
        if energy:
            a = 0.511
        q = 1
        if charge:
            q = e
        gamma, w = self.ts.get_particle(['gamma', 'w'], iteration=iteration,
                                        species=species, select=select)
        es = energy_spread(gamma, w)
        # needed values output as self.values? I'll see
        plt.hist(a*gamma, weights=q*in_ptcl_percent*w, **kwargs)
        print('Energy spread is {:3.1f}%'.format(es*100))

    def phase_space_hist(self, species, iteration, component1='z',
                         component2='uz', select=None, zeta_coord=False,
                         **kwargs):
        """
        Method that plots a 2D histogram of the particles phase space.

        Parameters:
        ----------
        species: str
            Select the particle specie among the available ones
            (Check them out in avail_species)
        iteration: int
            Selected iteration
        component1: str
            First phase space component of the phase space plot
            (Check the available components in avail_record_components)
        component2: str
            Second phase space component of the phase space plot
            (Check the available components in avail_record_components)
        select: dict
            Particle selector
        zeta_coord: bool
            If 'True' this sets the z values in co-moving frame
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

        comp1, comp2, weight = \
            self.ts.get_particle([component1, component2, 'w'],
                                 iteration=iteration, select=select,
                                 species=species)

        if component1 == 'z' and zeta_coord:
            t = self.ts.current_t
            comp1 -= c*t*1.e6

        if component2 == 'z' and zeta_coord:
            t = self.ts.current_t
            comp2 -= c*t*1.e6

        H, xedge, yedge = \
            np.histogram2d(comp1, comp2,
                           bins=bins, weights=weight,
                           density=density)
        H = H.T
        X, Y = np.meshgrid(xedge, yedge)
        H = np.ma.masked_where(H == 0, H)
        plt.pcolormesh(X, Y, H, cmap=cmap, alpha=alpha)
