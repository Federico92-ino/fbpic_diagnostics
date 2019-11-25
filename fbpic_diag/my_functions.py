"""
Set of functions by FA

"""
#Import section
import numpy as np
import matplotlib.pyplot as plt
from opmd_viewer import OpenPMDTimeSeries, ParticleTracker
import json
from scipy.constants import e, m_e, c, pi, epsilon_0


######################### Diag ###############################
class Diag(object):
    """
    A class to handle diagnostics of a plasma simulation; pass the path of hd5f files
    """
    def __init__(self, path):
        self.ts = OpenPMDTimeSeries(path)
        self.params = json.load( open('params.json'))
        self.iterations = self.ts.iterations
        self.t = self.ts.t
        self.avail_fields = self.ts.avail_fields
        self.avail_geom = self.ts.avail_geom
        self.avail_species = self.ts.avail_species
        self.avail_record_components = self.ts.avail_record_components

###################### read_properties #########################
    def read_properties (self, var_list, **kwargs):

        """
        Function to convert a OpenPMDTimeSeries.get_particle() array list output in a dict

        **Parameters**
            var_list:  list of strings of quantities to get
            **kwargs: the same parameters of .get_particle() method passed as keywords
        """
        dictionary = dict()

        for key in var_list:
            dictionary[key]=self.ts.get_particle([key], **kwargs)[0]
        return dictionary

#####################  emittance_l  #######################
    def emittance_l (self, x, ux, w):

        """
         Function to calculate bunches'normalized longitudinal emittance; the result is given in mm*mrad.

         **Parameters**
            x, ux: two 1darrays of  phase-space coords
            w: ndarray of particles' weights

         **Returns**
            emittance, beam size and momenta spread
        """


        #Longitudinal emittance

        x_mean = np.ma.average(x, weights=w)
        ux_mean = np.ma.average(ux, weights=w)
        sigma_x2 = np.ma.average((x-x_mean)**2, weights=w)
        sigma_ux2 = np.ma.average((ux-ux_mean)**2, weights=w)
        sigma_xux2 = (np.ma.average((x-x_mean)*(ux-ux_mean), weights=w))**2

        emit = np.sqrt(sigma_x2*sigma_ux2-sigma_xux2)
        return emit, np.sqrt(sigma_x2), np.sqrt(sigma_ux2)

#########################  emittance_t  #############################
    def emittance_t (self, x, ux, w):

        """
          Function to calculate bunches'normalized transverse emittance;
          the result is given in mm*mrad.

         **Parameters**
          x, ux: two ndarrays of  phase-space coords
          w: ndarray of particles' weights

         **Returns**
          emittance, beam size, momenta spread
        """
        #Transverse emittance

        sigma_x2 = np.ma.average(x**2, weights=w)
        sigma_ux2 = np.ma.average(ux**2,weights=w)
        sigma_xux = np.ma.average(x*ux, weights=w)

        emit=np.sqrt(sigma_x2*sigma_ux2 - sigma_xux**2)
        return emit, np.sqrt(sigma_x2), np.sqrt(sigma_ux2)

#####################  slice_emit  ###########################
    def slice_emit (self, dict, N):
        """
        Function to calculate slice emittances of a 'N sliced' bunch
        **Parameters**
            dict: a "read_properties" dictionary
            N: int, number of slices
        **Returns**
            S_prop: a dictionary of properties of each slice: emittance, size, momenta spread and phase_space of each slice
            Ph_space: an arrays' dictionary of 'x'-'ux'-'z' values of three slices taken at z_mean, (z_mean-z_dev) and (z_mean+z_dev)
            dz: longitudinal slices' thickness

        Note: here indexing of dict_keys labels over slices

        """
        dz = (dict['z'].max()-dict['z'].min())/N

        s_emit = list()
        s_sigma_x2 = list()
        s_sigma_ux2 = list()
        Z = list()
        X = list()
        UX = list()
        ZZ = list()

        a = dict['z'].argsort()
        x = dict['x'][a]
        ux = dict['ux'][a]
        w = dict['w'][a]
        dict['z'].sort()


        for n in range(N):
            inds = np.where( (dict['z'] >= dict['z'].min()+n*dz) & (dict['z'] <= dict['z'].min()+(n+1)*dz) )[0]

            Z.append(dict['z'][inds].mean())

            s_prop = self.emittance_t(x[inds], ux[inds], w[inds])
            s_emit.append(s_prop[0])
            s_sigma_x2.append(s_prop[1])
            s_sigma_ux2.append(s_prop[2])


        S_prop={'s_emit':s_emit,'s_sigma_x2':s_sigma_x2,'s_sigma_ux2':s_sigma_ux2,'z': Z}

        for n in range(-1,1):
            inds = np.where((dict['z'] >= dict['z'].mean()+np.sqrt((n*dict['z'].var()))-dz/2) & (dict['z'] <= dict['z'].mean()+np.sqrt((n*dict['z'].var()))+dz/2))
            X.append(x[inds])
            UX.append(ux[inds])
            ZZ.append(dict['z'][inds])

        Ph_space = {'x': X, 'ux': UX, 'z': ZZ}

        return S_prop, Ph_space, dz

############### lineout #####################
    def lineout(self, field_name, iteration, coord=None, theta=0, m='all', norm=False, **kwargs):
        """
        Method to get a lineout plot of passed field_name
        **Parameters**
         field_name: string, field to plot
         iteration: int, the same as usual
         coord, theta, m: same parameters of .get_field() method; same defaults (None, 0, 'all')
         norm: bool, optional;
                 If norm = True this set the usual normalization of specified field, i.e:
                    - e*n_e for charge density 'rho'
                    - m_e*c*omega_0/e for transverse 'E'
                    - m_e*c*omega_p/e for longitudinal 'E'
         **kwargs: keywords to pass to .pyplot.plot() function

        """

        E, info_e = self.ts.get_field(field=field_name, coord=coord, iteration=iteration, theta=theta, m=m)
        E0 = 1
        Nr = self.params['Nr']
        n_e = self.params['n_e']
        if norm:
            if field_name == 'rho':
                E0 = -e*n_e
            elif coord in ['x', 'y', 'r', 't']:
                omega0 = self.params['omega0']
                E0 = m_e *c*omega0/e
            else:
                omegap = self.params['omegap']
                E0 = m_e*c*omegap/e

        plt.plot(info_e.z*1.e6, E[Nr,:]/E0, **kwargs)

################# map ####################
    def map(self, field_name, iteration, coord=None, theta=0, m='all', norm = False, **kwargs):
        """
        Method to get a 2D-map of passed field_name
        **Parameters**
         field_name: string, field to plot
         iteration: int, the same as usual
         coord, theta, m: same parameters of .get_field() method; same defaults (None, 0, 'all')
         norm: bool, optional;
                 If norm = True this set the usual normalization of specified field, i.e:
                    - e*n_e for charge density 'rho'; this return normalized density
                    - m_e*c*omega_0/e for transverse 'E'
                    - m_e*c*omega_p/e for longitudinal 'E'
         **kwargs: keywords to pass to .pyplot.imshow() function

        """
        E, info_e = self.ts.get_field(field=field_name, coord=coord, iteration=iteration, theta=theta, m=m)
        E0 = 1
        n_e = self.params['n_e']
        if norm:
            if field_name == 'rho':
                E0 = -e*n_e
            elif coord in ['x', 'y', 'r', 't']:
                omega0 = self.params['omega0']
                E0 = m_e *c*omega0/e
            else:
                omegap = self.params['omegap']
                E0 = m_e*c*omegap/e

        plt.imshow(E/E0, extent=info_e.imshow_extent*1.e6, **kwargs)
        plt.colorbar()

################# bunch_properties_evolution ################
    def bunch_properties_evolution(self, select, species='electrons', ptcl_percent=1, **kwargs):
        """
        Method to select a bunch and to plot the evolution of
        its characteristics along propagation length

        **Parameters**
         select: dict or ParticleTracker object, optional
                - If `select` is a dictionary:
                then it lists a set of rules to select the particles, of the form
                'x' : [-4., 10.]   (Particles having x between -4 and 10 microns)
                'ux' : [-0.1, 0.1] (Particles having ux between -0.1 and 0.1 mc)
                'uz' : [5., None]  (Particles with uz above 5 mc)
                - If `select` is a ParticleTracker object:
                then it returns particles that have been selected at another
                iteration ; see the docstring of `ParticleTracker` for more info.
         species: string
                A string indicating the name of the species
                This is optional if there is only one species; default is 'electrons'.
         ptcl_percent: float
                A number in [0,1] range that tells the particles percent output from simulation;
                default is 1.
         **kwargs: keyword to pass to .pyplot.plot()
        **Returns**
         prop: dictionary
                A dict of bunch's properties values: emittance, beam size, momenta spread and beam charge
         fig, ax: Figure, Axes to handle the plot output

        """

        emit, sigma_x2, sigma_ux2, charge = list(), list(), list(), list()
        z = c*self.t*1.e6  #in microns

        for i in self.iterations:
            x, ux, w = self.ts.get_particle(['x','ux','w'], iteration=i, select=select, species=species)
            l, m, n = self.emittance_t(x, ux, w)
            emit.append(l)
            sigma_x2.append(m)
            sigma_ux2.append(n)
            charge.append(w.sum()*e/ptcl_percent)

        fig, ax = plt.subplots(2, 2, figsize=(10,10))

        ax[0,0].plot(z, emit, **kwargs), ax[0,0].set_title('emit')
        ax[0,1].plot(z, sigma_x2, **kwargs), ax[0,1].set_title('beam size')
        ax[1,0].plot(z, sigma_ux2, **kwargs), ax[1,0].set_title('momenta spread')
        ax[1,1].plot(z, charge, **kwargs), ax[1,1].set_title('charge')
        plt.tight_layout()

        emit = np.array(emit)
        sigma_x2 = np.array(sigma_x2)
        sigma_ux2 = np.array(sigma_ux2)
        charge = np.array(charge)

        prop={'emit':emit, 'sigma_x2':sigma_x2, 'sigma_ux2':sigma_ux2, 'charge':charge}

        return prop, fig, ax

    def phase_space_hist(self, species, iteration, component1='z', component2='uz', select=None, **kwargs):

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

        comp1, comp2, weight = self.ts.get_particle([component1, component2, 'w'], iteration=iteration, select=select, species=species)

        H, xedge, yedge = np.histogram2d(comp1, comp2, bins=bins, weights=weight, density=density)
        H = H.T
        X, Y = np.meshgrid(xedge, yedge)
        H = np.ma.masked_where(H == 0, H)
        plt.pcolormesh(X, Y, H, cmap=cmap, alpha=alpha)

