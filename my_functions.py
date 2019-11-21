
"""
Set of functions by FA

"""
#Import section
import numpy as np
import matplotlib.pyplot as plt
from opmd_viewer import OpenPMDTimeSeries
import json
from scipy.constants import e, m_e, c, pi, epsilon_0

######################### Diag ###############################
class Diag(object):

   def __init__(self, path):
      self.ts = OpenPMDTimeSeries(path)
      self.params = json.load( open('params.json')) 

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
         dictionary[key] = self.ts.get_particle([key], **kwargs)
      return dictionary

#####################  emittance_l  #######################
   def emittance_l (self, ar_list,weights):

      """
       Function to calculate bunches'normalized longitudinal emittance; the result is given in mm*mrad.

       **Parameters**
         ar_list: list of two ndarrays of  phase-space coords
         weights: ndarray of particles' weights   
   
       **Returns**
         emittance, beam size and momenta spread   
      """
   

      #Check ar_list
      if type(ar_list) is not list:
         print('The argument must be a list')


      if len(ar_list)>2 or len(ar_list)<1:
         print('ar_list must be long just two strings')
      
         return
      
 

      #Longitudinal emittance
   
   
      x, ux, w = ar_list[0], ar_list[1], weights
      x_mean = np.ma.average(x, weights=w)
      ux_mean = np.ma.average(ux, weights=w)  
      sigma_x2 = np.ma.average((x-x_mean)**2, weights=w)
      sigma_ux2 = np.ma.average((ux-ux_mean)**2, weights=w)
      sigma_xux2 = (np.ma.average((x-x_mean)*(ux-ux_mean), weights=w))**2 
   
      emit=np.sqrt(sigma_x2*sigma_ux2-sigma_xux2)
      return emit, np.sqrt(sigma_x2), np.sqrt(sigma_ux2)

#########################  emittance_t  #############################
   def emittance_t (self, ar_list,weights):

      """
        Function to calculate bunches'normalized transverse emittance;
        the result is given in mm*mrad.

       **Parameters**
        ar_list: list of two ndarrays of  phase-space coords
        weights: ndarray of particles' weights

       **Returns**
        emittance, beam size, momenta spread
      """


      #Check var_list
      if type(ar_list) is not list:
         print('The argument must be a list')


      if len(ar_list)>2 or len(ar_list)<1:
         print('ar_list must be long just two strings')
      
         return

      #Transverse emittance

      x, ux, w = ar_list[0], ar_list[1], weights
   
      sigma_x2 = np.ma.average(x**2, weights=w)
      sigma_ux2 = np.ma.average(ux**2,weights=w)
      sigma_xux = np.ma.average(x*ux, weights=w)  

      emit=np.sqrt(sigma_x2*sigma_ux2 - sigma_xux**2)
      return emit, np.sqrt(sigma_x2), np.sqrt(sigma_ux2)
   
#######################  bunche_charge  ############################
   def bunch_charge (self, charge, weights):
      """
       Function to calculate bunch charge

       **Parameters**
         charge: int; single particle charge 
         weights: ndarray of particles' weights in the bunch

      """
      b_charge = charge*weights.sum()
      return b_charge

#####################  slice_emit  ###########################
   def slice_emit (self, dict, N):
      """
      Function to calculate slice emittances of a 'N sliced' bunch
      **Parameters**
         dict: a "read_properties" dictionary
         N: int, number of slices 
      **Returns**
         S_prop: a dictionary of properties of each slice: emittance, size, momenta spread and phase_space of each slice
         Ph_space: an arrays' dictionary of 'x'-'ux' values of three slices taken at z_mean, (z_mean-z_var) and (z_mean+z_var)
         dz: longitudinal slices' thickness 

      Note: here indexing of dict_keys labels over slices   

      """
      dz = (dict['z'].max()-dict['z'].min())/N

      s_emit=[]
      s_sigma_x2=[]
      s_sigma_ux2=[]
      Z=[]
      X=[]
      UX=[]
      ZZ=[]

      a=dict['z'].argsort()
      x=dict['x'][a]
      ux=dict['ux'][a]
      w=dict['w'][a]
      dict['z'].sort()


      for n in range(N):
         inds = np.where( (dict['z'] >= dict['z'].min()+n*dz) & (dict['z'] <= dict['z'].min()+(n+1)*dz) )[0] 

         Z.append(dict['z'][inds].mean())

         s_prop = self.emittance_t( [x[inds], ux[inds]], w[inds])
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
   def lineout(self, field_name, coord, iteration, theta, m, norm = False,**kwargs):
      E, info_e = self.ts.get_field(field=field_name, coord=coord, iteration=iteration, theta=theta, m=m)
      E0 = 1
      Nr = self.params['Nr']
      n_e = self.params['n_e']
      if norm:
         if field_name is 'rho':
            E0 = -e*n_e
         elif coord in ['x','y','r','t']:
            omega0 = self.params['omega0']
            E0 = m_e *c*omega0/e
         else:
            omegap = self.params['omegap']
            E0 = m_e*c*omegap/e

      plt.plot(info_e.z*1.e6,E[Nr,:]/E0,**kwargs)

################# imshow ####################
   def imshow(self, field_name, coord, iteration, theta, m, norm = False, **kwargs):
      E, info_e = self.ts.get_field(field=field_name, coord=coord, iteration=iteration, theta=theta, m=m)
      E0 = 1
      n_e = self.params['n_e']
      if norm:
         if field_name is 'rho':
            E0 = -e*n_e
         elif coord in ['x','y','r','t']:
            omega0 = self.params['omega0']
            E0 = m_e *c*omega0/e
         else:
            omegap = self.params['omegap']
            E0 = m_e*c*omegap/e
      
      plt.imshow(E/E0, extent=info_e.imshow_extent*1.e6, **kwargs)      



   

                     
