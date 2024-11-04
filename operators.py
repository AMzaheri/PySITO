
import sys

import numpy as np
import copy

from   examples.seismic  import TimeAxis
from devito import Function,TimeFunction, Operator, Eq, solve
from examples.seismic import PointSource, Receiver
from examples.seismic.acoustic import AcousticWaveSolver

from devito import norm
from examples.seismic import PointSource, Receiver
from examples.seismic import RickerSource

from utils import grid_coord

#-----------------------------------------------------------------
#-----------------------------------------------------
def Laplacian(image, model): #, model, geometry):
    #dt = model.critical_dt
    #eqn = Eq(image, image.laplace)
    #op = Operator([eqn])
    #return op();
    nbl = model.nbl
    image_laplace = copy.deepcopy(image)
    laplace_arr = 4 * image.data[1:-1,1:-1]- image.data[0:-2,1:-1] - image.data[2:,1:-1] - image.data[1:-1,0:-2] - image.data[1:-1,2:]
    laplace_arr = laplace_arr /(model.spacing[0] * model.spacing[0])

    image_laplace.data[1:-1,1:-1] = laplace_arr[:,:]
    return image_laplace
#-----------------------------------------------------------

def laplacian_op(data, model, image_term):

    nbl = model.nbl
    if image_term:
        data_laplace = copy.deepcopy(data)
        data = data_laplace.data
    else:
        data_laplace = np.zeros(data.shape)
    laplace_arr = 4 * data[1:-1,1:-1]- data[0:-2,1:-1] - \
                  data[2:,1:-1] - data[1:-1,0:-2] - data[1:-1,2:]
    laplace_arr = laplace_arr /(model.spacing[0] * model.spacing[0])

    if image_term:
        data_laplace.data[1:-1,1:-1] = laplace_arr[:,:]
    else:
        data_laplace[1:-1,1:-1] = laplace_arr[:,:]
    return data_laplace
#-------------------------------------------
def second_deriv(image):  #, model, geometry):
    eqn = Eq(image, image.dy2.evaluate)
    op = Operator([eqn])
    return op();

#----------------------------------------------------------

def first_deriv(image):  #, model, geometry):
    eqn = Eq(image, image.dy.evaluate)
    op = Operator([eqn])
    return op();

#-----------------------------------------------

def calculate_alfa(image1, image2,  model, warp, inpara):
    """
	Calculate alfa a defined in equation 8 in XXX
        image1: monitor image, TimeFunction, migration images
        image2: baseline image
    """

    nbl = model.nbl


    image_residual = image1.data - image2.data

    image_dz = np.zeros(image2.data.shape)
    image_dz[:,0:-1] = np.diff(image2.data, axis=1)  #/model.spacing[0]    
    #image_dz[:,0:-1] = image2.data[:,1:]-image2.data[:,0:-1]

    image_dz2 = np.zeros(image2.data.shape)
    image_dz2[:,0:-1] = np.diff(image_dz, axis=1) #/model.spacing[0]    
    #image_dz2[:,0:-1] = image_dz[:,1:]-image_dz[:,0:-1]

    n = warp * image_dz
    d = image_dz ** 2 - image_dz2 * image_residual

    #zero values in the denominator causes spikes in the gradient
    # method 1
    #alfa_val = np.divide(n,d, out=np.zeros_like(n), where=d!=0)
    # method 1, didn't work
    #add water level to get rid of zeros in the denominator

    #d = d + 0.1 * np.max(d)
    #alfa_val = np.divide(n, d)


    #phi0=-(wrp0*wllwrpgrd1)/(phi0+water*sum(abs(phi0))/nw) ! calculates phi's numerator
    #phi0=agc_gain0*phi0                                    ! applies AGC for source
    #phi0=phi0*taper
    #d = d + 0.1 *np.sum(np.abs(d))

    d = d + inpara.water_level * np.sum(np.abs(d))/inpara.nshots
    alfa_val = np.divide(n, d)
   
    #from utils import smoothing_function
    #alf_val = smoothing_function(alfa_val, gauss_sigma=20) #boxcar_width=None, gauss_sigma=None, hanning_window=None):

    return alfa_val 

##--------------------------------------------------calculate adjoint source

def compute_adjSource(u0, alfa_val):

    """
    compute adjoint source according to eq xx in XX
   .
   """
    asrc = copy.deepcopy(u0)
    for i in range(u0.data.shape[0]):
        asrc.data[i] = np.multiply(alfa_val, u0.data[i])

    return asrc


    
#-------------------------------------find adjoint source locations
def find_asrc_coords(model, alfa):

    nbl = model.nbl

    nx, ny = model.shape
    x = np.linspace(0, model.domain_size[0], nx)
    y = np.linspace(0, model.domain_size[1], ny)

    (ix, iy) = np.where(alfa[nbl:-nbl,nbl:-nbl] !=0 )

    asrc_coords = np.vstack((x[ix],y[iy])).T

    return asrc_coords
   

#--------------------------------------- interpolate_asrc 

def interpolate_asrc(u0, model, time_axis, alfa):

    u = compute_adjSource(u0, alfa)
    
    coords = grid_coord(model)
    #coords = find_asrc_coords(model, alfa)
    asrc = Receiver(name='asrc', grid=model.grid,
                              time_range=time_axis,
                              coordinates=coords)
    a_term = asrc.interpolate(expr=u)

    op = Operator(a_term)
    op();

    return asrc, u
