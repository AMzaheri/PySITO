import os
import copy
import numpy as np
import scipy.io as sio

from devito import gaussian_smooth
from examples.seismic import Model, demo_model

from utils import plot_velocity, plot_toy_velocity, plot_image

#--------------------------------prepare_models
def prepare_models(inpara, modelname, grid=None, rank=0):

    Vp_orig = sio.loadmat(inpara.model_path)
    Vp = Vp_orig['velocity']/1000
    #-----------------------------------------------------
    # TEST
    #Vp[:, 0:51] =1.5
    #Vp[:, 51:151] = 2.5
    #Vp[:, 151:201] = 4.5
    #Vp[:, 201:243] = 3.5
    
    # END OF TEST
    #-------------------------------------------------------


    if modelname == 'baseline':
        model = Model(vp=Vp, origin=inpara.origin, \
                      shape=inpara.shape, spacing=inpara.spacing,\
                      space_order=inpara.space_order, \
                      nbl=inpara.nbl, bcs=inpara.bcs, \
                      fs=False, grid=grid)
        if rank == 0:
            plot_velocity(model, inpara.outpath, modelname)     
   
    elif modelname == 'smooth_baseline':
        model = Model(vp=Vp, origin=inpara.origin, \
                      shape=inpara.shape, spacing=inpara.spacing,\
                      space_order=inpara.space_order, nbl=inpara.nbl, \
                      bcs=inpara.bcs, fs=False, grid=grid)

        gaussian_smooth(model.vp, sigma=inpara.sigma[0])
        if rank == 0:
            plot_velocity(model, inpara.outpath, modelname)

    elif modelname == 'monitor':
        Vp_m = Vp
        #Vp_m[Vp_m >= 4.0]  = 4.02
        #Vp_m[Vp_m >= 4.1]  += .02 

        # 23/09/2023
        Vp_m[:,151:175] += 0.02

        #print(len(Vp_m[Vp_m >= 4.02]))
        
        model = Model(vp=Vp_m, origin=inpara.origin, \
                      shape=inpara.shape, spacing=inpara.spacing,\
                      space_order=inpara.space_order, \
                      nbl=inpara.nbl, bcs=inpara.bcs, \
                      fs=False, grid=grid)
        if rank == 0:
            plot_velocity(model, inpara.outpath, modelname)

            model_diff = copy.deepcopy(model)
            Vp_diff = np.zeros(Vp_m.shape)
            #Vp_diff[Vp_m >= 4.02] = .02
            Vp_diff[:, 151:175] = 0.02

            print(np.unique(Vp_diff))
            nbl = inpara.nbl
            model_diff.vp.data[nbl:-nbl,nbl:-nbl] = Vp_diff[:]
            plot_velocity(model_diff, inpara.outpath, 'diff_model')

    return model

#------------------------------------make gaussin
def gauss(x, amp, mean, std):
    return np.sqrt(amp)*np.exp(-(x-mean)**2/(2*std**2))

def gaussian_monitor(model, inpara, rank=0):

    nx, ny = model.shape
    x = np.linspace(0, model.domain_size[0], nx)
    y = np.linspace(0, model.domain_size[1], ny)
    xx, yy = np.meshgrid(x, y)

    nbl = model.nbl
    #Vp = model.vp.data[nbl:-nbl,nbl:-nbl]
    
    Vp_orig = sio.loadmat(inpara.model_path)
    Vp = Vp_orig['velocity']

    x_gauss =  gauss(xx, inpara.gauss_amp * np.mean(Vp) ,\
                     model.domain_size[0] /2, inpara.gauss_width)
    y_gauss =  gauss(yy, inpara.gauss_amp * np.mean(Vp),\
                    model.domain_size[1]/2, inpara.gauss_width)
    gauss2d = x_gauss * y_gauss
    #print(np.transpose(gauss2d).shape, Vp.shape,model.shape)
    
    Vp_m = Vp + np.transpose(gauss2d)
    gauss_model = Model(vp=Vp_m, origin=inpara.origin, \
                        shape=inpara.shape, spacing=inpara.spacing,\
                        space_order=inpara.space_order, \
                        nbl=inpara.nbl, bcs=inpara.bcs,\
                        fs=False, grid=model.grid)

    plot_velocity(gauss_model, inpara.outpath, \
                  'gauss_monitor_model')

    model_diff = copy.deepcopy(gauss_model)
    model_diff.vp.data[nbl:-nbl,nbl:-nbl] = \
                            np.transpose(gauss2d)[:]
    if rank == 0:
        plot_image(model_diff.vp.data, model_diff,\
                   inpara.outpath, 'gauss_monitor_diff', \
                   vmin=-np.max(gauss2d), vmax=np.max(gauss2d), \
                   cmap='seismic')

    return gauss_model
#--------------------------------------toy example
def prepare_toy_models(inpara):
    vp_m1 = sio.loadmat(inpara.model_path)
    vp_m1 = vp_m1['velocity']/1000

    modelb = Model(vp=vp_m1, origin=inpara.origin,\
                   shape=inpara.shape, spacing=inpara.spacing,\
                   space_order=inpara.space_order, \
                   nbl=inpara.nbl, bcs=inpara.bcs, \
                   fs=False)


    modelm = gaussian_monitor(modelb, inpara)
    modelb0 = Model(vp=vp_m1, origin=inpara.origin, \
                   shape=inpara.shape, spacing=inpara.spacing,\
                      space_order=inpara.space_order, \
                      nbl=inpara.nbl, bcs=inpara.bcs,
                      fs=False, grid=modelb.grid)
   
    gaussian_smooth(modelb0.vp, sigma=inpara.sigma[0])
    plot_toy_velocity(modelb0, inpara, 'smooth_model') 

    plot_toy_velocity(modelb, inpara, 'baseline') 
    #plot_toy_velocity(modelm, inpara, 'monitor') 


    return modelb, modelb0, modelm
