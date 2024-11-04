#-------------------------------------------------------------------
#   Filename:  run_idwt.py
#   Purpose:   Implementing Image Domain Wavefield Tomography
#   Developed by:    Afsaneh Mohammadzaheri
#   Email:     a.mohammadzaheri@leeds.ac.uk
#   License:   ?
#----------------------------------------------------
#------------------------------------Import libraries
import sys
import os, time
import numpy as np
import matplotlib.pyplot as plt

from devito import Function
from examples.seismic.acoustic import AcousticWaveSolver

from InputReader import Inparam
from ModelClass import prepare_models, gaussian_monitor, prepare_toy_models
from GeometryClass import prepare_geometry

from imaging_operator import migration_image, postStack_migration
from operators import Laplacian, calculate_alfa, laplacian_op

from WarpModule import find_warp

from gradient_op import compute_id_grad, compute_id_grad_2
from inversion_utils import update_velocity, cost_function
from utils import smooth_gradient

from utils import plot_image, plot_velocity
#try:
from termcolor import colored, cprint
#except:
#    os.system('conda install -c conda-forge termcolor')

#---------------------------------------- Read inputs

start_t = time.time()
infile = sys.argv[1]
cprint('------------------------------ Reading inputs', 'magenta')

inpara = Inparam(inp_idwt_file=infile)
#------------------------------------- prepare models

cprint('[ModelClass.py]: Creating Baseline and Monitor models', \
        'magenta')
toy_model = 0
if toy_model:
    modelb, modelb0, modelm = prepare_toy_models(inpara)
else:
    modelb = prepare_models(inpara, 'baseline')
    modelb0 = prepare_models(inpara, 'smooth_baseline', grid=modelb.grid)
    modelm = prepare_models(inpara, 'monitor', grid=modelb.grid)
#modelm = gaussian_monitor(modelb, inpara, rank=0)
#prepare_toy_models(inpara)

#print(modelb.critical_dt, modelb0.critical_dt, modelm.critical_dt)
#quit()
#--------------------------------------------geometry

cprint('[GeometryClass.pyINPUT]: Creating geometry objects', \
       'magenta')
geometry_devito, source_locations = prepare_geometry(modelb0, \
       inpara, 'devito')
#geometry_idwt = prepare_geometry(modelb0, inpara, 'idwt')

grad_geometry, source_locations_idwt, receiver_locations_idwt = \
       prepare_geometry(modelb0, inpara, 'grad_id')
#print(source_locations_idwt, receiver_locations_idwt)
#print(grad_geometry.time_axis, geometry_devito.time_axis)
#print(geometry_devito.rec_positions)

#print(modelb.grid == geometry_devito.grid)
#quit()
print('    ')
#-----------------------
# move to output_writer
if not os.path.exists(inpara.outpath):
    os.makedirs(inpara.outpath)
#----------------------------------- ------------inversion iteration

cost_history = []
gradient = np.zeros(modelb0.vp.shape)
nbl = inpara.nbl

for i_iter in range(inpara.n_iter):

    cost_val = 0

    if inpara.grad_type == 'post':

        cprint('Post-Stack IDWT: inversion iter: {}/ {}'\
                .format(i_iter+1, inpara.n_iter), 'magenta')
        #geometry_devito.src_positions[0, :] = source_locations[ishot, :]

        #solver = AcousticWaveSolver(modelb, geometry_devito, space_order=4)
        image_b = postStack_migration(modelb, modelb0, geometry_devito,\
                                      source_locations, i_iter,\
                                      inpara)
        #imageb =  Laplacian(image_b, modelb0)
        imageb = laplacian_op(image_b, modelb0, image_term=True)
        
        #print(np.max(imageb.data))
        image_vmax = 1e-3 * np.max(imageb.data)  #np.quantile(imageb.data, .95))
        plot_image(imageb.data, modelb0, inpara.outpath,
                 'baseline', vmin=-image_vmax, vmax=image_vmax, cmap='gray')
        #quit()
        image_m = postStack_migration(modelm, modelb0, geometry_devito, \
                                      source_locations, i_iter,\
                                      inpara)
        #imagem =  Laplacian(image_m, modelb0)
        imagem = laplacian_op(image_m, modelb0, image_term=True)


        plot_image(imagem.data, modelb0, inpara.outpath,
                 'monitor', vmin=-image_vmax, vmax=image_vmax, cmap='gray')

        #-------------------------------------------------------Warp 
        cprint('[Warping Function]: post-stack inversion iter = {}'\
               .format(i_iter+1), 'magenta')
        shift = np.zeros(imageb.data.shape)
        shift[nbl:-nbl, nbl:-nbl], mshift = find_warp(imageb.data[nbl:-nbl, nbl:-nbl],\
                                                   imagem.data[nbl:-nbl, nbl:-nbl], inpara)
        shift_vmax = np.max(shift)
        plot_image(shift, modelm, inpara.outpath,
                       'warp', vmin=-shift_vmax, vmax=shift_vmax, cmap='seismic')

        warped_image = np.zeros(imageb.shape)
        warped_image[nbl:-nbl, nbl:-nbl] = mshift[:,:]

        #image_vmax = np.max(warped_imageb)
        plot_image(warped_image, modelb0, inpara.outpath,
                 'warped', vmin=-image_vmax, vmax=image_vmax, cmap='gray')
        #quit()
        #---------------------------------plot trace
        plt.figure()
        plt.plot(imagem.data[301,150:], c='r', label='monitor')
        plt.plot(imageb.data[301,150:], c='b', label='baseline')
        plt.legend()
        plt.savefig('./outfiles/trace_base_monitor.png')
        plt.figure()
        plt.plot(imagem.data[301,150:], c='r', label='monitor')
        plt.plot(warped_image[301,150:], c='k', label='warped')
        plt.legend()
        plt.savefig('./outfiles/trace_warpedimg_monitor.png')

        plt.figure()
        plt.plot(warped_image[301,150:], c='k', label='warped')
        plt.plot(imageb.data[301,150:], c='b', label='baseline')
        plt.legend()
        plt.savefig('./outfiles/trace_base_warpedimg.png')
        quit()
        #----------------------------------------------------- cost
        cprint(f'[Cost function]: inversion iter = {i_iter+1}', 'magenta')
        cost_val += cost_function(shift, modelb0)

        #------------------------------------------------------mask

        #from MaskModule import target_oriented_mask
        #mask = target_oriented_mask(shift, inpara)
        #quit()
        #------------------------------------------------------alfa
        cprint('[Calculating alfa]: post-stack inversion iter = {}'\
               .format(i_iter+1), 'magenta')
        #alfa = calculate_alfa(imagem, imageb,  modelb0, shift, inpara)
        alfa = calculate_alfa(imageb, warped_image,  modelb0, shift, inpara)
        alfa = laplacian_op(alfa, modelb0, image_term=False)
        print('     ')
        # -------------------------------------gradient
        for ishot in range(inpara.nshots):
            cprint('[Post-stack gradient operator]: inversion iter = {}: source {}/{}'. \
                   format(i_iter +1, ishot+1, inpara.nshots), 'magenta')
            
            #grad = compute_id_grad(modelb0, grad_geometry, geometry_devito, alfa, inpara, \
            #                       i_iter, ishot, receiver_locations_idwt)
            # all receivers together
            #grad = compute_id_grad_2(modelb0, geometry_devito, alfa, inpara, \
            #                       i_iter, ishot, animation=True)
            from gradient_op import compute_id_grad_devito
            grad = compute_id_grad_devito(modelb0, grad_geometry, geometry_devito, alfa, inpara, \
                                   i_iter, ishot, receiver_locations_idwt)


            smooth_grad = smooth_gradient(grad.data[nbl:-nbl, nbl:-nbl], inpara.grad_smooth)

            #smooth_grad = mask * smooth_grad
            

            gradient[nbl:-nbl, nbl:-nbl] += smooth_grad

            #gradient[:,:] += grad.data[:,:]

#---------------------------------------------------Pre-stack IDWT
    elif inpara.grad_type == 'pre':
        #------------------------------------------------------Pre stack IDWT loop 
        for ishot in range(inpara.nshots):
            #shift_time_1 = time.time()
            geometry_devito.src_positions[0, :] = source_locations[ishot, :]

            #solver = AcousticWaveSolver(modelb, geometry_devito, space_order=4)

            #-------------------------------------------------migration
            cprint('[imaging operator]: inversion iter= {}:  source {}/{}'\
                   .format(i_iter+1, ishot+1, inpara.nshots), 'magenta')
            if i_iter == 0:
                image_b = migration_image(modelb, modelb0, geometry_devito, inpara)
                imageb =  Laplacian(image_b, modelb0)

            cprint('[imaging operator]: inversion iter = {}: source {}/{}'\
                   .format(i_iter+1, ishot+1, inpara.nshots), 'magenta')
            image_m = migration_image(modelm, modelb0, geometry_devito, inpara)
            imagem = Laplacian(image_m, modelb0) 
        
            #-------------------------------------------------------Warp 
            cprint('[Warping Function]: inversion iter = {}: source {}/{}'
                   .format(i_iter+1, ishot+1, inpara.nshots), 'magenta')
            shift = np.zeros(imageb.data.shape)
            shift[nbl:-nbl, nbl:-nbl], mshift = find_warp(imageb.data[nbl:-nbl, nbl:-nbl],\
                                                   imagem.data[nbl:-nbl, nbl:-nbl], inpara)
            plot_image(shift, modelm, inpara.outpath,
                       'warp', vmin=-np.max(shift), vmax=np.max(shift), cmap='seismic')
            #quit()
            #shift_time_2 = time.time()
            #print(f'Calculation of shift terminated in {shift_time_2 - shift_time_1}')


            warped_imageb = np.zeros(imageb.shape)
            warped_imageb[nbl:-nbl, nbl:-nbl] = mshift[:,:]
            #----------------------------------------------------- cost
            cprint(f'[Cost function]: inversion iter = {i_iter+1}', 'magenta')
            cost_val += cost_function(shift, modelb0)

            #------------------------------------------------------alfa
            cprint(f'[Calculating alfa]: inversion iter = {i_iter+1}: source {ishot+1}/{inpara.nshots}', 'magenta')
            #alfa = calculate_alfa(imagem, imageb,  modelb0, shift, inpara)
            alfa = calculate_alfa(imageb, warped_imageb,  modelb0, shift, inpara)

            print('     ')


#--------------------------------------------------------------------
#------------------------------------------------------------------
            # TEST
            #shift = np.zeros(imageb.data.shape)
            #alfa = np.zeros(imagem.data.shape)
            #alfa = shift
            #ix = int(alfa.shape[0]/2)
            #iz = 300
            #alfa[ix,iz] = 5.0
            # END OF TEST
#---------------------------------------------------------------
#---------------------------------------------------------------

            #--------------------------------------------------gradient       
            cprint(f'[Gradient operator]: inversion iter = {i_iter+1}: source {ishot+1}/{inpara.nshots}', 'magenta')
            stime_grad = time.time()
            if ishot == int(inpara.nshots/2.):
                anim = False
            else:
                anim = False
            #grad = compute_id_grad(modelb0, grad_geometry, geometry_devito, \
            #                       alfa, inpara, i_iter, ishot, receiver_locations_idwt)
            #grad = compute_id_grad_2(modelb0, geometry_devito, alfa, inpara, \
            #                        i_iter, ishot, animation=True)

            from gradient_op import compute_id_grad_devito
            grad = compute_id_grad_devito(modelb0, grad_geometry, geometry_devito, alfa, inpara, \
                                   i_iter, ishot, receiver_locations_idwt)

            smooth_grad = smooth_gradient(grad.data[nbl:-nbl, nbl:-nbl], inpara.grad_smooth)
            gradient[nbl:-nbl, nbl:-nbl] += smooth_grad
            #gradient[:,:] += grad.data[:,:]

            etime_grad = time.time()
            print(f'Calculation of gradient terminated in {etime_grad - stime_grad}')

    grad = Function(name="grad", grid=modelb0.grid)
    grad.data[:,:] =  gradient[:,:] / np.max(np.abs(gradient))
    

    if cost_val <= inpara.inv_tol:
        break
    else:
        #print(cost_val)
        cost_history.append(cost_val)
    #-----------------------------------------------velocity update
    cprint(f'[inversion_utils]: inversion iter = {i_iter+1}: updating velocity model', 'magenta')
    modelb0 = update_velocity(modelb0, inpara, grad)
    #quit()

#----------------------------------------------plot
cprint(f'[Plotting]: writing outputs', 'magenta')


image_data = np.max(np.quantile(imageb.data, .95))
plot_image(imageb.data, modelb0, inpara.outpath, 
            'baseline', vmin=-image_data, vmax=image_data, cmap='gray')
plot_image(imagem.data, modelb0, inpara.outpath, 
            'monitor', vmin=-image_data, vmax=image_data, cmap='gray')
dwarp = np.max(shift)
plot_image(shift, modelm, inpara.outpath,
            'warp', vmin=-dwarp, vmax=dwarp, cmap='seismic')

plot_image(alfa, modelm, inpara.outpath,
           'alfa', vmin=- np.max(alfa), vmax=np.max(alfa), cmap='seismic')

#gdata = np.max(np.quantile(grad.data[nbl:-nbl,nbl:-nbl], .95))
gdata = np.max(grad.data)
#plot_image(grad.data, modelm, inpara.outpath,
#            'gradient_nrec-%s_nsrc-%s' % (inpara.nrec, inpara.nshots),\
#              vmin=-.5, vmax=.5, cmap='seismic')

#------------------------------------------------------------
np.save(os.path.join(inpara.outpath, 'gradient_arr'), grad.data[nbl:-nbl, nbl:-nbl])
np.save(os.path.join(inpara.outpath, 'cost_history_arr'), np.array(cost_history))
np.save(os.path.join(inpara.outpath, 'warp_arr'), np.array(shift[nbl:-nbl, nbl:-nbl]))
#--------------------------------------------------------------


from gradient_op import plot_gradient
plot_gradient(grad, modelb0, geometry_devito.rec_positions,\
               source_locations, alfa, \
               'gradient', inpara, vmin=-gdata, vmax=gdata)

plot_velocity(modelb0, inpara.outpath, 'updated_model')

import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.array(cost_history))
plt.plot(cost_history, 'ro')
plt.title('Cost in %s iterations' % inpara.n_iter)
plt.xticks([0,1,2,3])
plt.savefig(os.path.join(inpara.outpath, 'cost_history.png'))
#-----------------------------end of the programme
end_t = time.time()
print(f'Program terminated: duratation: {end_t - start_t}')
