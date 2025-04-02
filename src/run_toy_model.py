
#------------------------------------Import libraries
import sys
import os, time
import numpy as np

import matplotlib.pyplot as plt
from gradient_op import plot_gradient

from devito import Function
from examples.seismic.acoustic import AcousticWaveSolver

from InputReader import Inparam
from ModelClass import prepare_models, prepare_toy_models
from GeometryClass import prepare_toy_geometry

from imaging_operator import migration_image, postStack_migration
from operators import Laplacian, calculate_alfa, laplacian_op

from WarpModule import find_warp

from gradient_op import compute_id_grad, compute_id_grad_2
from inversion_utils import update_velocity, cost_function
from utils import smooth_gradient

from utils import plot_image, plot_velocity
try:
    from termcolor import colored, cprint
except:
    os.system('conda install -c conda-forge termcolor')

#---------------------------------------- Read inputs

start_t = time.time()
infile = sys.argv[1]
cprint('------------------------------ Reading inputs', 'magenta')

inpara = Inparam(inp_idwt_file=infile)
#------------------------------------- prepare models
modelb, modelm = prepare_toy_models(inpara)
modelb0 = prepare_models(inpara, 'smooth_baseline', grid=modelb.grid)
print(modelb0.grid ==  modelm.grid)
#quit()
#--------------------------------------------geometry

cprint('[GeometryClass.pyINPUT]: Creating geometry objects', \
       'magenta')
geometry_devito, source_locations = prepare_toy_geometry(modelb0,\
                                    inpara, 'devito')

grad_geometry, source_locations_idwt, receiver_locations_idwt = \
       prepare_toy_geometry(modelb0, inpara, 'grad_id')
#print(source_locations_idwt, receiver_locations_idwt)
#print(source_locations)
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
        print(np.max(imageb.data))
        image_vmax = np.max(imageb.data)*1e-5  #np.quantile(imageb.data, .95))
        plot_image(imageb.data, modelb0, inpara.outpath,
                 'baseline', vmin=-image_vmax, vmax=image_vmax, cmap='gray')  

        quit()

        image_m = postStack_migration(modelm, modelb0, geometry_devito, \
                                      source_locations, i_iter,\
                                      inpara)
        #imagem =  Laplacian(image_m, modelb0)
        imagem = laplacian_op(image_m, modelb0, image_term=True)
        
        #image_data = .08 #np.max(np.quantile(imagem.data, .95))
        #plot_image(imagem.data, modelb0, inpara.outpath,
        #         'monitor', vmin=-image_data, vmax=image_data, cmap='gray')  

        #from MaskModule import plot_frequency_spectra, apply_filter
        #frequency_range = plot_frequency_spectra(imageb.data[nbl:-nbl, nbl:-nbl],\
        #                        imagem.data[nbl:-nbl, nbl:-nbl],\
        #                         modelb0.critical_dt, 1, inpara,\
        #                       'amplitude_spectra')
        #filt_imageb = np.zeros(imageb.data.shape)
        #filt_imageb[nbl:-nbl, nbl:-nbl] = apply_filter(imageb.data[nbl:-nbl, nbl:-nbl], \
        #                                     frequency_range, [50, 150], inpara, 1, 'bandpass')


        #image_data = .06 #np.max(np.quantile(filt_imageb, .95))
       # plot_image(filt_imageb, modelb0, inpara.outpath,
       #          'filtered_baseline', vmin=-image_data, vmax=image_data, cmap='gray')  
        #quit()
        #-------------------------------------------------------Warp 
        cprint('[Warping Function]: post-stack inversion iter = {}'\
               .format(i_iter+1), 'magenta')
        shift = np.zeros(imageb.data.shape)
        shift[nbl:-nbl, nbl:-nbl], mshift = find_warp(imageb.data[nbl:-nbl, nbl:-nbl],\
                                                   imagem.data[nbl:-nbl, nbl:-nbl], inpara)

        #print(modelb0.shape, shift.shape, imageb.data.shape)
        warped_imageb = np.zeros(imageb.shape)
        warped_imageb[nbl:-nbl, nbl:-nbl] = mshift[:,:]
        #print(mshift.shape)
                
        from utils import mask_function
        mask = mask_function(shift[nbl:-nbl, nbl:-nbl], inpara, i_iter+1)
        np.save(os.path.join(inpara.outpath, 'mask_arr'), mask)
        quit()
        #from MaskModule import mask_data
        #mask, masked_shift = mask_data(shift[nbl:-nbl,nbl:-nbl], modelb0, inpara)
        #shift[nbl:-nbl, nbl:-nbl] = masked_shift
        #dwarp = np.min(shift)
        #plot_image(shift, modelm, inpara.outpath,
        #        'warp', vmin=dwarp, vmax=-dwarp, cmap='seismic')
        #plt.figure()
        #plt.imshow(np.transpose(masked_shift), cmap='seismic_r')
        #plt.colorbar(shrink=.5)
        #plt.savefig(os.path.join(inpara.outpath, 'warp_image.png'))
        #plt.close()
        #quit()
        #----------------------------------------------------- cost
        cprint(f'[Cost function]: inversion iter = {i_iter+1}', 'magenta')
        cost_val += cost_function(shift, modelb0)

        #------------------------------------------------------mask

        #------------------------------------------------------alfa
        cprint('[Calculating alfa]: post-stack inversion iter = {}'\
              .format(i_iter+1), 'magenta')
        #alfa = calculate_alfa(imagem, imageb,  modelb0, shift, inpara)
        alfa = calculate_alfa(imageb, warped_imageb,  modelb0, shift, inpara)
        alfa = laplacian_op(alfa, modelb0, image_term=False)


        #from MaskModule import mask_data
        #mask, masked_alfa = mask_data(alfa[nbl:-nbl,nbl:-nbl], modelb0, inpara)
        #alfa[nbl:-nbl, nbl:-nbl] = masked_alfa
        #
        #smooth_alfa = smooth_gradient(alfa[nbl:-nbl, nbl:-nbl], inpara.grad_smooth)
        print('     ')
        plt.figure()
        plt.imshow(np.transpose(alfa[nbl:-nbl, nbl:-nbl]),\
                    cmap='gray')
        plt.colorbar(shrink=.5)
        plt.savefig(os.path.join(inpara.outpath, 'alfa_image.png'))
        plt.close()
        #quit()
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


            #gradient[:,:] += grad.data[:,:]

            #grad.data[:,:] = laplacian_op(grad.data, modelb0, image_term=False)

            smooth_grad = smooth_gradient(grad.data[nbl:-nbl, nbl:-nbl], inpara.grad_smooth)
            
            gradient[nbl:-nbl, nbl:-nbl] += smooth_grad

            gdata = np.max(smooth_grad)
            plt.figure()
            plt.imshow(np.transpose(smooth_grad),\
                    cmap='RdYlGn')
            plt.colorbar(shrink=.5)
            plt.savefig(os.path.join(inpara.outpath, 'gradient_%s.png' %ishot))
            plt.close()
            #quit()
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


            #smooth_alfa = smooth_gradient(alfa[nbl:-nbl, nbl:-nbl], inpara.grad_smooth)
            #alfa[nbl:-nbl, nbl:-nbl] = smooth_alfa
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
               'gradient', inpara, vmin=-gdata, vmax=gdata, cmap='gray')

plot_velocity(modelb0, inpara.outpath, 'updated_model')


plt.figure()
plt.plot(np.array(cost_history))
plt.plot(cost_history, 'ro')
plt.title('Cost in %s iterations' % inpara.n_iter)
plt.xticks([0,1,2,3])
plt.savefig(os.path.join(inpara.outpath, 'cost_history.png'))
#-----------------------------end of the programme
end_t = time.time()
print(f'Program terminated: duratation: {end_t - start_t}')
