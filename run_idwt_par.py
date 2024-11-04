
#-------------------------------------------------------------------
#   Filename:  run_idwt.py
#   Purpose:   Implementing Image Domain Wavefield Tomography
#   Developed by:    Afsaneh Mohammadzaheri
#   Email:     a.mohammadzaheri@leeds.ac.uk
#   License:   ?
#-------------------------------------------------------------------
# To run : https://arcdocs.leeds.ac.uk/usage/interactive.html
#---------------------------------------------------Import libraries
import sys
import os, time
import numpy as np
from mpi4py import MPI

from devito import Function
from examples.seismic.acoustic import AcousticWaveSolver

from InputReader import Inparam
from ModelClass import prepare_models, gaussian_monitor
from GeometryClass import prepare_geometry

from imaging_operator import migration_image
from operators import Laplacian, calculate_alfa

from WarpModule import find_warp

from gradient_op import compute_id_grad, plot_gradient
from inversion_utils import update_velocity, cost_function

from utils import plot_image, plot_velocity
#try:
from termcolor import colored, cprint
#except:
#    os.system('conda install -c conda-forge termcolor')
from parallel_helper import broadcast_inputs
#------------------------------------------------------------------
start_t = time.time()
infile = sys.argv[1]

#------------------------------------------------------- mpi initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#------------------------------------------------------ Read inputs
cprint('------------------------------ Reading inputs, processor : %s' % rank,\
        'magenta')

inpara = Inparam(inp_idwt_file=infile)

#------------------------------------- prepare models
#if rank == 0:
cprint('[ModelClass.py]: Creating Baseline and Monitor models, processor: %s' \
        % rank, 'magenta')
modelb = prepare_models(inpara, 'baseline', rank)
modelb0 = prepare_models(inpara, 'smooth_baseline', rank)

modelm = prepare_models(inpara, 'monitor', rank)
#modelm = gaussian_monitor(modelb, inpara, rank) 
'''
#--------------------------------------------geometry
    cprint('[GeometryClass.pyINPUT]: Creating geometry objects, rank %s' %rank, \
           'magenta')
    geometry_devito, source_locations = prepare_geometry(modelb, \
                                    inpara, 'devito')
    grad_geometry, source_locations_idwt, receiver_locations_idwt = \
                                   prepare_geometry(modelb, inpara, 'grad_id')

comm.Barrier()
# rank 0 broadcast inputs to workers
broadcast_inputs(comm, modelb, modelb0, modelm,\
                grad_geometry, source_locations_idwt, receiver_locations_idwt, \
                geometry_devito, source_locations)
'''
#-----------------------------------------
'''
if size > 1:
    partition_number = size - 1
else:
    partition_number = 1

nshot_per_rank = int(inpara.nshots / partition_number) + 1

if rank == 0:
    total_gradient = np.zeros(modelb0.vp.shape)
    cost_history = []

gradient_per_source = np.zeros(modelb0.vp.shape)
cost_val = 0.
total_cost_val = 0.
nbl = inpara.nbl
print('Test')

#comm.Barrier()

for i_iter in range(inpara.n_iter):

    cost_val = 0

    if inpara.grad_type == 'post':

        cprint('Post-Stack IDWT: inversion iter: {}/ {}'\
                .format(i_iter+1, inpara.n_iter), 'magenta')

        solver = AcousticWaveSolver(modelb, geometry_devito, space_order=4)
        image_b = postStack_migration(modelb, modelb0, geometry_devito,\
                                      solver, source_locations, i_iter)
        imageb =  Laplacian(image_b, modelb0)

        image_m = postStack_migration(modelm, modelb0, geometry_devito, \
                                      solver, source_locations, i_iter)
        imagem =  Laplacian(image_m, modelb0)

        #-------------------------------------------------------Warp 
        cprint('[Warping Function]: post-stack inversion iter = {}'\
               .format(i_iter+1), 'magenta')
        shift = np.zeros(imageb.data.shape)
        shift[nbl:-nbl, nbl:-nbl], mshift = find_warp(imageb.data[nbl:-nbl, nbl:-nbl],\
                                                   imagem.data[nbl:-nbl, nbl:-nbl], inpara)

        plot_image(shift, modelm, inpara.outpath,
                       'warp', vmin=-.2, vmax=.2, cmap='seismic')

        #----------------------------------------------------- cost
        cprint(f'[Cost function]: inversion iter = {i_iter+1}', 'magenta')
        cost_val += cost_function(shift, modelb0)

        #------------------------------------------------------alfa
        cprint('[Calculating alfa]: post-stack inversion iter = {}'\
               .format(i_iter+1), 'magenta')
        alfa = calculate_alfa(imagem, imageb,  modelb0, shift)

        print('     ')
        # -------------------------------------gradient
        for ishot in range(nshot_per_rank* rank, nshot_per_rank * (rank + 1)):
            cprint('[Post-stack gradient operator]: inversion iter = {}: source {}/{}'. \
                   format(i_iter +1, ishot+1, inpara.nshots), 'magenta')
            
            grad = compute_id_grad(modelb0, grad_geometry, geometry_devito, alfa, inpara, \
                                   i_iter, ishot, receiver_locations_idwt)
            # all receivers together
            #grad = compute_id_grad_2(modelb0, geometry_devito, alfa, inpara, \
            #                       i_iter, ishot, animation=True)
            #from gradient_op import compute_id_grad_devito
            #grad = compute_id_grad_devito(modelb0, grad_geometry, geometry_devito, alfa, inpara, \
            #                       i_iter, ishot, receiver_locations_idwt)


            gradient[:,:] += grad.data[:,:]
#---------------------------------------------------Pre-stack IDWT
    elif inpara.grad_type == 'pre':
        #------------------------------------------------------Pre stack IDWT loop 
        for ishot in range(nshot_per_rank* rank, nshot_per_rank * (rank + 1)):
            #shift_time_1 = time.time()
            geometry_devito.src_positions[0, :] = source_locations[ishot, :]

            solver = AcousticWaveSolver(modelb, geometry_devito, space_order=4)

            #-------------------------------------------------migration
            cprint('[imaging operator]: inversion iter= {}:  source {}/{}'\
                   .format(i_iter+1, ishot+1, inpara.nshots), 'magenta')
            if i_iter == 0:
                image_b = migration_image(modelb, modelb0, geometry_devito, solver)
                imageb =  Laplacian(image_b, modelb0)

            cprint('[imaging operator]: inversion iter = {}: source {}/{}'\
                   .format(i_iter+1, ishot+1, inpara.nshots), 'magenta')
            image_m = migration_image(modelm, modelb0, geometry_devito, solver)
            imagem = Laplacian(image_m, modelb0) 
        
            #-------------------------------------------------------Warp 
            cprint('[Warping Function]: inversion iter = {}: source {}/{}'
                   .format(i_iter+1, ishot+1, inpara.nshots), 'magenta')
            shift = np.zeros(imageb.data.shape)
            shift[nbl:-nbl, nbl:-nbl], mshift = find_warp(imageb.data[nbl:-nbl, nbl:-nbl],\
                                                   imagem.data[nbl:-nbl, nbl:-nbl], inpara)
            #plot_image(shift, modelm, inpara.outpath,
            #           'warp_%s' % ishot, vmin=-np.max(shift), vmax=np.max(shift), cmap='seismic')


            #----------------------------------------------------- cost
            cprint(f'[Cost function]: inversion iter = {i_iter+1}', 'magenta')
            cost_val += cost_function(shift, modelb0)

            #------------------------------------------------------alfa
            cprint(f'[Calculating alfa]: inversion iter = {i_iter+1}: source {ishot+1}/{inpara.nshots}', 'magenta')
            alfa = calculate_alfa(imagem, imageb,  modelb0, shift)

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
            if ishot == int(inpara.nshots/2.):
                anim = False
            else:
                anim = False
            grad = compute_id_grad(modelb0, grad_geometry, geometry_devito, \
                                   alfa, inpara, i_iter, ishot, receiver_locations_idwt)
            gradient_per_source[:,:] += grad.data[:,:]

    comm.Barrier()
    #gather partial results and add to the total sum
    comm.Reduce(total_gradient, gradient_per_source, op=MPI.SUM, root=0)
    comm.Reduce(total_cost_val, cost_val, op=MPI.SUM, root=0)
    if rank == 0 :  
        grad = Function(name="grad", grid=modelb0.grid)
        grad.data[:,:] =  gradient[:,:] / np.max(np.abs(gradient))
    

        if total_cost_val <= inpara.inv_tol:
            break
        else:
            cost_history.append(cost_val)
    #-----------------------------------------------velocity update
        cprint(f'[inversion_utils]: inversion iter = {i_iter+1}: updating velocity model', 'magenta')
        modelb0 = update_velocity(modelb0, inpara, grad)

    #----------------------------------------------plot
    cprint(f'[Plotting]: writing outputs', 'magenta')

    plot_image(imageb.data, modelb0, inpara.outpath, 
                'baseline', vmin=-4, vmax=4, cmap='gray')

    plot_image(imagem.data, modelb0, inpara.outpath, 
                'monitor', vmin=-4, vmax=4, cmap='gray')

    if inpara.grad_type == 'post':
        plot_image(shift, modelm, inpara.outpath,
                'warp', vmin=-np.max(shift), vmax=np.max(shift), cmap='seismic')

    plot_image(alfa, modelm, inpara.outpath,
               'alfa', vmin=- np.max(alfa), vmax=np.max(alfa), cmap='seismic')

    #gdata = np.max(np.quantile(grad.data[nbl:-nbl,nbl:-nbl], .95))
    gdata = np.max(grad.data)

    from gradient_op import plot_gradient
    plot_gradient(grad, modelb0, receiver_locations_idwt,\
                   source_locations_idwt, alfa, \
                   'gradient', inpara, vmin=-gdata, vmax=gdata)

    plot_velocity(modelb0, inpara.outpath, 'updated_model')

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(cost_history)
    plt.plot(cost_history, 'ro')
    plt.title('Cost in %s iterations' % inpara.n_iter)
    plt.xticks([0,1,2,3])
    plt.savefig(os.path.join(inpara.outpath, 'cost_history.png'))
    #-----------------------------end of the programme
end_t = time.time()
print(f'Program terminated: duratation: {end_t - start_t}')

'''
