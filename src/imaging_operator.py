import numpy as np

from utils import bandpass_filter, plot_shotrecord

from examples.seismic.acoustic import AcousticWaveSolver
from devito import TimeFunction, Operator, Eq, solve
from examples.seismic import PointSource
from devito import Function

from examples.seismic import AcquisitionGeometry

from MaskModule import mask_topmute_op
#------------------------------

def ImagingOperator(model, geometry, solver, image):
    # Define the wavefield with the size of the model and the time dimension
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)

    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                     save=geometry.nt)

    # Define the wave equation, but with a negated damping term
    eqn = model.m * v.dt2 - v.laplace + model.damp * v.dt.T

    # Use `solve` to rearrange the equation into a stencil expression
    stencil = Eq(v.backward, solve(eqn, v.backward))

    # Define residual injection at the location of the forward receivers
    dt = model.critical_dt
    residual = PointSource(name='residual', grid=model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)    
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / model.m)

    # Correlate u and v for the current time step and add it to the image
    image_update = Eq(image, image - u * v)

    return Operator([stencil] + res_term + [image_update],
                    subs=model.spacing_map)

#----------------------------------------------------create migration_image

def migration_image(model, model0, geometry, inparam):
 

    solver = AcousticWaveSolver(model, geometry, space_order=4)
    image = Function(name='image', grid=model.grid, space_order=2)
    op_imaging = ImagingOperator(model0, geometry, solver, image)

    # Generate synthetic data from true model
    true_d, _, _ = solver.forward(vp=model.vp)
    
    # Compute smooth data and full forward wavefield u0
    smooth_d, u0, _ = solver.forward(vp=model0.vp, save=True)

    # Compute gradient from the data residual  
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)

    # Mask
    if inparam.mask_rec_data:
        residual = PointSource(name='residual', grid=model.grid,
                               time_range=geometry.time_axis,
                               coordinates=geometry.rec_positions)
        residual_data = smooth_d.data - true_d.data
        masked_residual = mask_topmute_op(model0, residual_data, model, geometry)
        residual.data[:,:] = masked_residual[:,:]     
    # end mask
    else:
        residual = smooth_d.data - true_d.data

    op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt, 
               residual=residual)

    return image

#----------------------------------post stack 
def postStack_migration(model, model0, geometry, source_locations, i_iter, inparam):

    solver = AcousticWaveSolver(model, geometry, space_order=4)
    image = Function(name='image', grid=model.grid, space_order=2)
    op_imaging = ImagingOperator(model, geometry, solver, image)

    for i in range(len(source_locations)):
        print('[Post Stack imaging]: inversion iter = %d, source %d out of %d'\
               % (i_iter+1, i+1, len(source_locations)))

        # Update source location
        geometry.src_positions[0, :] = source_locations[i, :]
        #print(geometry.src_positions[0, :])
        # Generate synthetic data from true model
        true_d, _, _ = solver.forward(vp=model.vp)
        #print(f'data shape = {true_d.data.shape}')
    
        # Compute smooth data and full forward wavefield u0
        smooth_d, u0, _ = solver.forward(vp=model0.vp, save=True)

        # Compute gradient from the data residual  
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)
        
        if i == int(inparam.nshots/2):
            #print(np.max(true_d.data))
            plot_shotrecord((true_d.data), model, \
                            inparam.t0, inparam.tn, inparam.f0, inparam.nshots, \
                            inparam.outpath, 'receiver_data', cmap='gray')
        
        if inparam.filter_data_info[3]:
             true_d.data[:,:] = bandpass_filter(true_d.data, inparam)
             smooth_d.data[:,:] = bandpass_filter(smooth_d.data, inparam)


        if i == int(inparam.nshots/2):
            #print(np.max(true_d.data))
            plot_shotrecord((true_d.data), model, \
                            inparam.t0, inparam.tn, inparam.f0, inparam.nshots, \
                            inparam.outpath, 'filtered_receiver_data', cmap='gray')
        # Mask
        if inparam.mask_rec_data:
            residual = PointSource(name='residual', grid=model.grid,
                               time_range=geometry.time_axis,
                               coordinates=geometry.rec_positions)
            residual_data = smooth_d.data - true_d.data
            masked_residual = mask_topmute_op(model0, residual_data, inparam, geometry)
            residual.data[:,:] = masked_residual[:,:]
            
            if i == int(inparam.nshots/2):
                masked_true_d = mask_topmute_op(model0,true_d.data, inparam, geometry)
                plot_shotrecord((masked_true_d), model, \
                            inparam.t0, inparam.tn, inparam.f0, \
                            inparam.nshots,  inparam.outpath, 'RecAfterMask', cmap='gray')
                #print(masked_true_d.shape)
        # end mask
        else:
            residual = smooth_d.data - true_d.data



        op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt, 
               residual=residual)

    return image
