
from devito import Eq, solve, Operator
from devito import Function, TimeFunction
from examples.seismic import TimeAxis, RickerSource, Receiver
from utils import grid_coord
from operators import interpolate_asrc

import os
import numpy as np
import matplotlib.pyplot as plt
from wave_animation import make_animation
#-------------------------------- adjoint operator
def adjoint_op(model, rec_coordinates, src_coordinates, inparam, asrc):

    s = model.grid.stepping_dim.spacing
    time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    src = RickerSource(name='src', grid=model.grid, f0=inparam.f0,
                   npoint=1, time_range=time_range)
    src.coordinates.data[:, :] = src_coordinates[:,:] 


    coords = grid_coord(model)   
    rec = Receiver(name='rec', grid=model.grid, npoint=len(coords), time_range=time_range) 
    rec.coordinates.data[:,:] = coords[:,:]
    rec.data[:,:] = asrc.data[:,:]


    v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=2, save=time_range.num)      

    eqn = model.m * v.dt2 - v.laplace - model.damp * v.dt
    stencil = Eq(v.backward, solve(eqn, v.backward))

    rec_term = rec.inject(field=v.backward, expr=rec * s**2 / model.m)
    src_term = src.interpolate(expr=v)


    op_adj = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)
    op_adj(dt=model.critical_dt)

    return v

##-----------------------------------forward operator

def forward_op(model, src_coordinates, inparam):


    s = model.grid.stepping_dim.spacing
    time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    src = RickerSource(name='src', grid=model.grid, f0=inparam.f0,
                   npoint=1, time_range=time_range)
    src.coordinates.data[:, :] = src_coordinates[:,:]


    coords = grid_coord(model)   
    rec = Receiver(name='rec', grid=model.grid, npoint=len(coords), time_range=time_range) 
    rec.coordinates.data[:,:] = coords[:,:]

    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2, save=time_range.num)

    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))

    src_term = src.inject(field=u.forward, expr=src * s**2 / model.m)

    rec_term = rec.interpolate(expr=u.forward)

    op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

    op(dt=model.critical_dt)

    return u

#---------------------------------------back propagation operator

def backpropagate_op(model, rec_coordinates, src_coordinates, inparam):


    s = model.grid.stepping_dim.spacing
    time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    src = RickerSource(name='src', grid=model.grid, f0=inparam.f0,
                   npoint=len(rec_coordinates), time_range=time_range)
    src.coordinates.data[:, :] = rec_coordinates[:,:]


    coords = grid_coord(model)   
    rec = Receiver(name='rec', grid=model.grid, npoint=len(coords), time_range=time_range) 
    rec.coordinates.data[:,:] = coords[:,:]

    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2, save=time_range.num)

    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))

    src_term = src.inject(field=u.forward, expr=src * s**2 / model.m)

    rec_term = rec.interpolate(expr=u.forward)

    op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

    op(dt=model.critical_dt)

    return u

#----------------------gradient operator

def gradient_op(u, v, model):

    grad = Function(name='grad', grid=model.grid)

    eqn_g = Eq(grad, grad + u * v.dt2)
    #eqn_g = Eq(grad, u * v.dt2)  # creates empty image

    op_grad = Operator([eqn_g])

    op_grad(dt=model.critical_dt)

    return grad


#-----------------------------plot
def plot_gradient(grad, model, rec_coordinates, src_coordinates, alfa, imagename, inparam, vmin, vmax):

    nbl = model.nbl
    plt.figure()
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    plt.imshow(np.transpose(grad.data[nbl:-nbl,nbl:-nbl]), cmap='seismic', extent=extent
                        , vmin=vmin, vmax=vmax)
    plt.colorbar(shrink=.5)

    plt.scatter(1e-3*rec_coordinates[:, 0], 1e-3*rec_coordinates[:, 1],
                    s=25, c='green', marker='v')
    plt.scatter(1e-3*src_coordinates[:, 0], 1e-3*src_coordinates[:, 1],
                    s=25, c='k', marker='o')

    from operators import find_asrc_coords
    alfa_coord = find_asrc_coords(model, alfa)

    #plt.scatter(1e-3*alfa_coord[::200, 0], 1e-3*alfa_coord[::200, 1],
    #                s=10, c='k', marker='*')
                    
    plt.savefig(os.path.join(inparam.outpath, '%s.png' % imagename))

#---------------------------------
#---------------------------------- gradient computations
def compute_id_grad_sep(model, geometry, alfa, inparam, ishot, animation):

    time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    grad = Function(name='grad', grid=model.grid)
    rec_coordinates = np.empty((1, 2), dtype=np.float32)
    gradient = np.zeros(grad.data.shape)


    #Create forward wavefield
    us = forward_op(model, geometry.src_positions, inparam)
    # d_r = alfa x u_s
    dr, us_a = interpolate_asrc(us, model, time_range, alfa)

    for sta in range(len(geometry.rec_positions)):  

        print(f'[Gradient operator]: Source: {ishot+1}/{inparam.nshots}, station : {sta + 1}/{inparam.nrec}')
        rec_coordinates[:,:] = geometry.rec_positions[sta,:]

        #u_r
        ur =  backpropagate_op(model, rec_coordinates, geometry.src_positions, inparam)
        # d_s = alfa x u_r
        ds, ur_a = interpolate_asrc(ur, model, time_range, alfa)

        # source adjoint
        Ls = adjoint_op(model, rec_coordinates, geometry.src_positions, inparam, ds)

        # backpropagated adjoint
        Lr = adjoint_op(model, geometry.src_positions, rec_coordinates, inparam, dr)

        if animation:
            make_animation(Ls, model, inparam.outpath, 'adj')
            make_animation(Lr, model, inparam.outpath, 'adj_r')
            make_animation(ur, model, inparam.outpath, 'ur')
            make_animation(us, model, inparam.outpath, 'us')


        # compute gradient
        grad1 = gradient_op(us, Ls, model)
        grad2 = gradient_op(ur, Lr, model)
        gradient[:,:] = grad1.data[:,:] + grad2.data[:,:]

        grad.data[:,:] += gradient[:,:] 

        plot_gradient(grad1, model, rec_coordinates, geometry.src_positions, \
                      alfa, 'gradient_%s' % ishot, inparam, \
                      vmin=-np.max(grad.data), vmax=np.max(grad.data))
        #print(f'np.max(grad) = {np.max(grad.data)}')
    return grad

#----------------------------------------------------------
def  compute_id_grad_multiple_receivers(model, geometry, alfa, inparam, ishot, animation):

    time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    #u_r
    ur =  backpropagate_op(model, geometry.rec_positions, geometry.src_positions, inparam)
    # d_s = alfa x u_r
    ds, ur_a = interpolate_asrc(ur, model, time_range, alfa)
    
    # source adjoint
    Ls = adjoint_op(model, geometry.rec_positions, geometry.src_positions, inparam, ds)


    #Create forward wavefield
    us = forward_op(model, geometry.src_positions, inparam)
    print(ur.data.shape, alfa.shape)
    
    # d_r = alfa x u_s
    dr, us_a = interpolate_asrc(us, model, time_range, alfa)

    # backpropagated adjoint
    Lr = adjoint_op(model, geometry.src_positions, geometry.rec_positions, inparam, dr)

    if animation:
        make_animation(Ls, model, inparam.outpath, 'adj')
        make_animation(Lr, model, inparam.outpath, 'adj_r')
        make_animation(ur, model, inparam.outpath, 'ur')
        make_animation(us, model, inparam.outpath, 'us')

    # compute gradient
    grad1 = gradient_op(us, Ls, model)
    grad2 = gradient_op(ur, Lr, model)

    grad = Function(name='grad', grid=model.grid)
    grad.data[:,:] = grad1.data[:,:] + grad2.data[:,:]



    plot_gradient(grad1, model, geometry.rec_positions, geometry.src_positions, \
                  alfa, 'grad1_%s' % ishot, inparam, \
                  vmin=-np.max(grad.data), vmax=np.max(grad.data))


    plot_gradient(grad2, model, geometry.rec_positions, geometry.src_positions, \
                  alfa, 'grad2_%s' % ishot, inparam, \
                  vmin=-np.max(grad.data), vmax=np.max(grad.data))
    return grad
#-------------------------------------------------------------
#----------------------------------adjoint and gradient update together
def adjoint_op2(model, src_coordinates, inparam, asrc, u):

    s = model.grid.stepping_dim.spacing
    time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    src = RickerSource(name='src', grid=model.grid, f0=inparam.f0,
                   npoint=len(src_coordinates), time_range=time_range)
    src.coordinates.data[:, :] = src_coordinates[:,:] 


    coords = grid_coord(model)   
    rec = Receiver(name='rec', grid=model.grid, npoint=len(coords), time_range=time_range) 
    rec.coordinates.data[:,:] = coords[:,:]
    rec.data[:,:] = asrc.data[:,:]


    v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=2) #, save=time_range.num)      

    eqn = model.m * v.dt2 - v.laplace - model.damp * v.dt
    stencil = Eq(v.backward, solve(eqn, v.backward))

    rec_term = rec.inject(field=v.backward, expr=rec * s**2 / model.m)
    src_term = src.interpolate(expr=v)

    # gradient update
    grad = Function(name='grad', grid=model.grid)
    eqn_g = Eq(grad, grad + u * v.dt2)


    op_adj = Operator([stencil] + src_term + rec_term + [eqn_g], subs=model.spacing_map)
    op_adj(dt=model.critical_dt)

    return grad
#----------------------------------------compute gradient

def  compute_id_grad(model, geometry, alfa, inparam, ishot, receiver_locations, \
                     comm, rank, size):

    gradient = np.zeros(model.shape)
    total_receiver_gradient = np.zeros(model.shape)

    if size > 1:
        partition_number = size - 1
    else:
        partition_number = 1
     nrec_per_rank = int(inparam.nrec / partition_number) + 1

    #------------------------------------

    time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    if rank == 0:
        #Create forward wavefield
        us = forward_op(model, geometry.src_positions, inparam)
        # d_r = alfa x u_s
        dr, us_a = interpolate_asrc(us, model, time_range, alfa)

        # source adjoint
        grads = adjoint_op2(model, geometry.src_positions, inparam, dr, us)

    #comm.Barrier()
    for sta in range(nrec_per_rank * rank, nrec_per_rank * (rank+1)):
        print(f'[Gradient operator]: Source: {ishot+1}/{inparam.nshots}, station : {sta + 1}/{inparam.nrec}')
        geometry.rec_positions[:, :] = receiver_locations[sta, :]
        
        #u_r
        ur =  backpropagate_op(model, geometry.rec_positions, geometry.src_positions, inparam)
        # d_s = alfa x u_r
        ds, ur_a = interpolate_asrc(ur, model, time_range, alfa)

        # backpropagated adjoint
        receiver_gradinet = adjoint_op2(model, geometry.rec_positions, inparam, ds, ur)
        gradient[:,:] += receiver_gradinet.data[:,:]

        #if sta == 0 or sta == len(receiver_locations) -1 :
        #    from wave_animation import make_animation
        #    make_animation(ur, model, inparam.outpath, 'ur')
        #    make_animation(us, model, inparam.outpath, 'us')

        #if sta >= 10 and sta <= 20:
        #    plot_gradient(receiver_gradinet, model, geometry.rec_positions, geometry.src_positions, \
        #                  alfa, 'gradr_src-%s_rec-%s' % (ishot, sta), inparam, \
        #                  vmin=-np.max(receiver_gradinet.data), vmax=np.max(receiver_gradinet.data))

    comm.Barrier()
    #gather partial results and add to the total sum
    comm.Reduce(total_receiver_gradient, gradient, op=MPI.SUM, root=0)


    grad = Function(name='grad', grid=model.grid)
    grad.data[:,:] = grads.data[:,:] + total_receiver_gradient[:,:]


    #plot_gradient(grads, model, geometry.rec_positions, geometry.src_positions, \
    #              alfa, 'grads_%s' % ishot, inparam, \
    #              vmin=-np.max(grads.data), vmax=np.max(grads.data))

    return grad
