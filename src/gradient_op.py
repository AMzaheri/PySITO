from devito import Eq, solve, Operator
from devito import Function, TimeFunction
from examples.seismic import TimeAxis, RickerSource, Receiver, PointSource
from utils import grid_coord
from operators import interpolate_asrc

import os
import numpy as np
import matplotlib.pyplot as plt
from wave_animation import make_animation


##-----------------------------------forward operator

def forward_op(model, geometry, inparam):


    s = model.grid.stepping_dim.spacing

    src = RickerSource(name='src', grid=model.grid, f0=inparam.f0,
                   npoint=1, time_range=geometry.time_axis)
    src.coordinates.data[:, :] = geometry.src_positions[:,:]


    rec = Receiver(name='rec', grid=model.grid, 
                   npoint=len(geometry.rec_positions),
                   time_range=geometry.time_axis) 
    rec.coordinates.data[:,:] = geometry.rec_positions[:,:]

    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2, save=geometry.nt) #time_range.num)

    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))

    src_term = src.inject(field=u.forward, expr=src * s**2 / model.m)

    rec_term = rec.interpolate(expr=u.forward)

    op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)

    op(dt=model.critical_dt)

    return u, rec

#---------------------------------------back propagation operator

def backpropagate_op(forward_rec, model, geometry, inparam):


    s = model.grid.stepping_dim.spacing
    dt = model.critical_dt

    bpsrc = PointSource(name='bpsrc', grid=model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)
    try:
        bpsrc.data[:,:] = forward_rec.data[:,:]
    except:
        bpsrc.data[:,:] = forward_rec.reshape(bpsrc.data.shape)
    #print(np.max(forward_rec.data), forward_rec.data.shape)

    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2,
                     save=geometry.nt)

    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

    #stencil = Eq(u.backward, solve(pde, u.backward))
    stencil = Eq(u.forward, solve(pde, u.forward))

    #bpsrc_term = bpsrc.inject(field=u.backward, 
    #                          expr=bpsrc* dt**2 / model.m)

    bpsrc_term = bpsrc.inject(field=u.forward, 
                              expr=bpsrc* dt**2 / model.m)


    op = Operator([stencil] + bpsrc_term , subs=model.spacing_map)
    op(dt=model.critical_dt)

    return u


#-----------------------------plot
def plot_gradient(grad, model, rec_coordinates, src_coordinates, alfa, imagename, inparam, vmin, vmax, cmap='seismic'):

    nbl = model.nbl
    plt.figure()
    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    plt.imshow(np.transpose(grad.data[nbl:-nbl,nbl:-nbl]), cmap=cmap, extent=extent
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
    plt.close()

#----------------------------------adjoint and gradient update together
def ForwardAdjoint(model, geometry, inparam, asrc, u):

    s = model.grid.stepping_dim.spacing
    #time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    src = RickerSource(name='src', grid=model.grid, f0=inparam.f0,
                   npoint=len(geometry.src_positions), 
                   time_range=geometry.time_axis)
    src.coordinates.data[:, :] = geometry.src_positions[:,:] 


    coords = grid_coord(model)   
    rec = Receiver(name='rec', grid=model.grid, npoint=len(coords),
                    time_range=geometry.time_axis) 
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

#---------------------------------backward adjoint

def BackwardAdjoint(model, geometry, inparam, asrc, u):

    s = model.grid.stepping_dim.spacing


    coords = grid_coord(model)   
    Asrc = Receiver(name='asrc', 
                   grid=model.grid, npoint=len(coords),
                   time_range=geometry.time_axis) 
    Asrc.coordinates.data[:,:] = coords[:,:]
    Asrc.data[:,:] = asrc.data[:,:]


    v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=2) #, save=time_range.num)      

    eqn = model.m * v.dt2 - v.laplace - model.damp * v.dt
    stencil = Eq(v.backward, solve(eqn, v.backward))

    asrc_term = Asrc.inject(field=v.backward, 
                              expr=Asrc * s**2 / model.m)

    # gradient update
    grad = Function(name='grad', grid=model.grid)
    eqn_g = Eq(grad, grad + u * v.dt2)

    op_adj = Operator([stencil] + asrc_term + [eqn_g], subs=model.spacing_map)
    op_adj(dt=model.critical_dt)

    return grad

#---------------------------compute_grad_id_all_receivers

def compute_id_grad_2(model, geometry, alfa, inparam, i_iter, ishot, animation):

    gradr = Function(name='gradr', grid=model.grid)
    #gradient = np.zeros(gradr.data.shape)

    #time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    #Create forward wavefield
    us, forward_rec = forward_op(model, geometry, inparam)
    # d_r = alfa x u_s
    dr, us_a = interpolate_asrc(us, model, geometry.time_axis, alfa)
    print('ur calculation')
    #u_r
    ur =  backpropagate_op(forward_rec, model, geometry, inparam)
    print('ds calculation')
    # d_s = alfa x u_r
    ds, ur_a = interpolate_asrc(ur, model, geometry.time_axis, alfa)

    if animation :
        from wave_animation import make_animation
        make_animation(ur, model, inparam.outpath, 'ur')
        make_animation(us, model, inparam.outpath, 'us')


    # source adjoint
    grads = ForwardAdjoint(model, geometry, inparam, dr, us)
    # backpropagated adjoint
    gradr = BackwardAdjoint(model, geometry, inparam, ds, ur)

    grad = Function(name='grad', grid=model.grid)
    grad.data[:,:] = grads.data[:,:] + gradr.data[:,:]


    plot_gradient(gradr, model, geometry.rec_positions, geometry.src_positions, \
                  alfa, 'gradr_%s' % ishot, inparam, \
                  vmin=-np.max(gradr.data), vmax=np.max(gradr.data))


    plot_gradient(grads, model, geometry.rec_positions, geometry.src_positions, \
                  alfa, 'grads_%s' % ishot, inparam, \
                  vmin=-np.max(grads.data), vmax=np.max(grads.data))

    return grad
#----------------------------------------------compute grad station by station

def  compute_id_grad(model, geometry, geometry_devito, \
                     alfa, inparam, i_iter, ishot, receiver_locations):

    gradr = Function(name='gradr', grid=model.grid)

    #time_range = TimeAxis(start=inparam.t0, stop=inparam.tn, step=model.critical_dt)

    #Create forward wavefield
    us, forward_rec = forward_op(model, geometry_devito, inparam)
    # d_r = alfa x u_s
    dr, us_a = interpolate_asrc(us, model, geometry.time_axis, alfa)
    # source adjoint
    grads = ForwardAdjoint(model, geometry_devito, inparam, dr, us)

    #print(forward_rec.data.shape)
    #quit()

    for sta in range(len(receiver_locations)):
        print('[Gradient operator]: inversion: {}, source: {}/{}, station : {}/{}'\
                .format(i_iter+1, ishot+1, inparam.nshots, sta+1, inparam.nrec))
        geometry.rec_positions[:, :] = receiver_locations[sta, :]

        print('ur calculation')
        #u_r
        ur =  backpropagate_op(forward_rec.data[:,sta], model, geometry, inparam)
        print('ds calculation')
        # d_s = alfa x u_r
        ds, ur_a = interpolate_asrc(ur, model, geometry.time_axis, alfa)

        #if sta == 0 or sta == len(receiver_locations) -1 :
        #    from wave_animation import make_animation
        #    make_animation(ur, model, inparam.outpath, 'ur')
        #    make_animation(us, model, inparam.outpath, 'us')

        # backpropagated adjoint
        receiver_gradinet = BackwardAdjoint(model, geometry, inparam, ds, ur)
        gradr.data[:,:] += receiver_gradinet.data[:,:]

    grad = Function(name='grad', grid=model.grid)
    grad.data[:,:] = grads.data[:,:] + gradr.data[:,:]


    plot_gradient(gradr, model, geometry.rec_positions, geometry.src_positions, \
                  alfa, 'gradr_%s' % ishot, inparam, \
                  vmin=-np.max(gradr.data), vmax=np.max(gradr.data))


    plot_gradient(grads, model, geometry.rec_positions, geometry.src_positions, \
                  alfa, 'grads_%s' % ishot, inparam, \
                  vmin=-np.max(grads.data), vmax=np.max(grads.data))

    return grad
#---------------------------------------id gardient using devito grad
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import AcquisitionGeometry

def compute_id_grad_devito(model, grad_geometry, geometry_devito, \
                     alfa, inparam, i_iter, ishot, receiver_locations):

    forward_geometry = gradient_geometry(model, grad_geometry, inparam, 'forward')
    backward_geometry = gradient_geometry(model, grad_geometry, inparam, 'backward')

    grads = Function(name='grads', grid=model.grid)
    gradr = Function(name='gradr', grid=model.grid)
    grad = Function(name='grad', grid=model.grid)
    total_gradr = Function(name='total_gradr', grid=model.grid)

    #Create forward wavefield
    forward_solver = AcousticWaveSolver(model, forward_geometry, space_order=4)
    d_forward, us, _ = forward_solver.forward(vp=model.vp, save=True)
    # d_r = alfa x u_s
    dr, us_a = interpolate_asrc(us, model, forward_geometry.time_axis, alfa)

    # source adjoint
    forward_solver.jacobian_adjoint(rec=dr, u=us, vp=model.vp, grad=grads)

    for sta in range(len(receiver_locations)):
        print('[Gradient operator]: inversion: {}, source: {}/{}, station : {}/{}'\
                .format(i_iter+1, ishot+1, inparam.nshots, sta+1, inparam.nrec))
        backward_geometry.src_positions[:, :] = receiver_locations[sta, :]

        print('ur calculation')
        #u_r
        backward_solver = AcousticWaveSolver(model, backward_geometry, space_order=4)
        d_backward, ur, _ = backward_solver.forward(vp=model.vp, save=True)

        print('ds calculation')
        # d_s = alfa x u_r
        ds, ur_a = interpolate_asrc(ur, model, backward_geometry.time_axis, alfa)

        # backpropagated adjoint
        backward_solver.jacobian_adjoint(rec=ds, u=ur, vp=model.vp, grad=gradr)

        total_gradr.data[:,:] += gradr.data[:,:]

        if sta == 0 or sta == len(receiver_locations) -1 :
            from wave_animation import make_animation
            make_animation(ur, model, inparam.outpath, 'ur')
            make_animation(us, model, inparam.outpath, 'us')

    grad.data[:,:] = grads.data[:,:] + total_gradr.data[:,:]


    plot_gradient(gradr, model, grad_geometry.rec_positions, grad_geometry.src_positions, \
                  alfa, 'gradr_%s' % ishot, inparam, \
                  vmin=-np.max(gradr.data), vmax=np.max(gradr.data))


    plot_gradient(grads, model, grad_geometry.rec_positions, grad_geometry.src_positions, \
                  alfa, 'grads_%s' % ishot, inparam, \
                  vmin=-np.max(grads.data), vmax=np.max(grads.data))
    return grad

#---------------------------------------------gradient_geometry
def gradient_geometry(model, grad_geometry, inpara, geo_type):

    coords = grid_coord(model)

#    if geo_type == 'forward':
#        geometry = AcquisitionGeometry(model, coords, grad_geometry.src_positions,
#                               t0=inpara.t0, tn=inpara.tn, f0=inpara.f0, src_type='Ricker')
#    if geo_type == 'backward':
#        geometry = AcquisitionGeometry(model, coords, grad_geometry.rec_positions,
#                               t0=inpara.t0, tn=inpara.tn, f0=inpara.f0, src_type='Ricker')


    if geo_type == 'forward':
        geometry = AcquisitionGeometry(model, grad_geometry.rec_positions, grad_geometry.src_positions,
                               t0=inpara.t0, tn=inpara.tn, f0=inpara.f0, src_type='Ricker')
    if geo_type == 'backward':
        geometry = AcquisitionGeometry(model, grad_geometry.src_positions, grad_geometry.rec_positions,
                               t0=inpara.t0, tn=inpara.tn, f0=inpara.f0, src_type='Ricker')
    return geometry

