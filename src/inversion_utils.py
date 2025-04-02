import numpy as np
from devito import Function, Eq, Operator
from devito import Max, Min, norm
from examples.seismic import Model

#------------------------------------------cost function
def cost_function(warp, model):
#    """
#    m = argmin ||warp(m|m0;x,z)|| ** 2
#    """

    cost = Function(name='cost', grid=model.grid, space_order=2)

    cost.data[:,:] = warp[:,:]
    cost_val = .5 * norm(cost) ** 2

    return cost_val

#-----------------------------------------------------
    
def update_velocity_devito(model, inpara, grad):

    Vp = model.vp
    vmax = np.max(Vp)
    vmin = np.min(Vp)

    velocity = np.zeros(grad.data.shape)
    velocity[:,:] = Vp.data[:,:]

    squared_slowness = 1. /(velocity ** 2)
    update_step = inpara.learning_rate * np.mean(squared_slowness)
    sqared_slowness_update = squared_slowness - update_step  * grad.data
 
    updated_velocity = 1./np.sqrt(sqared_slowness_update)

    #Apply the new velocity in devito model
    velocity_perturbation = velocity - updated_velocity
    velocity_perturbation_holder = Function(name='perturbation', grid=model.grid)
    velocity_perturbation_holder.data[:,:] = velocity_perturbation[:,:]
    velocity_update = Vp - velocity_perturbation_holder
   
    update_eq = Eq(Vp, Max(Min(velocity_update, vmax), vmin))
    Operator(update_eq)()
    
#--------------------------------
   
def update_velocity(model, inpara, grad):

    nbl = model.nbl
    velocity = np.zeros(grad.data.shape)
    velocity[:,:] = model.vp.data[:,:]

    squared_slowness = 1. /(velocity ** 2)
    update_step = inpara.learning_rate * np.mean(squared_slowness)
    sqared_slowness_update = squared_slowness - update_step  * grad.data
 
    updated_velocity = 1./np.sqrt(sqared_slowness_update)
  
    updated_model = Model(vp=updated_velocity[nbl:-nbl,nbl:-nbl], origin=inpara.origin,\
                          shape=inpara.shape, spacing=inpara.spacing,
                          space_order=inpara.space_order, nbl=inpara.nbl, bcs=inpara.bcs)
    
    return updated_model
