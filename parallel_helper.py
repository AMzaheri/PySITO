

#----------------------------------------broadcast_inputs
def broadcast_inputs(comm, modelb, modelb0, modelm,\
                grad_geometry, source_locations_idwt, receiver_locations_idwt, \
                geometry_devito, source_locations):

    modelb = comm.bcast(modelb, root=0)
    modelb0 = comm.bcast(modelb0, root=0)
    modelm = comm.bcast(modelm, root=0)

    grad_geometry = comm.bcast(grad_geometry, root=0)
    source_locations_idwt = comm.Bcast(source_locations_idwt, root=0)
    receiver_locations_idwt = comm.Bcast(receiver_locations_idwt, root=0)
  
    geometry_devito = comm.Bcast(geometry_devito, root=0)
    source_locations = comm.Bcast(source_locations, root=0)


