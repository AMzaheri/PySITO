import numpy as np

from examples.seismic import AcquisitionGeometry

#---------------------------------prepare_acquisition
def prepare_geometry(model, inpara, geom_type):
    if geom_type == 'idwt':
        geometry_object = Geometry(inpara)
        geometry_object.define_geometry(model, inpara)
        return geometry_object

    if geom_type == 'devito':
        geometry_object, source_locations = prepare_devito_geometry(model, inpara)
        return geometry_object, source_locations
    
    if geom_type == 'grad_id':
        geometry_object, source_locations, receiver_locations = single_srcrec_devito_geometry(model, inpara)
        return geometry_object, source_locations, receiver_locations 

#----------------------------------prepare_devito_geometry
def prepare_devito_geometry(model, inpara):

    source_locations = np.empty((inpara.nshots, 2), dtype=np.float32)

    if inpara.nshots == 1 and inpara.nrec == 1:
        source_locations[:, 0] = model.domain_size[0] /4.
    elif inpara.nshots == 1 and inpara.nrec > 1:
        source_locations[:, 0] = model.domain_size[0] /2.
    elif inpara.nshots == 2:
        xlim = model.shape[0] *model.spacing[0]
        source_locations[:, 0] = np.array([xlim * .25, xlim * .75])
    elif inpara.nshots == 3:
        xlim = model.shape[0] *model.spacing[0]
        source_locations[:, 0] = np.array([xlim * .25, xlim * .5, xlim * .75])
    else:
        source_locations[:, 0] = np.linspace(0. + model.spacing[0], model.shape[0] * model.spacing[0], num=inpara.nshots)

    #else:
    #    source_locations[:, 0] = np.linspace(0., model.domain_size[0], num=inpara.nshots)

    source_locations[:, 1] = inpara.src_depth

    src_coordinates = np.empty((1, 2))
    if inpara.nshots == 1:
        src_coordinates[0, :] = source_locations[:,:]
    else:
        src_coordinates[0, :] = source_locations[0,:]

    rec_coordinates = np.empty((inpara.nrec,2))
    if inpara.nrec == 1:
        rec_coordinates[:, 0] = model.domain_size[0] * 3./4
    elif inpara.nrec == 5:
        xlim = model.shape[0] *model.spacing[0]
        rec_coordinates[:, 0] = np.array([xlim * .2, xlim * .4, xlim * .6, xlim * .8, xlim])
    else:
        rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=inpara.nrec)
    rec_coordinates[:, 1] = inpara.rec_depth

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, inpara.t0, inpara.tn, f0=inpara.f0, src_type=inpara.src_type)

    return geometry, source_locations

#---------------------------------- single source-receiver geometry
def single_srcrec_devito_geometry(model, inpara):

    source_locations = np.empty((inpara.nshots, 2), dtype=np.float32)
    receiver_locations = np.empty((inpara.nrec, 2), dtype=np.float32)

    if inpara.nshots == 1 and inpara.nrec == 1:
        source_locations[:, 0] = model.domain_size[0] /4.
    elif inpara.nshots == 1 and inpara.nrec > 1:
        source_locations[:, 0] = model.domain_size[0] /2.
    elif inpara.nshots == 2:
        xlim = model.shape[0] *model.spacing[0]
        source_locations[:, 0] = np.array([xlim * .25, xlim * .75])
    elif inpara.nshots == 3:
        xlim = model.shape[0] *model.spacing[0]
        source_locations[:, 0] = np.array([xlim * .25, xlim * .5, xlim * .75])
    else:
        source_locations[:, 0] = np.linspace(0., model.shape[0] * model.spacing[0], num=inpara.nshots)

    source_locations[:, 1] = inpara.src_depth

    src_coordinates = np.empty((1, 2))
    if inpara.nshots == 1:
        src_coordinates[0, :] = source_locations[:,:]
    else:
        src_coordinates[0, :] = source_locations[0,:]

    #-----------------------receivers

    if inpara.nrec == 1:
        receiver_locations[:, 0] = model.domain_size[0] * 3./4
    elif inpara.nrec == 2:
        xlim = model.shape[0] *model.spacing[0]
        receiver_locations[:, 0] = np.array([xlim * .25, xlim * .75])
    else:
        receiver_locations[:, 0] = np.linspace(0, model.domain_size[0], num=inpara.nrec)
    receiver_locations[:, 1] = inpara.rec_depth

    rec_coordinates = np.empty((1, 2))
    if inpara.nrec == 1:
        rec_coordinates[0, :] = receiver_locations[:,:]
    else:
        rec_coordinates[0, :] = receiver_locations[0,:]

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, inpara.t0, inpara.tn, f0=inpara.f0, src_type=inpara.src_type)

    return geometry, source_locations, receiver_locations

#----------------------------------Geometry class
class Geometry:
    def __init__(self, inpara):
        self.name = 'Geometry'
        self.src_coordinates = np.empty((inpara.nshots,2))
        self.rec_coordinates = np.empty((inpara.nrec,2))

    def define_geometry(self, model, inpara):
        
        source_locations = np.empty((inpara.nshots, 2), dtype=np.float32)
        if inpara.nshots == 1 and inpara.nrec ==1:
            source_locations[:, 0] = model.domain_size[0] / 4.
        elif inpara.nshots == 1 and inpara.nrec > 1:
            source_locations[:, 0] = model.domain_size[0] / 2.
        elif inpara.nshots == 3:
            xlim = model.shape[0] * model.spacing[0]
            source_locations[:, 0] = np.array([xlim * .25, xlim * .5, xlim * .75])
        else:
            source_locations[:, 0] = np.linspace(0., model.domain_size[0],num=inpara.nshots)
        source_locations[:, 1] = inpara.src_depth
   
        # receivers
        rec_coordinates = np.empty((inpara.nrec,2))
        if inpara.nrec == 1:
            rec_coordinates[:, 0] = model.domain_size[0] * 3./4
        else:
            rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=inpara.nrec)
        rec_coordinates[:, 1] = inpara.rec_depth

        self.src_coordinates[:,:] = source_locations[:,:]
        self.rec_coordinates[:,:] = rec_coordinates[:,:]


  
#----------------------------------toy_geometry
def prepare_toy_geometry(model, inpara, geo_type):
    if geo_type == 'devito':
        source_locations = np.empty((inpara.nshots, 2), dtype=np.float32)
        
        source_locations[:, 0] = np.linspace(0., \
                                 model.domain_size[0], num=inpara.nshots)
        source_locations[:, 1] = inpara.src_depth
        src_coordinates = np.empty((1, 2))
        src_coordinates[0, :] = source_locations[0,:]

        rec_coordinates = np.empty((inpara.nrec,2))
        rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], \
                                num=inpara.nrec)
        rec_coordinates[:, 1] = inpara.rec_depth

        geometry = AcquisitionGeometry(model, rec_coordinates, \
                       src_coordinates, inpara.t0, inpara.tn, \
                       f0=inpara.f0, src_type=inpara.src_type, \
                          grid=model.grid)
        return geometry, source_locations

    if geo_type == 'grad_id':
        source_locations = np.empty((inpara.nshots, 2), dtype=np.float32)
        receiver_locations = np.empty((inpara.nrec, 2), dtype=np.float32)
    
        source_locations[:, 0] = np.linspace(0.,model.domain_size[0], \
                                 num=inpara.nshots)
        source_locations[:, 1] = inpara.src_depth 
        src_coordinates = np.empty((1, 2))
        src_coordinates[0, :] = source_locations[0,:]

        receiver_locations[:, 0] = np.linspace(0, model.domain_size[0], num=inpara.nrec)
        receiver_locations[:, 1] = inpara.rec_depth
        rec_coordinates = np.empty((1, 2))
        rec_coordinates[0, :] = receiver_locations[0,:]

        geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, inpara.t0, inpara.tn, f0=inpara.f0, src_type=inpara.src_type)

    return geometry, source_locations, receiver_locations
