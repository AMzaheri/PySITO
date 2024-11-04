# To run parallel code in an interactive shell with 
# 2 cores for 10 minutes
# qrsh -l h_rt=00:10:00,h_vmem=4G -pe smp 2 -pty y bash
import sys
import numpy as np
import scipy.io as sio
from mpi4py import MPI
from examples.seismic import Model
from examples.seismic import AcquisitionGeometry
from InputReader import Inparam
from ModelClass import prepare_models
from GeometryClass import prepare_geometry


from termcolor import colored, cprint

#------------------------------------------------------- mpi initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#-------------------------------
#infile = sys.argv[1]
#inpara = Inparam(inp_idwt_file=infile)
#-----------------------------
#Vp_orig = sio.loadmat('/resstore/b0181/Data/Tyra/tyra-model-v1.mat')    
#Vp = Vp_orig['velocity']

#model = Model(vp=Vp, origin=(0.,0.), shape=(602, 241), spacing=(15., 15.),
#                      space_order=2, nbl=160, bcs="damp")


#modelb = prepare_models(inpara, 'baseline', rank)

#src_coordinates = 20. * np.ones((1,2))
#rec_coordinates = 40. * np.ones((1, 2))
#geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, inpara.t0, inpara.tn, f0=inpara.f0, src_type=inpara.src_type)
#geometry_devito, source_locations = prepare_geometry(model, \
 #                                   inpara, 'devito')

#grad_geometry, source_locations_idwt, receiver_locations_idwt = \
#                                   prepare_geometry(model, inpara, 'grad_id')


#------------------------------------------------------ Read inputs
cprint('------------------------------ Reading inputs, processor : %s' % rank,\
        'magenta')

infile = sys.argv[1]
inpara = Inparam(inp_idwt_file=infile)

#------------------------------------- prepare models
cprint('[ModelClass.py]: Creating Baseline and Monitor models, processor: %s' \
            % rank, 'magenta')
modelb = prepare_models(inpara, 'baseline', rank)
modelb0 = prepare_models(inpara, 'smooth_baseline', rank)
modelm = prepare_models(inpara, 'monitor', rank)
#modelm = gaussian_monitor(modelb, inpara, rank) 
#--------------------------------------------geometry
'''
cprint('[GeometryClass.pyINPUT]: Creating geometry objects, rank %s' %rank, \
           'magenta')
#geometry_devito, source_locations = prepare_geometry(modelb, \
#                                    inpara, 'devito')
#grad_geometry, source_locations_idwt, receiver_locations_idwt = \
#                                   prepare_geometry(modelb, inpara, 'grad_id')

print(' Done! ')
'''
