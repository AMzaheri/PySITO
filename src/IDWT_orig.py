 

import os

import numpy as np
from scipy import ndimage
import scipy.io as sio

import copy

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from examples.seismic import plot_velocity_am, plot_perturbation
from devito import gaussian_smooth
from examples.seismic import Model

from examples.seismic.acoustic import AcousticWaveSolver

#from examples.seismic import plot_image_am, plot_image
#from examples.seismic import plot_velocity
from utils import plot_velocity, plot_shotrecord
from operators import calculate_alfa, first_deriv, second_deriv, Laplacian

from devito import configuration
configuration['log-level'] = 'WARNING'

from devito import TimeFunction, Operator, Eq, solve
from examples.seismic import PointSource
from devito import Function, TimeFunction

font = {
#      'family' : 'Tahoma',
#     'weight' : 'bold',
      'size'   : 12}
plt.rc('font', **font)


#---------------------------------------------------------

#shape = (101, 101)
shape = (602, 241)
spacing = (15, 15)
nreceivers = 101
#nreceivers = 1
nshots = 1
if nshots <=10 :
    vmin=-1.*1e01
    vmax=1.*1e01
else:
    vmin=-1.*1e02
    vmax=1.*1e02

f0 = .01
t0 = 0.
tn = 3500.
nbl=60

inpath = '/resstore/b0181/Data/Tyra/tyra-model-v1.mat'
#outpath= '/home/home01/earamoh/Postdoc/src/IDWT/test/figs_Tyra-v1s%s' % nshots
outpath= '/home/home01/earamoh/Postdoc/src/IDWT/test/test' 
# Create true model
#Vp_orig = sio.loadmat('/nobackup/earamoh/Tyra/Tyra-Sjoerd/tyra-model-v1.mat')    # arc4
Vp_orig = sio.loadmat(inpath)
#Vp_orig = sio.loadmat('/nobackup/earamoh/Models/tyra-model.mat')    # arc3
# Vp = ndimage.rotate(Vp_orig['velocity'], -180)
Vp = Vp_orig['velocity']

#
model = Model(vp=Vp, origin=(0.,0.), shape=shape, spacing=spacing,
             space_order=2, nbl=nbl, bcs="damp")

# Create initial model and smooth the boundaries
model0 =  copy.deepcopy(model)
gaussian_smooth(model0.vp, sigma=(6,6))
# Plot the true and initial model and the perturbation between them
#from matlab_colormap import parula_cmap
plot_velocity(model, outpath, 'baseline_model')

# Create monitor model

Vp_m = Vp

Vp_m[Vp_m >= 4.0]  = 4.02


model_m = Model(vp=Vp_m, origin=(0.,0.), shape=shape, spacing=spacing,
             space_order=2, nbl=nbl, bcs="damp")
plot_velocity(model_m, outpath, 'monitor_model')


model_diff = copy.deepcopy(model_m)
Vp_diff = copy.copy(Vp_m)
Vp_diff[Vp_diff == 4.02] = .02
Vp_diff[Vp_m < 4.02] = 0.
print(len(Vp_diff[Vp_diff == .02]))
model_diff.vp.data[nbl:-nbl,nbl:-nbl] = Vp_diff[:]
plot_velocity(model_diff, outpath, 'diff_model')


# Define acquisition geometry: source
from examples.seismic import AcquisitionGeometry


# Prepare the varying source locations
source_locations = np.empty((nshots, 2), dtype=np.float32)
if nshots == 1:
    source_locations[:, 0] = (shape[0] * spacing[0]) / 2.
    #source_locations[:, 0] = (shape[0] * spacing[0]) / 4.
elif nshots == 3:
    xlim = shape[0] * spacing[0]
    source_locations[:, 0] = np.array([xlim * .25, xlim * .5, xlim * .75]) 
else:
    source_locations[:, 0] = np.linspace(0., shape[0] * spacing[0], num=nshots)
source_locations[:, 1] = 20.

# First, position source centrally in all dimensions, then set depth
src_coordinates = np.empty((1, 2))
if nshots == 1:
    src_coordinates[0, :] = source_locations[:,:]
else:
    src_coordinates[0, :] = source_locations[int(nshots/2)+1,:]
#src_coordinates[0, 1] = 20.  # Depth is 20m

# Define acquisition geometry: receivers

# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers,2))
if nreceivers == 1:
    rec_coordinates[:, 0] = model.domain_size[0] * 3./4
else:
    rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec_coordinates[:, 1] = 30.
#print(rec_coordinates[0, :])
#quit()

# Geometry

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')
# We can plot the time signature to see the wavelet
#geometry.src.show()

#geometry.rec_positions[:,0] = np.linspace(0, model.domain_size[0], num=nreceivers)
#geometry.rec_positions[:,1] = 30.
#print(geometry.rec_positions)

from utils import plot_velocity
plot_velocity(model0,outpath, 'base_init_model', \
                   #source=source_locations[::5, :],
                   source=source_locations[:, :],
                   receiver=geometry.rec_positions)  #[::2, :])


#-------------------------------------------------------

from examples.seismic.acoustic import AcousticWaveSolver

solver = AcousticWaveSolver(model, geometry, space_order=4)
true_d , _, _ = solver.forward(vp=model.vp)
plot_shotrecord((true_d.data), model, \
                        t0, tn, f0, nshots, \
                        outpath, 'RecBeforeMask', cmap='gray')
quit()
#Vp_m = Vp
#Vp_m[Vp_m <= 1.2] = 1.225


#model_m = Model(vp=Vp_m, origin=(0.,0.), shape=shape, spacing=spacing,
#             space_order=2, nbl=nbl, bcs="damp")



#solver_ = AcousticWaveSolver(model_m, geometry, space_order=4)
#true_d_m , _, _ = solver_.forward(vp=model_m.vp)
#plot_shotrecord((true_d.data - true_d_m.data), model, t0, tn, f0, 1,  outpath)
#quit()



# Define gradient operator for imaging
from devito import TimeFunction, Operator, Eq, solve
from examples.seismic import PointSource

def ImagingOperator(model, image):
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





# Run imaging loop over shots
from devito import Function

# Create image symbol and instantiate the previously defined imaging operator
image = Function(name='image', grid=model.grid, space_order=2)
#image = TimeFunction(name='image', grid=model.grid, time_order=1, space_order=2)
op_imaging = ImagingOperator(model, image)

for i in range(nshots):
    print('Imaging source %d out of %d' % (i+1, nshots))

    # Update source location
    geometry.src_positions[0, :] = source_locations[i, :]

    # Generate synthetic data from true model
    true_d, _, _ = solver.forward(vp=model.vp)
    #if i == int(len(source_locations)/2):
    #if i == 1:
    if i == 0:
        u_b = TimeFunction(name="u_", grid=model.grid, 
                 time_order=2, space_order=8,
                 save=geometry.nt)
        true_d_b, u_b, _ = solver.forward(vp=model.vp, save=True)

    # Compute smooth data and full forward wavefield u0
    smooth_d, u0, _ = solver.forward(vp=model0.vp, save=True)
    print(u0.data.shape)
    # Compute gradient from the data residual  
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)
    residual = smooth_d.data - true_d.data
    op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt, 
               residual=residual)

#-----------------------------------------------------------------
#
# manual Laplacian
#newimage = 4 * image.data[1:-1,1:-1]- image.data[0:-2,1:-1] - image.data[2:,1:-1] - image.data[1:-1,0:-2] - image.data[1:-1,2:]
#newimage = newimage /(model.spacing[0] * model.spacing[0])
imageb =  Laplacian(image, model)
#
#print(newimage.shape, image.shape, nbl)
from utils import plot_image
plot_image(imageb.data, model, outpath, 'baseline', vmin=vmin, vmax=vmax, cmap='gray')
#plt.figure()
#plt.imshow(np.transpose(newimage[(nbl-1):-(nbl-1),(nbl-1):-(nbl-1)]), cmap = 'gray',\
#           vmin = vmin, vmax=vmax)
#plt.colorbar(shrink=.5)
#plt.savefig(os.path.join(outpath, 'baseline_image_ManualLpalacian.png'))



#-----------------------------------------------------------------
# Create monitor model

#Vp_m = Vp

#Vp_m[Vp_m >= 4.0]  = 4.1


#model_m = Model(vp=Vp_m, origin=(0.,0.), shape=shape, spacing=spacing,
#             space_order=2, nbl=nbl, bcs="damp")
#plot_velocity(model_m, outpath, 'monitor_model')

# create monitor image

print('Creating monitor migration image')

image_m = Function(name='image', grid=model_m.grid, space_order=2)
op_imaging = ImagingOperator(model0, image_m)

for i in range(nshots):
    print('Imaging source %d out of %d' % (i+1, nshots))

    # Update source location
    geometry.src_positions[0, :] = source_locations[i, :]

    # Generate synthetic data from true model
    true_d, _, _ = solver.forward(vp=model_m.vp)
    
    #if i == int(len(source_locations)/2):
    #if i == 1:
    if i == 0:
        u_m = TimeFunction(name="u_", grid=model_m.grid, 
                 time_order=2, space_order=8,
                 save=geometry.nt)
        true_d_m, u_m, _ = solver.forward(vp=model_m.vp, save=True)

    # Compute smooth data and full forward wavefield u0
    smooth_d, u0, _ = solver.forward(vp=model0.vp, save=True)

    # Compute gradient from the data residual  
    v = TimeFunction(name='v', grid=model_m.grid, time_order=2, space_order=4)
    residual_m = smooth_d.data - true_d.data
    op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt, 
               residual=residual_m)

#-----------------------------------------------------------------
#
#plt.figure()
#plt.imshow(ndimage.rotate(np.diff(image_m.data, axis=1)[nbl:-nbl,nbl:-nbl], -90), cmap='gray', \
#                                  vmin=-1.e04,vmax=1.e04)
#plt.colorbar()
#plt.savefig(os.path.join(outpath, 'monitor_image.png'))
##quit()

# Laplacian

#newimage_m = 4 * image_m.data[1:-1,1:-1]- image_m.data[0:-2,1:-1] - image_m.data[2:,1:-1] - image_m.data[1:-1,0:-2] - image_m.data[1:-1,2:]
#newimage_m = newimage_m /(model.spacing[0]*model.spacing[0])

imagem = Laplacian(image_m, model_m)

plot_image(imagem.data, model_m, outpath, 'monitor', vmin, vmax, cmap='gray')
#plt.figure()
#plt.imshow(np.transpose(newimage_m[(nbl-1):-(nbl-1),(nbl-1):-(nbl-1)]),\
#          cmap = 'gray', vmin=vmin, vmax=vmax)
#plt.colorbar(shrink=.5)
#plt.savefig(os.path.join(outpath, 'monitor_image_Manual_Lpalacian.png'))
#

#------------------------------------------------movie
from wave_animation import wave_screenshot, make_animation

make_animation(u_b, model, geometry, outpath, 'base')
make_animation(u_m, model_m, geometry, outpath, 'monitor')
#screen shot

wave_screenshot(u_b, model, geometry, 500, outpath, 'base')
wave_screenshot(u_b, model, geometry, 800, outpath, 'base')
wave_screenshot(u_m, model_m, geometry, 1000, outpath, 'monitor')
#----------------------------------------------------------------------------

plot_shotrecord((true_d_b.data - true_d_m.data), model, t0, tn, f0, 1,  outpath, 'difference', cmap='gray')
plot_shotrecord(true_d_b.data, model, t0, tn, f0, 1,  outpath, 'baseline', cmap='gray')
plot_shotrecord(true_d_m.data, model_m, t0, tn, f0, 1,  outpath, 'monitor', cmap='gray')
##------------------------------------------------------------
## calculate warping: w(x,z,Xs)
#from utils import calculate_warping
from warp_function import warp_function

# quit()

#imageb = copy.deepcopy(image)
#imageb.data[1:-1,1:-1] = newimage[:,:]

#imagem = copy.deepcopy(image_m)
#imagem.data[1:-1,1:-1] = newimage_m[:,:]


#from scipy.io import savemat
#mdic = {"base": imageb.data[nbl:-nbl,nbl:-nbl], "monitor": imagem.data[nbl:-nbl,nbl:-nbl]}
#savemat(os.path.join( outpath, "both_images_laplacian_%s.mat" % nshots), mdic)


#mdic_ = {"base": image.data[nbl:-nbl,nbl:-nbl], "monitor": image_m.data[nbl:-nbl,nbl:-nbl]}
#savemat(os.path.join( outpath, "both_images_%s.mat" % nshots), mdic_)

# save shot records

#d_dic = {"db": true_d_b.data, "dm": true_d_m.data}
#savemat(os.path.join( outpath, "both_data_%s.mat" % nshots), d_dic)

#quit()

# save to text file
#np.savetxt('outfiles/baseline.txt', imageb.data[nbl:-nbl,nbl:-nbl])
#np.savetxt('outfiles/monitor.txt', imagem.data[nbl:-nbl,nbl:-nbl])


#plt.figure()
#plt.imshow(ndimage.rotate(imageb.data[nbl:-nbl,nbl:-nbl], -90), vmin=-2.*1e02, vmax=2.*1e02, cmap='gray')
#plt.colorbar(shrink=.5)
#plt.savefig(os.path.join(outpath, 'Baseline_image.png'))
#
#plt.figure()
#plt.imshow(ndimage.rotate(imagem.data[nbl:-nbl,nbl:-nbl], -90), vmin=-2.*1e02, vmax=2.*1e02, cmap='gray')
#plt.colorbar(shrink=.5)
#plt.savefig(os.path.join(outpath, 'Monitor_image.png'))


#plot_warp(imageb.data[nbl:-nbl,nbl:-nbl], imagem[nbl:-nbl,nbl:-nbl], 240, outpath, dtw_mod=2)

warp_arr = np.zeros(imagem.shape)
#warp_arr[nbl:-nbl, nbl:-nbl], warp_img  = warp_function(imagem.data[nbl:-nbl,nbl:-nbl], imageb.data[nbl:-nbl,nbl:-nbl])

# temporary


#from warp_function import run_matlab_warp
warp_img = np.zeros(imagem.shape)

#warp_arr.T[nbl:-nbl, nbl:-nbl], warp_img.T[nbl:-nbl, nbl:-nbl] = run_matlab_warp(imagem, imageb)

matwarpath = '/home/home01/earamoh/Postdoc/src/IDWT/Warping/outfiles'


warp_arr[nbl:-nbl:,nbl:-nbl] = sio.loadmat(os.path.join(matwarpath, 'shift_%s.mat' % nshots))['shift_%s' % nshots].T

warp_img[nbl:-nbl:,nbl:-nbl] = sio.loadmat(os.path.join(matwarpath, 'mshift_%s.mat' % nshots))['mshift_%s' % nshots].T

plot_image(warp_img, model_m, outpath, 'warped monitor', vmin, vmax, cmap='gray')
#plt.figure()
#plt.imshow(np.transpose(warp_img[nbl:-nbl,nbl:-nbl]), vmin=vmin, vmax=vmax, cmap='gray')
#plt.colorbar(shrink=.5)
#plt.savefig(os.path.join(outpath, 'warped_monitor_image.png'))
#plt.show()


#alfa = np.zeros(warp_arr.shape)
#alfa[nbl:-nbl,nbl:-nbl] = calculate_alfa(imagem, imageb, model, warp_arr)

# use the following: after changing the function in operator due to dividing by zero

alfa = calculate_alfa(imagem, imageb, model, warp_arr)
print(alfa.shape, warp_arr.shape)

alfa_copy = np.zeros(alfa.shape)
ix = int(alfa.shape[0]/2)
iz = 70
alfa_copy[ix,iz] = 5.0 

#plt.figure()
#plt.imshow(np.transpose(alfa), vmin=-1e-1, vmax=1e-1)
#plt.colorbar(shrink=.5)
#plt.savefig(os.path.join(outpath, 'alfa.png'))

# gradient
from operators import compute_gradient
#gradient = compute_gradient(model0, geometry, source_locations, alfa)
gradient = compute_gradient(model0, geometry, source_locations, alfa_copy)

print(np.min(gradient.data), np.max(gradient.data), gradient.data.shape)


#plot_image(gradient.data[nbl:-nbl,nbl:-nbl], model0, outpath, 'gradient_image',
# vmin=-2.e0, vmax=2.e0, cmap='seismic')

plt.figure()
plt.imshow(np.transpose(gradient.data[nbl:-nbl,nbl:-nbl]), cmap='seismic', vmin=-100,vmax=100)
plt.colorbar()
plt.savefig(os.path.join(outpath, 'grad_image.png'))
