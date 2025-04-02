import os
import numpy as np
from scipy import ndimage
from scipy.signal import butter, filtfilt

import copy

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


from examples.seismic import Model
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from devito import TimeFunction, Operator, Eq, solve
from examples.seismic import PointSource
from examples.seismic.acoustic import AcousticWaveSolver
from devito import Function

#------------------------------------- Geometry acuisition
def acquisition_geometry(model, inpara):

    # Prepare the varying source locations
    source_locations = np.empty((inpara.nshots, 2), dtype=np.float32)
    source_locations[:, 0] = np.linspace(0., inpara.shape[0] * inpara.spacing[0], num=inpara.nshots)
    source_locations[:, 1] = inpara.src_depth

    # First, position source centrally in all dimensions, then set depth
    src_coordinates = np.empty((1, 2))
    src_coordinates[0, :] = source_locations[int(inpara.nshots/2)+1,:]

    # Define acquisition geometry: receivers

    # Initialize receivers for synthetic and imaging data
    rec_coordinates = np.empty((inpara.nrec,2))
    rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=inpara.nrec)
    rec_coordinates[:, 1] = inpara.rec_depth

    # Geometry

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, inpara.t0, inpara.tn, f0=inpara.f0, src_type='Ricker')

    return geometry, source_locations
#------------------------------------------ plo_velocity
def plot_velocity(model, outpath, outfilename,\
                   source=None, receiver=None, colorbar=True, cmap='jet'):

    """
    Plot a two-dimensional velocity field from a seismic `Model`
    object. Optionally also includes point markers for sources and receivers.

    Parameters
    ----------
    model : Model
        Object that holds the velocity model.
    source : array_like or float
        Coordinates of the source point.
    receiver : array_like or float
        Coordinates of the receiver points.
    colorbar : bool
        Option to plot the colorbar.
    """
    from matlab_colormap import parula_cmap
    #cmap = parula_cmap() ; vmin=1.5; vmax=4.0
    if cmap == 'jet':
         vmin=1.0
         vmax=4.6
    plt.figure()


    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, 'vp', None) is not None:
        field = model.vp.data[slices]
    else:
        field = model.lam.data[slices]
    if outfilename == 'diff_model':
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["white", "red"])
        #cmap="jet"
        plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=0., vmax=np.max(field),
                      extent=extent)
    else:
        plot = plt.imshow(np.transpose(field), animated=True, cmap=cmap,
                      vmin=vmin, vmax=vmax,                     
                      extent=extent)
    if outfilename == 'monitor_model':
        ax = plt.gca()
        levels = [4.0, 4.02]
        ax.contour(np.transpose(field), levels=levels, colors='gray', origin='upper', 
                   extent=extent)

     
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')

    # Plot source points, if provided
    if receiver is not None:
        plt.scatter(1e-3*receiver[:, 0], 1e-3*receiver[:, 1],
                    s=25, c='green', marker='D')

    # Plot receiver points, if provided
    if source is not None:
        plt.scatter(1e-3*source[:, 0], 1e-3*source[:, 1],
                    s=25, c='red', marker='o')

    # Ensure axis limits
    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])
    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if outfilename == 'diff_model':
            cbar = plt.colorbar(plot, cax=cax, ticks=[0.,np.max(field)])
        else:
            cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label('Velocity (km/s)')

    if source is None:
        plt.savefig(os.path.join(outpath, '%s.png' % outfilename))
    else:
        plt.savefig(os.path.join(outpath, '%s_%srec%sm_%ssrc%sm.png' % (outfilename, \
	     len(receiver), receiver[0, 0], len(source), source[0,0] )))

    plt.close()
#--------------------------------------------

def plot_shotrecord(rec, model, t0, tn, f0, numsrc,  outpath, img_name , cmap, colorbar=True):
    """
    Plot a shot record (receiver values over time).

    Parameters
    ----------
    rec :
        Receiver data with shape (time, points).
    model : Model
        object that holds the velocity model.
    t0 : int
        Start of time dimension to plot.
    tn : int
        End of time dimension to plot.
    """
    plt.figure()
    scale = np.max(rec)/10.e3
    #scale = 1.
    extent = [model.origin[0], model.origin[0] + 1e-3*model.domain_size[0],
              1e-3*tn, t0]

    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cmap, extent=extent)
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')
    plt.title('%s' % img_name)

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if img_name == 'shift':
            cbar.set_label('Shift Value (s)')
        else:
            print('')

        plt.colorbar(plot, cax=cax)
    # set aspect ration: y is much shorter thatn x axis
    #ratio = 1.8
    #x_left, x_right = ax.get_xlim()
    #y_left, y_right = ax.get_ylim()
    #ax.set_aspect(abs((x_left - x_right)/(y_left - y_right))*ratio)

   # plt.show()
    plt.savefig(os.path.join(outpath, \
               'shotrecord_%s_%s-rec_%s-src.png'% (img_name, rec.shape[1], numsrc)))


#-----------------------------------------------
def plot_image(data, model, outpath, image_name, vmin, vmax, cmap='gray'):

    domain_size = 1.e-3 * np.array(model.domain_size)
    extent = [model.origin[0], model.origin[0] + domain_size[0],
              model.origin[1] + domain_size[1], model.origin[1]]
    nbl = model.nbl
    plt.figure()
    print(vmin, vmax)
    plt.imshow(np.transpose(data[nbl:-nbl,nbl:-nbl]),\
                cmap=cmap, vmin=vmin,vmax=vmax, extent=extent)
    plt.title(image_name)
    plt.xlabel('X position (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(shrink=.5)
    plt.savefig(os.path.join(outpath, '%s_image.png' % image_name))
    plt.close()
#----------------------------------- make adjoint source coordinates

def grid_coord(model):

    nx, ny = model.shape
    x = np.linspace(0, model.domain_size[0], nx)
    y = np.linspace(0, model.domain_size[1], ny)
    xv, yv = np.meshgrid(x, y)

    X = xv.flatten()
    Y = yv.flatten()

    coord = np.vstack((X,Y)).T

    return coord

#------------------------------------------ smoothing_function
def smoothing_function(img, boxcar_width=None, gauss_sigma=None, hanning_window=None):

    if boxcar_width is not None:
        # a uniform (boxcar) filter with a width of boxcar_width
        boxcar = ndimage.uniform_filter1d(img, boxcar_width, 1)
        return boxcar

    if gauss_sigma is not None:
        # a Gaussian filter with a standard deviation of gauss_sgima
        gauss = ndimage.gaussian_filter1d(img, gauss_sigma, 1)
        return gauss

    if hanning_windon is not None:
        kern = np.hanning(hanning_window)   # a Hanning window with width hanning_window
        kern /= kern.sum()      # normalize the kernel weights to sum to 1
        hanning = ndimage.convolve1d(img, kern, 1)    
        return hanning

#--------------------------------------smooth_gradient
def smooth_gradient(arr, n):
    '''
    from Sjoerd's mode_smooth.f90
    x, y: wrpgrd, totgrd1: gradient and smooth gradient
    n1,n2: nz,nx
    n: nsmooth
   
 '''    
    #print(arr.shape)
    #n2, n1 = x.shape
    #x = x.flatten()
    #y = np.zeros(x.shape)
    n2, n1 = arr.shape
    x = []
    for i2 in range(n2):
        for i1 in range(n1):
           x.append(arr[i2, i1])

    x = np.array(x)
    y = np.zeros(x.shape)

    for i2 in range(n2):
        for i1 in range(n1):
            fold = 0
            for jj in range(max(i2-n, 0), min(i2+n, n2)):
                for ii in range(max(i1-n, 0), min(i1+n, n1)):
                    y[i2* n1 + i1] = y[i2 * n1 + i1] + x[jj * n1 + ii]
                    fold += 1
            y[i2 * n1 + i1] = y[i2 * n1 + i1] / fold
        
    y = y.reshape((n2, n1))
    return y

#------------------------------------plot_toy_velocity
def plot_toy_velocity(model, inpara, filename, vmin=None, vmax=None):

    #print('Plot %s' %filename)
    nbl = model.nbl
    field = model.vp.data[nbl:-nbl, nbl:-nbl]
    plot = plt.imshow(np.transpose(field), cmap = 'jet', \
               vmin = vmin, vmax = vmax)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.set_label('Velocity (km/s)')

    plt.savefig(os.path.join(inpara.outpath, '%s.png' % filename))

    plt.close()

#-----------------------------------------------smooth_image
def smooth_using_kdTree(arr, model, inpara):

    
    nx, ny = model.shape
    #x = np.linspace(0, model.domain_size[0], nx)
    #y = np.linspace(0, model.domain_size[1], ny)

    x = []
    y = []
    for ix in range(nx):
        for iz in range(ny):
            x.append(ix * model.spacing[0])
            y.append(iz * model.spacing[1])
    x = np.array(x)
    y = np.array(y)

    xv, yv = np.meshgrid(x, y)
    print('smoothing function')

    # memory error when using all the mesh points to make the tree
    #from scipy import spatial
    #tree = spatial.KDTree(list(zip(xv.ravel(), yv.ravel())), leafsize=5)
    tree = spatial.cKDTree(list(zip(xv.ravel(), yv.ravel())))
    # 
    arr = arr.flatten()
    smooth_arr = np.zeros(arr.shape)
    print('KDTree done'); quit()

    for ix in range(nx):
        for iy in range(ny):
            xx = x[ix]
            yy = y[iy]
            point = np.array([xx, yy])
            idx = tree.query(point)[1]
            #nearest_points = tree.data[[e for e in idx]] 

            print(idx.shape); quit()       
            smooth_arr[ix * nx +iy] = np.sum(arr[idx]) /len(idx)         

    return smooth_arr 

#-------------------masking_function
def mask_function(warp, inpara, iter_num, baseline_velocity=None, updated_velocity=None):

    n1, n2 = warp.shape
    mask = np.zeros(warp.shape)
    phi = np.zeros(warp.shape)

    phi[:,0:-1] = np.diff(warp, axis=1)
    
    if iter_num == 1:
        for i1 in range(n1):
            for i2 in range(n2):
                for jj in range(0, n1):
                    for ii in range(0, n2):
                         #print(i1,i2,jj,ii);
                         mask[i1, i2] = mask[i1, i2] + phi[jj, ii]
            mask[i1, i2] = mask[i1, i2] / (4 * n1*n2)

    if iter_num > 1:
        for i1 in range(n1):
            for i2 in range(n2):
                for jj in range(max(i1-n1, 0), min(i1+n1, n1)):
                    for ii in range(max(i2-n2, 0), min(i2+n2, n2)):
                         mask[i1, i2] = mask[i1, i2] + \
                         abs((updated_velocity[jj+i1, ii+i2] - baseline_velocity[jj+i1, ii+i2]) \
                            / baseline_velocity[jj+i1, ii+i2])
            mask[i1, i2] = mask[i1, i2] / (4 * n1*n2)  

    plt.figure()
    plt.imshow(np.transpose(mask), cmap='seismic', \
               vmin=-np.max(mask), vmax=np.max(mask))
    plt.colorbar(shrink=.5)
    plt.savefig(os.path.join(inpara.outpath, 'kotsi_mask_%s.png' %iter_num))
    plt.close()

    return mask

#---------------------------------bandpass_filter
def bandpass_filter(data, inpara):
    if inpara.filter_data_info[3]:
        print('Filtering data')
        filtered_data = bandpass_filter_1d(data, inpara)
    if inpara.filter_image_info[3]:
        filtered_data = bandpass_filter_2d(data, inpara)
    return 
#---------------------------------bandpass rtm image
def bandpass_filter_2d(data, inpara): #lowcut, highcut, fs, order=4):
    """
    Apply a band-pass filter to the data.
    """
    power_spectrum_data = power_spectrum_2d(data)
    
    # filtering
    nyquist = 0.5 * inpara.filter_image_info[0]
    low = inpara.filter_image_info[1] / nyquist
    high = inpara.filter_image_info[2] / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)  # Apply along depth axis

    return filtered_data
#-------------------------------------------- power_spectrum
def power_spectrum_2d(image):
    # Compute the 2D Fourier Transform
    fft_image = np.fft.fft2(image)
    # Shift the zero frequency to the center
    fft_image_shifted = np.fft.fftshift(fft_image)
    # Compute the power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_image_shifted) ** 2
    return power_spectrum

# Compute power spectra for the baseline and monitor RTM images
#power_spectrum_baseline = compute_power_spectrum(rtm_baseline)
#power_spectrum_monitor = compute_power_spectrum(rtm_monitor)

# Average the power spectra for simplicity, or analyze them separately
#power_spectrum_avg = (power_spectrum_baseline + power_spectrum_monitor) / 2

#--------------------------------------------bandpass_filter_1d
def bandpass_filter_1d(seismic_data, inpara):
    """
    Apply a band-pass filter to a 1D seismic data array.
    """
    filtered_seismic_data = []
    nyquist = 0.5 * inpara.filter_data_info[0]
    low = inpara.filter_data_info[1] / nyquist
    high = inpara.filter_data_info[2] / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')

    for i in range(seismic_data.shape[1]):
        filtered_data = filtfilt(b, a, seismic_data[:,i])
        filtered_seismic_data.append(filtered_data)

    filtered_seismic_data = np.array(filtered_seismic_data)

    # plot the middle seismic trace
    i = int(filtered_seismic_data.shape[0]/2)
    plt.figure(figsize=(12, 6))
    # Plot original data
    plt.subplot(2, 1, 1)
    plt.plot(seismic_data[:,i], label='Original Data')
    plt.title(f'Receiver {i} - Original Data')
    plt.xlabel('Time Samples')
    plt.ylabel('Amplitude')
    plt.legend()
        
    # Plot filtered data
    plt.subplot(2, 1, 2)
    plt.plot(filtered_seismic_data[:,i], label='Filtered Data (Band-Pass)', color='orange')
    plt.title(f'Receiver {i} - Filtered Data')
    plt.xlabel('Time Samples')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(inpara.outpath, 'filtered_seismic_data_rec_%s.png' %i))

    return filtered_seismic_data
#--------------------------------------------power_spectrum_1d
def power_spectrum_1d(data):
    """
     # Compute power spectrum for each receiver and stack into an array
    """
    power_spectra = []
    for i in range(data.shape[1]):
        fft_data = np.fft.fft(data[:,i])
        power_spectrum = np.abs(fft_data) ** 2
        power_spectra.append(power_spectrum[:len(fft_data)//2])

    # Convert to a 2D array (receiver x frequency)
    power_spectra = np.array(power_spectra)

    # Create a heatmap of the power spectrum across receivers
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.log(power_spectra + 1), \
                cmap='viridis', cbar_kws={'label': 'Log Power Spectrum'})
    plt.xlabel('Frequency Index')
    plt.ylabel('Receiver Index')
    plt.title('Power Spectrum Across Receivers')
    plt.savefig(os.path.join*outpath, 'receiver_data_spectrum_heatmap')


