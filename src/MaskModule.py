import os
import numpy as np
from scipy import signal
import scipy.io as sio
from examples.seismic import Model
from examples.seismic.acoustic import AcousticWaveSolver
import matplotlib.pyplot as plt
#--------------------------------make receivers mask

def mask_topmute_op(model, rec_data, inpara, geometry):

    #Vp_orig = sio.loadmat(inpara.model_path)
    #Vp_orig = Vp_orig['velocity']
    nbl = model.nbl
    Vp_orig = model.vp.data[nbl:-nbl, nbl:-nbl]
    #Vp_water = np.mean(Vp_orig[:,0:100])
    Vp_water = 2.
    Vp = Vp_water * np.ones(Vp_orig.shape)
    #print(np.mean(Vp))
    model = Model(vp=Vp, origin=inpara.origin, shape=inpara.shape, spacing=inpara.spacing,
                      space_order=inpara.space_order, nbl=inpara.nbl, bcs=inpara.bcs
                      , fs=False)
    solver = AcousticWaveSolver(model, geometry, space_order=4)
    d, _, _ = solver.forward(vp=model.vp)

    mute_time = 250
    for ir in range(rec_data.shape[1]):
        for it in range(rec_data.shape[0]):
            if it < mute_time:
                rec_data[it, ir] = 0
    return rec_data
#---------------------------------
def make_receiver_mask(rec_data, model, geometry):

    '''

    # From Sjoerd Fortran code/ proccessing_mod.f90
    # Mask receiver data
    '''
    dz, dx = model.spacing
    nrec = len(geometry.rec_positions)
    tramp = int(model.critical_dt) #int(.1/model.critical_dt) # ramp length (should probably be external)
    mute_time_src = .2  #0.2  # mute time

    vp = 0.
    vp_in = np.zeros(model.vp.data.shape)
    vp_in[:,:] = model.vp.data[:,:]
    nz, nx = vp_in.shape[0], vp_in.shape[1]
    vp_in = vp_in.flatten()

    
    for ix in range(nx):
        #vp = vp + 1./sqrt(vp_in((ix-1)*nz+1))
        #vp = vp + 1./np.sqrt(vp_in[ix * nz + 1])
        vp = vp + vp_in[ix * nz]
    
    vp = vp / nx  # long mute
    #vp = 2.7  

    # calculate depth to first reflector
    d0 = nz
    for ix in range(nx):
        #iz = 1; d1 = nz
        iz = 0; d1 = nz -1 
        while (d1 == nz -1) and (iz < nz-1):
            #if 1./np.sqrt(vp_in[ix * nz +iz]) > vp:
            if vp_in[ix * nz +iz] > vp:
                d1=iz
            iz += 1

        if d0 > d1:
            d0 = d1
    d = d0

    src_positions = np.empty(geometry.src_positions.shape)
    #create mask
    rec_positions = np.empty(geometry.rec_positions.shape)
    mute_time = np.zeros(len(rec_positions))
    mask = np.zeros(rec_data.shape)

    for ishot in range(len(geometry.src_positions)):

        src_positions[ishot, 0] = 1.5 + geometry.src_positions[ishot,0] /dz
        src_positions[ishot, 1] = 1.5 + geometry.src_positions[ishot,1]/ dx
        for ir in range(nrec):
            rec_positions[ir,0] = 1.5 + geometry.rec_positions[ir,0]/ dz
            rec_positions[ir,1] = 1.5 + geometry.rec_positions[ir, 1]/ dx
          
            # calculate source receiver separation - assumes s
            xdis_src_rec = abs(src_positions[0,1] - rec_positions[ir,1]) * dx
            # calculate reflection distance
            zdis = (d - src_positions[0,0]) * dz
            reflection_dis = np.sqrt((2 * zdis) ** 2 + xdis_src_rec ** 2)

            # calculate mute time
            xdis_src_rec = xdis_src_rec + mute_time_src * vp
            if reflection_dis < xdis_src_rec: 
                reflection_dis = xdis_src_rec

            mute_time[ir] = int(reflection_dis / vp /model.critical_dt)
            if mute_time[ir] < 1:
                mute_time[ir] = 1  #why?
            #print(rec_data[:,ir] - mute_time[ir],  model.critical_dt); quit()

            # make mask
            j = 0; k=0
            #print(mute_time[ir], rec_data.shape[0], tramp)
            for it in range(rec_data.shape[0]):
                if it <= mute_time[ir]:
                    k += 1
                    mask[it, ir] = 0
                elif mute_time[ir] < it and it <= mute_time[ir] + tramp+1:
                    j += 1
                    mask[it, ir] = (it - mute_time[ir] - 1) / tramp
                elif it > mute_time[ir] + tramp+1:
                    mask[it, ir] = 1
    return mask

#--------------------------------
def mask_topmute_op_wrong(rec_data, model, geometry):
    '''
    #apply top mute mask !!!!!!!!
    '''
    
    masked_rec_data = np.ones(rec_data.shape)
    mask = make_receiver_mask(rec_data, model, geometry)
    #print(mask[mask[:,0]==0].shape,rec_data.shape); quit()
    for i in range(rec_data.shape[1]):
        masked_rec_data[:,i] = mask[:,i] * rec_data[:,i]
    
    return masked_rec_data

#-------------------------------- mask
def mask_data(data, model, inpara):

    mask = np.ones(data.shape)
    #print(mask.shape)
    for ix in range(model.shape[0]):
        for iz in range(model.shape[1]):
            xx = ix * model.spacing[0]
            zz = iz * model.spacing[1]

            if xx <= inpara.mask_xmax and xx >= inpara.mask_xmin:
                if zz >= inpara.mask_zmin and zz <= inpara.mask_zmax:
                    mask[ix,iz] = 0.
                    data[ix, iz] = 0.

    from utils import plot_image
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", "k"])
    plt.figure()
    plt.imshow(np.transpose(mask), cmap=cmap)
    plt.colorbar(shrink=.5)
    plt.savefig(os.path.join(inpara.outpath, 'warp_mask.png'))
    
    return mask, data


#------------------------------agc_nlop_for
def agc_op(data, window_length):
    '''
    Calculate weightings for agc applied to to data/ image
    From Sjoerd's agc_nlop_for Fortran code
    '''
    nx, nz = data.shape
    data = data.flatten()
    n = len(data)
    gain = np.zeros(data.shape)
    if window_length % 2 == 0: 
        window_length += 1
    # calculate desired mean !!!
    data_mean = np.sum(np.abs(data)) / n
    # first half of data !!!
    for i in range(int(np.ceil(window_length/2.))):
        li = (i+1) * 2 - 1
        if np.sum(np.abs(data[0:li-1])) == 0:
            gain[i] = 1
        else:
            gain[i] = data_mean * li/np.sum(np.abs(data[0: li-1]))
    
    #normal window !!!
    for i in range(n - window_length -1):
        idx = i + int(np.floor(window_length/2. - 1))
        gain[idx] = data_mean * window_length \
                    / np.sum(np.abs(data[i:i+window_length-1]))
        if np.sum(np.abs(data[0:li-1])) == 0: 
            gain[i] = 1
    
    # last l/2 data !!!
    for i in range(int(np.ceil(window_length/2.)) - 1, 0, -1):
        li = (i+1) * 2 -1
        idx = n - i 
        gain[idx] = data_mean * li \
                  / np.sum(np.abs(data[n-li+1:n-1]))
        if np.sum(np.abs(data[0:li-1])) == 0: 
            gain[i] = 1
    
    new_data = agc_lop(data, gain)

    return new_data.reshape((nx,nz)), gain.reshape((nx, nz))
#-------------------------------  
def agc_lop(data , gain):
    '''
    adjoint and forward based on linearisation
    '''
    new_data = data * gain

    return new_data

#--------------------------------------------frequency_spectra
# Define a basic function to calculate frequency spectra
# As we are using an fft we should really taper or window the data to prevent edge effects, 
# but for this demonstration we are going to conveniently ignore this

# An FFT returns a vector of complex values, in this case we are using the RFFT option to just return the real values 

def frequency_spectra(data, dt, fft_ax):
    """
    Calculate the frequency spectra
    """
    # Amplitude values
    
    # Get the absolute value of the Fourier coefficients
    fc = np.abs(np.fft.rfft(data, axis = fft_ax))
               
    # Take the mean to get the amplitude values of the spectra
    a = np.mean(fc) #, axis = (0, 1))
    #print(a, fc.shape, np.max(fc)); quit()
    # Get the frequency values corresponding to the coefficients
    # We need the length of the window and the sample interval in seconds 
    #dt = 4            
    dts = dt / 1000
    length = data.shape[fft_ax]
                
    f = np.fft.rfftfreq(length, d = dts)
    #print(f.shape, fc[1,:].shape); quit() 
    
    return f, fc

#----------------------------------------plot_spectra
def plot_frequency_spectra(datab, datam, dt, fft_ax, inpara, filename):

    freq, ampb = frequency_spectra(datab, dt, fft_ax)
    _,  ampm = frequency_spectra(datam, dt, fft_ax)
    #print(freq.shape, ampm, ampb); quit()

    plt.figure()  #figsize = (11,6))
    try:

        plt.plot(freq, ampm, color='b')
        plt.plot(freq, ampb, color='r')
    except:
        plt.plot(freq, ampm[fft_ax, :], color='b')
        plt.plot(freq, ampb[fft_ax, :], color='r')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend(labels =['Base', 'Monitor'])
    plt.title('Amplitude spectra')
    plt.savefig(os.path.join(inpara.outpath, '%s.png' % filename))

    return freq

#------------------------------------- filter_data
def define_filter(freq, frequency_cut, filter_type, inpara):

    # Generate a lowpass/highcut filter
    # Firstly we'll define the Nyquist frequency

    fs = 1 / .0005 #0.004
    nyq = fs / 2

    # We're going to generate a simple butterworth filter, 
    # The order (in this example we've used 7) determines the slope, larger is a steeper slope
    # The cutoff is the -3dB point and is given as a fraction of the nyquist frequency

    fcut = frequency_cut
    cutoff = [ f / nyq for f in fcut]
    #print(cutoff); quit()
    # lowpass, bandpass
    if filter_type == 'lowpass':
        hc_filt = signal.butter(7, cutoff[0], btype=filter_type, output='sos')
    if filter_type == 'bandpass':
        hc_filt = signal.butter(7, cutoff, btype=filter_type, output='sos')
    # Plot the response of the filter
     # The response is given angular freq (w), normalized to [0, pi]  and amplitude (h)

    w, h = signal.sosfreqz(hc_filt)

    # Convert to frequency
    # w_max / pi = nyq

    freq = nyq * w / np.pi

    # Plot
    plt.figure() #figsize = (8,6))

    plt.plot(freq, np.abs(h), 'b')        #plot the response
    for f in fcut:
        plt.axvline(f, color='k', ls='--') #plot the high cut frequency as a dashed black line
        plt.plot(f, 1/np.sqrt(2), 'ko')    #plot the -3dB (1/sqrt(2)) point as a black circle

    plt.xlim(0, nyq)
    plt.xlabel('Frequency (Hz)')
    plt.title('Filter response')
    plt.savefig(os.path.join(inpara.outpath, 'FilterResponse.png'))

    return hc_filt

#-----------------------------apply frequency filter
def apply_filter(data, freq, frequency_cut, inpara, filter_axis, filter_type):

    hc_filter = define_filter(freq, frequency_cut, filter_type, inpara)
    #print(data.shape);quit()

    filtered_data = signal.sosfiltfilt(hc_filter, data, axis=filter_axis)

    return filtered_data
