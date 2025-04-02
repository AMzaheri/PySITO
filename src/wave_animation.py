
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np
from scipy import ndimage

import os


def make_animation(u_, model, outpath, fname):

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, 'vp', None) is not None:
        Vp = model.vp.data[slices]

    #nt = geometry.nt
    nt = u_.data.shape[0]
    nbl = model.nbl

    fig = plt.figure()
    im = plt.imshow(np.transpose(u_.data[0,nbl:-nbl,nbl:-nbl]),
                cmap="Greys", animated=True, vmin=-1e-1, vmax=1e-1,
                extent=[model.origin[0], model.origin[0] + 1e-3 * model.shape[0] * model.spacing[0],
                        model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]])
    plt.xlabel('X position (km)',  fontsize=20)
    plt.ylabel('Depth (km)',  fontsize=20)
    plt.tick_params(labelsize=20)
    im2 = plt.imshow(np.transpose(Vp), cmap='jet',
                 extent=[model.origin[0], model.origin[0] + 1e-3 * model.shape[0] * model.spacing[0],
                         model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]], alpha=.4)

    def updatefig(i):
        im.set_array(np.transpose(u_.data[i*5,nbl:-nbl,nbl:-nbl]))
        return im, im2

    #anim = animation.FuncAnimation(fig, updatefig, frames=np.linspace(0, nt/5-1, nt//5, dtype=np.int64),blit=True, interval=50)
    #faster
    anim = animation.FuncAnimation(fig, updatefig, frames=np.linspace(0, nt/5-1, nt//50, dtype=np.int64),blit=True, interval=50)



    # gif format
    writergif = animation.PillowWriter() # fps=30) 
    anim.save(os.path.join(outpath, 'wavefield_movie_%s.gif' % fname), writer=writergif)
    # movie mp4
    # the writer is not available on arc
    # writervideo = animation.FFMpegWriter()

#----------------------------------------- plot screen shots
def wave_screenshot(u_, model, geometry, times, outpath, fname):

    fig = plt.figure(figsize=(15, 5))

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))
    if getattr(model, 'vp', None) is not None:
        Vp = model.vp.data[slices]

    nbl = model.nbl

    extent = [model.origin[0], model.origin[0] + 1e-3 * model.shape[0] * model.spacing[0],
              model.origin[1] + 1e-3*model.shape[1] * model.spacing[1], model.origin[1]]

    data_param = dict(vmin=-1e-1, vmax=1e-1, cmap=cm.Greys, aspect=1, extent=extent, interpolation='none')
    model_param = dict(cmap=cm.GnBu, aspect=1, extent=extent, alpha=.3)

    # ax0 = fig.add_subplot(131)
    fig, ax0 = plt.subplots(1)
    _ =plt.imshow(np.transpose(u_.data[times,nbl:-nbl,nbl:-nbl]), **data_param)
    _ = plt.imshow(np.transpose(Vp), **model_param)
    ax0.set_ylabel('Depth (km)',  fontsize=15)
    ax0.set_xlabel('X position (km)', fontsize=15)
    #ax1 = fig.add_subplot(132)
    #_ = plt.imshow(np.transpose(u_.data[times[1],nbl:-nbl,nbl:-nbl]), **data_param)
    #_ = plt.imshow(ndimage.rotate(Vp, -90), **model_param)
    #ax1.set_xlabel('X position (km)',  fontsize=14)
    #ax1.set_yticklabels([])

    #ax2 = fig.add_subplot(133)
    #_ = plt.imshow(np.transpose(u_.data[times[2],nbl:-nbl,nbl:-nbl]), **data_param)
    #_ = plt.imshow(ndimage.rotate(Vp ,-90), **model_param)
    #ax2.set_yticklabels([])

    plt.savefig(os.path.join(outpath, "wave_screenshot_%s_%s.png" % (fname, times)))

