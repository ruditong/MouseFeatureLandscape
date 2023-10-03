'''
calculate_FFT_statistics.py

For optimised images, perform FFT and calculate radial and axial averages.
'''

from utils import *
from glob import glob
from scipy import ndimage, signal
import re

def sort_nicely( l ): 
  """ Sort the given list in the way that humans expect. 
  """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  lsorted = sorted(l, key=alphanum_key )
  return lsorted

def load_images(fp, region):
    '''Load optimised images'''
    #fps = sort_nicely(glob(os.path.join(fp, f"optStim_masked_{region}", '*.png')))
    fps = glob(os.path.join(fp, region, '*.png'))
    ims = np.array([pl.imread(f)[:,:,0] for f in fps])
    return ims

def load_images2(fp, region):
    '''Load optimised images'''
    #fps = glob(os.path.join(fp, f"optStim_masked_{region}", '*.png'))
    fps = glob(os.path.join(fp, f"optStim_masked_{region}", '*.png'))
    ims = np.array([pl.imread(f)[:,:,0] for f in fps])
    return ims

def get_FFT(im):
    '''Perform FFT and return the centred power spectrum.'''
    # Remove DC
    im = im-im.mean()

    fft = np.fft.fftshift(np.fft.fft2(im))
    ps = np.abs(fft)**2
    ps = ps/ps.sum()
    return ps

def autocorrelation(x):
    x = x-x.mean()
    autocorr = signal.correlate2d(x,x, 'same')
    return autocorr

def get_average(ps, nbins_r, nbins_t):
    '''For a given power spectrum, calculate the radial and axial averages'''
    sx, sy = ps.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    
    # Compute radius and angle for each coordinate
    r = np.hypot(X-sx//2, Y-sy//2)
    theta = (np.arctan2(Y-sy//2, X-sx//2) + 2*np.pi)%(2*np.pi)

    # Calculate bins
    rmax = sx/2 -1
    rbin = (nbins_r * r/rmax).astype(np.int)
    tbin = (nbins_t * theta/(2*np.pi)).astype(np.int)
    tbin[r < 1] = -1
    tbin[r > 30] = -1

    # Calculate mean
    radial_mean = ndimage.mean(ps, labels=rbin, index=np.arange(nbins_r))
    axial_mean = ndimage.mean(ps, labels=tbin, index=np.arange(nbins_t))

    return radial_mean, axial_mean

if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    fp_ims = r'D:\Data\DeepMouse\Results\representative_images'
    #fp_ims = r'D:\Data\DeepMouse\Results\optstims\RF_aligned'
    savepath = r'D:\Data\DeepMouse\Results\image_analysis'

    nbins_r, nbins_t = 30, 24

    # Load images
    images = [load_images(fp_ims, region) for region in regions]
    
    # Perform fft
    ffts = [np.array([get_FFT(im) for im in ims]) for ims in images]
    # # Get autocorrelation
    # autocorr = [np.array([autocorrelation(im) for im in ims]) for ims in images]

    # # Calculate measures on autocorr
    # averages_raw = [np.array([get_average(ps, nbins_r=nbins_r, nbins_t=nbins_t) for ps in corr]) for corr in autocorr]
    # # Save
    # for i, region in enumerate(regions):
    #     np.save(os.path.join(savepath, f"FFT_statistics_autocorr_{region}_rep.npy"), averages_raw[i])

    # Perform averaging
    averages_raw = [np.array([get_average(ps, nbins_r=nbins_r, nbins_t=nbins_t) for ps in fft]) for fft in ffts]
    # Save
    for i, region in enumerate(regions):
        np.save(os.path.join(savepath, f"FFT_statistics_raw_{region}_rep.npy"), averages_raw[i])

    # Calculate the mean over all images
    fft_mean = np.mean([fft.mean(axis=0) for fft in ffts], axis=0)
    fft_std = np.std(np.concatenate([fft for fft in ffts], axis=0), axis=0)

    for i in range(len(ffts)):
        ffts[i] = (ffts[i]-fft_mean)/fft_std

    # Perform averaging
    averages = [np.array([get_average(ps, nbins_r=nbins_r, nbins_t=nbins_t) for ps in fft]) for fft in ffts]

    # Save
    for i, region in enumerate(regions):
        np.save(os.path.join(savepath, f"FFT_statistics_{region}_rep.npy"), averages[i])
