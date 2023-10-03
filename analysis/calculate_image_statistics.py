'''
calculate_image_statistics.py
'''

from utils import *
from glob import glob
from scipy import ndimage
from skimage.measure import regionprops_table
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
    fps = sort_nicely(glob(os.path.join(fp, f"optStim_masked_{region}", '*.png')))
    #fps = glob(os.path.join(fp, region, '*.png'))
    ims = np.array([pl.imread(f)[:,:,0] for f in fps])
    return ims

def get_luminance(im):
    '''Calculate the mean luminance'''
    return im.mean()

def get_rms_contrast(im):
    '''Calculate the mean luminance'''
    return im.std()

def detect_particles(im, white=True):
    '''Given a thresholded image, find all enclosed regions.
       Calculate the multiple statistics as specified in 
       https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops.
    '''
    if white: im_thresh = (im > 0.51).astype(int)
    else: im_thresh = (im < 0.49).astype(int)

    particles, nparticles = ndimage.label(im_thresh)
    regionprop = regionprops_table(particles, properties=['label', 'area', 'area_bbox', 
                                                         'axis_major_length', 'axis_minor_length', 
                                                         'centroid', 'eccentricity', 'orientation',
                                                         'solidity'])
    return regionprop

if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    fp_ims = r'D:\Data\DeepMouse\Results\representative_images'
    fp_ims = r'D:\Data\DeepMouse\Results\optstims\RF_aligned'
    savepath = r'D:\Data\DeepMouse\Results\image_analysis'

    # Load images
    images = [load_images(fp_ims, region) for region in regions]
    
    # 1. Calculate the mean luminance
    luminance = {region: np.array([get_luminance(im) for im in images[i]]) for i, region in enumerate(regions)}
    np.save(os.path.join(savepath, 'luminance.npy'), luminance)

    # 2. Calculate the contrast
    contrast = {region: np.array([get_rms_contrast(im) for im in images[i]]) for i, region in enumerate(regions)}
    np.save(os.path.join(savepath, 'contrast.npy'), contrast)

    # 3. Calculate particle properties for both black and white particles
    particle_white = {region: np.array([detect_particles(im, white=True) for im in images[i]]) for i, region in enumerate(regions)}
    particle_black = {region: np.array([detect_particles(im, white=False) for im in images[i]]) for i, region in enumerate(regions)}
    np.save(os.path.join(savepath, 'particle_white.npy'), particle_white)
    np.save(os.path.join(savepath, 'particle_black.npy'), particle_black)

