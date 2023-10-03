'''
utils.py

Utility functions.
'''
import os, glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as pl
import torch, torchvision, gc
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models import DNNActivations, FactorizedReadOut, DNN_readout_combined, ReadOut
from scipy.ndimage import center_of_mass, shift, median_filter, gaussian_filter,  label
from scipy.ndimage import binary_fill_holes, binary_opening, binary_dilation, binary_closing, binary_erosion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.morphology import skeletonize
from scipy.stats import mannwhitneyu
from scipy import interpolate
from scipy.io import loadmat
from torchvision.transforms import transforms


class SimpleImageNeuronDataset(torch.utils.data.Dataset):  
    def __init__(self,fp_ims, keys=None):
        '''
        Args
            file: Session name to load images and neural data
        '''
        self.fp_ims = fp_ims
        self.keys = keys
        self.images, self.mean, self.std = self.load_data()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((self.mean,self.mean,self.mean),(self.std,self.std,self.std))])

        self.len_dataset = len(self.images)
        self.Y = np.zeros(self.images.shape[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img = self.images[index]  # data
        y = self.Y[index]
        if self.transform:
            img = self.transform(img).type(torch.FloatTensor)
        return (img,y)

    def load_data(self):
        if type(self.fp_ims) == str:
            ims = np.load(self.fp_ims,allow_pickle=True).item()
            if self.keys is None:
                keys = ims.keys()
            else:
                keys = self.keys
            all_images = [ims[key] for key in keys]
            del ims

        elif type(self.fp_ims) == np.ndarray:
            all_images = self.fp_ims.copy()
        
        all_images = np.squeeze(all_images)
        im_mean, im_std = np.mean(all_images), np.std(all_images)
        return all_images, im_mean, im_std
    
def load_DNN(fp_conv, fp_readout, device='cpu', model_class='shallowConv_4', layer=16, pretrained=False, normalize=True, bias=True):
    '''Load in trained DNN + readout models'''
    DNN = DNNActivations(model_class=model_class,layer=layer,pretrained=pretrained, ckpt_path=None).to(device)
    DNN.load_state_dict(torch.load(fp_conv, map_location=torch.device(device)))
    # For FactorisedReadout, load in state dict first to check input and output size
    fact_readout_state_dict = torch.load(fp_readout, map_location=torch.device(device))
    out_size = fact_readout_state_dict['spatial'].shape[0]
    inp_size = [fact_readout_state_dict['features'].shape[1], fact_readout_state_dict['spatial'].shape[2], fact_readout_state_dict['spatial'].shape[3]]
    readout = FactorizedReadOut(inp_size=inp_size, out_size=out_size, bias=bias, normalize=normalize)
    readout.load_state_dict(fact_readout_state_dict)

    DNN = DNN.eval()
    readout = readout.eval()

    return DNN, readout

def load_DNN_untrained(fp_conv, fp_readout, device='cpu', model_class='shallowConv_4', layer=16, pretrained=False, normalize=True, bias=True):
    '''Load in trained DNN + readout models'''
    DNN = DNNActivations(model_class=model_class,layer=layer,pretrained=pretrained, ckpt_path=None).to(device)
    # For FactorisedReadout, load in state dict first to check input and output size
    fact_readout_state_dict = torch.load(fp_readout, map_location=torch.device(device))
    out_size = fact_readout_state_dict['spatial'].shape[0]
    inp_size = [fact_readout_state_dict['features'].shape[1], fact_readout_state_dict['spatial'].shape[2], fact_readout_state_dict['spatial'].shape[3]]
    readout = FactorizedReadOut(inp_size=inp_size, out_size=out_size, bias=bias, normalize=normalize)

    DNN = DNN.eval()
    readout = readout.eval()

    return DNN, readout

def load_dataset(fp_ims, batch_size=256):
    '''Load Imagenet images and return a pytorch dataloader'''
    dataset = SimpleImageNeuronDataset(fp_ims)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataset, dataloader

def get_average_RF(spatial, N=1):
    '''Calculate the average RF from the factorised readout and align to centre. Repeat N times for accuracy'''
    spatial_normalised = np.array([np.abs(rf/(np.sqrt((rf**2).sum())+ 1e-6)) for rf in spatial])
    spatial_shifted = spatial_normalised.copy()
    for n in range(N):
        coms = np.array([center_of_mass(rf) for rf in spatial_shifted])
        centre = (spatial_normalised.shape[1]-1)/2.

        spatial_shifted = np.array([shift(spatial_shifted[i], centre-coms[i], mode='wrap') for i in range(spatial_normalised.shape[0])])
        coms_shifted = np.array([center_of_mass(rf) for rf in spatial_shifted])

    return spatial_shifted, spatial_normalised, coms, coms_shifted

def clean_GPU():
    '''Removes gpu cache'''
    gc.collect()
    torch.cuda.empty_cache()

def load_models(fp_base, fp_nc, regions=['V1', 'LM', 'LI', 'POR', 'AL', 'RL'], device='cpu'):
    '''Load all models and predictions''' 
    fps_conv = [os.path.join(fp_base, region, f'{region}_shallowConv_4_untrained_e2e_16_DNN.pt') for region in regions]
    fps_readout = [os.path.join(fp_base, region, f'{region}_shallowConv_4_untrained_e2e_16_FactorizedReadout.pt') for region in regions]
    fps_pred = [os.path.join(fp_base, region, f'{region}_shallowConv_4_untrained_e2e_16_predictions.npy') for region in regions]

    models = []
    for i in range(len(regions)):
        DNN, readout = load_DNN(fps_conv[i], fps_readout[i], device=device)
        predictions = load_performance(fp_nc, regions[i])
        models.append([DNN, readout, predictions])

    return models

def load_performance(fp, region):
    '''Load in explainable variance explained'''
    fp_nc = os.path.join(fp, f'combined_noise_ceiling_{region}.npy') 
    fp_pred = os.path.join(fp, f'combined_r2_{region}.npy')

    nc = np.load(fp_nc)
    pred = np.load(fp_pred)
    performance = nc/pred

    return performance

def make_signmap(phasemap_a, phasemap_e, offset=0, median_size=15):
    '''Given two phase maps, create a sign map. If necessary, rotate the maps to align
       the gradients to be ~ 45 deg from the camera.
    '''
    # Calculate phase gradient
    grad_a = np.gradient(phasemap_a)
    grad_e = np.gradient(phasemap_e)
    # Calculate direction of gradient
    graddir_a = np.arctan2(grad_a[1], grad_a[0])
    graddir_e = np.arctan2(grad_e[1], grad_e[0])
    vdiff = np.multiply(np.exp(1j * graddir_a), np.exp(-1j * graddir_e))
    # Convert to sign
    sign_map = median_filter(np.sin(np.angle(vdiff)+offset), median_size)

    return sign_map


def circular_smoothing(data, sigma):
    '''Apply Gaussian filter to circular data.
     
       INPUT
       data  : Input data (numpy.array)
       sigma : SD of Gaussian

       OUTPUT
       data_smoothed : Smoothed data (numpy.array)
    '''

    data_sin = np.sin(data)
    data_cos = np.cos(data)

    data_sinf = gaussian_filter(data_sin, sigma=sigma)
    data_cosf = gaussian_filter(data_cos, sigma=sigma)

    data_smoothed = np.arctan2(data_sinf, data_cosf)
    return data_smoothed

def patch_edge(signmap, cutoff=0.2, nopen=1, nclose=1, ndilate=5, borderWidth=1):
    '''Given a sign map, find the edge between regions'''
    # Calculate cutoff
    sign_thresh = ((np.abs(signmap)**1.5) > cutoff)*np.sign(signmap)
    vis_borders = binary_fill_holes(binary_dilation(binary_opening(binary_closing(np.abs(sign_thresh), iterations=nclose), 
                                iterations=nopen), iterations=ndilate)).astype(np.int)

    # Generate borders for each patch
    # Remove noise
    sign_thresh = binary_opening(np.abs(sign_thresh), iterations=nopen).astype(np.int)

    # Identify patches
    patches, patch_i = label(sign_thresh)

    # Close each region
    patch_map = np.zeros_like(patches)
    for i in range(patch_i):
        curr_patch = np.zeros_like(patches)
        curr_patch[patches == i+1] = 1
        patch_map += binary_closing(curr_patch, iterations=nclose).astype(np.int)

    # Expand patches - directly adapted from NeuroAnalysisTools
    total_area = binary_dilation(patch_map, iterations=ndilate).astype(np.int)
    patch_border = total_area - patch_map

    patch_border = skeletonize(patch_border)

    if borderWidth > 1:
        patch_border = binary_dilation(patch_border, iterations=borderWidth-1).astype(np.float)

    patch_border = 1-patch_border
    #patch_border = gaussian_filter(1-patch_border, 1)
    #patch_border[patch_border < 0.5] = 0
    patch_border = np.ma.array(patch_border, mask=patch_border > 0.5)
    # patch_border[patch_border > 0.5] = np.nan
    

    return vis_borders, patch_border

def load_processed_data(fp):
    '''Load in processed neuron traces and save in a dictionary'''
    data = {'unique_stims'      : np.load(glob.glob(os.path.join(fp, '*unique_stims.npy'))[0], allow_pickle=1).item(),
            'unique_stims_raw'  : np.load(glob.glob(os.path.join(fp, '*unique_stims_raw.npy'))[0], allow_pickle=1).item(),
            'rep_stims'         : np.load(glob.glob(os.path.join(fp, '*rep_stims.npy'))[0], allow_pickle=1).item(),
            'rep_stims_raw'     : np.load(glob.glob(os.path.join(fp, '*rep_stims_raw.npy'))[0], allow_pickle=1).item(),
            'noise_ceiling'     : np.load(glob.glob(os.path.join(fp, '*noise_ceiling.npy'))[0], allow_pickle=1),
            'unique_s0'         : np.load(glob.glob(os.path.join(fp, '*unique_s0.npy'))[0], allow_pickle=1),
            }
    return data

def load_processed_region(fp, region):
    '''Load in all the data for a given region and save in dictionary'''
    dirs = glob.glob(os.path.join(fp, f'*{region}*'))
    data = [load_processed_data(f) for f in dirs]
    return data

def combine_data(data, typ='unique_stims', r2_thresh=0.6):
    '''Given a list of data dictionaries and preprocessing type, combine them into one dictionary.
       Data will be filtered by noise_ceiling set by r2_thresh.
       To account for missing data, add np.nan's
    '''
    # First figure out how big the final matrix will be
    ncs = np.concatenate([d['noise_ceiling'] for d in data])

def collapse_dictionary_to_matrix(stims):
    '''Training data for network is saved as a dictionary with keys=stimulus and value=(trial, neuron). 
       Convert this dictionary to a matrix of (stimulus, trial, neuron)
    '''
    mat = np.array([stims[key] for key in stims.keys()])
    return mat

def load_validation(fp, region):
    '''Load validation experiments and extract percentile and normalised responses'''
    fps = glob.glob(os.path.join(fp, region, '*.npy'))
    percentiles = []
    opt_stim = []
    all_stims = []

    for f in fps:
        data = np.load(f, allow_pickle=True).item()
        noise_ceiling = data['nc'] > 0.3
        mask = data['mask'] * noise_ceiling

        umask = data['umask'][mask][:,mask]
        all_stim = data['all_stim'][:,mask]

        scaler = StandardScaler()
        scaler.fit(np.vstack([umask, all_stim]))
        umask = scaler.transform(umask)
        all_stim = scaler.transform(all_stim)

        percentiles.append(np.array([(all_stim[:,i] <= np.diagonal(umask)[i]).sum()/all_stim.shape[0] for i in range(np.diagonal(umask).shape[0])]))
        opt_stim.append(np.diagonal(umask))
        all_stims.append(all_stim.ravel())
    
    percentiles = np.concatenate(percentiles)
    opt_stim = np.concatenate(opt_stim)
    all_stims = np.concatenate(all_stims)

    return percentiles, opt_stim, all_stims

def upsample_spatial_filter(filter_wt,input_size=8,output_size=135):
    '''Generate spatial filter at desired resolution'''
    x = np.linspace(0,output_size-(output_size%input_size),input_size)+(output_size%input_size)//2
    y = np.linspace(0,output_size-(output_size%input_size),input_size)+(output_size%input_size)//2
    f = interpolate.interp2d(x,y,filter_wt,kind='cubic')
    x_new = np.linspace(0,output_size-1,output_size)
    y_new = np.linspace(0,output_size-1,output_size)
    filter_wt_new = f(x_new,y_new)
    return filter_wt_new

def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Removes outliers and scales layout to between [0,1]."""

    # compute percentiles
    mins = np.percentile(layout, min_percentile, axis=(0))
    maxs = np.percentile(layout, max_percentile, axis=(0))

    # add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)

    # `clip` broadcasts, `[None]`s added only for readability
    clipped = np.clip(layout, mins, maxs)

    # embed within [0,1] along both axes
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)

    return clipped

def grayscale_tfo1(channels):
    assert channels==1 or channels==3, "Only 1 or 3 channels can be passed, currently {} passed".format(channels)
    def inner(image_t):
        return image_t.repeat(1,channels,1,1) 	# assuming input shape is (batch,channels,height,width)
    return inner

def plot_atlas(atlas, coords):
    '''Plot atlas given image array and corresponding coordinates'''
    N = np.max(coords)+1
    fig, ax = pl.subplots(figsize=(20,20))
    ax.set_xlim(0, N+1)
    ax.set_ylim(0, N+1)
    for i in range(coords[0].shape[0]):
        x, y = coords[0][i], coords[1][i]
        im = atlas[i]
        extent = 1
        ax.imshow(median_filter(im, 5), extent=[x-extent/2, x+extent/2, y-extent/2, y+extent/2], cmap='gray', vmin=0, vmax=1)
    return fig, ax

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


class OptStims(Dataset):
    '''Dataset to load in activation atlas images'''
    def __init__(self, fp, n_views=2, do_transform=True, transformations=None):
        '''
            Args: 
            fp (list)     : List of paths to atlas images
        '''
        self.fp = fp
        self.n_views = n_views
        self.ims, self.mean, self.std, self.labels = self._load_atlases()
        self.ims = torch.permute(torch.from_numpy(self.ims).float(), (0, 3,1,2))
        
        if transformations is None: transformations = self.get_simclr_pipeline_transform(32)
        self.transform = ContrastiveLearningViewGenerator(transformations, n_views)
        self.len_dataset = self.ims.shape[0]
        self.do_transform = do_transform

    @staticmethod
    def get_simclr_pipeline_transform(size):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        data_transforms = transforms.Compose([
                                              transforms.RandomResizedCrop(size=size),
                                              transforms.RandomAffine((-90,90), translate=(0.1,0.1), fill=0.5),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomInvert(),
                                              transforms.GaussianBlur(11, sigma=(0.1,2)),
                                              ])
        return data_transforms

    def __len__(self):
        return self.ims.shape[0]
    
    def _load_atlases(self):
        '''Load in optimised images and concatenate them. Images are 64x64 and MinMax scaled'''
        ims = []
        labels = []
        for i, fp in enumerate(self.fp):
            data = np.repeat(np.squeeze(np.load(fp))[...,None], 3, axis=-1)
            label = np.zeros(data.shape[0])+i
            ims.append(data)
            labels.append(label)
        
        ims = np.concatenate(ims, axis=0)
        labels = np.concatenate(labels, axis=0)

        print(f"Loaded {len(self.fp)} atlas sets with total shape {ims.shape}.")
        
        im_mean, im_std = ims.mean(), ims.std()
        return ims, im_mean, im_std, labels
    
    def __getitem__(self,index):
        img = self.ims[index]  
        label = self.labels[index]
        if self.do_transform:
            img = self.transform(img)
        return img, label

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]  

def px2deg(px):
    '''Convert pixel on screen to visual degrees'''
    px_per_cm = 1920/52. 
    deg = np.rad2deg(2 * np.arctan2(px/px_per_cm, 2*14))
    return deg

def sbx_get_ttlevents(fn):
    '''Load TTL events from scanbox events file.
       Based on sbx_get_ttlevents.m script.

       INPUT
       fn   : Filepath to events file

       OUTPUT
       evt  : List of TTL trigger time in units of frames (numpy.array)
    '''
    data = loadmat(fn)['ttl_events']
    if not(data.size == 0):
        evt = data[:,2]*256 + data[:,1]     # Not sure what this does
    else:
        evt = np.array([])

    return evt

def plot_stats(stats, regions, figsize=(6,5)):
    '''Plot pairwise statistics'''
    mask = np.tri(stats.shape[0], k=-1).T
    data_masked = np.ma.array(stats, mask=mask)
    fig, ax = pl.subplots(figsize=figsize)
    ax.imshow(data_masked < 0.05, cmap='Reds', vmin=0, vmax=1)
    # ax.imshow(data_masked, cmap='gray_r', vmin=0, vmax=1)

    for i in range(len(regions)):
        for j in range(len(regions)):
            if i == j: continue
            if data_masked.mask[i,j]: continue
            if stats[i,j] > 0.05: color = [0]*3
            else: color = [1] * 3
            #ax.text(j,i, f"{stats[i,j]:.2f}\n±{stats[i,j]:.2f}", ha='center', va='center', size=6, color=color)
            ax.text(j,i, f"{stats[i,j]:.0e}".replace("e-0", "e-"), ha='center', va='center', size=7, color=color)

    ax.set_xticks(range(stats.shape[0]))
    ax.set_xticklabels(regions)
    #ax.xaxis.tick_top()
    ax.set_yticks(range(stats.shape[0]))
    ax.set_yticklabels(regions)
    #ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    return fig