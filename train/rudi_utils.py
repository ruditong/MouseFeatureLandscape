import os, glob
import numpy as np
import matplotlib.pyplot as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import *
from scipy import interpolate
from scipy.ndimage import center_of_mass, shift
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap
from scipy.stats import binned_statistic_2d
from lucent.optvis.objectives import wrap_objective
from lucent.optvis import render, param, transform, objectives
from scipy.ndimage import gaussian_filter, median_filter
import scipy.fft as fft
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

class ImageNeuronDataset(torch.utils.data.Dataset):  
    def __init__(self,fp_ims):
        '''
        Args
            file: Session name to load images and neural data
        '''
        self.fp_ims = fp_ims
        self.images, self.mean, self.std = self.load_data()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((self.mean,self.mean,self.mean),(self.std,self.std,self.std))])

        self.len_dataset = len(self.images)
        self.Y = np.zeros(self.images.shape[0])

    def __len__(self):
        return self.len_dataset

    def __getitem__(self,index):
        img = self.images[index]  # data
        y = self.Y[index]
        if self.transform:
            img = self.transform(img).type(torch.FloatTensor)
        return (img,y)

    def load_data(self):
        ims = np.load(self.fp_ims,allow_pickle=True).item()
        all_images = [ims[key] for key in ims.keys()]
        del ims
        
        all_images = np.array(all_images)
        im_mean, im_std = np.mean(all_images), np.std(all_images)
        return all_images, im_mean, im_std


class ActivationAtlas(Dataset):
    '''Dataset to load in activation atlas images'''
    def __init__(self, fp):
        '''
            Args: 
            fp (list)     : List of paths to atlas images
        '''
        self.fp = fp
        self.ims, self.mean, self.std, self.labels = self._load_atlases()
        self.ims = torch.permute(torch.from_numpy(self.ims).float(), (0, 3,1,2))
        
        self.len_dataset = self.ims.shape[0]

    def __len__(self):
        return self.len_dataset
    
    def _load_atlases(self):
        '''Load in atlas images and concatenate them.'''
        ims = []
        labels = []
        for i, fp in enumerate(self.fp):
            data = np.repeat(np.squeeze(np.load(fp))[...,None], 3, axis=-1)
            label = np.zeros(data.shape[0])+i
            ims.append(data)
            labels.append(label)
        
        ims = np.concatenate(ims, axis=0)
        ims = (ims-ims.min())/(ims.max()-ims.min())
        labels = np.concatenate(labels, axis=0)

        print(f"Loaded {len(self.fp)} atlas sets with total shape {ims.shape}.")
        
        im_mean, im_std = ims.mean(), ims.std()
        return ims, im_mean, im_std, labels
    
    def __getitem__(self,index):
        img = self.ims[index]  
        label = self.labels[index]

        return img, label


class DNN_combined(nn.Module):
    def __init__(self,DNN1, DNN2):
        super().__init__()
        self.DNN1 = DNN1
        self.DNN2 = DNN2

    def forward(self,X):
        out1 = self.DNN1(X)
        out2 = self.DNN2(X)
        return out1+out2
    
def whiten_matrix(X):
    '''Calculates the precision matrix for whitening data'''
    cc = np.matmul(X.T, X) / len(X)
    cc = cc.astype("float32")
    S = np.linalg.inv(cc).astype("float32")
    return S

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

def svd_whiten(X):
    '''Whiten a matrix using SVD'''

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white

def upsample_spatial_filter(filter_wt,input_size=16,output_size=135):
    '''Generate spatial filter at desired resolution'''
    x = np.linspace(0,output_size-(output_size%input_size),input_size)+(output_size%input_size)//2
    y = np.linspace(0,output_size-(output_size%input_size),input_size)+(output_size%input_size)//2
    f = interpolate.interp2d(x,y,filter_wt,kind='cubic')
    x_new = np.linspace(0,output_size-1,output_size)
    y_new = np.linspace(0,output_size-1,output_size)
    filter_wt_new = f(x_new,y_new)
    return filter_wt_new

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
    
def calcualate_grid(embedding, activations, N=10):
    '''Group UMAP embedding into grid of size N and calculate the average activations'''
    bins = np.linspace(0,1,N+1)
    count, _, _, binnumber = binned_statistic_2d(x=embedding[:,0], y=embedding[:,1], values=None, statistic='count', bins=bins, expand_binnumbers=True)
    grid_activation = np.zeros((N, N, activations.shape[1]))
    for i in range(N):
        for j in range(N):
            acts = activations[(binnumber[0]==i)*(binnumber[1]==j)]
            grid_activation[i,j] = acts.mean(axis=0)
    return grid_activation, count

def get_orientation(im, resolution=16):
    '''Calculate the orientation tuning of an image by taking the radial average of the FFT spectrum'''
    f = np.fft.fftshift(np.fft.fft2(im-im.mean()))
    power = np.abs(f)

    x,y = np.meshgrid(np.arange(power.shape[1]), np.arange(power.shape[0]))
    r = np.linspace(-np.pi,np.pi, resolution)
    span = np.abs(r[0]-r[1])/2
    R = np.arctan2(y-power.shape[1]/2,x-power.shape[1]/2)
    rad_av = np.array([ np.mean(power[(R >= r[i]-span) & (R < r[i]+span)]) for i in range(r.shape[0])])
    rad_av = (rad_av-rad_av.min())/(rad_av.max()-rad_av.min())
    OS = (np.exp(2j*r)*rad_av).sum()
    OSI = np.absolute(OS)/rad_av.sum()
    pref_ori = np.angle(OS)
    return OSI, pref_ori/2

def process_optims(activation_atlas, median_size=8, rf_thresh=0.1):
    '''Apply RF and median filter to optimal images'''

    rf = (upsample_spatial_filter(np.median(spatial_shifted, axis=0), 8) > rf_thresh).astype(float)
    rf = gaussian_filter(rf, sigma=5)
    optim_median = np.array([median_filter(activation_atlas[i], median_size) for i in range(activation_atlas.shape[0])])
    optim_processed = np.array([optim_median[i]*rf + np.ones(activation_atlas[i].shape)*0.5*(1-rf) for i in range(activation_atlas.shape[0])])
    return optim_processed

def plot_activation_atlas(activation_atlas, mask, count, rf_thresh=0.1, median_size=8):
    optim_processed = process_optims(activation_atlas, median_size=median_size, rf_thresh=rf_thresh)

    N = mask.shape[0]
    fig, ax = pl.subplots(figsize=(20,20))
    idx = np.where(mask)
    ax.set_xlim(0, N+1)
    ax.set_ylim(0, N+1)
    for i in range(idx[0].shape[0]):
        x, y = idx[0][i], idx[1][i]   
        im = activation_atlas[i]*m + np.ones(activation_atlas[i].shape)*0.5*(1-m)
        extent = 1
        extent = np.min([(count[i]**(1/4))[0]*1.5, 1])
        ax.imshow(median_filter(im[35:135-35, 35:135-35], 5), extent=[x-extent/2, x+extent/2, y-extent/2, y+extent/2], cmap='gray', vmin=0, vmax=1)

    ax.axis('off')

@wrap_objective()
def alignment_loss(layer, vec, cossim_pow=1, S=None, batch=0):
    def inner(model):
        activity = model(layer) # (batch x neuron)
        activity = activity[batch]
        vec_ = vec
        if S is not None: vec_ = torch.matmul(vec_[None], S)[0]
        mag = torch.sqrt(torch.reduce_sum(activity))
        dot = torch.reduce_sum(activity * vec_)
        cossim = dot/(1e-4 + mag)
        cossim = torch.max(0.1, cossim)

        return - dot * cossim ** cossim_pow
    return inner

@wrap_objective()
def alignment_batch(layer, vec, cossim_pow, mask=None, device='cpu'):
    vec = vec.to(device).float()
    mask = mask.to(device)
    #if mask is not None: mask = mask.to(device)
    #else: mask = torch.ones(vec.shape[1], dtype=torch.bool)
    def inner(model):
        activity = model(layer) # (batch x neuron)
        dot = torch.diag(activity[:,mask] @ vec.T)  # (batch x batch)
        xnorm = torch.linalg.norm(activity, dim=1)
        ynorm = torch.linalg.norm(vec, dim=1)
        loss = (dot**cossim_pow / (xnorm*ynorm)**(cossim_pow-1)).mean()

        return -loss
    return inner

def alignment(y_pred, vec, cossim_pow, device='cpu'):
    vec =vec.to(device)
    dot = torch.diag(y_pred @ vec.T)  # (batch x batch)
    xnorm = torch.linalg.norm(y_pred, dim=1)
    ynorm = torch.linalg.norm(vec, dim=1)
    loss = (dot**cossim_pow / (xnorm*ynorm)**(cossim_pow-1)).mean()

    return -loss

@wrap_objective()
def tuning_curve_loss(layer, PC, tuning, mu, std, N=0, lamda=1, device='cpu'):
    '''
    PC      : Principal component axes (nPC x neuron)
    tuning  : Tuning vector (e.g. torch.arange(batch)) (batch)
    N       : Nth PC to visualise
    lamda   : Hyperparameter
    '''
    PC = PC.to(device)
    tuning = tuning.to(device).float()
    tuning = tuning/torch.linalg.norm(tuning)
    mu = mu.to(device).float()
    std = std.to(device).float()
    def inner(model):
        activity = model(layer) # (batch x neuron)
        activity_norm = (activity-mu)/std
        # Project into PC space
        pc_projection = activity_norm @ PC.T # (batch x neuron) @ (neuron x nPC) -> (batch x nPC)
        # Calculate alignment with nth PC
        align = torch.dot(tuning, pc_projection[:,N]).pow(4)/torch.linalg.norm(pc_projection[:,N]).pow(3)
        # Calculate activation of all other PCs
        act_res = pc_projection.mean() - pc_projection[:,N].mean()
        # Put everything together for loss
        loss = align + lamda*act_res

        return -loss
    return inner

def optStimLucent_atlas(DNN, readout, vec, inp_img_size=135, device='cpu', lr=3e-3, cossim_pow=4, layer='readout_fc', mask=None):
    readout_simple = ReadOut(readout_model=readout)  
    combined_model = DNN_readout_combined(DNN, readout_simple).to(device).eval()
    param_f = lambda:param.image(inp_img_size, batch=vec.shape[0], fft=True, decorrelate=True, channels=1)

    optimizer_vis = lambda params: torch.optim.Adam(params, lr=lr)		

    tot_objective = alignment_batch(layer, vec, cossim_pow, mask=mask, device=device)

    transforms = [transform.pad(2, mode='constant', constant_value=1),
                  transform.jitter(4),
                  transform.random_rotate(list(range(-10,10))),
                  transform.jitter(4),
                  grayscale_tfo1(3)]	
    imgs = render.render_vis(combined_model,tot_objective,optimizer=optimizer_vis,param_f=param_f,preprocess=False,fixed_image_size=inp_img_size,show_image=False,transforms=transforms) 	# transforms

    return imgs[0].transpose([0,3,1,2]) 	# return as (len(neurons),3,135,135)


def optStimLucent_PCA(DNN, readout, PC, tuning, N, pc_mean, pc_std, inp_img_size=135, device='cpu', lr=3e-3, layer='readout_fc', lamda=1):
    readout_simple = ReadOut(readout_model=readout)  
    combined_model = DNN_readout_combined(DNN, readout_simple).to(device).eval()
    param_f = lambda: param.image(inp_img_size, batch=tuning.shape[0], fft=True, decorrelate=True, channels=1)

    optimizer_vis = lambda params: torch.optim.Adam(params, lr=lr)		

    tot_objective = tuning_curve_loss(layer, PC, tuning, N=N, mu=pc_mean, std=pc_std, device=device, lamda=lamda)

    transforms = [transform.pad(2, mode='constant', constant_value=1),
                  #transform.jitter(4),
                  #transform.jitter(4),
                  #transform.jitter(8),
                  #transform.jitter(8),
                  transform.jitter(4),
                  transform.random_rotate(list(range(-10,10))),
                  transform.jitter(4),
                  grayscale_tfo1(3)]	
    imgs = render.render_vis(combined_model,tot_objective,optimizer=optimizer_vis,param_f=param_f,preprocess=False,fixed_image_size=inp_img_size,show_image=False,transforms=transforms) 	# transforms

    return imgs[0].transpose([0,3,1,2]) 	# return as (len(neurons),3,135,135)

def grayscale_tfo(channels):
    assert channels==1 or channels==3, "Only 1 or 3 channels can be passed, currently {} passed".format(channels)
    def inner(image_t):
        return torchvision.transforms.functional.rgb_to_grayscale(image_t,num_output_channels=channels)
    return inner

def grayscale_tfo1(channels):
    assert channels==1 or channels==3, "Only 1 or 3 channels can be passed, currently {} passed".format(channels)
    def inner(image_t):
        return image_t.repeat(1,channels,1,1) 	# assuming input shape is (batch,channels,height,width)
    return inner

def run_activation_atlas(DNN, readout, vec, batch=20, inp_img_size=135, device='cpu', lr=3e-3, cossim_pow=4, mask=None):
    ims = []
    for i in range(vec.shape[0]//batch + 1):
        vec_ = vec[i*batch:(i+1)*batch]
        im_batch = optStimLucent_atlas(DNN, readout, vec_, inp_img_size, device, lr, cossim_pow, mask=mask)
        ims.append(im_batch)
    return np.concatenate(ims, axis=0)


def run_PCA_tuning(DNN, readout, PC, tuning, pc_mean, pc_std, inp_img_size=135, device='cpu', lr=3e-3, lamda=1):
    ims = []
    # Loop over each PC and generate tuning curves
    for n in range(PC.shape[0]):
        im_n = optStimLucent_PCA(DNN, readout, PC, tuning, n, pc_mean=pc_mean, pc_std=pc_std, inp_img_size=inp_img_size, device=device, lr=3e-3, lamda=lamda)
        ims.append(im_n)
    return np.array(ims)

def load_dataset(fp_ims, batch_size=256):
    '''Load Imagenet images and return a pytorch dataloader'''
    dataset = ImageNeuronDataset(fp_ims)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataset, dataloader

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