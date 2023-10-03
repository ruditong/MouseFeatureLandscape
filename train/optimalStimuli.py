'''
Copied from optimalStimuli_all_animals.py, but modified so that it can be readily adapted to pooled neurons.
Major TODO: This script is adapted to --use_lucent, where optimal stimuli will have 1 channel. If we use DeepDream, OS will have 3 channels and will need to change certain parts of the script.
'''

import torch
import gc
from torch.utils import data
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
import time, os, glob
import argparse
import cv2
from dataloader import ImageNeuronDataset_v2
from shallow_simclr_backbone import BackBone
from models import DNNActivations, FactorizedReadOut, ReadOut, DNN_readout_combined
from train_utils import *
from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo.util import get_model_layers
from functools import reduce
from scipy import interpolate
from os_analysis import binarize_dilate_ConnectedComp_gaussian_RF, apply_RF

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

def set_axes_border_prop(ax,color,width):
    for side in ax.spines.keys():
        ax.spines[side].set_color(color)
        ax.spines[side].set_linewidth(width)
    return ax

def total_variation_loss(img):
    batches, channels, height, width = img.size()
    tot_var_horiz = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1],2).sum()
    tot_var_vert = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:],2).sum()
    return (tot_var_horiz+tot_var_vert)/(batches*channels*height*width)

def deepDream(img,DNN,readout,neurons,iterations,lr,total_variation_penalty=0.0):
    img = img.clone().to(device)
    inp = torch.autograd.Variable(img, requires_grad=True)
    neuron_idx_th = torch.tensor(neurons).unsqueeze(1).to(device)
    loss_arr = []
    optimizer = torch.optim.Adam([inp], lr=lr)
    DNN.eval()
    readout.eval()
    for _ in tqdm(range(iterations)):
        # DNN.zero_grad()
        # readout.zero_grad()
        # if inp.grad is not None:
        # 	inp.grad.zero_()

        optimizer.zero_grad()
        out = DNN(inp)
        out = readout(out)
        # loss = torch.gather(out,1,neuron_idx_th).sum() - total_variation_penalty*total_variation_loss(inp)
        loss = -torch.gather(out,1,neuron_idx_th).sum() + total_variation_penalty*total_variation_loss(inp)
        loss_arr.append(loss.item())
        loss.backward()
        # inp.data = inp.data + lr*inp.grad.data
        # inp.data = (inp.data-inp.data.min())/(inp.data.max()-inp.data.min()+0.0001)
        optimizer.step()
    inp = inp.detach().cpu().numpy()
    inp = (inp-inp.min())/(inp.max()-inp.min()+0.0001)
    # inp = np.clip(inp, 0, 1)
    return inp, loss_arr

def optStimLucent(DNN,readout,neurons,inp_img_size=135, device='cpu'):
    # breakpoint()
    readout_simple = ReadOut(readout_model=readout)  # TODO: Are ReadOut and DNN_readout_combined needed?
    combined_model = DNN_readout_combined(DNN,readout_simple).to(device).eval()
    # param_f = lambda:param.image(inp_img_size,batch=len(neurons),fft=True,decorrelate=False)
    param_f = lambda:param.image(inp_img_size,batch=len(neurons),fft=True,decorrelate=False,channels=1)
    # TRIED with channels = 3 --> noisier than without setting channels=3 --> very similar to when decorrelate=False
    # removed pad from transform --> scale of features reduces
    # tot_objective = None
    # for idx,neuron_idx in enumerate(neurons):
    # 	if tot_objective is not None:
    # 		tot_objective += objectives.channel("readout_fc",neuron_idx,batch=idx)
    # 	else:
    # 		tot_objective = objectives.channel("readout_fc",neuron_idx,batch=idx)
    # optimizer_vis = lambda params: torch.optim.Adam(params, 2e-3)		# for V1
    optimizer_vis = lambda params: torch.optim.Adam(params, 3e-3)		# for LM
    # optimizer_vis = lambda params: torch.optim.Adam(params, 5e-2)		# fastLR --> default param

    # tot_objective = reduce(lambda x,y: x+objectives.channel("readout_fc",y[0],batch=y[1]),list(zip(neurons,np.arange(len(neurons)))),0)
    tot_objective = sum([objectives.channel("readout_fc",neuron_idx,batch=idx) for idx,neuron_idx in enumerate(neurons)])
    # transforms = [transform.pad(4),transform.jitter(8),transform.random_rotate(list(range(-10,10))),transform.jitter(4),grayscale_tfo(3)]	#  + 2*list(range(-5,5)) + 5*list(range(-2,2))
    transforms = [transform.pad(4),transform.jitter(8),transform.random_rotate(list(range(-10,10))),transform.jitter(4),grayscale_tfo1(3)]	#  + 2*list(range(-5,5)) + 5*list(range(-2,2))
    imgs = render.render_vis(combined_model,tot_objective,optimizer=optimizer_vis,param_f=param_f,preprocess=False,fixed_image_size=inp_img_size,show_image=False,transforms=transforms) 	# transforms
    # debug_plot(imgs[0],'optStim_optim_slower_lessScaling_rotateMore')
    # breakpoint()
    return imgs[0].transpose([0,3,1,2]) 	# return as (len(neurons),3,135,135)

def optStimLucent_slow(DNN,readout,neurons,inp_img_size=135):
    # breakpoint()
    readout_simple = ReadOut(readout_model=readout)
    combined_model = DNN_readout_combined(DNN,readout_simple).to(device).eval()
    # param_f = lambda:param.image(inp_img_size,batch=len(neurons),fft=True,decorrelate=False)
    imgs = []
    for idx,neuron_idx in tqdm(enumerate(neurons)):
        param_f = lambda:param.image(inp_img_size,batch=1,fft=False,decorrelate=False,channels=1)
        optimizer_vis = lambda params: torch.optim.Adam(params, 3e-3)		# for LM
        tot_objective = objectives.channel("readout_fc",neuron_idx)
        transforms = [transform.pad(4),transform.jitter(8),transform.random_rotate(list(range(-10,10))),transform.jitter(4),grayscale_tfo1(3)]
        img = render.render_vis(combined_model,tot_objective,optimizer=optimizer_vis,param_f=param_f,preprocess=False,fixed_image_size=inp_img_size,show_image=False,transforms=transforms)
        imgs.append(img)
    # TRIED with channels = 3 --> noisier than without setting channels=3 --> very similar to when decorrelate=False
    # removed pad from transform --> scale of features reduces
    # tot_objective = None
    # for idx,neuron_idx in enumerate(neurons):
    # 	if tot_objective is not None:
    # 		tot_objective += objectives.channel("readout_fc",neuron_idx,batch=idx)
    # 	else:
    # 		tot_objective = objectives.channel("readout_fc",neuron_idx,batch=idx)
    # optimizer_vis = lambda params: torch.optim.Adam(params, 2e-3)		# for V1
    # optimizer_vis = lambda params: torch.optim.Adam(params, 3e-3)		# for LM
    # optimizer_vis = lambda params: torch.optim.Adam(params, 5e-2)		# fastLR --> default param

    # tot_objective = reduce(lambda x,y: x+objectives.channel("readout_fc",y[0],batch=y[1]),list(zip(neurons,np.arange(len(neurons)))),0)
    # tot_objective = sum([objectives.channel("readout_fc",neuron_idx,batch=idx) for idx,neuron_idx in enumerate(neurons)])
    # transforms = [transform.pad(4),transform.jitter(8),transform.random_rotate(list(range(-10,10))),transform.jitter(4),grayscale_tfo(3)]	#  + 2*list(range(-5,5)) + 5*list(range(-2,2))
    # transforms = [transform.pad(4),transform.jitter(8),transform.random_rotate(list(range(-10,10))),transform.jitter(4),grayscale_tfo1(3)]	#  + 2*list(range(-5,5)) + 5*list(range(-2,2))
    # imgs = render.render_vis(combined_model,tot_objective,optimizer=optimizer_vis,param_f=param_f,preprocess=False,fixed_image_size=inp_img_size,show_image=False,transforms=transforms) 	# transforms
    # debug_plot(imgs[0],'optStim_optim_slower_lessScaling_rotateMore')
    imgs = np.hstack(imgs)
    # debug_plot(imgs[0],"optStim_gray_new_test_slower")
    # breakpoint()
    return imgs[0].transpose([0,3,1,2]) 	# return as (len(neurons),3,135,135)

def debug_plot(imgs,figname):
    if len(imgs.shape)>3 and imgs.shape[-1]!=3 and imgs.shape[-1]!=1:
        print('Transposing input')
        imgs = np.transpose(imgs,[0,2,3,1])
    cols = 6
    rows = len(imgs)//cols + 1
    plt.figure(figname,figsize=(10,10))
    for idx,img in enumerate(imgs):
        plt.subplot(rows,cols,idx+1)
        if img.shape[-1]==3:
            plt.imshow(img)
        else:
            plt.imshow(img.squeeze(),cmap='gray')
    plt.savefig(figname+".png")
    plt.close(figname)

def rgb2gray(img):
    return np.dot(img[...,:3],[0.2989, 0.5870, 0.1140])

def upsample_spatial_filter(filter_wt,input_size=16,output_size=135):
    # breakpoint()
    x = np.linspace(0,output_size-(output_size%input_size),input_size)+(output_size%input_size)//2
    y = np.linspace(0,output_size-(output_size%input_size),input_size)+(output_size%input_size)//2
    f = interpolate.interp2d(x,y,filter_wt,kind='cubic')
    x_new = np.linspace(0,output_size-1,output_size)
    y_new = np.linspace(0,output_size-1,output_size)
    filter_wt_new = f(x_new,y_new)
    return filter_wt_new

parser = argparse.ArgumentParser(description="Deepdream to generate optimal stimuli")
# parser.add_argument("--data_folder", type=str,
#                     default='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
#                     help='Path to data files')
parser.add_argument("--image_data_folder",type=str,default='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
                    help='Path to directory in which DNN_images.npy is stores')
parser.add_argument("--response_data_folder",type=str,default='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
                    help='Path to rep_stims.npy and unique_stims.npy')
parser.add_argument("--area",type=str,default="V1",help="Visual cortex area to be used")
parser.add_argument("--model",type=str,default="VGG",help="Pretrained model architecture")
parser.add_argument("--layer",type=int,default=17,help="depth of DNN backbone to use for feature extraction")
parser.add_argument(
    "--save_dir",
    type=str,
    default="experiments/",
    help="path to save the experimental config, logs, model. This is not automatically generated.",
)
parser.add_argument("--use_lucent", default=False,action='store_true',help="Choose to use lucent to generate optStim (yields high quality images)")
parser.add_argument("--seed", type=int, default=12, help="random seed")
parser.add_argument("--bias", type=str, default="True", help="use bias for readout")
parser.add_argument("--normalize", type=str, default="False", help="normalize spatial weights")
parser.add_argument("--pretrained", type=str, default="True", help="use pretrained model weights or random weights")
parser.add_argument("--end_to_end", type=str, default="True", help="train backbone DNN as well")
parser.add_argument("--batch_size", type=int, default=20, help="batch size of neurons to generate stimuli")
parser.add_argument("--save_mode", type=str, default="all", help="to save OS for *good* neurons only or for *all* neurons. Either 'good' or 'all'.")
parser.add_argument("--ckpt_path", type=str, default="None", help="path to .pth file for shallow SimCLR checkpoint")
parser.add_argument("--normalize_by_noise_ceiling", type=str, default="True", help="Normalize pearson r of predictions by noise ceiling")
parser.add_argument("--load_existing_os_dict", default=False, help="if True, load existing grayscale_ColorNorm_good/all.npy file in save_dir")
args = parser.parse_args()
argsdict = args.__dict__
print(argsdict)

# Setting up cuda and seeds
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(argsdict["seed"])
np.random.seed(argsdict["seed"])

# Setting up rest of the arguments
# data_folder = argsdict["data_folder"]
image_data_folder = argsdict["image_data_folder"]
response_data_folder = argsdict["response_data_folder"]
area = argsdict["area"]
if argsdict["model"] =='VGG':
    model_class = models.vgg16
else:
    model_class = argsdict["model"]  # eg. shallowConvfeat_4
save_dir = argsdict["save_dir"]
saved_model_folder = os.path.join(save_dir,'')
use_lucent = argsdict['use_lucent']
layer = argsdict["layer"]
save_mode = argsdict["save_mode"]
pretrained = True if argsdict["pretrained"] == "True" or argsdict["pretrained"] == True else False
e2e = True if argsdict["end_to_end"] == "True" or argsdict["end_to_end"] == True else False
bias = True if argsdict["bias"] == "True" or argsdict["bias"] == True else False
normalize = True if argsdict["normalize"] == "True" or argsdict["normalize"] == True else False
r_reduction = '75per'
Files = ['part1','part2','part3','part4']
batch_size = argsdict['batch_size']
ckpt_path = argsdict["ckpt_path"]
normalize_by_noise_ceiling = True if argsdict["normalize_by_noise_ceiling"] == "True" or argsdict["normalize_by_noise_ceiling"] == True else False
load_existing_os_dict = True if argsdict["load_existing_os_dict"]=="True" or argsdict["load_existing_os_dict"]==True else False
total_variation_penalty = 0.08

model_fname = saved_model_folder+'{}_{}_{}{}_{}_DNN.pt'.format(area,argsdict["model"],
                                                                  'pretrained' if pretrained else 'untrained','_e2e' if e2e else '',layer)
readout_fname = saved_model_folder+'{}_{}_{}{}_{}_FactorizedReadout.pt'.format(area,argsdict["model"],
                                                                                  'pretrained' if pretrained else 'untrained','_e2e' if e2e else '',layer)

# good_neurons_idx, noise_ceiling, train_loader, val_loader, DNN, readout = load_define_data_model_factorized_v3(data_folder=data_folder,area=area,
#                                                                                                                files=Files,batch_size=batch_size, model_class=model_class, layer=layer, pretrained=pretrained,
#                                                                                                                bias=bias, normalize=normalize, ckpt_path=ckpt_path)
good_neurons_idx, noise_ceiling, train_loader, val_loader, DNN, readout = load_define_data_model_factorized_v4(
    response_data_folder=response_data_folder, image_data_folder=image_data_folder,batch_size=batch_size, model_class=model_class, layer=layer, pretrained=pretrained,
    bias=bias, normalize=normalize, ckpt_path=ckpt_path)
DNN.load_state_dict(torch.load(model_fname))
readout.load_state_dict(torch.load(readout_fname))
spatial_weights = readout.spatial.detach().cpu().numpy().squeeze()

all_neurons_r, r, r_75 = get_model_prediction_r(DNN=DNN,ReadOutModel=readout,
                                                                              val_loader=val_loader,noise_ceiling=noise_ceiling,normalize_by_noise_ceiling=normalize_by_noise_ceiling, verbose=True)

np.save(os.path.join(saved_model_folder,"{}_{}_{}{}_{}_predictions.npy".format(area,argsdict["model"],
                                                                                  ('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer)),all_neurons_r/noise_ceiling)

all_neurons_r_noise_ceil = all_neurons_r/noise_ceiling
ordering = np.argsort(-all_neurons_r_noise_ceil)
num_neurons_plot = np.sum(all_neurons_r_noise_ceil>=0)

if load_existing_os_dict:
    print(f"Loading existing os dict, which contains {num_neurons_plot} neurons")
    if save_mode == "good":
        existing_os_dict = os.path.join(saved_model_folder,"{}_{}_{}{}_{}_optStim{}_grayscale_ColorNorm_good.npy".format(area,argsdict["model"],
                                                                                                     ('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer,'_lucent' if use_lucent else ''))
    elif save_mode == "all":
        existing_os_dict = os.path.join(saved_model_folder,"{}_{}_{}{}_{}_optStim{}_grayscale_ColorNorm_all.npy".format(area,argsdict["model"],
                                                                                                     ('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer,'_lucent' if use_lucent else ''))
    optStim = np.load(existing_os_dict, allow_pickle=True).item()
else:
    plot_cols = 10
    print("Plotting {} neurons".format(num_neurons_plot))
    optimal_stim_idx_collection = []  # saves neurons with pred r > 0.7
    optimal_stim_collection = []
    optimal_stim_RF_collection = []
    optimal_stim_idx_collection_all = []  # saves neurons with pred r > 0
    optimal_stim_collection_all = []
    optimal_stim_RF_collection_all = []

    # save neuron idx, prediction, and the index used to label optimal stimuli into one npy
    saved_os_stats = {}
    saved_os_stats['saved_idx'] = ordering[:num_neurons_plot]  # matches optStim['neuron_idxs']
    saved_os_stats['actual_neuron_idx'] = good_neurons_idx[ordering[:num_neurons_plot]]  # matches experimental records
    saved_os_stats['predictions'] = all_neurons_r_noise_ceil[ordering[:num_neurons_plot]]
    np.save(os.path.join(saved_model_folder, "{}_{}_{}{}_{}_saved_os_stats.npy".format(area,argsdict["model"],('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer)), saved_os_stats)


    plot_rows = int(np.ceil(num_neurons_plot/plot_cols))
    fig_opt_stim = plt.figure('OptStim',figsize=(24,plot_rows*4))
    plt.title("Optimal Stimuli estimated for {} neurons: {}{}_{} and variation loss:{}".format(num_neurons_plot,
                                                                                               'pretrained' if pretrained else 'untrained','_e2e' if e2e else '',layer,total_variation_penalty))

    fig_opt_stim_filter = plt.figure('OptStim_filter',figsize=(24,plot_rows*4))
    plt.title("Optimal Stimuli estimated and filtered using spatial weights (RF) for {} neurons: {}{}_{} and variation loss:{}".format(num_neurons_plot,
                                                                                                                                       'pretrained' if pretrained else 'untrained','_e2e' if e2e else '',layer,total_variation_penalty))

    print("...Generating optimal stimuli...")
    for neurons_batch in range(1+num_neurons_plot//batch_size):
        start_idx = neurons_batch*batch_size
        end_idx = num_neurons_plot if neurons_batch==(num_neurons_plot//batch_size) else batch_size + neurons_batch*batch_size
        neuron_idxs = ordering[start_idx:end_idx]

        if use_lucent:
            opt_stim = optStimLucent(DNN=DNN,readout=readout,neurons=neuron_idxs, device=device)
        else:
            X,y,m = next(iter(val_loader))  # X: N,c,w,h;    y: N, n;    m: N,n
            inp_img = torch.randn((len(neuron_idxs),)+X[0,0].shape).unsqueeze(1).repeat(1,3,1,1)  # n_neurons, 3, 135, 135
            opt_stim,loss_arr = deepDream(img=inp_img, DNN=DNN, readout=readout, neurons=neuron_idxs,
                                          iterations=2000, lr=0.5, total_variation_penalty=total_variation_penalty)

            # plt.figure("LossCurves")
            # if num_neurons_plot//batch_size>0:
            #     plt.subplot(1+num_neurons_plot//batch_size,1,neurons_batch+1)
            # plt.plot(loss_arr)
            # plt.xlabel('Iterations')
            # plt.ylabel('Loss')

        for idx,neuron_idx in enumerate(neuron_idxs):  # neuron_idxs come from ordering
            # fig_opt_stim = plt.figure('OptStim')
            # rf = plt.subplot(plot_rows,plot_cols,start_idx+idx+1)

            color_norm = mcolors.Normalize(vmin=opt_stim[idx].min(),vmax=opt_stim[idx].max())
            optimal_stim_collection_all.append(color_norm(opt_stim[idx]))
            optimal_stim_idx_collection_all.append(neuron_idx)
            if all_neurons_r_noise_ceil[neuron_idx]>=0.7:
                optimal_stim_idx_collection.append(neuron_idx)
                optimal_stim_collection.append(color_norm(opt_stim[idx]))
            # if opt_stim[idx].shape[0]>1:
            #     rf.imshow(color_norm(opt_stim[idx]).squeeze().transpose([1,2,0]))
            # else:
            #     rf.imshow(color_norm(opt_stim[idx]).squeeze(),cmap='gray')
            # rf.set_title("({:.3f},{:.3f},{:.3f})".format(all_neurons_r[neuron_idx],noise_ceiling[neuron_idx],
            #                                              all_neurons_r_noise_ceil[neuron_idx]),{'fontsize':13})
            # if all_neurons_r_noise_ceil[neuron_idx]>=1.0:
            #     rf = set_axes_border_prop(rf,"green",4)
            # elif all_neurons_r_noise_ceil[neuron_idx]>=0.7:
            #     rf = set_axes_border_prop(rf,"gold",4)
            # else:
            #     rf = set_axes_border_prop(rf,"maroon",4)

            # fig_opt_stim_filter = plt.figure('OptStim_filter')
            # breakpoint()
            spatial_filter_neuron = upsample_spatial_filter(np.abs(spatial_weights[neuron_idx]),input_size=spatial_weights.shape[-1],output_size=135)
            spatial_filter_neuron_norm = (spatial_filter_neuron-spatial_filter_neuron.min())/(spatial_filter_neuron.max()-spatial_filter_neuron.min())
            color_norm = mcolors.Normalize(vmin=opt_stim[idx].min(),vmax=opt_stim[idx].max())
            optimal_stim_RF_collection_all.append(spatial_filter_neuron_norm)
            if all_neurons_r_noise_ceil[neuron_idx]>=0.7:
                optimal_stim_RF_collection.append(spatial_filter_neuron_norm)
            # rf = plt.subplot(plot_rows,plot_cols,start_idx+idx+1)
            # if opt_stim[idx].shape[0]>1:
            #     opt_stim_filtered = spatial_filter_neuron_norm*color_norm(opt_stim[idx])
            #     rf.imshow(opt_stim_filtered.squeeze().transpose([1,2,0]))
            # else:
            #     opt_stim_filtered = spatial_filter_neuron_norm*(2*color_norm(opt_stim[idx])-1)
            #     max_abs_val = np.max(np.abs(opt_stim_filtered))
            #     rf.imshow(opt_stim_filtered.squeeze(),cmap='seismic',vmin=-max_abs_val,vmax=max_abs_val)
            # rf.set_title("({:.3f},{:.3f},{:.3f})".format(all_neurons_r[neuron_idx],noise_ceiling[neuron_idx],
            #                                              all_neurons_r_noise_ceil[neuron_idx]),{'fontsize':13})
            # if all_neurons_r_noise_ceil[neuron_idx]>=1.0:
            #     rf = set_axes_border_prop(rf,"green",4)
            # elif all_neurons_r_noise_ceil[neuron_idx]>=0.7:
            #     rf = set_axes_border_prop(rf,"gold",4)
            # else:
            #     rf = set_axes_border_prop(rf,"maroon",4)

            # fig_opt_stim_gray_filter = plt.figure('OptStimGray_filter')
            # rf = plt.subplot(plot_rows,plot_cols,start_idx+idx+1)
            # if opt_stim[idx].shape[0]>1:
            #     rf.imshow(rgb2gray(opt_stim_filtered.transpose([1,2,0])),cmap='gray') 	#,vmin=-0.95,vmax=0.95
            # else:
            #     rf.imshow(opt_stim_filtered.squeeze(),cmap='gray',vmin=-max_abs_val,vmax=max_abs_val)
            # rf.set_title("({:.3f},{:.3f},{:.3f})".format(all_neurons_r[neuron_idx],noise_ceiling[neuron_idx],
            #                                              all_neurons_r_noise_ceil[neuron_idx]),{'fontsize':13})
            # if all_neurons_r_noise_ceil[neuron_idx]>=1.0:
            #     rf = set_axes_border_prop(rf,"green",4)
            # elif all_neurons_r_noise_ceil[neuron_idx]>=0.7:
            #     rf = set_axes_border_prop(rf,"gold",4)
            # else:
            #     rf = set_axes_border_prop(rf,"maroon",4)

    optimal_stim_idx_collection = np.array(optimal_stim_idx_collection)
    optimal_stim_collection = np.array(optimal_stim_collection)
    optimal_stim_RF_collection = np.array(optimal_stim_RF_collection)
    optimal_stim_idx_collection_all = np.array(optimal_stim_idx_collection_all)
    optimal_stim_collection_all = np.array(optimal_stim_collection_all)
    optimal_stim_RF_collection_all = np.array(optimal_stim_RF_collection_all)
    opt_stim_dict = {'optStims':optimal_stim_collection, 'RFs':optimal_stim_RF_collection, 'neuron_idxs': optimal_stim_idx_collection}
    opt_stim_dict_all = {'optStims':optimal_stim_collection_all, 'RFs':optimal_stim_RF_collection_all, 'neuron_idxs': optimal_stim_idx_collection_all}

    print("...Save OS and RF as .npy ....")  # Note that, here the RFs are not processed; you're literally slapping spatial weights onto the optimal stimuli
    if save_mode == 'good':
        print("Optimal Stimuli Bank consists of {} neurons".format(len(optimal_stim_collection)))
        np.save(os.path.join(saved_model_folder,"{}_{}_{}{}_{}_optStim{}_grayscale_ColorNorm_good.npy".format(area,argsdict["model"],
                                                                                                        ('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer,'_lucent' if use_lucent else '')),opt_stim_dict,allow_pickle=True)
        optStim = opt_stim_dict
    elif save_mode == "all":
        print("Optimal Stimuli Bank consists of {} neurons".format(len(optimal_stim_collection_all)))
        np.save(os.path.join(saved_model_folder,"{}_{}_{}{}_{}_optStim{}_grayscale_ColorNorm_all.npy".format(area,argsdict["model"],
                                                                                                         ('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer,'_lucent' if use_lucent else '')),opt_stim_dict_all,allow_pickle=True)
        optStim = opt_stim_dict_all

# =================================================================
# ============= Process RF and save RF.npy and PDFs ===============
# =================================================================

new_opt_stim_dict = {}
optStims = optStim['optStims']  # n_neurons, 1, 135, 135
RFs = optStim['RFs'] # n_neurons, 135, 136

binarized_RFs, dilated_binarized_RFs, dilated_binarized_ConnectedComp_RFs, dilated_binarized_ConnectedComp_gaussian_RFs, RF_stats = binarize_dilate_ConnectedComp_gaussian_RF(RFs)
masked_optStims, one_minus_masked_optStims = apply_RF(optStims, dilated_binarized_ConnectedComp_gaussian_RFs)
new_opt_stim_dict['optStims'] = optStims
new_opt_stim_dict['processed_RFs'] = dilated_binarized_ConnectedComp_gaussian_RFs  # smoothed mask
new_opt_stim_dict['neuron_idxs'] = optStim['neuron_idxs']
new_opt_stim_dict['preblur_RFs'] = dilated_binarized_ConnectedComp_RFs  # not smoothed mask

np.save(os.path.join(save_dir,"{}_{}_{}{}_{}_optStims_processed_RFs.npy".format(area,argsdict["model"],('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer)), new_opt_stim_dict)
np.save(os.path.join(save_dir,"{}_{}_{}{}_{}_processed_RF_stats.npy".format(area,argsdict["model"],('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer)), RF_stats)

plt.set_cmap("gray")
# Create figure
plot_cols = 10
plot_rows = int(np.ceil(len(optStims / plot_cols)))
fig_opt_stim_no_box = plt.figure('OptStim_no_box', figsize=(24, plot_rows * 4))  # , figsize=(24, plot_rows * 4)
for i_neuron in range(len(optStims)):
    os_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
    os_plot.imshow(optStims[i_neuron].squeeze(), vmin=0, vmax=1)

fig_original_rf = plt.figure('Original_RF', figsize=(24, plot_rows * 4))
for i_neuron in range(len(optStims)):
    os_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
    os_plot.imshow(RFs[i_neuron], vmin=0, vmax=1)

fig_opt_stim_masked = plt.figure('OptStim_masked', figsize=(24, plot_rows * 4))
for i_neuron in range(len(optStims)):
    os_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
    os_plot.imshow(masked_optStims[i_neuron], vmin=-1, vmax=1)

fig_opt_stim_one_minus_mask = plt.figure('OptStim_one_minus_mask', figsize=(24, plot_rows * 4))
for i_neuron in range(len(optStims)):
    os_plot = plt.subplot(plot_rows, plot_cols, i_neuron+1)
    os_plot.imshow(one_minus_masked_optStims[i_neuron], vmin=-1, vmax=1)

print("...Saving OS and processed RF into a big PDF...")
with PdfPages(os.path.join(save_dir, "{}_{}_{}{}_{}_optStims_processed_RFs.pdf".format(area,argsdict["model"],('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer))) as pdf:
    try:
        pdf.savefig(fig_opt_stim_no_box)
        plt.close(fig_opt_stim_no_box)
    except:
        pass
    try:
        pdf.savefig(fig_original_rf)
        plt.close(fig_original_rf)
    except:
        pass
    try:
        pdf.savefig(fig_opt_stim_masked)
        plt.close(fig_opt_stim_masked)
    except:
        pass
    try:
        pdf.savefig(fig_opt_stim_one_minus_mask)
        plt.close(fig_opt_stim_one_minus_mask)
    except:
        pass
    plt.close('all')

# =======================================================================
# Make directories and save optimal stimuli (og, masked, 1-masked) as png
# =======================================================================

big_dir = os.path.join(save_dir, "{}_{}_{}{}_{}_optimal_stimuli".format(area,argsdict["model"],('pretrained' if pretrained else 'untrained'),('_e2e' if e2e else ''),layer))
if not os.path.exists(big_dir):
    os.mkdir(big_dir)

print("...Saving OS as png...")
dir = os.path.join(big_dir, "optStim")
if not os.path.exists(dir):
    os.mkdir(dir)
for i, neuron_idx in enumerate(optStim['neuron_idxs']):
    opt_stim = np.moveaxis(new_opt_stim_dict['optStims'][i], 0, -1).repeat(3, axis=-1)
    plt.imsave(os.path.join(dir, f"{neuron_idx}.png"), opt_stim, cmap="gray")

# masked OS
print("Saving masked OS")
dir = os.path.join(big_dir, "optStim_masked")
if not os.path.exists(dir):
    os.mkdir(dir)
for i, neuron_idx in enumerate(optStim['neuron_idxs']):
    opt_stim_filtered = np.moveaxis(new_opt_stim_dict['processed_RFs'][i]*(2*new_opt_stim_dict['optStims'][i]-1), 0, -1).repeat(3, axis=-1)
    opt_stim_filtered = (opt_stim_filtered+1)/2
    plt.imsave(os.path.join(dir, f"{neuron_idx}.png"), opt_stim_filtered, cmap="gray", vmin=-np.max(np.abs(opt_stim_filtered)),vmax=np.max(np.abs(opt_stim_filtered)))

# 1-masked OS
print("Saving 1-masked OS")
dir = os.path.join(big_dir, "optStim_one_minus_masked")
if not os.path.exists(dir):
    os.mkdir(dir)
for i, neuron_idx in enumerate(optStim['neuron_idxs']):
    opt_stim_filtered = np.moveaxis((1-new_opt_stim_dict['processed_RFs'][i])*(2*new_opt_stim_dict['optStims'][i]-1), 0, -1).repeat(3, axis=-1)
    opt_stim_filtered = (opt_stim_filtered+1)/2
    plt.imsave(os.path.join(dir, f"{neuron_idx}.png"), opt_stim_filtered, cmap="gray", vmin=-np.max(np.abs(opt_stim_filtered)),vmax=np.max(np.abs(opt_stim_filtered)))

print("complete")
