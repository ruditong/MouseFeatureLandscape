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
import time, os, glob
import argparse
from models import DNNActivations, FactorizedReadOut, ReadOut
from train_utils import *
import matplotlib
matplotlib.use('Agg')

# Arguments
parser = argparse.ArgumentParser(description="Factorized readout model to predict neuronal response")
# parser.add_argument("--data_folder", type=str,
#                     default='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
#                     help='Path to data files')
parser.add_argument("--image_data_folder",type=str,default='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
                    help='Path to directory in which DNN_images.npy is stores')
parser.add_argument("--response_data_folder",type=str,default='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
                    help='Path to rep_stims.npy and unique_stims.npy')

parser.add_argument("--model", type=str, default="VGG", help="Pretrained model architecture")
parser.add_argument("--pretrained", type=str, default="True", help="use pretrained model weights or random weights")
parser.add_argument("--layer", type=int, default=17, help="depth of DNN backbone to use for feature extraction")
parser.add_argument("--area", type=str, default="V1", help="Visual cortex area to be used")
parser.add_argument("--bias", type=str, default="True", help="use bias for readout")
parser.add_argument("--normalize", type=str, default="False", help="normalize spatial weights")
parser.add_argument("--end_to_end", type=str, default="True", help="train backbone DNN as well")
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs to train")
parser.add_argument("--batch_size", type=int, default=256, help="size of one minibatch")
parser.add_argument("--weight_decay", type=float, default=5e-2, help="weight decay")
parser.add_argument("--l1_norm_weight", type=float, default=5e-2, help="L1 weight decay")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--lrd", type=float, default=5e-3, help="initial learning rate decay")
parser.add_argument(
    "--save_dir",
    type=str,
    default="experiments/",
    help="path to save the experimental config, logs, model. This is not automatically generated.",
)
parser.add_argument("--seed", type=int, default=13, help="random seed")
parser.add_argument("--save_model_freq", type=int, default=10, help="Save model every X epochs")
parser.add_argument("--save_model_pred", type=str, default=False, help="Save model prediction r for all neurons")
parser.add_argument("--r_reduction", type=str, default='mean', help="Mean or 75percentile of r scores")
parser.add_argument("--plot_trends", type=str, default="False", help="Plot train and validation loss plots")
parser.add_argument("--verbose", type=str, default="False", help="Print debug statements")
parser.add_argument("--ckpt_path", type=str, default="None", help="path to .pth file for shallow SimCLR checkpoint")
parser.add_argument("--normalize_by_noise_ceiling", type=str, default="True", help="Normalize pearson r of predictions by noise ceiling")
args = parser.parse_args()
argsdict = args.__dict__

# Setting up cuda and seeds
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")  # DO THIS FOR MILA CLUSTER
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # DO THIS FOR MNI GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # DO THIS FOR MNI GPU
torch.manual_seed(argsdict["seed"])
np.random.seed(argsdict["seed"])

# Setting rest of the arguments
# data_folder = argsdict["data_folder"]
image_data_folder = argsdict["image_data_folder"]
response_data_folder = argsdict["response_data_folder"]
if argsdict["model"] == 'VGG':
    model_class = models.vgg16
else:
    model_class = argsdict["model"]  # eg. shallowConvfeat_4
pretrained = True if argsdict["pretrained"] == "True" or argsdict["pretrained"] == True else False
layer = argsdict["layer"]
area = argsdict["area"]
bias = True if argsdict["bias"] == "True" or argsdict["bias"] == True else False
normalize = True if argsdict["normalize"] == "True" or argsdict["normalize"] == True else False
end_to_end = True if argsdict["end_to_end"] == "True" or argsdict["end_to_end"] == True else False
epochs = argsdict["num_epochs"]
batch_size = argsdict["batch_size"]
weight_decay = argsdict["weight_decay"]
l1_norm_weight = argsdict["l1_norm_weight"]
learning_rate = argsdict["lr"]
learning_rate_decay = argsdict["lrd"]
save_dir = argsdict["save_dir"]
tmp_save_dir = argsdict["save_dir"]  # DO THIS ON MNI GPU
# tmp_save_dir = os.environ['SLURM_TMPDIR'] # DO THIS ON MILA CLUSTER
save_model_freq = argsdict["save_model_freq"]
save_model_pred = True if argsdict["save_model_pred"] == "True" or argsdict["save_model_pred"] == True else False
r_reduction = argsdict["r_reduction"]
plot_trends = True if argsdict["plot_trends"] == "True" or argsdict["plot_trends"] == True else False
verbose = True if argsdict["verbose"] == "True" or argsdict["verbose"] == True else False
ckpt_path = argsdict["ckpt_path"]
normalize_by_noise_ceiling = True if argsdict["normalize_by_noise_ceiling"] == "True" or argsdict["normalize_by_noise_ceiling"] == True else False

def copy_saved_model(curr_dir, final_dir):
    if verbose:
        tqdm.write("Moving saved model from {} to {}".format(curr_dir, final_dir))
    model_files = glob.glob(os.path.join(curr_dir, '*.pt'))
    for file in model_files:
        os.system('mv {} {}'.format(os.path.join(curr_dir, os.path.basename(file)), final_dir))


Files = ['part1', 'part2', 'part3', 'part4']

print(argsdict)

filename_check = os.path.join(save_dir, "{}_{}{}_{}_predictions.npy".format(area,
                                                                            (
                                                                                'pretrained' if pretrained else 'untrained'),
                                                                            ('_e2e' if end_to_end else ''), layer))
assert not os.path.exists(filename_check), "Area {} because it already exists!".format(area)

# Setting up seeds
torch.manual_seed(argsdict["seed"])
np.random.seed(argsdict["seed"])
# noise_ceil_dict = np.load(os.path.join(data_folder, area, 'noise_ceiling_dict.npy'), allow_pickle=True).item()
# all_good_neurons = 0
# for key in noise_ceil_dict.keys():
#     all_good_neurons += len(noise_ceil_dict[key]['noise_ceiling_arr'])
# assert all_good_neurons > 0, "No good neurons for this region"
gc.collect()

random_r_arr = []
best_r_arr = []
best_r_epoch_arr = []
train_r_arr = []
random_r_75_arr = []
best_r_75_arr = []
best_r_75_epoch_arr = []
train_r_75_arr = []

# good_neurons_idx, noise_ceiling, train_loader, val_loader, DNN, readout = load_define_data_model_factorized_v3(data_folder=data_folder,
#     area=area,
#     files=Files, batch_size=batch_size, model_class=model_class, layer=layer, pretrained=pretrained,
#     bias=bias, normalize=normalize, ckpt_path=ckpt_path)
good_neurons_idx, noise_ceiling, train_loader, val_loader, DNN, readout = load_define_data_model_factorized_v5(
    response_data_folder=response_data_folder, image_data_folder=image_data_folder,batch_size=batch_size, model_class=model_class, layer=layer, pretrained=pretrained,
                                                                                                              bias=bias, normalize=normalize, ckpt_path=ckpt_path)
print(f"Noise ceiling: mean {noise_ceiling.mean():.4f} (min {noise_ceiling.min():.4f}, max {noise_ceiling.max():.4f}), std {noise_ceiling.std():.4f}")
# ============= FOR MULTI_DAY TESTING =============
# model_fname = '/home/mila/l/lindongy/MouseNeuronPredict-MNIMila/DNNexperiments/models_train/experiments/multiday/day1/RL_test_VGG_pretrained_e2e_28_DNN.pt'
# readout_fname = '/home/mila/l/lindongy/MouseNeuronPredict-MNIMila/DNNexperiments/models_train/experiments/multiday/day1/RL_test_VGG_pretrained_e2e_28_FactorizedReadout.pt'
# DNN.load_state_dict(torch.load(model_fname))
# readout.load_state_dict(torch.load(readout_fname))
# DNN.eval()
# readout.eval()
# DNN = DNN.to(device)
# readout = readout.to(device)
# ===================================================

if end_to_end:
    optimizer = optim.Adam(list(DNN.parameters()) + list(readout.parameters()), lr=learning_rate,
                           weight_decay=weight_decay)
    train_fn = train_e2e
else:
    optimizer = optim.Adam(readout.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_fn = train
random_loss, random_r, random_r_75 = test_v2(DNN=DNN, ReadOutModel=readout,val_loader=val_loader,
                                            noise_ceiling=noise_ceiling,normalize_by_noise_ceiling=normalize_by_noise_ceiling,verbose=verbose)
evaluate_on_train(DNN=DNN, ReadOutModel=readout, train_loader=train_loader, noise_ceiling=noise_ceiling, normalize_by_noise_ceiling=normalize_by_noise_ceiling,verbose=verbose)
best_r = random_r
best_r_epoch = 0
train_r = random_r
best_r_75 = random_r_75
best_r_75_epoch = 0
train_r_75 = random_r_75
if plot_trends:
    train_loss_trend = []
    val_loss_trend = [random_loss]
    val_loss_ind = [0]
for epoch in tqdm(range(1, epochs + 1)):
    gc.collect()
    tr_loss = train_fn(DNN=DNN, ReadOutModel=readout, optimizer=optimizer,
                       train_loader=train_loader, epoch=epoch, learning_rate=learning_rate,
                       learning_rate_decay=learning_rate_decay, l1_norm_weight=l1_norm_weight, verbose=verbose)
    if plot_trends:
        train_loss_trend.append(tr_loss)
    if epoch % save_model_freq == 0:
        # val_loss, r, r_noise_ceil, r_75_noise_ceil = test(DNN=DNN,ReadOutModel=readout,
        # 	val_loader=val_loader,noise_ceiling=noise_ceiling,reduction=r_reduction,verbose=verbose)
        val_loss, r, r_75 = test_v2(DNN=DNN, ReadOutModel=readout,val_loader=val_loader, noise_ceiling=noise_ceiling,normalize_by_noise_ceiling=normalize_by_noise_ceiling,
                                                                verbose=verbose)
        if plot_trends:
            val_loss_trend.append(val_loss)
            val_loss_ind.append(epoch)
        if (r_reduction == 'mean' and r >= best_r) or (
                r_reduction == '75per' and r_75 >= best_r_75):
            best_r = r
            best_r_epoch = epoch
            best_r_75 = r_75
            best_r_75_epoch = epoch
            train_r, train_r_75 = evaluate_on_train(DNN=DNN, ReadOutModel=readout,
                                                      train_loader=train_loader, noise_ceiling=noise_ceiling,normalize_by_noise_ceiling=normalize_by_noise_ceiling,
                                                      verbose=verbose)
            torch.save(DNN.state_dict(), os.path.join(tmp_save_dir,
                                                      "{}_{}_{}{}_{}_DNN.pt".format(area, argsdict["model"],
                                                                                       (
                                                                                           'pretrained' if pretrained else 'untrained'),
                                                                                       ('_e2e' if end_to_end else ''),
                                                                                       layer)))
            torch.save(readout.state_dict(), os.path.join(tmp_save_dir,
                                                          "{}_{}_{}{}_{}_FactorizedReadout.pt".format(area,
                                                                                                         argsdict[
                                                                                                             "model"], (
                                                                                                             'pretrained' if pretrained else 'untrained'),
                                                                                                         (
                                                                                                             '_e2e' if end_to_end else ''),
                                                                                                         layer)))
            if save_model_pred:
                all_neurons_r, r, r_75 = get_model_prediction_r(DNN=DNN,
                                                              ReadOutModel=readout,
                                                              val_loader=val_loader,
                                                              noise_ceiling=noise_ceiling,normalize_by_noise_ceiling=normalize_by_noise_ceiling,
                                                              verbose=verbose)
if verbose:
    print("Best Validation Pearson r wrt noise ceiling: ", best_r, best_r_epoch)

if plot_trends:
    fig, ax1 = plt.subplots()
    ln1 = ax1.plot(train_loss_trend, label='Train Loss', color='b')
    ax2 = ax1.twinx()
    ln2 = ax2.plot(val_loss_ind, val_loss_trend, label='Val Loss', color='r')
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs)
    # plt.savefig(os.path.join(os.environ['SLURM_TMPDIR'],'{}_r_trend.png'.format(layer[:-3])))
    plt.savefig(os.path.join(save_dir, '{}_{}_{}{}_loss_trend.png'.format(area, argsdict["model"],
                                                                             (
                                                                                 'pretrained_' if pretrained else 'untrained_') + str(
                                                                                 layer),
                                                                             ('_e2e' if end_to_end else ''))))
    plt.close()
copy_saved_model(tmp_save_dir, save_dir)
if save_model_pred:
    np.save(os.path.join(save_dir, "{}_{}_{}{}_{}_predictions.npy".format(area, argsdict["model"],
                                                                             (
                                                                                 'pretrained' if pretrained else 'untrained'),
                                                                             ('_e2e' if end_to_end else ''), layer)),
            all_neurons_r)
random_r_arr.append(random_r)
best_r_arr.append(best_r)
best_r_epoch_arr.append(best_r_epoch)
train_r_arr.append(train_r)
random_r_75_arr.append(random_r_75)
best_r_75_arr.append(best_r_75)
best_r_75_epoch_arr.append(best_r_75_epoch)
train_r_75_arr.append(train_r_75)

random_r_arr = np.array(random_r_arr)
best_r_arr = np.array(best_r_arr)
best_r_epoch_arr = np.array(best_r_epoch_arr)
train_r_arr = np.array(train_r_arr)
random_r_75_arr = np.array(random_r_75_arr)
best_r_75_arr = np.array(best_r_75_arr)
best_r_75_epoch_arr = np.array(best_r_75_epoch_arr)
train_r_75_arr = np.array(train_r_75_arr)

print("For {}, random r_pred was {:.4f} and best r_pred (mean) is {:.4f} ".format(
    ('pretrained_' if pretrained else 'untrained_') + ('e2e_' if end_to_end else '') + str(layer), random_r_arr.mean(),
    best_r_arr.mean()) + " after {:.2f} epochs [Train r is {:.4f}]".format(
    best_r_epoch_arr.mean(), train_r_arr.mean()))

print("For {}, random r_pred was {:.4f} and best r_pred (75per) is {:.4f}".format(
    ('pretrained_' if pretrained else 'untrained_') + ('e2e_' if end_to_end else '') + str(layer),
    random_r_75_arr.mean(),
    best_r_75_arr.mean()) + " after {:.2f} epochs [Train r is {:.4f}]".format(
    best_r_75_epoch_arr.mean(), train_r_75_arr.mean()))
