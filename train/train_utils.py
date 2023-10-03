import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt
from dataloader import ImageNeuronDataset_v2, ImageNeuronDataset_v3, ImageNeuronDataset_v4
from models import DNNActivations, FactorizedReadOut, LinearRegression
import gc, os
# data_folder = '/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data'  # TODO: change data folder

def load_define_data_model_factorized_v5(response_data_folder,
                                         image_data_folder='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
                                         batch_size=256,model_class=models.vgg16,layer=11,pretrained=True,ckpt_path=None,bias=True,normalize=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    noise_ceiling_dict = np.load(os.path.join(response_data_folder,"noise_ceiling_dict.npy"), allow_pickle=True).item()
    good_neurons_idx = noise_ceiling_dict['good_neurons_idx']
    noise_ceiling_arr = noise_ceiling_dict['noise_ceiling_arr']

    dataset = ImageNeuronDataset_v4(response_data_folder=response_data_folder,image_data_folder=image_data_folder,
                                    good_neurons_idx=good_neurons_idx, mode='test', tr_portion=1.0)
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)
    X,y,m = next(iter(loader)) # X: N,c,w,h;    y: N, n;    m: N,n
    if model_class != "linear":
        DNN = DNNActivations(model_class=model_class,layer=layer,pretrained=pretrained, ckpt_path=ckpt_path).to(device)
        with torch.no_grad():
            X_feats = DNN(X.to(device))  # X_feats: N,256,16,16
        # breakpoint()
        readout = FactorizedReadOut(inp_size=X_feats.size()[1:],out_size=y.size(1),bias=bias,normalize=normalize).to(device) # inp_size: 256,16,16; out_size: n_good_neurons
        with torch.no_grad():
            tmp_out = readout(X_feats)  # tmp_out: N,n_good_neurons
        del X,X_feats,tmp_out
    else:
        linear_model = LinearRegression(inp_size=X.size()[1:], out_size=y.size(1)).to(device)
        del X
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_dataset = ImageNeuronDataset_v4(response_data_folder=response_data_folder,image_data_folder=image_data_folder, good_neurons_idx=good_neurons_idx,mode='train')
    val_dataset = ImageNeuronDataset_v4(response_data_folder=response_data_folder,image_data_folder=image_data_folder, good_neurons_idx=good_neurons_idx,mode='test')
    # setting the image data mean to mean of entire dataset
    all_images = np.concatenate([train_dataset.images,val_dataset.images])
    img_mean = all_images.mean()
    img_std = all_images.std()
    del all_images
    train_dataset.mean = img_mean
    train_dataset.std = img_std
    train_dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((img_mean,img_mean,img_mean),(img_std,img_std,img_std))])
    val_dataset.mean = img_mean
    val_dataset.std = img_std
    val_dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((img_mean,img_mean,img_mean),(img_std,img_std,img_std))])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    if model_class != "linear":
        return good_neurons_idx, noise_ceiling_arr, train_loader, val_loader, DNN, readout
    else:
        return good_neurons_idx, noise_ceiling_arr, train_loader, val_loader, linear_model



def load_define_data_model_factorized_v4(response_data_folder,
                                         image_data_folder='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
                                         batch_size=256,model_class=models.vgg16,layer=11,pretrained=True,ckpt_path=None,bias=True,normalize=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    noise_ceiling_dict = np.load(os.path.join(response_data_folder,"noise_ceiling_dict.npy"), allow_pickle=True).item()
    good_neurons_idx = noise_ceiling_dict['good_neurons_idx']
    noise_ceiling_arr = noise_ceiling_dict['noise_ceiling_arr']

    dataset = ImageNeuronDataset_v3(response_data_folder=response_data_folder,image_data_folder=image_data_folder,
                                    good_neurons_idx=good_neurons_idx, mode='test', tr_portion=1.0)
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)
    X,y,m = next(iter(loader)) # X: N,c,w,h;    y: N, n;    m: N,n
    if model_class != "linear":
        DNN = DNNActivations(model_class=model_class,layer=layer,pretrained=pretrained, ckpt_path=ckpt_path).to(device)
        with torch.no_grad():
            X_feats = DNN(X.to(device))  # X_feats: N,256,16,16
        # breakpoint()
        readout = FactorizedReadOut(inp_size=X_feats.size()[1:],out_size=y.size(1),bias=bias,normalize=normalize).to(device) # inp_size: 256,16,16; out_size: n_good_neurons
        with torch.no_grad():
            tmp_out = readout(X_feats)  # tmp_out: N,n_good_neurons
        del X,X_feats,tmp_out
    else:
        linear_model = LinearRegression(inp_size=X.size()[1:], out_size=y.size(1)).to(device)
        del X
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_dataset = ImageNeuronDataset_v3(response_data_folder=response_data_folder,image_data_folder=image_data_folder, good_neurons_idx=good_neurons_idx,mode='train')
    val_dataset = ImageNeuronDataset_v3(response_data_folder=response_data_folder,image_data_folder=image_data_folder, good_neurons_idx=good_neurons_idx,mode='test')
    # setting the image data mean to mean of entire dataset
    all_images = np.concatenate([train_dataset.images,val_dataset.images])
    img_mean = all_images.mean()
    img_std = all_images.std()
    del all_images
    train_dataset.mean = img_mean
    train_dataset.std = img_std
    train_dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((img_mean,img_mean,img_mean),(img_std,img_std,img_std))])
    val_dataset.mean = img_mean
    val_dataset.std = img_std
    val_dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((img_mean,img_mean,img_mean),(img_std,img_std,img_std))])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    if model_class != "linear":
        return good_neurons_idx, noise_ceiling_arr, train_loader, val_loader, DNN, readout
    else:
        return good_neurons_idx, noise_ceiling_arr, train_loader, val_loader, linear_model


def load_define_data_model_factorized_v3(data_folder='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
                                         area='V1',files=['part1','part2','part3','part4'],
                                         batch_size=256,model_class=models.vgg16,layer=11,pretrained=True,ckpt_path=None,bias=True,normalize=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    good_neurons_idx_list = []
    noise_ceiling_list = []
    noise_ceiling_dict = np.load(os.path.join(data_folder,area,'noise_ceiling_dict.npy'),allow_pickle=True).item()
    for fidx,file in enumerate(files):
        if file not in noise_ceiling_dict.keys():
            continue
        good_neurons_idx = noise_ceiling_dict[file]['good_neurons_idx']
        if len(good_neurons_idx)==0:
            continue
        noise_ceiling = noise_ceiling_dict[file]['noise_ceiling_arr']
        good_neurons_idx_list.append(good_neurons_idx)
        noise_ceiling_list.append(noise_ceiling)
    if len(good_neurons_idx_list)==0:
        return None, None, None, None, None, None
    good_neurons_idx_list = np.concatenate(good_neurons_idx_list)
    noise_ceiling_list = np.concatenate(noise_ceiling_list)

    # dataset = ImageNeuronDataset_v2(data_folder=data_folder,area=area,files=files,mode='test',time_indices=np.arange(8,12))
    dataset = ImageNeuronDataset_v2(data_folder=data_folder,area=area,files=files,mode='test',time_indices=np.arange(0,10))

    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)
    X,y,m = next(iter(loader)) # X: N,c,w,h;    y: N, n;    m: N,n
    if model_class != "linear":
        DNN = DNNActivations(model_class=model_class,layer=layer,pretrained=pretrained, ckpt_path=ckpt_path).to(device)
        with torch.no_grad():
            X_feats = DNN(X.to(device))  # X_feats: N,256,16,16
        # breakpoint()
        readout = FactorizedReadOut(inp_size=X_feats.size()[1:],out_size=y.size(1),bias=bias,normalize=normalize).to(device) # inp_size: 256,16,16; out_size: n_good_neurons
        with torch.no_grad():
            tmp_out = readout(X_feats)  # tmp_out: N,n_good_neurons
        del X,X_feats,tmp_out
    else:
        linear_model = LinearRegression(inp_size=X.size()[1:], out_size=y.size(1)).to(device)
        del X
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # train_dataset = ImageNeuronDataset_v2(data_folder=data_folder,area=area,files=files,mode='train',time_indices=np.arange(8,12))
    # val_dataset = ImageNeuronDataset_v2(data_folder=data_folder,area=area,files=files,mode='test',time_indices=np.arange(8,12))
    train_dataset = ImageNeuronDataset_v2(data_folder=data_folder,area=area,files=files,mode='train',time_indices=np.arange(0,10))
    val_dataset = ImageNeuronDataset_v2(data_folder=data_folder,area=area,files=files,mode='test',time_indices=np.arange(0,10))
    # setting the image data mean to mean of entire dataset
    all_images = np.concatenate([train_dataset.images,val_dataset.images])
    img_mean = all_images.mean()
    img_std = all_images.std()
    del all_images
    train_dataset.mean = img_mean
    train_dataset.std = img_std
    train_dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((img_mean,img_mean,img_mean),(img_std,img_std,img_std))])
    val_dataset.mean = img_mean
    val_dataset.std = img_std
    val_dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((img_mean,img_mean,img_mean),(img_std,img_std,img_std))])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    if model_class != "linear":
        return good_neurons_idx_list, noise_ceiling_list, train_loader, val_loader, DNN, readout
    else:
        return good_neurons_idx_list, noise_ceiling_list, train_loader, val_loader, linear_model


def norm_dot_product(output, target, mask):
    assert output.shape == target.shape, "Output ({}) and target ({}) shapes should match!".format(output.shape, target.shape)
    assert output.shape == mask.shape, "Output ({}) and mask ({}) shapes should match!".format(output.shape, mask.shape)
    # Normalize output and target
    output = F.normalize(output*mask, dim=0)  # Normalize the N-element vector for each neuron, N=batch size
    target = F.normalize(target*mask, dim=0)
    dot_product = 0
    for i_neuron in range(target.shape[-1]):
        ############################## RT EDIT ##############################
        dot_product += torch.dot(output[:, i_neuron], target[:, i_neuron])
        #dot_product -= torch.sum((output[:, i_neuron]-target[:, i_neuron])**2)
    return dot_product


def train(DNN,ReadOutModel,optimizer,train_loader,epoch,learning_rate=0.001,learning_rate_decay=0.01,l1_norm_weight=1e-3,verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_interval = 5
    DNN.eval()
    ReadOutModel.train()
    if learning_rate_decay is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = (learning_rate)/(1+epoch*learning_rate_decay)
    epoch_loss = 0
    for batch_idx, (data,target,mask) in enumerate(train_loader):
        # breakpoint()
        # tqdm.write("{:.4f} {:.4f} {:.4f} {:.4f}".format(data.mean(),data.std(),data.min(),data.max()))
        target = target.to(device)  # N x n_neurons
        data = data.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            feats = DNN(data)
        output = ReadOutModel(feats)
        # breakpoint()
        loss = -norm_dot_product(output, target, mask)  # Want to maximize dot product
        # loss = F.mse_loss(output*mask, target*mask, reduction='sum')
        loss += l1_norm_weight*ReadOutModel.l1()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
        if verbose and batch_idx % log_interval == 0:
            tqdm.write("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data),
                                                                                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        del target,data,mask,feats,loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return epoch_loss/len(train_loader.dataset)


def train_e2e(DNN,ReadOutModel,optimizer,train_loader,epoch,learning_rate=0.001,learning_rate_decay=0.01,l1_norm_weight=1e-3,verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_interval = 5
    DNN.train()
    ReadOutModel.train()
    if learning_rate_decay is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = (learning_rate)/(1+epoch*learning_rate_decay)
    epoch_loss = 0
    for batch_idx, (data,target,mask) in enumerate(train_loader):
        # tqdm.write("{:.4f} {:.4f} {:.4f} {:.4f}".format(data.mean(),data.std(),data.min(),data.max()))
        target = target.to(device)
        data = data.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        feats = DNN(data)
        output = ReadOutModel(feats)
        loss = -norm_dot_product(output, target, mask)
        # loss = F.mse_loss(output*mask, target*mask, reduction='sum')
        loss += l1_norm_weight*ReadOutModel.l1()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
        if verbose and batch_idx % log_interval == 0:
            tqdm.write("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data),
                                                                                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        del target,data,mask,feats,output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return epoch_loss/len(train_loader.dataset)


def train_simple_model(model,optimizer,train_loader,epoch,learning_rate=0.001,learning_rate_decay=0.01,l1_norm_weight=1e-3,verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_interval = 5
    model.train()
    if learning_rate_decay is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = (learning_rate)/(1+epoch*learning_rate_decay)
    epoch_loss = 0
    for batch_idx, (data,target,mask) in enumerate(train_loader):

        target = target.to(device)  # N x n_neurons
        data = data.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = -norm_dot_product(output, target, mask)  # Want to maximize dot product
        # loss = F.mse_loss(output*mask, target*mask, reduction='sum')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
        if verbose and batch_idx % log_interval == 0:
            tqdm.write("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data),
                                                                                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        del target,data,mask,loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return epoch_loss/len(train_loader.dataset)


def compute_neuron_r2_score(ground_truths,predictions,masks):
    assert ground_truths.shape == predictions.shape, "Ground truths ({}) and predictions ({}) should have same shape".format(
        ground_truths.shape,predictions.shape)
    assert predictions.shape == masks.shape, "Predictions ({}) and masks ({}) should have same shape".format(
        predictions.shape,masks.shape)
    # breakpoint()
    r2_arr = [r2_score(ground_truths[masks[:,idx],idx],predictions[masks[:,idx],idx]) for idx in range(masks.shape[1])]
    return np.array(r2_arr)


def compute_pearson_r(ground_truths, predictions, masks):
    assert ground_truths.shape == predictions.shape, "Ground truths ({}) and predictions ({}) should have same shape".format(
        ground_truths.shape,predictions.shape)
    assert predictions.shape == masks.shape, "Predictions ({}) and masks ({}) should have same shape".format(
        predictions.shape,masks.shape)
    pearson_r_arr = [(stats.pearsonr(ground_truths[masks[:,idx],idx],predictions[masks[:,idx],idx]))[0] for idx in range(masks.shape[1])]
    return np.array(pearson_r_arr)


def evaluate_on_train(DNN,ReadOutModel,train_loader,noise_ceiling,normalize_by_noise_ceiling=True,verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DNN.eval()
    ReadOutModel.eval()
    loss = 0
    ground_truths = torch.Tensor(0)
    predictions = torch.Tensor(0)
    all_masks = torch.BoolTensor(0)
    with torch.no_grad():
        for data, target, mask in train_loader:
            # breakpoint()
            target = target.to(device)
            data = data.to(device)
            mask = mask.to(device)
            feats = DNN(data)
            output = ReadOutModel(feats)
            loss -= norm_dot_product(output, target, mask)
            # loss = F.mse_loss(output*mask, target*mask, reduction='sum')
            pred = output.cpu()
            predictions = torch.cat((predictions,pred),dim=0)
            ground_truths = torch.cat((ground_truths,target.cpu()),dim=0)
            all_masks = torch.cat((all_masks,mask.cpu()),dim=0)
            del data, target, mask, feats, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        loss /= len(train_loader.dataset)
        predictions = predictions.numpy()
        ground_truths = ground_truths.numpy()
        all_masks = all_masks.numpy()
        pred_r = compute_pearson_r(ground_truths,predictions,all_masks)
        pred_r_75_percentile = np.percentile(pred_r, 75)
        pred_r_75_percentile_arr = pred_r[pred_r>=pred_r_75_percentile]
        # pred_r2 = compute_neuron_r2_score(ground_truths,predictions,all_masks)
        r_noise_percent = np.maximum(pred_r,np.zeros(pred_r.shape))/noise_ceiling
        # tqdm.write("Train set: r_score: {:.4f}\n".format(pred_r))
        r_noise_percent_75percentile = np.percentile(r_noise_percent,75)
        r_noise_percent_75percentile_arr = r_noise_percent[r_noise_percent>=r_noise_percent_75percentile]
        if verbose:
            tqdm.write("Train set: Avg loss: {:.4f}, r_score: {:.4f}\n".format(loss,np.mean(pred_r)))
    if normalize_by_noise_ceiling:
        return np.mean(r_noise_percent), np.mean(r_noise_percent_75percentile_arr)
    else:
        return np.mean(pred_r), np.mean(pred_r_75_percentile_arr)


def evaluate_on_train_simple_model(model,train_loader,noise_ceiling,normalize_by_noise_ceiling=True,verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss = 0
    ground_truths = torch.Tensor(0)
    predictions = torch.Tensor(0)
    all_masks = torch.BoolTensor(0)
    with torch.no_grad():
        for data, target, mask in train_loader:
            # breakpoint()
            target = target.to(device)
            data = data.to(device)
            mask = mask.to(device)
            output = model(data)
            loss -= norm_dot_product(output, target, mask)
            # loss = F.mse_loss(output*mask, target*mask, reduction='sum')
            pred = output.cpu()
            predictions = torch.cat((predictions,pred),dim=0)
            ground_truths = torch.cat((ground_truths,target.cpu()),dim=0)
            all_masks = torch.cat((all_masks,mask.cpu()),dim=0)
            del data, target, mask, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        loss /= len(train_loader.dataset)
        predictions = predictions.numpy()
        ground_truths = ground_truths.numpy()
        all_masks = all_masks.numpy()
        pred_r = compute_pearson_r(ground_truths,predictions,all_masks)
        pred_r_75_percentile = np.percentile(pred_r, 75)
        pred_r_75_percentile_arr = pred_r[pred_r>=pred_r_75_percentile]
        # pred_r2 = compute_neuron_r2_score(ground_truths,predictions,all_masks)
        r_noise_percent = np.maximum(pred_r,np.zeros(pred_r.shape))/noise_ceiling
        # tqdm.write("Train set: r_score: {:.4f}\n".format(pred_r))
        r_noise_percent_75percentile = np.percentile(r_noise_percent,75)
        r_noise_percent_75percentile_arr = r_noise_percent[r_noise_percent>=r_noise_percent_75percentile]
        if verbose:
            tqdm.write("Train set: Avg loss: {:.4f}, r_score: {:.4f}\n".format(loss,np.mean(pred_r)))
    if normalize_by_noise_ceiling:
        return np.mean(r_noise_percent), np.mean(r_noise_percent_75percentile_arr)
    else:
        return np.mean(pred_r), np.mean(pred_r_75_percentile_arr)


def test_v2(DNN,ReadOutModel,val_loader,noise_ceiling,normalize_by_noise_ceiling=True,verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DNN.eval()
    ReadOutModel.eval()
    test_loss = 0
    ground_truths = torch.Tensor(0)
    predictions = torch.Tensor(0)
    all_masks = torch.BoolTensor(0)
    with torch.no_grad():
        for data, target, mask in val_loader:
            #breakpoint()
            target = target.to(device) # batch_size x n_neurons
            data = data.to(device)  # batch_size x C x H x W
            mask = mask.to(device)  # batch_size x n_neurons
            feats = DNN(data)
            output = ReadOutModel(feats)
            test_loss -= norm_dot_product(output, target, mask).cpu().item()
            # test_loss += F.mse_loss(output*mask, target*mask, reduction='sum')
            pred = output.cpu()
            predictions = torch.cat((predictions,pred),dim=0)  # stack batches together
            ground_truths = torch.cat((ground_truths,target.cpu()),dim=0)
            all_masks = torch.cat((all_masks,mask.cpu()),dim=0)
            #breakpoint()
            del data, target, mask, output, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        test_loss /= len(val_loader.dataset)
        predictions = predictions.numpy()
        ground_truths = ground_truths.numpy()
        all_masks = all_masks.numpy()
        pred_r = compute_pearson_r(ground_truths,predictions,all_masks)
        pred_r_75_percentile = np.percentile(pred_r, 75)
        pred_r_75_percentile_arr = pred_r[pred_r>=pred_r_75_percentile]
        # pred_r2 = compute_neuron_r2_score(ground_truths,predictions,all_masks)
        r_noise_percent = np.maximum(pred_r,np.zeros(pred_r.shape))/noise_ceiling
        r_noise_percent_75percentile = np.percentile(r_noise_percent,75)
        r_noise_percent_75percentile_arr = r_noise_percent[r_noise_percent>=r_noise_percent_75percentile]
        if verbose:
            if normalize_by_noise_ceiling:
                tqdm.write("{:.4f} ({:.4f},{:.4f}) {:.4f}({:.4f},{:.4f})".format(r_noise_percent.mean(),r_noise_percent.min(),r_noise_percent.max(),
                                                                                 r_noise_percent_75percentile_arr.mean(),r_noise_percent_75percentile_arr.min(),r_noise_percent_75percentile_arr.max()))
            else:
                tqdm.write("{:.4f} ({:.4f},{:.4f}) {:.4f}({:.4f},{:.4f})".format(pred_r.mean(),pred_r.min(),pred_r.max(),
                                                                                 pred_r_75_percentile_arr.mean(),pred_r_75_percentile_arr.min(),pred_r_75_percentile_arr.max()))
                tqdm.write(f"Noise ceiling: Mean {noise_ceiling.mean()} (Min {noise_ceiling.min()}, Max {noise_ceiling.max()})")
            tqdm.write("Test set: Avg loss: {:.4f}, r_score: {:.4f}\n".format(test_loss,np.mean(pred_r)))
    if normalize_by_noise_ceiling:
        return test_loss, np.mean(r_noise_percent), np.mean(r_noise_percent_75percentile_arr)
    else:
        return test_loss, np.mean(pred_r), np.mean(pred_r_75_percentile_arr)

def test_simple_model(model,val_loader,noise_ceiling,normalize_by_noise_ceiling=True,verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    ground_truths = torch.Tensor(0)
    predictions = torch.Tensor(0)
    all_masks = torch.BoolTensor(0)
    with torch.no_grad():
        for data, target, mask in val_loader:
            #breakpoint()
            target = target.to(device) # batch_size x n_neurons
            data = data.to(device)  # batch_size x C x H x W
            mask = mask.to(device)  # batch_size x n_neurons
            output = model(data)
            test_loss -= norm_dot_product(output, target, mask).cpu().item()
            # test_loss += F.mse_loss(output*mask, target*mask, reduction='sum')
            pred = output.cpu()
            predictions = torch.cat((predictions,pred),dim=0)  # stack batches together
            ground_truths = torch.cat((ground_truths,target.cpu()),dim=0)
            all_masks = torch.cat((all_masks,mask.cpu()),dim=0)
            #breakpoint()
            del data, target, mask, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        test_loss /= len(val_loader.dataset)
        predictions = predictions.numpy()
        ground_truths = ground_truths.numpy()
        all_masks = all_masks.numpy()
        pred_r = compute_pearson_r(ground_truths,predictions,all_masks)
        pred_r_75_percentile = np.percentile(pred_r, 75)
        pred_r_75_percentile_arr = pred_r[pred_r>=pred_r_75_percentile]
        # pred_r2 = compute_neuron_r2_score(ground_truths,predictions,all_masks)
        r_noise_percent = np.maximum(pred_r,np.zeros(pred_r.shape))/noise_ceiling
        r_noise_percent_75percentile = np.percentile(r_noise_percent,75)
        r_noise_percent_75percentile_arr = r_noise_percent[r_noise_percent>=r_noise_percent_75percentile]
        if verbose:
            if normalize_by_noise_ceiling:
                tqdm.write("{:.4f} ({:.4f},{:.4f}) {:.4f}({:.4f},{:.4f})".format(r_noise_percent.mean(),r_noise_percent.min(),r_noise_percent.max(),
                                                                                 r_noise_percent_75percentile_arr.mean(),r_noise_percent_75percentile_arr.min(),r_noise_percent_75percentile_arr.max()))
            else:
                tqdm.write("{:.4f} ({:.4f},{:.4f}) {:.4f}({:.4f},{:.4f})".format(pred_r.mean(),pred_r.min(),pred_r.max(),
                                                                                 pred_r_75_percentile_arr.mean(),pred_r_75_percentile_arr.min(),pred_r_75_percentile_arr.max()))
                tqdm.write(f"Noise ceiling: Mean {noise_ceiling.mean()} (Min {noise_ceiling.min()}, Max {noise_ceiling.max()})")
            tqdm.write("Test set: Avg loss: {:.4f}, r_score: {:.4f}\n".format(test_loss,np.mean(pred_r)))
    if normalize_by_noise_ceiling:
        return test_loss, np.mean(r_noise_percent), np.mean(r_noise_percent_75percentile_arr)
    else:
        return test_loss, np.mean(pred_r), np.mean(pred_r_75_percentile_arr)


def get_model_prediction_r(DNN,ReadOutModel,val_loader,noise_ceiling,normalize_by_noise_ceiling=True,verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DNN.eval()
    ReadOutModel.eval()
    test_loss = 0
    ground_truths = torch.Tensor(0)
    predictions = torch.Tensor(0)
    all_masks = torch.BoolTensor(0)
    with torch.no_grad():
        for data, target, mask in val_loader:
            target = target.to(device)
            data = data.to(device)
            mask = mask.to(device)
            feats = DNN(data)
            output = ReadOutModel(feats)
            test_loss -= norm_dot_product(output, target, mask)
            # loss = F.mse_loss(output*mask, target*mask, reduction='sum')
            pred = output.cpu()
            predictions = torch.cat((predictions,pred),dim=0)
            ground_truths = torch.cat((ground_truths,target.cpu()),dim=0)
            all_masks = torch.cat((all_masks,mask.cpu()),dim=0)
            del data, target, mask, output, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        test_loss /= len(val_loader.dataset)
        predictions = predictions.numpy()
        ground_truths = ground_truths.numpy()
        pred_r = compute_pearson_r(ground_truths,predictions,all_masks)
        pred_r_75_percentile = np.percentile(pred_r, 75)
        pred_r_75_percentile_arr = pred_r[pred_r>=pred_r_75_percentile]
        # pred_r2 = compute_neuron_r2_score(ground_truths,predictions,all_masks)
        r_noise_percent = np.maximum(pred_r,np.zeros(pred_r.shape))/noise_ceiling
        r_noise_percent_75percentile = np.percentile(r_noise_percent,75)
        r_noise_percent_75percentile_arr = r_noise_percent[r_noise_percent>=r_noise_percent_75percentile]
        if verbose:
            if normalize_by_noise_ceiling:
                tqdm.write("{:.4f} ({:.4f},{:.4f}) {:.4f}({:.4f},{:.4f})".format(r_noise_percent.mean(),r_noise_percent.min(),r_noise_percent.max(),
                                                                                 r_noise_percent_75percentile_arr.mean(),r_noise_percent_75percentile_arr.min(),r_noise_percent_75percentile_arr.max()))
            else:
                tqdm.write("{:.4f} ({:.4f},{:.4f}) {:.4f}({:.4f},{:.4f})".format(pred_r.mean(),pred_r.min(),pred_r.max(),
                                                                                 pred_r_75_percentile_arr.mean(),pred_r_75_percentile_arr.min(),pred_r_75_percentile_arr.max()))
                tqdm.write(f"Noise ceiling: Mean {noise_ceiling.mean()} (Min {noise_ceiling.min()}, Max {noise_ceiling.max()})")
            tqdm.write("Test set: Avg loss: {:.4f}, r_score: {:.4f}".format(test_loss,np.mean(pred_r)))
    if normalize_by_noise_ceiling:
        return pred_r, np.mean(r_noise_percent), np.mean(r_noise_percent_75percentile_arr)  # NOTE: no more np.mean(pred_r)
    else:
        return pred_r, np.mean(pred_r), np.mean(pred_r_75_percentile_arr)


def get_model_prediction_r_simple_model(model,val_loader,noise_ceiling,normalize_by_noise_ceiling=True,verbose=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    ground_truths = torch.Tensor(0)
    predictions = torch.Tensor(0)
    all_masks = torch.BoolTensor(0)
    with torch.no_grad():
        for data, target, mask in val_loader:
            target = target.to(device)
            data = data.to(device)
            mask = mask.to(device)
            output = model(data)
            test_loss -= norm_dot_product(output, target, mask)
            # loss = F.mse_loss(output*mask, target*mask, reduction='sum')
            pred = output.cpu()
            predictions = torch.cat((predictions,pred),dim=0)
            ground_truths = torch.cat((ground_truths,target.cpu()),dim=0)
            all_masks = torch.cat((all_masks,mask.cpu()),dim=0)
            del data, target, mask, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        test_loss /= len(val_loader.dataset)
        predictions = predictions.numpy()
        ground_truths = ground_truths.numpy()
        pred_r = compute_pearson_r(ground_truths,predictions,all_masks)
        pred_r_75_percentile = np.percentile(pred_r, 75)
        pred_r_75_percentile_arr = pred_r[pred_r>=pred_r_75_percentile]
        # pred_r2 = compute_neuron_r2_score(ground_truths,predictions,all_masks)
        r_noise_percent = np.maximum(pred_r,np.zeros(pred_r.shape))/noise_ceiling
        r_noise_percent_75percentile = np.percentile(r_noise_percent,75)
        r_noise_percent_75percentile_arr = r_noise_percent[r_noise_percent>=r_noise_percent_75percentile]
        if verbose:
            if normalize_by_noise_ceiling:
                tqdm.write("{:.4f} ({:.4f},{:.4f}) {:.4f}({:.4f},{:.4f})".format(r_noise_percent.mean(),r_noise_percent.min(),r_noise_percent.max(),
                                                                                 r_noise_percent_75percentile_arr.mean(),r_noise_percent_75percentile_arr.min(),r_noise_percent_75percentile_arr.max()))
            else:
                tqdm.write("{:.4f} ({:.4f},{:.4f}) {:.4f}({:.4f},{:.4f})".format(pred_r.mean(),pred_r.min(),pred_r.max(),
                                                                                 pred_r_75_percentile_arr.mean(),pred_r_75_percentile_arr.min(),pred_r_75_percentile_arr.max()))
                tqdm.write(f"Noise ceiling: Mean {noise_ceiling.mean()} (Min {noise_ceiling.min()}, Max {noise_ceiling.max()})")
            tqdm.write("Test set: Avg loss: {:.4f}, r_score: {:.4f}".format(test_loss,np.mean(pred_r)))
    if normalize_by_noise_ceiling:
        return pred_r, np.mean(r_noise_percent), np.mean(r_noise_percent_75percentile_arr)  # NOTE: no more np.mean(pred_r)
    else:
        return pred_r, np.mean(pred_r), np.mean(pred_r_75_percentile_arr)
