import numpy as np
from tqdm import tqdm
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import math
import matplotlib.pyplot as plt
from shallow_simclr_backbone import BackBone, streamNet, dualstreamNet
from collections import OrderedDict

class DNNActivations(nn.Module):
    def __init__(self,model_class=models.vgg16, layer=11, pretrained=True, ckpt_path=None):
        '''
        Args
            model_class: model class to generate activations from, e.g. - models.vgg16; model_3d.DPC_RNN
            layer: Layer number to generate activations, eg. - 11 (1st layer indeced at 1)
            pretrained - True/False flag to save activations from pretrained or untrained network
        '''
        super().__init__()
        if model_class == models.vgg16 or model_class == models.vgg19:
            self.model = model_class(pretrained=pretrained)
            self.model_layers = [module for module in self.model.modules() if type(module)!=nn.Sequential] # 0th item is whole network
            self.model_layers = self.model_layers[1:] 	# remove first element because it's the whole model anyways
            self.layer = layer-1
            self.outputs = []
        elif model_class == models.resnet18 or model_class == models.resnet34 or model_class == models.resnet50 or model_class == models.resnet101 or model_class == models.resnet152:
            if pretrained:
                self.model = model_class(weights='IMAGENET1K_V1')
            else:
                self.model = model_class()
            self.model_layers = [module for module in self.model.modules() if type(module)!=nn.Sequential] # 0th item is whole network
            self.model_layers = self.model_layers[1:] 	# remove first element because it's the whole model anyways
            self.layer = layer-1
            self.outputs = []
        elif model_class == models.alexnet:
            self.model = model_class(pretrained=pretrained)
            self.model_layers = [module for module in self.model.modules() if type(module)!=nn.Sequential] # 0th item is whole network
            self.model_layers = self.model_layers[1:] 	# remove first element because it's the whole model anyways
            self.layer = layer-1
            self.outputs = []
        elif model_class == 'streamNet':
            self.model = streamNet()
            # TODO: if pretrained...
            self.model_layers = [module for module in self.model.modules() if type(module)!=nn.Sequential]
            self.model_layers = self.model_layers[1:] 	# remove first element because it's the whole model anyways
            self.layer = layer-1  # layer is 1-indexed, i.e. 1=Conv2d, 2=BN, 3=ReLU,...
            self.outputs = []
        elif model_class == 'dualstreamNet':
            self.model = dualstreamNet()
            # TODO: if pretrained...
            self.model_layers = [module for module in self.model.modules() if type(module)!=nn.Sequential]
            self.model_layers = self.model_layers[1:] 	# remove first element because it's the whole model anyways
            self.layer = layer-1  # layer is 1-indexed, i.e. 1=Conv2d, 2=BN, 3=ReLU,...
            self.outputs = []
        else:
            assert type(model_class) == str, "model_class should be a str, eg. shallowConvfeatdualstream_4"
            assert 'proj' not in model_class, "model_class should not contain projector. Replace 'proj' with 'feat'."
            if pretrained:
                assert ckpt_path is not None, "Must provide ckpt_path for pretrained shallowConv networks"
            self.model = BackBone(name=model_class,
                                  dataset=None,  # shouldn't really matter for shallowConv
                                  projector_dim=None,
                                  hidden_dim=None
                                  )
            if pretrained:
                ckpt = torch.load(ckpt_path)['model']  # ordered dict
                ckpt = OrderedDict([(k[9:], v) if 'backbone.' in k else (k, v) for k, v in ckpt.items()])
                for (k,v) in ckpt.copy().items():
                    if 'proj' in k:
                        del ckpt[k]
                self.model.load_state_dict(ckpt)
                print("Loaded pretrained weights from {}".format(ckpt_path))
            self.model_layers = [module for module in self.model.modules() if type(module)!=nn.Sequential]
            self.model_layers = self.model_layers[1:] 	# remove first element because it's the whole model anyways
            # self.model_layers: 0=Conv2d, 1=BN, 2=ReLU, 3=Conv2d, 4=BN, 5=ReLU... regardless of streams
            self.layer = layer-1  # layer is 1-indexed, i.e. 1=Conv2d, 2=BN, 3=ReLU,...
            self.outputs = []


    def forward(self, x):
        self.outputs = []
        def get_activations(module,inp,out):
            self.outputs.append(out)
        h = self.model_layers[self.layer].register_forward_hook(get_activations)
        out = self.model(x)
        if len(self.outputs[0].size())==2:
            self.outputs[0] = self.outputs[0].unsqueeze(1).unsqueeze(2)
        return self.outputs[0]


class FactorizedReadOut(nn.Module):
    def __init__(self,inp_size,out_size,bias=True,normalize=True):
        '''
        Args
            inp_size: shape of input as c x w x h
            out_size: number of units in output layer
            bais: True/False flag to have extra bias parameter
            normalize: True/False flag to normalize the spatial weights

        '''
        super().__init__()
        self.inp_size = inp_size
        self.out_size = out_size
        self.normalize = normalize
        c,w,h = self.inp_size
        self.spatial = nn.Parameter(torch.Tensor(self.out_size,1,w,h))
        self.features = nn.Parameter(torch.Tensor(self.out_size,c,1,1))
        if bias:
            bias = nn.Parameter(torch.Tensor(self.out_size))
            self.register_parameter('bias',bias)
        else:
            self.register_parameter('bias',None)

        self.initialize()

    @property
    def normalized_spatial(self):
        if self.normalize:
            weight = self.spatial/(self.spatial.pow(2).sum(2,keepdim=True).sum(3,keepdim=True).sqrt().expand_as(self.spatial) + 1e-6)
            ### RT edit ###
            weight = torch.abs(weight)
        else:
            weight = self.spatial
        return weight

    @property
    def weight(self):
        c,w,h = self.inp_size
        weight = self.normalized_spatial.expand(self.out_size,c,w,h) * self.features.expand(self.out_size,c,w,h)
        weight = weight.view(self.out_size,-1)
        return weight

    def l1(self):
        c,w,h = self.inp_size
        ret = (self.normalized_spatial.view(self.out_size,-1).abs().sum(1,keepdim=True)
               * self.features.view(self.out_size,-1).abs().sum(1)).sum()
        ret = ret/(self.out_size*c*w*h)
        return ret

    def initialize(self, init_noise=1e-3):
        self.spatial.data.normal_(0,init_noise)
        self.features.data.normal_(0,init_noise)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self,x):
        assert len(x.size())==4, "x should be a 4D tensor, current shape: {}".format(str(x.size()))
        N = x.size(0)
        y = x.view(N,-1) @ self.weight.t()
        if self.bias is not None:
            y = y+self.bias.expand_as(y)
        return y

    def __repr__(self):
        return ('normalized' if self.normalize else '') + self.__class__.__name__ + ' (' + '{} x {} x {}'.format(
            *self.inp_size) + ' -> ' + str(self.out_size) + ')'

class DNN_readout_combined(nn.Module):
    def __init__(self,DNN,readout):
        super().__init__()
        assert type(DNN).__name__ == 'DNNActivations' and type(readout).__name__ == 'ReadOut', "DNNActivations and ReadOut objects are needed respectively"
        self.DNN_model = DNN
        self.readout = readout

    def forward(self,X):
        out = self.DNN_model(X)
        out = self.readout(out)
        return out

class ReadOut(nn.Module):
    def __init__(self,inp_size=None,out_size=None,readout_model=None):
        super(ReadOut, self).__init__()
        assert (inp_size is not None and out_size is not None) or readout_model is not None, "At least input & output size or some (Factorized) readout model is needed"
        # breakpoint()
        if readout_model is not None:
            c,w,h = readout_model.inp_size
            inp_size = c*w*h
            out_size = readout_model.out_size
            self.fc = nn.Linear(inp_size,out_size,bias=True if readout_model.bias is not None else False)
            if readout_model.bias is not None:
                self.fc.bias.data = readout_model.bias.data
            self.fc.weight.data = readout_model.weight.data
        else:
            self.fc = nn.Linear(inp_size,out_size)

    def forward(self, x):
        N = x.size(0)
        x = x.view(N,-1)
        x = self.fc(x)
        return x

class LinearRegression(torch.nn.Module):

    def __init__(self, inp_size=None, out_size=None):
        super(LinearRegression, self).__init__()
        if len(inp_size) > 1:  # if inp_size is like (135,135,3)
            inp_size = inp_size[0]*inp_size[1]*inp_size[2]
        self.linear = torch.nn.Linear(inp_size, out_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        predict_y = self.linear(x)
        return predict_y

class ResNetSimCLR(nn.Module):
    '''ResNet backbone used for SimCNE'''

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def forward(self, x):
        return self.backbone(x)


class SimCNE(object):

    def __init__(self, device='cpu', n_views=2, epochs=1, temperature=1, d0=None, savepath=None, **kwargs):
        self.device=device
        self.n_views = n_views
        self.epochs = epochs
        self.temperature = temperature
        self.model = kwargs['model'].to(self.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.d0 = d0
        self.savepath = savepath

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(features.shape[0]//self.n_views) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # Train on lowrank features to prevent dimensionality collapse
        features_lowrank = features[:,:self.d0]

        # SimCLR loss
        #features_lowrank = F.normalize(features_lowrank, dim=1)
        #similarity_matrix = torch.matmul(features_lowrank, features_lowrank.T)

        # SimCNE loss
        similarity_matrix = torch.log( 1/(1+torch.cdist(features_lowrank.unsqueeze(0), features_lowrank.unsqueeze(0)).squeeze().pow(2)) +1e-9)


        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # Temperature applied to similarity, variable only becomes logits when passed to CrossEntropyLoss
        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader):
        n_iter = 0
        epoch_loss = []
        for epoch_counter in tqdm(range(self.epochs)):
            l = []
            for images, _ in train_loader:
                images = torch.cat(images, dim=0)

                images = images.to(self.device)

                features = self.model(images)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                loss.backward()
                l.append(loss.item())

                self.optimizer.step()

                n_iter += 1

            epoch_loss.append(np.mean(l))
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            # Save checkpoints
            if epoch_counter%100 == 0:
                if self.savepath is not None:
                    torch.save(self.model.state_dict(), os.path.join(self.savepath, f'SimCLR_renet18_temp{self.temperature}_epoch{epoch_counter}.pth'))
                
        return epoch_loss

