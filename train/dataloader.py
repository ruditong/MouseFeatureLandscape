import torch
import torchvision
import sys
from calculate_noise_ceiling import *
# data_folder = '/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data'

class ImageNeuronDataset_v2(torch.utils.data.Dataset):  # TODO: keep or remove v2/v3 flags?
    def __init__(self,data_folder='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
                 area='V1',files=['part1','part2','part3','part4'],
                 mode='train',time_indices=np.arange(8,12),tr_portion=1.0):
        '''
        Args
            file: Session name to load images and neural data
        '''
        assert mode in ['train','test'], "mode should be `train` or `test`"
        self.mode = mode
        self.tr_portion = tr_portion
        self.areas = area
        # self.animal = animal
        self.total_rep_resp, self.total_unique_resp, self.total_rep_stims, self.total_unique_stims, self.incomplete_stims = self.pool_neurons(data_folder, area, time_indices)
        self.image_keys_dict, self.images,self.mean,self.std = self.load_data(data_folder,area,files)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((self.mean,self.mean,self.mean),(self.std,self.std,self.std))])
        self.all_neural_activities, self.neural_activities_mean, self.neural_activities_std, self.neural_activities_mask = self.load_neural_data(
            data_folder,area,files)
        if self.mode=='test':
            # We need mean as well as trial-wise neural activities
            self.trial_neural_activities, self.neural_activities = self.all_neural_activities
        else:
            # We only have mean neural activities (only 1 trial per image was shown to animal)
            self.neural_activities = self.all_neural_activities
            self.trial_neural_activities = None
        self.neural_activities = torch.Tensor(self.neural_activities) if self.neural_activities is not None else None
        self.trial_neural_activities = torch.Tensor(self.trial_neural_activities) if self.trial_neural_activities is not None else None
        self.neural_activities_mean = torch.Tensor(self.neural_activities_mean) if self.neural_activities_mean is not None else None
        self.neural_activities_std = torch.Tensor(self.neural_activities_std) if self.neural_activities_std is not None else None
        self.neural_activities_mask = torch.BoolTensor(self.neural_activities_mask) if self.neural_activities_mask is not None else None
        assert len(self.images)==len(self.neural_activities), ("Number of images (" + str(len(self.images))
                                                               +") and Neural activity ("+str(len(self.neural_activities))+") should match!!")
        self.len_dataset = len(self.images)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self,index):
        img = self.images[index]  # data
        neural_activity = self.neural_activities[index]  # target
        neural_activity_mask = self.neural_activities_mask[index]  # mask
        if self.transform:
            img = self.transform(img).type(torch.FloatTensor)
        return (img,neural_activity,neural_activity_mask)

    def pool_neurons(self, data_folder, area, indices):
        total_rep_stims, total_unique_stims, incomplete_stims = create_stimuli_bank(data_folder, area, verbose=False)
        incomplete_stims = identify_missing_stimuli(data_folder, area, total_rep_stims, total_unique_stims, incomplete_stims, verbose=False)
        total_rep_resp, total_unique_resp = load_pool_neurons(data_folder, area, total_rep_stims, total_unique_stims, incomplete_stims, indices)
        return total_rep_resp, total_unique_resp, total_rep_stims, total_unique_stims, incomplete_stims

    def load_data(self, data_folder, area, files):
        image_collection = np.load(os.path.join(data_folder,'DNN_images.npy'),allow_pickle=True)
        noise_ceiling_dict = np.load(os.path.join(data_folder,area,'noise_ceiling_dict.npy'),allow_pickle=True).item()
        all_images = []
        image_keys_dict = {}
        for file in files:
            if not file in self.total_rep_stims.keys(): # No data for this session
                continue
            if len(noise_ceiling_dict[file]['good_neurons_idx'])==0:  # No good neurons in this session
                continue
            img_keys = []
            img_arr = []
            if self.mode == 'test':
                for key in self.total_rep_stims[file]:
                    if key not in self.incomplete_stims['rep']:
                        img_arr.append(image_collection.item()[key])
                        img_keys.append(key)
            else:
                for key in self.total_unique_stims[file]:
                    if key not in self.incomplete_stims['unique']:
                        img_arr.append(image_collection.item()[key])
                        img_keys.append(key)
            img_arr = np.array(img_arr)
            all_images.append(img_arr)
            image_keys_dict[file] = img_keys
        del image_collection
        if len(all_images)>0:
            all_images = np.concatenate(all_images)
            imgs_mean = all_images.mean()
            imgs_std = all_images.std()
        else:
            all_images = np.array(all_images)
            imgs_mean = None
            imgs_std = None
        return image_keys_dict, all_images, imgs_mean, imgs_std

    def load_neural_data(self, data_folder, area, files):
        noise_ceiling_dict = np.load(os.path.join(data_folder,area,'noise_ceiling_dict.npy'),allow_pickle=True).item()
        good_neurons_dict = {}  # keys = [0,1,2,3], values = good_neurons_idx of that part
        neurons_fidx_list = []  # list (len=total_good_neurons) to indicate which part each good neuron is from

        # Load good neurons
        total_good_neurons = 0
        for fidx,file in enumerate(files):
            if file not in noise_ceiling_dict.keys():
                continue
            good_neurons = noise_ceiling_dict[file]['good_neurons_idx']
            total_good_neurons += len(good_neurons)
            good_neurons_dict[fidx] = good_neurons
            if len(good_neurons)==0:
                continue
            neurons_fidx_list.append((1+fidx)*np.ones(len(good_neurons),))
        if len(neurons_fidx_list)>0:  # at least one session
            neurons_fidx_list = np.concatenate(neurons_fidx_list)
        else:
            return np.array([]), None, None, np.array([])

        # Load masks
        # masks = []
        masks = {}  # keys = [0,1,2,3], value = list(len=total_good_neurons) of T (if this good neuron appears in this part) or F
        for fidx,file in enumerate(files):
            if fidx not in good_neurons_dict.keys() or len(good_neurons_dict[fidx])==0:
                continue
            masks[fidx] = neurons_fidx_list==(fidx+1)
        # masks = np.array(masks)

        # Load responses
        all_responses = []
        if self.mode=='test':
            all_trial_responses = []  # for calculating trial-weighted r2
        mean_neuronal_responses = []
        std_neuronal_responses = []
        all_masks = []
        for fidx,file in enumerate(files):
            # breakpoint()
            if fidx not in good_neurons_dict.keys() or len(good_neurons_dict[fidx])==0:
                continue
            good_neurons = good_neurons_dict[fidx]
            if self.mode=='test':
                Y1 = self.total_rep_resp[file][..., good_neurons]  # n_stims x n_trials x len(good_neurons)
                Y1_mean = np.mean(Y1,axis=1)  # average across trials  # n_stims x len(good_neurons)
                # Y1_mean = np.median(Y1,axis=1)  # median across trials  # TODO: mean or median?
            else:
                Y1 = self.total_unique_resp[file][..., good_neurons]
                Y1_mean = Y1  # average across trials but there's only one trial  # n_stims x len(good_neurons)
                # Y1_mean = np.mean(Y1,axis=1) # average across trials
                # Y1_mean = np.median(Y1,axis=1) # median across trials
            Y = np.zeros((Y1_mean.shape[0],total_good_neurons))  # n_stim x total_good_neurons
            Y[:,masks[fidx]] = Y1_mean
            all_responses.append(Y)
            if self.mode=='test':
                all_trial_Y = np.zeros((Y1.shape[0],Y1.shape[1],total_good_neurons))  # n_stims x n_trials x total_good_neurons
                all_trial_Y[:,:,masks[fidx]] = Y1
                all_trial_responses.append(all_trial_Y)
            mean_neuronal_responses.append(np.mean(Y1_mean,axis=0)) # mean across stimuli
            std_neuronal_responses.append(np.std(Y1_mean,axis=0)) # std across stimuli
            all_masks.append(np.array([masks[fidx]]*Y.shape[0]))  # create mask array
        all_responses = np.concatenate(all_responses)
        if self.mode=='test':
            all_trial_responses = np.concatenate(all_trial_responses)
        mean_neuronal_responses = np.concatenate(mean_neuronal_responses)
        std_neuronal_responses = np.concatenate(std_neuronal_responses)
        all_masks = np.concatenate(all_masks)
        assert len(all_responses)==len(all_masks), ("Number of responses (" + str(len(all_responses))
                                                    +") and masks ("+str(len(all_masks))+") should match!!")
        if self.mode=='test':
            return (all_trial_responses,all_responses), mean_neuronal_responses, std_neuronal_responses, all_masks
        return all_responses, mean_neuronal_responses, std_neuronal_responses, all_masks


class ImageNeuronDataset_v3(torch.utils.data.Dataset):  # TODO: keep or remove v2/v3 flags?
    def __init__(self, image_data_folder, response_data_folder, good_neurons_idx, mode='train',tr_portion=1.0):
        """
        Assume rep_stim_and_resp and unique_stim_and_resp are pre-processed data from **one area one animal one session**.
        Assume we're not pooling across sessions or animals.
        :param data_folder: path to the directory in which DNN_images.npy is saed
        :param rep_stim_and_resp: dict with keys = image, value = response matrix (n_trials x n_neurons). n_trials = 10
        :param unique_stim_and_resp: dict with keys = image, value = response matrix (n_trials x n_neurons). n_trials = 1
        :param good_neurons_idx: list with index of neurons whose noise ceiling surpasses the threshold
        :param mode: 'train' or 'test'. Default = 'train'
        :param tr_portion: Default = 1.0
        """
        assert mode in ['train','test'], "mode should be `train` or `test`"
        self.mode = mode
        self.tr_portion = tr_portion
        self.image_data_folder = image_data_folder
        self.rep_stim_and_resp = np.load(os.path.join(response_data_folder, "rep_stims.npy"), allow_pickle=True).item()
        self.unique_stim_and_resp = np.load(os.path.join(response_data_folder, "unique_stims.npy"), allow_pickle=True).item()
        self.good_neurons_idx = good_neurons_idx
        self.image_keys, self.images,self.mean,self.std = self.load_data()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((self.mean,self.mean,self.mean),(self.std,self.std,self.std))])
        self.all_neural_activities, self.neural_activities_mean, self.neural_activities_std, self.neural_activities_mask = self.load_neural_data()
        if self.mode=='test':
            # We need mean as well as trial-wise neural activities
            self.trial_neural_activities, self.neural_activities = self.all_neural_activities
        else:
            # We only have mean neural activities (only 1 trial per image was shown to animal)
            self.neural_activities = self.all_neural_activities
            self.trial_neural_activities = None
        self.neural_activities = torch.Tensor(self.neural_activities) if self.neural_activities is not None else None
        self.trial_neural_activities = torch.Tensor(self.trial_neural_activities) if self.trial_neural_activities is not None else None
        self.neural_activities_mean = torch.Tensor(self.neural_activities_mean) if self.neural_activities_mean is not None else None
        self.neural_activities_std = torch.Tensor(self.neural_activities_std) if self.neural_activities_std is not None else None
        self.neural_activities_mask = torch.BoolTensor(self.neural_activities_mask) if self.neural_activities_mask is not None else None
        assert len(self.images)==len(self.neural_activities), ("Number of images (" + str(len(self.images))
                                                               +") and Neural activity ("+str(len(self.neural_activities))+") should match!!")
        self.len_dataset = len(self.images)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self,index):
        img = self.images[index]  # data
        neural_activity = self.neural_activities[index]  # target
        neural_activity_mask = self.neural_activities_mask[index]  # mask
        if self.transform:
            img = self.transform(img).type(torch.FloatTensor)
        return (img,neural_activity,neural_activity_mask)

    def load_data(self):  # Load image data as input
        image_collection = np.load(os.path.join(self.image_data_folder,'DNN_images.npy'),allow_pickle=True)
        img_keys = []
        img_arr = []
        if self.mode == 'test':
            for key in list(self.rep_stim_and_resp.keys()):
                img_arr.append(image_collection.item()[key])
                img_keys.append(key)
        else:
            for key in list(self.unique_stim_and_resp.keys()):
                img_arr.append(image_collection.item()[key])
                img_keys.append(key)
        img_arr = np.array(img_arr)
        del image_collection
        if len(img_keys)>0:
            imgs_mean = img_arr.mean()
            imgs_std = img_arr.std()
        else:
            imgs_mean = None
            imgs_std = None
        return img_keys, img_arr, imgs_mean, imgs_std

    def load_neural_data(self):
        if len(self.good_neurons_idx)==0:
            return np.array([]), None, None, np.array([])

        # Load responses
        if self.mode=='test':
            rep_resp = []
            for rep_stim in list(self.rep_stim_and_resp.keys()):
                rep_resp.append(self.rep_stim_and_resp[rep_stim])
            rep_resp = np.asarray(rep_resp)
            Y1 = rep_resp[..., self.good_neurons_idx]  # n_stims x n_trials x len(good_neurons)
            Y1_mean = np.mean(Y1,axis=1)  # average across trials  # n_stims x len(good_neurons)
        else:
            unique_resp = []
            for unique_stim in list(self.unique_stim_and_resp.keys()):
                unique_resp.append(self.unique_stim_and_resp[unique_stim])
            unique_resp = np.squeeze(np.asarray(unique_resp))
            Y1 = unique_resp[..., self.good_neurons_idx]
            Y1_mean = Y1  # average across trials but there's only one trial  # n_stims x len(good_neurons)
        mean_neuronal_responses = np.mean(Y1_mean,axis=0)  # mean across stimuli
        std_neuronal_responses = np.std(Y1_mean,axis=0)  # std across stimuli
        neural_activities_mask = np.ones_like(Y1_mean)  # Assume data is from one single session and there's no need to pool sessions
        if self.mode=='test':
            return (Y1, Y1_mean), mean_neuronal_responses, std_neuronal_responses, neural_activities_mask
        return Y1_mean, mean_neuronal_responses, std_neuronal_responses, neural_activities_mask
    

class ImageNeuronDataset_v4(torch.utils.data.Dataset):  # TODO: keep or remove v2/v3 flags?
    def __init__(self, image_data_folder, response_data_folder, good_neurons_idx, mode='train',tr_portion=1.0):
        """
        Assume rep_stim_and_resp and unique_stim_and_resp are pre-processed data from **one area one animal one session**.
        Assume we're not pooling across sessions or animals.
        :param data_folder: path to the directory in which DNN_images.npy is saed
        :param rep_stim_and_resp: dict with keys = image, value = response matrix (n_trials x n_neurons). n_trials = 10
        :param unique_stim_and_resp: dict with keys = image, value = response matrix (n_trials x n_neurons). n_trials = 1
        :param good_neurons_idx: list with index of neurons whose noise ceiling surpasses the threshold
        :param mode: 'train' or 'test'. Default = 'train'
        :param tr_portion: Default = 1.0
        """
        assert mode in ['train','test'], "mode should be `train` or `test`"
        # Set attributes
        self.mode = mode
        self.tr_portion = tr_portion
        self.image_data_folder = image_data_folder

        # Load in data dictionaries (keys == stim, [trial, neuron])
        self.rep_stim_and_resp = np.load(os.path.join(response_data_folder, "rep_stims.npy"), allow_pickle=True).item()
        self.unique_stim_and_resp = np.load(os.path.join(response_data_folder, "unique_stims.npy"), allow_pickle=True).item()

        # Load in images and set up Standard Scaler
        self.image_keys, self.images,self.mean,self.std = self.load_data()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((self.mean,self.mean,self.mean),(self.std,self.std,self.std))])
        
        # Not sure what this is
        self.all_neural_activities, self.neural_activities_mean, self.neural_activities_std, self.neural_activities_mask = self.load_neural_data()

        if self.mode=='test':
            # We need mean as well as trial-wise neural activities
            self.trial_neural_activities, self.neural_activities = self.all_neural_activities
        else:
            # We only have mean neural activities (only 1 trial per image was shown to animal)
            self.neural_activities = self.all_neural_activities
            self.trial_neural_activities = None
        
        self.neural_activities = torch.Tensor(self.neural_activities) if self.neural_activities is not None else None
        self.trial_neural_activities = torch.Tensor(self.trial_neural_activities) if self.trial_neural_activities is not None else None
        self.neural_activities_mean = torch.Tensor(self.neural_activities_mean) if self.neural_activities_mean is not None else None
        self.neural_activities_std = torch.Tensor(self.neural_activities_std) if self.neural_activities_std is not None else None
        self.neural_activities_mask = torch.BoolTensor(self.neural_activities_mask) if self.neural_activities_mask is not None else None
        assert len(self.images)==len(self.neural_activities), ("Number of images (" + str(len(self.images))
                                                               +") and Neural activity ("+str(len(self.neural_activities))+") should match!!")
        self.len_dataset = len(self.images)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self,index):
        img = self.images[index]  # data
        neural_activity = self.neural_activities[index]  # target
        #neural_activity_mask = self.neural_activities_mask[index]  # mask
        neural_activity_mask = (neural_activity != 0)
        if self.transform:
            img = self.transform(img).type(torch.FloatTensor)
        return (img,neural_activity,neural_activity_mask)

    def load_data(self):  # Load image data as input
        image_collection = np.load(os.path.join(self.image_data_folder,'DNN_images.npy'),allow_pickle=True)
        img_keys = []
        img_arr = []
        if self.mode == 'test':
            for key in list(self.rep_stim_and_resp.keys()):
                img_arr.append(image_collection.item()[key])
                img_keys.append(key)
        else:
            for key in list(self.unique_stim_and_resp.keys()):
                img_arr.append(image_collection.item()[key])
                img_keys.append(key)
        img_arr = np.array(img_arr)
        del image_collection
        if len(img_keys)>0:
            imgs_mean = img_arr.mean()
            imgs_std = img_arr.std()
        else:
            imgs_mean = None
            imgs_std = None
        return img_keys, img_arr, imgs_mean, imgs_std

    def load_neural_data(self):
        # Load responses
        if self.mode=='test':
            rep_resp = np.array([self.rep_stim_and_resp[key] for key in self.rep_stim_and_resp.keys()])
            Y1_mean = np.mean(rep_resp,axis=1)  # average across trials  # n_stims x len(good_neurons)
        else:
            unique_resp = np.array([self.unique_stim_and_resp[key] for key in self.unique_stim_and_resp.keys()])
            Y1_mean = np.mean(unique_resp,axis=1)  # average across trials but there's only one trial  # n_stims x len(good_neurons)

        mean_neuronal_responses = np.mean(Y1_mean,axis=0)  # mean across stimuli
        std_neuronal_responses = np.std(Y1_mean,axis=0)  # std across stimuli
        neural_activities_mask = np.ones_like(Y1_mean)  # Assume data is from one single session and there's no need to pool sessions
        #print(Y1_mean.shape)
        if self.mode=='test':
            return (rep_resp, Y1_mean), mean_neuronal_responses, std_neuronal_responses, neural_activities_mask
        return Y1_mean, mean_neuronal_responses, std_neuronal_responses, neural_activities_mask