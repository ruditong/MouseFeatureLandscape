# TODO: load data; pool neurons; noise ceiling; select good neurons
'''
Assume data: zscore=True / corr_filter=False / pretrial_frame=5 / mode=F. Unpooled data in area/animal/session
'''
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score
import itertools
from scipy import stats
import matplotlib
# matplotlib.use('Agg')


def neuronal_response(A,indices=np.arange(8,12)):
    if indices is None:
        indices = list(range(A.shape[-1]))
    return np.sum(A[...,indices],axis=-1)


def identify_missing_stimuli(data_folder, area, total_rep_stims, total_unique_stims, incomplete_stims, verbose=False):
    '''
    Find stimuli that are missing from one or more sessions and add them to "incomplete_stims"
    :param data_folder: path to the data directory (for all regions)
    :param area: str
    :param total_rep_stims:  dict, keys = session name, value = 100-item list
    :param total_unique_stims: dict, key = session name, value = 2400-item list
    :param incomplete_stims: dict, keys = 'rep' or 'unique', value = list of images that lacks response
    :return:
    '''
    data_folder = os.path.join(data_folder,area)
    animal_list = glob.glob(os.path.join(data_folder,'*'))
    animal_list = [os.path.basename(a) for a in animal_list]
    animal_list.sort()
    if 'noise_ceiling_dict.npy' in animal_list:
        animal_list.remove('noise_ceiling_dict.npy')
    for ani_idx,animal in enumerate(animal_list):
        animal_data_folder = os.path.join(data_folder,animal)
        files = glob.glob(os.path.join(animal_data_folder,'*',''))
        files = [os.path.basename(os.path.dirname(f)) for f in files]
        files.sort()
        for fidx,f in enumerate(['part1', 'part2', 'part3', 'part4']):
            if f not in files:
                continue
            rep_data = np.load(os.path.join(animal_data_folder,f,'rep_stims.npy'), allow_pickle=True).item()
            unique_data = np.load(os.path.join(animal_data_folder,f,'unique_stims.npy'), allow_pickle=True).item()
            rep_stims = list(rep_data.keys())
            unique_stims = list(unique_data.keys())
            # if len(rep_stims) < len(total_rep_stims[f]):
            for stim in total_rep_stims[f]:
                if stim not in rep_stims:
                    if verbose:
                        print("Missing image {}".format(stim))
                    incomplete_stims['rep'].append(stim)
                    # total_rep_stims[f].remove(stim)
            # if len(unique_stims) < len(total_unique_stims[f]):
            for stim in total_unique_stims[f]:
                if stim not in unique_stims:
                    if verbose:
                        print("Missing image {}".format(stim))
                    incomplete_stims['unique'].append(stim)
                    # total_unique_stims[f].remove(stim)
    return incomplete_stims


def create_stimuli_bank(data_folder, area="V1", verbose=False):
    '''
    Create a bank of stimuli for a given region
    :param data_folder: path to the data directory (for all regions)
    :param area: str
    :return: total_rep_stims: dict, keys = session name, value = 100-item list
    :return: total_unique_stims: dict, key = session name, value = 2400-item list
    :return: incomplete_stims: dict, keys = 'rep' or 'unique', value = list of images that lacks response
    '''
    data_folder = os.path.join(data_folder,area)
    animal_list = glob.glob(os.path.join(data_folder,'*'))
    animal_list = [os.path.basename(a) for a in animal_list]
    animal_list.sort()
    if 'noise_ceiling_dict.npy' in animal_list:
        animal_list.remove('noise_ceiling_dict.npy')
    total_rep_stims = {}  # key = session name, value = 100-item list
    total_unique_stims = {}  # key = session name, value = 2400-item list
    incomplete_stims = {'rep': [], 'unique': []}  # stimulus to which we don't have responses from all neurons
    for ani_idx,animal in enumerate(animal_list):
        animal_data_folder = os.path.join(data_folder,animal)
        files = glob.glob(os.path.join(animal_data_folder,'*',''))
        files = [os.path.basename(os.path.dirname(f)) for f in files]
        files.sort()
        if verbose:
            print(f'Creating stimulus bank for animal {animal}')
        for fidx,f in enumerate(['part1', 'part2', 'part3', 'part4']):
            if f not in files:
                continue

            if f not in list(total_rep_stims.keys()):  # a new part
                total_rep_stims[f] = []
            if f not in list(total_unique_stims.keys()):
                total_unique_stims[f] = []

            rep_data = np.load(os.path.join(animal_data_folder,f,'rep_stims.npy'), allow_pickle=True).item()
            unique_data = np.load(os.path.join(animal_data_folder,f,'unique_stims.npy'), allow_pickle=True).item()
            rep_stims = list(rep_data.keys())
            unique_stims = list(unique_data.keys())
            if verbose:
                print(f'{f}, {len(rep_stims)} rep stims, {len(unique_stims)} unique stims')

            for stim in rep_stims:
                # if stim not in total_rep_stims[f] and stim not in incomplete_stims['rep']:
                try:
                    tmp = np.stack(rep_data[stim])
                    # assert tmp.shape[-1]==25
                    assert len(tmp)==10
                except:
                    if verbose:
                        print("Skipping image {}".format(stim))
                    incomplete_stims['rep'].append(stim)
                    continue
                if stim not in total_rep_stims[f]:
                    total_rep_stims[f].append(stim)  # only append name (str), not image

            for stim in unique_stims:
                # if stim not in total_unique_stims[f] and stim not in incomplete_stims['unique']:
                try:
                    tmp = np.stack(unique_data[stim])
                    # assert tmp.shape[-1]==25
                except:
                    if verbose:
                        print("Skipping image {}".format(stim))
                    incomplete_stims['unique'].append(stim)
                    continue
                if stim not in total_unique_stims[f]:
                    total_unique_stims[f].append(stim)  # only append name (str), not image

    return total_rep_stims, total_unique_stims, incomplete_stims


def chain_unique_resp(data_folder, area, total_rep_stims, total_unique_stims, incomplete_stims):
    data_folder = os.path.join(data_folder,area)  # TODO: change data folder
    animal_list = glob.glob(os.path.join(data_folder,'*'))
    animal_list = [os.path.basename(a) for a in animal_list]
    animal_list.sort()
    if 'noise_ceiling_dict.npy' in animal_list:
        animal_list.remove('noise_ceiling_dict.npy')
    chained_rep_resp = {}  # key = session name, value = 100-item list
    chained_unique_resp = {}  # key = session name, value = 2400-item list

    for ani_idx,animal in enumerate(animal_list):
        animal_data_folder = os.path.join(data_folder,animal)
        files = glob.glob(os.path.join(animal_data_folder,'*',''))
        files = [os.path.basename(os.path.dirname(f)) for f in files]
        files.sort()

        for fidx, f in enumerate(['part1', 'part2', 'part3', 'part4']):
            if f not in files:
                continue

            rep_data = np.load(os.path.join(animal_data_folder,f,'rep_stims.npy'), allow_pickle=True).item()
            unique_data = np.load(os.path.join(animal_data_folder,f,'unique_stims.npy'), allow_pickle=True).item()
            Ca_traces_arr = []
            for rep_stim in total_rep_stims[f]:
                if rep_stim not in incomplete_stims['rep']:
                    Ca_traces_arr.append(np.stack(rep_data[rep_stim]))
            Ca_traces_arr = np.array(Ca_traces_arr)  # n_stims x n_trials x n_neurons x 25
            Ca_traces_arr = np.transpose(Ca_traces_arr, axes=[2, 0, 1, 3])  # n_neurons x n_stims x n_trials x 25
            n_neurons = Ca_traces_arr.shape[0]
            n_stims = Ca_traces_arr.shape[1]
            n_trials = Ca_traces_arr.shape[2]
            n_frames = Ca_traces_arr.shape[-1]
            chained_rep_resp[f] = np.reshape(Ca_traces_arr, (n_neurons, n_stims*n_trials*n_frames))

            Ca_traces_arr = []
            for unique_stim in total_unique_stims[f]:  # for each image
                if unique_stim not in incomplete_stims['unique']:
                    Ca_traces_arr.append(np.stack(unique_data[unique_stim]))
            Ca_traces_arr = np.array(Ca_traces_arr).squeeze()  # n_stims x n_neurons x 25
            Ca_traces_arr = np.transpose(Ca_traces_arr, axes=[1, 0, 2])  # n_neurons x n_stims x 25
            n_neurons = Ca_traces_arr.shape[0]
            n_stims = Ca_traces_arr.shape[1]
            n_frames = Ca_traces_arr.shape[-1]
            chained_unique_resp[f] = np.reshape(Ca_traces_arr, (n_neurons, n_stims*n_frames))

    return chained_rep_resp, chained_unique_resp


def load_pool_neurons(data_folder, area, total_rep_stims, total_unique_stims, incomplete_stims, indices=np.arange(8,12)):
    """
    Load neuronal response data into rep_stims = (n_stims, n_trials, n_neurons) matrix or unique_stims = (n_stims, n_neurons) matrix
    that is pooled across animals for a given area.
    :param data_folder: None or str
    :param area: str
    :param total_rep_stims, total_unique_stims, incomplete_stims: from previous functions
    :param indices: default = np.arange(8,12)
    :return: rep_resp, unique_resp
    """
    data_folder = os.path.join(data_folder,area)  # TODO: change data folder

    animal_list = glob.glob(os.path.join(data_folder,'*'))
    animal_list = [os.path.basename(a) for a in animal_list]
    animal_list.sort()
    if 'noise_ceiling_dict.npy' in animal_list:
        animal_list.remove('noise_ceiling_dict.npy')
    total_rep_resp = {}  # key = session name, value = 100-item list
    total_unique_resp = {}  # key = session name, value = 2400-item list

    for ani_idx,animal in enumerate(animal_list):
        animal_data_folder = os.path.join(data_folder,animal)
        files = glob.glob(os.path.join(animal_data_folder,'*',''))
        files = [os.path.basename(os.path.dirname(f)) for f in files]
        files.sort()

        for fidx, f in enumerate(['part1', 'part2', 'part3', 'part4']):
            if f not in files:
                continue

            if f not in list(total_rep_resp.keys()):  # a new part
                total_rep_resp[f] = []  # list of matrices,to be stacked
            if f not in list(total_unique_resp.keys()):
                total_unique_resp[f] = []

            rep_data = np.load(os.path.join(animal_data_folder,f,'rep_stims.npy'), allow_pickle=True).item()
            unique_data = np.load(os.path.join(animal_data_folder,f,'unique_stims.npy'), allow_pickle=True).item()

            Ca_traces_arr = []
            for rep_stim in total_rep_stims[f]:
                if rep_stim not in incomplete_stims['rep']:
                    Ca_traces_arr.append(np.stack(rep_data[rep_stim]))
            Ca_traces_arr = np.array(Ca_traces_arr)
            total_rep_resp[f].append(neuronal_response(Ca_traces_arr,indices=indices))  # n_stims x n_trials x n_neurons

            Ca_traces_arr = []
            for unique_stim in total_unique_stims[f]:  # for each image
                if unique_stim not in incomplete_stims['unique']:
                    Ca_traces_arr.append(np.stack(unique_data[unique_stim]))
            Ca_traces_arr = np.array(Ca_traces_arr)
            total_unique_resp[f].append(np.squeeze(neuronal_response(Ca_traces_arr,indices=indices)))  # n_stims x n_neurons
    assert len(total_rep_resp) == len(total_unique_resp), "# of sessions mismatch between rep and unique data"
    for f in list(total_rep_resp.keys()):
        total_rep_resp[f] = np.concatenate(total_rep_resp[f], axis=-1)  # concatenate across neurons from different animals
        total_unique_resp[f] = np.concatenate(total_unique_resp[f], axis=-1)
    return total_rep_resp, total_unique_resp


def calculate_pearson_r(data_folder, area, total_rep_resp, threshold, n_half_split, plot=True, save=False):
    """
    For each neurons, randomly half-splitting the 10 trials for n_half_split times and calculate the average pearson r between the means of the two halves.
    :param total_rep_resp: dict
    :param threshold: float, pearson r2 threshold for selecting good neurons
    :return: total_pearson_r: dict, keys = sessions, values = pearson r for all neurons (not just good neurons)
    :return noise_ceiling: dict, keys = sessions, value = dict with keys = ['good_neurons_idx', 'noise_ceiling_arr']
    """
    total_pearson_r = {}
    noise_ceiling_dict = {}
    for fidx, f in enumerate(list(total_rep_resp.keys())):
        A = total_rep_resp[f]
        r_array_all_splits = []
        noise_ceiling_dict[f] = {}
        for i_half_split in range(n_half_split):
            r_array_one_split = []
            shuff_idx = np.random.permutation(A.shape[1])
            test_idx = shuff_idx[:int(A.shape[1]/2)]  # first 5 random idx
            train_idx = shuff_idx[int(A.shape[1]/2):]  # last 5 random idx
            gt_neuronal_resp = np.mean(A[:,test_idx,:],axis=1)  # taking the mean of the two halves
            pred_neuronal_resp = np.mean(A[:,train_idx,:],axis=1)
            # r2_array_all_splits.append(r2_score(gt_neuronal_resp,pred_neuronal_resp,multioutput='raw_values'))  # calculate the R2 score between the two halves
            for i_neuron in range(A.shape[-1]):
                pearsonr, pval = stats.pearsonr(gt_neuronal_resp[:, i_neuron], pred_neuronal_resp[:, i_neuron])
                r_array_one_split.append(pearsonr)
            r_array_one_split = np.array(r_array_one_split)
            r_array_all_splits.append(r_array_one_split)  # calculate the pearson correlation between the two halves
        total_pearson_r[f] = np.mean(np.array(r_array_all_splits), axis=0)  # average across n splits
        noise_ceiling_dict[f]['good_neurons_idx'] = np.where(total_pearson_r[f] >= threshold)[0]
        noise_ceiling_dict[f]['noise_ceiling_arr'] = total_pearson_r[f][np.where(total_pearson_r[f] >= threshold)[0]]
    if plot:
        plt.figure()
        total_good_neuron = 0
        total_neuron = 0
        for f in list(noise_ceiling_dict.keys()):  # for each part
            plt.hist(total_pearson_r[f], bins=100, label=f, alpha=0.5)
            total_good_neuron += len(noise_ceiling_dict[f]['good_neurons_idx'])
            total_neuron += total_rep_resp[f].shape[-1]
        plt.axvline(x=threshold,color='k',linestyle='--')
        plt.xlabel('half-split pearson r of each neuron',size=10)
        plt.ylabel('Number of neurons',size=10)
        plt.title('Area {}, {}/{} neurons'.format(area, total_good_neuron, total_neuron),fontsize=16)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend()
        # plt.show()
        plt.savefig(f"experiments/{area}_noise_ceiling_distribution.svg", format="svg")
    if save:
        np.save(os.path.join(data_folder, area, "noise_ceiling_dict.npy"), noise_ceiling_dict)
    return total_pearson_r, noise_ceiling_dict


def calculate_pearson_r_v2(response_data_folder, pearsonr_threshold=0.6, n_half_split=50, plot=True, save=True):
    """
    calculate noise ceiling
    :param rep_stim_and_resp: dict. keys = images. values = response matrix (n_trials x n_neurons)
    :param data_folder: where noise_ceiling_dict will be saved
    :param pearsonr_threshold: default = .6
    :param n_half_split: default = 50
    :return noise_ceiling: noise ceiling of *good* neurons.
    :return good_neurons_idx: indices of neurons whose noise ceiling > pearsonr_threshold
    """
    r_array_all_splits = []
    noise_ceiling_dict = {}
    rep_resp = []
    rep_stim_and_resp = np.load(os.path.join(response_data_folder, "rep_stims.npy"), allow_pickle=True).item()
    for rep_stim in list(rep_stim_and_resp.keys()):
        rep_resp.append(rep_stim_and_resp[rep_stim])
    rep_resp = np.asarray(rep_resp)  # n_stims x n_trials x n_neurons
    for i_half_split in range(n_half_split):
        r_array_one_split = []
        shuff_idx = np.random.permutation(rep_resp.shape[1])
        test_idx = shuff_idx[:int(rep_resp.shape[1]/2)]  # first 5 random idx
        train_idx = shuff_idx[int(rep_resp.shape[1]/2):]  # last 5 random idx
        gt_neuronal_resp = np.mean(rep_resp[:,test_idx,:],axis=1)  # taking the mean of the two halves
        pred_neuronal_resp = np.mean(rep_resp[:,train_idx,:],axis=1)
        # r2_array_all_splits.append(r2_score(gt_neuronal_resp,pred_neuronal_resp,multioutput='raw_values'))  # calculate the R2 score between the two halves
        for i_neuron in range(rep_resp.shape[-1]):
            pearsonr, pval = stats.pearsonr(gt_neuronal_resp[:, i_neuron], pred_neuronal_resp[:, i_neuron])
            ############################## RT EDIT ##############################
            r_array_one_split.append(pearsonr)
            ############################## RT EDIT ##############################
        r_array_one_split = np.array(r_array_one_split)
        r_array_all_splits.append(r_array_one_split)  # calculate the pearson correlation between the two halves
    total_pearson_r = np.mean(np.array(r_array_all_splits), axis=0)  # average across n splits
    noise_ceiling_dict['good_neurons_idx'] = np.where(total_pearson_r >= pearsonr_threshold)[0]
    noise_ceiling_dict['noise_ceiling_arr'] = total_pearson_r[np.where(total_pearson_r >= pearsonr_threshold)[0]]
    if plot:
        plt.figure()
        plt.hist(total_pearson_r, bins=100, alpha=0.5)
        plt.axvline(x=pearsonr_threshold,color='k',linestyle='--')
        plt.xlabel('half-split pearson r of each neuron',size=10)
        plt.ylabel('Number of neurons',size=10)
        plt.title('{}/{} neurons'.format(len(noise_ceiling_dict['good_neurons_idx']), rep_resp.shape[-1]),fontsize=16)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(response_data_folder, "noise_ceiling_distribution.svg"), format="svg")
    if save:
        np.save(os.path.join(response_data_folder, "noise_ceiling_dict.npy"), noise_ceiling_dict)



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Factorized readout model to predict neuronal response")
    parser.add_argument("--response_data_folder",type=str,default='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data/pre_validation',
                        help='Path to neuronal responses, rep_stim_and_resp and unique_stim_and_resp')
    args = parser.parse_args()
    argsdict = args.__dict__
    response_data_folder = argsdict['response_data_folder']
    pearsonr_threshold = -1  # for choosing good neurons with pearson r
    n_half_split = 50  # for calculating pearson r
    calculate_pearson_r_v2(response_data_folder=response_data_folder, pearsonr_threshold=pearsonr_threshold, n_half_split=n_half_split, plot=True, save=True)

    # import argparse
    # parser = argparse.ArgumentParser(description="Factorized readout model to predict neuronal response")
    # parser.add_argument("--data_folder",type=str,default='/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data',
    #                 help='Path to data files')
    # parser.add_argument("--area", type=str, default='V1', help='Visual cortex area to be used')
    # args = parser.parse_args()
    # argsdict = args.__dict__
    # data_folder = argsdict['data_folder']
    # area = argsdict['area']
    # # data_folder = '/home/mila/l/lindongy/linclab_folder/linclab_users/MouseNeuronPredict_data'
    # # area = "dTeA"
    # pearsonr_threshold = 0.6  # for choosing good neurons with pearson r
    # n_half_split = 50  # for calculating pearson r
    # # indices = np.arange(8,12)
    # indices = np.arange(0,10)
    # total_rep_stims, total_unique_stims, incomplete_stims = create_stimuli_bank(data_folder, area, verbose=True)
    # incomplete_stims = identify_missing_stimuli(data_folder, area, total_rep_stims, total_unique_stims, incomplete_stims, verbose=True)
    # #chained_rep_resp, chained_unique_resp = chain_unique_resp(data_folder, area, total_rep_stims, total_unique_stims, incomplete_stims)
    # #np.save("/Users/dongyanlin/Desktop/V1_rep_part3.npy", chained_rep_resp['part3'])
    # #np.save("/Users/dongyanlin/Desktop/V1_unique_part3.npy", chained_unique_resp['part3'])
    # total_rep_resp, total_unique_resp = load_pool_neurons(data_folder, area, total_rep_stims, total_unique_stims, incomplete_stims, indices)
    # total_pearson_r, noise_ceiling_dict = calculate_pearson_r(data_folder, area, total_rep_resp, pearsonr_threshold, n_half_split, plot=True, save=True)
