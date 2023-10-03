'''
combine_data.py

Combines data from each brain region into a common file for DNN training.
'''

import os, glob
import numpy as np
import matplotlib.pyplot as pl

def load_data(fp):
    '''Load in data and save in dictionary'''
    data = {'unique_stims'      : np.load(glob.glob(os.path.join(fp, '*unique_stims.npy'))[0], allow_pickle=1).item(),
            'unique_stims_raw'  : np.load(glob.glob(os.path.join(fp, '*unique_stims_raw.npy'))[0], allow_pickle=1).item(),
            'rep_stims'         : np.load(glob.glob(os.path.join(fp, '*rep_stims.npy'))[0], allow_pickle=1).item(),
            'rep_stims_raw'         : np.load(glob.glob(os.path.join(fp, '*rep_stims_raw.npy'))[0], allow_pickle=1).item(),
            'noise_ceiling'     : np.load(glob.glob(os.path.join(fp, '*noise_ceiling.npy'))[0]),
            'unique_s0'         : np.load(glob.glob(os.path.join(fp, '*unique_s0.npy'))[0]),
            }
    return data

def load_region(fp, region='V1'):
    '''Load in all the data for a given region and save in dictionary'''
    # Get all folderpaths
    dirs = glob.glob(os.path.join(fp, f'*{region}*'))
    data = [load_data(f) for f in dirs]
    return data

def combine_data(data, typ='unique_stims', r2_thresh=0.6):
    '''Given a list of data dictionaries and preprocessing type, combine them into one dictionary.
       Data will be filtered by noise_ceiling set by r2_thresh.
       To account for missing data, add np.nan's
    '''
    # First figure out how big the final matrix will be
    masks = [d['noise_ceiling'] > r2_thresh for d in data]
    total_neurons = np.array([m.sum() for m in masks])
    print(f'Total number of neurons: {total_neurons.sum()} ({total_neurons})')

    # Create a histogram of all noise ceiling
    ncs = np.concatenate([d['noise_ceiling'] for d in data])
    fig, ax = pl.subplots()
    ax.hist(ncs, bins=np.linspace(-0.2, 1, 44), color='k')
    ax.axvline(r2_thresh, ls='--', color=[.5]*3)
    ax.set_title(f"{total_neurons.sum()}/{np.array([m.shape[0] for m in masks]).sum()} in {len(masks)} recordings")
    ax.set_xlabel('Spearman Brown rho')

    # Find the union of all keys
    allkeys = set().union(*[d[typ].keys() for d in data])
    print(f'Found {len(allkeys)} unique keys')

    # Get the number of trials
    trials = data[0][typ][next(iter(data[0][typ].keys()))].shape[0]

    # Now loop over each key and generate the combined matrix, filling any missing data with np.nans
    # The matrices have shape (trial, neuron)
    combined_data = {}
    for key in allkeys:
        matrix = np.zeros((trials, total_neurons.sum())) 
        for i, d in enumerate(data):
            try:
                m = d[typ][key][:,masks[i]]
                matrix[:,total_neurons[:i].sum() : total_neurons[:i].sum()+total_neurons[i]] = m
            except KeyError:
                continue
        combined_data[key] = matrix

    return combined_data, fig

if __name__ == '__main__':
    regions = ['V1', 'LM', 'AL', 'LI', 'POR', 'RL']
    #regions = ['LM', 'LI', 'AL']
    fp = r'D:\Data\DeepMouse\Processed'
    savepath = r'D:\Data\DeepMouse\Processed\Combined_raw'

    typ = 'rep_stims_raw'
    r2_thresh = 0.5

    for region in regions:
        data = load_region(fp, region=region)
        combined_data, fig = combine_data(data, typ=typ, r2_thresh=r2_thresh)

        np.save(os.path.join(savepath, f'{region}_combined_{typ}.npy'), combined_data, allow_pickle=True)
        fig.savefig(os.path.join(savepath, f'{region}_noise_ceiling_combined.svg'))