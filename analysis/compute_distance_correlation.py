'''
compute_distance_correlation.py

Infer functional hierarchy via distance correlation of RF aligned activations.
Perform both shuffling and permutation test. In each case, shuffle/permute along both
neuron and stimulus axes.
'''

from utils import *
import dcor

def subsample(activations1, activations2, nneuron=200, nstim=1000):
    '''Given two activation matrices (stim x neuron), subsample both dimensions'''
    # Subsample stimuli and use same sample for both matrices
    idx_stim = np.random.choice(np.arange(activations1.shape[0]), replace=False, size=nstim)

    # Subsample neurones separately for each region
    idx_neuron1 = np.random.choice(np.arange(activations1.shape[1]), replace=False, size=nneuron)
    idx_neuron2 = np.random.choice(np.arange(activations2.shape[1]), replace=False, size=nneuron)

    return idx_stim, idx_neuron1, idx_neuron2

def shuffle_data(data):
    '''Shuffle each row'''
    data_copy = data.copy()
    for i in data_copy: np.random.shuffle(i)
    return data_copy

def distance_correlation(activations, nneuron=200, nstim=1000, nrepeat=100):
    '''Run distance correlation analysis'''
    cc = np.zeros([len(activations), len(activations), nrepeat])
    pbar = tqdm(total= int((len(activations)+1)*len(activations)/2)*nrepeat )
    for i in range(len(activations)):
        for j in range(i, len(activations)):
            for k in range(nrepeat):
                # Get permutations
                idx_stim, idx_neuron1, idx_neuron2 = subsample(activations[i], activations[j], nneuron, nstim)
                cc[i,j,k] = dcor.distance_correlation(activations[i][idx_stim][:,idx_neuron1], activations[j][idx_stim][:,idx_neuron2])
                cc[j,i,k] = cc[i,j,k]
                pbar.update(1)
    return cc

def partial_distance_correlation(activations, nneuron, nstim, nrepeat):
    '''Run partical distance correlation analysis'''
    cc = np.zeros([len(activations), len(activations), nrepeat])
    pbar = tqdm(total= int((len(activations)+1)*len(activations)/2)*nrepeat )
    for i in range(len(activations)):
        for j in range(i, len(activations)):
            for k in range(nrepeat):
                # Create matrix of all other regions
                z = np.hstack([activations[x] for x in range(len(activations)) if x not in (i,j)])

                # Get permutations
                idx_stim, idx_neuron1, idx_neuron2 = subsample(activations[i], activations[j], nneuron, nstim)
                cc[i,j,k] = dcor.partial_distance_correlation(activations[i][idx_stim][:,idx_neuron1], activations[j][idx_stim][:,idx_neuron2], z[idx_stim])
                cc[j,i,k] = cc[i,j,k]
                pbar.update(1)
    
    return cc

def partial_distance_correlation_stratified(activations, nneuron, nstim, nrepeat, shuffle=False):
    '''Run partical distance correlation analysis but with each region conditioned individually'''
    cc = np.zeros([len(activations), len(activations), len(activations), nrepeat])
    pbar = tqdm(total= int(len(activations)*(len(activations)-1)/2*nrepeat*(len(activations)-2)) )
    for i in range(len(activations)):
        for j in range(i, len(activations)):
            if i == j: continue
            for r in range(len(activations)):
                if r in (i,j): continue
                # Create matrix of all other regions
                z = activations[r]
                if shuffle: z = shuffle_data(z)
                for k in range(nrepeat):
                    # Get permutations
                    idx_stim, idx_neuron1, idx_neuron2 = subsample(activations[i], activations[j], nneuron, nstim)
                    cc[i,j,r,k] = dcor.partial_distance_correlation(activations[i][idx_stim][:,idx_neuron1], activations[j][idx_stim][:,idx_neuron2], z[idx_stim])
                    cc[j,i,r,k] = cc[i,j,r,k]
                    pbar.update(1)
    
    return cc


if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'D:\Data\DeepMouse\Results_raw\representational_similarity'
    fp_act = r'D:\Data\DeepMouse\Results_raw\activations\RF_aligned'

    nneuron=200
    nstim=1000 
    nrepeat=100

    # Load activations
    activations = [StandardScaler().fit_transform(np.load(os.path.join(
                   fp_act, f"activations_{region}.npy"))) for region in regions]
    
    dcor_all = distance_correlation(activations, nneuron, nstim, nrepeat)
    np.save(os.path.join(savepath, f'dcor_nneuron-{nneuron}_nstim-{nstim}.npy'), dcor_all)

    pdcor_all = partial_distance_correlation(activations, nneuron, nstim, nrepeat)
    np.save(os.path.join(savepath, f'pdcor_nneuron-{nneuron}_nstim-{nstim}.npy'), pdcor_all)

    # pdcor_all_strat = partial_distance_correlation_stratified(activations, nneuron, nstim, nrepeat)
    # np.save(os.path.join(savepath, f'pdcor_strat_nneuron-{nneuron}_nstim-{nstim}.npy'), pdcor_all_strat)

    # pdcor_all_strat0 = partial_distance_correlation_stratified(activations, nneuron, nstim, nrepeat, shuffle=True)
    # np.save(os.path.join(savepath, f'pdcor_strat00_nneuron-{nneuron}_nstim-{nstim}.npy'), pdcor_all_strat0)
    