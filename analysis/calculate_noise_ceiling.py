'''
calculate_noise_ceiling.py

Calculate the noise ceiling of repeated presentation of stimuli via leave-one-out correlation.
'''

from utils import *
from scipy.stats import pearsonr

def leave_one_out_correlation(data):
    '''Given a data matrix trial x stimuli, calculate the leave-one-out pearson correlation'''
    data = data.T
    stim_mask = data.mean(axis=0) != 0

    cc = []
    for i in range(data.shape[0]):
        mask = np.ones(data.shape[0]).astype(bool)
        mask[i] = False
        cc.append(pearsonr(data[mask][:,stim_mask].mean(axis=0), data[~mask][:,stim_mask].mean(axis=0))[0]**2)
    
    return np.mean(cc)

def split_half_correlation(data, N=100):
    '''Given a data matrix trial x stimuli, calculate the leave-one-out pearson correlation'''
    data = data.T
    stim_mask = data.mean(axis=0) != 0

    cc = []
    for i in range(N):
        mask = np.arange(data.shape[0])
        np.random.shuffle(mask)
        cc.append(pearsonr(data[mask[:data.shape[0]//2]][:,stim_mask].mean(axis=0), data[mask[data.shape[0]//2:]][:,stim_mask].mean(axis=0))[0]**2)
    
    return np.mean(cc)

def tolias_ev(data):
    '''Given a data matrix trial x stimuli, calculate the leave-one-out pearson correlation'''
    data = data.T
    stim_mask = data.mean(axis=0) != 0

    total_var = data[:,stim_mask].var()
    noise_var = data[:,stim_mask].var(axis=0).mean()
    EV = (total_var-noise_var)/total_var

    return EV

if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'D:\Data\DeepMouse\Results_raw\noise_ceiling'
    # Filepath to data
    fp_combined = r'D:\Data\DeepMouse\Processed\Combined_raw'

    # Load in the noise ceiling for all regions
    rep_stims = [collapse_dictionary_to_matrix(np.load(os.path.join(fp_combined, region, 'rep_stims.npy'), allow_pickle=True).item()) for region in regions]

    # For each region, calculate the noise ceiling for each neuron
    for i, region in enumerate(regions):
        print(f"Processing {region}")
        ncs = np.array([leave_one_out_correlation(rep_stims[i][:,:,j]) for j in range(rep_stims[i].shape[-1])])
        print(f"Median r2 = {np.median(ncs)}")
        np.save(os.path.join(savepath, f"combined_noise_ceiling_{region}.npy"), ncs)

        # ncs = np.array([split_half_correlation(rep_stims[i][:,:,j]) for j in range(rep_stims[i].shape[-1])])
        # print(f"Median r2 = {np.median(ncs)}")
        # np.save(os.path.join(savepath, f"combined_noise_ceiling_50_{region}.npy"), ncs)

        ncs = np.array([tolias_ev(rep_stims[i][:,:,j]) for j in range(rep_stims[i].shape[-1])])
        print(f"Median r2 = {np.median(ncs)}")
        np.save(os.path.join(savepath, f"combined_noise_ceiling_tolias_{region}.npy"), ncs)


