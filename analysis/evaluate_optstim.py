'''
evalute_optstim.py

For each optstim, get its response. Then remove anything below threshold of -1 stdev.
'''

from utils import *

if __name__ == '__main__':
    # Load activations and predictions
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    fp_cross = r'D:\Data\DeepMouse\Results_raw\generalisation'
    fp_nc = r'D:\Data\DeepMouse\Results_raw\noise_ceiling'
    savepath = r'D:\Data\DeepMouse\Results_raw\optstims\RF_aligned'
    activations = [np.load(os.path.join(fp_cross, f"cross_presentation_{region}.npy"), allow_pickle=True).item() for region in regions]

    # Load predictions
    predictions = [load_performance(fp_nc, region) > 0.3 for region in regions]

    # For each region, get the response and mask
    thresh = -1
    for i, region in enumerate(regions):
        mask = np.diagonal(StandardScaler().fit_transform(activations[i][region][predictions[i]].T)) > thresh 
        np.save(os.path.join(savepath, f'good_optstim_{region}.npy'), mask)
