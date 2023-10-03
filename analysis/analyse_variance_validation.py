'''
analyse_variance_validation.py

For validation experiments, for each cell, calculate the variance for optimised vs natural stimuli
'''

from utils import *
from sklearn.preprocessing import StandardScaler

def load_all_data(fp_base, regions, axis=0):
    '''Given path to folder containing validation data, import all data and extract
       umask and allstim
    '''
    results = {}
    for i, region in enumerate(regions):
        fps = glob.glob(os.path.join(fp_base, region, r'*.npy'))
        umask = []
        natstim = []
        for fp in fps:
            data = np.load(fp, allow_pickle=True).item()
            mask = data['mask']
            combined = np.concatenate([data['umask'][:,mask], data['all_stim'][:,mask]], axis=0)
            scaler = StandardScaler()
            scaler.fit(combined)

            # umask.append(scaler.transform(data['umask'][:,mask]).var(axis=axis, ddof=1))
            # natstim.append(scaler.transform(data['all_stim'][:,mask]).var(axis=axis, ddof=1))
            # umask.append(np.linalg.norm(scaler.transform(data['umask'][:,mask]), axis=axis))
            # natstim.append(np.linalg.norm(scaler.transform(data['all_stim'][:,mask]), axis=axis))
            # umask.append(np.mean(scaler.transform(data['umask'][:,mask]), axis=axis))
            # natstim.append(np.mean(scaler.transform(data['all_stim'][:,mask]), axis=axis))
            umask.append(np.percentile(scaler.transform(data['umask'][:,mask]), 90, axis=axis))
            natstim.append(np.percentile(scaler.transform(data['all_stim'][:,mask]), 90, axis=axis))

        umask = np.concatenate(umask)
        natstim = np.concatenate(natstim)
        results[region] = [umask, natstim]

    return results

if __name__ == '__main__':
    # Filepaths and parameters
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'D:\Data\DeepMouse\Results\tuning_dynamics'
    fp_val = r'D:\Data\DeepMouse\Results\validation'

    # results = load_all_data(fp_val, regions, axis=0)
    # np.save(os.path.join(savepath, r'neuronvariance_validation.npy'), results)
    # results = load_all_data(fp_val, regions, axis=1)
    # np.save(os.path.join(savepath, r'popvariance_validation.npy'), results)

    # results = load_all_data(fp_val, regions, axis=1)
    # np.save(os.path.join(savepath, r'L2_validation.npy'), results)

    # results = load_all_data(fp_val, regions, axis=1)
    # np.save(os.path.join(savepath, r'mean_validation.npy'), results)

    results = load_all_data(fp_val, regions, axis=0)
    np.save(os.path.join(savepath, r'percentile90_validation.npy'), results)
