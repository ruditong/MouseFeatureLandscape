'''
calculate_linear_predictability.py

Perform pairwise linear regression across regions.
'''

from utils import *
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

def get_adjusted_r2(r2, n, p):
    '''Calculate the adjusted r2 given sample size n and #independent variables p'''
    r2_adj = 1-(1-r2)*(n-1)/(n-p-1)
    return r2_adj

def shuffle_data(data):
    '''Shuffle each row'''
    data_copy = data.copy()
    for i in data_copy: np.random.shuffle(i)
    return data_copy

def main(activations, N=24, cv=5, shuffle=False):
    '''Hierarchy via PLSRegression'''
    # stim, neuron
    model = PLSRegression(n_components=N)
    scores = np.zeros((len(activations), len(activations), cv))
    pbar = tqdm(total=len(activations)**2)
    
    for i in range(len(activations)):
        for j in range(len(activations)):
            if i == j: 
                scores[i,j] = 1
                continue   

            if shuffle: X = shuffle_data(activations[i])
            else: X = activations[i]

            # Calculate sample size and number of independent variables
            n = np.round(activations[i].shape[0]*(1-1/cv))
            p = activations[i].shape[1]
            s = cross_val_score(model, X, activations[j], cv=cv)
            s = get_adjusted_r2(s, n, p)
            scores[i,j] = s
            pbar.update(1)

    return scores

if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'D:\Data\DeepMouse\Results\representational_similarity'
    fp_act = r'D:\Data\DeepMouse\Results\activations\RF_aligned'

    nPC = 25
    cv = 5

    # Load activations
    activations = [StandardScaler().fit_transform(np.load(os.path.join(
                   fp_act, f"activations_{region}.npy"))) for region in regions]
    
    # scores = main(activations, N=nPC, cv=cv)
    # np.save(os.path.join(savepath, f'linear_predicatbility_pls_cv-{cv}.npy'), scores)

    scores0 = main(activations, N=nPC, cv=cv, shuffle=True)
    np.save(os.path.join(savepath, f'linear_predicatbility_pls0_cv-{cv}.npy'), scores0)