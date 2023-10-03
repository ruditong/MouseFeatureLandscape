'''
calculate_spatial_mask.py

Calculate the size of the spatial mask for each neuron.
'''

from utils import *
from scipy.optimize import curve_fit

def get_r2(y, ypred):
    SSres = ((y-ypred)**2).sum()
    SStot = ((y-y.mean())**2).sum()
    r2 = 1-SSres/SStot
    return r2

def fit_gaussian(data):
    '''Fit a 2D gaussian given data matrix N x N'''
    data_flat = data.ravel()
    # Create x and y meshgrid
    X, Y = np.meshgrid(np.arange(data.shape[0])-data.shape[0]/2, np.arange(data.shape[1])-data.shape[1]/2)
    # Guess initial params (amplitude, xo, yo, sigmax, sigmay, theta, offset)
    max_coord = np.array(np.unravel_index(np.argmax(data), data.shape)) - data.shape[0]/2
    p0 = (1, max_coord[1], max_coord[0], 15, 15, 0, data.min())
    # Fit

    popt, pcov = curve_fit(twoD_Gaussian, (X.ravel(), Y.ravel()), data_flat, p0=p0)

    # Calculate r2
    spatial_pred = twoD_Gaussian((X,Y), *popt)
    r2 = get_r2(data_flat, spatial_pred.ravel())

    return popt, r2

def main(readout, predictions, pred_thresh=0.3):
    '''Fit 2D Gaussian RFs to all spatial masks'''
    # Extract the spatial layer
    spatial = readout.spatial.detach().cpu().numpy()
    spatial = spatial[predictions > pred_thresh]
    spatial_normalised = np.array([np.abs(rf/(np.sqrt((rf**2).sum())+ 1e-6)) for rf in spatial])
    spatial_upsampled = np.array([upsample_spatial_filter(spatial_normalised[i], 8, 135) for i in range(spatial_normalised.shape[0])])

    # Fit a Gaussian to each spatial mask
    popts, r2 = [], []
    for s in tqdm(spatial_upsampled):
        try: popt, r = fit_gaussian(s)
        except: popt, r = [np.nan]*7, -1
        popts.append(popt)
        r2.append(r)

    popts = np.array(popts)
    r2 = np.array(r2)

    return popts, r2

if __name__ == '__main__':
    savepath = r'D:\Data\DeepMouse\Results_raw\representational_similarity'
    # Load in models and dataset
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    fp_base = r'D:\Data\DeepMouse\Processed\Combined_raw'
    fp_nc = r'D:\Data\DeepMouse\Results_raw\noise_ceiling'
    device = 'cuda:0'
    models = load_models(fp_base, fp_nc, regions=regions, device=device)
    

    # Loop through regions and save activations
    N = 10
    align_RFs = True
    pred_thresh=0.3

    for i, region in enumerate(regions):
        print(f'Processing {region}')
        DNN, readout, predictions = models[i]
        popts, r2 = main(readout, predictions, pred_thresh)
        output = {'popts': popts, 'r2': r2}
        np.save(os.path.join(savepath, f'spatial_mask_fit_{region}'), output)