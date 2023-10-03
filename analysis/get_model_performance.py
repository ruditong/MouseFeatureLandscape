'''
get_model_performance.py

Calculate the performance of DNN model on test data. Performance is assessed as the squared Pearson r.
'''

from utils import *
from scipy.stats import pearsonr

def get_r2(data, prediction):
    '''Given a data matrix trial x stimuli, calculate the leave-one-out pearson correlation'''
    data = data.T
    stim_mask = data.mean(axis=0) != 0

    cc = pearsonr(data[:,stim_mask].mean(axis=0), prediction[stim_mask])[0]**2
    
    return cc

def get_r2_sample(data, prediction, N=500):
    '''Given a data matrix trial x stimuli, calculate the leave-one-out pearson correlation'''
    data = data.T
    stim_mask = data.mean(axis=0) != 0

    cc = []
    for i in range(N):
        mask = np.arange(data.shape[0])
        np.random.shuffle(mask)
        cc.append(pearsonr(data[mask[:data.shape[0]//2]][:,stim_mask].mean(axis=0),prediction[stim_mask])[0]**2)    

    return np.mean(cc)

def main(fp, fp_ims, device='cpu', batch_size=64):
    '''Get model performance'''
    clean_GPU()
    # Load model
    fp_conv = glob.glob(os.path.join(fp, r'*_shallowConv_4_untrained_e2e_16_DNN.pt'))[0]
    fp_readout = glob.glob(os.path.join(fp, r'*_shallowConv_4_untrained_e2e_16_FactorizedReadout.pt'))[0]
    DNN, readout = load_DNN(fp_conv, fp_readout, device=device, model_class='shallowConv_4', layer=16, pretrained=False, normalize=True, bias=True)
    DNN = DNN.eval()
    readout = readout.eval()

    # Load dataset
    images = np.load(os.path.join(fp, 'rep_stims.npy'), allow_pickle=True).item()
    image_labels = list(images.keys())
    data = collapse_dictionary_to_matrix(images)

    dataset = SimpleImageNeuronDataset(fp_ims, keys=image_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Get activations
    DNN.to(device)
    readout.to(device)
    
    activations = np.vstack([ readout(DNN(batch.cuda())).cpu().detach().numpy() for batch,y in dataloader ])
    print(activations.shape, data.shape)
    r2 = np.array([get_r2(data[:,:,i], activations[:,i]) for i in range(data.shape[2])])

    return r2

def main0(fp, fp_ims, device='cpu', batch_size=64):
    clean_GPU()
    '''Get model performance'''
    # Load model
    fp_conv = glob.glob(os.path.join(fp, r'*_shallowConv_4_untrained_e2e_16_DNN.pt'))[0]
    fp_readout = glob.glob(os.path.join(fp, r'*_shallowConv_4_untrained_e2e_16_FactorizedReadout.pt'))[0]
    DNN, readout = load_DNN_untrained(fp_conv, fp_readout, device=device, model_class='shallowConv_4', layer=16, pretrained=False, normalize=True, bias=True)
    DNN = DNN.eval()
    readout = readout.eval()

    # Load dataset
    images = np.load(os.path.join(fp, 'rep_stims.npy'), allow_pickle=True).item()
    image_labels = list(images.keys())
    data = collapse_dictionary_to_matrix(images)

    dataset = SimpleImageNeuronDataset(fp_ims, keys=image_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Get activations
    DNN.to(device)
    readout.to(device)
    
    activations = np.vstack([ readout(DNN(batch.cuda())).cpu().detach().numpy() for batch,y in dataloader ])
    r2 = np.array([get_r2(data[:,:,i], activations[:,i]) for i in range(data.shape[2])])

    return r2

if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'D:\Data\DeepMouse\Results_raw\noise_ceiling'
    fp_ims = r'D:\Data\Models\DNN_images.npy'
    fp_model = r'D:\Data\DeepMouse\Processed\Combined_raw'

    fps = [os.path.join(fp_model, region) for region in regions]

    for i, region in enumerate(regions):
        print(f"Processing {region}")
        r2 = main(fps[i], fp_ims, device='cuda:0', batch_size=512)
        print(f'Median r2 = {np.median(r2)}')
        np.save(os.path.join(savepath, f"combined_r2_{region}.npy"), r2)

        # r2 = main0(fps[i], fp_ims, device='cuda:0', batch_size=512)
        # print(f'Median r2 = {np.median(r2)}')
        # np.save(os.path.join(savepath, f"combined_r2_random_{region}.npy"), r2)