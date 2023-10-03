'''
test_generalisation_insilico.py

Test whether representative images preferentially activate each region in the in silico model.
'''

from utils import *

def main(DNN, readout, dataloader, predictions, thresh_pred=0.3, align_RFs=True, mean_RF=False, N=10, device='cpu'):
    '''Given a model and dataloader, present all stimuli and return the activations. Exclude neurons that do
       not pass the predictions threshold'''
    clean_GPU()

    # Create prediction mask to exclude low prediction neurons
    mask = predictions > thresh_pred
    print(f"Number of neurons left after prediction threshold: {mask.sum()}")

    if align_RFs:
        spatial = readout.spatial.detach().cpu().numpy()
        spatial_shifted, spatial_normalised, coms, coms_shifted = get_average_RF(np.squeeze(spatial), N=N)
        # Overwrite spatial mask
        readout.spatial = nn.Parameter(torch.from_numpy(spatial_shifted.reshape(spatial.shape)))
    if mean_RF:
        spatial_median = np.median(spatial_shifted, axis=0)
        readout.spatial = nn.Parameter(torch.from_numpy(np.repeat(spatial_median[None], spatial.shape[0], axis=0)[:,None]))

    # Construct full model
    combined_model = DNN_readout_combined(DNN, ReadOut(readout_model=readout)).to(device).eval()

    # Run all stimuli through model (stim, neuron)
    activations = np.vstack([ combined_model(batch.cuda()).cpu().detach().numpy() for batch,y in dataloader ])[:,mask]

    return activations

def embed_images(ims):
    '''Embed a list of images N x 64 x 64 into N x 135 x 135'''
    new_ims = np.zeros((ims.shape[0], 135, 135)) + 0.5
    new_ims[:,135//2-32:135//2+32,135//2-32:135//2+32] = ims
    new_ims = np.repeat(new_ims[:,None], repeats=3, axis=1)
    new_ims = np.transpose(new_ims, axes=(0,2,3,1))
    return new_ims

if __name__ == '__main__':
    # Load images and embed
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    fp_base = r'D:\Data\DeepMouse\Processed\Combined_raw'
    fp_ims = r'D:\Data\DeepMouse\Results_raw\optstims\RF_aligned'
    fp_nc = r'D:\Data\DeepMouse\Results_raw\noise_ceiling'
    savepath = r'D:\Data\DeepMouse\Results_raw\generalisation'
    
    # Load images
    for i, regioni in enumerate(regions):
        print(f"Running {regioni}")
        fp_conv = os.path.join(fp_base, regioni, f'{regioni}_shallowConv_4_untrained_e2e_16_DNN.pt')
        fp_readout = os.path.join(fp_base, regioni, f'{regioni}_shallowConv_4_untrained_e2e_16_FactorizedReadout.pt') 
        
        predictions = load_performance(fp_nc, regioni)
        DNN, readout = load_DNN(fp_conv, fp_readout, device='cuda:0', model_class='shallowConv_4', layer=16, pretrained=False, normalize=True, bias=True)
        activations = {}
        for j, regionj in enumerate(regions):
            images = np.squeeze(np.load(os.path.join(fp_ims, f"optStim_masked_{regionj}.npy")))
            # Embed 64x64 images into 135x135
            images = embed_images(images)
            dataset = SimpleImageNeuronDataset(images)
            dataloader = DataLoader(dataset, batch_size=256)

            activations[regionj] = main(DNN, readout, dataloader, predictions, thresh_pred=0.3, align_RFs=True, mean_RF=False, N=10, device='cuda:0')

        np.save(os.path.join(savepath, f'cross_presentation_{regioni}.npy'), activations)

