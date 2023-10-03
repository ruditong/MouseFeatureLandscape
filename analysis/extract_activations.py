'''
extract_activations.py

Extract model activations to 10k natural images.
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


if __name__ == '__main__':
    savepath = r'D:\Data\DeepMouse\Results_raw\activations\RF_aligned'
    savepath2 = r'D:\Data\DeepMouse\Results_raw\activations\mean_RF'

    # Load in models and dataset
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    fp_base = r'D:\Data\DeepMouse\Processed\Combined_raw'
    fp_nc = r'D:\Data\DeepMouse\Results_raw\noise_ceiling'
    device = 'cuda:0'
    models = load_models(fp_base, fp_nc, regions=['V1', 'LM', 'LI', 'POR', 'AL', 'RL'], device=device)

    batch_size = 512
    fp_ims = r'D:\Data\Models\DNN_images_half.npy'
    dataset, dataloader = load_dataset(fp_ims, batch_size=batch_size)

    # Loop through regions and save activations
    N = 10
    align_RFs = True
    thresh_pred=0.3

    for i, region in enumerate(regions):
        DNN, readout, predictions = models[i]
        activations = main(DNN, readout, dataloader, predictions, thresh_pred=thresh_pred, align_RFs=align_RFs, N=N, device=device)
        np.save(os.path.join(savepath, f'activations_{region}'), activations)

        # Set all neurones to same RF
        activations = main(DNN, readout, dataloader, predictions, thresh_pred=thresh_pred, align_RFs=align_RFs, mean_RF=True, N=N, device=device)
        np.save(os.path.join(savepath2, f'activations_{region}'), activations)
