'''
get_image_embedding.py

Embed optimised images and return embedding coordinates.
'''

from utils import *
from models import *
from umap import UMAP

def load_SimCNE(fp_params, out_dim):
    '''Load a pretrained SimCNE model'''
    model_params = torch.load(fp_params)
    model = ResNetSimCLR(out_dim=out_dim, base_model="resnet18")
    model.load_state_dict(model_params)
    return model


def get_latents(model, dataset, device='cpu'):
    '''Sample latent space'''
    model.eval()
    dataset.do_transform = False
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    latents = []
    for X,y in tqdm(dataloader):
        X = X.to(device)
        Y = model(X)
        latents.append(Y.detach().cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    return latents

if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'D:\Data\DeepMouse\Results_raw\neuron_topology'
    fp_opt = r'D:\Data\DeepMouse\Results_raw\optstims\RF_aligned'
    fp_nc = r'D:\Data\DeepMouse\Results_raw\noise_ceiling'
    masks = np.concatenate([np.load(os.path.join(fp_opt, f'good_optstim_{region}.npy')) for region in regions])
    fp_ims = [os.path.join(fp_opt, f'optStim_masked_{region}.npy') for region in regions]

    predictions = [load_performance(fp_nc, region) for region in regions]
    pred_mask = np.concatenate(predictions) > 0.3

    # Parameters
    out_dim = 256
    device='cuda:0'
    n_neighbors = 30
    min_dist=0.1

    # Load dataset
    dataset = OptStims(fp_ims, n_views=2, do_transform=False, transformations=None)
    dataset.ims = dataset.ims[pred_mask][masks]
    dataset.labels = dataset.labels[pred_mask][masks]
    images = dataset.ims.cpu().numpy()[:,0]
    labels = dataset.labels

    # Load Affine SimCNE
    fp_params_full = r'D:\Data\DeepMouse\Results_raw\simcne\affine\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    latents = get_latents(model, dataset, device=device)
    latents_stratified = {region: latents[labels == i] for i, region in enumerate(regions)}
    np.save(os.path.join(savepath, f'SimCNE_affine_latents.npy'), latents_stratified)

    # Perform umap
    embedding = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean').fit_transform(latents)
    # Save as dictionary
    embedding_stratified = {region: embedding[labels == i] for i, region in enumerate(regions)}

    np.save(os.path.join(savepath, f'SimCNE_affine_embedding.npy'), embedding_stratified)
