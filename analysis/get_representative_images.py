'''
get_representative_images.py

For each region, find the most representative images using a kNN approach.
The most representative images are those that are surrounded most to other
images of the same region. 
Find images with highest self-consistency, then either choose images that are
as different as possible in this set or run GMM to look for groups and sample
randomly from the groups.
'''

from utils import *
from sklearn.neighbors import KNeighborsClassifier


def main(embedding, labels, images, regions, k, N):
    '''Given a matrix of image embeddings (images x N) and labels (images,) find the
       most representative.
    '''
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(embedding, labels)
    Y = model.predict_proba(embedding) # images, len(regions)

    repr_ims = []
    raw_prob = {}
    for i, label in enumerate(np.unique(labels)):
        prob = Y[labels==label][:,i]
        idx = np.argsort(prob)[::-1]
        repr_ims.append(images[labels==label][idx[:N]])
        raw_prob[regions[i]] = prob

    return repr_ims, raw_prob

def main2(embedding, labels, regions, k, N):
    '''Given a matrix of image embeddings (images x N) and labels (images,) find the
       most representative.
    '''
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(embedding, labels)
    Y = model.predict_proba(embedding) # images, len(regions)

    raw_prob = {}
    for i, label in enumerate(np.unique(labels)):
        prob = Y[labels==label]
        raw_prob[regions[i]] = prob

    return raw_prob


if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'D:\Data\DeepMouse\Results_raw\representative_images'
    fp_opt = r'D:\Data\DeepMouse\Results_raw\optstims\RF_aligned'
    fp_nc = r'D:\Data\DeepMouse\Results_raw\noise_ceiling'
    fp_embed = r'D:\Data\DeepMouse\Results_raw\neuron_topology\SimCNE_affine_embedding.npy'
    fp_ims = [os.path.join(fp_opt, f'optStim_masked_{region}.npy') for region in regions]
    masks = np.concatenate([np.load(os.path.join(fp_opt, f'good_optstim_{region}.npy')) for region in regions])

    predictions = [load_performance(fp_nc, region) for region in regions]
    pred_mask = np.concatenate(predictions) > 0.3

    # Parameters
    k = 100
    N = 200

    # Load dataset
    dataset = OptStims(fp_ims, n_views=2, do_transform=False, transformations=None)
    dataset.ims = dataset.ims[pred_mask][masks]
    dataset.labels = dataset.labels[pred_mask][masks]
    images = dataset.ims.cpu().numpy()[:,0]
    labels = dataset.labels

    # Load embeddings
    embedding = np.load(fp_embed, allow_pickle=1).item() # neuron x 256
    embedding = np.concatenate([embedding[region] for region in regions], axis=0)

    # Find representative images
    repr_ims, raw_prob = main(embedding, labels, images, regions, k, N) # repr_ims = len(regions), N x image_dim

    # Save images
    for i, region in enumerate(regions):
        for j, im in enumerate(repr_ims[i]):
            pl.imsave(os.path.join(savepath, region, f"{j}.png"), im, cmap='gray', vmin=0, vmax=1)

    np.save(os.path.join(savepath, 'raw_probabilities.npy'), raw_prob)

