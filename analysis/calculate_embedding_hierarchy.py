'''
calculate_embedding_hierarchy.py

Test which embedding of optimised stimulus space fits best with representational hierarchy.
We build a hierarchy by kNN in the embedding space and looking at the confusion matrix as a measure of local overlap.
'''

from utils import *
from models import *
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def get_confusion(latents, labels, nneighbour_range=(5,20), n=10, split=0.8):
    '''Calculate the confusion matrix for a range of nneighbours'''
    nlabels = np.unique(labels).shape[0]
    split_n = int(labels.shape[0]*split)
    conf_matrix = np.zeros( (len(nneighbour_range), n, nlabels, nlabels) )   # nneighbour x n x regions x regions 

    for i, nneighbours in tqdm(enumerate(nneighbour_range)):
        idx_train, idx_test = split_labels(labels, split=split, N=n)
        for j in range(n):
            kNN = KNeighborsClassifier(n_neighbors=nneighbours)
            # Split data into random subset
            # idx = np.arange(labels.shape[0])
            # np.random.shuffle(idx)
            # kNN.fit(latents[idx[:split_n]], labels[idx[:split_n]])
            kNN.fit(latents[idx_train[j]], labels[idx_train[j]])
            Y = kNN.predict(latents[idx_test[j]])
            conf_matrix[i,j] = confusion_matrix(labels[idx_test[j]], Y, normalize='true')

    return conf_matrix

def get_probability(latents, labels, nneighbour_range=(5,20), n=10, split=0.8):
    '''Calculate the confusion matrix for a range of nneighbours'''
    nlabels = np.unique(labels).shape[0]
    split_n = int(labels.shape[0]*split)
    conf_matrix = np.zeros( (len(nneighbour_range), n, nlabels, nlabels) )   # nneighbour x n x regions x regions 

    for i, nneighbours in tqdm(enumerate(nneighbour_range)):
        idx_train, idx_test = split_labels(labels, split=split, N=n)
        for j in range(n):
            kNN = KNeighborsClassifier(n_neighbors=nneighbours)
            kNN.fit(latents[idx_train[j]], labels[idx_train[j]])
            Y = kNN.predict_proba(latents[idx_test[j]])
            for label in np.unique(labels):
                conf_matrix[i,j,int(label)] = Y[labels[idx_test[j]] == int(label)].mean(axis=0)

    return conf_matrix

def split_labels(labels, split=0.8, N=100):
    '''Given a list of labels, split each label into two groups of ratio 'split' N times.'''
    idx = np.arange(labels.shape[0])
    idx_labels = [idx[labels == i] for i in np.unique(labels)]

    splits_train, splits_test = [], []
    nsplit = int(split*np.min([idx_label.shape[0] for idx_label in idx_labels]))
    for i in range(N):
        # For each label, shuffle and split into two halves
        for x in idx_labels: np.random.shuffle(x)
        idx_train = np.concatenate([idx_label[:nsplit] for idx_label in idx_labels])
        idx_test = np.concatenate([idx_label[nsplit:] for idx_label in idx_labels])
        splits_train.append(idx_train.copy())
        splits_test.append(idx_test.copy())

    return np.array(splits_train).astype(int), np.array(splits_test).astype(int)

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
    fp_ims = [os.path.join(fp_opt, f'optStim_masked_{region}.npy') for region in regions]
    masks = np.concatenate([np.load(os.path.join(fp_opt, f'good_optstim_{region}.npy')) for region in regions])

    predictions = [load_performance(fp_nc, region) for region in regions]
    pred_mask = np.concatenate(predictions) > 0.3

    # Parameters
    out_dim = 256
    device='cuda:0'
    nneighbour_range = [10, 20, 50]
    n = 100
    split = 0.75
    nPC = 256

    # Load dataset
    dataset = OptStims(fp_ims, n_views=2, do_transform=False, transformations=None)
    dataset.ims = dataset.ims[pred_mask][masks]
    dataset.labels = dataset.labels[pred_mask][masks]
    images = dataset.ims.cpu().numpy()[:,0]
    labels = dataset.labels

    # Calculate overlap for each embedding space
    # 1.1 Full SimCNE
    fp_params_full = r'D:\Data\DeepMouse\Results\simcne\full\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    latents1 = get_latents(model, dataset, device=device)
    #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    conf_matrix1 = get_confusion(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    np.save(os.path.join(savepath, 'confusion_matrix_SimCNEfull.npy'), conf_matrix1)

    # # 1.2 Scale SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\scale\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_confusion(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'confusion_matrix_SimCNEscale.npy'), conf_matrix1)

    # # 1.3 Rotation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\rotation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_confusion(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'confusion_matrix_SimCNErotation.npy'), conf_matrix1)

    # # 1.4 Translation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\translation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_confusion(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'confusion_matrix_SimCNEtranslation.npy'), conf_matrix1)

    # # 1.5 Scale + Rotation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\scale_rotation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_confusion(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'confusion_matrix_SimCNEscalerotation.npy'), conf_matrix1)

    # # 1.6 Scale + Translation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\scale_translation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_confusion(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'confusion_matrix_SimCNEscaletranslation.npy'), conf_matrix1)

    # # 1.7 Rotation + Translation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\rotation_translation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_confusion(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'confusion_matrix_SimCNErotationtranslation.npy'), conf_matrix1)

    # 1.8 Affine SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results_raw\simcne\affine\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_confusion(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'confusion_matrix_SimCNEaffine.npy'), conf_matrix1)

    # # 2. Pixel space
    # latents2 = images.reshape(dataset.ims.shape[0], -1)
    # latents2 = PCA(n_components=nPC).fit_transform(latents2)
    # conf_matrix2 = get_confusion(latents2, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'confusion_matrix_pixel.npy'), conf_matrix2)

    # # 3. FFT space
    # latents3 = np.array([np.abs(np.fft.fftshift(np.fft.fft2(im)))**2 for im in images])
    # # Remove DC
    # latents3[:, latents3.shape[1]//2, latents3.shape[2]//2] = 0
    # latents3 = latents3.reshape(latents3.shape[0], -1)
    # latents3 = PCA(n_components=nPC).fit_transform(latents3.reshape(latents3.shape[0], -1))
    # conf_matrix3 = get_confusion(latents3, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'confusion_matrix_FFT.npy'), conf_matrix3)

    # Calculate overlap using probabilities for each embedding space
    # 1.1 Full SimCNE
    fp_params_full = r'D:\Data\DeepMouse\Results\simcne\full\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    latents1 = get_latents(model, dataset, device=device)
    #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    conf_matrix1 = get_probability(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    np.save(os.path.join(savepath, 'probability_matrix_SimCNEfull.npy'), conf_matrix1)

    # # 1.2 Scale SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\scale\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_probability(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'probability_matrix_SimCNEscale.npy'), conf_matrix1)

    # # 1.3 Rotation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\rotation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_probability(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'probability_matrix_SimCNErotation.npy'), conf_matrix1)

    # # 1.4 Translation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\translation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_probability(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'probability_matrix_SimCNEtranslation.npy'), conf_matrix1)

    # # 1.5 Scale + Rotation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\scale_rotation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_probability(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'probability_matrix_SimCNEscalerotation.npy'), conf_matrix1)

    # # 1.6 Scale + Translation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\scale_translation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_probability(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'probability_matrix_SimCNEscaletranslation.npy'), conf_matrix1)

    # # 1.7 Rotation + Translation SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results\simcne\rotation_translation\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_probability(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'probability_matrix_SimCNErotationtranslation.npy'), conf_matrix1)

    # # 1.8 Affine SimCNE
    # fp_params_full = r'D:\Data\DeepMouse\Results_raw\simcne\affine\SimCNE_resnet18_outdim256_temp0.1_epoch500_d0None.pth'
    # model = load_SimCNE(fp_params_full, out_dim=out_dim).to(device)
    # latents1 = get_latents(model, dataset, device=device)
    # #latents1 = PCA(n_components=nPC).fit_transform(latents1)
    # conf_matrix1 = get_probability(latents1, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'probability_matrix_SimCNEaffine.npy'), conf_matrix1)

    # # 2. Pixel space
    # latents2 = images.reshape(dataset.ims.shape[0], -1)
    # latents2 = PCA(n_components=nPC).fit_transform(latents2)
    # conf_matrix2 = get_probability(latents2, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'probability_matrix_pixel.npy'), conf_matrix2)

    # # 3. FFT space
    # latents3 = np.array([np.abs(np.fft.fftshift(np.fft.fft2(im)))**2 for im in images])
    # # Remove DC
    # latents3[:, latents3.shape[1]//2, latents3.shape[2]//2] = 0
    # latents3 = latents3.reshape(latents3.shape[0], -1)
    # latents3 = PCA(n_components=nPC).fit_transform(latents3.reshape(latents3.shape[0], -1))
    # conf_matrix3 = get_probability(latents3, labels, nneighbour_range=nneighbour_range, n=n, split=split)
    # np.save(os.path.join(savepath, 'probability_matrix_FFT.npy'), conf_matrix3)