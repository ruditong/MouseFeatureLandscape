'''
compute_neuron_overlap.py

Compute how much a given neuron overlaps with other regions.
For each neuron, count the labels of the k nearest neighbours. 
Calculate the entropy of this distribution.

Next, for each region, calculate the pairwise correlation across regions (i.e.
those receiving inputs). If this correlation is negative, it suggests separate
input streams. Vice versa, a positive correlation suggests joint streams. In order
to compare the correlation values, estimate a Dirichlet distribution given the
mean probabilities for each region.
'''

from utils import *
from scipy.special import xlogy
import dirichlet
from sklearn.metrics import pairwise_distances


def get_neuron_entropy(pdistance, labels, k=10):
    '''Given a pairwise distance matrix, find the kNN and construct the
       distribution of labels. Then calculate the entropy and normalise.
    '''
    bins = np.arange(labels.min(), labels.max()+2)-0.5
    N = bins.shape[0]-1
    counts = []
    for i in range(pdistance.shape[0]):
        mask = labels != labels[i]
        arr_sorted = np.argsort(pdistance[i][mask])[::-1]
        neighbours = labels[mask][arr_sorted[:k]]
        # Get counts
        count = np.histogram(neighbours, bins=bins, density=True)[0]
        #count = count * N / (N-1)
        counts.append(count)

    counts = np.array(counts)
    entropy = -xlogy(counts, counts).sum(axis=1)/np.log(N-1)

    return entropy, counts

def get_connectivity(counts, labels, label, Nshuffle=100):
    '''Calculate the pairwise connectivity given the selected label'''
    # Get neurons of interest
    mask = labels == label
    prob = counts[mask]
    # Remove column corresponding to self-similarity
    mask_self = np.ones(counts.shape[1]).astype(bool)
    mask_self[label] = False
    prob = prob[:,mask_self]

    # Estimate dirichlet
    prob = prob + 1e-9
    prob = prob/prob.sum(axis=1)[:,None]
    D0 = dirichlet.mle(prob)

    # Now perform pairwise correlations
    connectivity = 1-pairwise_distances(prob.T, metric='correlation')

    # Now generate null distribution
    # connectivity0 = np.array([1-pairwise_distances(np.random.dirichlet(D0, size=prob.shape[0]).T,
    #                                                metric='correlation') for i in range(Nshuffle)])
    connectivity0 = np.array([1-pairwise_distances(np.random.multinomial(20, pvals=prob.mean(axis=0), size=prob.shape[0]).T,
                                                   metric='correlation') for i in range(Nshuffle)])
    
    # Now compare
    connectivity_z = (connectivity-connectivity0.mean(axis=0))/connectivity0.std(axis=0)

    return connectivity, connectivity0, connectivity_z


if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'D:\Data\DeepMouse\Results\neuron_topology'
    
    # Load pdist and labels
    corr = np.load(os.path.join(savepath, r'pdist_correlation.npy'))
    labels = np.load(os.path.join(savepath, r'pdist_correlation_labels.npy'))

    # Convert correlation to distance
    pdistance = 1-np.abs(corr)
    k = 20
    Nshuffle = 1000

    entropy, counts = get_neuron_entropy(pdistance, labels, k=k)
    np.save(os.path.join(savepath, f'neuron_overlap_entropy_k{k}.npy'), entropy)
    np.save(os.path.join(savepath, f'neuron_overlap_counts_k{k}.npy'), counts)

    # Now loop over each region and compute pairwise connectivity
    connectivities = {}
    fig, ax = pl.subplots(ncols=3, nrows=2)
    ax = ax.reshape(-1)
    for i, region in enumerate(regions):
        connectivity, connectivity0, connectivity_z = get_connectivity(counts, labels, label=i, Nshuffle=Nshuffle)
        connectivities[region] = [connectivity, connectivity0, connectivity_z]
        ax[i].imshow(connectivity_z, cmap='bwr', vmin=-3, vmax=3)
        ax[i].axis('off')
        ax[i].set_title(region)

    np.save(os.path.join(savepath, f'neuron_overlap_conenctivity_k{k}.npy'), connectivities)
    pl.show()