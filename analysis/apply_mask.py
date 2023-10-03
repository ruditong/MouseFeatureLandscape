'''
apply_mask.py

Apply the fitted spatial mask to optimal images and crop to size.
'''

from utils import *


if __name__ == '__main__':
    savepath = r'D:\Data\DeepMouse\Results_raw\optstims\RF_aligned'
    fp_mask = r'D:\Data\DeepMouse\Results_raw\representational_similarity'
    fp_opt = r'D:\Data\DeepMouse\Results_raw\optstims\RF_aligned'
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']

    r2_thresh = 0.6
    sigma=5

    # Load in spatial masks
    masks = [np.load(os.path.join(fp_mask, f'spatial_mask_fit_{region}.npy'), allow_pickle=1).item() for region in regions]
    # Load in optimal images
    optstims = [np.load(os.path.join(fp_opt, f'optStim_{region}.npy')) for region in regions]
    imsize = optstims[0][0].shape[1]
    
    for i, region in enumerate(regions):
        popts, r2 = masks[i]['popts'], masks[i]['r2'] 
        optstim = optstims[i]
        r2_mask = r2 > r2_thresh
        print(f"After r2 threshold {r2_mask.sum()} neurons remain out of {r2.shape[0]}")
        # Calculate the median popt
        popts_m = np.nanmedian(popts[r2_mask], axis=0)
        # Set mu to 0
        popts_m[1:3] = 0

        # Find cutoff
        stdev = np.mean([twoD_Gaussian( (popts_m[3]*1.96, 0), *popts_m), twoD_Gaussian((0, popts_m[4]*1.96), *popts_m)])
        
        # Construct the Gaussian
        X, Y = np.meshgrid(np.arange(imsize)-imsize/2, np.arange(imsize)-imsize/2)
        spatial_mask = twoD_Gaussian((X.ravel(), Y.ravel()), *popts_m).reshape((imsize, imsize)) > stdev
        # Apply gaussian smoothing
        spatial_mask = gaussian_filter(spatial_mask.astype(float), sigma)

        # Apply mask to images
        bgr = np.zeros((imsize, imsize))+0.5
        optstim = optstim * spatial_mask[None,None] + (bgr*(1-spatial_mask))[None,None]
        
        # Crop images to 64x64
        optstim = optstim[:,:,imsize//2-32:imsize//2+32, imsize//2-32:imsize//2+32]

        # Save
        np.save(os.path.join(savepath, f'optStim_masked_{region}.npy'), optstim)
        # Also save individually
        if not os.path.isdir(os.path.join(savepath, f'optStim_masked_{region}')):
            os.mkdir(os.path.join(savepath, f'optStim_masked_{region}'))

        for j, im in tqdm(enumerate(optstim)):
            pl.imsave(os.path.join(savepath, f'optStim_masked_{region}', f'{region}_{j}.png'), np.squeeze(im), vmin=0, vmax=1, cmap='gray')
