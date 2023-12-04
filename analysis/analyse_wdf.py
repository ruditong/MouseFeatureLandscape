'''
analyse_wdf.py

Analyse widefield validation experiments.
In widefield mode, brain regions were presented optimised images from all regions.
In addition, retinotopic mapping is performed.

1. Find HVAs by retinotopic mapping and segment pixels.
2. For each region, calculate the average activity across pixels.
3. Perform bandpass filtering to remove noise (high pass at 2*stimulus frequency)
4. Get the response kernel for each region.
5. Calculate the average activity to each class of optimised images.
'''

from utils import *
import skvideo.io
from scipy import signal
from scipy import ndimage
import skimage.transform as transform

def load_mj2_mask(fn, mask, start_frame=0, end_frame=0, resize=None, index=None):
    '''Load the video file specified in path and average for each masked region.
       File should be in .mj2 format.
       Uses ffmpeg backend and scikit-video to load video file.

       INPUT
       fn           : Filepath to video file (.mj2 format)
       mask         : Mask array that contains labels for each region
       start_frame  : First frame to read (count from 0)
       end_frame    : Last frame to read 
       resize       : Factor to resize image to. Skip if None

       OUTPUT
       vid          : Video file as numpy.array (frames x dim1 x dim2)
    '''

    # Create input and output parameters for ffmpeg
    inputparameters = {'-ss' : '%s'%(start_frame)}  # -ss seeks to the position/frame in video file
    outputparameters = {'-pix_fmt' : 'gray16be'}     # specify to import as uint16, otherwise it's uint8
    if end_frame == 0: num_frames = 0
    else: num_frames = end_frame

    # Import video file as numpy.array
    labels = np.array(index)
    vidreader = skvideo.io.vreader(fn, inputdict=inputparameters, outputdict=outputparameters, num_frames=num_frames)
    # Prepare output
    for (i, frame) in tqdm(enumerate(vidreader)):                     # vreader is faster than vread, but frame by frame
        if i == 0: 
            imagesize = (int(frame.shape[0]*resize), int(frame.shape[1]*resize))
            vid = np.zeros( (labels.shape[0], num_frames) )

        vid[:,i] = ndimage.mean(np.squeeze(transform.resize(frame, imagesize)), labels=mask.astype(int), index=labels)       # Resize in place for performance

    return vid

def preprocess_video(vid, T=2, fr=10):
    '''Run a high pass filter on video data'''
     # Remove DC component of image (i.e. average intensity value)
    data_ds_rel = vid - np.nanmean(vid, axis=0)
    # Apply high pass filter (subtract out frequencies below 2 cycles of the stimulus)
    cutfreq = 1/(2*T)
    b, a = signal.butter(1, cutfreq/(0.5*fr), 'low')
    data_lowfreq = signal.filtfilt(b, a, data_ds_rel, axis=0)
    data_filt = data_ds_rel - data_lowfreq
    return data_filt

def extract_stimuli(vid, ttl, window=(7, 15)):
    '''Given a video matrix (region, frames) and ttls, extract stimuli'''
    data = np.array([vid[:,i-window[0]:i+window[1]] for i in ttl])
    return data

def read_logfile(fp):
    '''Loads stimulus log file. Each line is a dictionary containing stimulus information'''
    with open(fp, 'r') as file:
        log = file.read().splitlines()
    
    header = log.pop(0)
    log = [eval(i) for i in log]

    return log, header

def get_log_label(log, regions):
    '''Assign number to each region in log'''
    labels = [l['name'] for l in log]
    idx = np.zeros(len(labels))
    for i, region in enumerate(regions):
        mask = np.array([region in label for label in labels])
        idx[mask] = i
    return idx

if __name__ == '__main__':
    # Filepaths and parameters
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'F:\DeepMouse\RT_114\wdf\20230804\wdf'
    fp_mask = r'F:\DeepMouse\RT_114\wdf\20230804\wdf\patch_map.npy'
    fp_vid = r'F:\DeepMouse\RT_114\wdf\20230804\wdf\wdf_000_000.mj2'
    fp_log = r'F:\DeepMouse\RT_114\wdf\20230804\wdf\20230804-133250_log.txt'

    #index = [6,7,9,11,8,3]
    index = [5,8,11,12,9,7]
    T = 2
    fr = 10

    mask = np.load(fp_mask)
    ttls = sbx_get_ttlevents(fp_vid + r'_events')

    vid = load_mj2_mask(fp_vid, mask, start_frame=0, end_frame=ttls[-1]+20, resize=0.5, index=index)
    

    vid_filt = np.array([preprocess_video(vid[i], T=T, fr=fr) for i in range(vid.shape[0])])
    ttls = ttls[1:-1] # Remove wait period

    np.save(os.path.join(savepath, r'segmented_activity.npy'), vid_filt)

    data = extract_stimuli(vid_filt, ttls, window=(7, 15))

    # Now normalise data
    data_m, data_s = data[:,:,:5].mean(axis=(0,-1)), data[:,:,:5].std(axis=(0,-1))
    data_norm = (data-data_m[None,:,None])/data_s[None,:,None]
    np.save(os.path.join(savepath, r'segmented_activity_norm.npy'), data_norm)

    # Extract TTL labels
    log, header = read_logfile(fp_log)
    # Remove waits
    log = log[1:-1]
    labels = get_log_label(log, regions)
    np.save(os.path.join(savepath, r'labels.npy'), labels)

    results = {'segmented' : vid,
               'segmented_norm' : data_norm,
               'labels': labels}
    
    np.save(os.path.join(savepath, r'results.npy'), results)