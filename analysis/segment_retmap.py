'''
segment_retmap.py

Segment retinotopic map into HVAs and return labelled mask.
'''

from utils import *
import skvideo.io
from scipy import signal
import skimage.transform as transform

def load_mj2(fn, start_frame=0, end_frame=0, resize=None):
    '''Load the video file specified in path.
       File should be in .mj2 format.
       Uses ffmpeg backend and scikit-video to load video file.

       INPUT
       fn           : Filepath to video file (.mj2 format)
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
    else: num_frames = end_frame-start_frame

    # Import video file as numpy.array
    vidreader = skvideo.io.vreader(fn, inputdict=inputparameters, outputdict=outputparameters, num_frames=end_frame)
    for (i, frame) in enumerate(vidreader):                     # vreader is faster than vread, but frame by frame
        if i == 0: 
            imagesize = (int(frame.shape[0]*resize), int(frame.shape[1]*resize))
            vid = np.zeros( (end_frame, imagesize[0], imagesize[1]) )
        
        vid[i,:,:] = np.squeeze(transform.resize(frame, imagesize))       # Resize in place for performance

    return vid

def get_fft(fvid, fttl, timetot, nrep, fr):
    '''Get FFT map of retinotopic mapping data'''
    T = timetot/nrep
    nimagetot = int(np.round(timetot*fr))
    fstim = 1/T

    # Load video
    ttl = sbx_get_ttlevents(fttl)
    start_frame = ttl[ttl > 0][0]-1
    vid = load_mj2(fvid, start_frame=start_frame, end_frame=nimagetot, resize=0.5)

    # Remove DC component of image (i.e. average intensity value)
    data_ds_rel = vid - np.nanmean(vid, axis=0)
    # Apply high pass filter (subtract out frequencies below 2 cycles of the stimulus)
    cutfreq = 1/(2*T)
    b, a = signal.butter(1, cutfreq/(0.5*fr), 'low')
    data_lowfreq = signal.filtfilt(b, a, data_ds_rel, axis=0)
    data_filt = data_ds_rel - data_lowfreq
    
    # Perform fft
    t = np.arange(data_filt.shape[0])*(1/fr)
    data_fft = np.apply_along_axis(lambda x: x @ np.exp(2j * np.pi * fstim * t), 0, data_filt)
    return data_fft

def combine_maps(fft1, fft2):
    '''Average two fft maps'''
    phase = np.angle(fft1/fft2)/2
    amplitude = (np.abs(fft1)+np.abs(fft2))/2
    return phase, amplitude

def get_sign_map(phase1, phase2):
    '''Given two phase maps at 90 deg angle, calculate the sign of the gradient'''
    # Calculate the gradient of each phase - output is list of gradx and grady
    grad1 = np.gradient(phase1)
    grad2 = np.gradient(phase2)

    # Calculate gradient direction
    graddir1 = np.arctan2(grad1[1], grad1[0])
    graddir2 = np.arctan2(grad2[1], grad2[0])

    # Calculate phase difference
    vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))
    sign_map = np.sin(np.angle(vdiff))

    return sign_map

def segment_map(sign_map):
    '''Given a sign map, segment into different areas'''
    sign = median_filter(sign_map, 15)

    # Calculate cutoff
    cutoff = 1.5*np.nanstd(sign)
    # Treshold
    sign_thresh = np.zeros_like(sign)
    sign_thresh[sign > cutoff] = 1
    sign_thresh[sign < -cutoff] = -1

    # Remove noise
    sign_thresh = binary_opening(np.abs(sign_thresh), iterations=1).astype(np.int)

    # Identify patches
    patches, patch_i = label(sign_thresh)

    # Close each region
    patch_map = np.zeros_like(patches)
    for i in range(patch_i):
        curr_patch = np.zeros_like(patches)
        curr_patch[patches == i+1] = 1
        patch_map += binary_erosion(binary_closing(curr_patch, iterations=1), iterations=2).astype(np.int) * (i+1)

    return patch_map

if __name__ == '__main__':
    # Filepaths and parameters
    savepath = r'F:\DeepMouse\RT_117\wdf\20230802\wdf'
    fps = [r'F:\DeepMouse\RT_117\wdf\20230802\wdf\wdf_000_a11.mj2',
           r'F:\DeepMouse\RT_117\wdf\20230802\wdf\wdf_000_a12.mj2',
           r'F:\DeepMouse\RT_117\wdf\20230802\wdf\wdf_000_e11.mj2',
           r'F:\DeepMouse\RT_117\wdf\20230802\wdf\wdf_000_e12.mj2']    # a11, a12, e11, e12
    fr = 10
    nrep = 10
    timetot = [280,280,170,170]

    # For each video, load and calculate the fft map
    ffts = [get_fft(fvid, fvid+r'_events', timetot[i], nrep, fr) for i,fvid in enumerate(fps)]

    # Combine maps
    phase_a, amp_a = combine_maps(ffts[0], ffts[1])
    phase_e, amp_e = combine_maps(ffts[2], ffts[3])

    # Get sign map
    sign_map = get_sign_map(phase_a, phase_e)

    # Get patches
    patch_map = segment_map(sign_map)

    np.save(os.path.join(savepath, 'sign_map.npy'), sign_map)
    np.save(os.path.join(savepath, 'patch_map.npy'), patch_map)

    fig, ax = pl.subplots()
    ax.imshow(sign_map, cmap='bwr')
    ax.axis('off')
    fig.savefig(os.path.join(savepath, 'sign_map.png'), bbox_inches='tight')