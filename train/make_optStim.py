'''
Generate optimal stimuli by first aligning receptive field centres
'''

import argparse
from rudi_utils import *

def optStimLucent(DNN,readout,neurons,inp_img_size=135, device='cpu'):
    # breakpoint()
    readout_simple = ReadOut(readout_model=readout)  # TODO: Are ReadOut and DNN_readout_combined needed?
    combined_model = DNN_readout_combined(DNN,readout_simple).to(device).eval()
    # param_f = lambda:param.image(inp_img_size,batch=len(neurons),fft=True,decorrelate=False)
    param_f = lambda:param.image(inp_img_size,batch=len(neurons),fft=True,decorrelate=False,channels=1)
    # TRIED with channels = 3 --> noisier than without setting channels=3 --> very similar to when decorrelate=False
    # removed pad from transform --> scale of features reduces
    # tot_objective = None
    # for idx,neuron_idx in enumerate(neurons):
    # 	if tot_objective is not None:
    # 		tot_objective += objectives.channel("readout_fc",neuron_idx,batch=idx)
    # 	else:
    # 		tot_objective = objectives.channel("readout_fc",neuron_idx,batch=idx)
    # optimizer_vis = lambda params: torch.optim.Adam(params, 2e-3)		# for V1
    optimizer_vis = lambda params: torch.optim.Adam(params, 3e-3)		# for LM
    # optimizer_vis = lambda params: torch.optim.Adam(params, 5e-2)		# fastLR --> default param

    # tot_objective = reduce(lambda x,y: x+objectives.channel("readout_fc",y[0],batch=y[1]),list(zip(neurons,np.arange(len(neurons)))),0)
    tot_objective = sum([objectives.channel("readout_fc",neuron_idx,batch=idx) for idx,neuron_idx in enumerate(neurons)])
    # transforms = [transform.pad(4),transform.jitter(8),transform.random_rotate(list(range(-10,10))),transform.jitter(4),grayscale_tfo(3)]	#  + 2*list(range(-5,5)) + 5*list(range(-2,2))
    transforms = [transform.pad(4),transform.jitter(8),transform.random_rotate(list(range(-10,10))),transform.jitter(4),grayscale_tfo1(3)]	#  + 2*list(range(-5,5)) + 5*list(range(-2,2))
    imgs = render.render_vis(combined_model,tot_objective,optimizer=optimizer_vis,param_f=param_f,preprocess=False,fixed_image_size=inp_img_size,show_image=False,transforms=transforms) 	# transforms
    # debug_plot(imgs[0],'optStim_optim_slower_lessScaling_rotateMore')
    # breakpoint()
    return imgs[0].transpose([0,3,1,2]) 	# return as (len(neurons),3,135,135)


def main(fp_conv, fp_readout, batch_size=64, device='cpu', model_class='shallowConv_4', layer=16, 
         normalize=True, bias=True, verbose=True, savepath=None, N=10, ignore_RF=False):
    '''Make optimal stimuli'''

    # Construct DNN and FactorisedReadout modules
    DNN, readout = load_DNN(fp_conv, fp_readout, device, model_class, layer, normalize=normalize, bias=bias)
    if verbose: print("Dataset and models loaded.")
    
    # Calculate the average receptive field, align to centre, and set all neurons to same RF
    spatial = readout.spatial.detach().cpu().numpy()
    spatial_shifted, spatial_normalised, coms, coms_shifted = get_average_RF(np.squeeze(spatial), N=N)
    # Overwrite spatial mask
    if ignore_RF:
        rf = np.median(spatial_shifted, axis=0)
        rf = rf/np.sqrt((rf**2).sum())
        rf = np.abs(rf)
        rf = np.repeat(rf[None,:,:], repeats=spatial.shape[0], axis=0)
        readout.spatial = nn.Parameter(torch.from_numpy(rf.reshape(spatial.shape)))
    else:
        readout.spatial = nn.Parameter(torch.from_numpy(spatial_shifted.reshape(spatial.shape)))

    # Construct full model
    combined_model = DNN_readout_combined(DNN, ReadOut(readout_model=readout)).to(device).eval()
    if verbose: print("Neurons aligned")

    # Run lucent in batches
    all_ims = []
    idx = np.arange(spatial.shape[0])
    for i in range(spatial.shape[0]//batch_size + 1):
        start_idx = i*batch_size
        end_idx = min([(i+1)*batch_size, spatial.shape[0]])
        all_ims.append(optStimLucent(DNN, readout, idx[start_idx:end_idx], inp_img_size=135, device=device))

    all_ims = np.concatenate(all_ims, axis=0)
    all_ims = (all_ims-all_ims.min())/(all_ims.max()-all_ims.min())

    np.save(os.path.join(savepath, 'optStim.npy'), all_ims)

    return all_ims


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate activation atlas")
    parser.add_argument("--model_path", type=str, default=r"D:\Trenholm Lab\Rudi\Ronan_experiment\Models\combined\V1",
                    help='Path to directory with model weights')
    parser.add_argument("--save_path", type=str, default=r"None",
                    help='Path to save directory')
    parser.add_argument("--model", type=str, default="shallowConv_4", help="Pretrained model architecture")
    parser.add_argument("--layer", type=int, default=16, help="depth of DNN backbone to use for feature extraction")
    parser.add_argument("--bias", type=str, default="True", help="use bias for readout")
    parser.add_argument("--normalize", type=str, default="True", help="normalize spatial weights")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for image generation")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--N", type=int, default=10, help="Iterations for RF alignment")
    parser.add_argument("--ignore_RF", type=str, default='False', help="If true, use same RF for every neuron")

    # Get parameters
    tmp = parser.parse_args()
    args = tmp.__dict__
    print(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    model_path = args["model_path"]
    save_path = model_path if args['save_path'] == 'None' else args['save_path']
    model = args["model"]
    layer = args['layer']
    bias = True if args['bias']=='True' else False
    normalize = True if args['normalize']=='True' else False
    batch_size = args['batch_size']
    N = args['N']
    ignore_RF = True if args['ignore_RF']=='True' else False


    # Run
    fp_conv = glob.glob(os.path.join(model_path, '*DNN.pt'))[0]
    fp_readout = glob.glob(os.path.join(model_path, '*FactorizedReadout.pt'))[0]

    main(fp_conv=fp_conv, fp_readout=fp_readout, batch_size=batch_size, device=device, model_class=model, layer=layer, 
         normalize=normalize, bias=bias, verbose=True, savepath=save_path, N=N, ignore_RF=ignore_RF)
