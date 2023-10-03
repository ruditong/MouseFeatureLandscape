'''
train_SimCNE.py

Train SimCNE models on optimal stimuli for custom embedding.
'''

from utils import *
from models import *

def main(fp_ims, transformations, epochs, savepath, out_dim=256, lr=1e-3, device='cpu', temperature=1, d0=None):
    '''Train SimCNE'''
    clean_GPU()
    # Load dataset
    dataset = OptStims(fp_ims, n_views=2, do_transform=True, transformations=transformations)
    trainloader = DataLoader(dataset, batch_size=batch_size)

    # Construct model
    model = ResNetSimCLR(base_model='resnet18', out_dim=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0,
                                                           last_epoch=-1)

    dataset.do_transform = True
    with torch.cuda.device(0):
        simclr = SimCNE(model=model, optimizer=optimizer, scheduler=scheduler, device=device,  
                        n_views=2, epochs=epochs, temperature=temperature, d0=d0, savepath=savepath)
        
        loss = simclr.train(trainloader)

    fig, ax = pl.subplots()
    ax.plot(loss)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    
    return model, loss, fig


if __name__ == '__main__':
    regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
    savepath = r'D:\Data\DeepMouse\Results_raw\simcne\affine'
    fp_opt = r'D:\Data\DeepMouse\Results_raw\optstims\RF_aligned'
    fp_ims = [os.path.join(fp_opt, f'optStim_masked_{region}.npy') for region in regions]

    # Parameters
    batch_size = 7910//2
    epochs = 500
    out_dim = 256
    lr = 1e-3
    device='cuda:0'
    temperature=0.1
    d0=None

    # Transformations
    transformations = [transforms.Compose([
                                        transforms.RandomResizedCrop(size=32),
                                        transforms.RandomAffine((-90,90), translate=(0.1,0.1), fill=0.5),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomInvert(),
                                        transforms.GaussianBlur(11, sigma=(0.1,2)),
                                        ]),

                        transforms.Compose([
                                        transforms.RandomResizedCrop(size=32),
                                        #transforms.GaussianBlur(11, sigma=(0.1,2)),
                                        ]),

                        transforms.Compose([
                                        transforms.Resize(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation((-90,90)),
                                        #transforms.GaussianBlur(11, sigma=(0.1,2)),
                                        ]),
                                    
                        transforms.Compose([
                                        transforms.Resize(32),
                                        transforms.RandomAffine((0,0), translate=(0.1,0.1), fill=0.5),
                                        #transforms.GaussianBlur(11, sigma=(0.1,2)),
                                        ]),

                        transforms.Compose([
                                        transforms.RandomResizedCrop(size=32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation((-90,90)),
                                        #transforms.GaussianBlur(11, sigma=(0.1,2)),
                                        ]),

                        transforms.Compose([
                                        transforms.RandomResizedCrop(size=32),
                                        transforms.RandomAffine((0,0), translate=(0.1,0.1), fill=0.5),
                                        #transforms.GaussianBlur(11, sigma=(0.1,2)),
                                        ]),

                        transforms.Compose([
                                        transforms.Resize(32),
                                        transforms.RandomAffine((-90,90), translate=(0.1,0.1), fill=0.5),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        #transforms.GaussianBlur(11, sigma=(0.1,2)),
                                        ]),

                        transforms.Compose([
                                        transforms.RandomResizedCrop(size=32),
                                        transforms.RandomAffine((-90,90), translate=(0.1,0.1), fill=0.5),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        #transforms.GaussianBlur(11, sigma=(0.1,2)),
                                        ]),
    ]


    # Train model
    model, loss, fig = main(fp_ims, savepath=savepath, transformations=transformations[-1], epochs=epochs, out_dim=out_dim, 
                            lr=lr, device=device, temperature=temperature, d0=d0)
    
    torch.save(model.state_dict(), os.path.join(savepath, f'SimCNE_resnet18_outdim{out_dim}_temp{temperature}_epoch{epochs}_d0{d0}.pth'))
    np.save(os.path.join(savepath, 'SimCNE_loss.npy'), loss)
    fig.savefig(os.path.join(savepath, 'loss.png'))


