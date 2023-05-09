import matplotlib.pyplot as plt
import torch
def main():
    for t in range(7):
        root = f'train_interrupt_{t}.pth'
        x = torch.load(root,map_location = "cpu")
        ins,outs = x['x'],x['y']
        ins = ins.detach().numpy()
        nrows = ins.shape[0]
        ncols = ins.shape[1]
        fig,axs = plt.subplots(nrows,ncols,figsize = (24,24))
        for i in range(nrows):
            for j in range(ncols):
                ax = axs[i,j]
                ax.imshow(ins[i,j])
        fig.savefig(f'interrupt_{t}.png')
        plt.close()
        
if __name__ == '__main__':
    main()