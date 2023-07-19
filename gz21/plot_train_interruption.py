import matplotlib.pyplot as plt
import torch
import itertools
import numpy as np
def main():
    filename = 'train_interrupt_0_.pth'
    tip = torch.load(filename)
    inp = tip['true_result'].detach().cpu().numpy()
    shps = inp.shape[:2]
    rngs = [range(shp) for shp in shps]
    fig,axs = plt.subplots(shps[0],shps[1],figsize = (shps[1]*6,shps[0]*4))
    for i,j in itertools.product(*rngs):
        img = inp[i,j]
        axs[i,j].imshow(np.log10(np.abs(img[::-1])))
    fig.savefig('train_interruption.png')
if __name__ == '__main__':
    main()