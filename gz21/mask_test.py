import torch
import numpy as np
import matplotlib.pyplot as plt
def main():
    pth = "train_interrupt_1.pth"
    tf = torch.load(pth,map_location = "cpu")
    # print({
    #     key:val.shape for key,val in tf.items()
    # })
    # return
    mask,output,input_ = tf['mask'].detach().numpy(),tf['output'].detach().numpy(),tf['input'].numpy()
    masked = output*mask
    # fig,axs = plt.subplots(1,3,figsize = (35,12))
    
    # axs[0].imshow(input_[0,0,::-1].squeeze())
    # axs[1].imshow(output[0,0,::-1].squeeze())
    # axs[2].imshow(masked[0,0,::-1].squeeze())
    # plt.savefig('output.png')
    # return
    
    for i in range(4):
        vals = masked[:,i]
        vals = np.sort(vals.reshape([-1]))
        dvals = vals[1:] - vals[:-1]
        plt.semilogy(dvals,'.')#imshow(vals)
        plt.savefig(f'masked_vals_{i}.png')
        plt.close()

if "__main__" == __name__:
    main()