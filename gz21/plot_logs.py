import matplotlib.pyplot as plt




def main():
    # paths = [
    #     '/scratch/cg3306/climate/subgrid/gz21/slurm/echo/train_32889750_1.out',
    #     '/scratch/cg3306/climate/temp/gz21/slurm/echo/train_33168641_1.out',
    #     '/scratch/cg3306/climate/temp/gz21/slurm/echo/train_33199949_1.out',
    #     '/scratch/cg3306/climate/temp/gz21/slurm/echo/train_33199951_1.out',
    #     '/scratch/cg3306/climate/temp/gz21/slurm/echo/train_33199952_1.out',
        
    # ]
    
    # paths = [
    #     '/scratch/cg3306/climate/temp/gz21_new/gz21/slurm/echo/train_33199965_1.out',
    #     '/scratch/cg3306/climate/temp/gz21_new/gz21/slurm/echo/train_33199964_1.out',
    #     '/scratch/cg3306/climate/temp/gz21_new/gz21/slurm/echo/train_33199939_1.out',
    # ]
    # paths = [        
    #     '/scratch/cg3306/climate/temp/gz21_new/gz21/slurm/echo/train_33199964_1.out',
    #     '/scratch/cg3306/climate/temp/gz21_new/gz21/slurm/echo/train_33199965_1.out',
    #     '/scratch/cg3306/climate/temp/gz21_new/gz21/slurm/echo/train_33199939_1.out'
    # ]

    paths = [        
        '/scratch/cz3056/CNN_train/Arthur_model/gz21/slurm/echo/gtrain_35529888_1.out',
        '/scratch/cz3056/CNN_train/Arthur_model/gz21/slurm/echo/gtrain_35625547_1.out'
    ]
    

    # labels = [
    #     'cem_20230511_1_four_regions_fixed_branch',   
    #     'cem_20230512_four_regions_fixed_branch',
    #     'cem_20230512_1_four_regions_fixed_branch',         
    # ]

    labels = [
        'global_landmasks_1', 
        'global_landmasks_2'        
    ]
    fig,axs = plt.subplots(len(paths),3,figsize = (20,30))
    titles = [
        'running_avg_trainloss',
        'trainloss',
        'testloss',
    ]
    xlabels = [
        'per 20 updates',
        'epochs',
        'epochs'
    ]
    for i in range(3):
        for j,(path,label) in enumerate(zip(paths,labels)):
            log = read_log(path)
            
            v1 = [x for x in zip(*log)][i]
            v1 = [x for x in v1 if x is not None]
            axs[j,i].plot(v1,label = label)
            axs[j,i].grid()
            axs[j,i].set_title(f"{label} - {titles[i]}")
    fig.savefig('logs_global.png')
        # continue
        # log1 = read_log(paths[0])
        # log2 = read_log(paths[1])
        # v1 = [x for x in zip(*log1)][i]    
        # v2 = [x for x in zip(*log2)][i]    
        
        # v1 = [x for x in v1 if x is not None]
        # v2 = [x for x in v2 if x is not None]
        # # if len(v1) < len(v2):
        # #     v1 = v1 + [v1[-1]]*(len(v2) - len(v1))
        # # if len(v2) < len(v1):
        # #     v2 = v2 + [v2[-1]]*(len(v1) - len(v2))
        # diff = [abs(a - b) for a,b in zip(v2,v1)]
        
        # axs[1,i].semilogy(diff)
        # axs[1,i].set_xlabel(xlabels[i])
        # axs[1,i].set_title(titles[i] + ' difference - all values')
        # axs[0,i].semilogy(diff[:16],'*')
        # axs[0,i].set_xlabel(xlabels[i])
        # axs[0,i].set_title(titles[i] + ' difference - first 16 values')
    # fig.savefig('logs.png')
    
def read_log(path):
    with open(path,'r') as f:
        lines1 = f.readlines()
    
    logs_dict = []
    
    for i,line in enumerate(lines1):
        val = read_loss_val(line)
        if val is None:
            continue
        logs_dict.append(val)
    return logs_dict
def read_loss_val(line):
    line = line.strip()
    if 'Train loss for this epoch is' in line:
        train_loss_value = line.split()[-1]
        return None,float(train_loss_value),None
    if 'Test loss for this epoch is' in line:
        train_loss_value = line.split()[-1]
        return None,None,float(train_loss_value)
    if 'Loss value' not in line:
        return None
    lossval = line.split()[2].replace(',','')
    return float(lossval),None,None

if __name__ == '__main__':
    main()