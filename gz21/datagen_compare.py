from gz21.data.utils import find_latest_data_run,load_data_from_past,load_data_from_run
import numpy as np
import matplotlib.pyplot as plt
from yaml import Loader, Dumper,load

def main():
    runs = find_latest_data_run()
    ds0 = load_data_from_run(runs['run_id'])
    ds1 = load_data_from_past()
    ds0 = ds0.sel(xu_ocean = slice(-280,80))
    with open('gz21/training_subdomains.yaml') as f:
        subdomains = load(f,Loader)
    
    for dom in range(4):
        print(f'domain = {dom}')
        subdomain_ = subdomains[dom][1]
        kwargs = dict(
            xu_ocean = slice(subdomain_['lon_min'],subdomain_['lon_max']),
            yu_ocean = slice(subdomain_['lat_min'],subdomain_['lat_max']),
        )
        ds1_ = ds1.sel(**kwargs).isel(time = 0)
        ds0_ = ds0.sel(**kwargs).isel(time = 0)
        err = ds0_ - ds1_
        logerr = np.log10(np.abs(err)/np.abs(ds0_))
        columns = [ds0_,ds1_,err,logerr]
        names = ['Arthur','Cem','Err','Log10RelErr']
        for key in logerr.keys():
            fig,axs = plt.subplots(1,len(names),figsize = (20,5))
            for coli,nm  in enumerate(names):
                columns[coli][key].plot(ax = axs[coli])
                axs[coli].set_title(nm)
            fig.suptitle(f'{key} on domain #{dom}')
            plt.savefig(f'images/{key}_{dom}.png')
            plt.close()
            

if __name__ == '__main__':
    main()