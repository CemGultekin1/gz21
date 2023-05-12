from gz21.data.utils import load_data_from_past
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

path = '/scratch/cg3306/climate/temp/gz21/temp/tmpj5ob8e89/forcing.zarr'
import xarray as xr
ds = xr.open_zarr(path)
print(ds)
ds1 = load_data_from_past()


t = 5
for key in ds.data_vars.keys():
    u0 = ds[key].isel(time =t)
    u1 = ds1[key].isel(time =t)

    relerr = -np.log10(np.abs(u0 - u1)/(np.abs(u0)+np.abs(u1))*2)
    relerr.plot(vmin = 0)
    plt.title(f"#digits of accuracy for {key}, at time = {u0.time.values.item()}")
    plt.savefig(f'{key}.png')
    plt.close()
