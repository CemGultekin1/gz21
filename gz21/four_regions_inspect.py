from gz21.data.utils import load_data_from_past
import matplotlib.pyplot as plt
import matplotlib
from gz21.paths import CM2P6_SURFACE_UVT
import numpy as np
ds = load_data_from_past()

xslices = [slice(-50, -20, None), slice(-180, -162, None), slice(-110, -92, None), slice(-48, -30, None)]
yslices = [slice(35, 50, None), slice(-40, -25, None), slice(-20, -5, None), slice(0, 15, None)]

size = (38, 45)

def crop(i):
    us = ds.usurf.isel(time = 0).sel(xu_ocean = xslices[i],yu_ocean = yslices[i])
    sep = np.array(us.shape) - np.array(size)
    leftsep = sep//2
    rightsep = sep - leftsep
    if rightsep[0]>0:
        us = us.isel(yu_ocean = slice(leftsep[0],-rightsep[0]))
    else:
        us = us.isel(yu_ocean = slice(leftsep[0],len(us.yu_ocean)))
    if rightsep[1]>0:
        us = us.isel(xu_ocean = slice(leftsep[1],-rightsep[1]))
    else:
        us = us.isel(xu_ocean = slice(leftsep[1],len(us.xu_ocean)))
    return us

cropped_xslices = []
cropped_yslices = []
for i in range(4):
    u = crop(i)
    cropped_xslices.append(
        slice(*u.xu_ocean.values[[0,-1]])
    )
    cropped_yslices.append(
        slice(*u.yu_ocean.values[[0,-1]])
    )

import xarray as xr
# masks = xr.open_dataset("/scratch/zanna/data/cm2.6/GFDL_CM2_6_grid.nc")
ds = xr.open_zarr(CM2P6_SURFACE_UVT)
fu = ds.usurf.isel(time = 100)
# print(ds)
# raise Exception
# fu = xr.where(np.isnan(fu), 1, fu )
# fu.plot()
# plt.savefig('global.png')

cmap = matplotlib.cm.bwr
cmap.set_bad('gray',1.)

fig,axs = plt.subplots(2,2,figsize = (20,20))
for i in range(4):
    ic = i%2
    ir = i//2
    ax = axs[ir,ic]
    subfu = fu.sel(
        xu_ocean = cropped_xslices[i],yu_ocean = cropped_yslices[i]
    )
    # subfu = xr.where(np.isnan(subfu),1,subfu)
    subfu.plot(ax = ax,cmap = cmap)
    ax.set_title(f"region#{i}")
fig.savefig('four_regions_after_cropping.png')
plt.close()
    
fig,axs = plt.subplots(2,2,figsize = (20,20))
for i in range(4):
    ic = i%2
    ir = i//2
    ax = axs[ir,ic]
    subfu = fu.sel(
        xu_ocean = xslices[i],yu_ocean = yslices[i]
    )
    # subfu = xr.where(np.isnan(subfu),1,subfu)
    subfu.plot(ax = ax,cmap = cmap)
    ax.set_title(f"region#{i}")
fig.savefig('four_regions_before_cropping.png')
plt.close()
    
