from gz21.data.pangeo_catalog import get_patch_from_file
from gz21.data.coarse import eddy_forcing
# from gz21.data.utils import cyclize_dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import torch
from gz21.paths import LANDMASKS
import os

def expand_for_cnn_spread(land_mask:xr.DataArray,cnn_field_of_view:int):
    ones_sig = np.ones((cnn_field_of_view,cnn_field_of_view))
    nplandmask = land_mask.values.squeeze()
    nplandmask = convolve2d(nplandmask,ones_sig,boundary='wrap',mode = 'same')
    nplandmask = np.where(nplandmask>0,1,0)
    # spread = (cnn_field_of_view - 1)//2
    # if spread == 0:
    #     return land_mask
    # land_mask = land_mask.copy().isel(xu_ocean = slice(spread,-spread),yu_ocean = slice(spread,-spread))
    land_mask.data = nplandmask.reshape(land_mask.shape)
    return land_mask



class CoarseGridLandMask:
    def __init__(self,factor:int = 4,cnn_field_of_view:int = 21,torch_flag:bool = True) -> None:
        self.factor =factor
        self.cnn_field_of_view = cnn_field_of_view
        self._interior_land_mask = None
        self._land_mask = None
        hsint = str(abs(hash((factor,cnn_field_of_view))))
        self._memory_location = os.path.join(LANDMASKS, hsint + '.nc')
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_flag =torch_flag
    def generate_masks(self,):
        patch_data, grid_data = get_patch_from_file(1,None,0,'usurf', 'vsurf') 
        # patch_data = cyclize_dataset(patch_data, 'xu_ocean', self.factor)
        # grid_data = cyclize_dataset(grid_data, 'xu_ocean', self.factor)
        patch_data = xr.where(np.isnan(patch_data), 1,0)
        patch_data = patch_data.load()
        grid_data = grid_data.load()
        forcing = eddy_forcing(patch_data, grid_data, scale=self.factor)
        forcing = np.abs(forcing)
        interior_land_mask = forcing.S_x + forcing.S_y + forcing.usurf + forcing.vsurf
        interior_land_mask = xr.where(interior_land_mask>0,1,0)
        _interior_land_mask = 1 - expand_for_cnn_spread(interior_land_mask,self.cnn_field_of_view)
        
        usurf = patch_data.usurf
        usurf = usurf.coarsen({'xu_ocean': int(self.factor),'yu_ocean': int(self.factor)},boundary='trim')
        land_density = usurf.mean()
        land_mask = xr.where(land_density >= 0.5,1,0)
        _land_mask= 1 - expand_for_cnn_spread(land_mask,self.cnn_field_of_view)
        _interior_land_mask.name = 'interior'
        _land_mask.name = 'default'
        masks = xr.merge([_interior_land_mask,_land_mask])
        masks = masks.isel(time = 0)
        return masks
    def save_to_file(self,):
        masks = self.generate_masks()
        masks.to_netcdf(self._memory_location)
    def read_from_file(self,):
        return xr.open_dataset(self._memory_location)
    @property
    def interior_land_mask(self,):
        if self._interior_land_mask is None:
            masks = self.read_from_file()
            self._interior_land_mask = masks.interior
            if self.torch_flag:
                vals = self._interior_land_mask.values.squeeze()
                vals = np.stack([vals],axis = 0)                
                self._interior_land_mask = torch.from_numpy(vals).to(dtype = torch.float32)
        return self._interior_land_mask
    @property
    def land_mask(self,):
        if self._land_mask is None:
            masks = self.read_from_file()
            self._land_mask = masks.default
            if self.torch_flag:
                vals = self._land_mask.values.squeeze()
                vals = np.stack([vals],axis = 0)
                self._land_mask = torch.from_numpy(vals).to(dtype = torch.float32)
        return self._land_mask



def main():
    cglm = CoarseGridLandMask(torch_flag=False,cnn_field_of_view=1)
    cglm.save_to_file()
    
    cglm.interior_land_mask.plot()
    plt.savefig('coarse_interior_land_mask.png')
    plt.close()

    cglm.land_mask.plot()
    plt.savefig('coarse_land_mask.png')
    plt.close()
    
    cglm = CoarseGridLandMask(torch_flag=False,cnn_field_of_view=21)
    cglm.save_to_file()
    
    cglm.interior_land_mask.plot()
    plt.savefig('coarse_interior_land_mask_expanded.png')
    plt.close()

    cglm.land_mask.plot()
    plt.savefig('coarse_land_mask_expanded.png')
    plt.close()
    
if __name__ == '__main__':
    main()