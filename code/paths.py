import os
def read_root():
    with open('root.txt') as root_file:
        x = root_file.readlines()[0]
        x = x.strip()
    return x

ROOT = read_root()
OUTPUTS = os.path.join(ROOT,'outputs')
TEMP = os.path.join(ROOT,'temp')

for f in (OUTPUTS,TEMP):
    if not os.path.exists(f):
        os.makedirs(f)
        
CM2P6_SURFACE_UVT = '/scratch/as15415/Data/CM26_Surface_UVT.zarr'
LAB_CM2P6_PATH = '/scratch/zanna/data/cm2.6'
GRID_DATA = os.path.join(LAB_CM2P6_PATH,'GFDL_CM2_6_grid.nc')
CM2P6_SURFACE_1PCT_CO2_UVT = os.path.join(LAB_CM2P6_PATH,'surface_1pct_co2.zarr')


