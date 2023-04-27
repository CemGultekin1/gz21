import os
def read_made_dictionary():
    file1 = open('root.txt', 'r')
    lines = file1.readlines()
    file1.close()
    files = {}
    for line in lines:
        line = line.strip()
        key,value = [x.strip() for x in line.split('=')]
        files[key] = value
    return files
files = read_made_dictionary()
SLURM_JOBS = os.path.abspath('slurm/jobs')
SLURM_ECHO = os.path.abspath('slurm/echo')
TEMP = os.path.abspath('temp')
for directory in [TEMP,SLURM_ECHO,SLURM_JOBS]:
    if not os.path.exists(directory):
        os.makedirs(directory)
    

EXT3 = os.path.abspath(files['EXTFILE'])
CUDA_SINGULARITY = os.path.abspath(files['CUDA_SINGULARITY'])

CM2P6_SURFACE_UVT = '/scratch/as15415/Data/CM26_Surface_UVT.zarr'
LAB_CM2P6_PATH = '/scratch/zanna/data/cm2.6'
GRID_DATA = os.path.join(LAB_CM2P6_PATH,'GFDL_CM2_6_grid.nc')
CM2P6_SURFACE_1PCT_CO2_UVT = os.path.join(LAB_CM2P6_PATH,'surface_1pct_co2.zarr')


