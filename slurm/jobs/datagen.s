#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1
#SBATCH --mem=200GB
#SBATCH --job-name=datagen
#SBATCH --output=/scratch/cg3306/climate/subgrid/gz21/slurm/echo/datagen_%A_%a.out
#SBATCH --error=/scratch/cg3306/climate/subgrid/gz21/slurm/echo/datagen_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
echo "$(date)"
module purge
singularity exec --nv --overlay /scratch/cg3306/climate/subgrid/gz21/overlay-15GB-500K.ext3:ro\
	 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "
		source /ext3/env.sh;
		mlflow run -e data . --env-manager local --experiment-name data --run-name full;
	"
echo "$(date)"