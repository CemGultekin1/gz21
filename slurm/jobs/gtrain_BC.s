#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --array=1
#SBATCH --mem=60GB
#SBATCH --job-name=gtrain
#SBATCH --output=/scratch/cz3056/CNN_train/Arthur_model/gz21_stencil_size/gz21/slurm/echo/gtrain_%A_%a.out
#SBATCH --error=/scratch/cz3056/CNN_train/Arthur_model/gz21_stencil_size/gz21/slurm/echo/gtrain_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
echo "$(date)"
module purge
singularity exec --nv --overlay /scratch/cz3056/CNN_train/Arthur_model/gz21/overlay-15GB-500K.ext3:ro\
	 /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif /bin/bash -c "
		source /ext3/env.sh;
		mlflow run -e global-train-FullyCNN_BC . --env-manager local --experiment-name gtrain --run-name full;
	"
echo "$(date)"