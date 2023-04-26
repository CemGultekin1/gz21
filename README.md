# How to test the build
Build a singularity and install all the dependencies into the default conda environment `base`
```
make
```
To use the singularity in a read only mode
```
make interactive-singularity-read-only
```
Or to use it with writing permits
```
make interactive-singularity-writing-permitted
```

You need to see `Singularity> ` if you are inside the singularity. In order to exit the singularity
use the command `exit`.

Once inside the singularity, use the follow command to load the environment.
```
source /ext3/env.sh
```

The `mlflow` steps are defined in `MLproject`. For a test, the following will coarse-grain a couple time instances of high resolution data and save the created coarse-grid variables as an mlflow artifact
into `mlruns/<experiment_id>/<run_id>/artifacts/forcing/`. 

```
mlflow run -e data-test . --env-manager local --experiment-name data_generation --run-name test
```

To access the generated data, the above entered params `experiment_name` and `run_name` can be used as provided in `peek.py`. 
Inside `MLproject`, `peek` is another entry point which plots `time=0` fields of the data. The plots are 
saved as artifacts similar to the data. 

```
mlflow run -e peek . --env-manager local --experiment-name data_generation --run-name peek
```

Creation of the slurm tasks are done with
```
mlflow run -e data-test . --env-manager local --experiment-name data --run-name test
```
The jobs are saved to `slurm/jobs` the outputs are directed to `slurm/echo`. 
To change the specifics of the `*.sbatch` files, edit `gz21/slurm/jobs.py`.