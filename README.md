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
into `mlruns/<experiment_id>/<run_id>/artifacts/forcing.zarr`. 

```
mlflow run -e data-test . --env-manager local --experiment-name data --run-name test
```
Above `--env-manager local` ensures the locally existing environment is used. `experiment_name` and `run_name` are arbitrary. 
But it might be useful to keep a convention to search for past runs. For a full run of data generation use
```
mlflow run -e data . --env-manager local --experiment-name data --run-name full
```

To access the generated data, the above entered params `experiment_name` and `run_name` can be used as provided in `peek.py`. 
Inside `MLproject`, `peek` is another entry point which plots `time=0` fields of the data. The plots are 
saved as artifacts similar to the data. 

```
mlflow run -e peek . --env-manager local --experiment-name data --run-name peek
```

Creation of the slurm tasks are done with

```
mlflow run -e slurm-jobs . --env-manager local --experiment-name slurm_jobs_gen
```

The jobs are saved to `slurm/jobs` the outputs are directed to `slurm/echo`. 
To change the specifics of the `*.sbatch` files, edit `gz21/slurm/jobs.py`.

To run a training job use 

```
mlflow run -e four-regions-train . --env-manager local --experiment-name train --run-name full
```

or you can use slurm. The following line can be run only from outside of the singularity.

```
sbatch slurm/jobs/r4train.s
```

All of these processes use `tempfile` library. `tempfile` opens randomly named folders inside the folder `temp`
to store things such as neural network weights or coarse-grained data during the execution.
However, currently the code does not clean `temp` after the execution is done. This is because
the processes take long to execute and may raise an error. When the proper `tempfile` system is used
the partial progress gets deleted automatically. This way it is possible see the partial process.
But don't forget to clean `temp` folder once in a while.

Updating the environment packages can be done through `requirements.txt`. 
If need to install new packages, enter the singularity with a writing permit and 
install the packages. Once it is done, if you want to keep the changes as a dependency 
on github update `requirements.txt` using the following line. Note that
`pipreqs` will only consider those libraries that are imported in the code.

```
python3 -m  pipreqs.pipreqs . --force
```