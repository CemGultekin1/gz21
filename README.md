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

The `mlflow` steps are defined in `MLproject`. For a test, the following will coarse-grain a couple time instances of high resolution data and save the created coarse-grid variables into `temp` under a random folder name.

```
mlflow run code --env-manager local
```