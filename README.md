# How to test the build
Build a singularity and install all the dependencies inside a conda environment `pangeo`.
This also creates a file `gz21/root.txt` to personalize the paths.
```
make setup-greene
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

Employ `pangeo` environment by using the command line
```
source /ext3/env.sh
conda activate pangeo
```
It should look like `(pangeo) Singularity> `. Use `mlflow` with the processes defined in `gz21/MLproject`

For a test, the following will coarse-grain a couple time instances of high resolution data and save the created coarse-grid variables into `temp` under a random folder name.

```
mlflow run code --env-manager local
```