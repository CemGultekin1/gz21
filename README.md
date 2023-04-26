# How to test the build
Build a singularity and install all the dependencies inside a conda environment `pangeo`.
This also creates a file `code/root.txt` to personalize the paths.
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
You need to see `Singularity> ` if you are now inside the singularity. We can now employ `pangeo` 
environment using
```
source /ext3/env.sh
conda activate pangeo
```
It should look like `(pangeo) Singularity> `. We can now use `mlflow` with the processes defined in `code/MLproject`

For a test, the following will coarse-grain high resolution data and save coarse-grid variables into `temp`.

```
mlflow run code --env-manager local
```