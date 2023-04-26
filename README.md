# How to test the build
Build a singularity and install all the dependencies inside a conda environment "pangeo".
This also creates a file 'code/root.txt' to personalize the paths.
'make setup-greene'
To use the singularity in a read only mode
'make interactive-singularity-read-only'
Or to use it with writing permits
'make interactive-singularity-writing-permitted'
Once inside the singularity, employ 'pangeo' using
'source /ext3/env.sh'
'conda activate pangeo'