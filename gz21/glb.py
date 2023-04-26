from gz21.paths import OUTPUTS
import matplotlib.pyplot as plt
import xarray as xr
import os
import mlflow

def secure_plots_dir():
    PLOTS = os.path.join(OUTPUTS,'plots')
    if not os.path.exists(PLOTS):
        os.makedirs(PLOTS)
    return PLOTS
def main():
    experiment = mlflow.get_experiment("0")
    plots_dir = secure_plots_dir()
    print("Artifact Location: {}".format(experiment.artifact_location))

    
        


if __name__ == '__main__':
    main()