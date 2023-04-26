import tempfile
import matplotlib.pyplot as plt
import xarray as xr
import os
from os.path import join
import mlflow
from mlflow.tracking import MlflowClient

def main():
    client = MlflowClient()

    # Retrieve Experiment information
    experiment = client.get_experiment_by_name('data_generation')
    runs = {run.info.run_name:run for run in client.search_runs(experiment.experiment_id)}
    run = runs['test']
    data_address = run.info.artifact_uri

    path = os.path.join(data_address,'forcing')
    ds = xr.open_zarr(path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        temploc = os.path.join(tmpdirname,'peek')
        os.makedirs(temploc)
        for key in ds.data_vars.keys():
            ds[key].isel(time = 0).plot()
            plt.savefig(join(temploc,f'{key}.png'))
            plt.close()
        mlflow.log_artifact(temploc)
        


if __name__ == '__main__':
    main()