import mlflow

def workflow():
    with mlflow.start_run() as active_run:
            mlflow.run(".", "data", env_manager='local')
if __name__ == '__main__':
    workflow()  