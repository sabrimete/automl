# ===========================
# Module: Backend setup (H2O, MLflowy)
# Author: Kenneth Leung
# Last Modified: 02 Jun 2022
# ===========================
# Command to execute script locally: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Command to run Docker image: docker run -d -p 8000:8000 <fastapi-app-name>:latest

import pandas as pd
import io
import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import json
import random

from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from utils.data_processing import match_col_types, separate_id_col

from sklearn.metrics import accuracy_score

# Create FastAPI instance
app = FastAPI()

# Initiate H2O instance and MLflow client
h2o.init()
client = MlflowClient()


# Create POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: bytes = File(...)):
    # Load best model (based on logloss) amongst all experiment runs
    all_exps = [exp.experiment_id for exp in client.list_experiments()]
    runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
    # run_id, exp_id = runs.loc[runs['metrics.log_loss'].idxmax()]['run_id'], runs.loc[runs['metrics.log_loss'].idxmax()]['experiment_id']
    run_id, exp_id = runs.loc[0]['run_id'], runs.loc[0]['experiment_id']
    print(f'Loading best model: Run {run_id} of Experiment {exp_id}')
    best_model = mlflow.h2o.load_model(f"mlruns/{exp_id}/{run_id}/artifacts/model/")

    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    test_h2o = h2o.H2OFrame(test_df)
    y_train = test_df['Response']
    # Separate ID column (if any)
    id_name, X_id, X_h2o = separate_id_col(test_h2o)

    # Match test set column types with train set
    X_h2o = match_col_types(X_h2o)

    # Generate predictions with best model (output is H2O frame)
    preds = best_model.predict(X_h2o)
    
    # Apply processing if dataset has ID column
    if id_name is not None:
        preds_list = preds.as_data_frame()['predict'].tolist()
        id_list = X_id.as_data_frame()[id_name].tolist()
        preds_final = dict(zip(id_list, preds_list))
    else:
        preds_final = preds.as_data_frame()['predict'].tolist()

    # Convert predictions into JSON format
    json_compatible_item_data = jsonable_encoder(preds_final)
    output = JSONResponse(content=json_compatible_item_data) 
    accuracy = accuracy_score(y_train, preds_final)
    return output, accuracy

@app.post("/train")
async def train(file: bytes = File(...)):

    # Get parsed experiment name
    experiment_name = 'deneme' + str(random.randint(1, 1000000))

    # Create MLflow experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = client.get_experiment_by_name(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name)
    
    mlflow.set_experiment(experiment_name)

    # Print experiment details
    print(f"Name: {experiment_name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")
    print(f"Tracking uri: {mlflow.get_tracking_uri()}")

    # Import data directly as H2O frame (default location is data/processed)
    file_obj = io.BytesIO(file)
    train_df = pd.read_csv(file_obj)
    main_frame = h2o.H2OFrame(train_df)

    # # Save column data types of H2O frame (for matching with test set during prediction)
    # with open('data/processed/train_col_types.json', 'w') as fp:
    #     json.dump(main_frame.types, fp)

    # Set predictor and target columns
    target = 'Response'
    predictors = [n for n in main_frame.col_names if n != target]

    # Factorize target variable so that autoML tackles classification problem
    main_frame[target] = main_frame[target].asfactor()

    # Setup and wrap AutoML training with MLflow
    with mlflow.start_run():
        aml = H2OAutoML(
                        max_models=1, # Run AutoML for n base models
                        seed=42, 
                        balance_classes=True, # Target classes imbalanced, so set this as True
                        sort_metric='logloss', # Sort models by logloss (metric for multi-classification)
                        verbosity='info', # Turn on verbose info
                        exclude_algos = ['GLM', 'DRF'], # Specify algorithms to exclude
                    )
        
        # Initiate AutoML training
        aml.train(x=predictors, y=target, training_frame=main_frame)
        
        # Set metrics to log
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("AUC", aml.leader.auc())
        
        # Log and save best model (mlflow.h2o provides API for logging & loading H2O models)
        mlflow.h2o.log_model(aml.leader, artifact_path="model")
        
        model_uri = mlflow.get_artifact_uri("model")
        print(f'AutoML best model saved in {model_uri}')
        
        # Get IDs of current experiment run
        exp_id = experiment.experiment_id
        run_id = mlflow.active_run().info.run_id
        
        # Save leaderboard as CSV
        lb = get_leaderboard(aml, extra_columns='ALL')
        lb_path = f'mlruns/{exp_id}/{run_id}/artifacts/model/leaderboard.csv'
        lb.as_data_frame().to_csv(lb_path, index=False) 
        print(f'AutoML Complete. Leaderboard saved in {lb_path}')
        print()
        print(aml.leaderboard.as_data_frame())
        print(type(aml.leaderboard.as_data_frame()))
        return aml.leaderboard.as_data_frame()

@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to the End to End AutoML Pipeline Project for Insurance Cross-Sell</h2>
    <p> The H2O model and FastAPI instances have been set up successfully </p>
    <p> You can view the FastAPI UI by heading to localhost:8000 </p>
    <p> Proceed to initialize the Streamlit UI (frontend/app.py) to submit prediction requests </p>
    </body>
    """
    # content = """
    # <body>
    # <form action="/predict/" enctype="multipart/form-data" method="post">
    # <input name="file" type="file" multiple>
    # <input type="submit">
    # </form>
    # </body>
    # """
    return HTMLResponse(content=content)