import numpy as np
import pandas as pd
import io
import h2o
import json
from h2o.automl import H2OAutoML, get_leaderboard


from fastapi import FastAPI, File, Form, UploadFile, Request, Body
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import random


import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


from google.cloud import storage
from sklearn.metrics import accuracy_score

app = FastAPI()
global_leaderboard = None
model = None
# configure CORS middleware
origins = ["*"]  # Replace * with your specific domain if you don't want to allow all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initiate H2O instance and MLflow client
h2o.init()
TRACKING_SERVER_HOST = "https://mlflow-server-6r72er7ega-uc.a.run.app" # fill in with the external IP of the compute engine instance
mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
client = MlflowClient(TRACKING_SERVER_HOST)


@app.post("/save_models")
async def save_models(model_ids: list[str] = Body(...)):
    global global_leaderboard

    if global_leaderboard is None:
        return {"message": "No leaderboard available"}

    # Create MLflow experiment
    experiment_name = 'deneme' + str(random.randint(1, 1000000))
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

    for model_id in model_ids:

        with mlflow.start_run():
            model = h2o.get_model(model_id)

            # Log and save best model (mlflow.h2o provides API for logging & loading H2O models)
            mlflow.h2o.log_model(model, artifact_path='model')
            model_uri = mlflow.get_artifact_uri('model')
            print(f'model saved in {model_uri}')

        # Perform any action you want with the selected models
        # For example, you can save the models to a file or database
    return {"message": model_uri}

@app.get("/run_names")
async def get_run_names():
    all_exps = [exp.experiment_id for exp in client.search_experiments()]
    runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ACTIVE_ONLY)
    runs = runs[runs['status'] != 'FAILED']
    return runs['tags.mlflow.runName'].values.tolist()

@app.post("/run_info")
async def get_run_info(run_name: str = Body(...)):
    all_exps = [exp.experiment_id for exp in client.search_experiments()]
    run_info = client.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ACTIVE_ONLY, filter_string=f"tags.mlflow.runName = '{run_name}'")[0]
    return json.dumps(run_info.to_dictionary())

@app.post("/predict")
async def predict(request: Request):
    form_data = await request.form()
    file = form_data["file"].file
    file_obj = io.BytesIO(file.read())
    predict_df = pd.read_csv(file_obj)
    predict_frame = h2o.H2OFrame(predict_df)

    all_exps = [exp.experiment_id for exp in client.search_experiments()]
    runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
    # run_id, exp_id = runs.loc[runs['metrics.log_loss'].idxmax()]['run_id'], runs.loc[runs['metrics.log_loss'].idxmax()]['experiment_id']
    run_id, exp_id, artifact_uri = runs.loc[0]['run_id'], runs.loc[0]['experiment_id'], runs.loc[0]['artifact_uri']
    print(f'Loading best model: Run {run_id} of Experiment {exp_id}')

    last_model = mlflow.h2o.load_model(artifact_uri+'/model')

    predictions = last_model.predict(predict_frame).as_data_frame()['predict'].values
    json_compatible_item_data = jsonable_encoder(predictions.tolist())
    output = JSONResponse(content=json_compatible_item_data)
    print(output)
    return output

@app.post("/train")
async def train(request: Request):
    form_data = await request.form()
    file = form_data["file"].file
    trainString = form_data.get("target_string")
    max_runtime_secs = int(form_data.get("max_runtime_secs") or 300)
    max_models = int(form_data.get("max_models") or 6)
    # if(max_models == 1):
    #     max_models = 2
    nfolds = int(form_data.get("nfolds") or 5)
    seed = int(form_data.get("seed") or 42)
    include_algos = form_data.get("include_algos")
    print(include_algos)
    if include_algos:
        include_algos = json.loads(include_algos)
    else:
        include_algos = None
    print("PARAMS ",trainString, max_runtime_secs, max_models, nfolds, seed, include_algos)

    file_obj = io.BytesIO(file.read())
    train_df = pd.read_csv(file_obj)
    main_frame = h2o.H2OFrame(train_df)

    y = trainString
    x = [n for n in main_frame.col_names if n != y]

    
    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=max_models,
        nfolds=nfolds,
        seed=seed,
        include_algos=include_algos
    )
    aml.train(x=x, y=y, training_frame=main_frame)

    

    lb = aml.leaderboard
    global global_leaderboard
    global_leaderboard = aml.leaderboard
    lb.head(rows=lb.nrows)

    response =  aml.leaderboard.as_data_frame(use_pandas=True).to_json()
    print(response)
    print(type(response))
    return response

@app.post("/unsupervised-train")
async def train(request: Request):
    form_data = await request.form()
    file = form_data["file"].file
    algo = form_data.get("algo")
    predictors = [col for col in file.col_names]
    if algo == "kmeans":
        kmeans = h2o.estimators.H2OKMeansEstimator(
            k=int(form_data.get("number_of_clusters") or 5),          # number of clusters
            standardize=True,  # standardize the data before clustering
            seed = int(form_data.get("seed") or 42)      # set the seed for reproducibility
        )
        kmeans.train(x=predictors, training_frame=file)
    elif algo == "iforest":
        iforest = h2o.estimators.H2OIsolationForestEstimator(
            ntrees=int(form_data.get("ntrees") or 100),          # number of trees in the forest
            max_depth=int(form_data.get("max_depth") or 20),        # maximum depth of each tree
            sample_size=int(form_data.get("sample_size") or 256),     # size of the sample used to train each tree
            seed=int(form_data.get("seed") or 42)           # set the seed for reproducibility
        )
        iforest.train(x=predictors, training_frame=file)

    experiment_name = 'deneme' + str(random.randint(1, 1000000))
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


    with mlflow.start_run():
        if algo == "kmeans":
            model = h2o.get_model(kmeans.model_id)
        elif algo == "iforest":
            model = h2o.get_model(iforest.model_id)
        # Log and save best model (mlflow.h2o provides API for logging & loading H2O models)
        mlflow.h2o.log_model(model, artifact_path='model')
        model_uri = mlflow.get_artifact_uri('model')
        print(f'model saved in {model_uri}')
    
    return {"message": model_uri}


@app.post("/dev-supervised-train")
async def train(request: Request):
    form_data = await request.form()
    file = form_data["file"].file
    algo = form_data.get("algo")
    response = form_data.get("response")
    predictors = [col for col in file.col_names if col != response]

    # Split the data into training and validation sets
    train, valid = file.split_frame(ratios=[0.7], seed=1234)

    if algo == "gbm":
        gbm = h2o.estimators.H2OGradientBoostingEstimator(
            ntrees=int(form_data.get("ntrees") or 100),          # number of trees in the forest
            max_depth=int(form_data.get("max_depth") or 20),         # maximum depth of each tree
            learn_rate=float(form_data.get("learn_rate") or 0.1),      # learning rate of the algorithm
            seed=int(form_data.get("seed") or 42)                # set the seed for reproducibility
        )
        gbm.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    elif algo == "glm":
        # Create a GLM model
        glm = h2o.estimators.H2OGeneralizedLinearEstimator(
            family=str(form_data.get("family") or "binomial"),   # use binomial family for binary classification
            alpha=float(form_data.get("alpha") or 0.5),           # specify the alpha regularization parameter
            lambda_=float(form_data.get("lambda") or 1e-5),        # specify the lambda regularization parameter
            seed=int(form_data.get("seed") or 42)            # set the seed for reproducibility
        )
        # Train the GLM model on the training set
        glm.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    elif algo == "xgb":
        # Create an XGBoost model
        xgb = h2o.estimators.H2OXGBoostEstimator(
            ntrees=int(form_data.get("ntrees") or 100),          # number of trees in the forest
            max_depth=int(form_data.get("max_depth") or 5),         # maximum depth of each tree
            learn_rate=float(form_data.get("learn_rate") or 0.1),      # learning rate of the algorithm
            seed=int(form_data.get("seed") or 42)         # set the seed for reproducibility
        )
        # Train the XGBoost model on the training set
        xgb.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
    elif algo == "rf":
        # Create a Random Forest model
        rf = h2o.estimators.H2ORandomForestEstimator(
            ntrees=int(form_data.get("ntrees") or 100),          # number of trees in the forest
            max_depth=int(form_data.get("max_depth") or 5),         # maximum depth of each tree
            min_rows=int(form_data.get("min_rows") or 10),         # maximum depth of each tree      # specify the minimum number of rows in each leaf node
            seed=int(form_data.get("seed") or 42)         # set the seed for reproducibility
        )
        # Train the Random Forest model on the training set
        rf.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

    experiment_name = 'deneme' + str(random.randint(1, 1000000))
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


    with mlflow.start_run():
        if algo == "gbm":
            model = h2o.get_model(gbm.model_id)
        elif algo == "glm":
            model = h2o.get_model(glm.model_id)
        elif algo == "xgb":
            model = xgb.get_model(xgb.model_id)
        elif algo == "rf":
            model = h2o.get_model(rf.model_id)
        # Log and save best model (mlflow.h2o provides API for logging & loading H2O models)
        mlflow.h2o.log_model(model, artifact_path='model')
        model_uri = mlflow.get_artifact_uri('model')
        print(f'model saved in {model_uri}')
    
    return {"message": model_uri}

@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to the End to End AutoML Pipeline Project</h2>
    </body>
    """
    return HTMLResponse(content=content)
