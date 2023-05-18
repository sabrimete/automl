import numpy as np
import pandas as pd
import io
import h2o
import json
import uvicorn
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

@app.post("/predict")
async def predict(request: Request):
    form_data = await request.form()

    run_name = form_data["run_name"]

    file = form_data["file"].file
    file_obj = io.BytesIO(file.read())
    predict_df = pd.read_csv(file_obj)
    predict_frame = h2o.H2OFrame(predict_df)

    all_exps = [exp.experiment_id for exp in client.search_experiments()]
    run_info = client.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ACTIVE_ONLY, filter_string=f"tags.mlflow.runName = '{run_name}'")[0].to_dictionary()

    model = mlflow.h2o.load_model(run_info['info']['artifact_uri']+"/model")

    predictions = model.predict(predict_frame).as_data_frame()['predict'].values
    json_compatible_item_data = jsonable_encoder(predictions.tolist())
    output = JSONResponse(content=json_compatible_item_data)
    print(output)
    return output


@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to the Inference Server</h2>
    </body>
    """
    return HTMLResponse(content=content)
