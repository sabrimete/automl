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
global_leaderboard = None
model_path = None
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

@app.post("/save_models")
async def save_models(model_ids: list[str] = Body(...)):
    global global_leaderboard
    global model_path
    if global_leaderboard is None:
        return {"message": "No leaderboard available"}
    for model_id in model_ids:
        model = h2o.get_model(model_id)
        model_path = h2o.save_model(model = model, path ='sample_data/', force = True)

        # Perform any action you want with the selected models
        # For example, you can save the models to a file or database
    return {"message": model_path}


@app.post("/predict")
async def predict(request: Request):
    global model_path
    form_data = await request.form()
    file = form_data["file"].file
    file_obj = io.BytesIO(file.read())
    predict_df = pd.read_csv(file_obj)
    predict_frame = h2o.H2OFrame(predict_df)


    # run_id, exp_id = runs.loc[runs['metrics.log_loss'].idxmax()]['run_id'], runs.loc[runs['metrics.log_loss'].idxmax()]['experiment_id']
    last_model = h2o.load_model(model_path)

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
    max_runtime_secs = int(form_data.get("max_runtime_secs") or 1000)
    max_models = int(form_data.get("max_models") or 6)
    # if(max_models == 1):
    #     max_models = 2
    nfolds = int(form_data.get("nfolds") or 5)
    seed = int(form_data.get("seed") or 42)
    include_algos = form_data.get("include_algos")
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

    if(include_algos is None):
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            nfolds=nfolds,
            seed=seed
        )
    else:
        aml = H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            nfolds=nfolds,
            seed=seed,
            include_algos=include_algos
        )
    aml.train(x=x, y=y, training_frame=main_frame)

    

    lb = get_leaderboard(aml, extra_columns='ALL')
    global global_leaderboard
    global_leaderboard = lb
    lb.head(rows=lb.nrows)

    response =  lb.as_data_frame(use_pandas=True).to_json()
    print(response)
    print(type(response))
    return response

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
    return HTMLResponse(content=content)
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

