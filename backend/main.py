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


h2o.init()

@app.post("/save_models")
async def save_models(model_ids: list[str] = Body(...)):
    global global_leaderboard

    if global_leaderboard is None:
        return {"message": "No leaderboard available"}

    for model_id in model_ids:
        model = h2o.get_model(model_id)
        model_path = h2o.save_model(model=model, path='sample_model/', force=True)


        # Perform any action you want with the selected models
        # For example, you can save the models to a file or database
    return {"message": model_path}
    


@app.post("/predict")
async def predict(request: Request):
    form_data = await request.form()
    file = form_data["file"].file
    file_obj = io.BytesIO(file.read())
    predict_df = pd.read_csv(file_obj)
    predict_frame = h2o.H2OFrame(predict_df)

    # path = 'gs://deneme-bucket-1/model/XGBoost_1_AutoML_1_20230322_115208' 
    global model
    path = '/app/sample_model/' + model.model_id
    loaded_model = h2o.load_model(path)

    predictions = loaded_model.predict(predict_frame).as_data_frame()['predict'].values
    json_compatible_item_data = jsonable_encoder(predictions.tolist())
    output = JSONResponse(content=json_compatible_item_data)
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

    # model = aml.leader
    # model_path = h2o.save_model(model=model, path='sample_model/', force=True)
    # model_path = h2o.save_model(model=model, path='gs://deneme-bucket-1/model/', force=True)
    # print(model_path)

    response =  aml.leaderboard.as_data_frame(use_pandas=True).to_json()
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
