import numpy as np
import pandas as pd
import io
import h2o
from h2o.automl import H2OAutoML, get_leaderboard

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import storage
from sklearn.metrics import accuracy_score

app = FastAPI()

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

@app.post("/predict")
async def predict(request: Request):
    form_data = await request.form()
    file = form_data["file"].file
    file_obj = io.BytesIO(file.read())
    predict_df = pd.read_csv(file_obj)
    predict_frame = h2o.H2OFrame(predict_df)

    path = 'gs://deneme-bucket-1/model/XGBoost_1_AutoML_1_20230322_115208' 
    loaded_model = h2o.load_model(path)

    predictions = loaded_model.predict(predict_frame).as_data_frame()['predict'].values
    json_compatible_item_data = jsonable_encoder(predictions.tolist())
    output = JSONResponse(content=json_compatible_item_data)
    return output

@app.post("/train")
async def train(request: Request):
    form_data = await request.form()
    file = form_data["file"].file
    trainString = form_data["targetString"]
    file_obj = io.BytesIO(file.read())
    train_df = pd.read_csv(file_obj)
    main_frame = h2o.H2OFrame(train_df)

    y = trainString
    x = [n for n in main_frame.col_names if n != y]

    aml = H2OAutoML(max_models=1, seed=42, balance_classes=False)
    aml.train(x=x, y=y, training_frame=main_frame)

    lb = aml.leaderboard
    lb.head(rows=lb.nrows)

    model = aml.leader
    model_path = h2o.save_model(model=model, path='gs://deneme-bucket-1/model/', force=True)
    print(model_path)

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
