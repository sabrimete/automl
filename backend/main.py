# ===========================
# Module: Backend setup (H2O, MLflowy)
# Author: Kenneth Leung
# Last Modified: 02 Jun 2022
# ===========================
# Command to execute script locally: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Command to run Docker image: docker run -d -p 8000:8000 <fastapi-app-name>:latest
import numpy as np
import pandas as pd
import io
import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import pickle


from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path

from sklearn.metrics import accuracy_score


from google.cloud import storage

# Create FastAPI instance
app = FastAPI()

# Initiate H2O instance and MLflow client
h2o.init()


# Create POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: bytes = File(...)):
    client = storage.Client()
    bucket = client.get_bucket('deneme-bucket-1')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    y_train = test_df['Survived']
    test_frame = h2o.H2OFrame(test_df)

    path = 'gs://deneme-bucket-1/model/XGBoost_1_AutoML_1_20230322_115208' 

    loaded_model = h2o.load_model(path)

    predictions = loaded_model.predict(test_frame).as_data_frame()['predict'].values
    # print('predictions')
    # print(predictions)
    accuracy = accuracy_score(y_train, np.asarray(predictions))
    json_compatible_item_data = jsonable_encoder(predictions.tolist())
    output = JSONResponse(content=json_compatible_item_data)
    return output
    # Convert predictions into JSON format
    # json_compatible_item_data = jsonable_encoder(predictions)
    # output = JSONResponse(content=json_compatible_item_data) 
    # accuracy = accuracy_score(y_train, np.asarray(predictions))
    # print('output, accuracy')
    # print(output, accuracy)
    # return output, accuracy

@app.post("/train")
async def train(file: bytes = File(...)):
    print('here')
    # Import data directly as H2O frame (default location is data/processed)
    file_obj = io.BytesIO(file)
    train_df = pd.read_csv(file_obj)
    main_frame = h2o.H2OFrame(train_df)

    # Set predictor and target columns
    y = 'Survived'
    x = [n for n in main_frame.col_names if n != y]

    # callh20automl function
    aml = H2OAutoML(max_models=1, # Run AutoML for n base models
                    seed=42, 
                    # exclude_algos =['DeepLearning'],
                    # stopping_metric ='logloss',
                    # sort_metric ='logloss',
                    balance_classes = False
    )
    # train model and record time % time
    aml.train(x = x, y = y, training_frame = main_frame)


    # View the H2O aml leaderboard
    lb = aml.leaderboard
    # Print all rows instead of 10 rows
    lb.head(rows = lb.nrows)

    model = aml.leader
    model_path = h2o.save_model(model=model, path='gs://deneme-bucket-1/model/', force=True)
    print(model_path)

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