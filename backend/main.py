import numpy as np
import pandas as pd
import io
import h2o
from h2o.automl import H2OAutoML, get_leaderboard

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

from google.cloud import storage
from sklearn.metrics import accuracy_score

app = FastAPI()
h2o.init()

@app.post("/predict")
async def predict(request: Request):
    form_data = await request.form()
    file = form_data["file"].file
    predictString = form_data["predictString"]
    file_obj = io.BytesIO(file.read())
    test_df = pd.read_csv(file_obj)
    y_train = test_df[predictString].values
    test_frame = h2o.H2OFrame(test_df)

    path = 'gs://deneme-bucket-1/model/XGBoost_1_AutoML_1_20230322_115208' 
    loaded_model = h2o.load_model(path)

    predictions = loaded_model.predict(test_frame).as_data_frame()['predict'].values
    accuracy = accuracy_score(y_train, np.asarray(predictions))
    json_compatible_item_data = jsonable_encoder(predictions.tolist())
    output = JSONResponse(content=json_compatible_item_data)
    return output

@app.post("/train")
async def train(trainString: str = Form(...), file: bytes = File(...)):
    file_obj = io.BytesIO(file)
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
    return HTMLResponse(content=content)
