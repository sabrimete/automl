import nicegui as ng
import requests
import pandas as pd
import io
import json
# Define receive and send functions for NiceGUI communication
def receive():
    return ng.get_events()

def send(update):
    ng.update(update)

ng.app("CMPE492AutoML Project", receive=receive, send=send)


# Set FastAPI endpoint
# endpoint = 'http://localhost:8000/predict'
predict_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/predict' # Specify this path for Dockerization to work
train_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/train'

# ng.html("<h2>Train Dataset</h2>")
train_csv = ng.file_uploader(" ", extensions=["csv"])
if train_csv is not None:
    train_df = pd.read_csv(train_csv)
    ng.subheader("Sample of Uploaded Dataset")
    ng.table(train_df.head())

    # Convert dataframe to BytesIO object (for parsing as file into FastAPI later)
    train_bytes_obj = io.BytesIO()
    train_df.to_csv(train_bytes_obj, index=False)  # write to BytesIO buffer
    train_bytes_obj.seek(0) # Reset pointer to avoid EmptyDataError

    files = {"file": ('train_dataset.csv', train_bytes_obj, "multipart/form-data")}

    # Upon click of button
    if ng.button('Start Train'):
        if len(train_df) == 0:
            ng.markdown("## Train Dataset")
        else:
            with ng.spinner('Training in Progress. Please Wait...'):
                output = requests.post(train_endpoint, 
                                       files=files,
                                       timeout=8000)
            ng.success('Success!')
            # ng.text(output)
            # ng.text(output.json())

# ng.html("<h2>Test Dataset</h2>")
test_csv = ng.file_uploader("  ", extensions=["csv"])
if test_csv is not None:
    test_df = pd.read_csv(test_csv)
    ng.subheader("Sample of Uploaded Dataset")
    ng.table(test_df.head())

    # Convert dataframe to BytesIO object (for parsing as file into FastAPI later)
    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)  # write to BytesIO buffer
    test_bytes_obj.seek(0) # Reset pointer to avoid EmptyDataError

    files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

    # Upon click of button
    if ng.button('Start Prediction'):
        if len(test_df) == 0:
            ng.markdown("## Train Dataset")
        else:
            with ng.spinner('Prediction in Progress. Please Wait...'):
                output = requests.post(predict_endpoint, 
                                       files=files,
                                       timeout=8000)
            ng.success('Success! Click Download button below to get prediction results (in JSON format)')
            # ng.download(json.dumps(output.json()), "automl_prediction_results.json")
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8501))
    ng.serve(port=port, address='0.0.0.0')