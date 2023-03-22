import streamlit as st
import httpx
import pandas as pd
import io
import json

st.title('CMPE492AutoML Project')

# Set FastAPI endpoint
# endpoint = 'http://localhost:8000/predict'
predict_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/predict' # Specify this path for Dockerization to work
train_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/train'


st.subheader('Train Dataset')
train_csv = st.file_uploader(' ', type=['csv'], accept_multiple_files=False)

if train_csv:
    train_df = pd.read_csv(train_csv)
    st.subheader('Sample of Uploaded Dataset')
    st.write(train_df.head())

    # Convert dataframe to BytesIO object (for parsing as file into FastAPI later)
    train_bytes_obj = io.BytesIO()
    train_df.to_csv(train_bytes_obj, index=False)  # write to BytesIO buffer
    train_bytes_obj.seek(0) # Reset pointer to avoid EmptyDataError

    files = {"file": ('train_dataset.csv', train_bytes_obj, "multipart/form-data")}

    # Upon click of button
    if st.button('Start Train'):
        if len(train_df) == 0:
            st.write("Please upload a valid Train dataset!")  # handle case with no image
        else:
            with st.spinner('Training in Progress. Please Wait...'):
                with httpx.Client(http2=True) as client:
                    response = client.post(train_endpoint,
                                           files=files,
                                           timeout=8000)
            st.success('Success!')
            st.write(response)
            st.write(response.json())
            # pd.DataFrame.from_dict(output.json(), orient="index")
            # print(type(df))
            # print(df)
            # st.write(df)

st.subheader('Test Dataset')
test_csv = st.file_uploader('  ', type=['csv'], accept_multiple_files=False)

# Upon upload of file (to test using test.csv from data/processed folder)
if test_csv:
    test_df = pd.read_csv(test_csv)
    st.subheader('Sample of Uploaded Dataset')
    st.write(test_df.head())

    # Convert dataframe to BytesIO object (for parsing as file into FastAPI later)
    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)  # write to BytesIO buffer
    test_bytes_obj.seek(0) # Reset pointer to avoid EmptyDataError

    files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

    # Upon click of button
    if st.button('Start Prediction'):
        if len(test_df) == 0:
            st.write("Please upload a valid test dataset!")  # handle case with no image
        else:
            with st.spinner('Prediction in Progress. Please Wait...'):
                with httpx.Client(http2=True) as client:
                    response = client.post(predict_endpoint,
                                           files=files,
                                           timeout=8000)
            st.success('Success! Click Download button below to get prediction results (in JSON format)')
            # st.write(output.json()[1])
            # st.download_button(
            #     label='Download',
            #     data=json.dumps(output.json()[0]['body']), # Download as JSON file object
            #     file_name='automl_prediction_results.json'
            # )
