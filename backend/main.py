import io
import json
from h2o.automl import H2OAutoML, get_leaderboard

from fastapi import FastAPI, File, Form, UploadFile, Request, Body, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


import random
from base64 import encodebytes

import h2o
import mlflow
import mlflow.h2o
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from h2o.estimators import H2OKMeansEstimator
from matplotlib import pyplot as plt
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score

app = FastAPI()
global_leaderboard = None
unsupervised_file = None
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
TRACKING_SERVER_HOST = "https://mlflow-server-6r72er7ega-uc.a.run.app"  # fill in with the external IP of the compute engine instance
mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
client = MlflowClient(TRACKING_SERVER_HOST)


def increment_array_int(first, last, increment):
    if first is None or last is None or increment is None or int(increment) <= 0:
        return None
    first = int(first)
    last = int(last)
    increment = int(increment)
    array = []
    current = first
    while current <= last:
        array.append(current)
        current += increment
    return array


def increment_array_float(first, last, increment):
    if first is None or last is None or increment is None or int(increment) <= 0:
        return None
    first = float(first)
    last = float(last)
    increment = float(increment)
    array = []
    current = first
    while current <= last:
        array.append(current)
        current += increment
    return array


@app.post("/save_models")
async def save_models(model_ids: list[str] = Body(...), train_file_name: str = Body(...)):
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
            mlflow.set_tag("mlflow.runName", model_id)

            score_history = model.scoring_history()
            metrics_list = ["accuracy", "auc", "logloss", "rmse", "mse", "mae", "rmsle", "residual_deviance", "mean_residual_deviance"]
            metrics = {}
            for metric in metrics_list:
                if hasattr(model, metric):
                    try:
                        metrics[metric] = getattr(model, metric)()
                    except Exception:
                        pass
            
            if score_history is not None:
                if 'timestamp' in score_history.columns:
                    score_history['timestamp'] = pd.to_datetime(score_history['timestamp']).astype(int) // 10**9
                non_numeric_cols = score_history.select_dtypes(exclude=[np.number]).columns
                score_history = score_history.drop(columns=non_numeric_cols)
                score_history = score_history.iloc[-1].to_dict()
                metrics = {**score_history, **metrics}

            metrics = {k: v for k, v in metrics.items() if v is not None}
            mlflow.log_metrics(metrics)

            mlflow.set_tag("train_file", train_file_name)

            params = model.actual_params
            for key in list(params.keys()):
                if not isinstance(params[key], (str, int, float)):
                    del params[key]
            mlflow.log_params(params)

            # Log and save best model (mlflow.h2o provides API for logging & loading H2O models)
            mlflow.h2o.log_model(model, artifact_path='model')
            model_uri = mlflow.get_artifact_uri('model')
            print(f'model saved in {model_uri}')

        # Perform any action you want with the selected models
        # For example, you can save the models to a file or database
    return {"message": model_uri}


@app.get("/runs")
async def get_runs():
    all_exps = [exp.experiment_id for exp in client.search_experiments()]
    runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ACTIVE_ONLY)
    runs = runs[runs['status'] != 'FAILED']
    runs = runs.to_dict('records')
    names = [run["tags.mlflow.runName"] for run in runs]
    timestamps = [run["metrics.timestamp"] for run in runs]
    train_files = [run["tags.train_file"] for run in runs]
    return json.dumps({"names": names, "timestamps": timestamps, "train_files": train_files})


@app.post("/run_info")
async def get_run_info(run_name: str = Body(...)):
    all_exps = [exp.experiment_id for exp in client.search_experiments()]
    run_info = client.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ACTIVE_ONLY,
                                  filter_string=f"tags.mlflow.runName = '{run_name}'")[0]
    return json.dumps(run_info.to_dictionary())


@app.post("/heatmap")
async def heatmap(request: Request):
    print("heatmap")
    form_data = await request.form()
    file = form_data["file"].file
    file_obj = io.BytesIO(file.read())
    data = pd.read_csv(file_obj)

    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    return Response(content=buffer.getvalue(), media_type="image/png")


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

    last_model = mlflow.h2o.load_model(artifact_uri + '/model')

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
    print("PARAMS ", trainString, max_runtime_secs, max_models, nfolds, seed, include_algos)

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

    lb = h2o.automl.get_leaderboard(aml, extra_columns='ALL')
    global global_leaderboard
    global_leaderboard = lb
    lb.head(rows=lb.nrows)

    response = lb.as_data_frame(use_pandas=True).to_json()
    print(response)
    print(type(response))
    return response


@app.post("/unsupervised-train-suggest")
async def unsupervisedTrainSuggest(request: Request):
    global unsupervised_file
    form_data = await request.form()
    data = form_data["file"].file
    file_obj = io.BytesIO(data.read())
    data_pd = h2o.H2OFrame(pd.read_csv(file_obj))
    unsupervised_file = data_pd

    # Convert the H2OFrame object to a pandas DataFrame
    data_df = data_pd.as_data_frame()

    # Extract the labels from the first line
    labels = data_df.columns

    for i in range(len(labels)):
        if str(data_df[labels[i]][0]) == "nan":
            return {"error": "Please remove string columns from the dataset."}

    # Define the range of k values to try
    k_values = range(2, 10)

    # Initialize lists to store the evaluation scores
    silhouette_scores = []
    sse = []

    # Perform k-means clustering for each k value
    for k in k_values:
        # Create the k-means model
        kmeans = H2OKMeansEstimator(k=k, seed=123)
        kmeans.train(training_frame=data_pd)

        # Compute the silhouette score
        predictions = kmeans.predict(data_pd)
        silhouette_scores.append(
            sklearn.metrics.silhouette_score(data_df, predictions.as_data_frame(), metric='euclidean'))

        # Compute the sum of squared errors (SSE)
        sse.append(kmeans.tot_withinss())

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    elbowImage = encodebytes(buffer.getvalue()).decode('ascii')
    elbowImage = elbowImage.replace('\n', '')
    buffer.close()

    # Plot the silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    silhouetteImage = encodebytes(buffer.getvalue()).decode('ascii')
    silhouetteImage = silhouetteImage.replace('\n', '')
    buffer.close()

    # Find the optimal value of k based on the silhouette score
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print("Optimal value of k: ", optimal_k)

    response = {
        "elbowImage": elbowImage,
        "silhouetteImage": silhouetteImage,
        "optimal_k": optimal_k,
    }

    return response


@app.post("/unsupervised-train-final")
async def unsupervisedTrainFinal(request: Request):
    global unsupervised_file
    form_data = await request.form()
    optimal_k = int(form_data.get("optimal_k"))
    data_pd = unsupervised_file

    # Convert the H2OFrame object to a pandas DataFrame
    data_df = data_pd.as_data_frame()

    # Extract the labels from the first line
    labels = data_df.columns

    # Train the final k-means model with the optimal k value
    final_kmeans = H2OKMeansEstimator(k=optimal_k, seed=123)
    final_kmeans.train(training_frame=data_pd)

    # Get the cluster assignments for each data point
    predictions = final_kmeans.predict(data_pd)
    clusters = predictions.as_data_frame()['predict'].values
    centroids = np.array(final_kmeans.centers())

    data_np = data_pd.as_data_frame().values
    # Compute the distances of each data point to the closest centroid
    distances = np.linalg.norm(data_np - centroids[clusters], axis=1)

    # Set a threshold for outlier detection (adjust as needed)
    threshold = np.percentile(distances, 95)  # 95th percentile as a threshold

    # Identify outliers as data points with distances above the threshold
    outliers = np.where(distances > threshold)[0]

    responseCluster = []
    responseOutliers = []
    for i in range(optimal_k):
        cluster_data = data_np[clusters == i]
        cluster_outliers = cluster_data[np.isin(np.where(clusters == i)[0], outliers)]
        responseCluster.append(cluster_data)
        responseOutliers.append(cluster_outliers)

    # Convert numpy arrays to nested lists
    data = [arr.tolist() for arr in responseCluster]
    responseCluster = {f"cluster {i + 1}": arr for i, arr in enumerate(data)}

    data = [arr.tolist() for arr in responseOutliers]
    responseOutliers = {f"outlier {i + 1}": arr for i, arr in enumerate(data)}

    # Prepare the plot based on the number of dimensions
    num_dimensions = data_pd.shape[1]

    # Generate colors for clusters and outliers
    num_clusters = len(np.unique(clusters))
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    outlier_color = 'red'

    finalImage = []
    # Plot the data with clusters and outliers
    if num_dimensions == 2:
        # Plot for 2D data
        plt.figure(figsize=(10, 6))
        for i in range(optimal_k):
            cluster_data = data_np[clusters == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=cluster_colors[i], alpha=0.7,
                        label=f'Cluster {i + 1}')
        plt.scatter(data_np[outliers, 0], data_np[outliers, 1], color=outlier_color, marker='x', label='Outliers')
        plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='*', s=200, edgecolors='k', label='Centroids')
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title('K-Means Clustering Result with Outliers')
        plt.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        finalImage = encodebytes(buffer.getvalue()).decode('ascii')
        finalImage = finalImage.replace('\n', '')
        buffer.close()

    elif num_dimensions == 3:
        # Plot for 3D data
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(optimal_k):
            cluster_data = data_np[clusters == i]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], color=cluster_colors[i], alpha=0.7,
                       label=f'Cluster {i + 1}')
        ax.scatter(data_np[outliers, 0], data_np[outliers, 1], data_np[outliers, 2],
                   color=outlier_color, marker='x', label='Outliers')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   color='red', marker='*', s=200, edgecolors='k', label='Centroids')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.set_title('K-Means Clustering Result with Outliers')
        ax.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        finalImage = encodebytes(buffer.getvalue()).decode('ascii')
        finalImage = finalImage.replace('\n', '')
        buffer.close()
    else:
        print("Cannot plot for data with more than 3 dimensions.")

    response = {
        "finalImage": finalImage,
        "clusters": responseCluster,
        "outliers": responseOutliers,
    }

    return response


@app.post("/manual-supervised-train")
async def manualSupervisedTrain(request: Request):
    form_data = await request.form()
    a = form_data["file"].file
    file_obj = io.BytesIO(a.read())
    file = h2o.H2OFrame(pd.read_csv(file_obj))
    algo = form_data.get("algo")
    response = form_data.get("target_string")
    predictors = [col for col in file.col_names if col != response]

    # Split the data into training and validation sets
    train, valid = file.split_frame(ratios=[0.7], seed=1234)

    gbmGridSearchParameters = {
        "ntrees": increment_array_int(form_data.get("ntrees_first"), form_data.get("ntrees_last"),
                                      form_data.get("ntrees_step")) or [10, 20, 30],
        "max_depth": increment_array_int(form_data.get("max_depth_first"), form_data.get("max_depth_last"),
                                         form_data.get("max_depth_step")) or [10, 20, 30],
        "learn_rate": increment_array_float(form_data.get("learn_rate_first"), form_data.get("learn_rate_last"),
                                            form_data.get("learn_rate_step")) or [0.1, 0.2, 0.3],
    }

    glmGridSearchParameters = {
        "alpha": increment_array_float(form_data.get("alpha_first"), form_data.get("alpha_last"),
                                       form_data.get("alpha_step")) or [0.1, 0.2, 0.3],
        "lambda": increment_array_float(form_data.get("lambda_first"), form_data.get("lambda_last"),
                                        form_data.get("lambda_step")) or [0.1, 0.2, 0.3],
    }

    xgbGridSearchParameters = {
        "ntrees": increment_array_int(form_data.get("ntrees_first"), form_data.get("ntrees_last"),
                                      form_data.get("ntrees_step")) or [10, 20, 30],
        "max_depth": increment_array_int(form_data.get("max_depth_first"), form_data.get("max_depth_last"),
                                         form_data.get("max_depth_step")) or [10, 20, 30],
        "learn_rate": increment_array_float(form_data.get("learn_rate_first"), form_data.get("learn_rate_last"),
                                            form_data.get("learn_rate_step")) or [0.1, 0.2, 0.3],
    }

    rfGridSearchParameters = {
        "ntrees": increment_array_int(form_data.get("ntrees_first"), form_data.get("ntrees_last"),
                                      form_data.get("ntrees_step")) or [10, 20, 30],
        "max_depth": increment_array_int(form_data.get("max_depth_first"), form_data.get("max_depth_last"),
                                         form_data.get("max_depth_step")) or [10, 20, 30],
    }

    if algo == "gbm":
        gbm = h2o.estimators.H2OGradientBoostingEstimator(
            min_rows=(form_data.get("min_rows") or 1),
            sample_rate=(form_data.get("sample_rate") or 1),
            col_sample_rate=(form_data.get("col_sample_rate") or 1),
            min_split_improvement=(form_data.get("min_split_improvement") or 0.01),
            distribution=str(form_data.get("distribution") or "gaussian"),
            seed=int(42),
        )

        gridSearch = h2o.grid.H2OGridSearch(gbm, gbmGridSearchParameters)
        gridSearch.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

        # create response for all models
        response = {'model_id': {}, 'rmse': {}, 'mse': {}, 'mae': {}, 'rmsle': {}, 'mean_residual_deviance': {},
                    'params': {}}
        for i in range(len(gridSearch.get_grid(sort_by='rmse', decreasing=False).models)):
            model = gridSearch.get_grid(sort_by='rmse', decreasing=False).models[i]
            response["model_id"][str(i)] = model.model_performance(valid)._metric_json["model"]["name"]
            response["rmse"][str(i)] = model.rmse()
            response["mse"][str(i)] = model.mse()
            response["mae"][str(i)] = model.mae()
            response["rmsle"][str(i)] = model.rmsle()
            response["mean_residual_deviance"][str(i)] = model.mean_residual_deviance()
            response["params"][str(i)] = {}
            response["params"][str(i)]["ntrees"] = model.actual_params['ntrees']
            response["params"][str(i)]["max_depth"] = model.actual_params['max_depth']
            response["params"][str(i)]["learn_rate"] = model.actual_params['learn_rate']
            response["params"][str(i)]["min_rows"] = model.actual_params['min_rows']
            response["params"][str(i)]["sample_rate"] = model.actual_params['sample_rate']
            response["params"][str(i)]["col_sample_rate"] = model.actual_params['col_sample_rate']
            response["params"][str(i)]["min_split_improvement"] = model.actual_params['min_split_improvement']
            response["params"][str(i)]["distribution"] = model.actual_params['distribution']

        return response

    elif algo == "glm":
        # Create a GLM model
        glm = h2o.estimators.H2OGeneralizedLinearEstimator(
            family=str(form_data.get("family") or "binomial"),  # use binomial family for binary classification
            max_iterations=(form_data.get("max_iterations") or 100),
            link=str(form_data.get("link") or "logit"),
            missing_values_handling=str(form_data.get("missing_values_handling") or "MeanImputation"),
            seed=int(42)  # set the seed for reproducibility
        )

        gridSearch = h2o.grid.H2OGridSearch(glm, glmGridSearchParameters)
        gridSearch.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

        # create response for all models
        response = {'model_id': {}, 'rmse': {}, 'mse': {}, 'mae': {}, 'rmsle': {}, 'mean_residual_deviance': {},
                    'params': {}}
        for i in range(len(gridSearch.get_grid(sort_by='rmse', decreasing=False).models)):
            model = gridSearch.get_grid(sort_by='rmse', decreasing=False).models[i]
            response["model_id"][str(i)] = model.model_performance(valid)._metric_json["model"]["name"]
            response["rmse"][str(i)] = model.rmse()
            response["mse"][str(i)] = model.mse()
            response["mae"][str(i)] = None
            response["rmsle"][str(i)] = None
            response["mean_residual_deviance"][str(i)] = None
            response["params"][str(i)] = {}
            response["params"][str(i)]["alpha"] = model.actual_params['alpha']
            response["params"][str(i)]["lambda"] = model.actual_params['lambda']
            response["params"][str(i)]["family"] = model.actual_params['family']
            response["params"][str(i)]["max_iterations"] = model.actual_params['max_iterations']
            response["params"][str(i)]["link"] = model.actual_params['link']
            response["params"][str(i)]["missing_values_handling"] = model.actual_params['missing_values_handling']

        return response

    elif algo == "xgb":
        # Create an XGBoost model
        xgb = h2o.estimators.H2OXGBoostEstimator(
            seed=int(42),  # set the seed for reproducibility
            min_rows=(form_data.get("min_rows") or 1),
            sample_rate=(form_data.get("sample_rate") or 1),
            col_sample_rate=(form_data.get("col_sample_rate") or 1),
            min_split_improvement=(form_data.get("min_split_improvement") or 0.01),
            booster=str(form_data.get("booster") or "gbtree"),
        )

        gridSearch = h2o.grid.H2OGridSearch(xgb, xgbGridSearchParameters)
        gridSearch.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

        # create response for all models
        response = {'model_id': {}, 'rmse': {}, 'mse': {}, 'mae': {}, 'rmsle': {}, 'mean_residual_deviance': {},
                    'params': {}}
        for i in range(len(gridSearch.get_grid(sort_by='rmse', decreasing=False).models)):
            model = gridSearch.get_grid(sort_by='rmse', decreasing=False).models[i]
            response["model_id"][str(i)] = model.model_performance(valid)._metric_json["model"]["name"]
            response["rmse"][str(i)] = model.rmse()
            response["mse"][str(i)] = model.mse()
            response["mae"][str(i)] = model.mae()
            response["rmsle"][str(i)] = model.rmsle()
            response["mean_residual_deviance"][str(i)] = model.mean_residual_deviance()
            response["params"][str(i)] = {}
            response["params"][str(i)]["ntrees"] = model.actual_params['ntrees']
            response["params"][str(i)]["max_depth"] = model.actual_params['max_depth']
            response["params"][str(i)]["learn_rate"] = model.actual_params['learn_rate']
            response["params"][str(i)]["min_rows"] = model.actual_params['min_rows']
            response["params"][str(i)]["sample_rate"] = model.actual_params['sample_rate']
            response["params"][str(i)]["col_sample_rate"] = model.actual_params['col_sample_rate']
            response["params"][str(i)]["min_split_improvement"] = model.actual_params['min_split_improvement']
            response["params"][str(i)]["booster"] = model.actual_params['booster']

        return response

    elif algo == "rf":
        # Create a Random Forest model
        rf = h2o.estimators.H2ORandomForestEstimator(
            min_rows=int(form_data.get("min_rows") or 10),
            sample_rate=(form_data.get("sample_rate") or 1),
            col_sample_rate_per_tree=(form_data.get("col_sample_rate_per_tree") or 1),
            min_split_improvement=(form_data.get("min_split_improvement") or 0.01),
            mtries=(form_data.get("mtries") or -1),
            # maximum depth of each tree      # specify the minimum number of rows in each leaf node
            seed=int(42)  # set the seed for reproducibility
        )

        gridSearch = h2o.grid.H2OGridSearch(rf, rfGridSearchParameters)
        gridSearch.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

        # create response for all models
        response = {'model_id': {}, 'rmse': {}, 'mse': {}, 'mae': {}, 'rmsle': {}, 'mean_residual_deviance': {},
                    'params': {}}
        for i in range(len(gridSearch.get_grid(sort_by='rmse', decreasing=False).models)):
            model = gridSearch.get_grid(sort_by='rmse', decreasing=False).models[i]
            response["model_id"][str(i)] = model.model_performance(valid)._metric_json["model"]["name"]
            response["rmse"][str(i)] = model.rmse()
            response["mse"][str(i)] = model.mse()
            response["mae"][str(i)] = model.mae()
            response["rmsle"][str(i)] = model.rmsle()
            response["mean_residual_deviance"][str(i)] = model.mean_residual_deviance()
            response["params"][str(i)] = {}
            response["params"][str(i)]["ntrees"] = model.actual_params['ntrees']
            response["params"][str(i)]["max_depth"] = model.actual_params['max_depth']
            response["params"][str(i)]["min_rows"] = model.actual_params['min_rows']
            response["params"][str(i)]["sample_rate"] = model.actual_params['sample_rate']
            response["params"][str(i)]["col_sample_rate_per_tree"] = model.actual_params['col_sample_rate_per_tree']
            response["params"][str(i)]["min_split_improvement"] = model.actual_params['min_split_improvement']
            response["params"][str(i)]["mtries"] = model.actual_params['mtries']

        return response


@app.get("/")
async def main():
    content = """
    <body>
    <h2> Welcome to the End to End AutoML Pipeline Project</h2>
    </body>
    """
    return HTMLResponse(content=content)


