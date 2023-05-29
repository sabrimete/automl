import React, { useState } from "react";
import styles from './User.module.css';
import Leaderboard from './Leaderboard';
import PacmanLoader from "react-spinners/PacmanLoader";
import PropagateLoader from "react-spinners/PropagateLoader";
import RingLoader from "react-spinners/RingLoader";
import Papa from "papaparse";
import Button from '@mui/material/Button';
import ReactVirtualizedTable from "./VirtualTable";

const predict_endpoint = 'https://inference-6r72er7ega-uc.a.run.app/predict';
// const all_models_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/run_names';
const all_models_endpoint = 'http://localhost:8000/runs';
// const one_model_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/run_info';
const one_model_endpoint = 'http://localhost:8000/run_info';
const train_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/train';
const save_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/save_models';

const User = () => {
  const [trainFile, setTrainFile] = useState(null);
  const [trainFileLabel, setTrainFileLabel] = useState(" Choose file ");
  const [predictFile, setPredictFile] = useState(null);
  const [modelId, setModelId] = useState("");
  const [predictFileLabel, setPredictFileLabel] = useState(" Choose file ");
  const [targetString, setTargetString] = useState("");
  const [maxRuntimeSecs, setMaxRuntimeSecs] = useState("");
  const [maxModels, setMaxModels] = useState("");
  const [nfolds, setNfolds] = useState("");
  const [seed, setSeed] = useState("");
  const [selectedAlgos, setSelectedAlgos] = useState([]);
  const [leaderboardData, setLeaderboardData] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [trainLoading, setTrainLoading] = useState(false);
  const [predictLoading, setpredictLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [columnNames, setColumnNames] = useState([]);
  const [columnInsights, setColumnInsights] = useState([]);
  const [columnsToDrop, setColumnsToDrop] = useState(new Set());
  const [heatmap, setHeatmap] = useState(null);
  const [oneModelData, setOneModelData] = useState(null);

  const analyzeColumns = (data) => {
    const filteredData = data.map((row) => {
      const newRow = { ...row };
      columnsToDrop.forEach((col) => {
        delete newRow[col];
      });
      return newRow;
    });

    const insights = filteredData.reduce(
      (acc, row) => {
        Object.entries(row).forEach(([key, value]) => {
          if (!acc[key]) {
            acc[key] = {
              type: null,
              unique_values: new Set(),
              null_count: 0,
              min: Number.POSITIVE_INFINITY,
              max: Number.NEGATIVE_INFINITY,
              sum: 0,
              counter: 0,
            };
          }
  
          if (value === "" || value === null) {
            acc[key].null_count += 1;
          } else {
            if (!isNaN(value)) {
              const numValue = Number(value);
              acc[key].min = Math.min(acc[key].min, numValue);
              acc[key].max = Math.max(acc[key].max, numValue);
              acc[key].sum += numValue;
              acc[key].counter += 1;
            }
  
            acc[key].unique_values.add(value);
          }
        });
  
        return acc;
      },
      {}
    );
  
    const result = Object.entries(insights).map(([name, data]) => ({
      name,
      type: isNaN(Array.from(data.unique_values)[0]) ? "string" : "number",
      unique_values: data.unique_values.size,
      null_count: data.null_count,
      min: isNaN(Array.from(data.unique_values)[0]) ? null : (data.min === Number.POSITIVE_INFINITY ? null : data.min),
      max: isNaN(Array.from(data.unique_values)[0]) ? null : (data.max === Number.NEGATIVE_INFINITY ? null : data.max),
      mean: isNaN(Array.from(data.unique_values)[0]) ? null : (data.sum / (data.counter - data.null_count)).toFixed(2),
    }));
  
    setColumnInsights(result);
  };

  const handleTrainFileChange = (e) => {
    setTrainFile(e.target.files[0]);
    setTrainFileLabel(e.target.files[0].name);
    const fileReader = new FileReader();
    fileReader.onload = async (event) => {
    const fileContent = event.target.result;

    // Extract the column names
    const parsedData = Papa.parse(fileContent, { header: true });
    const firstLine = fileContent.split("\n")[0];
    const columns = firstLine.split(",");

    // Update the column names state
    setColumnNames(columns);
    analyzeColumns(parsedData.data);
  };
  fileReader.readAsText(e.target.files[0]);
  };

  const handlePredictFileChange = (e) => {
    setPredictFile(e.target.files[0]);
    setPredictFileLabel(e.target.files[0].name);
  };

  const handleModelIdChange = (e) => {
    setModelId(e.target.value);
  };

  const handleTargetStringChange = (e) => {
    setTargetString(e.target.value);
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setTrainLoading(true);
  
    const formData = new FormData();
    formData.append("file", trainFile);
    formData.append("target_string", targetString);
    if (maxRuntimeSecs) formData.append("max_runtime_secs", maxRuntimeSecs);
    if (maxModels) formData.append("max_models", maxModels);
    if (nfolds) formData.append("nfolds", nfolds);
    if (seed) formData.append("seed", seed);
    if (selectedAlgos.length != 0) formData.append("include_algos", JSON.stringify(selectedAlgos));
  
    const response = await fetch(train_endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    const parsedJsonData = Object.keys(JSON.parse(data).model_id).map((key) => {
      const model = {};
      for (const prop in JSON.parse(data)) {
        model[prop] = JSON.parse(data)[prop][key];
      }
      return model;
    });
    
    setLeaderboardData(parsedJsonData);    
    setTrainLoading(false);
  };

  const handlePredictSubmit = async (e) => {
    e.preventDefault();
    setpredictLoading(true);
    const formData = new FormData();
    formData.append("file", predictFile);
    formData.append("run_name", modelId);

    const response = await fetch(predict_endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setpredictLoading(false);
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
    
    // Create an object URL for the blob object
    const url = URL.createObjectURL(blob);
    
    // Create a link element
    const link = document.createElement('a');
    
    // Set the href and download attributes for the link
    link.href = url;
    link.download = 'predict_response.json';
    
    // Append the link to the body
    document.body.appendChild(link);
    
    // Simulate click
    link.click();
    
    // Remove the link after download
    document.body.removeChild(link);
  };

  const getAllModels = async (e) => {
    var container = document.getElementById("responseContainer");
    e.preventDefault();
    // fetch(all_models_endpoint)  // Replace with your actual backend endpoint URL
    // .then(function(response) {
    //   return response.json();
    // })
    // .then(function(responseData) {
    //   // Iterate over the response data and create a paragraph for each item
    //   responseData.forEach(function(item) {
    //     var paragraph = document.createElement("p");
    //     paragraph.textContent = item;
    //     container.appendChild(paragraph);
    //   });
    // })
    // .catch(function(error) {
    //   console.log('Error:', error);
    // });
  };

  const getOneModel = async (e) => {

    var cont = document.getElementById("responseModel");
    console.log(modelId);
    e.preventDefault();
    const response = await fetch(one_model_endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'text/plain'
      },
      body: modelId,
    });
  
    console.log(response);
    const responseData = await response.json();
    setOneModelData(responseData);
  };

  const saveSelectedModels = async () => {
    setSaveLoading(true);
    const response = await fetch(save_endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(selectedModels),
    });

    if (response.ok) {
      console.log("Selected models saved successfully");
      alert("Selected models are saved successfully");
      setLeaderboardData(null);
      setSelectedModels([]);
    } else {
      console.error("Error saving selected models");
    }
    setSaveLoading(false);
  };
  return (
    <div className={styles.AutoMLPipeline__container}>
      <div className={styles.predictSection}>
        <h2>Predict</h2>
        <div className={styles.predictForm}>
        <form onSubmit={getAllModels}>
        <strong> Get Information of the Models: </strong> <br />
        <Button style={{ width: "250px", height: "50px", margin: "10px"}} color="secondary" variant="contained" type="submit"><strong>Get All Models</strong></Button>
        <div id="responseContainer">
          <ReactVirtualizedTable></ReactVirtualizedTable>
        </div>
        </form>
        <br></br>
        <form onSubmit={getOneModel}>
          <label htmlFor="modelId"> <strong> Or you can get your model by ID: </strong></label>
          <input
            type="text"
            id="modelId"
            name="modelId"
            value={modelId}
            onChange={(e) => handleModelIdChange(e)}
          />
          <br />
        <Button style={{ width: "250px", height: "50px", margin: "10px"}} color="secondary" variant="contained" type="submit"><strong> Get Model by Id</strong></Button>
        <div id="responseModel">
        {oneModelData && (
        <pre>{JSON.stringify(oneModelData, null, 2)}</pre>
      )}
        </div>
        </form>
        </div>
        <br />
        <br />
        <br />
        <form onSubmit={handlePredictSubmit}>
          <label htmlFor="predictFile"> <strong> Choose Your Test File: </strong></label>
          <input
            type="file"
            id="predictFile"
            name="predictFile"
            onChange={(e) => handlePredictFileChange(e)}
          />
          <br></br>
          <label htmlFor="modelId"> <strong> Specify Model: </strong></label>
          <input
            type="text"
            id="modelId"
            name="modelId"
            value={modelId}
            onChange={(e) => handleModelIdChange(e)}
          />
          <br />
          
        <Button style={{ width: "300px", height: "50px", margin: "10px"}} color="secondary" variant="contained" type="submit"><strong>Predict by This Model</strong></Button>
        </form>
        {predictLoading && (
        <div className={styles.loadingSection}>
          <PropagateLoader color="#7b1fa2" size={50} />
        </div>
      )}
      </div>

    </div>
  );
};

export default User;