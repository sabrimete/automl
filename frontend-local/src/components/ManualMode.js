import React, { useState } from "react";
import styles from './Manual.module.css';
import Leaderboard from './Leaderboard';
import PacmanLoader from "react-spinners/PacmanLoader";
import PropagateLoader from "react-spinners/PropagateLoader";
import RingLoader from "react-spinners/RingLoader";
import Papa from "papaparse";
import Button from '@mui/material/Button';
import { createTheme, ThemeProvider } from '@mui/material/styles';

const predict_endpoint = 'https://inference-6r72er7ega-uc.a.run.app/predict';
const unsupervised_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/unsupervised-train';
const supervised_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/manual-supervised-train';
const save_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/save_models';

const Manual = () => {
  const [mode, setMode] = React.useState('user');
  const [unsupervisedFile, setUnsupervisedFile] = useState(null);
  const [unsupervisedLoading, setunsupervisedLoading] = useState(false);
  const [trainFile, setTrainFile] = useState(null);
  const [trainFileLabel, setTrainFileLabel] = useState("Choose file");
  const [predictFile, setPredictFile] = useState(null);
  const [predictFileLabel, setPredictFileLabel] = useState("Choose file");
  const [targetString, setTargetString] = useState("");
  const [maxRuntimeSecs, setMaxRuntimeSecs] = useState("");
  const [maxModels, setMaxModels] = useState("");
  const [nfolds, setNfolds] = useState("");
  const [seed, setSeed] = useState("");
  const [selectedAlgos, setSelectedAlgos] = useState("glm");
  const [selectedModels, setSelectedModels] = useState([]);
  const [trainLoading, setTrainLoading] = useState(false);
  const [predictLoading, setpredictLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [columnNames, setColumnNames] = useState([]);
  const [columnInsights, setColumnInsights] = useState([]);
  const [columnsToDrop, setColumnsToDrop] = useState(new Set());
  const [isModel, setIsModel] = useState(false);
  const [modelId, setModelId] = useState("");

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

  const handleAlgoSelectChange = (e) => {
    setSelectedAlgos(e.target.value);
  };

  const handleTargetStringChange = (e) => {
    setTargetString(e.target.value);
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

  const handlePredictFileChange = (e) => {
    setPredictFile(e.target.files[0]);
    setPredictFileLabel(e.target.files[0].name);
  };

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


  const handleUnsupervisedFileChange = (e) => {
    setUnsupervisedFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setTrainLoading(true);
  
    const formData = new FormData();
    formData.append("file", trainFile);
    formData.append("target_string", targetString);
    formData.append("algo", selectedAlgos);
    console.log(selectedAlgos);
  
    const response = await fetch(supervised_endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    var container = document.getElementById("responseContainer");
    container.innerHTML = "";
    let table = document.createElement('table');

    // Create table header
    let thead = document.createElement('thead');
    let headerRow = document.createElement('tr');

    for (let key in data) {
        let th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    }

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create table body
    
    let tbody = document.createElement('tbody');
    let dataRow = document.createElement('tr');
    
    for (let key in data) {
      if(key==="model_id"){
        setModelId(data[key][0]);
      }
      console.log(key);
        let td = document.createElement('td');
        td.textContent = data[key][0];
        dataRow.appendChild(td);
    }

    tbody.appendChild(dataRow);
    table.appendChild(tbody);
    container.appendChild(table);
    // Add the table to the body of the page
    console.log(data);
    
    setTrainLoading(false);
    setIsModel(true);
  };

  const theme = createTheme({
    palette: {
      primary: {
        main: '#42a5f5',
      },
      secondary: {
        main: '#42a5f5',
      }
    },
  });

  const handleUnsupervisedSubmit = async (e) => {
    e.preventDefault();
    setunsupervisedLoading(true);
    const formData = new FormData();
    formData.append("file", unsupervisedFile);

    const response = await fetch(unsupervised_endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    console.log(data);
    setunsupervisedLoading(false);
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
      setSelectedModels([]);
    } else {
      console.error("Error saving selected models");
    }
    setSaveLoading(false);
  };

  
  return (    
  <div>
    <div className={styles.navmanual}>
      <nav>
      <ThemeProvider theme={theme}>
        <Button style={{color: 'white', margin: 5}} size="small" color="primary" variant="contained" onClick={() => setMode('unsupervised')}>Unsupervised Training</Button>
        <Button style={{color: 'white', margin: 5}} size="small" color="secondary" variant="contained" onClick={() => setMode('supervised')}>Supervised Training</Button>
        </ThemeProvider>
      </nav>
    </div>
    {mode === 'unsupervised' && 
      <div className={styles.unsupervisedSection}>
        <h2>TRAIN</h2>
        <form onSubmit={handleUnsupervisedSubmit}>
          <label htmlFor="unsupervisedFile"> <strong> Upload Your Train File </strong></label>
          <input
            type="file"
            id="unsupervisedFile"
            name="unsupervisedFile"
            onChange={(e) => handleUnsupervisedFileChange(e)}
          />
          <br />
            <Button style={{ width: "200px", height: "50px", margin: "10px"}} color="info" variant="contained" type="submit"><strong>TRAIN UNSUPERVISED</strong></Button>
        </form>
        {unsupervisedLoading && (
        <div className={styles.loadingSection}>
          <PropagateLoader color="#4A90E2" size={50} />
        </div>
        )}
      </div>}
      {mode === 'supervised' && 
      <div className={styles.supervisedSection}>
        <h2>TRAIN</h2>
        <form onSubmit={handleSubmit}>
        <label htmlFor="trainFile"> <strong>Choose Your Train File  </strong></label>
        <input
          type="file"
          id="trainFile"
          name="trainFile"
          onChange={(e) => handleTrainFileChange          (e)}
          />
          
          <label htmlFor="targetString">
            {columnNames.length > 0
              ? "Select the target column! "
              : "Upload your train.csv file first! "}
          </label>
          {columnNames.length > 0 ? (
          <select
            id="targetString"
            name="targetString"
            value={targetString}
            onChange={(e) => handleTargetStringChange(e)}
          >
            {columnNames.map((columnName) => (
              <option key={columnName} value={columnName}>
                {columnName}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            id="targetString"
            name="targetString"
            value={targetString}
            onChange={(e) => handleTargetStringChange(e)}
            disabled
          />
        )}
            <br/>
            <br/>
            <label htmlFor="algos"><strong>Select Your Supervised Algorithm</strong> </label>
            <br/>
            <select name="algos" id="algos" value={selectedAlgos} onChange={handleAlgoSelectChange}>
              <option value="glm">GLM</option>
              <option value="rf">Random Forest</option>
              <option value="gbm">GBM</option>
              <option value="xgb">XGBoost</option>
            </select>
            <br></br>
            <Button style={{ width: "200px", height: "50px", margin: "10px"}} color="info" variant="contained" type="submit"><strong>TRAIN SUPERVISED</strong></Button>
        </form>
      </div>}
      {trainLoading && (
        <div className={styles.loadingSection}>
          <PropagateLoader color="#4A90E2" size={50} />
        </div>
        )}
        <div id="responseContainer" className={styles.leaderboardContainer}>
        {isModel && (
        <Button onClick={saveSelectedModels} style={{ width: "300px", height: "50px", margin: "10px"}} color="info" variant="contained" type="submit"><strong>Save This Model</strong></Button>
        )}
        {saveLoading && (
        <div className={styles.loadingSection}>
          <RingLoader color="#4A90E2" size={100} />
        </div>
        
      )
      
      }
        </div>
        <form onSubmit={handlePredictSubmit}>
          <label htmlFor="predictFile"> <strong> Choose Your Test File: </strong></label>
          <input
            type="file"
            id="predictFile"
            name="predictFile"
            onChange={(e) => handlePredictFileChange(e)}
          />
          <br />
          
        <Button style={{ width: "300px", height: "50px", margin: "10px"}} color="info" variant="contained" type="submit"><strong>Predict by This Recent Model</strong></Button>
        </form>
        {predictLoading && (
        <div className={styles.loadingSection}>
          <PropagateLoader color="#7b1fa2" size={50} />
        </div>
      )}
  </div>
  );
};

export default Manual;