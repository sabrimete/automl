import React, { useState, useEffect } from "react";
import styles from './Manual.module.css';
import Leaderboard from './Leaderboard';
import PacmanLoader from "react-spinners/PacmanLoader";
import MinimumDistanceSlider from './SliderMinimum'
import PropagateLoader from "react-spinners/PropagateLoader";
import RingLoader from "react-spinners/RingLoader";
import Papa from "papaparse";
import Button from '@mui/material/Button';
import { createTheme, ThemeProvider } from '@mui/material/styles';

const predict_endpoint = 'https://inference-6r72er7ega-uc.a.run.app/predict';
const unsupervised_endpoint = 'http://backend-6r72er7ega-uc.a.run.app/unsupervised-train-suggest';
const unsupervised_final_endpoint = 'http://backend-6r72er7ega-uc.a.run.app/unsupervised-train-final';
const supervised_endpoint = 'http://backend-6r72er7ega-uc.a.run.app/manual-supervised-train';
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
  const [selectedAlgo, setSelectedAlgo] = useState("glm");
  const [selectedModels, setSelectedModels] = useState([]);
  const [trainLoading, setTrainLoading] = useState(false);
  const [hyperParamLoading, setHyperParamLoading] = useState(false);
  const [predictLoading, setpredictLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [columnNames, setColumnNames] = useState([]);
  const [columnInsights, setColumnInsights] = useState([]);
  const [columnsToDrop, setColumnsToDrop] = useState(new Set());
  const [isModel, setIsModel] = useState(false);
  const [modelId, setModelId] = useState("");
  const [elbowData, setElbowData] = useState("");
  const [finalData, setFinalData] = useState("");
  const [silhouetteData, setSilhouetteData] = useState("");
  const [optimalK, setOptimalK] = useState(0);
  const [selectedOption, setSelectedOption] = useState('');
  const [value1, setValue1] = useState([0, 40]);
  const [value, setValue] = useState(1);

  useEffect(() => {
    console.log('value1', value1);
    console.log('value', value);
  }, [value1, value]);

  const handleChange = (event) => {
    setSelectedOption(event.target.value);
  };

  const renderOptions = () => {
    const options = [];
    for (let i = 1; i <= 10; i++) {
      options.push(<option key={i} value={i}>{i}</option>);
    }
    return options;
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

  const handleAlgoSelectChange = (e) => {
    setSelectedAlgo(e.target.value);
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
    formData.append("algo", selectedAlgo);
    formData.append("ntrees_first", value1[0]);
    formData.append("ntrees_last", value1[1]);
    formData.append("ntrees_step", value);
    console.log(selectedAlgo);
  
    const response = await fetch(supervised_endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    console.log(data);
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

    // Assuming all arrays in `data` are of the same length
    for (let i = 0; i < Object.keys(data).length; i++) {
        let dataRow = document.createElement('tr');
        for (let key in data) {
            let td = document.createElement('td');
            td.textContent = data[key][i];  // Change from `data[key][0]` to `data[key][i]`
            dataRow.appendChild(td);
        }
        tbody.appendChild(dataRow); 
    }

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
        main: '#27632a',
      },
      secondary: {
        main: '#357a38',
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
    setOptimalK(data['optimal_k']);
    setElbowData(data['elbowImage']);
    setSilhouetteData(data['silhouetteImage'])
    setunsupervisedLoading(false);
    setFinalData("");
  };

  const handleUnsupervisedFinal = async (e) => {
    e.preventDefault();
    setunsupervisedLoading(true);
    const formData = new FormData();
    formData.append("optimal_k", optimalK);
    const response = await fetch(unsupervised_final_endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    console.log(optimalK);
    console.log(data['finalImage'])
    setOptimalK(0);
    if(data['finalImage']){
      setFinalData(data['finalImage']);
    }
    else {
      setFinalData("");
    }
    console.log(finalData);
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
    <ThemeProvider theme={theme}>
  <div>
    <div className={styles.navmanual}>
      <nav>

        <Button style={{color: 'white', margin: 5}} size="small" color="primary" variant="contained" onClick={() => setMode('unsupervised')}>Unsupervised Training</Button>
        <Button style={{color: 'white', margin: 5}} size="small" color="primary" variant="contained" onClick={() => setMode('supervised')}>Hyperparameter Optimization</Button>

      </nav>
    </div>

  <div className="styles.AutoMLPipeline__container">
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
            <Button style={{ width: "200px", height: "50px", margin: "10px"}} color="primary" variant="contained" type="submit"><strong>TRAIN UNSUPERVISED</strong></Button>
        </form>
        {unsupervisedLoading && (
        <div className={styles.loadingSection}>
          <PropagateLoader color="#4A90E2" size={50} />
        </div>
        )}
        {optimalK != 0 && (
          <div>
          <div className={styles.images_container}>
            <img src={`data:image/jpeg;base64,${elbowData}`} alt='elbow_image' style={{ width: '700px', height: 'auto' }} />
            <img src={`data:image/jpeg;base64,${silhouetteData}`} style={{ width: '700px', height: 'auto' }} />
          </div>
          <p> Optimal K we found is {optimalK}. <br/> However, you can select a k value between 1 and 10. </p>
          <select value={selectedOption} onChange={handleChange}>
            <option value="">Select an option</option>
            {renderOptions()}
          </select>
          <form onSubmit={handleUnsupervisedFinal}>
          <Button style={{ width: "200px", height: "50px", margin: "10px"}} color="info" variant="contained" type="submit"><strong>GET THE FINAL</strong></Button>
          </form>
          </div>
        )}
        {finalData!= "" && (
        <div className={styles.images_container}>
          <img src={`data:image/jpeg;base64,${finalData}`} alt='final_image' style={{ width: '1000px', height: 'auto' }} />
        </div>)
        }
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
            <select name="algos" id="algos" value={selectedAlgo} onChange={handleAlgoSelectChange}>
              <option value="glm">GLM</option>
              <option value="rf">Random Forest</option>
              <option value="gbm">GBM</option>
              <option value="xgb">XGBoost</option>
            </select>
            <br></br>
            <br></br>
            <br></br>
            {selectedAlgo == "gbm" && 
            <div className="hyperparam_options">
              <MinimumDistanceSlider title={'ntrees'}  minDistance={10} initValues={[0, 40]} initValue={1} value1={value1} setValue1={setValue1} value={value} setValue={setValue} />
            </div>}
            <Button onClick={handleSubmit}  style={{ width: "200px", height: "50px", margin: "10px"}} color="primary" variant="contained" type="submit"><strong>TRAIN SUPERVISED</strong></Button>
        </form>
      </div>}
      {mode === 'supervised' && trainLoading && (
        <div className={styles.loadingSection}>
          <PropagateLoader color="#27632a" size={50} />
        </div>
        )}
        <div id="responseContainer" className={styles.leaderboardContainer}>
        {isModel && (
        <Button onClick={saveSelectedModels} style={{ width: "300px", height: "50px", margin: "10px"}} color="primary" variant="contained" type="submit"><strong>Save This Model</strong></Button>
        )}
        {saveLoading && (
        <div className={styles.loadingSection}>
          <RingLoader color="#27632a" size={100} />
        </div>
      )}
        </div>
  </div>

  </div>
  </ThemeProvider>
  );
};

export default Manual;