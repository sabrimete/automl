import React, { useState, useEffect } from "react";
import styles from './Manual.module.css';
import Leaderboard from './Leaderboard';
import PacmanLoader from "react-spinners/PacmanLoader";
import Slider from './Slid'
import PropagateLoader from "react-spinners/PropagateLoader";
import RingLoader from "react-spinners/RingLoader";
import Papa from "papaparse";
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import CollapsibleTable from "./CollapsibleTable";
import { createTheme, ThemeProvider } from '@mui/material/styles';

const predict_endpoint = 'https://inference-6r72er7ega-uc.a.run.app/predict';
const unsupervised_endpoint = 'http://localhost:8000/unsupervised-train-suggest';
const unsupervised_final_endpoint = 'http://localhost:8000/unsupervised-train-final';
const supervised_endpoint = 'http://localhost:8000/manual-supervised-train';
const save_endpoint = 'http://localhost:8000/save_models';

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
  const [slid1, setSlid1] = useState([]);
  const [slid2, setSlid2] = useState([]);
  const [slid3, setSlid3] = useState([]);
  const [step1, setStep1] = useState(null);
  const [step2, setStep2] = useState(null);
  const [step3, setStep3] = useState(null);
  const [outliers, setOutliers] = useState(null);
  const [clusters, setClusters] = useState(null);
  const [response, setResponse] = useState(null);

  // useEffect(() => {
  // }, [value1, value]);

  const handleChange = (event) => {
    setOptimalK(event.target.value);
  };

  const handleSelectedChange = (newSelected) => {
    setSelectedModels(newSelected);
  };

  const handleSlidChange1 = (newSlid) => {
    setSlid1(newSlid);
  };

  const handleSlidChange2 = (newSlid) => {
    setSlid2(newSlid);
  };
  const handleSlidChange3 = (newSlid) => {
    setSlid3(newSlid);
  };

  const handleStepChange1 = (newStep) => {
    setStep1(newStep);
  };

  const handleStepChange2 = (newStep) => {
    setStep2(newStep);
  };

  const handleStepChange3 = (newStep) => {
    setStep3(newStep);
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

  const handleDownloadClusters = (e) => {
    clusters.click();
  };

  const handleDownloadOutliers = (e) => {
    outliers.click();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setTrainLoading(true);
  
    const formData = new FormData();
    formData.append("file", trainFile);
    formData.append("target_string", targetString);
    formData.append("algo", selectedAlgo);
    if(slid1[0] != null && slid1[1] != null && step1 != null){  
      formData.append("ntrees_first", slid1[0]);
      formData.append("ntrees_last", slid1[1]);
      formData.append("ntrees_step", step1);
    }
    if(slid2[0] != null && slid2[1] != null && step2 != null){  
      formData.append("max_depth_first", slid2[0]);
      formData.append("max_depth_last", slid2[1]);
      formData.append("max_depth_step", step2);
    }
    if(slid3[0] != null && slid3[1] != null && step3 != null){  
      formData.append("learn_rate_first", slid3[0]);
      formData.append("learn_rate_last", slid3[1]);
      formData.append("learn_rate_step", step3);
    }

  
    const response = await fetch(supervised_endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setResponse(data);
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
    setOptimalK(0);
    if(data['finalImage']){
      setFinalData(data['finalImage']);
    }
    else {
      setFinalData("");
    }
    setunsupervisedLoading(false);
    const blob = new Blob([JSON.stringify(data['clusters'], null, 2)], {type: "application/json"});
    const blob2 = new Blob([JSON.stringify(data['outliers'], null, 2)], {type: "application/json"});
    
    // Create an object URL for the blob object
    const url = URL.createObjectURL(blob);
    const url2 = URL.createObjectURL(blob2);
    
    // Create a link element
    const link = document.createElement('a');
    const link2 = document.createElement('a');
    
    // Set the href and download attributes for the link
    link.href = url;
    link2.href = url2;
    link.download = 'clusters_response.json';
    link2.download = 'outliers_response.json';
    
    setOutliers(link2);
    setClusters(link);
    // Append the link to the body
    // document.body.appendChild(link);
    
    // // Simulate click
    // // link.click();
    
    // // Remove the link after download
    // document.body.removeChild(link);
  };

  const saveSelectedModels = async () => {
    setSaveLoading(true);
    const response = await fetch(save_endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_ids: selectedModels,
        train_file_name: "<your_train_file_name>",
      }),
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

        <Box
          component="form"
          sx={{
            '& > :not(style)': { m: 0, width: 'auto', height: 'auto' },
          }}
          noValidate
          autoComplete="off"
        >
          <strong> Choose Your Train File</strong> <br></br>
        <TextField id="filled-basic"  color="primary" type="file" variant="filled" onChange={(e) => handleUnsupervisedFileChange(e)}  />
        </Box>
          <br />
            <Button style={{ width: "200px", height: "50px", margin: "10px"}} color="primary" variant="contained" type="submit"><strong>TRAIN UNSUPERVISED</strong></Button>
        </form>
        {unsupervisedLoading && (
        <div className={styles.loadingSection}>
          <PropagateLoader color="#27632a" size={50} />
        </div>
        )}
        {optimalK != 0 && (
          <div>
          <div className={styles.images_container}>
            <img src={`data:image/jpeg;base64,${elbowData}`} alt='elbow_image' style={{ width: '700px', height: 'auto' }} />
            <img src={`data:image/jpeg;base64,${silhouetteData}`} style={{ width: '700px', height: 'auto' }} />
          </div>
          <p> Optimal K we found is {optimalK}. <br/> However, you can select a k value between 1 and 10. </p>
          <Box
              component="form"
              sx={{
                '& > :not(style)': { m: 1, width: '25ch' },
              }}
              noValidate
              autoComplete="on"
            >
          <FormControl>
          <InputLabel id="demo-simple-select-label">K Value</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={optimalK}
            label="OptimalK"
            onChange={handleChange}
          >
            <MenuItem value={1}>1</MenuItem>
            <MenuItem value={2}>2</MenuItem>
            <MenuItem value={3}>3</MenuItem>
            <MenuItem value={4}>4</MenuItem>
            <MenuItem value={5}>5</MenuItem>
            <MenuItem value={6}>6</MenuItem>
            <MenuItem value={7}>7</MenuItem>
            <MenuItem value={8}>8</MenuItem>
            <MenuItem value={9}>9</MenuItem>
            <MenuItem value={10}>10</MenuItem>
          </Select>
        </FormControl>
        </Box>
          <form onSubmit={handleUnsupervisedFinal}>
          <Button style={{ width: "200px", height: "50px", margin: "10px"}} color="primary" variant="contained" type="submit"><strong>GET THE FINAL</strong></Button>
          </form>
          </div>
        )}
        {finalData!= "" && (
        <div className={styles.images_container_final}>
          <img src={`data:image/jpeg;base64,${finalData}`} alt='final_image' style={{ alignContent:'center', width: '1000px', height: 'auto' }} />
          <div>
          <form onSubmit={handleDownloadClusters}>
          <Button style={{ width: "200px", height: "50px", margin: "10px"}} color="primary" variant="contained" type="submit"><strong>Download Clusters</strong></Button>
          </form>
          <form onSubmit={handleDownloadOutliers}>
          <Button style={{ width: "200px", height: "50px", margin: "10px"}} color="primary" variant="contained" type="submit"><strong>Download Outliers</strong></Button>
          </form>
          </div>
        </div>)
        }
      </div>}
      {mode === 'supervised' && 
      <div className={styles.supervisedSection}>
        <h2>TRAIN</h2>
        <form onSubmit={handleSubmit}>
        <Box
          component="form"
          sx={{
            '& > :not(style)': { m: 0, width: 'auto', height: 'auto' },
          }}
          noValidate
          autoComplete="off"
        >
          <strong> Choose Your Train File</strong> <br></br>
        <TextField id="filled-basic"  color="developer" type="file" variant="filled" onChange={(e) => handleTrainFileChange(e)}  />
        </Box>
        {/* <label htmlFor="trainFile"> <strong>Choose Your Train File  </strong></label>
        <input
          type="file"
          id="trainFile"
          name="trainFile"
          onChange={(e) => handleTrainFileChange(e)}
          /> */}
          <label htmlFor="targetString">
            {columnNames.length > 0
              ? <strong>Select the target column </strong>
              : <strong>Upload your train.csv file first! </strong>}
          </label>
          <br></br>
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
            <Box
              component="form"
              sx={{
                '& > :not(style)': { m: 1, width: '25ch' },
              }}
              noValidate
              autoComplete="on"
            >
            <FormControl>
              <InputLabel id="demo-simple-select-label">Algorithm</InputLabel>
              <Select
                labelId="demo-simple-select-label"
                id="demo-simple-select"
                value={selectedAlgo}
                label="SelectedAlgo"
                onChange={handleAlgoSelectChange}
              >
                <MenuItem value={"glm"}>GLM</MenuItem>
                <MenuItem value={"rf"}>Random Forest</MenuItem>
                <MenuItem value={"gbm"}>GBM</MenuItem>
                <MenuItem value={"xgb"}>XGBoost</MenuItem>
              </Select>
              
            </FormControl>
            </Box>
            <br></br>
            {selectedAlgo == "gbm" && 
            <div className="hyperparam_options">
              <Slider title={'ntrees'}  min={0} max={50} step={1} onSlidChange={handleSlidChange1} onStepChange={handleStepChange1}/>
              <Slider title={'max_depth'}  min={0} max={50} step={1} onSlidChange={handleSlidChange2} onStepChange={handleStepChange2}/>
              <Slider title={'learn_rate'}  min={0.0} max={0.5} step={0.01} onSlidChange={handleSlidChange3} onStepChange={handleStepChange3}/>

            </div>}
            <Button onClick={handleSubmit}  style={{ width: "200px", height: "50px", margin: "10px"}} color="primary" variant="contained" type="submit"><strong>TRAIN SUPERVISED</strong></Button>
        </form>
        {response!=null && <CollapsibleTable response={response} onSelectChange={handleSelectedChange}></CollapsibleTable>}
      </div>}
      {mode === 'supervised' && trainLoading && (
        <div className={styles.loadingSection}>
          <PacmanLoader color="#27632a" size={50} />
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