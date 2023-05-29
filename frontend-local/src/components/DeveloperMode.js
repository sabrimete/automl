import React, { useState } from "react";
import styles from './Developer.module.css';
import Leaderboard from './Leaderboard';
import PacmanLoader from "react-spinners/PacmanLoader";
import PropagateLoader from "react-spinners/PropagateLoader";
import RingLoader from "react-spinners/RingLoader";
import Papa from "papaparse";
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';

const predict_endpoint = 'https://inference-6r72er7ega-uc.a.run.app/predict';
const train_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/train';
const save_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/save_models';
const heatmap_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/heatmap';

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
      width: 250,
    },
  },
};

const names = [
  'GLM',
  'DeepLearning',
  'DRF',
  'GBM',
  'XGBoost',
  'StackedEnsemble',
];

function getStyles(name, selectedAlgo, theme) {
  return {
    fontWeight:
      selectedAlgo.indexOf(name) === -1
        ? theme.typography.fontWeightRegular
        : theme.typography.fontWeightMedium,
  };
}

const theme = createTheme({
  palette: {
    user: {
      // Purple and green play nicely together.
      main: '#008394',
    },
    developer: {
      // This is green.A700 as hex.
      main: '#00695f',
    },
    manual: {
      // This is green.A700 as hex.
      main: '#4caf50',
    },
    predict: {
      main: '#8bc34a',
    },
  },
});

const Developer = () => {
  const [trainFile, setTrainFile] = useState(null);
  const [trainFileLabel, setTrainFileLabel] = useState("Choose file");
  const [predictFile, setPredictFile] = useState(null);
  const [predictFileLabel, setPredictFileLabel] = useState("Choose file");
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

  const handleHeatmapButtonClick = async () => {
    if (heatmap) {
      URL.revokeObjectURL(heatmap);
      setHeatmap(null);
    }
  
    const formData = new FormData();
    formData.append("file", trainFile);
  
    const response = await fetch(heatmap_endpoint, {
      method: "POST",
      body: formData,
    });
  
    if (!response.ok) {
      console.error('Server response was not ok');
      return;
    }
  
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    setHeatmap(objectUrl);
  };
  

  const handleColumnDropChange = (e) => {
    const columnName = e.target.name;
    if (e.target.checked) {
      setColumnsToDrop((prev) => new Set([...prev, columnName]));
    } else {
      setColumnsToDrop((prev) => {
        const newSet = new Set([...prev]);
        newSet.delete(columnName);
        return newSet;
      });
    }
  };

  const handleDropColumnsClick = () => {
    // Filter out the dropped columns from columnNames
    const updatedColumnNames = columnNames.filter(
      (columnName) => !columnsToDrop.has(columnName)
    );
    setColumnNames(updatedColumnNames);
  
    // Filter out the dropped columns from columnInsights
    const updatedColumnInsights = columnInsights.filter(
      (insight) => !columnsToDrop.has(insight.name)
    );
    setColumnInsights(updatedColumnInsights);
  
    // Filter out the dropped columns from the original train data
    const fileReader = new FileReader();
    fileReader.onload = async (event) => {
      const fileContent = event.target.result;
      const parsedData = Papa.parse(fileContent, { header: true });
      const filteredData = parsedData.data.map((row) => {
        const newRow = { ...row };
        columnsToDrop.forEach((col) => {
          delete newRow[col];
        });
        return newRow;
      });
  
      const csvString = Papa.unparse(filteredData);

      // convert string to Blob
      const csvBlob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
  
      // convert Blob to File
      const csvFile = new File([csvBlob], "updated_train.csv", { type: "text/csv" });
  
      setTrainFile(csvFile);
    };
    fileReader.readAsText(trainFile);
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

  const handleTargetStringChange = (e) => {
    setTargetString(e.target.value);
  };

  const handleMaxRuntimeSecsChange = (e) => {
    setMaxRuntimeSecs(e.target.value);
  };

  const handleMaxModelsChange = (e) => {
    setMaxModels(e.target.value);
  };

  const handleNfoldsChange = (e) => {
    setNfolds(e.target.value);
  };

  const handleSeedChange = (e) => {
    setSeed(e.target.value);
  };

  const handleAlgoSelectChange = (event) => {
    const {
      target: { value },
    } = event;
    setSelectedAlgos(
      // On autofill we get a stringified value.
      typeof value === 'string' ? value.split(',') : value,
    );
  };

  // const handleAlgoSelectChange = (e) => {
  //   const options = e.target.options;
  //   const selected = [];
  //   for (let i = 0; i < options.length; i++) {
  //     if (options[i].selected) {
  //       selected.push(options[i].value);
  //     }
  //   }
  //   setSelectedAlgos(selected);
  // };

  
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
    const parsedData = JSON.parse(data);
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
    <ThemeProvider theme={theme}>
    <div className={styles.AutoMLPipeline__container}>
      <div className={styles.trainSection}>
      <h2>Train</h2>
      <form onSubmit={handleSubmit}>
        <Box
          component="form"
          sx={{
            '& > :not(style)': { m: 0, width: 'auto' },
          }}
          noValidate
          autoComplete="off"
        >
          <strong> Choose Your Train File</strong> <br></br>
        <TextField id="filled-basic" color="developer" type="file" variant="filled" onChange={(e) => handleTrainFileChange(e)}  />
        </Box>
        {/* <label htmlFor="trainFile"> <strong>Choose Your Train File  </strong></label>
        <input
          type="file"
          id="trainFile"
          name="trainFile"
          onChange={(e) => handleTrainFileChange          (e)}
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
          <br/> <br/>
        <strong> Specify the Desired Parameters</strong> 
            <div className={styles.inputs}>    
            <Box
              component="form"
              sx={{
                '& > :not(style)': { m: 1, width: '25ch' },
              }}
              noValidate
              autoComplete="on"
            >
            <TextField id="outlined-number" color="developer" label="Max Runtime Secs" type="number" variant="outlined" onChange={(e) => handleMaxRuntimeSecsChange(e)}  />
            </Box>
            {/* <label htmlFor="maxRuntimeSecs">Max Runtime Secs</label> <input
            placeholder="Default is 3600"
            type="number"
            id="maxRuntimeSecs"
            name="maxRuntimeSecs"
            value={maxRuntimeSecs}
            onChange={(e) => handleMaxRuntimeSecsChange(e)} />
            <br></br> */}
            <Box
              component="form"
              sx={{
                '& > :not(style)': { m: 1, width: '25ch' },
              }}
              noValidate
              autoComplete="on"
            >
            <TextField id="outlined-number" color="developer" label="Max Models" type="number" variant="outlined" onChange={(e) => handleMaxModelsChange(e)}/>
            </Box>
            {/* <label htmlFor="maxModels">Max Models </label>
            <input
              placeholder="Default is None"
              type="number"
              id="maxModels"
              name="maxModels"
              value={maxModels}
              onChange={(e) => handleMaxModelsChange(e)}
            /> */}
            <Box
              component="form"
              sx={{
                '& > :not(style)': { m: 1, width: '25ch' },
              }}
              noValidate
              autoComplete="on"
            >
            <TextField id="outlined-number" color="developer" label="nFolds" type="number" variant="outlined" onChange={(e) => handleNfoldsChange(e)}/>
            </Box>
            {/* <label htmlFor="nfolds">Nfolds </label>
            <input
              placeholder="Default is -1"
              type="number"
              id="nfolds"
              name="nfolds"
              value={nfolds}
              onChange={(e) => handleNfoldsChange(e)}
            /> */}
            {/* <label htmlFor="seed">Seed </label>
            <input
              placeholder="Default is None"
              type="number"
              id="seed"
              name="seed"
              value={seed}
              onChange={(e) => handleSeedChange(e)}
            /> */}
            <Box
              component="form"
              sx={{
                '& > :not(style)': { m: 1, width: '25ch' },
              }}
              noValidate
              autoComplete="on"
            >
            <TextField id="outlined-number" color="developer" label="Seed" type="number" variant="outlined" onChange={(e) => handleSeedChange(e)}/>
            </Box>
            <Box
              component="form"
              sx={{
                '& > :not(style)': { m: 1, width: '25ch' },
              }}
              noValidate
              autoComplete="on"
            >
              <FormControl fullWidth>
                <InputLabel id="demo-simple-select-label">Algorithms</InputLabel>
                
                <Select
                  labelId="demo-simple-select-label"
                  id="demo-simple-select"
                  multiple
                  value={selectedAlgos}
                  label="Algorithms"
                  onChange={handleAlgoSelectChange}
                  MenuProps={MenuProps}
                >{names.map((name) => (
                  <MenuItem
                    key={name}
                    value={name}
                    style={getStyles(name, selectedAlgos, theme)}
                  >
                    {name}
                  </MenuItem>
                ))}
                </Select>
              </FormControl>
            </Box>
            {/* <label htmlFor="algos">Algorithms </label>
            <select multiple name="algos" id="algos" onChange={handleAlgoSelectChange}>
              <option value="GLM">GLM</option>
              <option value="DeepLearning">DeepLearning</option>
              <option value="DRF">DRF</option>
              <option value="GBM">GBM</option>
              <option value="XGBoost">XGBoost</option>
              <option value="StackedEnsemble">StackedEnsemble</option>
            </select> */}
            <br></br>
            </div>
            <Button style={{ width: "100px", height: "50px", margin: "10px"}} color="success" variant="contained" type="submit"><strong>Train</strong></Button>
        </form>
        {columnInsights.length > 0 && (
        <div className={styles.columnInsights}>
          <h3>Column Insights</h3>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Unique Values</th>
                <th>Missing Values</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Drop Column</th>
              </tr>
            </thead>
            <tbody>
              {columnInsights.map((insight) => (
                <tr key={insight.name} className={insight.name === targetString ? styles.highlight : ''}>
                  <td>{insight.name}</td>
                  <td>{insight.type}</td>
                  <td>{insight.unique_values}</td>
                  <td>{insight.null_count}</td>
                  <td>{insight.min}</td>
                  <td>{insight.max}</td>
                  <td>{insight.mean}</td>
                  <td>
                  <input
                    type="checkbox"
                    id={`drop_${insight.name}`}
                    name={insight.name}
                    onChange={handleColumnDropChange}
                  />
                </td>
                </tr>
              ))}
            </tbody>
          </table>

          <Button style={{ width: "100px", height: "50px", margin: "10px"}} color="success" variant="contained" type="submit"><strong></strong></Button>
          <button onClick={handleDropColumnsClick}>Drop Selected Columns</button>
          <button onClick={handleHeatmapButtonClick}>Get the Heatmap</button>
        </div>
      )}

        {trainLoading && (
        <div className={styles.loadingSection}>
          <PacmanLoader color="#1b5e20" size={50} />
        </div>
      )}
      </div>

      {leaderboardData && (
        <div className={styles.leaderboardContainer}>
          <Leaderboard
            data={leaderboardData}
            selectedModels={selectedModels}
            setSelectedModels={setSelectedModels}
            onSave={saveSelectedModels}
          />
        {saveLoading && (
        <div className={styles.loadingSection}>
          <RingLoader color="#1b5e20" size={100} />
        </div>
      )}
        </div>
      )}
      {heatmap && (
        <div className={styles.heatmapContainer}>
          <img src={heatmap} alt="heatmap" />
        </div>
      )}
    </div>
    </ThemeProvider>
  );
  
};

export default Developer;