import React, { useState } from "react";
import styles from './User.module.css';
import Leaderboard from './Leaderboard';
import PacmanLoader from "react-spinners/PacmanLoader";
import RingLoader from "react-spinners/RingLoader";
import Papa from "papaparse";
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import { createTheme, ThemeProvider } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    user: {
      // Purple and green play nicely together.
      main: '#008394',
    },
    developer: {
      // This is green.A700 as hex.
      main: '#009688',
    },
    manual: {
      // This is green.A700 as hex.
      main: '#4caf50',
    },
    predict: {
      main: '#618833',
    },
  },
});

const train_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/train';
const save_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/save_models';

const User = () => {
  const [trainFile, setTrainFile] = useState(null);
  const [targetString, setTargetString] = useState("");
  const [leaderboardData, setLeaderboardData] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [trainLoading, setTrainLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [columnNames, setColumnNames] = useState([]);
  const [columnInsights, setColumnInsights] = useState([]);
  const [columnsToDrop, setColumnsToDrop] = useState(new Set());

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

  const handleTargetStringChange = (e) => {
    setTargetString(e.target.value);
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setTrainLoading(true);
  
    const formData = new FormData();
    formData.append("file", trainFile);
    formData.append("target_string", targetString);
  
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

  const saveSelectedModels = async () => {
    setSaveLoading(true);
    const response = await fetch(save_endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_ids: selectedModels,
        train_file_name: "filename",
      }),
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
        <br />
          <Button style={{ width: "100px", height: "50px", color:"white", margin: "10px"}} color="user" variant="contained" type="submit"><strong>Train</strong></Button>
        </form>
        {trainLoading && (
        <div className={styles.loadingSection}>
          <PacmanLoader color="#008394" size={50} />
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
          <RingLoader color="008394" size={100} />
        </div>
      )}
        </div>
      )}
    </div>
    </ThemeProvider>
  );
};

export default User;