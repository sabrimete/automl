// Leaderboard.js
import React, { useState, useMemo } from 'react';
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

const Leaderboard = ({ data, selectedModels, setSelectedModels, onSave }) => {

  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  console.log(data);
  const handleModelSelect = (modelId) => {
    if (selectedModels.includes(modelId)) {
      setSelectedModels(selectedModels.filter((id) => id !== modelId));
    } else {
      setSelectedModels([...selectedModels, modelId]);
    }
  };

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const SortArrow = () => (
    <span>{'↓'}</span>
  );

  const sortedData = React.useMemo(() => {
    const dataCopy = [...data];
    if (sortConfig.key !== null) {
      dataCopy.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    return dataCopy;
  }, [data, sortConfig]);
  return (
    <ThemeProvider theme={theme}>
    <div>
      <table>
        <thead>
          <tr>
            <th>Select</th>
            <th onClick={() => handleSort('model_id')}> Model ID  <SortArrow/> </th>
            <th onClick={() => handleSort('mean_residual_deviance')}>Mean Residual Deviance <SortArrow/> </th>
            <th onClick={() => handleSort('rmse')}>RMSE  <SortArrow/> </th>
            <th onClick={() => handleSort('mse')}>MSE  <SortArrow/> </th>
            <th onClick={() => handleSort('mae')}>MAE  <SortArrow/> </th>
            <th onClick={() => handleSort('rmsle')}>RMSLE <SortArrow/> </th>
            <th onClick={() => handleSort('training_time_ms')}>Training Time as ms <SortArrow/> </th>
            <th onClick={() => handleSort('predict_time_per_row_ms')}>Predict Time per Row as ms <SortArrow/> </th>
          </tr>
        </thead>
        <tbody>
          {Array.isArray(sortedData) &&
            sortedData.map((model, index) => (
              console.log(model),
              <tr key={model.model_id} style={{ backgroundColor: index % 2 === 0 ? '#f2f2f2' : 'white' }} >
                <td>
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model.model_id)}
                    onChange={() => handleModelSelect(model.model_id)}
                  />
                </td>
                <td>{model.model_id}</td>
                <td>{model.mean_residual_deviance}</td>
                <td>{model.rmse}</td>
                <td>{model.mse}</td>
                <td>{model.mae}</td>
                <td>{model.rmsle}</td>
                <td>{model.training_time_ms}</td>
                <td>{model.predict_time_per_row_ms}</td>
              </tr>
            ))}
        </tbody>
      </table>
      <Button onClick={onSave} style={{ width: "300px", height: "50px", margin: "10px", color: "white"}} color="user" variant="contained" type="submit"><strong>Save Selected Models</strong></Button>
    </div>
    </ThemeProvider>
  );
};

export default Leaderboard;
