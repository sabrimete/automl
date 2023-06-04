import React, { useState, useMemo } from 'react';
import Button from '@mui/material/Button';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Checkbox from '@mui/material/Checkbox';

const theme = createTheme({
  palette: {
    user: {
      main: '#008394',
    },
    developer: {
      main: '#009688',
    },
    manual: {
      main: '#4caf50',
    },
    predict: {
      main: '#618833',
    },
  },
});

const Leaderboard = ({ data, selectedModels, setSelectedModels, onSave }) => {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

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

  const SortArrow = () => <span>{sortConfig.direction === 'asc' ? '▲' : '▼'}</span>;

  const sortedData = useMemo(() => {
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
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Select</TableCell>
                <TableCell onClick={() => handleSort('model_id')}>
                  Model ID <SortArrow />
                </TableCell>
                <TableCell onClick={() => handleSort('mean_residual_deviance')}>
                  Mean Residual Deviance <SortArrow />
                </TableCell>
                <TableCell onClick={() => handleSort('rmse')}>RMSE <SortArrow /></TableCell>
                <TableCell onClick={() => handleSort('mse')}>MSE <SortArrow /></TableCell>
                <TableCell onClick={() => handleSort('mae')}>MAE <SortArrow /></TableCell>
                <TableCell onClick={() => handleSort('rmsle')}>RMSLE <SortArrow /></TableCell>
                <TableCell onClick={() => handleSort('training_time_ms')}>
                  Training Time as ms <SortArrow />
                </TableCell>
                <TableCell onClick={() => handleSort('predict_time_per_row_ms')}>
                  Predict Time per Row as ms <SortArrow />
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Array.isArray(sortedData) &&
                sortedData.map((model, index) => (
                  <TableRow
                    key={model.model_id}
                    style={{ backgroundColor: index % 2 === 0 ? '#f5f5f5' : '#ffffff' }}
                  >
                    <TableCell>
                      <Checkbox
                        checked={selectedModels.includes(model.model_id)}
                        onChange={() => handleModelSelect(model.model_id)}
                      />
                    </TableCell>
                    <TableCell>{model.model_id}</TableCell>
                    <TableCell>{model.mean_residual_deviance}</TableCell>
                    <TableCell>{model.rmse}</TableCell>
                    <TableCell>{model.mse}</TableCell>
                    <TableCell>{model.mae}</TableCell>
                    <TableCell>{model.rmsle}</TableCell>
                    <TableCell>{model.training_time_ms}</TableCell>
                    <TableCell>{model.predict_time_per_row_ms}</TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
        <Button variant="contained" style={{ width: "250px", height: "50px", backgroundColor: '#008394', marginTop: '10px' }} onClick={onSave}>
          Save Selected Models
        </Button>
      </div>
    </ThemeProvider>
  );
};

export default Leaderboard;
