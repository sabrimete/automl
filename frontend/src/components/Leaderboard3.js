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
import styles from './Developer.module.css';

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

export default function Leaderboard({ columnInsight, targetString, handleDropColumnsClick, handleHeatmapButtonClick }) {
  const [selected, setSelected] = useState([]);

  const handleSelect = (event, id) => {
    const selectedIndex = selected.indexOf(id);
    const newSelected = [...selected];
  
    if (selectedIndex === -1) {
      newSelected.push(id);
    } else {
      newSelected.splice(selectedIndex, 1);
    }
  
    setSelected(newSelected);
  };

  const isSelected = (id) => selected.indexOf(id) !== -1;

  return (
    <ThemeProvider theme={theme}>
      <div>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Select</TableCell>
                <TableCell>
                  Name
                </TableCell>
                <TableCell>
                  Type
                </TableCell>
                <TableCell>Unique Values</TableCell>
                <TableCell>Null Count</TableCell>
                <TableCell>Min</TableCell>
                <TableCell>Max</TableCell>
                <TableCell>Mean</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Array.isArray(columnInsight) &&
                columnInsight.map(insight => (
                  <TableRow
                    key={insight.name}
                    className={insight.name === targetString ? styles.highlight : ''}
                    // style={{ backgroundColor: index % 2 === 0 ? '#f5f5f5' : '#ffffff' }}
                  >
                    <TableCell>
                      <Checkbox
                        onClick={(event) => handleSelect(event, insight.name)}
                        checked={isSelected(insight.name)}
                      />
                    </TableCell>
                    <TableCell>{insight.name}</TableCell>
                    <TableCell>{insight.type}</TableCell>
                    <TableCell>{insight.unique_values}</TableCell>
                    <TableCell>{insight.null_count}</TableCell>
                    <TableCell>{insight.min}</TableCell>
                    <TableCell>{insight.max}</TableCell>
                    <TableCell>{insight.mean}</TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
        <Button variant="contained" style={{ width: "250px", height: "50px", backgroundColor: '#00695f', marginTop: '10px', marginRight: '20px' }} onClick={() => handleDropColumnsClick(selected)}>
            Drop Selected Columns
          </Button>
          <Button variant="contained" style={{ width: "220px", height: "50px", backgroundColor: '#00695f', marginTop: '10px', marginLeft: '20px'  }} onClick={handleHeatmapButtonClick}>
            Get the Heatmap
          </Button>
      </div>
    </ThemeProvider>
  );
};

