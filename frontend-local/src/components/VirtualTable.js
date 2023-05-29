import * as React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import { TableVirtuoso } from 'react-virtuoso';

function createData(id, data) {
    return { 
      id, 
      name: data[0], 
      timestamp: data[1], 
      trainFile: data[2] 
    };
  }

const columns = [
  {
    width: 200,
    label: 'Name',
    dataKey: 'name',
  },
  {
    width: 120,
    label: 'Timestamp',
    dataKey: 'timestamp',
    numeric: true,
  },
  {
    width: 120,
    label: 'Train File',
    dataKey: 'trainFile',
  },
];

const VirtuosoTableComponents = {
  Scroller: React.forwardRef((props, ref) => (
    <TableContainer component={Paper} {...props} ref={ref} />
  )),
  Table: (props) => (
    <Table {...props} sx={{ borderCollapse: 'separate', tableLayout: 'fixed' }} />
  ),
  TableHead,
  TableRow: ({ item: _item, ...props }) => <TableRow {...props} />,
  TableBody: React.forwardRef((props, ref) => <TableBody {...props} ref={ref} />),
};

function fixedHeaderContent() {
  return (
    <TableRow>
      {columns.map((column) => (
        <TableCell
          key={column.dataKey}
          variant="head"
          align={column.numeric || false ? 'right' : 'left'}
          style={{ width: column.width }}
          sx={{
            backgroundColor: 'background.paper',
          }}
        >
          {column.label}
        </TableCell>
      ))}
    </TableRow>
  );
}

function rowContent(_index, row) {
  return (
    <React.Fragment>
      {columns.map((column) => (
        <TableCell
          key={column.dataKey}
          align={column.numeric || false ? 'right' : 'left'}
        >
          {row[column.dataKey]}
        </TableCell>
      ))}
    </React.Fragment>
  );
}
export default function ReactVirtualizedTable() {
    const [data, setData] = React.useState([]);
    const all_models_endpoint = 'http://localhost:8000/runs';
  
    React.useEffect(() => {
      fetch(all_models_endpoint)
      .then(response => response.json())
      .then(responseData => {
        const parsedData = [["GBM_1_AutoML_24_20230526_145408", 1685112849.0, "filename"], ["DeepLearning_1_AutoML_24_20230526_145408", 1685112850.0, "filename"], ["GLM_1_AutoML_24_20230526_145408", 1685112848.0, "filename"], ["XRT_1_AutoML_24_20230526_145408", 1685112850.0, "filename"], ["GBM_grid_1_AutoML_24_20230526_145408_model_1", 1685112851.0, "filename"], ["GBM_4_AutoML_24_20230526_145408", 1685112850.0, "filename"], ["DRF_1_AutoML_24_20230526_145408", 1685112850.0, "filename"], ["GBM_5_AutoML_24_20230526_145408", 1685112850.0, "filename"], ["GBM_3_AutoML_24_20230526_145408", 1685112850.0, "filename"], ["GBM_2_AutoML_24_20230526_145408", 1685112850.0, "filename"], ["StackedEnsemble_BestOfFamily_1_AutoML_24_20230526_145408", NaN, "filename"], ["StackedEnsemble_AllModels_1_AutoML_24_20230526_145408", NaN, "filename"], ["XGBoost_1_AutoML_2_20230526_11245", 1685063567.0, "train3.csv"], ["StackedEnsemble_AllModels_1_AutoML_2_20230526_11245", NaN, "train3.csv"], ["StackedEnsemble_BestOfFamily_1_AutoML_1_20230526_11055", NaN, "train3.csv"], ["GBM_1_AutoML_23_20230526_34004", 1685072405.0, "filename"], ["GLM_1_AutoML_23_20230526_34004", 1685072404.0, "filename"], ["XRT_1_AutoML_23_20230526_34004", 1685072406.0, "filename"]];
        const rows = parsedData.map((item, index) =>
          createData(index, item)
        );
        setData(rows);
      })
      .catch(error => {
        console.log('Error:', error);
      });
    }, [all_models_endpoint]);
  
    return (
      <Paper style={{ height: 400, width: '100%' }}>
        <TableVirtuoso
          data={data}
          components={VirtuosoTableComponents}
          fixedHeaderContent={fixedHeaderContent}
          itemContent={rowContent}
        />
      </Paper>
    );
  }
  