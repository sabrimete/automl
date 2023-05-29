import * as React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import { TableVirtuoso } from 'react-virtuoso';

const all_models_endpoint = 'http://localhost:8000/runs';

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
    width: 50,
    label: 'Timestamp',
    dataKey: 'timestamp',
    numeric: true,
  },
  {
    width: 40,
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
  
    React.useEffect(() => {
      fetch(all_models_endpoint)
      .then(response => response.json())
      .then(response => {
        console.log(typeof response);
        const formattedString = response.replace(/'/g, '"').replace(/NaN/g, "null");; // Replaces single quotes with double quotes

        const array = JSON.parse(formattedString);
        const rows = [];
        for (let index = 0; index < array.length; index++) {
          const item = array[index];
          const row = createData(index, item);
          rows.push(row);
        }
        setData(rows);
      })
      .catch(error => {
        console.log('Error:', error);
      });
    }, [all_models_endpoint]);
  
    return (
      <Paper style={{ height: 400, width: 800}}>
        <TableVirtuoso
          data={data}
          components={VirtuosoTableComponents}
          fixedHeaderContent={fixedHeaderContent}
          itemContent={rowContent}
        />
      </Paper>
    );
  }
  