import * as React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import { TableVirtuoso } from 'react-virtuoso';

const all_models_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/runs';

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
    width: 250,
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
  const timestamp = new Date(row.timestamp * 1000)
  console.log(timestamp);
  const day = String(timestamp.getDate()).padStart(2, '0');
  const month = String(timestamp.getMonth() + 1).padStart(2, '0'); // Month is zero-based
  const year = timestamp.getFullYear();
  console.log(day,month,year);

  // Extract the time components
  let hours = timestamp.getHours();
  const minutes = String(timestamp.getMinutes()).padStart(2, '0');
  const time = `${hours}:${minutes}`;

  // Construct the formatted timestamp
  const formattedTimestamp = `${day}/${month}/${year}, ${time}`;
  console.log(formattedTimestamp);
  return (
    <React.Fragment>
      {columns.map((column) => (
        <TableCell
          key={column.dataKey}
          align={column.numeric || false ? 'right' : 'left'}
        >
          {column.dataKey === 'timestamp' ? formattedTimestamp : row[column.dataKey]}
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
        const formattedString = response.replace(/'/g, '"').replace(/NaN/g, "null"); // Replaces single quotes with double quotes

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
      <Paper elevation={7} style={{display: 'inline-block', height: 400, width: 800}}>
        <TableVirtuoso
          data={data}
          components={VirtuosoTableComponents}
          fixedHeaderContent={fixedHeaderContent}
          itemContent={rowContent}
        />
      </Paper>
    );
  }
  