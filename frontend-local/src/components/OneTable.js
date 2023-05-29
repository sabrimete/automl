import * as React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import Box from '@mui/material/Box';
import { TableVirtuoso } from 'react-virtuoso';
import { createTheme, ThemeProvider } from '@mui/material/styles';

const all_models_endpoint = 'https://localhost:8000/runs';

const theme = createTheme({
  palette: {
    onetable: {
      // Purple and green play nicely together.
      main: '#33691e',
    },
  },
});

function createData(id, data) {
    return { 
      id, 
      name: data[0], 
      timestamp: data[1]
    };
  }

const columns = [
  {
    width: 50,
    dataKey: 'name',
  },
  {
    width: 100,
    dataKey: 'timestamp',
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
export default function OneTable() {
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
        itemContent={rowContent}
      />
    </Paper>
  );
}

// export default function OneTable() {
//     const modelId = "DeepLearning_1_AutoML_24_20230526_145408";
//     const [data, setData] = React.useState([]);
//     const [info, setInfo] = React.useState([]);
//     const [metrics, setMetrics] = React.useState([]);
//     const [params, setParams] = React.useState([]);
//     const [tags, setTags] = React.useState([]);
//     console.log("here");
//     React.useEffect(() => {
//       const fetchData = async () => {
//         try {
//           const response = await fetch(one_model_endpoint, {
//             method: 'POST',
//             headers: {
//               'Content-Type': 'text/plain'
//             },
//             body: modelId,
//           });
    
//           const data = await response.json();
          
//           const parsedJSON = JSON.parse(data);
//           const infoArray = Object.entries(parsedJSON.info);
//           const metricsArray = Object.entries(parsedJSON.data.metrics);
//           const paramsArray = Object.entries(parsedJSON.data.params);
//           const tagsArray = Object.entries(parsedJSON.data.tags);
//           const infos = [];
//           for (let index = 0; index < infoArray.length; index++) {
//             const item = infoArray[index];
//             const row = createData(index, item);
//             infos.push(row);
//           }
//           setInfo(infos);
//           const metrics = [];
//           for (let index = 0; index < metricsArray.length; index++) {
//             const item = metricsArray[index];
//             const row = createData(index, item);
//             metrics.push(row);
//           }
//           setMetrics(metrics);
//           const params = [];
//           for (let index = 0; index < paramsArray.length; index++) {
//             const item = paramsArray[index];
//             const row = createData(index, item);
//             params.push(row);
//           }
//           setParams(params);
//           const tags = [];
//           for (let index = 0; index < tagsArray.length; index++) {
//             const item = tagsArray[index];
//             const row = createData(index, item);
//             tags.push(row);
//           }
//           setTags(tags);
//         } catch (error) {
//           console.log('Error:', error);
//         }
//       }
    
//       fetchData();
//     }, [one_model_endpoint, modelId]);
    
  
//     return (
//       <ThemeProvider theme={theme}>
//       <Box
//       sx={{
//         display: 'flex',
//         flexDirection: 'column',
//         alignItems: 'center',
//         '& > *': {
//           m: 1,
//         },
//       }}
//     >
//       <ButtonGroup variant="text" aria-label="text button group" color="onetable">
//         <Button onClick={() => setData(info)}>Info</Button>
//         <Button onClick={() => setData(metrics)}>Metrics</Button>
//         <Button onClick={() => setData(params)}>Parameters</Button>
//         <Button onClick={() => setData(tags)}>Tags</Button>
//       </ButtonGroup>
//       <Paper style={{ height: 400, width: 800}}>
//         <TableVirtuoso
//           data={data}
//           components={VirtuosoTableComponents}
//           itemContent={rowContent}
//         />
//       </Paper>
//       </Box>
//       </ThemeProvider>
//     );
//   }
  