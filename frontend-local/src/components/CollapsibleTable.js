import React, { useState, useEffect } from "react";
import PropTypes from 'prop-types';
import Box from '@mui/material/Box';
import Collapse from '@mui/material/Collapse';
import IconButton from '@mui/material/IconButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import { TableVirtuoso } from 'react-virtuoso';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import TableContainer from '@mui/material/TableContainer';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import Checkbox from '@mui/material/Checkbox';

function descendingComparator(a, b, orderBy) {
  if (b[orderBy] < a[orderBy]) {
    return -1;
  }
  if (b[orderBy] > a[orderBy]) {
    return 1;
  }
  return 0;
}

function getComparator(order, orderBy) {
  return order === 'desc'
    ? (a, b) => descendingComparator(a, b, orderBy)
    : (a, b) => -descendingComparator(a, b, orderBy);
}

function stableSort(array, comparator) {
  const stabilizedThis = array.map((el, index) => [el, index]);
  stabilizedThis.sort((a, b) => {
    const order = comparator(a[0], b[0]);
    if (order !== 0) {
      return order;
    }
    return a[1] - b[1];
  });
  return stabilizedThis.map((el) => el[0]);
}

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



const columns = [
  {
    width: 30,
    label: 'select',
    dataKey: 'select',
  },
  {
    width: 200,
    label: 'model_id',
    dataKey: 'model_id',
  },
  {
    width: 50,
    label: 'rmse',
    dataKey: 'rmse',
  },
  {
    width: 40,
    label: 'mse',
    dataKey: 'mse',
  },
  {
    width: 40,
    label: 'mae',
    dataKey: 'mae',
  },
  {
    width: 40,
    label: 'rmsle',
    dataKey: 'rmsle',
  },
  {
    width: 40,
    label: 'mean_residual_deviance',
    dataKey: 'mean_residual_deviance',
  },
];

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

function createData([select, model_id, rmse, mse, mae, rmsle, mean_residual_deviance, params]) {
  return {
    select,
    model_id,
    rmse,
    mse,
    mae,
    rmsle,
    mean_residual_deviance,
    params,
  };
}


export default function CollapsibleTable({response, onSelectChange}) {
  const [selected, setSelected] = useState([]);
  const [expandedRows, setExpandedRows] = useState([]);

  const handleSelect = (event, id) => {
    const selectedIndex = selected.indexOf(id);
    const newSelected = [...selected];
  
    if (selectedIndex === -1) {
      newSelected.push(id);
    } else {
      newSelected.splice(selectedIndex, 1);
    }
  
    setSelected(newSelected);
    onSelectChange(newSelected);
  };
  

  const isSelected = (id) => selected.indexOf(id) !== -1;
  const isRowExpanded = (model_id) => expandedRows.includes(model_id);

  const handleRowExpand = (model_id) => {
    const isExpanded = expandedRows.includes(model_id);
    const newExpandedRows = isExpanded
      ? expandedRows.filter((id) => id !== model_id)
      : [...expandedRows, model_id];
    setExpandedRows(newExpandedRows);
  };

  const rows = [];
  const keys = [];
  for (let key in response) {
    keys.push(key);
  }
  for (let i = 0; i < keys.length; i++) {
    const row = [false];
    for (let key in response) {
        row.push(response[key][i]);  
    }
    const rowTemp = createData(row);
    rows.push(rowTemp);
  }
  
  function renderParamsTable(params) {
    return (
      <Table>
        <TableBody>
          {Object.entries(params).map(([key, value]) => (
            <TableRow key={key}>
              <TableCell component="th" scope="row">
                {key}
              </TableCell>
              <TableCell>{value}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    );
  }

  function rowContent(_index, row) {
    const isExpanded = isRowExpanded(row.model_id);
    return (
      <React.Fragment>
        <TableCell padding="checkbox" style={{ width: columns[0].width }}>
          <Checkbox
            color="primary"
            onClick={(event) => handleSelect(event, row.model_id)}
            checked={isSelected(row.model_id)}
          />
        </TableCell>
        {columns.slice(1).map((column) => (
          <TableCell
            key={column.dataKey}
            align={column.numeric || false ? "right" : "left"}
            style={{
              width: column.width,
              padding: "16px",
            }}
          >
            <Box width="100%">{row[column.dataKey]}</Box>
          </TableCell>
        ))}
        <TableCell padding="checkbox" style={{ width: columns[columns.length - 1].width }}>
          <IconButton
            aria-label="expand row"
            size="small"
            onClick={() => handleRowExpand(row.model_id)}
          >
            {isExpanded ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
          </IconButton>
        </TableCell>
      </React.Fragment>
    );
  }
  
  // function rowContent(_index, row) {
  //   const isExpanded = isRowExpanded(row.model_id);
  //   return (
  //     <React.Fragment>
  //       <TableCell padding="checkbox" style={{ width: columns[0].width }}>
  //         <Checkbox
  //           color="primary"
  //           onClick={(event) => handleSelect(event, row.model_id)}
  //           checked={isSelected(row.model_id)}
  //         />
  //       </TableCell>
  //       {columns.slice(1).map((column, index) => (
  //         <TableCell
  //           key={column.dataKey}
  //           align={column.numeric || false ? 'right' : 'left'}
  //           style={{ width: column.width }}
  //         >
  //           <Box width={column.width}>
  //             {row[column.dataKey]}
  //           </Box>
  //         </TableCell>
  //       ))}
  //       <TableCell padding="checkbox" style={{ width: columns[columns.length - 1].width }}>
  //         <IconButton
  //           aria-label="expand row"
  //           size="small"
  //           onClick={() => handleRowExpand(row.model_id)}
  //         >
  //           {isExpanded ? (
  //             <KeyboardArrowUpIcon />
  //           ) : (
  //             <KeyboardArrowDownIcon />
  //           )}
  //         </IconButton>
  //       </TableCell>
  //     </React.Fragment>
  //   );
  // }
  
  return (
    <Paper style={{ height: 500, width: 1400, margin:"auto"}}>
      <TableVirtuoso
        data={rows}
        fixedHeaderContent={fixedHeaderContent}
        components={VirtuosoTableComponents}
        itemContent={(index, row) => (
          <React.Fragment>
            <TableRow>
              {rowContent(index, row)}
            </TableRow>
            <TableRow>
              <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={columns.length + 1}>
                <Collapse in={isRowExpanded(row.model_id)} timeout="auto" unmountOnExit>
                  {renderParamsTable(row.params)}
                </Collapse>
              </TableCell>
            </TableRow>
          </React.Fragment>
        )}
      />
    </Paper>
  );
}