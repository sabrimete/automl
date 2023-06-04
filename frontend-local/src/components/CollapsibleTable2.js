import React, { useState, useEffect } from "react";
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import IconButton from '@mui/material/IconButton';
import Paper from '@mui/material/Paper';
import { TableVirtuoso } from 'react-virtuoso';
import Checkbox from '@mui/material/Checkbox';
import Modal from '@mui/material/Modal';
import Preview from './Preview';
import Box from '@mui/material/Box';
import PreviewIcon from '@mui/icons-material/Preview';

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

const columns = [
  {
    width: 30,
    label: 'select',
    dataKey: 'select',
  },
  {
    width: 500,
    label: 'model_id',
    dataKey: 'model_id',
  },
  {
    width: 150,
    label: 'rmse',
    dataKey: 'rmse',
  },
  {
    width: 140,
    label: 'mse',
    dataKey: 'mse',
  },
  {
    width: 140,
    label: 'mae',
    dataKey: 'mae',
  },
  {
    width: 140,
    label: 'rmsle',
    dataKey: 'rmsle',
  },
  {
    width: 160,
    label: 'mean_residual_deviance',
    dataKey: 'mean_residual_deviance',
  },
  {
    width: 70,
    label: 'parameters',
    dataKey: 'parameters',
  },
];


export default function CollapsibleTable2({response, onSelectChange}) {
    const [selected, setSelected] = useState([]);
    const [expandedRows, setExpandedRows] = useState([]);
    const [openModal, setOpenModal] = useState(false);
    const [ID, setID] = useState();
    const [sortColumn, setSortColumn] = useState("");
    const [sortOrder, setSortOrder] = useState("asc");
    

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
      console.log(selected);
    };

    const isSelected = (id) => selected.indexOf(id) !== -1;

    const handleRowExpand = (model_id) => {
      const isExpanded = expandedRows.includes(model_id);
      const newExpandedRows = isExpanded
        ? expandedRows.filter((id) => id !== model_id)
        : [...expandedRows, model_id];
      setExpandedRows(newExpandedRows);
    };

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


  function fixedHeaderContent() {
    const handleSort = (column) => {
      if (sortColumn === column.dataKey) {
        // If the same column is clicked again, toggle the sort order
        setSortOrder(sortOrder === "asc" ? "desc" : "asc");
      } else {
        // If a new column is clicked, set it as the sort column with ascending order
        setSortColumn(column.dataKey);
        setSortOrder("asc");
      }
    };

    return (
      <TableRow>
        {updatedColumns.map((column, index) => (
          <TableCell
            key={index}
            variant="head"
            align={column.numeric || false ? 'right' : 'left'}
            style={{ width: column.width }}
            sx={{
              backgroundColor: 'background.paper',
            }}
          >
            <div
            style={{ cursor: "pointer" }}
            onClick={() => handleSort(column)}
          >
            {column.label}
            {sortColumn === column.dataKey && (
              <span>{sortOrder === "asc" ? " ▲" : " ▼"}</span>
            )}
            {sortColumn !== column.dataKey && " ▼"}
          </div>
          </TableCell>
        ))}
      </TableRow>
    );
  }

  function rowContent(_index, row) {
    console.log(row);
    return (
      <React.Fragment>
        <TableCell padding="checkbox" style={{ width: updatedColumns[0].width }}>
            <Checkbox
              color="primary"
              align="center"
              onClick={(event) => handleSelect(event, row.model_id)}
              checked={isSelected(row.model_id)}
            />
          </TableCell>
          {updatedColumns.slice(1,-1).map((column, index) => (
            <TableCell
              key={index}
              align={column.numeric || false ? "right" : "left"}
              style={{
                width: column.width,
                padding: "16px",
              }}
            >
              <Box width="100%">{row[column.dataKey]}</Box>
            </TableCell>
          ))}

          <TableCell padding="checkbox" style={{ width: updatedColumns[updatedColumns.length - 1].width }}>
          <IconButton
            aria-label="expand row"
            size="small"
            onClick={() => {
              handleRowExpand(row.model_id);
            } }
          >
          <PreviewIcon margin="auto" onClick={() => {
            setOpenModal(true);
            setID(row.model_id);
          }}/>
          </IconButton>
        </TableCell>
        
      </React.Fragment>
    );
  }
  const sortedRows = React.useMemo(() => {
    const comparator = (a, b) => {
      const aValue = a[sortColumn];
      const bValue = b[sortColumn];
      if (aValue < bValue) {
        return sortOrder === "asc" ? -1 : 1;
      } else if (aValue > bValue) {
        return sortOrder === "asc" ? 1 : -1;
      } else {
        return 0;
      }
    };
  
    return sortColumn ? rows.slice().sort(comparator) : rows;
  }, [rows, sortColumn, sortOrder]);

  const filteredColumns = columns.filter((column) => {
    return rows.some((row) => row[column.dataKey] !== null);
  });
  
  const updatedColumns = [...filteredColumns];
  
  return (
    <div>
        <Paper style={{ height: 400, width: '100%' }}>
        <TableVirtuoso
          data={sortedRows}
          components={VirtuosoTableComponents}
          fixedHeaderContent={fixedHeaderContent}
          itemContent={rowContent}
          />
      </Paper>
      <Modal
        open={openModal}
        onClose={() => setOpenModal(false)}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
        >
        <Preview setOpenModal={setOpenModal} data={ sortedRows.filter(row => row.model_id === ID)}/>
      </Modal>
    </div>
  );
}
