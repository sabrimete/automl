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


const Preview = (props) => {

    console.log(props.data[0]);

    return (
    <div style={styles.container}>
      <Table>
        <TableBody>
          {Object.entries(props.data[0].params).map(([key, value]) => (
            <TableRow key={key}>
              <TableCell component="th" scope="row">
                {key}
              </TableCell>
              <TableCell>{value}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default Preview;

const styles = {
  container: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: 'auto',
    width: '70vh',
    background: 'white',
    padding: '100px',
    margin: 'auto',
    marginTop: '200px',
    borderRadius: '20px',
  },
  content: {
    background: 'white',
    padding: '20px',
  },
};
