import * as React from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';
import MuiInput from '@mui/material/Input';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';

const Input = styled(MuiInput)`
  width: 42px;
`;

export default function RangeSlider({title, min, max, step, onSlidChange, onStepChange}) {
  const [value, setValue] = React.useState([10, 30]);
  const [value1, setValue1] = React.useState('2');

  const handleChange = (event, newValue) => {
    setValue(newValue);
    console.log(value);
    onSlidChange(value);
  };

  const handleInputChange = (event) => {
    const newValue = event.target.value === '' ? '' : Number(event.target.value);
    setValue1(newValue);
    console.log(newValue);
    onStepChange(newValue);
  };

  return (
    <Box sx={{ width: 300, margin: "auto"}}>
      <Typography id="input-slider" gutterBottom>
        {title}
      </Typography>
      <Grid container spacing={2} alignItems="center">
      <Grid item xs>
      <Slider
        min={min}
        step={step}
        max={max}
        getAriaLabel={() => 'Temperature range'}
        value={value}
        onChange={handleChange}
        valueLabelDisplay="auto"
      />
      </Grid>
      <Grid item xs>
        
      <Input
        value={value1}
        size="small"
        onChange={handleInputChange}
        inputProps={{
          step: 1,
          min: 0,
          max: 3,
          type: 'number',
          'aria-labelledby': 'input-slider',
        }}
      />
      </Grid>
      </Grid>
    </Box>

  );
}