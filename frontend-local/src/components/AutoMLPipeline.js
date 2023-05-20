// App.js
import React from 'react';
import UserMode from './UserMode';
import DeveloperMode from './DeveloperMode';
import ManualMode from './ManualMode';
import styles from './AutoMLPipeline.module.css';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import Button from '@mui/material/Button';

const theme = createTheme({
  palette: {
    primary: {
      // Purple and green play nicely together.
      main: '#ba68c8',
    },
    secondary: {
      // This is green.A700 as hex.
      main: '#4caf50',
    },
    third: {
      // This is green.A700 as hex.
      main: '#03a9f4',
    },
  },
});

export default function App() {
  const [mode, setMode] = React.useState('user');

  return (
    <div>
      <div className={styles.navbar}>
        <nav>
        <ThemeProvider theme={theme}>
          <Button style={{color: 'white', margin: 10}} color="primary" variant="contained" onClick={() => setMode('user')}>User Mode</Button>
          <Button style={{color: 'white', margin: 10}} color="secondary" variant="contained" onClick={() => setMode('developer')}>Developer Mode</Button>
          <Button style={{color: 'white', margin: 10}} color="third" variant="contained" onClick={() => setMode('manual')}>Manual Mode</Button>
          </ThemeProvider>
        </nav>
        
      </div>
      {mode === 'user' && <UserMode />}
      {mode === 'developer' && <DeveloperMode />}
      {mode === 'manual' && <ManualMode />}
    </div>
  );
}
