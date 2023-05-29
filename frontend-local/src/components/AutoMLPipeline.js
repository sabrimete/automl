// App.js
import * as React from 'react';
import UserMode from './UserMode';
import DeveloperMode from './DeveloperMode';
import ManualMode from './ManualMode';
import PredictMode from './Predict';
import styles from './AutoMLPipeline.module.css';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import Button from '@mui/material/Button';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Menu from '@mui/material/Menu';
import MenuIcon from '@mui/icons-material/Menu';
import Container from '@mui/material/Container';
import MenuItem from '@mui/material/MenuItem';
import AdbIcon from '@mui/icons-material/Adb';
import headerBackground from "../assets/header.png";
import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
  header: {
    backgroundImage: `url(${headerBackground})`,
  },
}));

const pages = ['user', 'developer', 'manual', 'predict'];

const theme = createTheme({
  palette: {
    user: {
      // Purple and green play nicely together.
      main: '#00bcd4',
    },
    developer: {
      // This is green.A700 as hex.
      main: '#009688',
    },
    manual: {
      // This is green.A700 as hex.
      main: '#4caf50',
    },
    predict: {
      main: '#8bc34a',
    },
  },
});

export default function App() {
  const classes = useStyles();

  const [mode, setMode] = React.useState('user');
  const [anchorElNav, setAnchorElNav] = React.useState(null);

  const handleOpenNavMenu = (event) => {
    setAnchorElNav(event.currentTarget);
  };

  const handleCloseNavMenu = () => {
    setAnchorElNav(null);
  };

  return (
    <div>
      <ThemeProvider theme={theme}>
      <AppBar position="static" color='secondary' className={classes.header} >
      <Container maxWidth="xl">
      <Toolbar disableGutters>
      <AdbIcon sx={{ display: { xs: 'none', md: 'flex' }, mr: 1 }} />
      <Typography
            variant="h4"
            noWrap
            component="a"
            href="/"
            sx={{
              mr: 2,
              display: { xs: 'none', md: 'flex' },
              fontFamily: 'monospace',
              fontWeight: 700,
              letterSpacing: '.3rem',
              color: 'inherit',
              textDecoration: 'none',
              margin: '10px',
            }}
          >
            ModelLab
          </Typography>
          <Box sx={{ flexGrow: 1, display: { xs: 'flex', md: 'none' } }}>
          {/* <Menu
              id="menu-appbar"
              anchorEl={anchorElNav}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'left',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'left',
              }}
              open={Boolean(anchorElNav)}
              onClose={handleCloseNavMenu}
              sx={{
                display: { xs: 'block', md: 'none' },
              }}
            >
              {pages.map((page) => (
                
                <MenuItem key={page} onClick={handleCloseNavMenu}>
                  <Typography textAlign="center" color='white'>{page}</Typography>
                </MenuItem>
              ))}
            </Menu> */}
          </Box>
          <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' } }}>
            {pages.map((page) => (
              <Button
                key={page}
                color={page}
                variant="contained"
                onClick={() => setMode(page)}
                sx={{ my: 2, color: 'white', margin: 1,  display: 'block'}}
              >
                {page}
              </Button>
            ))}
            
          </Box>
        </Toolbar>
        </Container>
      </AppBar>
      </ThemeProvider>
      {mode === 'user' && <UserMode />}
      {mode === 'developer' && <DeveloperMode />}
      {mode === 'manual' && <ManualMode />}
      {mode === 'predict' && <PredictMode />}
    </div>
  );
}
