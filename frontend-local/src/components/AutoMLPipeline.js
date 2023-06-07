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
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Menu from '@mui/material/Menu';
import MenuIcon from '@mui/icons-material/Menu';
import MenuItem from '@mui/material/MenuItem';
import AdbIcon from '@mui/icons-material/Adb';
import headerBackground from "../assets/header2.jpg";
import cobraLogo from "../assets/cobra-removebg-preview.png";
import { makeStyles } from "@material-ui/core/styles";
import { withTheme } from '@emotion/react';



const useStyles = makeStyles((theme) => ({
  header: {
    backgroundImage: `linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url(${headerBackground})`,
    backgroundSize: 'cover',
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center',
    paddingTop: '5px',
    paddingDown: '5px',
    boxSizing: 'border-box',
  },
  logo: {
    backgroundImage: `url(${cobraLogo})`,
    backgroundRepeat: 'no-repeat',
    backgroundSize: 'contain',
    filter: 'brightness(0) invert(1)',
    width: '45px',
    height: '45px',
    marginRight: '1rem',
    marginLeft: '2rem',
    color: "white",
    zIndex: 1,
  },
}));


const pages = ['Domain-Expert', 'Data-Scientist', 'Manual', 'Predict'];

const theme = createTheme({
  palette: {
    'Domain-Expert': {
      // Purple and green play nicely together.
      main: '#00bcd4',
    },
    'Data-Scientist': {
      // This is green.A700 as hex.
      main: '#009688',
    },
    'Manual': {
      // This is green.A700 as hex.
      main: '#4caf50',
    },
    'Predict': {
      main: '#8bc34a',
    },
  },
});

export default function App() {
  const classes = useStyles();

  const [mode, setMode] = React.useState('Domain-Expert');
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
      <div className={classes.logo} sx={{ display: { xs: 'none', md: 'flex' }, marginRight: 1 }} />
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
              color: 'white',
              textDecoration: 'none',
              margin: '10px',
            }}
          >
            CobraLab
          </Typography>
          {/* <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' } }}>
            
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
            
          </Box> */}
          <Box sx={{ flexGrow: 1, display: { xs: 'flex', md: 'none' } }}>
            <IconButton
              size="large"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleOpenNavMenu}
              color="inherit"
            >
              <MenuIcon />
            </IconButton>
            <Menu
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
                <MenuItem key={page} onClick={() => setMode(page)}>
                  <Typography textAlign="center">{page}</Typography>
                </MenuItem>
              ))}
            </Menu>
          </Box>
          <Typography
            variant="h5"
            noWrap
            component="a"
            href=""
            sx={{
              mr: 2,
              display: { xs: 'flex', md: 'none' },
              flexGrow: 1,
              fontFamily: 'monospace',
              fontWeight: 700,
              letterSpacing: '.3rem',
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            CobraLab
          </Typography>
          <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' } }}>
            {pages.map((page) => (
              <Button
                key={page}
                onClick={() => setMode(page)}
                color={page}
                variant="contained"
                sx={{ my: 2, color: 'white', margin: 1,  display: 'block' }}
              >
                {page}
              </Button>
            ))}
          </Box>
        </Toolbar>
        </Container>
      </AppBar>
      </ThemeProvider>
      {mode === 'Domain-Expert' && <UserMode />}
      {mode === 'Data-Scientist' && <DeveloperMode />}
      {mode === 'Manual' && <ManualMode />}
      {mode === 'Predict' && <PredictMode />}
    </div>
  );
}
