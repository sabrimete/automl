const predict_endpoint = 'http://0.0.0.0:8000/predict';
const train_endpoint = 'http://0.0.0.0:8000/train';
// const predict_endpoint = 'http://backend:8000/predict';
// const train_endpoint = 'http://backend:8000/train';
// predict_endpoint = 'http://host.docker.internal:8000/predict' 
// train_endpoint = 'http://host.docker.internal:8000/train'
function uploadTrainFile() {
    const input = document.getElementById("trainFile");
    const file = input.files[0];
    const targetString = document.getElementById("targetString").value;

    if (targetString == '') {
        alert('Please enter the name of the target column.');
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("targetString", targetString);

    showTrainStatus('Uploading train file...');

    fetch(train_endpoint, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        hideTrainStatus();
        if (response.ok) {
            console.log("Train file uploaded successfully.");
        } else {
            console.error("Error uploading train file.");
        }
    })
    .catch(error => {
        hideTrainStatus();
        console.error(error);
    });
}


function uploadPredictFile() {
    const input = document.getElementById("predictFile");
    const file = input.files[0];
    const predictString = document.getElementById("predictString").value;

    if (predictString == '') {
        alert('Please enter the name of the string column.');
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("predictString", predictString);

    showPredictStatus('Uploading predict file...');

    fetch(predict_endpoint, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        hidePredictStatus();
        if (response.ok) {
            console.log("Predict file uploaded successfully.");
        } else {
            console.error("Error uploading predict file.");
        }
    })
    .catch(error => {
        hidePredictStatus();
        console.error(error);
    });
}

function updateTrainFileLabel() {
    const input = document.getElementById("trainFile");
    const label = document.getElementById("trainFileLabel");
    label.textContent = input.files[0].name;
}

function updatePredictFileLabel() {
    const input = document.getElementById("predictFile");
    const label = document.getElementById("predictFileLabel");
    label.textContent = input.files[0].name;
}

function showTrainStatus(message) {
    const status = document.getElementById("trainStatus");
    status.textContent = message;
    status.innerHTML += '<div class="spinner"></div>';
}

function hideTrainStatus() {
    const status = document.getElementById("trainStatus");
    status.textContent = '';
}

function showPredictStatus(message) {
    const status = document.getElementById("predictStatus");
    status.textContent = message;
    status.innerHTML += '<div class="spinner"></div>';
}

function hidePredictStatus() {
    const status = document.getElementById("predictStatus");
    status.textContent = '';
}
