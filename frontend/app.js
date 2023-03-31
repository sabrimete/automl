const predict_endpoint = 'http://0.0.0.0:8000/predict';
const train_endpoint = 'http://0.0.0.0:8000/train';
const save_endpoint = "http://0.0.0.0:8000/save_models"

// predict_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/predict' 
// train_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/train'
function uploadTrainFile() {
    const input = document.getElementById("trainFile");
    const file = input.files[0];
    const target_string = document.getElementById("target_string").value;
    const max_runtime_secs = document.getElementById("max_runtime_secs").value;
    const max_models = document.getElementById("max_models").value;
    const nfolds = document.getElementById("nfolds").value;
    const seed = document.getElementById("seed").value;
    const algoSelect = document.getElementById("algo-select");
    const selectedAlgos = Array.from(algoSelect.selectedOptions).map((option) => option.value);

    console.log("selectedalgos ", selectedAlgos);

    if (target_string == '') {
        alert('Please enter the name of the target column.');
        return;
    }
    if (max_models!= 0 && selectedAlgos.length > max_models) {
        alert('Max models value should be bigger than number of selected algorithms');
        return;
    }


    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_string", target_string);
    formData.append("max_runtime_secs", max_runtime_secs);
    formData.append("max_models", max_models);
    formData.append("nfolds", nfolds);
    formData.append("seed", seed);
    formData.append("include_algos", JSON.stringify(selectedAlgos));
    console.log("selectedalgos ", JSON.stringify(selectedAlgos));

    showTrainStatus('Uploading train file...');

    fetch(train_endpoint, {
    method: 'POST',
    body: formData
    })
    .then(response => {
        hideTrainStatus();
        if (response.ok) {
            return response.text(); // Return the response text here
        } else {
            console.error("Error uploading train file.");
        }
    })
    .then(text => {
        displayLeaderboard(text, max_models);
    })

    .catch(error => {
        hideTrainStatus();
        console.error(error);
    });
    }

function uploadPredictFile() {
    const input = document.getElementById("predictFile");

    const file = input.files[0];

    const formData = new FormData();
    formData.append("file", file);

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
function displayLeaderboard(jsonText, model_num) {
    var jsonObject = JSON.parse(jsonText);
    var jsonObject = JSON.parse(jsonObject);
    const leaderboard = document.createElement("table");
    const header = leaderboard.createTHead();
    const headerRow = header.insertRow();
  
    // Add the new column for the selection checkbox
    const selectHeaderCell = document.createElement("th");
    selectHeaderCell.textContent = "Select";
    headerRow.appendChild(selectHeaderCell);
  
    const columns = [
      "model_id",
      "rmse",
      "mse",
      "mae",
      "rmsle",
      "mean_residual_deviance",
    ];
  
    columns.forEach((column) => {
      const headerCell = document.createElement("th");
      headerCell.textContent = column;
      headerRow.appendChild(headerCell);
    });
  
    const tbody = document.createElement("tbody");
    leaderboard.appendChild(tbody);
  
    const numRows = Object.keys(jsonObject["model_id"]).length;
    for (let i = 0; i < numRows; i++) {
        const row = tbody.insertRow();
      
        const checkboxCell = row.insertCell();
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.setAttribute("data-model-id", jsonObject["model_id"][i.toString()]);
        checkboxCell.appendChild(checkbox);
      
        columns.forEach((column) => {
          const cell = row.insertCell();
          const key = column.replace(" ", "_").toLowerCase();
          if (jsonObject[key] && jsonObject[key][i.toString()]) {
            if (key === "model_id") {
              cell.textContent = jsonObject[key][i.toString()];
            } else {
              cell.textContent = parseFloat(jsonObject[key][i.toString()]).toFixed(4);
            }
          } else {
            cell.textContent = "";
          }
        });
      }
      
    const saveButton = document.createElement("button");
    saveButton.textContent = "Submit Selected Models";
    // saveButton.addEventListener("click", () => saveSelectedModels(leaderboard));
    saveButton.addEventListener("click", () => {
        // Display a confirmation dialog box
        const confirmed = confirm("Are you sure about the selected models?\nAll the models are gonna disappear after this step!");
        
        // Check if the user confirmed the action
        if (confirmed) {
            // Call the saveSelectedModels function
            saveSelectedModels(leaderboard, saveButton);
        } else {
            // Do nothing
        }
    });
    const container = document.getElementById("jsonLeaderboard");
    container.innerHTML = "";
    container.appendChild(leaderboard);
    container.appendChild(saveButton); // Append the save button after the table
  }

  
  function saveSelectedModels(table, saveButton) {
    const checkboxes = table.querySelectorAll("input[type='checkbox']");
    const selectedModels = [];
  
    checkboxes.forEach((checkbox) => {
      if (checkbox.checked) {
        const modelId = checkbox.getAttribute("data-model-id");
        selectedModels.push(modelId);
      }
    });
  
    if (selectedModels.length > 0) {
      console.log("Selected models:", selectedModels);
      sendSelectedModels(selectedModels, table, saveButton);
      // Send the selected model IDs to your server or perform any other actions
    } else {
      console.log("No models selected");
    }
  }
  
  async function sendSelectedModels(modelIds, table, saveButton) {
    console.log(JSON.stringify(modelIds))
    const response = await fetch(save_endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(modelIds),
    })
    .then(response => {
        hidePredictStatus();
        if (response.ok) {
            console.log("Selected models sent successfully");
            alert("Selected models are saved successfully");
            table.innerHTML = "";
            saveButton.remove();
            
            return response.text();
        } else {
            console.error("Error sending selected models");
        }
    })

    .catch(error => {
        hidePredictStatus();
        console.error(error);
    });
  }
  
  
  
  
// function displayLeaderboard(jsonText, model_num) {
//     var jsonObject = JSON.parse(jsonText);
//     var jsonObject = JSON.parse(jsonObject);
//     const leaderboard = document.createElement("table");
//     const header = leaderboard.createTHead();
//     const headerRow = header.insertRow();
  
//     const columns = [
//       "model_id",
//       "rmse",
//       "mse",
//       "mae",
//       "rmsle",
//       "mean_residual_deviance",
//     ];
  
//     columns.forEach((column) => {
//       const headerCell = document.createElement("th");
//       headerCell.textContent = column;
//       headerRow.appendChild(headerCell);
//     });
  
//     const tbody = document.createElement("tbody");
//     leaderboard.appendChild(tbody);
  
//     const numRows = Object.keys(jsonObject["model_id"]).length;
  
//     for (let i = 0; i < numRows; i++) {
//       const row = tbody.insertRow();
//       columns.forEach((column) => {
//         const cell = row.insertCell();
//         const key = column.replace(" ", "_").toLowerCase();
//         if (jsonObject[key] && jsonObject[key][i.toString()]) {
//           if (key === "model_id") {
//             cell.textContent = jsonObject[key][i.toString()];
//           } else {
//             cell.textContent = parseFloat(jsonObject[key][i.toString()]).toFixed(4);
//           }
//         } else {
//           cell.textContent = "";
//         }
//       });
//     }
  
//     const container = document.getElementById("jsonLeaderboard");
//     container.innerHTML = "";
//     container.appendChild(leaderboard);
//   }

//   function saveSelectedModels() {
//     const table = document.getElementById("leaderboard");
//     const checkboxes = table.querySelectorAll("input[type='checkbox']");
//     const selectedModelIds = [];
  
//     checkboxes.forEach((checkbox) => {
//       if (checkbox.checked) {
//         selectedModelIds.push(checkbox.value);
//       }
//     });
  
//     // Send the selectedModelIds to your server or use it as needed
//     console.log("Selected Model IDs: ", selectedModelIds);
//   }
  