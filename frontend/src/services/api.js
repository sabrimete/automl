const predict_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/predict';
const train_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/train';
const save_endpoint = 'https://backend-6r72er7ega-uc.a.run.app/save_models';

export async function uploadTrainFile(formData) {
  try {
    const response = await fetch(train_endpoint, {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const text = await response.text();
      return { success: true, data: text };
    } else {
      console.error('Error uploading train file.');
      return { success: false, error: 'Error uploading train file.' };
    }
  } catch (error) {
    console.error(error);
    return { success: false, error };
  }
}

export async function uploadPredictFile(formData) {
  try {
    const response = await fetch(predict_endpoint, {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      console.log('Predict file uploaded successfully.');
      return { success: true };
    } else {
      console.error('Error uploading predict file.');
      return { success: false, error: 'Error uploading predict file.' };
    }
  } catch (error) {
    console.error(error);
    return { success: false, error };
  }
}

export async function sendSelectedModels(modelIds) {
  try {
    const response = await fetch(save_endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(modelIds),
    });

    if (response.ok) {
      console.log('Selected models sent successfully');
      return { success: true };
    } else {
      console.error('Error sending selected models');
      return { success: false, error: 'Error sending selected models' };
    }
  } catch (error) {
    console.error(error);
    return { success: false, error };
  }
}
