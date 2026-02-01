const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const fileMeta = document.getElementById('fileMeta');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const labelEl = document.getElementById('label');
const confidenceEl = document.getElementById('confidence');
const topkEl = document.getElementById('topk');
const dropzone = document.getElementById('dropzone');

let currentFile = null;

dropzone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropzone.classList.add('hover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('hover');
});

dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropzone.classList.remove('hover');
  const file = event.dataTransfer.files[0];
  if (file) {
    handleFile(file);
  }
});

fileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    handleFile(file);
  }
});

predictBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  predictBtn.disabled = true;
  predictBtn.textContent = 'Running...';
  labelEl.textContent = 'Analyzing';
  confidenceEl.textContent = '?';
  topkEl.innerHTML = '';

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed');
    }

    labelEl.textContent = data.label;
    confidenceEl.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    topkEl.innerHTML = data.topk
      .map(
        (item) =>
          `<span><strong>${item.class_name}</strong><em>${(item.confidence * 100).toFixed(2)}%</em></span>`
      )
      .join('');
  } catch (error) {
    labelEl.textContent = 'Error';
    confidenceEl.textContent = error.message;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Run classification';
  }
});

resetBtn.addEventListener('click', () => {
  currentFile = null;
  fileInput.value = '';
  previewImage.style.display = 'none';
  previewImage.src = '';
  fileMeta.textContent = '';
  predictBtn.disabled = true;
  resetBtn.disabled = true;
  labelEl.textContent = 'Awaiting input';
  confidenceEl.textContent = '?';
  topkEl.innerHTML = '';
});

function handleFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (event) => {
    previewImage.src = event.target.result;
    previewImage.style.display = 'block';
  };
  reader.readAsDataURL(file);

  fileMeta.textContent = `${file.name} ? ${(file.size / 1024).toFixed(1)} KB`;
  predictBtn.disabled = false;
  resetBtn.disabled = false;
}
