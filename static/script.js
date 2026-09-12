(() => {
  const canvas = document.querySelector('#drawing-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const slider = document.querySelector('#brush-size');
  const brushValue = document.querySelector('#brush-value');
  const clearButton = document.querySelector('#clear-button');
  const prediction = document.querySelector('#prediction');
  const message = document.querySelector('#message');
  const loading = document.querySelector('#loading');
  const status = document.querySelector('#prediction-status');
  const probabilityList = document.querySelector('#probability-list');
  let model = null;
  let drawing = false;
  let lastPoint = null;
  let debounceTimer = null;
  let predictionVersion = 0;
  let predictionInProgress = false;
  let predictionQueued = false;
  let hasInk = false;

  probabilityList.innerHTML = Array.from({ length: 10 }, (_, digit) => `
    <div class="probability-row"><strong>${digit}</strong><div class="bar-track"><div class="bar-fill"></div></div><output>0.0%</output></div>
  `).join('');
  const rows = [...document.querySelectorAll('.probability-row')];

  function configureCanvas() {
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#000'; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.lineWidth = Number(slider.value);
  }

  function setStatus(kind, label) { status.className = `status ${kind}`; status.innerHTML = `<i></i>${label}`; }

  function pointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    const source = (event.touches && event.touches[0]) || event;
    return { x: (source.clientX - rect.left) * canvas.width / rect.width, y: (source.clientY - rect.top) * canvas.height / rect.height };
  }

  function startDrawing(event) {
    if (event.cancelable) event.preventDefault();
    drawing = true; lastPoint = pointFromEvent(event);
    ctx.beginPath(); ctx.arc(lastPoint.x, lastPoint.y, ctx.lineWidth / 2, 0, Math.PI * 2); ctx.fillStyle = '#000'; ctx.fill();
    hasInk = true;
  }

  function moveDrawing(event) {
    if (!drawing) return;
    if (event.cancelable) event.preventDefault();
    const point = pointFromEvent(event);
    ctx.beginPath(); ctx.moveTo(lastPoint.x, lastPoint.y); ctx.lineTo(point.x, point.y); ctx.stroke();
    lastPoint = point; hasInk = true; queuePrediction();
  }

  function endDrawing(event) {
    if (!drawing) return;
    if (event?.cancelable) event.preventDefault();
    drawing = false; lastPoint = null; queuePrediction();
  }

  function queuePrediction() {
    if (!model || !hasInk) return;
    if (predictionInProgress) { predictionQueued = true; return; }
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(predictCanvas, 200);
  }

  function makeInputTensor() {
    const source = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const ink = new Uint8ClampedArray(canvas.width * canvas.height);
    let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;
    for (let y = 0; y < canvas.height; y += 1) for (let x = 0; x < canvas.width; x += 1) {
      const pixel = (y * canvas.width + x) * 4;
      const value = 255 - (0.299 * source[pixel] + 0.587 * source[pixel + 1] + 0.114 * source[pixel + 2]);
      ink[y * canvas.width + x] = value;
      if (value > 20) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
    }
    if (maxX < 0) throw new Error('Canvas is empty. Draw a digit before predicting.');

    minX = Math.max(0, minX - 4); maxX = Math.min(canvas.width - 1, maxX + 4);
    minY = Math.max(0, minY - 4); maxY = Math.min(canvas.height - 1, maxY + 4);
    const cropWidth = maxX - minX + 1, cropHeight = maxY - minY + 1;
    const crop = document.createElement('canvas'); crop.width = cropWidth; crop.height = cropHeight;
    const cropContext = crop.getContext('2d'); const cropPixels = cropContext.createImageData(cropWidth, cropHeight);
    for (let y = 0; y < cropHeight; y += 1) for (let x = 0; x < cropWidth; x += 1) {
      const value = ink[(minY + y) * canvas.width + minX + x], index = (y * cropWidth + x) * 4;
      cropPixels.data[index] = cropPixels.data[index + 1] = cropPixels.data[index + 2] = value; cropPixels.data[index + 3] = 255;
    }
    cropContext.putImageData(cropPixels, 0, 0);
    const scale = 20 / Math.max(cropWidth, cropHeight);
    const width = Math.max(1, Math.round(cropWidth * scale)), height = Math.max(1, Math.round(cropHeight * scale));
    const resized = document.createElement('canvas'); resized.width = width; resized.height = height;
    resized.getContext('2d').drawImage(crop, 0, 0, width, height);
    const pixels = resized.getContext('2d').getImageData(0, 0, width, height).data;
    const centered = new Float32Array(28 * 28), startX = Math.floor((28 - width) / 2), startY = Math.floor((28 - height) / 2);
    for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) centered[(startY + y) * 28 + startX + x] = pixels[(y * width + x) * 4];

    let sum = 0, weightedX = 0, weightedY = 0;
    centered.forEach((value, index) => { sum += value; weightedX += (index % 28) * value; weightedY += Math.floor(index / 28) * value; });
    const offsetX = Math.round(13.5 - weightedX / sum), offsetY = Math.round(13.5 - weightedY / sum), aligned = new Float32Array(28 * 28);
    for (let y = 0; y < 28; y += 1) for (let x = 0; x < 28; x += 1) {
      const targetX = x + offsetX, targetY = y + offsetY;
      if (targetX >= 0 && targetX < 28 && targetY >= 0 && targetY < 28) aligned[targetY * 28 + targetX] = centered[y * 28 + x] / 255;
    }
    return tf.tensor4d(aligned, [1, 28, 28, 1], 'float32');
  }

  async function predictCanvas() {
    if (predictionInProgress) { predictionQueued = true; return; }
    predictionInProgress = true;
    const version = ++predictionVersion; loading.hidden = false; message.textContent = 'Processing canvas…';
    let input = null;
    let output = null;
    try {
      input = makeInputTensor(); output = model.predict(input);
      const probabilities = Array.from(await output.data());
      if (version !== predictionVersion) return;
      const digit = probabilities.indexOf(Math.max(...probabilities)); prediction.textContent = digit;
      rows.forEach((row, index) => { const value = probabilities[index]; row.querySelector('.bar-fill').style.width = `${value * 100}%`; row.querySelector('output').textContent = `${(value * 100).toFixed(1)}%`; row.classList.toggle('highest', index === digit); });
      message.textContent = 'Prediction updated in realtime.';
    } catch (error) { if (version === predictionVersion) message.textContent = error.message || 'Prediction failed.'; }
    finally {
      if (input) input.dispose();
      if (output) output.dispose();
      predictionInProgress = false;
      if (predictionQueued) { predictionQueued = false; queuePrediction(); }
      if (version === predictionVersion) loading.hidden = true;
    }
  }

  function resetResults() {
    predictionVersion += 1;
    clearTimeout(debounceTimer);
    predictionQueued = false;
    hasInk = false;
    prediction.textContent = '-';
    loading.hidden = true;
    message.textContent = model ? 'Draw a digit to begin.' : 'Loading TensorFlow.js model…';
    rows.forEach(row => { row.querySelector('.bar-fill').style.width = '0%'; row.querySelector('output').textContent = '0.0%'; row.classList.remove('highest'); });
  }

  async function loadModel() {
    try {
      await tf.ready(); model = await tf.loadLayersModel('./model/model.json');
      const warmup = model.predict(tf.zeros([1, 28, 28, 1])); warmup.dispose();
      setStatus('ready', 'Prediction Ready'); message.textContent = 'Draw a digit to begin.';
    } catch (error) { setStatus('error', 'Model Unavailable'); message.textContent = 'Model belum tersedia. Jalankan python convert_model.py, lalu push folder model.'; console.error(error); }
  }

  slider.addEventListener('input', () => { ctx.lineWidth = Number(slider.value); brushValue.value = `${slider.value} px`; brushValue.textContent = `${slider.value} px`; });
  clearButton.addEventListener('click', () => { configureCanvas(); resetResults(); });
  canvas.addEventListener('mousedown', startDrawing); canvas.addEventListener('mousemove', moveDrawing); canvas.addEventListener('mouseup', endDrawing); canvas.addEventListener('mouseleave', endDrawing);
  canvas.addEventListener('touchstart', startDrawing, { passive: false }); canvas.addEventListener('touchmove', moveDrawing, { passive: false }); canvas.addEventListener('touchend', endDrawing, { passive: false }); canvas.addEventListener('touchcancel', endDrawing, { passive: false });
  configureCanvas(); loadModel();
})();
