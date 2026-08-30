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
  const rows = [...document.querySelectorAll('.probability-row')];
  let drawing = false;
  let lastPoint = null;
  let debounceTimer = null;
  let requestId = 0;
  let hasInk = false;

  function configureCanvas() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#000000';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = Number(slider.value);
  }

  function pointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    const source = event.touches ? event.touches[0] : event;
    return {
      x: (source.clientX - rect.left) * (canvas.width / rect.width),
      y: (source.clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  function drawDot(point) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#000000';
    ctx.fill();
  }

  function startDrawing(event) {
    if (event.cancelable) event.preventDefault();
    drawing = true;
    lastPoint = pointFromEvent(event);
    drawDot(lastPoint);
    hasInk = true;
  }

  function moveDrawing(event) {
    if (!drawing) return;
    if (event.cancelable) event.preventDefault();
    const point = pointFromEvent(event);
    ctx.beginPath();
    ctx.moveTo(lastPoint.x, lastPoint.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPoint = point;
    hasInk = true;
    queuePrediction();
  }

  function endDrawing(event) {
    if (!drawing) return;
    if (event && event.cancelable) event.preventDefault();
    drawing = false;
    lastPoint = null;
    queuePrediction();
  }

  function queuePrediction() {
    if (!window.MODEL_READY || !hasInk) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(predictCanvas, 200);
  }

  async function predictCanvas() {
    const thisRequest = ++requestId;
    loading.hidden = false;
    message.textContent = 'Processing canvas…';
    try {
      const response = await fetch('/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: canvas.toDataURL('image/png') }),
      });
      const data = await response.json();
      if (thisRequest !== requestId) return;
      if (!response.ok) throw new Error(data.error || 'Prediction request failed.');
      updateResults(data.prediction, data.probabilities);
      message.textContent = 'Prediction updated in realtime.';
      status.className = 'status ready';
      status.innerHTML = '<i></i>Prediction Ready';
    } catch (error) {
      if (thisRequest !== requestId) return;
      message.textContent = error.message;
      status.className = 'status error';
      status.innerHTML = '<i></i>Prediction Error';
    } finally {
      if (thisRequest === requestId) loading.hidden = true;
    }
  }

  function updateResults(digit, probabilities) {
    prediction.textContent = digit;
    const highest = Math.max(...probabilities);
    rows.forEach((row, digitIndex) => {
      const value = probabilities[digitIndex] || 0;
      row.querySelector('.bar-fill').style.width = `${value * 100}%`;
      row.querySelector('output').textContent = `${(value * 100).toFixed(1)}%`;
      row.classList.toggle('highest', value === highest);
    });
  }

  function resetResults() {
    requestId += 1;
    clearTimeout(debounceTimer);
    hasInk = false;
    prediction.textContent = '-';
    message.textContent = 'Draw a digit to begin.';
    loading.hidden = true;
    rows.forEach((row) => {
      row.querySelector('.bar-fill').style.width = '0%';
      row.querySelector('output').textContent = '0.0%';
      row.classList.remove('highest');
    });
  }

  slider.addEventListener('input', () => { ctx.lineWidth = Number(slider.value); brushValue.value = `${slider.value} px`; brushValue.textContent = `${slider.value} px`; });
  clearButton.addEventListener('click', () => { configureCanvas(); resetResults(); });
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', moveDrawing);
  canvas.addEventListener('mouseup', endDrawing);
  canvas.addEventListener('mouseleave', endDrawing);
  canvas.addEventListener('touchstart', startDrawing, { passive: false });
  canvas.addEventListener('touchmove', moveDrawing, { passive: false });
  canvas.addEventListener('touchend', endDrawing, { passive: false });
  canvas.addEventListener('touchcancel', endDrawing, { passive: false });
  configureCanvas();
})();
