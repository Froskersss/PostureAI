const video   = document.getElementById('webcam');
const canvas  = document.getElementById('overlay');
const ctx     = canvas.getContext('2d');
const status  = document.getElementById('status');
const display = document.getElementById('shoulder-val');

// Shoulder landmark indices in MediaPipe
const L_SHOULDER = 11;
const R_SHOULDER = 12;

function getShoulderAngle(landmarks) {
  const L = landmarks[L_SHOULDER];
  const R = landmarks[R_SHOULDER];
  const dx = R.x - L.x;
  const dy = R.y - L.y;
  let angle = Math.abs(Math.atan2(dy, dx) * (180 / Math.PI));
  if (angle > 90) angle = 180 - angle;
  return Math.round(angle);
}

function drawSkeleton(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const W = canvas.width;
  const H = canvas.height;

  const L = landmarks[L_SHOULDER];
  const R = landmarks[R_SHOULDER];

  const lx = L.x * W;
  const ly = L.y * H;
  const rx = R.x * W;
  const ry = R.y * H;

  // Draw shoulder line
  ctx.beginPath();
  ctx.moveTo(lx, ly);
  ctx.lineTo(rx, ry);
  ctx.strokeStyle = '#c8f060';
  ctx.lineWidth = 3;
  ctx.stroke();

  // Draw dots
  [{ x: lx, y: ly }, { x: rx, y: ry }].forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  });
}

const pose = new Pose({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});

pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: false,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

pose.onResults((results) => {
  if (!results.poseLandmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const landmarks = results.poseLandmarks;
  drawSkeleton(landmarks);

  const angle = getShoulderAngle(landmarks);
  display.innerHTML = angle + '<span class="unit">°</span>';

  // Shared output for teammates
  window.postureCV = { shoulderAngle: angle };

  console.log('Shoulder angle:', angle);
});

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });

    video.srcObject = stream;

    video.onloadedmetadata = () => {
      const camera = new Camera(video, {
        onFrame: async () => {
          await pose.send({ image: video });
        },
        width: 640,
        height: 480,
      });

      camera.start();
      status.textContent = 'Camera running — stand in frame';
      status.className = 'ok';
    };

  } catch (err) {
    status.textContent = 'Camera error: ' + err.message;
    status.className = 'error';
    console.error(err);
  }
}

startCamera();

// to run: http://localhost:8080/postureai/

