import { MnistData } from './data.js';
import { MNISTModel } from './model.js';

let model;
let isDrawing = false;
let chart = null;

// Canvas setup
const drawingCanvas = document.getElementById('drawingCanvas');
const drawingCtx = drawingCanvas.getContext('2d');
const debugCanvas = document.getElementById('debugCanvas');
const debugCtx = debugCanvas.getContext('2d');
const predictDiv = document.getElementById('predictionResult');

// Set white background
drawingCtx.fillStyle = 'white';
drawingCtx.fillStyle = 'black';
drawingCtx.lineWidth = 20;
drawingCtx.lineCap = 'round';

// Drawing handlers
drawingCanvas.addEventListener('mousedown', startDrawing);
drawingCanvas.addEventListener('mousemove', draw);
drawingCanvas.addEventListener('mouseup', stopDrawing);
drawingCanvas.addEventListener('mouseout', stopDrawing);

document.getElementById('clearButton').addEventListener('click', clearCanvas);
document.getElementById('predictButton').addEventListener('click', predict);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = drawingCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    drawingCtx.beginPath();
    drawingCtx.moveTo(x, y);
    drawingCtx.lineTo(x, y);
    drawingCtx.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

function clearCanvas() {
    drawingCtx.fillStyle = 'white';
    drawingCtx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    drawingCtx.fillStyle = 'black';
    predictDiv.innerHTML = 'Draw a digit to see the prediction.';
    if (chart) {
        chart.destroy();
    }
}

async function predict() {
    // Scale down to 28x28
    debugCtx.drawImage(drawingCanvas, 0, 0, debugCanvas.width, debugCanvas.height);
    
    // Get image data and prepare for model
    const imageData = debugCtx.getImageData(0, 0, debugCanvas.width, debugCanvas.height);
    const input = new Float32Array(28 * 28);
    
    // Convert to grayscale and normalize
    for (let i = 0; i < imageData.data.length; i += 4) {
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];
        const gray = (r + g + b) / 3;
        input[i / 4] = 1 - gray / 255;
    }

    // Make prediction
    const tensor = tf.tensor(input, [28, 28, 1]);
    // log the non zero pixels in input
    const nonZeroPixels = input.filter(value => value > 0).length;
    console.log('Non-zero pixels:', nonZeroPixels);
    const prediction = await model.predict(tensor);
    const probabilities = await prediction.data();
    
    updateChart(probabilities);
    predictDiv.innerHTML = `Predicted digit: ${probabilities.indexOf(Math.max(...probabilities))}`;
    tensor.dispose();
    prediction.dispose();
}

function updateChart(probabilities) {
    const ctx = document.getElementById('predictionChart');
    
    if (chart) {
        chart.destroy();
    }
    
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            datasets: [{
                label: 'Probability',
                data: Array.from(probabilities),
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// Initialize and train model
async function init() {
    console.log('Loading MNIST data...');
    const data = new MnistData();
    console.log("object created - Loading data...");
    await data.load();
    
    console.log('Creating and training model...');
    model = new MNISTModel();
    console.log("training model...");
    const [xTrain, yTrain] = data.getTrainData();
    
    await model.train(xTrain, yTrain, 25, 2048);
    console.log('Training complete! You can now draw digits and make predictions.');
}

init();