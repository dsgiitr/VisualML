import * as tfvis from "@tensorflow/tfjs-vis";
const statusElement = document.getElementById("status");
const messageElement = document.getElementById("message");
const imagesElement = document.getElementById("images");
const visualiseElement = document.getElementById("log");

export function logStatus(message) {
    statusElement.innerText = message;
}

export function logVisualise(message) {
    visualiseElement.innerText = message;
}
export function trainingLog(message) {
    messageElement.innerText = `${message}\n`;
    console.log(message);
}

export function showTestResults(batch, predictions, labels) {
    const testExamples = batch.xs.shape[0];
    imagesElement.innerHTML = "";
    for (let i = 0; i < testExamples; i++) {
        const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);
        const div = document.createElement("div");
        div.className = "pred-container";

        const canvas = document.createElement("canvas");
        canvas.className = "prediction-canvas";
        draw(image.flatten(), canvas);

        const pred = document.createElement("div");

        const prediction = predictions[i];
        const label = labels[i];
        const correct = prediction === label;

        pred.className = `pred ${correct ? "pred-correct" : "pred-incorrect"}`;
        pred.innerText = `pred: ${prediction}`;

        div.appendChild(pred);
        div.appendChild(canvas);

        imagesElement.appendChild(div);
    }
}

export function showLayer(output, div) {
    div.innerHTML = "";
    const numfilters = output.shape[3];
    const size = output.shape[1];
    for (let i = 0; i < numfilters; i++) {
        const image = output.slice([0, 0, 0, i], [1, size, size, 1]);
        const canvas = document.createElement("canvas");
        canvas.className = "layer-canvas";
        canvas.style.marginTop = "1em";
        canvas.style.marginBottom = "1em";
        show(image.flatten(), canvas, size);
        div.appendChild(canvas);
    }
}

export function showDense(output, div) {
    div.innerHTML = "";
    const numunits = output.shape[1];
    for (let i = 0; i < numunits; i++) {
        const unit = output.slice([0, i], [1, 1]);
        const canvas = document.createElement("canvas");
        canvas.className = "layer-canvas";
        canvas.style.marginTop = "1em";
        canvas.style.marginBottom = "1em";
        showFC(unit.flatten(), canvas);
        div.appendChild(canvas);
    }
}

const lossLabelElement = document.getElementById("loss-label");
const accuracyLabelElement = document.getElementById("accuracy-label");
const lossValues = [
    [],
    []
];

export function plotLoss(batch, loss, set) {
    const series = set === "train" ? 0 : 1;
    lossValues[series].push({ x: batch, y: loss });
    const lossContainer = document.getElementById("loss-canvas");
    tfvis.render.linechart(
        lossContainer, { values: lossValues, series: ["train", "validation"] }, {
            xLabel: "Batch #",
            yLabel: "Loss",
            width: 400,
            height: 300,
        }
    );
    lossLabelElement.innerText = `Last Loss: ${loss.toFixed(3)}`;
}

const accuracyValues = [
    [],
    []
];
export function plotAccuracy(batch, accuracy, set) {
    const accuracyContainer = document.getElementById("accuracy-canvas");
    const series = set === "train" ? 0 : 1;
    accuracyValues[series].push({ x: batch, y: accuracy });
    tfvis.render.linechart(
        accuracyContainer, { values: accuracyValues, series: ["train", "validation"] }, {
            xLabel: "Batch #",
            yLabel: "Accuracy",
            width: 400,
            height: 300,
        }
    );
    accuracyLabelElement.innerText = `Last Accuracy: ${(accuracy * 100).toFixed(
    1
  )}%`;
}

export function show(image, canvas, size) {
    const [width, height] = [size, size];
    canvas.width = 4.5 * width;
    canvas.height = 4.5 * height;
    const ctx = canvas.getContext("2d");
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = data[i] * 255;
        imageData.data[j + 1] = data[i] * 255;
        imageData.data[j + 2] = data[i] * 255;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    ctx.drawImage(canvas, 0, 0, 4 * canvas.width, 4 * canvas.height);
}

export function showFC(unit, canvas) {
    const [width, height] = [1, 1];
    canvas.width = 10 * width;
    canvas.height = 10 * height;
    const ctx = canvas.getContext("2d");
    const imageData = new ImageData(width, height);
    const data = unit.dataSync();
    imageData.data[0] = data[0] * 255;
    imageData.data[1] = data[0] * 255;
    imageData.data[2] = data[0] * 255;
    imageData.data[3] = 255;
    ctx.putImageData(imageData, 0, 0);
    ctx.drawImage(canvas, 0, 0, 4 * canvas.width, 4 * canvas.height);
}

export function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = data[i] * 255;
        imageData.data[j + 1] = data[i] * 255;
        imageData.data[j + 2] = data[i] * 255;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

export function getTrainEpochs() {
    return Number.parseInt(document.getElementById("train-epochs").value);
}

export function getLearningRate() {
    return Number.parseFloat(document.getElementById("learning-rate").value);
}

export function getBatchSize() {
    return Number.parseInt(document.getElementById("batch-size").value);
}

export function getOptimizer() {
    return document.getElementById("optimizer").value;
}

export function setTrainButtonCallback(callback) {
    const trainButton = document.getElementById("train");
    trainButton.addEventListener("click", () => {
        trainButton.setAttribute("disabled", true);
        callback();
    });
}