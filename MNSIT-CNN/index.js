import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { IMAGE_H, IMAGE_W, MnistData } from "./data.js";

// This is a helper class for drawing loss graphs and MNIST images to the
// window. For the purposes of understanding the machine learning bits, you can
// largely ignore it
import * as ui from "./ui.js";
/**
 * Creates a convolutional neural network (Convnet) for the MNIST data.
 *
 * @returns {tf.Model} An instance of tf.Model.
 */
function createConvModel() {
  // Create a sequential neural network model. tf.sequential provides an API
  // for creating "stacked" models where the output from one layer is used as
  // the input to the next layer.
  const model = tf.sequential();

  // The first layer of the convolutional neural network plays a dual role:
  // it is both the input layer of the neural network and a layer that performs
  // the first convolution operation on the input. It receives the 28x28 pixels
  // black and white images. This input layer uses 16 filters with a kernel size
  // of 5 pixels each. It uses a simple RELU activation function which pretty
  // much just looks like this: __/
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_H, IMAGE_W, 1],
      kernelSize: 3,
      filters: 8,
      activation: "relu",
    })
  );

  // After the first layer we include a MaxPooling layer. This acts as a sort of
  // downsampling using max values in a region instead of averaging.
  // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  // Our third layer is another convolution, this time with 32 filters.
  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 16, activation: "relu" })
  );

  // Max pooling again.
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  // Add another conv2d layer.
  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
  );

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({ units: 10, activation: "relu" }));

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
  // represent numbers, but it's the same idea if you had classes that
  // represented other entities like dogs and cats (two output classes: 0, 1).
  // We use the softmax function as the activation for the output layer as it
  // creates a probability distribution over our 10 classes so their output
  // values sum to 1.
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  return model;
}

/**
 * Creates a model consisting of only flatten, dense and dropout layers.
 *
 * The model create here has approximately the same number of parameters
 * (~31k) as the convnet created by `createConvModel()`, but is
 * expected to show a significantly worse accuracy after training, due to the
 * fact that it doesn't utilize the spatial information as the convnet does.
 *
 * This is for comparison with the convolutional network above.
 *
 * @returns {tf.Model} An instance of tf.Model.
 */

/**
 * This callback type is used by the `train` function for insertion into
 * the model.fit callback loop.
 *
 * @callback onIterationCallback
 * @param {string} eventType Selector for which type of event to fire on.
 * @param {number} batchOrEpochNumber The current epoch / batch number
 * @param {tf.Logs} logs Logs to append to
 */

/**
 * Compile and train the given model.
 *
 * @param {tf.Model} model The model to train.
 * @param {onIterationCallback} onIteration A callback to execute every 10
 *     batches & epoch end.
 */
async function train(model, onIteration) {
  ui.logStatus("Training model...");

  // Now that we've defined our model, we will define our optimizer. The
  // optimizer will be used to optimize our model's weight values during
  // training so that we can decrease our training loss and increase our
  // classification accuracy.

  // We are using rmsprop as our optimizer.
  // An optimizer is an iterative method for minimizing an loss function.
  // It tries to find the minimum of our loss function with respect to the
  // model's weight parameters.
  var optimizer = ui.getOptimizer();
  var eta = ui.getLearningRate();
  if (optimizer === "RMSprop") {
    optimizer = tf.train.rmsprop(eta);
  } else if (optimizer === "Adam") {
    optimizer = tf.train.adam(eta);
  } else {
    optimizer = tf.train.sgd(eta);
  }

  // We compile our model by specifying an optimizer, a loss function, and a
  // list of metrics that we will use for model evaluation. Here we're using a
  // categorical crossentropy loss, the standard choice for a multi-class
  // classification problem like MNIST digits.
  // The categorical crossentropy loss is differentiable and hence makes
  // model training possible. But it is not amenable to easy interpretation
  // by a human. This is why we include a "metric", namely accuracy, which is
  // simply a measure of how many of the examples are classified correctly.
  // This metric is not differentiable and hence cannot be used as the loss
  // function of the model.
  model.compile({
    optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Batch size is another important hyperparameter. It defines the number of
  // examples we group together, or batch, between updates to the model's
  // weights during training. A value that is too low will update weights using
  // too few examples and will not generalize well. Larger batch sizes require
  // more memory resources and aren't guaranteed to perform better.
  var batchSize = ui.getBatchSize();

  // Leave out the last 15% of the training data for validation, to monitor
  // overfitting during training.
  const validationSplit = 0.15;

  // Get number of training epochs from the UI.
  var trainEpochs = ui.getTrainEpochs();

  // We'll keep a buffer of loss and accuracy values over time.
  let trainBatchCount = 0;

  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const totalNumBatches =
    Math.ceil((trainData.xs.shape[0] * (1 - validationSplit)) / batchSize) *
    trainEpochs;

  // During the long-running fit() call for model training, we include
  // callbacks, so that we can plot the loss and accuracy values in the page
  // as the training progresses.
  let valAcc;
  await model.fit(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;
        ui.logStatus(
          `Training... (` +
            `${((trainBatchCount / totalNumBatches) * 100).toFixed(1)}%` +
            ` complete). To stop training, refresh or close page.`
        );
        ui.plotLoss(trainBatchCount, logs.loss, "train");
        ui.plotAccuracy(trainBatchCount, logs.acc, "train");
        if (onIteration && batch % 10 === 0) {
          onIteration("onBatchEnd", batch, logs);
        }
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        valAcc = logs.val_acc;
        ui.plotLoss(trainBatchCount, logs.val_loss, "validation");
        ui.plotAccuracy(trainBatchCount, logs.val_acc, "validation");
        if (onIteration) {
          onIteration("onEpochEnd", epoch, logs);
        }
        await tf.nextFrame();
      },
    },
  });

  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  ui.logStatus(
    `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
      `Final test accuracy: ${testAccPercent.toFixed(1)}%`
  );
  // await model.save("localstorage://CNN");
}

/**
 * Show predictions on a number of test examples.
 *
 * @param {tf.Model} model The model to be used for making the predictions.
 */
async function showPredictions(model) {
  const testExamples = 14;
  const examples = data.getTestData(testExamples);

  // Code wrapped in a tf.tidy() function callback will have their tensors freed
  // from GPU memory after execution without having to call dispose().
  // The tf.tidy callback runs synchronously.
  tf.tidy(() => {
    const output = model.predict(examples.xs);

    // tf.argMax() returns the indices of the maximum values in the tensor along
    // a specific axis. Categorical classification tasks like this one often
    // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
    // one element for each output class. All values in the vector are 0
    // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
    // output from model.predict() will be a probability distribution, so we use
    // argMax to get the index of the vector element that has the highest
    // probability. This is our prediction.
    // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
    // dataSync() synchronously downloads the tf.tensor values from the GPU so
    // that we can use them in our normal CPU JavaScript code
    // (for a non-blocking version of this function, use data()).
    const axis = 1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(examples, predictions, labels);
  });
}

function createModel() {
  let model;
  model = createConvModel();
  return model;
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

let weights;
ui.setTrainButtonCallback(async () => {
  ui.logStatus("Loading MNIST data...");
  await load();

  ui.logStatus("Creating model...");
  const model = createModel();
  model.summary();

  ui.logStatus("Starting model training...");
  await train(model, () => showPredictions(model));
  weights = model.getWeights();
});

ui.setVisualiseButton0Callback(async () => {
  await visualiseLayer0();
});

ui.setVisualiseButton1Callback(async () => {
  await visualiseLayer1();
});
ui.setVisualiseButton2Callback(async () => {
  await visualiseLayer2();
});
ui.setVisualiseButton3Callback(async () => {
  await visualiseLayer3();
});
ui.setVisualiseButton4Callback(async () => {
  await visualiseLayer4();
});
ui.setVisualiseButton5Callback(async () => {
  await visualiseLayer5();
});
ui.setVisualiseButton6Callback(async () => {
  await visualiseLayer6();
});
ui.setVisualiseButton7Callback(async () => {
  await visualiseLayer7();
});

console.log("Yes");
let model;
var canvasWidth = 280;
var canvasHeight = 280;
var canvasStrokeStyle = "white";
var canvasLineJoin = "round";
var canvasLineWidth = 10;
var canvasBackgroundColor = "black";
var canvasId = "canvas";
var clickX = new Array();
var clickY = new Array();
var clickD = new Array();
var drawing;
var canvasBox = document.getElementById("canvas_box");
var canvas = document.createElement("canvas");

async function initModel() {
  model = createConvModel();
  model.setWeights(weights);
}
canvas.setAttribute("width", canvasWidth);
canvas.setAttribute("height", canvasHeight);
canvas.setAttribute("id", canvasId);
canvas.style.backgroundColor = canvasBackgroundColor;
canvasBox.appendChild(canvas);
if (typeof G_vmlCanvasManager != "undefined") {
  canvas = G_vmlCanvasManager.initElement(canvas);
}

const ctx = canvas.getContext("2d");
//---------------------
// MOUSE DOWN function
//---------------------
$("#canvas").mousedown(function (e) {
  var rect = canvas.getBoundingClientRect();
  var mouseX = e.clientX - rect.left;
  var mouseY = e.clientY - rect.top;
  drawing = true;
  addUserGesture(mouseX, mouseY);
  drawOnCanvas();
});

//-----------------------
// TOUCH START function
//-----------------------
canvas.addEventListener(
  "touchstart",
  function (e) {
    if (e.target == canvas) {
      e.preventDefault();
    }

    var rect = canvas.getBoundingClientRect();
    var touch = e.touches[0];

    var mouseX = touch.clientX - rect.left;
    var mouseY = touch.clientY - rect.top;

    drawing = true;
    addUserGesture(mouseX, mouseY);
    drawOnCanvas();
  },
  false
);

$("#canvas").mousemove(function (e) {
  if (drawing) {
    var rect = canvas.getBoundingClientRect();
    var mouseX = e.clientX - rect.left;
    var mouseY = e.clientY - rect.top;
    addUserGesture(mouseX, mouseY, true);
    drawOnCanvas();
  }
});

canvas.addEventListener(
  "touchmove",
  function (e) {
    if (e.target == canvas) {
      e.preventDefault();
    }
    if (drawing) {
      var rect = canvas.getBoundingClientRect();
      var touch = e.touches[0];

      var mouseX = touch.clientX - rect.left;
      var mouseY = touch.clientY - rect.top;

      addUserGesture(mouseX, mouseY, true);
      drawOnCanvas();
    }
  },
  false
);

$("#canvas").mouseup(function (e) {
  drawing = false;
});

canvas.addEventListener(
  "touchend",
  function (e) {
    if (e.target == canvas) {
      e.preventDefault();
    }
    drawing = false;
  },
  false
);

//----------------------
// MOUSE LEAVE function
//----------------------
$("#canvas").mouseleave(function (e) {
  drawing = false;
});

canvas.addEventListener(
  "touchleave",
  function (e) {
    if (e.target == canvas) {
      e.preventDefault();
    }
    drawing = false;
  },
  false
);

function addUserGesture(x, y, dragging) {
  clickX.push(x);
  clickY.push(y);
  clickD.push(dragging);
}

//-------------------
// RE DRAW function
//-------------------
function drawOnCanvas() {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.strokeStyle = canvasStrokeStyle;
  ctx.lineJoin = canvasLineJoin;
  ctx.lineWidth = canvasLineWidth;

  for (var i = 0; i < clickX.length; i++) {
    ctx.beginPath();
    if (clickD[i] && i) {
      ctx.moveTo(clickX[i - 1], clickY[i - 1]);
    } else {
      ctx.moveTo(clickX[i] - 1, clickY[i]);
    }
    ctx.lineTo(clickX[i], clickY[i]);
    ctx.closePath();
    ctx.stroke();
  }
}

$("#clear-button").click(async function () {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  clickX = new Array();
  clickY = new Array();
  clickD = new Array();
  $(".prediction-text").empty();
  $("#result_box").addClass("d-none");
});

function preprocessCanvas(image) {
  // resize the input image to target size of (1, 28, 28)
  let tensor = tf.browser
    .fromPixels(image)
    .resizeNearestNeighbor([28, 28])
    .mean(2)
    .expandDims(2)
    .expandDims()
    .toFloat();
  return tensor.div(255.0);
}

$("#predict-button").click(async function () {
  await initModel();
  // get image data from canvas
  var imageData = canvas.toDataURL();

  // preprocess canvas
  let tensor = preprocessCanvas(canvas);

  // make predictions on the preprocessed image tensor
  let predictions = await model.predict(tensor).data();

  // get the model's prediction results
  let results = Array.from(predictions);
  const data = [
    { index: 0, value: 100 * results[0] },
    { index: 1, value: 100 * results[1] },
    { index: 2, value: 100 * results[2] },
    { index: 3, value: 100 * results[3] },
    { index: 4, value: 100 * results[4] },
    { index: 5, value: 100 * results[5] },
    { index: 6, value: 100 * results[6] },
    { index: 7, value: 100 * results[7] },
    { index: 8, value: 100 * results[8] },
    { index: 9, value: 100 * results[9] },
  ];

  // Render to visor
  const surface = { name: "Bar chart", tab: "Charts" };
  tfvis.render.barchart(surface, data);
  tfvis.visor();
  // display the predictions in chart
  // $("#result_box").removeClass("d-none");
  // displayChart(results);
  // displayLabel(results);
});
//------------------------------
// Chart to display predictions
//------------------------------
// var chart = "";
// var firstTime = 0;
// function loadChart(label, data, modelSelected) {
//   var ctx = document.getElementById("chart_box").getContext("2d");
//   chart = new Chart(ctx, {
//     // The type of chart we want to create
//     type: "bar",

//     // The data for our dataset
//     data: {
//       labels: label,
//       datasets: [
//         {
//           label: modelSelected + " prediction",
//           backgroundColor: "#f50057",
//           borderColor: "rgb(255, 99, 132)",
//           data: data,
//         },
//       ],
//     },

//     // Configuration options go here
//     options: {},
//   });
// }

// //----------------------------
// // display chart with updated
// // drawing from canvas
// //----------------------------
// function displayChart(data) {
//   var select_option = "CNN";

//   const label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
//   if (firstTime == 0) {
//     loadChart(label, data, select_option);
//     firstTime = 1;
//   } else {
//     chart.destroy();
//     loadChart(label, data, select_option);
//   }
//   document.getElementById("chart_box").style.display = "block";
// }

function displayLabel(data) {
  var max = data[0];
  var maxIndex = 0;

  for (var i = 1; i < data.length; i++) {
    if (data[i] > max) {
      maxIndex = i;
      max = data[i];
    }
  }
  return maxIndex;
}
//   $(".prediction-text").html(
//     "Predicting you draw <b>" +
//       maxIndex +
//       "</b> with <b>" +
//       Math.trunc(max * 100) +
//       "%</b> confidence"
//   );
// }

let output = [];
async function cla() {
  var imageData = canvas.toDataURL();
  let tensor = preprocessCanvas(canvas);
  const num = 1;
  const model = createModel();
  model.setWeights(weights);
  var layers = model.layers;
  output[0] = layers[0].apply(tensor);
  output[1] = layers[1].apply(output[0]);
  output[2] = layers[2].apply(output[1]);
  output[3] = layers[3].apply(output[2]);
  output[4] = layers[4].apply(output[3]);
  output[5] = layers[5].apply(output[4]);
  output[6] = layers[6].apply(output[5]);
  output[7] = layers[7].apply(output[6]);
}

async function visualiseLayer0() {
  await cla();
  ui.showLayer(output[0], document.getElementById("Layer0"));
}
async function visualiseLayer1() {
  await cla();
  ui.showLayer(output[1], document.getElementById("Layer1"));
}
async function visualiseLayer2() {
  await cla();
  ui.showLayer(output[2], document.getElementById("Layer2"));
}
async function visualiseLayer3() {
  await cla();
  ui.showLayer(output[3], document.getElementById("Layer3"));
}
async function visualiseLayer4() {
  await cla();
  ui.showLayer(output[4], document.getElementById("Layer4"));
}
async function visualiseLayer5() {
  await cla();
  ui.showDense(output[5], document.getElementById("Layer5"));
}
async function visualiseLayer6() {
  await cla();
  ui.showDense(output[6], document.getElementById("Layer6"));
}
async function visualiseLayer7() {
  await cla();
  ui.showDense(output[7], document.getElementById("Layer7"));
}
