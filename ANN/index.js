/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import * as data from './data';

import * as ui from './ui';

let model;

const params = ui.loadTrainParametersFromUI();

/**
 * Train a `tf.Model` to recognize Iris flower type.
 *
 * @param trainDataset A tf.Dataset object yielding features and targets. The
 *   features must be of shape [numTrainExamples, 4], while the targets must be
 *   [numTrainExamples, 3]. The four feature dimensions include the
 *   petal_length, petal_width, sepal_length and sepal_width.  The target is
 *   one-hot encoded labels of the three iris categories.
 * @param validataionDataset A tf.Dataset of the same format as the trainDataset
 *   for use in validation.
 * @returns The trained `tf.Model` instance.
 */
async function trainModel(trainDataset, validationDataset,arr) {
  ui.status('Training model... Please wait.');

  

  // Define the topology of the model: two dense layers.
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: arr[0],
    activation: 'sigmoid',
    inputShape: [data.IRIS_NUM_FEATURES]
  }));
  var i=0;
  for(i=1;i<arr.length;i++){
    model.add(tf.layers.dense({units: arr[i], activation: 'sigmoid'}));
  }
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  model.summary();
  const surface = { name: 'Model Summary', tab: 'Model Inspection'};
  tfvis.show.modelSummary(surface, model);

  var optimizer = params.optimizer;

  if (optimizer === "RMSprop") {
    optimizer = tf.train.rmsprop(params.learningRate);
  } else if (optimizer === "Adam") {
    optimizer = tf.train.adam(params.learningRate);
  } else {
    optimizer = tf.train.sgd(params.learningRate);
  }
  
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const trainLogs = [];
  const lossContainer = document.getElementById('lossCanvas');
  const accContainer = document.getElementById('accuracyCanvas');
  const beginMs = performance.now();
  // Call `model.fit` to train the model.
  await model.fitDataset(trainDataset, {
    epochs: params.epochs,
    validationData: validationDataset,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        const secPerEpoch =
            (performance.now() - beginMs) / (1000 * (epoch + 1));
        ui.status(
            `Training model... Approximately ` +
            `${secPerEpoch.toFixed(4)} seconds per epoch`);
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ['loss', 'val_loss'])
        tfvis.show.history(accContainer, trainLogs, ['acc', 'val_acc'])
        const [{xs: xTest, ys: yTest}] = await validationDataset.toArray();
        calculateAndDrawConfusionMatrix(model, xTest, yTest);
      },
    }
  });

  const secPerEpoch = (performance.now() - beginMs) / (1000 * params.epochs);
  ui.status(
      `Model training complete:  ${secPerEpoch.toFixed(4)} seconds per epoch`);
  
  return model;
}

/**
 * Run inference on manually-input Iris flower data.
 *
 * @param model The instance of `tf.Model` to run the inference with.
 */
async function predictOnManualInput(model) {
  if (model == null) {
    ui.setManualInputWinnerMessage('ERROR: Please load or train model first.');
    return;
  }

  // Use a `tf.tidy` scope to make sure that WebGL memory allocated for the
  // `predict` call is released at the end.
  tf.tidy(() => {
    // Prepare input data as a 2D `tf.Tensor`.
    const inputData = ui.getManualInputData();
    const input = tf.tensor2d([inputData], [1, 4]);

    // Call `model.predict` to get the prediction output as probabilities for
    // the Iris flower categories.

    const predictOut = model.predict(input);
    const logits = Array.from(predictOut.dataSync());
    const winner = data.IRIS_CLASSES[predictOut.argMax(-1).dataSync()[0]];
    ui.setManualInputWinnerMessage(winner);
    ui.renderLogitsForManualInput(logits);
  });
}

/**
 * Draw confusion matrix.
 */
async function calculateAndDrawConfusionMatrix(model, xTest, yTest) {
  const [preds, labels] = tf.tidy(() => {
    const preds = model.predict(xTest).argMax(-1);
    const labels = yTest.argMax(-1);
    return [preds, labels];
  });

  const confMatrixData = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = document.getElementById('confusion-matrix');
  tfvis.render.confusionMatrix(
      container, {values: confMatrixData, labels: data.IRIS_CLASSES},
      {shadeDiagonal: true},
  );

  tf.dispose([preds, labels]);
}

/**
 * Run inference on some test Iris flower data.
 *
 * @param model The instance of `tf.Model` to run the inference with.
 * @param testDataset A tf.Dataset object yielding features and targets. The
 *   features must be of shape [numTrainExamples, 4], while the targets must be
 *   [numTrainExamples, 3]. The four feature dimensions include the
 *   petal_length, petal_width, sepal_length and sepal_width.  The target is
 *   one-hot encoded labels of the three iris categories.
 */
async function evaluateModelOnTestData(model, testDataset) {
  ui.clearEvaluateTable();
  const [{xs: xTest, ys: yTest}] = await testDataset.toArray();
  const xData = xTest.dataSync();
  const yTrue = yTest.argMax(-1).dataSync();
  const predictOut = model.predict(xTest);
  const yPred = predictOut.argMax(-1);
  ui.renderEvaluateTable(xData, yTrue, yPred.dataSync(), predictOut.dataSync());
  calculateAndDrawConfusionMatrix(model, xTest, yTest);
  predictOnManualInput(model);

  
}

const HOSTED_MODEL_JSON_URL =
    'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json';
/**
 * The main function of the Iris demo.
 */
var arr=Array();
var maxi=0;
async function iris() {
  const testFraction = 0.15;
  let [trainDataset, testDataset] = await data.getIrisData(testFraction);
  // Batch datasets.
  trainDataset = trainDataset.batch(params.batch_size);
  testDataset = testDataset.batch(params.batch_size);
  var x=0;
  document.getElementById('button1').addEventListener('click',async()=>{
            arr[x] = Number(document.getElementById('#ofneuron').value);
            if(arr[x]>maxi){
              maxi=arr[x];
            }
            document.getElementById('#ofneuron').value="0";
            
            
           x++;
           document.getElementById('pos').value=x+1;
  });
  document.getElementById('train-from-scratch')
      .addEventListener('click', async () => {
        model = await trainModel(trainDataset, testDataset,arr);
       
        await evaluateModelOnTestData(model, testDataset);
        
      });
      

  ui.status('Standing by.');

  document.getElementById('structure').addEventListener('click',async()=>{
    
    var graph = document.getElementById("graph");
    graph.width=1000;
    graph.height=75*maxi;
    const check=document.getElementById("myCheck").checked;
    
    
    var canvasOffset = $("#graph").offset();
    var offsetX = canvasOffset.left;
    var offsetY = canvasOffset.top;
    

    var tipCanvas = document.getElementById("tip");
    
    var tipCtx = tipCanvas.getContext("2d");
    
    var nn=Array();
    nn[0]=data.IRIS_NUM_FEATURES;
    var i;
    for(i=0;i<arr.length;i++){
      nn[i+1]=arr[i];
    }
    nn[i+1]=3;
    
    var l=Number(document.getElementById('pos').value)+1;
    var ctx = graph.getContext("2d");
    var graph;
    var arr1=Array();
    var count1,count2;
    for(i=0;i<l;i++){
     arr1[i]=((75*maxi)/2)-60*(nn[i]/2);
    }
    var j,k;

    var dots = [];

  
    for(i=0;i<l;i++){
      if(i<l-1){
       var a= model.layers[i].getWeights()[0].arraySync();
       
      }
      count1=arr1[i];
       for(j=0;j<nn[i];j++){
           count2=arr1[i+1];
           if(i+1<l){
               for(k=0;k<nn[i+1];k++){
                  ctx.beginPath(); 
                  ctx.moveTo(150*(i+1),count1);
                  ctx.lineTo(150*(i+2),count2);
                  if(check==true){
                    if(Number(a[j][k])>0){
                      var b=255-100*a[j][k];
                     ctx.lineWidth=5*a[j][k]; 
                    ctx.strokeStyle='rgb(0,0,'+b+')';
                    }
                    else{
                      var c=255+100*a[j][k];
                      ctx.lineWidth=-5*a[j][k];
                    ctx.strokeStyle='rgb('+c+',0,0)';
                    }
                  }
                  else{
                  ctx.strokeStyle='black';
                  }
                  ctx.stroke();
                  count2+=60;
                }
               }

            ctx.beginPath();
            ctx.lineWidth=1;
            ctx.arc(150*(i+1),count1, 10,0,2*Math.PI);
            ctx.fillStyle='rgb(200,255,0)';
            ctx.fill();
            ctx.strokeStyle='black';
            ctx.stroke();
            if(i==0){
              dots.push({
                x: 150*(i+1),
                y: count1,
                r: 10,
                rXr: 100,
                tip1:"layer no.: "+(i+1),
                tip2:"input no.: "+(j+1),
                tip3:""
                
                
                });
            }

            else if(i+1<l){
            dots.push({
              x: 150*(i+1),
              y: count1,
              r: 10,
              rXr: 100,
              tip1:"layer no.: "+(i+1),
              tip2:"neuron no.: "+(j+1),
              tip3:"activation:  'SIGMOID' " 
              
              
              });
            }
            else{
              dots.push({
                x: 150*(i+1),
                y: count1,
                r: 10,
                rXr: 100,
                tip1:"layer no.: "+(i+1),
                tip2:"neuron no.: "+(j+1),
                tip3:"activation:  'SOFTMAX' " 
                
                
            });
            }

            count1+=60;
        }
    }
    dots.push({
      x: 150*(i+1),
      y:((75*maxi)/2)-30,
      r:30,
      rXr:900,
      tip1:"",
      tip2:"predicted class",
      tip3:""

    });
    // request mousemove events
    
    // show tooltip when mouse hovers over dot
    graph.addEventListener("mousemove",function(e){
     var mouseX=parseInt(e.pageX-offsetX);
     var mouseY=parseInt(e.pageY-offsetY);
      // Put your mousemove stuff here
      var hit = false;
      for (var m = 0; m < dots.length; m++) {
          var dot = dots[m];
          var dx = mouseX - dot.x;
          var dy = mouseY - dot.y;
          if (dx * dx + dy * dy < dot.rXr) {
              tipCanvas.style.left = (e.pageX + 10) + "px";
              tipCanvas.style.top = (e.pageY + 10) + "px";
              tipCtx.clearRect(0, 0, tipCanvas.width, tipCanvas.height);
              tipCtx.font = '15px Verdana';
              
              if(m<=(dots.length-1)){
              tipCtx.fillText(dot.tip1, 10, 20);
              tipCtx.fillText(dot.tip2,10,40);
              tipCtx.fillText(dot.tip3,10,60);
              }
              
              hit = true;
              break;
          }
      }
      if (!hit) { tipCanvas.style.left = "-200px"; }
    });


    count1=arr1[l-1];
    for(k=0;k<3;k++){
      
      ctx.beginPath(); 
      ctx.moveTo(150*(i),count1);
      ctx.lineTo(150*(i+1),(((75*maxi)/2)-30));
      ctx.strokeStyle='black';
      ctx.stroke();
      count1+=60;
    }
    ctx.beginPath();
    ctx.arc(150*(i+1),((75*maxi)/2)-30, 30,0,2*Math.PI);
    ctx.fillStyle='rgb(200,255,0)';
    ctx.fill();
    ctx.strokeStyle='black';
    ctx.stroke();
    ctx.font = "15px Arial";
    ctx.fillStyle = "black";
    ctx.textAlign = "center";
    ctx.fillText("output", 150*(i+1),(((75*maxi)/2)-30));
    
    function canvas_arrow(context, fromx, fromy, tox, toy) {
      var headlen = 10; // length of head in pixels
      var dX = tox - fromx;
      var dY = toy - fromy;
      var angle = Math.atan2(dY, dX);
      context.moveTo(fromx, fromy);
      context.lineTo(tox, toy);
      context.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
      context.moveTo(tox, toy);
      context.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
      context.strokeStyle="black";
    }
    count1=arr1[0];
    for(i=0;i<4;i++){
      ctx.beginPath();
      canvas_arrow(ctx,30, count1, 140, count1);
      ctx.stroke();
      count1+=60;
    }
    if(check){
    ctx.font = "15px Arial";
    ctx.fillStyle = "black";
    ctx.textAlign = "center";
    ctx.fillText("Positive Weights", 900,20);
    ctx.font = "15px Arial";
    ctx.fillStyle = "black";
    ctx.textAlign = "center";
    ctx.fillText("Negative Weights", 900,40);
    ctx.beginPath(); 
    ctx.moveTo(800,20);
    ctx.lineTo(830,20);
    ctx.strokeStyle='blue';
    ctx.stroke();
    ctx.beginPath(); 
    ctx.moveTo(800,40);
    ctx.lineTo(830,40);
    ctx.strokeStyle='red';
    ctx.stroke();
    }
  });


 

  ui.wireUpEvaluateTableCallbacks(() => predictOnManualInput(model));
}

iris();


