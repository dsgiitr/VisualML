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
import * as loader from './loader';
import * as ui from './ui';

let model;


/**
 * Train a `tf.Model` to recognize Iris flower type.
 *
 * @param xTrain Training feature data, a `tf.Tensor` of shape
 *   [numTrainExamples, 4]. The second dimension include the features
 *   petal length, petalwidth, sepal length and sepal width.
 * @param yTrain One-hot training labels, a `tf.Tensor` of shape
 *   [numTrainExamples, 3].
 * @param xTest Test feature data, a `tf.Tensor` of shape [numTestExamples, 4].
 * @param yTest One-hot test labels, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 * @returns The trained `tf.Model` instance.
 */




async function trainModel(xTrain, yTrain, xTest, yTest) {
  ui.status('Training model... Please wait.');

  // Define the topology of the model
  const params = ui.loadTrainParametersFromUI();
  const model = tf.sequential();

  var neurons=params.numNeurons;
  var a_f = params.a_f;
  var layers = params.numLayers;
  model.add(tf.layers.dense(
      {units: neurons, activation: a_f, inputShape: [xTrain.shape[1]]}));

  var i=0;
  for(i=1;i<layers-1;i++)
  {
    model.add(tf.layers.dense({units: neurons, activation: a_f}));
  }

  model.add(tf.layers.dense({units: 3, activation: "softmax"}));
  model.summary();
  

  const optimizer = tf.train.adam(params.learningRate);
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
  const history = await model.fit(xTrain, yTrain, {
    epochs: params.epochs,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        const secPerEpoch =
            (performance.now() - beginMs) / (1000 * (epoch + 1));
        ui.status(`Training model... Approximately ${
            secPerEpoch.toFixed(4)} seconds per epoch`)
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ['loss', 'val_loss'])
        tfvis.show.history(accContainer, trainLogs, ['acc', 'val_acc']);
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
 * Run inference on some test Iris flower data.
 *
 * @param model The instance of `tf.Model` to run the inference with.
 * @param xTest Test data feature, a `tf.Tensor` of shape [numTestExamples, 4].
 * @param yTest Test true labels, one-hot encoded, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 */
async function evaluateModelOnTestData(model, xTest, yTest) {
  ui.clearEvaluateTable();

  tf.tidy(() => {
    const xData = xTest.dataSync();
    const yTrue = yTest.argMax(-1).dataSync();
    const predictOut = model.predict(xTest);
    const yPred = predictOut.argMax(-1);
    ui.renderEvaluateTable(
        xData, yTrue, yPred.dataSync(), predictOut.dataSync());
  });

  predictOnManualInput(model);
}

const HOSTED_MODEL_JSON_URL =
    'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json';

/**
 * The main function of the Iris demo.
 */
async function iris() {
  
  const [xTrain, yTrain, xTest, yTest] = data.getIrisData(0.15);

  const localLoadButton = document.getElementById('load-local');
  const localSaveButton = document.getElementById('save-local');
  const localRemoveButton = document.getElementById('remove-local'); 


  document.getElementById('train-from-scratch')
      .addEventListener('click', async () => {
        model = await trainModel(xTrain, yTrain, xTest, yTest);
        await evaluateModelOnTestData(model, xTest, yTest);
        localSaveButton.disabled = false;
        /*
       for (let i = 0; i < model.getWeights().length; i++) {
    console.log(model.getWeights()[i].dataSync());
}
*/
      });   

  document.getElementById('model-summary')
      .addEventListener('click', async() => {
        const surface = { name: 'Model Summary', tab: 'Model Inspection'};
        tfvis.show.modelSummary(surface, model);
      });        
    

  if (await loader.urlExists(HOSTED_MODEL_JSON_URL)) {
    ui.status('Model available: ' + HOSTED_MODEL_JSON_URL);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      ui.clearEvaluateTable();
      model = await loader.loadHostedPretrainedModel(HOSTED_MODEL_JSON_URL);
      await predictOnManualInput(model);
      localSaveButton.disabled = false;
    });
  }

  localLoadButton.addEventListener('click', async () => {
    model = await loader.loadModelLocally();
    await predictOnManualInput(model);
  });

  localSaveButton.addEventListener('click', async () => {
    await loader.saveModelLocally(model);
    await loader.updateLocalModelStatus();
  });

  localRemoveButton.addEventListener('click', async () => {
    await loader.removeModelLocally();
    await loader.updateLocalModelStatus();
  });

  await loader.updateLocalModelStatus();

  ui.status('Standing by.');
  ui.wireUpEvaluateTableCallbacks(() => predictOnManualInput(model));
}

iris();


document.getElementById('show-nn-architecture')
      .addEventListener('click', async() => {
        var n_neurons=Number(document.getElementById("num-neurons").value);
        var n_layers=Number(document.getElementById("num-layers").value);
        var h=110*(n_neurons+1);
        var w=250*n_layers;
        var canvas=document.getElementById("myCanvas");
        canvas.width=w;
        canvas.height=h;
        var mid;
        if(n_neurons%2==0)
        {
          mid=Math.floor(n_neurons/2);
        }
        else
        {
          mid=Math.floor(n_neurons/2)+1;
        }
        
        var ctx2=canvas.getContext("2d");
        ctx2.strokeStyle="#800000";

        for(var r=1;r<=4;r++)
        {
          for(var s=1;s<=n_neurons;s++)
          {
            ctx2.moveTo(110,110*r);
            ctx2.lineTo(110*2,110*s);
            ctx2.stroke();  
          }
        }
        for(var i=2;i<=n_layers-1;i++)
        {
           for(var j=1;j<=n_neurons;j++)
           {
              for(var k=1;k<=n_neurons;k++)
              {
                fnc(ctx2,i,j,k);
              }
           }
        }
        for(var m=1;m<=n_neurons;m++)
        {
          for(var n=-1;n<=1;n++)
          {
            ctx2.moveTo(110*n_layers,110*m);
            ctx2.lineTo(110*n_layers+110,110*(mid+n));
            ctx2.stroke();  
          }
        }
        for(var m=-1;m<=1;m++)
        {
            ctx2.moveTo(110*(n_layers+1),110*(mid+m));
            ctx2.lineTo(115*(n_layers+2),110*(mid));
            ctx2.stroke();  
        }
        function fnc(ci,i,j,k)
        {
          ci.moveTo(110*i,110*j);
          ci.lineTo(110*(i+1),110*k);
          ci.stroke();
        }
        var ctx=canvas.getContext("2d");
        for(var i=1;i<=n_neurons;i++)
        {
           for(var j=2;j<=n_layers;j++)
           {
              ctx.beginPath();
              ctx.arc(110*j,110*i,30,0,2*Math.PI);
              ctx.lineWidth=1;
              ctx.strokeStyle="#800000";
              ctx.fillStyle="#FFDAB9";
              ctx.fill();
              ctx.stroke();
           }
        }
        for(var m=-1;m<=1;m++)
        {
          ctx.beginPath();
          ctx.arc(110*(n_layers+1),110*(mid+m),30,0,2*Math.PI);
          ctx.fill();
          ctx.stroke();
        }
        for(var m=1;m<=4;m++)
        {
          ctx.beginPath();
          ctx.arc(110,110*m,30,0,2*Math.PI);
          ctx.fill();
          ctx.stroke();
        }
        
        ctx.beginPath();
        ctx.arc(115*(n_layers+2),110*(mid),50,0,2*Math.PI);
        ctx.fill();
        ctx.stroke();

        ctx.font = "20px Arial";
        ctx.strokeText("OUTPUT",115*(n_layers+2)-40,110*(mid))+15;
      }); 


