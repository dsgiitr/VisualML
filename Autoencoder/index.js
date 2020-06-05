




// This is a helper class for loading and managing MNIST data specifically.
// It is a useful example of how you could create your own data manager class
// for arbitrary data though. It's worth a look :)
import {IMAGE_H, IMAGE_W, MnistData} from './datas.js';

// This is a helper class for drawing loss graphs and MNIST images to the
// window. For the purposes of understanding the machine learning bits, you can
// largely ignore it
import * as ui from './ui.js';


function createConvModel(n_layers,n_units,hidden) {

  this.latent_dim =  Number(hidden);                                         //final dimension of hidden layer
  this.n_layers = Number(n_layers);                                           //how many hidden layers in encoder and decoder
  this.n_units =  Number(n_units);                                         //output dimension of each layer
  this.img_shape = [28,28];
  this.img_units = this.img_shape[0] * this.img_shape[1];
  // build the encoder
  var i = tf.input({shape: this.img_shape})
  var h = tf.layers.flatten().apply(i)

  for (var j=0; j<this.n_layers; j++) {
    var h = tf.layers.dense({units: this.n_units, activation:'relu'}).apply(h)              //n hidden
  }

  var o = tf.layers.dense({units: this.latent_dim}).apply(h)                                //1 final
  this.encoder = tf.model({inputs: i, outputs: o});

  // build the decoder
  var i = h = tf.input({shape: this.latent_dim})
  for (var j=0; j<this.n_layers; j++) {                                                     //n hidden
    var h = tf.layers.dense({units: this.n_units, activation:'relu'}).apply(h)
  }
  var o = tf.layers.dense({units: this.img_units}).apply(h)                                 //1 final
  var o = tf.layers.reshape({targetShape: this.img_shape}).apply(o)
  this.decoder = tf.model({inputs: i, outputs: o})

  // stack the autoencoder
  var i = tf.input({shape: this.img_shape})
  var z = this.encoder.apply(i)                                                             //z is hidden code

  var o = this.decoder.apply(z)
  this.auto = tf.model({inputs: i, outputs: o})

}
let epochs;

async function train(model) {

  const e=document.getElementById('batchsize')
  const batchSize = Number(e.value)


  // Leave out the last 15% of the training data for validation, to monitor
  // overfitting during training.
  const validationSplit = 0;

  // Get number of training epochs from the UI.
  const element=document.getElementById('train-epochs')
  const trainEpochs = Number(element.value)
  const lr = Number(document.getElementById('lr').value)

  const ele=document.getElementById('new')
  ele.innerHTML="Training..."
  epochs=Number(epochs)+Number(trainEpochs)

  // We'll keep a buffer of loss and accuracy values over time.
  let trainBatchCount = 0;

  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const totalNumBatches =
      Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
      trainEpochs;

  // During the long-running fit() call for model training, we include
  // callbacks, so that we can plot the loss and accuracy values in the page
  // as the training progresses.
  const y=trainData.xs.reshape([-1,28,28])

  model.auto.compile({optimizer: 'adam', loss: 'meanSquaredError', lr: lr})

  await model.auto.fit(y, y, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,

  });
await showPredictions(model,epochs);                                      //Trivial Samples of autoencoder

}

/**
 *
 */
async function showPredictions(model,epochs) {                              //Trivial Samples of autoencoder
  const testExamples = 10;
  const examples = data.getTestData(testExamples);


  tf.tidy(() => {
    const output = model.auto.predict(examples.xs.reshape([-1,28,28]));



    ui.showTestResults(examples.xs.reshape([-1,28,28]),output,epochs);
  });
}





var data,vis=Number(500);
async function run(){
  data = new MnistData();
  await data.load();
}

var model;
async function load() {
  const n_units=document.getElementById('n_units').value
  const n_layers=document.getElementById('n_layers').value
  const hidden=document.getElementById('hidden').value
  model = new createConvModel(n_layers,n_units,hidden);
  const elem=document.getElementById('new')
  elem.innerHTML="Model Created!!!"
  epochs=0;
  vis=Number(document.getElementById('vis').value);
}
async function runtrain(){
  await train(model);
  vis=Number(document.getElementById('vis').value);
}


// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.

load()                                                       //load model






var container=document.getElementById('cn')               //plot2d area
const canvas=document.getElementById('celeba-scene')


      // sample from the latent space at obj.x, obj.y
function sample(obj) {                                    //plotting
  obj.x = (obj.x - 0.5) * vis;
  obj.y = (obj.y - 0.5) * vis;
  // convert 10, 50 into a vector
  var y = tf.tensor2d([[obj.x, obj.y]]);
  // sample from region 10, 50 in latent space

  var prediction = model.decoder.predict(y).dataSync();

  for(var i=0;i<prediction.length;i++){prediction[i]+=50;prediction[i]/=100;}
  prediction=(tf.tensor(prediction)).toFloat();

  const inputMax = prediction.max();
const inputMin = prediction.min();
prediction= prediction.sub(inputMin).div(inputMax.sub(inputMin));                         //scaling
  prediction=prediction.reshape([28,28]);

  prediction=prediction.mul(255).toInt();


  // log the prediction to the browser console
  tf.browser.toPixels(prediction, canvas);
}



function plot2d(){
  load();
  const decision=Number(document.getElementById("hidden").value)
  if(decision===Number(2)){
    canvas.style.display="block";
  window.c2d = new Controls2D({ onDrag: sample, container: container });
  sample({x:0,y:0});}
  else {  var context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display="none";                                                  //clearing canvas for higher dimensions
    container.innerHTML="";}
}

plot2d();

const el=document.getElementById('Create')                                                //listeners
el.addEventListener('click', plot2d);
const element=document.getElementById('train')
element.addEventListener('click', runtrain);

document.addEventListener('DOMContentLoaded', run);
