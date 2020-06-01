




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
  this.auto.compile({optimizer: 'adam', loss: 'meanSquaredError', lr: 0.1})
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
  let valAcc;
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





let data;
async function run(){
  data = new MnistData();
  await data.load();
}

let model;
async function load() {
  const n_units=document.getElementById('n_units').value
  const n_layers=document.getElementById('n_layers').value
  const hidden=document.getElementById('hidden').value
  model = new createConvModel(n_layers,n_units,hidden);
  const elem=document.getElementById('new')
  elem.innerHTML="Model Created!!!"
  epochs=0;
}
async function runtrain(){
  await train(model);
}


// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.
const element=document.getElementById('train')
element.addEventListener('click', runtrain);
const elemen=document.getElementById('Create')
elemen.addEventListener('click', load);
document.addEventListener('DOMContentLoaded', run);
