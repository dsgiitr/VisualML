




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
  var i = tf.input({shape: this.img_shape});
  var h = tf.layers.flatten().apply(i);

  for (var j=0; j<this.n_layers; j++) {
    var h = tf.layers.dense({units: this.n_units, activation:'relu'}).apply(h);           //n hidden
  }

  var o = tf.layers.dense({units: this.latent_dim}).apply(h);                               //1 final
  this.encoder = tf.model({inputs: i, outputs: o});

  // build the decoder
  var i = h = tf.input({shape: this.latent_dim});
  for (var j=0; j<this.n_layers; j++) {                                                     //n hidden
    var h = tf.layers.dense({units: this.n_units, activation:'relu'}).apply(h);
  }
  var o = tf.layers.dense({units: this.img_units}).apply(h)   ;                              //1 final
  var o = tf.layers.reshape({targetShape: this.img_shape}).apply(o);
  this.decoder = tf.model({inputs: i, outputs: o});

  // stack the autoencoder
  var i = tf.input({shape: this.img_shape});
  var z = this.encoder.apply(i);                                                          //z is hidden code

  var o = this.decoder.apply(z);
  this.auto = tf.model({inputs: i, outputs: o});

}
let epochs=0,trainEpochs,batch;
var trainData;
var testData;
var b;var model;

async function train(model) {

  const e=document.getElementById('batchsize');
  batch = Number(e.value);
  const validationSplit = Number(0);

  // Get number of training epochs from the UI.
  const element=document.getElementById('train-epochs');
  trainEpochs = Number(element.value);
  const lr = Number(document.getElementById('lr').value);


  epochs=Number(epochs)+Number(trainEpochs);

  const y=trainData.xs.reshape([-1,28,28]);

  model.auto.compile({optimizer: 'adam', loss: 'meanSquaredError', lr: lr});
  const onBatchEnd=loadbar;

  await model.auto.fit(y, y, {
    batchSize:batch,
    validationSplit:validationSplit,
    epochs: trainEpochs,
    callbacks: [{
    onBatchEnd: loadbar,}],
  });
await showPredictions(model,epochs);                                      //Trivial Samples of autoencoder

}



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
  trainData = data.getTrainData();
  testData = data.getTestData();
}


async function load() {
  var ele=document.getElementById('barc');
  ele.style.display="none";
  const n_units=document.getElementById('n_units').value;
  const n_layers=document.getElementById('n_layers').value;
  const hidden=document.getElementById('hidden').value;
  model = new createConvModel(n_layers,n_units,hidden);
  const elem=document.getElementById('new')
  elem.innerHTML="Model Created!!!"
  epochs=0;
  vis=Number(document.getElementById('vis').value);
}

load();

async function runtrain(){
  var ele=document.getElementById('barc');
  ele.style.display="block";
  var elem=document.getElementById('new');
  elem.innerHTML="";
  b=0;
  await train(model);
  vis=Number(document.getElementById('vis').value);
}

function loadbar(){
  var element=document.getElementById("bar");
  element.style.width=Math.min(Math.ceil((b*100*batch)/(trainEpochs*55000)),100)+'%';
  element.innerHTML=Math.min(Math.ceil((b*100*batch)/(trainEpochs*55000)),100)+'%';
  b++;
  console.log(b);
}



function normaltensor(prediction){
    for(var i=0;i<prediction.length;i++){prediction[i]+=50;prediction[i]/=100;}
    prediction=(tf.tensor(prediction)).toFloat();

    const inputMax = prediction.max();
    const inputMin = prediction.min();
    prediction= prediction.sub(inputMin).div(inputMax.sub(inputMin));
    return prediction;}
function normal(prediction){
  const inputMax = prediction.max();
  const inputMin = prediction.min();
  prediction= prediction.sub(inputMin).div(inputMax.sub(inputMin));
  return prediction;
}


var container=document.getElementById('cn');               //plot2d area
const canvas=document.getElementById('celeba-scene');
const mot=document.getElementById('mot');
var cont=mot.getContext('2d');

function sample(obj) {                                    //plotting
  obj.x = (obj.x) * vis;
  obj.y = (obj.y) * vis;
  // convert 10, 50 into a vector
  var y = tf.tensor2d([[obj.x, obj.y]]);
  // sample from region 10, 50 in latent space

  var prediction = model.decoder.predict(y).dataSync();

                         //scaling
  prediction=normaltensor(prediction);
  prediction=prediction.reshape([28,28]);

  prediction=prediction.mul(255).toInt();


  // log the prediction to the browser console
  tf.browser.toPixels(prediction, canvas);
}

var mouse={x:0,y:0};
sample(mouse);
cont.fillStyle = "#DDDDDD";
cont.fillRect(0,0,mot.width,mot.height);
mot.addEventListener('mousemove', function(e) {
    mouse.x = (e.pageX - this.offsetLeft)*3.43;
    mouse.y = (e.pageY - this.offsetTop)*1.9;
}, false);

mot.addEventListener('mousedown', function(e) {
    mot.addEventListener('mousemove', on, false);
}, false);

mot.addEventListener('mouseup', function() {
    mot.removeEventListener('mousemove', on, false);
}, false);

var on= function() {
  cont.fillStyle = "#BBBBBB";
    cont.fillRect(0,0,mot.width,mot.height);
    cont.fillStyle="#000000";
    cont.fillRect(mouse.x-10,mouse.y-10, 40, 20);
    sample(mouse);
};







function plot2d(){
  load();
  const decision=Number(document.getElementById("hidden").value);
  if(decision===Number(2)){
    container.style.display="block";
  }
  else {  var context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    container.style.display="none";                                                  //clearing canvas for higher dimensions
    }
}


const el=document.getElementById('Create')                                                //listeners
el.addEventListener('click', plot2d);
const element=document.getElementById('train')
element.addEventListener('click', runtrain);

document.addEventListener('DOMContentLoaded', run);
document.addEventListener('DOMContentLoaded',plot2d);                                                  //load model







const canv=document.getElementById('canv');
const outcanv=document.getElementById('outcanv');
var ct = outcanv.getContext('2d');

var ctx = canv.getContext('2d');

function clear(){
    ctx.clearRect(0, 0, canv.width, canv.height);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canv.width, canv.height);
    ct.clearRect(0, 0, outcanv.width, outcanv.height);
    ct.fillStyle = "#DDDDDD";
    ct.fillRect(0, 0, outcanv.width, outcanv.height);
}
document.getElementById('clear').addEventListener('click',clear);
document.getElementById('save').addEventListener('click',rundraw);
clear();


  var mouse = {x: 0, y: 0};
  var last_mouse = {x: 0, y: 0};

  /* Mouse Capturing Work */
  canv.addEventListener('mousemove', function(e) {
      last_mouse.x = mouse.x;
      last_mouse.y = mouse.y;

      mouse.x = (e.pageX - this.offsetLeft)/1.34;
      mouse.y = (e.pageY - this.offsetTop)/2.7;
  }, false);


  /* Drawing on Paint App */
  ctx.lineWidth = 15;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'white';

  canv.addEventListener('mousedown', function(e) {
      canv.addEventListener('mousemove', onPaint, false);
  }, false);

  canv.addEventListener('mouseup', function() {
      canv.removeEventListener('mousemove', onPaint, false);
  }, false);

  var onPaint = function() {
      ctx.beginPath();
      ctx.moveTo(last_mouse.x, last_mouse.y);
      ctx.lineTo(mouse.x, mouse.y);
      ctx.closePath();
      ctx.stroke();
  };

function rundraw(){
  var sm=tf.browser.fromPixels(canv,1);
  sm=sm.toFloat().resizeNearestNeighbor([28,28]).reshape([-1,28,28]);
  sm=normal(sm);

  var pr=model.auto.predict(sm).dataSync();
  pr=normal(tf.tensor(pr).toFloat()).reshape([28,28]).mul(255.0).toInt();
  tf.browser.toPixels(pr, outcanv);
}
