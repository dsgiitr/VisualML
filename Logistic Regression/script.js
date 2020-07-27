
var x1,x2,y,N=1,initial_weight,weights_calculated , cost_history,m1,m2,n1,n2,degree;
var dv=document.getElementById('plt');

async function data(csvUrl) {
   const csvDataset = tf.data.csv(
     csvUrl,{hasHeader:1});

   var a=await csvDataset.toArray();
   var x1=new Array(),x2=new Array(),y=new Array();
   for(var i=0;i<a.length;i++){
     x1[i]=a[i].X1;
     x2[i]=a[i].X2;
     y[i]=a[i].Y;
   }
   return [x2,x1,y];
}

async function loaddata(){
  var t=document.getElementById('data').value;
  var dt;
  dt=await data(t+'.csv');

  x1=dt[0];
  x2=dt[1];
  y=dt[2];
  N=x1.length;
  x1=tf.tensor(x1);
  x2=tf.tensor(x2);
  y=tf.tensor(y);

  x1=x1.sub(x1.mean());
  x1=x1.div(x1.max().sub(x1.min())).mul(4.0);
  //x1=x1.add(1.0);
  x2=x2.sub(x2.mean());
  x2=x2.div(x2.max().sub(x2.min())).mul(4.0);
  //x2=x2.add(1.0);
  y=y.sub(y.min());
  y=y.div(y.max().sub(y.min()));

  n2=await x2.max().data();
  n1=await x2.min().data();
  m2=await x1.max().data();
  m1=await x1.min().data();
  x1=x1.dataSync();
  x2=x2.dataSync();
  y=y.dataSync();
}


document.getElementById('data').addEventListener('click',loaddata);

async function polynomial_features(x1,x2, degree){
  var res=new Array(x1.length);
    for(var k=0;k<x1.length;k++){
      res[k]=new Array((degree+1)*(degree+1));
      for(var i=0;i<=degree;i++){
        for(var j=0;j<=degree;j++){
          res[k][i*((degree+1))+j]=(Number(x1[k])**i)*(Number(x2[k])**j);}}}

    return tf.tensor2d(res);
}
function sigmoid(z){
    return tf.tensor(1).div(z.mul(-1).exp().add(1));}

async function update_weights(features, labels, weights, lr,r){
  const z = tf.dot(features,weights);
  var predictions = sigmoid(z);

  var gradient = (features.transpose()).dot((predictions.sub(labels)));
  gradient=gradient.div(N);
  gradient = gradient.mul(lr);
  gradient=gradient.add(weights.mul(2*r).div(weights.shape));
  weights=weights.sub(gradient);

  return weights;
}

async function cost_function(features, labels, weights,r){
    var z = tf.dot(features,weights);
    var h = sigmoid(z);
    var term1 = labels.mul(tf.log(h.add(0.0001)));
    var term2= (labels.mul(-1).add(1)).mul(tf.log(h.mul(-1).add(1).add(0.0001)));
    var J = term1.add(term2).div(-N);
    var sum=Number(await J.sum().data());
    var res=await weights.dataSync();
    for(var i=0;i<res.length;i++){
      sum+=(Number(r)*res[i]*res[i])/res.length;
    }
    return sum;
}
async function train(features, labels, weights, lr, iters=100,r=0){
    cost_history =new Array();
    var bar=document.getElementById('bar');
    for(var i=0;i<iters;i++){
        bar.style.width=Math.ceil(i*100/(iters-1))+'%';
        bar.innerHTML=Math.ceil(i*100/(iters-1))+'%';

        weights = await update_weights(features, labels, weights, lr,r);
        var cost = await cost_function(features, labels, weights,r);
        cost_history.push({x:i,y:Number(cost)});

        // Log Progress
         if (i % 10 == 5)
             console.log(cost);
      }
    return [weights, cost_history]
}

async function trainclick(){
  var ele=document.getElementById('barc');
  ele.style.display="block";
  await loaddata();
  degree=Number(document.getElementById('d').value);
  var lr=tf.pow(tf.tensor(10),Number(document.getElementById('l').value));
  var r=Number(document.getElementById('r').value);
  initial_weight = tf.zeros([((degree+1)*(degree+1))]).toFloat();
  weights_calculated=initial_weight;
  var epoch=Number(document.getElementById('epoch').value);

  var X=await polynomial_features(x1,x2,degree);
  y=tf.tensor(y).toFloat();
  var res=await train(X, y , initial_weight,lr,epoch,r);
  weights_calculated =res[0]; cost_history=res[1];
  await plotdecisionboundary();
  ele.style.display="none";
  var bar=document.getElementById('bar');
  bar.style.width='0%';
  bar.innerHTML='';
}
document.addEventListener('DOMContentLoaded',trainclick());

document.getElementById('train').addEventListener('click',trainclick);


async function plotdecisionboundary(){
  y=y.dataSync();
  var rs=Number(document.getElementById('res').value);
  var res=Number(rs);
  m1=Number(m1);
  m2=Number(m2);
  n1=Number(n1);
  n2=Number(n2);
  var xx=new Array();
  for(var i=m1;i<=m2;i+=(m2-m1)/res){
    var xa=await tf.linspace(n1,n2,res).dataSync();
    var xb=await tf.ones([res]).toFloat().mul(Number(i)).dataSync();
    var xc=await polynomial_features(xb,xa,degree);
    var z = tf.dot(xc,weights_calculated).dataSync();
    for(var j=Number(0);j<z.length-1;j++){
    if(Number(z[j])*Number(z[j+1])<Number(0)){
      xx.push({x:Number(xa[j]),y:Number(i)});
    }}
  }
  var a1=new Array(),a2=new Array();
  for(var i=0;i<x2.length;i++){
    if(y[i]!=1)
      a1.push({x:(x2[i]),y:(x1[i])});
    else a2.push({x:(x2[i]),y:(x1[i])});
  }
  console.log(xx);
  var series = ['decision boundary','0','1'];
  var data = { values: [xx,a1,a2], series };
  var surface=document.getElementById('pl');
  surface.style.display='block';
  surface.innerHTML='';
  tfvis.render.scatterplot(surface, data,{xLabel:'X1',yLabel:'X2',fontSize:15});

  series = ['loss'];
  data = { values: [cost_history], series };
  surface=document.getElementById('training');
  surface.style.display='block';
  surface.innerHTML='';
  tfvis.render.linechart(surface, data,{xLabel:'Epochs',yLabel:'Loss',fontSize:15});
}
