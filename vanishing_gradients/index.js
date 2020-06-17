 

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
import Plotly from 'plotly.js-dist';

 import * as data from './data';





//acc: a variable to get the correct index in grads tensor
// xl: x coordinates for graph of loss vs epoch
// yl: y coordinates for ......................
//xtr_a: x coordinates for graph of train accuracy and epoch
//ytr_a: y .................................................
//xte_a: x coordinates for test accuracy vs epoch
//yte_a: y........................................

  var acc=0;
  var xl=Array();
  var yl=Array();

  var xtr_a=Array();
  var ytr_a=Array();

  var xte_a=Array();
  var yte_a=Array();

document.getElementById('show-nn-architecture')
      .addEventListener('click', async() => {
        xl=[];yl=[];
        xtr_a=[];ytr_a=[];
        xte_a=[];yte_a=[];
        
        console.clear();


        var a_f=document.getElementById("activations_f");
        var layers=Number(document.getElementById('num-layers').value);
        var neurons=Number(document.getElementById('num-neurons').value);
        var batch=Number(document.getElementById('batch').value);
        var l_r=Number(document.getElementById('lr').value);
        var iter=Number(document.getElementById('iter').value);

  var [xTrain, yTrain, xTest, yTest] = data.getIrisData(0.2);
  


  var tgrads;



var w=Array();
var b=Array();
if(layers>1)
{

  w[0]        =  tf.variable(tf.randomNormal([4, neurons],0,0.5));
  b[0]        =  tf.variable(tf.randomNormal([1, neurons],0,0.5));


  for(var i=1;i<layers-1;i++)
  {
    
     w[i] = tf.variable(tf.randomNormal([neurons, neurons],0,0.5));
     b[i] = tf.variable(tf.randomNormal([1,neurons],0,0.5));
    
  }


  w[layers-1] =  tf.variable(tf.randomNormal([neurons, 3],0,0.5));
  b[layers-1] =  tf.variable(tf.randomNormal([1, 3],0,0.5));
}
else
{
  w[0]        =  tf.variable(tf.randomNormal([4, 3],0,0.5));
  b[0]        =  tf.variable(tf.randomNormal([1, 3],0,0.5));

} 




   function model(w,b,x)
  {
     var output = (tf.dot(x,w[0])).add(b[0]);

       if(a_f=="leakyRelu")
       {
          for(var i=1;i<layers;i++)
         {
            output=(tf.dot(tf.leakyRelu(output),w[i])).add(b[i]);
         }
       }
       else
        if(a_f=="sigmoid")
        {
          for(var i=1;i<layers;i++)
         {
            output=(tf.dot(tf.sigmoid(output),w[i])).add(b[i]);
         }
        }
        else
          if(a_f=="relu")
          {
            for(var i=1;i<layers;i++)
           {
              output=(tf.dot(tf.relu(output),w[i])).add(b[i]);
           }
          }
          else
          {
            for(var i=1;i<layers;i++)
           {
              output=(tf.dot(tf.tanh(output),w[i])).add(b[i]);
           }
          }
          return output;
  }
for(var e=0;e<iter;e++)
{
  const random_train = tf.util.createShuffledIndices(120);
  const random_test = tf.util.createShuffledIndices(30);
  var total_loss=0;
  for(var m=0;m<(120/batch);m++)
  {
    var x=[];
    var y=[];
    
    if(m>(120/batch)-1)
    {
      for(var i=0;i<120-m*batch;i++)
      {
        x.push(xTrain.slice([random_train[m*batch+i],0],[1,-1]).arraySync());
        y.push(yTrain.slice([random_train[m*batch+i],0],[1,-1]).arraySync());
      } 
    }
    else
    {
      for(var i=0;i<batch;i++)
      {
        x.push(xTrain.slice([random_train[m*batch+i],0],[1,-1]).arraySync());
        y.push(yTrain.slice([random_train[m*batch+i],0],[1,-1]).arraySync());
      }
    }
    x=tf.tensor(x).squeeze();
    y=tf.tensor(y).squeeze();

    var f=() =>{
         
         
      
         var loss   = tf.losses.softmaxCrossEntropy(y,model(w,b,x));
         
         return loss;
       }
       var {value, grads} = tf.variableGrads(f);
      total_loss+=value.arraySync();
       
      
      


       for(var j=0;j<layers;j++)
       {
        w[j]=tf.variable(tf.sub(w[j],grads[2*j+acc].mul(l_r)));
        b[j]=tf.variable(tf.sub(b[j],grads[2*j+acc+1].mul(l_r)));
       }

      
       
       acc+=2*layers;

       tgrads=grads;
      
  }
  console.log("Loss for epoch("+e+"):"+total_loss/(Math.ceil(120/batch)));
  yl.push(total_loss/(Math.ceil(120/batch)));


    var train_o = model(w,b,xTrain);
    var probs   = train_o.softmax();

    var test_o = model(w,b,xTest);
    var probstest   = tf.softmax(test_o);

    var max_idx_ptr = tf.argMax(probs,1).arraySync();
    var max_idx_train = tf.argMax(yTrain,1).arraySync();

    var max_idx_pte = tf.argMax(probstest,1).arraySync();
    var max_idx_test = tf.argMax(yTest,1).arraySync();

    

    var correct_train=0;
    for(var i=0;i<120;i++)
    {
      if(max_idx_ptr[i]==max_idx_train[i])
      {
        correct_train+=1;
      }
    }
    ytr_a.push(correct_train/120);


    var correct_test=0;
    for(var i=0;i<30;i++)
    {
      if(max_idx_pte[i]==max_idx_test[i])
      {
        correct_test+=1;
      }
    }
    yte_a.push(correct_test/30);

}//end of epoch loop  
   acc-=2*layers;
   

//printing test loss
   var test_loss=tf.losses.softmaxCrossEntropy(yTest,model(w,b,xTest));
   //console.log(test_loss.arraySync());
   var tl=document.getElementById("testloss");
    var ctx3=tl.getContext("2d");
    ctx3.clearRect(0, 0, 850, 30);
    ctx3.beginPath();
    ctx3.strokeStyle="black";
    ctx3.font="20px Arial";
    ctx3.fillText("Also, Loss on validation set after ("+iter+") epochs of training is: "+test_loss.arraySync(),5,20);

  
  
  var abs_grads=Array();



  for(var i=0;i<layers;i++)
  {
    abs_grads[i] = tf.abs(tgrads[2*i+acc]).arraySync();
  }


  for(var i=0;i<layers;i++)
  {
    for(var j=0;j<abs_grads[i].length;j++)
    {
      for(var k=0;k<abs_grads[i][j].length;k++)
      {
        if(abs_grads[i][j][k]==0)
        {
          abs_grads[i][j][k]+=0.0001;
        }
        abs_grads[i][j][k]=Math.log10(abs_grads[i][j][k]);
      }
    }
  }


  
var max= -100000000;
var min= 100000000;
for(var i=0;i<layers;i++)
  {
    for(var j=0;j<abs_grads[i].length;j++)
    {
      for(var k=0;k<abs_grads[i][j].length;k++)
      {
        max=Math.max(abs_grads[i][j][k],max);
        min=Math.min(abs_grads[i][j][k],min);
      }
    }
  }



  for(var i=0;i<layers;i++)
  {
    for(var j=0;j<tgrads[2*i+acc].arraySync().length;j++)
    {
      for(var k=0;k<abs_grads[i][j].length;k++)
      {
        abs_grads[i][j][k]=((abs_grads[i][j][k]-min)/(max-min))+0.07;
      }
    }
  }





//r is radius of neuron,d_h is horizontal dist. b/w centres of neurons
// h is height of canvas and w its width
//b_n boundary of neurons



        
    
        var r=30;
        var d_h=120;
        var d_v=120;
        var h
        var w = d_h*(layers+3)+100;
        var b_n=1
        if(neurons<4)
        {
          h = d_v*5;
        }
        else
        {
          h = d_v*(neurons+1);
        }
        var canvas=document.getElementById("myCanvas");
        canvas.width=w;
        canvas.height=h;
        var mid;
        mid=h/2;
  

        
        var ctx=canvas.getContext("2d");

        var cds=Array();
        cds.push(Array(4));
        for(var m=1;m<layers;m++)
        {
          cds.push(Array(neurons));
        }
        cds.push(Array(3));


    async function circles()
    {

         for(var m=-1;m<=2;m++)
        {
          ctx.beginPath();
          ctx.arc(d_h,(mid-(d_v/2))+m*d_v,r,0,2*Math.PI);
          cds[0][m+1]=(mid-(d_v/2)+m*d_v);
          ctx.lineWidth=b_n;
          ctx.fillStyle="#FFFF99";
          ctx.fill();
          ctx.stroke();
        }

        
          var i=2;
          while(i<layers+2)
          {
            var len=abs_grads[i-2][0].length;
            var cnt=1;
            if(len%2==0)
            {
              var j= -(len/2);
              while(cnt<=len&&j<(len/2))
              {
                ctx.beginPath();
                ctx.arc(d_h*i,(mid+(d_v/2))+j*d_v,r,0,2*Math.PI);
                cds[i-1][cnt-1]=(mid+(d_v/2)+j*d_v);
                ctx.lineWidth=b_n;
                ctx.fillStyle="#FFFF99";
                ctx.fill();
                ctx.stroke();
                cnt++;
                j++;
              }
            }
            else
            {
              var j = -Math.floor(len/2);
              while(cnt<=len&&j<=(len/2))
              {
                ctx.beginPath();
                ctx.arc(d_h*i,(mid+j*d_v),r,0,2*Math.PI);
                cds[i-1][cnt-1]=(mid+j*d_v);
                ctx.lineWidth=b_n;
                ctx.fillStyle="#FFFF99";
                ctx.fill();
                ctx.stroke();
                cnt++;
                j++;
              }
            }
            i++;
          }
        
        
        ctx.beginPath();
        ctx.arc(d_h*(layers+2),mid,40,0,2*Math.PI);
        ctx.fillStyle="#FFFF99";
        ctx.fill();
        ctx.stroke();

        ctx.font = "17px Arial";
        ctx.strokeText("OUTPUT",d_h*(layers+2)-33,mid+9);


      }

      circles();

      

       var ctx2=canvas.getContext("2d");

       for(var m=0;m<layers;m++)
       {
        for(var n=0;n<cds[m].length;n++)
        {
          for(var p=0;p<cds[m+1].length;p++)
          { 
                ctx2.beginPath();
                ctx2.lineWidth=4;
                ctx2.moveTo(d_h*(m+1),cds[m][n]);
                ctx2.lineTo(d_h*(m+2),cds[m+1][p]);
                if(tgrads[2*m+acc].arraySync()[n][p]<0)
                {
                  ctx2.strokeStyle="rgba(255,0,0,"+abs_grads[m][n][p]+")";
                }
                else
                {
                  ctx2.strokeStyle="rgba(0,0,255,"+abs_grads[m][n][p]+")";
                }
                ctx2.stroke();
          }
        }
       }


       for(var m=0;m<3;m++)
       {
         ctx2.beginPath();
         ctx2.lineWidth=4;
         ctx2.moveTo(d_h*(layers+1),cds[layers][m]);
         ctx2.lineTo(d_h*(layers+2),mid);
         ctx2.strokeStyle="black";
         ctx2.stroke();
       }

       for(var m=0;m<4;m++)
       {
         ctx2.beginPath();
         ctx2.lineWidth=4;
         ctx2.moveTo(50,cds[0][m]);
         ctx2.lineTo(d_h,cds[0][m]);
         ctx2.strokeStyle="black";
         ctx2.stroke();
       }
       ctx2.beginPath();
       ctx2.fillStyle="black"
       ctx2.font = "17px Arial";
       ctx2.fillText("Petal length",1,cds[0][0]+17);
       ctx2.fillText("Petal width", 1,cds[0][1]+17);
       ctx2.fillText("Sepal length",1,cds[0][2]+17);
       ctx2.fillText("Sepal width",1,cds[0][3]+17);
       ctx2.beginPath();
       ctx2.fillStyle="black"
       ctx2.font = "17px Arial";
       ctx2.font = "25px Arial";
       ctx2.fillText(" Average Training Loss vs epoch",20,h-5);
       circles();
      
      acc+=4*layers;

for(var z=0;z<iter;z++)
{
  xl.push(z);
  xte_a.push(z);
  xtr_a.push(z);
}

    
    

  Plotly.newPlot('graph', [{
  x: xl,
  y: yl,
  line: {simplify: false},
}], {}, {showSendToCloud:true});

function plot() {
  Plotly.animate('graph', {
    data: [{y: yl}],
    traces: [0],
    layout: {}
  }, {
    transition: {
      duration: 500,
      easing: 'cubic-in-out'
    },
    frame: {
      duration: 500
    }
  })
}
plot();



Plotly.newPlot('graph2', [{
  x: xtr_a,
  y: ytr_a,
  line: {simplify: false},
}], {}, {showSendToCloud:true});

function plot2() {
  Plotly.animate('graph2', {
    data: [{y: ytr_a}],
    traces: [0],
    layout: {}
  }, {
    transition: {
      duration: 500,
      easing: 'cubic-in-out'
    },
    frame: {
      duration: 500
    }
  })
}
plot2();



Plotly.newPlot('graph3', [{
  x: xte_a,
  y: yte_a,
  line: {simplify: false},
}], {}, {showSendToCloud:true});

function plot3() {
  Plotly.animate('graph3', {
    data: [{y: yte_a}],
    traces: [0],
    layout: {}
  }, {
    transition: {
      duration: 500,
      easing: 'cubic-in-out'
    },
    frame: {
      duration: 500
    }
  })
}
plot3();


var tracanvas=document.getElementById("tra");
var tra =  tracanvas.getContext("2d");
tra.beginPath();
tra.font="25px Arial";
tra.fillText("Train accuracy(120 egs) vs epoch => ",5,20);

var teacanvas=document.getElementById("tea");
var tea =  teacanvas.getContext("2d");
tea.beginPath();
tea.font="25px Arial";
tea.fillText("Test accuracy(30 egs) vs epoch => ",5,20);

} );


