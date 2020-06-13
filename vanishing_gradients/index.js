 

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






  var acc=0;
document.getElementById('show-nn-architecture')
      .addEventListener('click', async() => {
        var a_f=document.getElementById("activations_f");
        var layers=Number(document.getElementById('num-layers').value);
        var neurons=Number(document.getElementById('num-neurons').value);
        var batch=Number(document.getElementById('batch').value);


  var [xTrain, yTrain, xTest, yTest] = data.getIrisData(0);
  var random=Math.floor(Math.random() * 146)
  var x  =  tf.slice(xTrain,[random,0],[batch,-1]);
  var y  =  tf.slice(yTrain,[random,0],[batch,-1]);



var w=Array()
var b=Array()
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
  var f=() =>{
       var output = (x.dot(w[0])).add(b[0]);

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
       
    
       var loss   = tf.losses.softmaxCrossEntropy(y,output);
       return loss;
}

   


  var {value, grads} = tf.variableGrads(f);
  
  var abs_grads=Array();



  for(var i=0;i<layers;i++)
  {
    abs_grads[i] = tf.abs(grads[2*i+acc]).arraySync();
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
    for(var j=0;j<grads[2*i+acc].arraySync().length;j++)
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
                if(grads[2*m+acc].arraySync()[n][p]<0)
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
       circles();
      
      acc+=2*layers;

} );      