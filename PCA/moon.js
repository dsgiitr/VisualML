import * as tf from '@tensorflow/tfjs';
import * as numeric from 'numeric';
import * as Plotly from 'plotly.js/dist/plotly.js';

const D2 = 2;
const N2= 50;


function render2DPrediction2(ctx, xs) {
    const xsOriginal = xs.dataSync();
    const c1Data = [];
    const c2Data = [];
    for (let i = 0; i < xsOriginal.length; i += 2) 
    {
        if (i < 2*N2) 
        {
            c1Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1] });
        }
        else 
        {
            c2Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1] });
        }
    }
    const c1X = [];
    const c1Y = [];
    for (const data1 of c1Data) {
        c1X.push(data1.x);
        c1Y.push(data1.y);
    }
    var trace1 = {
        x: c1X, y: c1Y,
        mode: 'markers',
        marker: {
            size: 8,
            color: 'rgba(255, 0, 0, 0.8)',
            line: {
                width: 0.5
            },
            opacity: 1.0
        },
        text: "Class 1",
        type: 'scatter'
    };
    const c2X = [];
    const c2Y = [];
    for (const data2 of c2Data) {
        c2X.push(data2.x);
        c2Y.push(data2.y);
    }
    var trace2 = {
        x: c2X, y: c2Y,
        mode: 'markers',
        marker: {
            size: 8,
            color: 'rgba(128, 0, 256, 0.8)',
            line: {
                width: 0.5
            },
            opacity: 1.0
        },
        text: "Class 2",
        type: 'scatter'
    };


    var data = [trace1, trace2];
    var layout = {
        width: 500,
        height: 500,
        margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 0
        },
        scene: {
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' }
        },
        title: "Data"
    };
    Plotly.newPlot(ctx, data, layout);
    return ctx;
}


function render1DPrediction2(ctx, xs) {
    const xsOriginal = xs.dataSync();
    const c1Data = [];
    const c2Data = [];

    for (let i = 0; i < xsOriginal.length; i += 1) 
    {
        if (i < N1) 
        {
            c1Data.push({ x: xsOriginal[i] , y: 0});
        }
        else 
        {
            c2Data.push({ x: xsOriginal[i] , y: 0});
        }
    }
    const c1X = [];
    const c1Y = [];
    for (const data1 of c1Data) {
        c1X.push(data1.x);
        c1Y.push(data1.y);
    }
    var trace1 = {
        x: c1X, y:c1Y,
        mode: 'markers',
        marker: {
            size: 8,
            color: 'rgba(255, 0, 0, 0.8)',
            line: {
                width: 0.5
            },
            opacity: 1.0
        },
        text: "Class 1",
        type: 'scatter'
    };
    const c2X = [];
    const c2Y = [];
    for (const data2 of c2Data) {
        c2X.push(data2.x);
        c2Y.push(data2.y);
    }
    var trace2 = {
        x: c2X, y: c2Y,
        mode: 'markers',
        marker: {
            size: 8,
            color: 'rgba(128, 0, 256, 0.8)',
            line: {
                width: 0.5
            },
            opacity: 1.0
        },
        text: "Class 2",
        type: 'scatter'
    };
    

    var data = [trace1, trace2];
    var layout = {
        width: 500,
        height: 500,
        scene: {
            xaxis: { title: 'X' },
        },
        title: "Data"
    };
    Plotly.newPlot(ctx, data, layout);
    return ctx;
}


async function pca3(xs, nComponents) {
    const batch = xs.shape[0];
    const meanValues = xs.mean(0);
    const sub = tf.sub(xs, meanValues);
    const covariance = tf.matMul(sub.transpose(), sub);//(3,3)
    //Numeric does not recognize tensor type of Tensorflow.js,Hence we need to convert
    //the tensor into javascript array
    const covarianceData = tf.util.toNestedArray([D2, D2], covariance.dataSync());
    const eig = numeric.eig(covarianceData);
    console.log("eigen values");
    console.log(eig.lambda);
    //returns eigen vectors and eigen values
    console.log("eigenvectors");
    console.log(eig.E);
    console.log(eig.E.x);
    const components = tf.tensor(eig.E.x);
    const eigenvectors = tf.tensor(eig.E.x).slice([0, 0], [-1, nComponents]);
    //eig.E returns eigen vectors in form of numeric-tensor
    //eig.E.x returns eigen vectors in raw array form
    //which is then converted into tf tensor to apply matrix multiplication
    return [components, tf.matMul(sub, eigenvectors)];
}




async function kpca(xs, gamma=15, nComponents=2) 
{
    const km = Array();
    //Calculating the squared Euclidean distances for every pair of points
    //in the 100*2 dimensional dataset.
    for(let i=0; i<100; i++)
    {
        for(let j=0; j<100; j++)
        {
            km.push(tf.norm(tf.sub(xs.gather([j],0),xs.gather([i],0))).arraySync());
        }
    }
    const sq_dists=tf.square(km);
    //console.log(sq_dists.arraySync());

    //Computing the 100*100 kernel matrix.
    var K = tf.exp(tf.mul(tf.tensor1d([(-gamma)]),sq_dists));
    //console.log(K.arraySync());
    var K_sym= K.reshape([100,100]);
    //console.log(K_sym.arraySync());


    //Centering the symmetric 100*100 kernel matrix.
    one_n = tf.div(tf.ones([100,100]),tf.tensor1d([100]));
    K_sym = tf.add( tf.sub( tf.sub(K_sym, tf.matMul(one_n, K_sym))  ,  tf.matMul(K_sym, one_n) ) , tf.matMul(tf.matMul(one_n,K_sym),one_n)); 

    var K_sym_data = tf.util.toNestedArray([100, 100], K_sym.dataSync());

    const eig = numeric.eig(K_sym_data);
    

    
    var temp = Array();
    for(var i=0;i<nComponents;i++)
    {
        temp.push(i);
    }
    const eigenvectors = tf.tensor(eig.E.x).gather(temp,1);
    //eig.E returns eigen vectors in form of numeric-tensor
    //eig.E.x returns eigen vectors in raw array form
    //which is then converted into tf tensor to apply matrix multiplication
    //console.log(eigenvectors.arraySync());
    return  eigenvectors;
}



const moon_data  =    [[ 1.00000000e+00,  0.00000000e+00],
                       [ 9.97945393e-01,  6.40702200e-02],
                       [ 9.91790014e-01,  1.27877162e-01],
                       [ 9.81559157e-01,  1.91158629e-01],
                       [ 9.67294863e-01,  2.53654584e-01],
                       [ 9.49055747e-01,  3.15108218e-01],
                       [ 9.26916757e-01,  3.75267005e-01],
                       [ 9.00968868e-01,  4.33883739e-01],
                       [ 8.71318704e-01,  4.90717552e-01],
                       [ 8.38088105e-01,  5.45534901e-01],
                       [ 8.01413622e-01,  5.98110530e-01],
                       [ 7.61445958e-01,  6.48228395e-01],
                       [ 7.18349350e-01,  6.95682551e-01],
                       [ 6.72300890e-01,  7.40277997e-01],
                       [ 6.23489802e-01,  7.81831482e-01],
                       [ 5.72116660e-01,  8.20172255e-01],
                       [ 5.18392568e-01,  8.55142763e-01],
                       [ 4.62538290e-01,  8.86599306e-01],
                       [ 4.04783343e-01,  9.14412623e-01],
                       [ 3.45365054e-01,  9.38468422e-01],
                       [ 2.84527587e-01,  9.58667853e-01],
                       [ 2.22520934e-01,  9.74927912e-01],
                       [ 1.59599895e-01,  9.87181783e-01],
                       [ 9.60230259e-02,  9.95379113e-01],
                       [ 3.20515776e-02,  9.99486216e-01],
                       [-3.20515776e-02,  9.99486216e-01],
                       [-9.60230259e-02,  9.95379113e-01],
                       [-1.59599895e-01,  9.87181783e-01],
                       [-2.22520934e-01,  9.74927912e-01],
                       [-2.84527587e-01,  9.58667853e-01],
                       [-3.45365054e-01,  9.38468422e-01],
                       [-4.04783343e-01,  9.14412623e-01],
                       [-4.62538290e-01,  8.86599306e-01],
                       [-5.18392568e-01,  8.55142763e-01],
                       [-5.72116660e-01,  8.20172255e-01],
                       [-6.23489802e-01,  7.81831482e-01],
                       [-6.72300890e-01,  7.40277997e-01],
                       [-7.18349350e-01,  6.95682551e-01],
                       [-7.61445958e-01,  6.48228395e-01],
                       [-8.01413622e-01,  5.98110530e-01],
                       [-8.38088105e-01,  5.45534901e-01],
                       [-8.71318704e-01,  4.90717552e-01],
                       [-9.00968868e-01,  4.33883739e-01],
                       [-9.26916757e-01,  3.75267005e-01],
                       [-9.49055747e-01,  3.15108218e-01],
                       [-9.67294863e-01,  2.53654584e-01],
                       [-9.81559157e-01,  1.91158629e-01],
                       [-9.91790014e-01,  1.27877162e-01],
                       [-9.97945393e-01,  6.40702200e-02],
                       [-1.00000000e+00,  1.22464680e-16],
                       [ 0.00000000e+00,  5.00000000e-01],
                       [ 2.05460725e-03,  4.35929780e-01],
                       [ 8.20998618e-03,  3.72122838e-01],
                       [ 1.84408430e-02,  3.08841371e-01],
                       [ 3.27051370e-02,  2.46345416e-01],
                       [ 5.09442530e-02,  1.84891782e-01],
                       [ 7.30832427e-02,  1.24732995e-01],
                       [ 9.90311321e-02,  6.61162609e-02],
                       [ 1.28681296e-01,  9.28244800e-03],
                       [ 1.61911895e-01, -4.55349012e-02],
                       [ 1.98586378e-01, -9.81105305e-02],
                       [ 2.38554042e-01, -1.48228395e-01],
                       [ 2.81650650e-01, -1.95682551e-01],
                       [ 3.27699110e-01, -2.40277997e-01],
                       [ 3.76510198e-01, -2.81831482e-01],
                       [ 4.27883340e-01, -3.20172255e-01],
                       [ 4.81607432e-01, -3.55142763e-01],
                       [ 5.37461710e-01, -3.86599306e-01],
                       [ 5.95216657e-01, -4.14412623e-01],
                       [ 6.54634946e-01, -4.38468422e-01],
                       [ 7.15472413e-01, -4.58667853e-01],
                       [ 7.77479066e-01, -4.74927912e-01],
                       [ 8.40400105e-01, -4.87181783e-01],
                       [ 9.03976974e-01, -4.95379113e-01],
                       [ 9.67948422e-01, -4.99486216e-01],
                       [ 1.03205158e+00, -4.99486216e-01],
                       [ 1.09602303e+00, -4.95379113e-01],
                       [ 1.15959990e+00, -4.87181783e-01],
                       [ 1.22252093e+00, -4.74927912e-01],
                       [ 1.28452759e+00, -4.58667853e-01],
                       [ 1.34536505e+00, -4.38468422e-01],
                       [ 1.40478334e+00, -4.14412623e-01],
                       [ 1.46253829e+00, -3.86599306e-01],
                       [ 1.51839257e+00, -3.55142763e-01],
                       [ 1.57211666e+00, -3.20172255e-01],
                       [ 1.62348980e+00, -2.81831482e-01],
                       [ 1.67230089e+00, -2.40277997e-01],
                       [ 1.71834935e+00, -1.95682551e-01],
                       [ 1.76144596e+00, -1.48228395e-01],
                       [ 1.80141362e+00, -9.81105305e-02],
                       [ 1.83808810e+00, -4.55349012e-02],
                       [ 1.87131870e+00,  9.28244800e-03],
                       [ 1.90096887e+00,  6.61162609e-02],
                       [ 1.92691676e+00,  1.24732995e-01],
                       [ 1.94905575e+00,  1.84891782e-01],
                       [ 1.96729486e+00,  2.46345416e-01],
                       [ 1.98155916e+00,  3.08841371e-01],
                       [ 1.99179001e+00,  3.72122838e-01],
                       [ 1.99794539e+00,  4.35929780e-01],
                       [ 2.00000000e+00,  5.00000000e-01]];


const data  = tf.tensor2d(moon_data, [100, 2]);       
          


async function main4(xs) 
{
    
    const [axes, pcaXs] = await pca3(xs, 2);
    xs_new = axes.concat(xs);
    
    const ctx3 = document.getElementById('2dplot-m');
    render2DPrediction2(ctx3, xs);
    const ctx4 = document.getElementById('0-dim-m');
    render1DPrediction2(ctx4, xs.gather([0],1));
    const ctx5 = document.getElementById('1-dim-m');
    render1DPrediction2(ctx5, xs.gather([1],1));
    
    
    console.log("Variance of pca");
    variance(pcaXs);

   

    const ctx7 = document.getElementById('0-dim-pca-m');
    render1DPrediction2(ctx7, pcaXs.gather([0],1));

    const ctx8 = document.getElementById('1-dim-pca-m');
    render1DPrediction2(ctx8, pcaXs.gather([1],1));


    const ctx10 = document.getElementById('0-1-dim-pca-m');
    render2DPrediction2(ctx10, pcaXs.gather([0,1],1));

}

async function main5(xs,gamma) 
{
    
    const kpcaXs = await kpca(xs,gamma);
    const ctx11 = document.getElementById('kpca');
    render2DPrediction2(ctx11, kpcaXs);

    const ctx12 = document.getElementById('kpca-0');
    render1DPrediction2(ctx12, kpcaXs.gather([0],1));

    const ctx13 = document.getElementById('kpca-1');
    render1DPrediction2(ctx13, kpcaXs.gather([1],1));
}

document.getElementById('Show4')
      .addEventListener('click', async() => {
        console.clear();
        main4(data);
        });

document.getElementById('Show5')
      .addEventListener('click', async() => {
        console.clear();
        var g = Number(document.getElementById("gamma").value);
        main5(data,g);
        });         
