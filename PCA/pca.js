import * as tf from '@tensorflow/tfjs';
import * as numeric from 'numeric';
import * as Plotly from 'plotly.js/dist/plotly.js';


const N = 30;
const D = 3;


function render3DPrediction(ctx, xs, proj = false) {
    let xsOriginal = xs.dataSync();
    //console.log(xsOriginal);
    if (proj) 
    {
        const dir = xsOriginal.slice(0, 9);
        xsOriginal = xsOriginal.subarray(9);
        const c1dirX = [0, 5 * dir[0]];
        const c2dirX = [0, 5 * dir[1]];
        const c3dirX = [0, 5 * dir[2]];
        const c1dirY = [0, 5 * dir[3]];
        const c2dirY = [0, 5 * dir[4]];
        const c3dirY = [0, 5 * dir[5]];
        const c1dirZ = [0, 5 * dir[6]];
        const c2dirZ = [0, 5 * dir[7]];
        const c3dirZ = [0, 5 * dir[8]];
        var c1dir = {
            x: c1dirX,
            y: c1dirY,
            z: c1dirZ,
            mode: 'lines',
            opacity: 1.0,
            line: {
                width: 5,
                color: 'red',
                colorscale: 'Viridis'
            },
            name: "PC 1",
            type: 'scatter3d'
        };
        var c2dir = {
            x: c2dirX,
            y: c2dirY,
            z: c2dirZ,
            mode: 'lines',
            opacity: 1.0,
            line: {
                width: 5,
                color: 'green',
                colorscale: 'Viridis'
            },
            name: "PC 2",
            type: 'scatter3d'
        };
        var c3dir = {
            x: c3dirX,
            y: c3dirY,
            z: c3dirZ,
            mode: 'lines',
            opacity: 1.0,
            line: {
                width: 5,
                color: 'blue',
                colorscale: 'Viridis'
            },
            name: "PC 3",
            type: 'scatter3d'
        };
    }


    //c1,c2,c3 are classes
    const c1Data = [];
    const c2Data = [];
    const c3Data = [];
    for (let i = 0; i < xsOriginal.length; i += 3) {
        if (i < D*N) {
            c1Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1], z: xsOriginal[i + 2] });
        }
        else if (D*N <= i && i < 2*D* N) {
            c2Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1], z: xsOriginal[i + 2] });
        }
        else {
            c3Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1], z: xsOriginal[i + 2] });
        }
    }
    const c1X = [];
    const c1Y = [];
    const c1Z = [];
    for (const data1 of c1Data) {
        c1X.push(data1.x);
        c1Y.push(data1.y);
        c1Z.push(data1.z);
    }
    var trace1 = {
        x: c1X, y: c1Y, z: c1Z,
        mode: 'markers',
        marker: {
            size: 8,
            color: 'rgba(0, 128, 256, 0.8)',
            line: {
                width: 0.5
            },
            opacity: 1.0
        },
        text: "Class 1",
        type: 'scatter3d'
    };
    const c2X = [];
    const c2Y = [];
    const c2Z = [];
    for (const data2 of c2Data) {
        c2X.push(data2.x);
        c2Y.push(data2.y);
        c2Z.push(data2.z);
    }
    var trace2 = {
        x: c2X, y: c2Y, z: c2Z,
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
        type: 'scatter3d'
    };
    const c3X = [];
    const c3Y = [];
    const c3Z = [];
    for (const data3 of c3Data) {
        c3X.push(data3.x);
        c3Y.push(data3.y);
        c3Z.push(data3.z);
    }
    var trace3 = {
        x: c3X, y: c3Y, z: c3Z,
        mode: 'markers',
        marker: {
            size: 8,
            color: 'rgba(128, 256, 0, 0.8)',
            line: {
                width: 0.5
            },
            opacity: 1.0
        },
        text: "Class 3",
        type: 'scatter3d'
    };
    
    if (proj) 
    {
        var data = [trace1, trace2, trace3, c1dir, c2dir, c3dir];
    } 
    else 
    {
        var data = [trace1, trace2, trace3];
    }
    var layout = {
        width: 600,
        height: 600,
        margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 0
        },
        scene: {
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' },
            zaxis: { title: 'Z' },
        },
        title: "Data"
    };
    Plotly.newPlot(ctx, data, layout);
    return ctx;
}



function render2DPrediction(ctx, xs) {
    const xsOriginal = xs.dataSync();
    const c1Data = [];
    const c2Data = [];
    const c3Data = [];
    for (let i = 0; i < xsOriginal.length; i += 2) {
        if (i <2*N) {
            c1Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1] });
        }
        else if (2*N <= i && i < 4* N) {
            c2Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1] });
        }
        else {
            c3Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1] });
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
            color: 'rgba(0, 128, 256, 0.8)',
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
    const c3X = [];
    const c3Y = [];
    for (const data3 of c3Data) {
        c3X.push(data3.x);
        c3Y.push(data3.y);
    }
    var trace3 = {
        x: c3X, y: c3Y,
        mode: 'markers',
        marker: {
            size: 8,
            color: 'rgba(128, 256, 0, 0.8)',
            line: {
                width: 0.5
            },
            opacity: 1.0
        },
        text: "Class 3",
        type: 'scatter'
    };
    var data = [trace1, trace2, trace3];
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



function render1DPrediction(ctx, xs) {
    const xsOriginal = xs.dataSync();
    const c1Data = [];
    const c2Data = [];
    const c3Data = [];
    for (let i = 0; i < xsOriginal.length; i += 1) {
        if (i < N) {
            c1Data.push({ x: xsOriginal[i] , y: 0});
        }
        else if (N <= i && i < 2 * N) {
            c2Data.push({ x: xsOriginal[i] , y: 0});
        }
        else {
            c3Data.push({ x: xsOriginal[i] , y: 0});
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
            color: 'rgba(0, 128, 256, 0.8)',
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
    const c3X = [];
    const c3Y = [];
    for (const data3 of c3Data) {
        c3X.push(data3.x);
        c3Y.push(data3.y);
    }
    var trace3 = {
        x: c3X, y: c3Y,
        mode: 'markers',
        marker: {
            size: 8,
            color: 'rgba(128, 256, 0, 0.8)',
            line: {
                width: 0.5
            },
            opacity: 1.0
        },
        text: "Class 3",
        type: 'scatter'
    };
    var data = [trace1, trace2, trace3];
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

async function pca(xs, nComponents) {
    const batch = xs.shape[0];
    const meanValues = xs.mean(0);
    const sub = tf.sub(xs, meanValues);
    const covariance = tf.matMul(sub.transpose(), sub);//(3,3)
    //Numeric does not recognize tensor type of Tensorflow.js,Hence we need to convert
    //the tensor into javascript array
    const covarianceData = tf.util.toNestedArray([D, D], covariance.dataSync());
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


function variance(xs) {
    const v = xs.sub(xs.mean(0)).pow(2).mean();
    console.log(v.dataSync());
}




async function main(xs) {
    const xs_t = xs.gather([0, 1, 2], 1);
    const xs1 = xs.gather([0, 1], 1);
    console.log("Variance of xs1");
    variance(xs1);
    const xs2 = xs.gather([0, 2], 1);
    console.log("Variance of xs2");
    variance(xs2);
    const xs3 = xs.gather([1, 2], 1);
    console.log("Variance of xs3");
    variance(xs3);

    const [axes, pcaXs] = await pca(xs, 3);
    xs_new = axes.concat(xs);
    
    const div = document.getElementById('3dplot');
    render3DPrediction(div, xs_new, proj =true);
    /*
    const ctx1 = document.getElementById('0-1-dim');
    render2DPrediction(ctx1, xs1);
    const ctx2 = document.getElementById('0-2-dim');
    render2DPrediction(ctx2, xs2);
    const ctx3 = document.getElementById('1-2-dim');
    render2DPrediction(ctx3, xs3);
    */
    const ctx4 = document.getElementById('0-dim');
    render1DPrediction(ctx4, xs.gather([0],1));
    const ctx5 = document.getElementById('1-dim');
    render1DPrediction(ctx5, xs.gather([1],1));
    const ctx6 = document.getElementById('2-dim');
    render1DPrediction(ctx6, xs.gather([2],1));
    


    const div2 = document.getElementById('3dplot-pca');

    //const pcaXs = await pca(xs, 3);
    console.log("Variance of pca");
    variance(pcaXs);

    render3DPrediction(div2, pcaXs);

    const ctx7 = document.getElementById('0-dim-pca');
    render1DPrediction(ctx7, pcaXs.gather([0],1));

    const ctx8 = document.getElementById('1-dim-pca');
    render1DPrediction(ctx8, pcaXs.gather([1],1));

    const ctx9 = document.getElementById('2-dim-pca');
    render1DPrediction(ctx9, pcaXs.gather([2],1));

    const ctx10 = document.getElementById('0-1-dim-pca');
    render2DPrediction(ctx10, pcaXs.gather([0,1],1));
}


async function main2(xs) {
    const xs_t = xs.gather([0, 1, 2], 1);
    const xs1 = xs.gather([0, 1], 1);
    console.log("Variance of xs1");
    variance(xs1);
    const xs2 = xs.gather([0, 2], 1);
    console.log("Variance of xs2");
    variance(xs2);
    const xs3 = xs.gather([1, 2], 1);
    console.log("Variance of xs3");
    variance(xs3);

    const [axes, pcaXs] = await pca(xs, 3);
    xs_new = axes.concat(xs);
    
    const div = document.getElementById('3dplot-out');
    render3DPrediction(div, xs_new, proj =true);
    /*
    const ctx1 = document.getElementById('0-1-dim-out');
    render2DPrediction(ctx1, xs1);
    const ctx2 = document.getElementById('0-2-dim-out');
    render2DPrediction(ctx2, xs2);
    const ctx3 = document.getElementById('1-2-dim-out');
    render2DPrediction(ctx3, xs3);
    */
    const ctx4 = document.getElementById('0-dim-out');
    render1DPrediction(ctx4, xs.gather([0],1));
    const ctx5 = document.getElementById('1-dim-out');
    render1DPrediction(ctx5, xs.gather([1],1));
    const ctx6 = document.getElementById('2-dim-out');
    render1DPrediction(ctx6, xs.gather([2],1));
    


    const div2 = document.getElementById('3dplot-pca-out');

    //const pcaXs = await pca(xs, 3);
    console.log("Variance of pca");
    variance(pcaXs);

    render3DPrediction(div2, pcaXs);

    const ctx7 = document.getElementById('0-dim-pca-out');
    render1DPrediction(ctx7, pcaXs.gather([0],1));

    const ctx8 = document.getElementById('1-dim-pca-out');
    render1DPrediction(ctx8, pcaXs.gather([1],1));

    const ctx9 = document.getElementById('2-dim-pca-out');
    render1DPrediction(ctx9, pcaXs.gather([2],1));

    const ctx10 = document.getElementById('0-1-dim-pca-out');
    render2DPrediction(ctx10, pcaXs.gather([0,1],1));
}

const c1 = tf.randomNormal([N, D]).add([1.0, 0.0, 0.0]);
const c2 = tf.randomNormal([N, D]).add([-1.0, 0.0, 0.0]);
const c3 = tf.randomNormal([N, D]).add([0.0, 1.0, 1.0]);



document.getElementById('Show2')
      .addEventListener('click', async() => {
        console.clear()
        var vo1 = Number(document.getElementById("f1").value);
        var vo2 = Number(document.getElementById("f2").value);
        var vo3  = Number(document.getElementById("f3").value);
        const outlier = tf.tensor2d([vo1,vo2,vo3],[1,3]);
        const xs = c1.concat(c2).concat(c3).concat(outlier);
        main2(xs);
        });

document.getElementById('Show1')
      .addEventListener('click', async() => {
        console.clear()
        const xs = c1.concat(c2).concat(c3);
        main(xs);
        });      
