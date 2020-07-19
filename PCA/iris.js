const D1 = 4;
const N1 = 50;


function render3DPrediction1(ctx, xs, name, proj = false) {
    let xsOriginal = xs.dataSync();
    if (proj) {
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
        if (i < 3 * N1) {
            c1Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1], z: xsOriginal[i + 2] });
        } else if (3 * N1 <= i && i < 6 * N1) {
            c2Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1], z: xsOriginal[i + 2] });
        } else {
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
        x: c1X,
        y: c1Y,
        z: c1Z,
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
        x: c2X,
        y: c2Y,
        z: c2Z,
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
        x: c3X,
        y: c3Y,
        z: c3Z,
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

    if (proj) {
        var data = [trace1, trace2, trace3, c1dir, c2dir, c3dir];
    } else {
        var data = [trace1, trace2, trace3];
    }
    var layout = {
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
        autosize: true,
        title: name
    };
    Plotly.newPlot(ctx, data, layout);
    return ctx;
}



function render2DPrediction1(ctx, xs, name) {
    const xsOriginal = xs.dataSync();
    const c1Data = [];
    const c2Data = [];
    const c3Data = [];
    for (let i = 0; i < xsOriginal.length; i += 2) {
        if (i < 2 * N1) {
            c1Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1] });
        } else if (2 * N1 <= i && i < 4 * N1) {
            c2Data.push({ x: xsOriginal[i], y: xsOriginal[i + 1] });
        } else {
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
        x: c1X,
        y: c1Y,
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
        x: c2X,
        y: c2Y,
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
        x: c3X,
        y: c3Y,
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
        autosize: true,
        title: name
    };
    Plotly.newPlot(ctx, data, layout);
    return ctx;
}



function render1DPrediction1(ctx, xs, name) {
    const xsOriginal = xs.dataSync();
    const c1Data = [];
    const c2Data = [];
    const c3Data = [];
    for (let i = 0; i < xsOriginal.length; i += 1) {
        if (i < N1) {
            c1Data.push({ x: xsOriginal[i], y: 0 });
        } else if (N1 <= i && i < 2 * N1) {
            c2Data.push({ x: xsOriginal[i], y: 0 });
        } else {
            c3Data.push({ x: xsOriginal[i], y: 0 });
        }
    }
    const c1X = [];
    const c1Y = [];
    for (const data1 of c1Data) {
        c1X.push(data1.x);
        c1Y.push(data1.y);
    }
    var trace1 = {
        x: c1X,
        y: c1Y,
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
        x: c2X,
        y: c2Y,
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
        x: c3X,
        y: c3Y,
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
        scene: {
            xaxis: { title: 'X' },
        },
        autosize: true,
        title: name
    };
    Plotly.newPlot(ctx, data, layout);
    return ctx;
}



var xl = Array();
var yl = Array();
yl.push(0);
for (var z = 0; z <= D1; z++) {
    xl.push(z);
}

async function pca2(xs, nComponents) {
    const batch = xs.shape[0];
    const meanValues = xs.mean(0);
    const sub = tf.sub(xs, meanValues);
    const covariance = tf.matMul(sub.transpose(), sub); //(3,3)
    //Numeric does not recognize tensor type of Tensorflow.js,Hence we need to convert
    //the tensor into javascript array
    const covarianceData = tf.util.toNestedArray([D1, D1], covariance.dataSync());
    const eig = numeric.eig(covarianceData);
    //returns eigen vectors and eigen values
    console.log("eigen values");
    console.log(eig.lambda.x);

    yl.push(eig.lambda.x[0]);
    yl.push(eig.lambda.x[0] + eig.lambda.x[1]);
    yl.push(eig.lambda.x[0] + eig.lambda.x[1] + eig.lambda.x[2]);
    yl.push(eig.lambda.x[0] + eig.lambda.x[1] + eig.lambda.x[2] + eig.lambda.x[3]);

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

function ret_variance(xs) {
    const v = xs.sub(xs.mean(0)).pow(2).mean();
    return v.arraySync();
}

//150 egs
const iris_data = [
    [5.1, 3.5, 1.4, 0.2, 0],
    [4.9, 3.0, 1.4, 0.2, 0],
    [4.7, 3.2, 1.3, 0.2, 0],
    [4.6, 3.1, 1.5, 0.2, 0],
    [5.0, 3.6, 1.4, 0.2, 0],
    [5.4, 3.9, 1.7, 0.4, 0],
    [4.6, 3.4, 1.4, 0.3, 0],
    [5.0, 3.4, 1.5, 0.2, 0],
    [4.4, 2.9, 1.4, 0.2, 0],
    [4.9, 3.1, 1.5, 0.1, 0],
    [5.4, 3.7, 1.5, 0.2, 0],
    [4.8, 3.4, 1.6, 0.2, 0],
    [4.8, 3.0, 1.4, 0.1, 0],
    [4.3, 3.0, 1.1, 0.1, 0],
    [5.8, 4.0, 1.2, 0.2, 0],
    [5.7, 4.4, 1.5, 0.4, 0],
    [5.4, 3.9, 1.3, 0.4, 0],
    [5.1, 3.5, 1.4, 0.3, 0],
    [5.7, 3.8, 1.7, 0.3, 0],
    [5.1, 3.8, 1.5, 0.3, 0],
    [5.4, 3.4, 1.7, 0.2, 0],
    [5.1, 3.7, 1.5, 0.4, 0],
    [4.6, 3.6, 1.0, 0.2, 0],
    [5.1, 3.3, 1.7, 0.5, 0],
    [4.8, 3.4, 1.9, 0.2, 0],
    [5.0, 3.0, 1.6, 0.2, 0],
    [5.0, 3.4, 1.6, 0.4, 0],
    [5.2, 3.5, 1.5, 0.2, 0],
    [5.2, 3.4, 1.4, 0.2, 0],
    [4.7, 3.2, 1.6, 0.2, 0],
    [4.8, 3.1, 1.6, 0.2, 0],
    [5.4, 3.4, 1.5, 0.4, 0],
    [5.2, 4.1, 1.5, 0.1, 0],
    [5.5, 4.2, 1.4, 0.2, 0],
    [4.9, 3.1, 1.5, 0.1, 0],
    [5.0, 3.2, 1.2, 0.2, 0],
    [5.5, 3.5, 1.3, 0.2, 0],
    [4.9, 3.1, 1.5, 0.1, 0],
    [4.4, 3.0, 1.3, 0.2, 0],
    [5.1, 3.4, 1.5, 0.2, 0],
    [5.0, 3.5, 1.3, 0.3, 0],
    [4.5, 2.3, 1.3, 0.3, 0],
    [4.4, 3.2, 1.3, 0.2, 0],
    [5.0, 3.5, 1.6, 0.6, 0],
    [5.1, 3.8, 1.9, 0.4, 0],
    [4.8, 3.0, 1.4, 0.3, 0],
    [5.1, 3.8, 1.6, 0.2, 0],
    [4.6, 3.2, 1.4, 0.2, 0],
    [5.3, 3.7, 1.5, 0.2, 0],
    [5.0, 3.3, 1.4, 0.2, 0],
    [7.0, 3.2, 4.7, 1.4, 1],
    [6.4, 3.2, 4.5, 1.5, 1],
    [6.9, 3.1, 4.9, 1.5, 1],
    [5.5, 2.3, 4.0, 1.3, 1],
    [6.5, 2.8, 4.6, 1.5, 1],
    [5.7, 2.8, 4.5, 1.3, 1],
    [6.3, 3.3, 4.7, 1.6, 1],
    [4.9, 2.4, 3.3, 1.0, 1],
    [6.6, 2.9, 4.6, 1.3, 1],
    [5.2, 2.7, 3.9, 1.4, 1],
    [5.0, 2.0, 3.5, 1.0, 1],
    [5.9, 3.0, 4.2, 1.5, 1],
    [6.0, 2.2, 4.0, 1.0, 1],
    [6.1, 2.9, 4.7, 1.4, 1],
    [5.6, 2.9, 3.6, 1.3, 1],
    [6.7, 3.1, 4.4, 1.4, 1],
    [5.6, 3.0, 4.5, 1.5, 1],
    [5.8, 2.7, 4.1, 1.0, 1],
    [6.2, 2.2, 4.5, 1.5, 1],
    [5.6, 2.5, 3.9, 1.1, 1],
    [5.9, 3.2, 4.8, 1.8, 1],
    [6.1, 2.8, 4.0, 1.3, 1],
    [6.3, 2.5, 4.9, 1.5, 1],
    [6.1, 2.8, 4.7, 1.2, 1],
    [6.4, 2.9, 4.3, 1.3, 1],
    [6.6, 3.0, 4.4, 1.4, 1],
    [6.8, 2.8, 4.8, 1.4, 1],
    [6.7, 3.0, 5.0, 1.7, 1],
    [6.0, 2.9, 4.5, 1.5, 1],
    [5.7, 2.6, 3.5, 1.0, 1],
    [5.5, 2.4, 3.8, 1.1, 1],
    [5.5, 2.4, 3.7, 1.0, 1],
    [5.8, 2.7, 3.9, 1.2, 1],
    [6.0, 2.7, 5.1, 1.6, 1],
    [5.4, 3.0, 4.5, 1.5, 1],
    [6.0, 3.4, 4.5, 1.6, 1],
    [6.7, 3.1, 4.7, 1.5, 1],
    [6.3, 2.3, 4.4, 1.3, 1],
    [5.6, 3.0, 4.1, 1.3, 1],
    [5.5, 2.5, 4.0, 1.3, 1],
    [5.5, 2.6, 4.4, 1.2, 1],
    [6.1, 3.0, 4.6, 1.4, 1],
    [5.8, 2.6, 4.0, 1.2, 1],
    [5.0, 2.3, 3.3, 1.0, 1],
    [5.6, 2.7, 4.2, 1.3, 1],
    [5.7, 3.0, 4.2, 1.2, 1],
    [5.7, 2.9, 4.2, 1.3, 1],
    [6.2, 2.9, 4.3, 1.3, 1],
    [5.1, 2.5, 3.0, 1.1, 1],
    [5.7, 2.8, 4.1, 1.3, 1],
    [6.3, 3.3, 6.0, 2.5, 2],
    [5.8, 2.7, 5.1, 1.9, 2],
    [7.1, 3.0, 5.9, 2.1, 2],
    [6.3, 2.9, 5.6, 1.8, 2],
    [6.5, 3.0, 5.8, 2.2, 2],
    [7.6, 3.0, 6.6, 2.1, 2],
    [4.9, 2.5, 4.5, 1.7, 2],
    [7.3, 2.9, 6.3, 1.8, 2],
    [6.7, 2.5, 5.8, 1.8, 2],
    [7.2, 3.6, 6.1, 2.5, 2],
    [6.5, 3.2, 5.1, 2.0, 2],
    [6.4, 2.7, 5.3, 1.9, 2],
    [6.8, 3.0, 5.5, 2.1, 2],
    [5.7, 2.5, 5.0, 2.0, 2],
    [5.8, 2.8, 5.1, 2.4, 2],
    [6.4, 3.2, 5.3, 2.3, 2],
    [6.5, 3.0, 5.5, 1.8, 2],
    [7.7, 3.8, 6.7, 2.2, 2],
    [7.7, 2.6, 6.9, 2.3, 2],
    [6.0, 2.2, 5.0, 1.5, 2],
    [6.9, 3.2, 5.7, 2.3, 2],
    [5.6, 2.8, 4.9, 2.0, 2],
    [7.7, 2.8, 6.7, 2.0, 2],
    [6.3, 2.7, 4.9, 1.8, 2],
    [6.7, 3.3, 5.7, 2.1, 2],
    [7.2, 3.2, 6.0, 1.8, 2],
    [6.2, 2.8, 4.8, 1.8, 2],
    [6.1, 3.0, 4.9, 1.8, 2],
    [6.4, 2.8, 5.6, 2.1, 2],
    [7.2, 3.0, 5.8, 1.6, 2],
    [7.4, 2.8, 6.1, 1.9, 2],
    [7.9, 3.8, 6.4, 2.0, 2],
    [6.4, 2.8, 5.6, 2.2, 2],
    [6.3, 2.8, 5.1, 1.5, 2],
    [6.1, 2.6, 5.6, 1.4, 2],
    [7.7, 3.0, 6.1, 2.3, 2],
    [6.3, 3.4, 5.6, 2.4, 2],
    [6.4, 3.1, 5.5, 1.8, 2],
    [6.0, 3.0, 4.8, 1.8, 2],
    [6.9, 3.1, 5.4, 2.1, 2],
    [6.7, 3.1, 5.6, 2.4, 2],
    [6.9, 3.1, 5.1, 2.3, 2],
    [5.8, 2.7, 5.1, 1.9, 2],
    [6.8, 3.2, 5.9, 2.3, 2],
    [6.7, 3.3, 5.7, 2.5, 2],
    [6.7, 3.0, 5.2, 2.3, 2],
    [6.3, 2.5, 5.0, 1.9, 2],
    [6.5, 3.0, 5.2, 2.0, 2],
    [6.2, 3.4, 5.4, 2.3, 2],
    [5.9, 3.0, 5.1, 1.8, 2],
];

const numExamples = iris_data.length;
const tensor_dataset = tf.tensor2d(iris_data, [numExamples, 5]);





async function main3(xs) {
    const xs_t = xs.gather([0, 1, 2, 3], 1);


    const [axes, pcaXs] = await pca2(xs, 4);
    xs_new = axes.concat(xs);

    //const div = document.getElementById('3dplot');
    //render3DPrediction(div, xs_new, proj =true);

    const ctx4 = document.getElementById('0-dim-iris');
    render1DPrediction1(ctx4, xs.gather([0], 1), "1st Feature");
    const ctx5 = document.getElementById('1-dim-iris');
    render1DPrediction1(ctx5, xs.gather([1], 1), "2nd Feature");
    const ctx6 = document.getElementById('2-dim-iris');
    render1DPrediction1(ctx6, xs.gather([2], 1), "3rd Feature");
    const ctx11 = document.getElementById('3-dim-iris');
    render1DPrediction1(ctx11, xs.gather([3], 1), "4th Feature");

    console.log("Variance of pca");
    variance(pcaXs);


    yl.push(0);



    Plotly.newPlot('var-pca', [{
        x: xl,
        y: yl,
        line: { simplify: false },
    }], {}, { showSendToCloud: true });

    function plot() {
        Plotly.animate(
            "var-pca", {
                data: [{ y: yl }],
                traces: [0],
                layout: {
                    autosize: true,
                    title: "Explained Variance",
                },
            }, {
                transition: {
                    duration: 500,
                    easing: "cubic-in-out",
                },
                frame: {
                    duration: 500,
                },
            }
        );
    }
    plot();

    const div2 = document.getElementById('3dplot-pca-iris');
    render3DPrediction1(div2, pcaXs.gather([0, 1, 2], 1), "3D PCA plot");

    const ctx7 = document.getElementById('0-dim-pca-iris');
    render1DPrediction1(ctx7, pcaXs.gather([0], 1), "1st Principal Component");

    const ctx8 = document.getElementById('1-dim-pca-iris');
    render1DPrediction1(ctx8, pcaXs.gather([1], 1), "2nd Principal Component");

    const ctx9 = document.getElementById('2-dim-pca-iris');
    render1DPrediction1(ctx9, pcaXs.gather([2], 1), "3rd Principal Component");

    const ctx12 = document.getElementById('3-dim-pca-iris');
    render1DPrediction1(ctx12, pcaXs.gather([3], 1), "4th Principal Component");

    const ctx10 = document.getElementById('0-1-dim-pca-iris');
    render2DPrediction1(ctx10, pcaXs.gather([0, 1], 1), "2D PCA plot");
}

document.getElementById('Show3')
    .addEventListener('click', async() => {
        console.clear()
        const xs1 = tensor_dataset.gather([0, 1, 2, 3], 1);
        main3(xs1);
    });