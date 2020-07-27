var svm = require("svm");

var N = 10; //number of data points
var data = new Array(N);
var labels = new Array(N);
var wb; // weights and offset structure
var ss = 50.0; // scaling factor for drawing
var trainstats;
var dirty = true;

var degree_value = 3;
var rbfKernelSigma = 1.0;
var c_sig = 0.050;
var alpha = 0.32;
var svmC = 10.0;
var a_value = 1.0;

var SVM = new svm.SVM();

var kernelid = 0;
var c = document.getElementById("NPGcanvas");
var ctx = c.getContext('2d');

var covg = document.getElementById("covg");
var supp = document.getElementById("supp");
var kern = document.getElementById("kern");
var cdiv = document.getElementById("c");
var sig = document.getElementById("sig");
var a = document.getElementById("a");
var deg = document.getElementById("deg");
var alp = document.getElementById("alp");
var csig = document.getElementById("csig");

data[0] = [-0.4326, 1.1909];
data[1] = [3.0, 4.0];
data[2] = [0.1253, -0.0376];
data[3] = [0.2877, 0.3273];
data[4] = [-1.1465, 0.1746];
data[5] = [1.8133, 2.1139];
data[6] = [2.7258, 3.0668];
data[7] = [1.4117, 2.0593];
data[8] = [4.1832, 1.9044];
data[9] = [1.8636, 1.1677];
labels[0] = 1;
labels[1] = 1;
labels[2] = 1;
labels[3] = 1;
labels[4] = 1;
labels[5] = -1;
labels[6] = -1;
labels[7] = -1;
labels[8] = -1;
labels[9] = -1;

setChange(10);

function myinit() {
    retrainSVM();
}

function poly() {
    return function(v1, v2) {
        var s = 0;
        for (var q = 0; q < v1.length; q++) { s += ((v1[q] * v2[q]) + a_value) }
        return Math.pow(s, degree_value);
    }

}

function sigmoid() {
    return function(v1, v2) {
        var s = 0;
        for (var q = 0; q < v1.length; q++) { s += (v1[q] * v2[q]) }
        return Math.tanh(((alpha * s) + c_sig));
    }
}

function hideSig() {
    document.getElementById("sigreport").style.display = "none";
    document.getElementById("slider2").style.display = "none";
}

function hideDeg() {
    document.getElementById("degreport").style.display = "none";
    document.getElementById("slider3").style.display = "none";
}

function hidea() {
    document.getElementById("areport").style.display = "none";
    document.getElementById("slider4").style.display = "none";
}

function hidealp() {
    document.getElementById("alpreport").style.display = "none";
    document.getElementById("slider5").style.display = "none";
}

function hidecsig() {
    document.getElementById("csigreport").style.display = "none";
    document.getElementById("slider6").style.display = "none";
}

function showSig() {
    document.getElementById("report").appendChild(document.getElementById("sigreport"));
    document.getElementById("rider").appendChild(document.getElementById("slider2"));
    document.getElementById("sigreport").style.display = "block";
    document.getElementById("slider2").style.display = "block";
}

function showDeg() {
    document.getElementById("report").appendChild(document.getElementById("degreport"));
    document.getElementById("rider").appendChild(document.getElementById("slider3"));
    document.getElementById("degreport").style.display = "block";
    document.getElementById("slider3").style.display = "block";
}

function showa() {
    document.getElementById("preport").appendChild(document.getElementById("areport"));
    document.getElementById("provider").appendChild(document.getElementById("slider4"));
    document.getElementById("areport").style.display = "block";
    document.getElementById("slider4").style.display = "block";
}

function showalp() {
    document.getElementById("report").appendChild(document.getElementById("alpreport"));
    document.getElementById("rider").appendChild(document.getElementById("slider5"));
    document.getElementById("alpreport").style.display = "block";
    document.getElementById("slider5").style.display = "block";
}

function showcsig() {
    document.getElementById("preport").appendChild(document.getElementById("csigreport"));
    document.getElementById("provider").appendChild(document.getElementById("slider6"));
    document.getElementById("csigreport").style.display = "block";
    document.getElementById("slider6").style.display = "block";
}

function retrainSVM() {

    if (kernelid == 0) {
        trainstats = SVM.train(data, labels, { kernel: 'linear', C: svmC });
        wb = SVM.getWeights();
        hideSig();
        hidea();
        hidealp();
        hidecsig();
        hideDeg();
    } else if (kernelid == 1) {
        trainstats = SVM.train(data, labels, { kernel: 'rbf', rbfsigma: rbfKernelSigma, C: svmC });
        showSig();
        hidea();
        hidealp();
        hidecsig();
        hideDeg();
    } else if (kernelid == 2) {
        trainstats = SVM.train(data, labels, {
            kernel: poly(),
            C: svmC
        });
        hideSig();
        showa();
        hidealp();
        hidecsig();
        showDeg();
    } else if (kernelid == 3) {
        trainstats = SVM.train(data, labels, {
            kernel: sigmoid(),
            rbfsigma: rbfKernelSigma,
            C: svmC
        });
        hideSig();
        hidea();
        showalp();
        showcsig();
        hideDeg();
    }

    dirty = true; // to redraw screen
}

function update() {}



function draw() {
    if (!dirty) return;

    ctx.clearRect(0, 0, WIDTH, HEIGHT);

    // draw decisions in the grid
    var density = 4.0;
    for (var x = 0.0; x <= WIDTH; x += density) {
        for (var y = 0.0; y <= HEIGHT; y += density) {
            var dec = SVM.marginOne([(x - WIDTH / 2) / ss, (y - HEIGHT / 2) / ss]);
            if (dec > 0) ctx.fillStyle = '#4169E1';
            else ctx.fillStyle = '#ffef00';
            ctx.fillRect(x - density / 2 - 1, y - density - 1, density + 2, density + 2);
        }
    }


    // draw datapoints. Draw support vectors larger
    ctx.strokeStyle = 'rgb(0,0,0)';
    for (var i = 0; i < N; i++) {

        if (labels[i] == 1) ctx.fillStyle = '#0080FF';
        else ctx.fillStyle = '#FFAA00';

        if (SVM.alpha[i] > 1e-2) ctx.lineWidth = 3; // distinguish support vectors
        else ctx.lineWidth = 1;

        drawCircle(data[i][0] * ss + WIDTH / 2, data[i][1] * ss + HEIGHT / 2, Math.floor(5));
    }

    // if linear kernel, draw decision boundary and margin lines
    if (kernelid == 0) {

        var xs = [-5, 5];
        var ys = [0, 0];
        ys[0] = (-wb.b - wb.w[0] * xs[0]) / wb.w[1];
        ys[1] = (-wb.b - wb.w[0] * xs[1]) / wb.w[1];
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        // wx+b=0 line
        ctx.moveTo(xs[0] * ss + WIDTH / 2, ys[0] * ss + HEIGHT / 2);
        ctx.lineTo(xs[1] * ss + WIDTH / 2, ys[1] * ss + HEIGHT / 2);
        // wx+b=1 line
        ctx.moveTo(xs[0] * ss + WIDTH / 2, (ys[0] - 1.0 / wb.w[1]) * ss + HEIGHT / 2);
        ctx.lineTo(xs[1] * ss + WIDTH / 2, (ys[1] - 1.0 / wb.w[1]) * ss + HEIGHT / 2);
        // wx+b=-1 line
        ctx.moveTo(xs[0] * ss + WIDTH / 2, (ys[0] + 1.0 / wb.w[1]) * ss + HEIGHT / 2);
        ctx.lineTo(xs[1] * ss + WIDTH / 2, (ys[1] + 1.0 / wb.w[1]) * ss + HEIGHT / 2);

        for (var i = 0; i < N; i++) {
            if (SVM.alpha[i] < 1e-2) continue;
            if (labels[i] == 1) {
                ys[0] = (1 - wb.b - wb.w[0] * xs[0]) / wb.w[1];
                ys[1] = (1 - wb.b - wb.w[0] * xs[1]) / wb.w[1];
            } else {
                ys[0] = (-1 - wb.b - wb.w[0] * xs[0]) / wb.w[1];
                ys[1] = (-1 - wb.b - wb.w[0] * xs[1]) / wb.w[1];
            }
            var u = (data[i][0] - xs[0]) * (xs[1] - xs[0]) + (data[i][1] - ys[0]) * (ys[1] - ys[0]);
            u = u / ((xs[0] - xs[1]) * (xs[0] - xs[1]) + (ys[0] - ys[1]) * (ys[0] - ys[1]));
            var xi = xs[0] + u * (xs[1] - xs[0]);
            var yi = ys[0] + u * (ys[1] - ys[0]);
            ctx.moveTo(data[i][0] * ss + WIDTH / 2, data[i][1] * ss + HEIGHT / 2);
            ctx.lineTo(xi * ss + WIDTH / 2, yi * ss + HEIGHT / 2);
        }
        ctx.stroke();
    }

    ctx.fillStyle = 'rgb(0,0,0)';
    covg.innerHTML = "Converged in " + trainstats.iters + " iterations";

    var numsupp = 0;
    for (var i = 0; i < N; i++) { if (SVM.alpha[i] > 1e-2) numsupp++; }
    supp.innerHTML = "No. of Support Vectors: " + numsupp;

    kern.innerHTML = "Using Linear kernel";
    cdiv.innerHTML = "C = " + svmC.toPrecision(2);
    if (kernelid == 0) {
        document.getElementById("linear_info").style.display = "block";
        document.getElementById("poly_info").style.display = "none";
        document.getElementById("rbf_info").style.display = "none";
        document.getElementById("sigmoid_info").style.display = "none";
        kern.innerHTML = "Using Linear kernel";
        cdiv.innerHTML = "C = " + svmC.toPrecision(2);
    }
    if (kernelid == 1) {
        document.getElementById("linear_info").style.display = "none";
        document.getElementById("poly_info").style.display = "none";
        document.getElementById("rbf_info").style.display = "block";
        document.getElementById("sigmoid_info").style.display = "none";
        kern.innerHTML = "Using Gaussian kernel";
        cdiv.innerHTML = "C = " + svmC.toPrecision(2);
        sig.style.display = "list-item";
        csig.style.display = "none";
        a.style.display = "none";
        alp.style.display = "none";
        deg.style.display = "none";
        sig.innerHTML = "Gaussian Kernel Sigma: " + rbfKernelSigma.toPrecision(2);
    }
    if (kernelid == 2) {
        document.getElementById("linear_info").style.display = "none";
        document.getElementById("poly_info").style.display = "block";
        document.getElementById("rbf_info").style.display = "none";
        document.getElementById("sigmoid_info").style.display = "none";
        kern.innerHTML = "Using Polynomial kernel";
        cdiv.innerHTML = "C = " + svmC.toPrecision(2);
        deg.style.display = "list-item";
        sig.style.display = "none";
        alp.style.display = "none";
        csig.style.display = "none";
        deg.innerHTML = "Polynomial Kernel Degree: " + degree_value;
        a.style.display = "list-item";
        a.innerHTML = "Polynomial Kernel a: " + a_value.toPrecision(2);
    }
    if (kernelid == 3) {
        document.getElementById("linear_info").style.display = "none";
        document.getElementById("poly_info").style.display = "none";
        document.getElementById("rbf_info").style.display = "none";
        document.getElementById("sigmoid_info").style.display = "block";
        kern.innerHTML = "Using Sigmoid kernel";
        cdiv.innerHTML = "C = " + svmC.toPrecision(2);
        alp.style.display = "list-item";
        sig.style.display = "none";
        deg.style.display = "none";
        a.style.display = "none";
        alp.innerHTML = "Sigmoid Kernel alpha: " + alpha.toPrecision(2);
        csig.style.display = "list-item";
        csig.innerHTML = "Sigmoid Kernel c: " + c_sig.toPrecision(2);
    }

}

function drawCircle(x, y, r) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2, true);
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
}

function mouseClick(x, y, shiftPressed) {

    // add datapoint at location of click
    data[N] = [(x - WIDTH / 2) / ss, (y - HEIGHT / 2) / ss];
    labels[N] = shiftPressed ? 1 : -1;
    N += 1;

    // retrain the svm
    retrainSVM();
}

function keyUp(key) {

    if (key == 67) { // 'c'

        // clear the points
        data = data.splice(0, 10);
        labels = labels.splice(0, 10);
        N = 10;
        retrainSVM();
    }

    if (key == 76) { // 'l'
        // Switch to linear kernel

        kernelid = 0;
        retrainSVM();
    }

    if (key == 82) { // 'r'
        // Switch to rbf kernel

        kernelid = 1;
        retrainSVM();
    }

    if (key == 80) { // 'p'
        // Switch to polynomial kernel

        kernelid = 2;
        retrainSVM();
    }

    if (key == 83) { // 's'
        // Switch to sigmoid kernel

        kernelid = 3;
        retrainSVM();
    }

    if (key == 85) { // 'u'
        // Undo
        data = data.splice(0, N - 1);
        labels = labels.splice(0, N - 1);
        N = N - 1;
        retrainSVM();
    }
}

function keyDown(key) {}


// UI stuff
function refreshC(event, ui) {
    var logC = ui.value;
    svmC = Math.pow(10, logC);
    $("#creport").text("C = " + svmC.toPrecision(2));
    retrainSVM();
}

function eventClick(e) {
    var x;
    var y;
    if (e.pageX || e.pageY) {
        x = e.pageX;
        y = e.pageY;
    } else {
        x = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
        y = e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
    }
    x -= canvas.offsetLeft;
    y -= canvas.offsetTop;
    mouseClick(x, y, e.shiftKey);
}

function eventKeyUp(e) {
    var keycode = ('which' in e) ? e.which : e.keyCode;
    keyUp(keycode);
}

function eventKeyDown(e) {
    var keycode = ('which' in e) ? e.which : e.keyCode;
    keyDown(keycode);
}

function setChange(FPS) {

    canvas = document.getElementById('NPGcanvas');
    ctx = canvas.getContext('2d');

    WIDTH = canvas.width;
    HEIGHT = canvas.height;
    canvas.addEventListener('click', eventClick, false);
    document.addEventListener('keyup', eventKeyUp, true);
    document.addEventListener('keydown', eventKeyDown, true);
    setInterval(main, 1000 / FPS);

    myinit();
}

function main() {
    update();
    draw();
}

function refreshC(event, ui) {
    var logC = ui.value;
    svmC = Math.pow(10, logC);
    $("#creport").text("C = " + svmC.toPrecision(2));
    retrainSVM();
}

function refreshSig(event, ui) {
    var logSig = ui.value;
    rbfKernelSigma = Math.pow(10, logSig);
    $("#sigreport").text("Gaussian Kernel sigma = " + rbfKernelSigma.toPrecision(2));
    retrainSVM();
}

function refreshDeg(event, ui) {
    var logDeg = ui.value;
    degree_value = logDeg;
    $("#degreport").text("Polynomial Kernel degree = " + degree_value);
    retrainSVM();
}

function refreshA(event, ui) {
    var logA = ui.value;
    a_value = Math.pow(10, logA);
    $("#areport").text("Polynomial Kernel a = " + a_value.toPrecision(2));
    retrainSVM();

}

function refreshAlpha(event, ui) {
    var logAlpha = ui.value;
    alpha = Math.pow(10, logAlpha);
    $("#alpreport").text("Sigmoid Kernel alpha = " + alpha.toPrecision(2));
    retrainSVM();

}

function refreshCsig(event, ui) {
    var logCsig = ui.value;
    c_sig = Math.pow(10, logCsig);
    $("#csigreport").text("Sigmoid Kernel c-sigma = " + c_sig.toPrecision(2));
    retrainSVM();
}



$(function() {
    // for C parameter
    $("#slider1").slider({
        orientation: "horizontal",
        animate: "slow",
        slide: refreshC,
        max: 2.0,
        min: -2.0,
        step: 0.1,
        value: 0.0
    });

    // for rbf kernel sigma
    $("#slider2").slider({
        orientation: "horizontal",
        animate: "slow",
        slide: refreshSig,
        max: 2.0,
        min: -2.0,
        step: 0.1,
        value: 0.0
    });

    $("#slider3").slider({
        orientation: "horizontal",
        animate: "slow",
        slide: refreshDeg,
        max: 8,
        min: 2,
        step: 1,
        value: 3
    });

    $("#slider4").slider({
        orientation: "horizontal",
        animate: "slow",
        slide: refreshA,
        max: 2.0,
        min: -2.0,
        step: 0.1,
        value: 0.0
    });

    $("#slider5").slider({
        orientation: "horizontal",
        animate: "slow",
        slide: refreshAlpha,
        max: 2.0,
        min: -2.0,
        step: 0.1,
        value: 0.0
    });

    $("#slider6").slider({
        orientation: "horizontal",
        animate: "slow",
        slide: refreshCsig,
        max: 2.0,
        min: -2.0,
        step: 0.1,
        value: 0.0
    });

    $("#slider1").css('border-color', 'rgb(255,0,0)');
    $("#slider2").css('border-color', 'rgb(0,180,0)');
    $("#slider3").css('border-color', 'rgb(0,180,0)');
    $("#slider4").css('border-color', 'rgb(0,0,255)');
    $("#slider5").css('border-color', 'rgb(0,180,0)');
    $("#slider6").css('border-color', 'rgb(0,0,255)');
    $("#slider1").css('background', 'rgb(247, 188, 188)');
    $("#slider2").css('background', 'rgb(192, 247, 188)');
    $("#slider3").css('background', 'rgb(192, 247, 188)');
    $("#slider4").css('background', 'rgb(188, 217, 247)');
    $("#slider5").css('background', 'rgb(192, 247, 188)');
    $("#slider6").css('background', 'rgb(188, 217, 247)');
    $(".ui-slider-handle").css('background', 'rgb(255,255,204)');
});

$(document).ready(function() {


    $("#collapseInfo").click(function() {
        $(this).toggleClass("active");
        if ($(this).hasClass("active")) {
            $(this).text("Show Less..");
        } else {
            $(this).text("Show More..");
        }
    });

    $("#collapseInst").click(function() {
        $(this).toggleClass("active");
        if ($(this).hasClass("active")) {
            $(this).text("Hide Instructions..");
        } else {
            $(this).text("Show Instructions..");
        }
    });
});

var prevKernel;
var prevData;
var prevLabel;

function getPrevKernel() {
    return prevKernel;
}

function getPrevData() {
    return prevData;
}

function getPrevLabel() {
    return prevLabel;
}

function setPrevKernel(kid) {
    prevKernel = kid;
}

function setPrevData(x) {
    prevData = x;
}

function setPrevLabel(y) {
    prevLabel = y;
}