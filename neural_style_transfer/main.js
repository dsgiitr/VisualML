import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';
tf.ENV.set('WEBGL_PACK', false);
import links from './links';

class Main {
    constructor() {

        this.fileSelect = document.getElementById('file-select');

        // Initialize model selection
        this.modelSelectStyle = document.getElementById('model-select-style');
        this.modelSelectStyle.onchange = (net) => {
            if (net.target.value === 'mobilenet') {
                this.disableStylizeButtons();
                this.loadMobileNetStyleModel().then(model => {
                    this.styleNet = model;
                }).finally(() => this.enableStylizeButtons());
            } else if (net.target.value === 'inception') {
                this.disableStylizeButtons();
                this.loadInceptionStyleModel().then(model => {
                    this.styleNet = model;
                }).finally(() => this.enableStylizeButtons());
            }
        }

        this.modelSelectTransformer = document.getElementById('model-select-transformer');
        this.modelSelectTransformer.onchange = (net) => {
            if (net.target.value === 'original') {
                this.disableStylizeButtons();
                this.loadOriginalTransformerModel().then(model => {
                    this.transformNet = model;
                }).finally(() => this.enableStylizeButtons());
            } else if (net.target.value === 'separable') {
                this.disableStylizeButtons();
                this.loadSeparableTransformerModel().then(model => {
                    this.transformNet = model;
                }).finally(() => this.enableStylizeButtons());
            }
        }

        this.initalizeWebcamVariables();
        this.initializeStyleTransfer();

        Promise.all([
            this.loadMobileNetStyleModel(),
            this.loadSeparableTransformerModel(),
        ]).then(([styleNet, transformNet]) => {
            console.log('Loaded styleNet');
            this.styleNet = styleNet;
            this.transformNet = transformNet;
            this.enableStylizeButtons()
        });
    }

    async loadMobileNetStyleModel() {
        if (!this.mobileStyleNet) {
            this.mobileStyleNet = await tf.loadGraphModel(
                'saved_model_style_js/model.json');
        }

        return this.mobileStyleNet;
    }

    async loadInceptionStyleModel() {
        if (!this.inceptionStyleNet) {
            this.inceptionStyleNet = await tf.loadGraphModel(
                'saved_model_style_inception_js/model.json');
        }

        return this.inceptionStyleNet;
    }

    async loadOriginalTransformerModel() {
        if (!this.originalTransformNet) {
            this.originalTransformNet = await tf.loadGraphModel(
                'saved_model_transformer_js/model.json'
            );
        }

        return this.originalTransformNet;
    }

    async loadSeparableTransformerModel() {
        if (!this.separableTransformNet) {
            this.separableTransformNet = await tf.loadGraphModel(
                'saved_model_transformer_separable_js/model.json'
            );
        }

        return this.separableTransformNet;
    }

    async gradualStyler(n = 10, i = 1) {
        this.startStyling(this.stylized, this.styleRatio * (i / n)).finally(() => {
            if (i < n) {
                var j = i + 1;
                var progressBar = document.getElementById("Bar");
                document.getElementById("counter").innerHTML = "Epochs: " + j * 40;
                var width = 100 * (j / n);
                if (width >= 102) {
                    progressBar.style.width = 0;
                } else {
                    progressBar.style.width = width + "%";
                }
                this.gradualStyler(n, i + 1);
            } else {
                this.enableStylizeButtons();
            }
        });
    }

    initializeStyleTransfer() {
        // Initialize images
        this.contentImg = document.getElementById('content-img');
        this.contentImg.onerror = () => {
            alert("Error loading " + this.contentImg.src + ".");
        }
        this.styleImg = document.getElementById('style-img');
        this.styleImg.onerror = () => {
            alert("Error loading " + this.styleImg.src + ".");
        }
        this.stylized = document.getElementById('stylized');

        // Initialize images
        this.contentImgSlider = document.getElementById('content-img-size');
        this.connectImageAndSizeSlider(this.contentImg, this.contentImgSlider);
        this.styleImgSlider = document.getElementById('style-img-size');
        this.styleImgSquare = document.getElementById('style-img-square');
        this.connectImageAndSizeSlider(this.styleImg, this.styleImgSlider, this.styleImgSquare);

        this.styleRatio = 1.0
        this.styleRatioSlider = document.getElementById('stylized-img-ratio');
        this.styleRatioSlider.oninput = (net) => {
            this.styleRatio = net.target.value / 100.;
        }

        // Initialize buttons
        this.styleButton = document.getElementById('style-button');
        this.styleButton.onclick = () => {
            this.disableStylizeButtons();
            this.gradualStyler(25, 0)
        };
        this.randomizeButton = document.getElementById('randomize');
        this.randomizeButton.onclick = () => {
            this.styleRatioSlider.value = getRndInteger(0, 100);
            this.contentImgSlider.value = getRndInteger(256, 400);
            this.styleImgSlider.value = getRndInteger(100, 400);
            this.styleRatioSlider.dispatchEvent(new Event("input"));
            this.contentImgSlider.dispatchEvent(new Event("input"));
            this.styleImgSlider.dispatchEvent(new Event("input"));
            if (getRndInteger(0, 1)) {
                this.styleImgSquare.click();
            }
        }

        // Initialize selectors
        this.contentSelect = document.getElementById('content-select');
        this.contentSelect.onchange = (net) => this.setImage(this.contentImg, net.target.value);
        this.contentSelect.onclick = () => this.contentSelect.value = '';
        this.styleSelect = document.getElementById('style-select');
        this.styleSelect.onchange = (net) => this.setImage(this.styleImg, net.target.value);
        this.styleSelect.onclick = () => this.styleSelect.value = '';
    }

    connectImageAndSizeSlider(img, slider, square) {
        slider.oninput = (net) => {
            img.height = net.target.value;
            if (img.style.width) {
                img.style.width = img.height + "px";
            }
        }
        if (square !== undefined) {
            square.onclick = (net) => {
                if (net.target.checked) {
                    img.style.width = img.height + "px";
                } else {
                    img.style.width = '';
                }
            }
        }
    }

    setImage(element, selectedValue) {
        if (selectedValue === 'file') {
            console.log('file selected');
            this.fileSelect.onchange = (net) => {
                const f = net.target.files[0];
                const fileReader = new FileReader();
                fileReader.onload = ((e) => {
                    element.src = e.target.result;
                });
                fileReader.readAsDataURL(f);
                this.fileSelect.value = '';
            }
            this.fileSelect.click();
        } else if (selectedValue === 'pic') {
            this.openModal(element);
        } else if (selectedValue === 'random') {
            const randomNumber = Math.floor(Math.random() * links.length);
            element.src = links[randomNumber];
        } else {
            element.src = 'images/' + selectedValue + '.jpg';
        }
    }

    enableStylizeButtons() {
        this.styleButton.disabled = false;
        this.modelSelectStyle.disabled = false;
        this.modelSelectTransformer.disabled = false;
        this.styleButton.textContent = 'Stylize';
    }

    disableStylizeButtons() {
        this.styleButton.disabled = true;
        this.modelSelectStyle.disabled = true;
        this.modelSelectTransformer.disabled = true;
    }

    async startStyling(canvas, styleRatio) {
        await tf.nextFrame();
        this.styleButton.textContent = 'Generating style representation';
        await tf.nextFrame();
        let bottleneck = await tf.tidy(() => {
            return this.styleNet.predict(tf.browser.fromPixels(this.styleImg).toFloat().div(tf.scalar(255)).expandDims());
        })
        if (styleRatio !== 1.0) {
            this.styleButton.textContent = 'Generating identity style representation';
            await tf.nextFrame();
            const identityBottleneck = await tf.tidy(() => {
                return this.styleNet.predict(tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims());
            })
            const styleBottleneck = bottleneck;
            bottleneck = await tf.tidy(() => {
                const styleBottleneckScaled = styleBottleneck.mul(tf.scalar(styleRatio));
                const identityBottleneckScaled = identityBottleneck.mul(tf.scalar(1.0 - styleRatio));
                return styleBottleneckScaled.addStrict(identityBottleneckScaled)
            })
            styleBottleneck.dispose();
            identityBottleneck.dispose();
        }
        this.styleButton.textContent = 'Stylizing image...';
        await tf.nextFrame();
        const stylized = await tf.tidy(() => {
            return this.transformNet.predict([tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims(), bottleneck]).squeeze();
        })
        await tf.browser.toPixels(stylized, canvas);
        bottleneck.dispose();
        stylized.dispose();
    }

    initalizeWebcamVariables() {
        this.camModal = $('#cam-modal');

        this.snapButton = document.getElementById('snap-button');
        this.webcamVideoElement = document.getElementById('webcam-video');

        navigator.getUserMedia = navigator.getUserMedia ||
            navigator.webkitGetUserMedia || navigator.mozGetUserMedia ||
            navigator.msGetUserMedia;

        this.camModal.on('hidden.bs.modal', () => {
            this.stream.getTracks()[0].stop();
        })

        this.camModal.on('shown.bs.modal', () => {
            navigator.getUserMedia({
                    video: true
                },
                (stream) => {
                    this.stream = stream;
                    this.webcamVideoElement.srcObject = stream;
                    this.webcamVideoElement.play();
                },
                (err) => {
                    console.error(err);
                }
            );
        })
    }

    openModal(element) {
        this.camModal.modal('show');
        this.snapButton.onclick = () => {
            const hiddenCanvas = document.getElementById('hidden-canvas');
            const hiddenContext = hiddenCanvas.getContext('2d');
            hiddenCanvas.width = this.webcamVideoElement.width;
            hiddenCanvas.height = this.webcamVideoElement.height;
            hiddenContext.drawImage(this.webcamVideoElement, 0, 0,
                hiddenCanvas.width, hiddenCanvas.height);
            const imageDataURL = hiddenCanvas.toDataURL('image/jpg');
            element.src = imageDataURL;
            this.camModal.modal('hide');
        };
    }

    async benchmark() {
        const x = tf.randomNormal([1, 256, 256, 3]);
        const bottleneck = tf.randomNormal([1, 1, 1, 100]);

        let styleNet = await this.loadInceptionStyleModel();
        let time = await this.benchmarkStyle(x, styleNet);
        styleNet.dispose();

        styleNet = await this.loadMobileNetStyleModel();
        time = await this.benchmarkStyle(x, styleNet);
        styleNet.dispose();

        let transformNet = await this.loadOriginalTransformerModel();
        time = await this.benchmarkTransform(
            x, bottleneck, transformNet);
        transformNet.dispose();

        transformNet = await this.loadSeparableTransformerModel();
        time = await this.benchmarkTransform(
            x, bottleneck, transformNet);
        transformNet.dispose();

        x.dispose();
        bottleneck.dispose();
    }

    async benchmarkStyle(x, styleNet) {
        const profile = await tf.profile(() => {
            tf.tidy(() => {
                const dummyOut = styleNet.predict(x);
                dummyOut.print();
            });
        });
        console.log(profile);
        const time = await tf.time(() => {
            tf.tidy(() => {
                for (let i = 0; i < 10; i++) {
                    const y = styleNet.predict(x);
                    y.print();
                }
            })
        });
        console.log(time);
    }

    async benchmarkTransform(x, bottleneck, transformNet) {
        const profile = await tf.profile(() => {
            tf.tidy(() => {
                const dummyOut = transformNet.predict([x, bottleneck]);
                dummyOut.print();
            });
        });
        console.log(profile);
        const time = await tf.time(() => {
            tf.tidy(() => {
                for (let i = 0; i < 10; i++) {
                    const y = transformNet.predict([x, bottleneck]);
                    y.print();
                }
            })
        });
        console.log(time);
    }
}

window.onload = function() {
    var dwn = document.getElementById('btndownload'),
        canvas = document.getElementById('stylized'),
        context = canvas.getContext('2d');
    dwn.onclick = function() {
        download(canvas, 'design.png');
    }
}

function download(canvas, filename) {
    var lnk = document.createElement('a'),
        e;
    lnk.download = filename;
    lnk.href = canvas.toDataURL("image/png;base64");
    if (document.createEvent) {
        e = document.createEvent("MouseEvents");
        e.initMouseEvent("click", true, true, window,
            0, 0, 0, 0, 0, false, false, false,
            false, 0, null);

        lnk.dispatchEvent(e);
    } else if (lnk.fireEvent) {
        lnk.fireEvent("onclick");
    }
}

function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

window.addEventListener('load', () => new Main());