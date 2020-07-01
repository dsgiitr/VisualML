
import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';
tf.ENV.set('WEBGL_PACK', false);  // This needs to be done otherwise things run very slow v1.0.4
import links from './links';

class Main {
  constructor() {
    if (window.mobilecheck()) {
      document.getElementById('mobile-warning').hidden = false;
    }

    this.fileSelect = document.getElementById('file-select');

    // Initialize model selection
    this.modelSelectStyle = document.getElementById('model-select-style');
    this.modelSelectStyle.onchange = (evt) => {
      if (evt.target.value === 'mobilenet') {
        this.disableStylizeButtons();
        this.loadMobileNetStyleModel().then(model => {
          this.styleNet = model;
        }).finally(() => this.enableStylizeButtons());
      } else if (evt.target.value === 'inception') {
        this.disableStylizeButtons();
        this.loadInceptionStyleModel().then(model => {
          this.styleNet = model;
        }).finally(() => this.enableStylizeButtons());
      }
    }

    this.modelSelectTransformer = document.getElementById('model-select-transformer');
    this.modelSelectTransformer.onchange = (evt) => {
      if (evt.target.value === 'original') {
        this.disableStylizeButtons();
        this.loadOriginalTransformerModel().then(model => {
          this.transformNet = model;
        }).finally(() => this.enableStylizeButtons());
      } else if (evt.target.value === 'separable') {
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
      navigator.getUserMedia(
        {
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

  async gradualStyler(n = 10, i = 1) {
    this.startStyling(this.stylized, this.styleRatio * (i / n)).finally(() => {
      if (i < n) {
        var j = i + 1;
        var progressBar = document.getElementById("Bar");
        document.getElementById("counter").innerHTML = "Epochs: " + j*40;
        var width = 100 * (j / n);
        if (width >= 102) {
          progressBar.style.width = 0;
        } else {
          progressBar.style.width = width + "%";
        }
        this.gradualStyler(n, i + 1);
      }
      else {
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
    this.styleRatioSlider.oninput = (evt) => {
      this.styleRatio = evt.target.value/100.;
    }

    // Initialize buttons
    this.styleButton = document.getElementById('style-button');
    this.styleButton.onclick = () => {
      this.disableStylizeButtons();
      this.gradualStyler(25,0)
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
    this.contentSelect.onchange = (evt) => this.setImage(this.contentImg, evt.target.value);
    this.contentSelect.onclick = () => this.contentSelect.value = '';
    this.styleSelect = document.getElementById('style-select');
    this.styleSelect.onchange = (evt) => this.setImage(this.styleImg, evt.target.value);
    this.styleSelect.onclick = () => this.styleSelect.value = '';
  }

  connectImageAndSizeSlider(img, slider, square) {
    slider.oninput = (evt) => {
      img.height = evt.target.value;
      if (img.style.width) {
        // If this branch is triggered, then that means the image was forced to a square using
        // a fixed pixel value.
        img.style.width = img.height+"px";  // Fix width back to a square
      }
    }
    if (square !== undefined) {
      square.onclick = (evt) => {
        if (evt.target.checked) {
          img.style.width = img.height+"px";
        } else {
          img.style.width = '';
        }
      }
    }
  }

  // Helper function for setting an image
  setImage(element, selectedValue) {
    if (selectedValue === 'file') {
      console.log('file selected');
      this.fileSelect.onchange = (evt) => {
        const f = evt.target.files[0];
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
      const randomNumber = Math.floor(Math.random()*links.length);
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
    this.styleButton.textContent = 'Generating 100D style representation';
    await tf.nextFrame();
    let bottleneck = await tf.tidy(() => {
      return this.styleNet.predict(tf.browser.fromPixels(this.styleImg).toFloat().div(tf.scalar(255)).expandDims());
    })
    if (styleRatio !== 1.0) {
      this.styleButton.textContent = 'Generating 100D identity style representation';
      await tf.nextFrame();
      const identityBottleneck = await tf.tidy(() => {
        return this.styleNet.predict(tf.browser.fromPixels(this.contentImg).toFloat().div(tf.scalar(255)).expandDims());
      })
      const styleBottleneck = bottleneck;
      bottleneck = await tf.tidy(() => {
        const styleBottleneckScaled = styleBottleneck.mul(tf.scalar(styleRatio));
        const identityBottleneckScaled = identityBottleneck.mul(tf.scalar(1.0-styleRatio));
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
    bottleneck.dispose();  // Might wanna keep this around
    stylized.dispose();
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

window.onload = function(){
  var dwn = document.getElementById('btndownload'),
      canvas = document.getElementById('stylized'),
      context = canvas.getContext('2d');
      dwn.onclick = function(){
        download(canvas, 'design.png');
      }    
}

function download(canvas, filename) {
  var lnk = document.createElement('a'), e;
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
  return Math.floor(Math.random() * (max - min + 1) ) + min;
}

window.mobilecheck = function() {
  var check = false;
  (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
  return check;
};
window.addEventListener('load', () => new Main());
