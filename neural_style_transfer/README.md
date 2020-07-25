## Neural style transfer
This is an implementation of an arbitrary style transfer algorithm
running purely in the browser using TensorFlow.js. As with all neural 
style transfer algorithms, a neural network attempts to "draw" one 
picture, the Content (usually a photograph), in the style of another, 
the Style (usually a painting). 

## Features
* The browser allows the user to select custom image from their
desktop or even capture image using the webcam.
* It works proficiently for any given image and style.
* The user can select the extent of style transfer, the user wants
to apply on the original image.
* The user is displayed of how the image transitions progressively
as the training occurs.

## Running locally for development

To run it locally, you must install Yarn and run the following command at the repository's root to get all the dependencies.

```bash
yarn run prep
```

Verify that there is a .babelrc.json file present if it isn't in json format rename the file from .babelrc to .babelrc.json.

Then, you can run

```bash
yarn run start
```

You can then browse to `localhost:9966` to view the application.

## Contributors
1. Ankit Biswas
2. Shruti Gour
3. Akshit Mittal

## Credits
* Reiichiro Nakano for this [blog](https://magenta.tensorflow.org/blog/2018/12/20/style-transfer-js/) which I took reference from.
* Authors of the [arbitrary style transfer](https://arxiv.org/abs/1705.06830) paper.
* The Magenta repository for [arbitrary style transfer](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization).
* Authors of [the MobileNet-v2 paper](https://arxiv.org/abs/1801.04381).
* Authors of the paper describing [neural network knowledge distillation](https://arxiv.org/abs/1503.02531).

  


