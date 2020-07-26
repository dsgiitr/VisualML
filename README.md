
<p align="center">
  <img width="60%" src="visualml.png" />
</p>

-----------------------------------------------------------------------------------------------------------
# Visual Machine Learning

Visual Machine Learning contains a set of Machine Learning and Deep Learning interactive visualisation demos for developing intuition. These demos are developed using [TensorFlow.js](https://js.tensorflow.org) and can be executed directly in your browser. This project is an extension of ML examples from [tfjs-examples](https://github.com/tensorflow/tfjs-examples). We implement new demos, as well as, add additional features into the ones that already existed in TFJS. 

Some examples may require web-gl enabled browsers and viewers may experience latency during executing the demos based on the device. 

# Overview of Demos

<table>
  <tr>
    <th>Example name</th>
    <th>Demo link</th>
    <th>Input data type</th>
    <th>Task type</th>
    <th>Model type</th>
    <th>Training</th>
    <th>Inference</th>
  <tr>
    <td><a href="./ANN">ANN</a></td>
    <td><a href="https://dsgiitr.github.io/ann-demo">🔗</a></td>
    <td>Iris Dataset</td>
    <td>View NN architecture, View Confusion Matrix</td>
    <td>Multilayer perceptron</td>
    <td>Browser</td>
    <td>Browser</td>
  </tr>
  <tr>
    <td><a href="./Autoencoder">Autoencoder</a></td>
    <td><a href="https://dsgiitr.github.io/autoencoder-demo">🔗</a></td>
    <td>MNIST dataset</td>
    <td>Visualising Latent Space</td>
    <td>Autoencoder</td>
    <td>Browser</td>
    <td>Browser</td>
  </tr>
  <tr>
    <td><a href="./Logistic Regression">Logistic Regression</a></td>
    <td><a href="https://dsgiitr.github.io/logistic-regression-demo">🔗</a></td>
    <td>Various 2D data</td>
    <td>Visualising Decision Boundary</td>
    <td>Logistic Regression</td>
    <td>Browser</td>
    <td>Browser</td>
  </tr>
  <tr>
    <td><a href="./MNIST-CNN">MNIST-CNN</a></td>
    <td><a href="https://dsgiitr.github.io/mnist-cnn-demo">🔗</a></td>
    <td>MNIST</td>
    <td>Visualising Activations</td>
    <td>CNN</td>
    <td>Browser</td>
    <td>Browser</td>
  </tr>
  <tr>
    <td><a href="./PCA">PCA</a></td>
    <td><a href="https://dsgiitr.github.io/pca-demo">🔗</a></td>
    <td>Various</td>
    <td>Visualising Principal Components & projected dimensions</td>
    <td>PCA</td>
    <td>Browser</td>
    <td>Browser</td>
  </tr>
  <tr>
    <td><a href="./SVM">SVM</a></td>
    <td><a href="https://dsgiitr.github.io/svm-demo">🔗</a></td>
    <td>2D Dataset</td>
    <td>Visualising Support Vectors and Kernels</td>
    <td>SMO</td>
    <td>Browser</td>
    <td>Browser</td>
  </tr>
  <tr>
    <td><a href="./neural_style_transfer">Neural Style Transfer</a></td>
    <td><a href="https://dsgiitr.github.io/neural-style-transfer-tfjs">🔗</a></td>
    <td>Image Data</td>
    <td>Visualising Style Transfer using MobileNet</td>
    <td>Style Transfer</td>
    <td>Browser</td>
    <td>Browser</td>
  </tr>
  <tr>
    <td><a href="./vanishing_gradients">Vanishing Gradients</a></td>
    <td><a href="https://dsgiitr.github.io/vanishing-gradients-demo">🔗</a></td>
    <td>Iris Dataset</td>
    <td>Developing Intuition how Relu Fixes Vanishing Gradients</td>
    <td>Neural Networks</td>
    <td>Browser</td>
    <td>Browser</td>
  </tr>
</table>

# Dependencies

All the examples require the following dependencies to be installed.

 - Node.js version 8.9 or higher
 - [NPM CLI](https://docs.npmjs.com/cli/npm) OR [Yarn](https://yarnpkg.com/en/)

## How to build?
`cd` into the directory

If you are using `yarn`:

```sh
cd MNIST-CNN
yarn
yarn watch
```

If you are using `npm`:
```sh
cd MNIST-CNN
npm install
npm run watch
```

### Details

The convention is that each example contains two scripts:

- `yarn watch` or `npm run watch`: This starts and generates a local development HTML server tracking filesystem for changes, supporting hot-reloading.

- `yarn build` or `npm run build`: generates a `dist/` folder which contains the build artifacts and can be used for deployment.

## Contributing

If you want to contribute a demo, please reach out to us on
[Github issues](https://github.com/dsgiitr/VisualML/issues)
before sending us a pull request as we are trying to keep this set of examples
small and highly curated.

## Acknowledgements

* [tfjs-examples](https://github.com/tensorflow/tfjs-examples)
