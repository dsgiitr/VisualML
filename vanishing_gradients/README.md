# Vanishing Gradients.
This demo helps to visualize the problem of **vanishing gradients** (particularly w.r.t to use of different 
activation functions) using **TensorFlow.js** and **Plotly.js**.

## Features

* Analyse various activation functions by plotting them.
* Train a fully connected neural network using custom parameters.
* Visualize the vanishing gradients in the nn-architecture printed.
* Opacity of links between different nodes represent magnitude of positive and negative gradients separately.
* Visualise accuracy and Loss curves.

## Dependencies
These dependencies are required to be installed.

* Node.js version 8.9 or higher
* [Yarn](https://classic.yarnpkg.com/en/docs/install/#debian-stable)

### How to build an example
```cd``` into the directory

If you are using ```yarn```:
```
cd vanishing_gradients
yarn
yarn watch
```


#### Details
The package contains two scripts:

```yarn watch```: starts a local development HTTP server which watches the filesystem for changes so you can edit the code (JS or HTML) and see changes when you refresh the page immediately.

```yarn build```: generates a dist/ folder which contains the build artifacts and can be used for deployment.

### Contributors

* [Sahil Goyal](https://github.com/sahilg06)

### References

* [Activation functions](https://github.com/Polarisation/tfjs-activation-functions)