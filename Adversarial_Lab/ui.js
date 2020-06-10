import * as attacks from './attacks';
import * as tf from '@tensorflow/tfjs';

const mobilenet = require('@tensorflow-models/mobilenet');


export async function Generate(){
  const model = await mobilenet.load();
  const epsilon = parseInt(document.getElementById("upload-file").eps.value);
  const raw_image = document.getElementById("upload-file").files[0];
  console.log(raw_image)
  const image = tf.browser.fromPixels(raw_image)
  const adv_tensor = attacks.fgsm(model, image, epsilon);
  return adv_tensor;
}