

export function showTestResults(batch, output,epochs) {
  const testExamples = batch.shape[0];
  const element=document.getElementById('new')
  element.innerHTML="<span style='display:block;'>After epochs = "+epochs+"</span>"
  for (let i = 0; i < testExamples; i++) {
    const image = batch.slice([i, 0], [1, batch.shape[1]]);

    const out =output.slice([i, 0], [1, batch.shape[1]]);

    const div = document.createElement('div');
    div.className = 'pred-container';

    const canvas = document.createElement('canvas');
    canvas.className = 'prediction-canvas';
    draw(image.flatten(), canvas);
    const canvas1 = document.createElement('canvas');
    canvas1.className = 'prediction-canvas';
    draw(out.flatten(), canvas1);


    div.appendChild(canvas1);
    div.appendChild(canvas);

    element.appendChild(div);
  }
}



export function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
