import * as tf from '@tensorflow/tfjs';


export async function fgsm(model, image, epsilon){
    const adv_image = tf.tidy(()=>{
        const image_var = tf.variable(image);
        const f = () => model.execute();
        const g = tf.grad(f);
        const grads = g(image_var);
        return update(image, grads, epsilon);
    });
    return adv_image;
}

export function update(image, grads, epsilon){
    const adv_image = tf.tidy(()=>{
        return image.add(grads.sign().mul(tf.scalar(epsilon)));
    });
    return adv_image;
}