import { Iris } from 'machinelearn/datasets';

(async function() {
    const irisData = new Iris();
    const {
        data, // returns the iris data (X)
        targets, // list of target values (y)
        labels, // list of labels
        targetNames, // list of short target labels
        description // dataset description
    } = await irisData.load(); // loads the data internally
})();

import { SVR } from 'machinelearn/svm';

const svm = new SVR();
svm.loadASM().then((loadedSVM) => {
    loadedSVM.fit([
        [0, 0],
        [1, 1]
    ], [0, 1]);
    loadedSVM.predict([
        [1, 1]
    ]); // [0.9000000057898799]
});