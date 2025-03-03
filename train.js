const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const natural = require("natural");

const tokenizer = new natural.WordTokenizer();
const trainingData = JSON.parse(fs.readFileSync("training-data.json", "utf-8"));

const inputTexts = trainingData.map(d => tokenizer.tokenize(d.input.toLowerCase()).join(" "));
const outputTexts = trainingData.map(d => d.output);

const maxWords = 20;
const wordSet = new Set(inputTexts.join(" ").split(" "));
const wordIndex = Array.from(wordSet).reduce((acc, word, i) => {
    acc[word] = i + 1;
    return acc;
}, {});

const encodeText = text => {
    const tokens = tokenizer.tokenize(text.toLowerCase());
    return Array.from({ length: maxWords }, (_, i) => wordIndex[tokens[i]] || 0);
};

const xTrain = tf.tensor2d(inputTexts.map(encodeText));
const yTrain = tf.tensor2d(outputTexts.map((_, i) => [i]));

const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [maxWords] }));
model.add(tf.layers.dense({ units: trainingData.length, activation: "softmax" }));
model.compile({ optimizer: "adam", loss: "sparseCategoricalCrossentropy", metrics: ["accuracy"] });

async function trainModel() {
    console.log("🧠 Melatih model AI...");
    await model.fit(xTrain, yTrain, { epochs: 100 });
    await model.save(`file://./model`);
    console.log("✅ Model berhasil disimpan!");
}

module.exports = { trainModel };
