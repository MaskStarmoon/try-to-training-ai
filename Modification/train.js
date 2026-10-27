
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const natural = require("natural");

const tokenizer = new natural.WordTokenizer();

const ROOT = __dirname;
const DATA_PATH = path.join(ROOT, "training-data.json");
const MODEL_DIR = path.join(ROOT, "model");
const CANDIDATE_DIR = path.join(ROOT, "model-candidate");

const MAX_VOCAB = 500;
const MAX_SAMPLES = 1000;
const HIDDEN_UNITS = 16;
const EPOCHS = 25;

function tokenize(text) {
    return tokenizer.tokenize(
        String(text).toLowerCase()
    );
}

function buildVocabulary(data) {
    const frequency = new Map();

    for (const item of data) {
        for (const word of new Set(tokenize(item.input))) {
            frequency.set(word, (frequency.get(word) || 0) + 1);
        }
    }

    return [...frequency.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, MAX_VOCAB)
        .map(([word]) => word);
}

function encodeText(text, vocabulary) {
    const words = new Set(tokenize(text));

    return vocabulary.map(word => words.has(word) ? 1 : 0);
}

function shuffle(data) {
    const result = [...data];

    for (let i = result.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [result[i], result[j]] = [result[j], result[i]];
    }

    return result;
}

async function evaluateModel(model, data, vocabulary, labels) {
    if (!data.length) return null;

    let correct = 0;

    for (const item of data) {
        const target = labels.indexOf(item.output);
        if (target < 0) continue;

        const input = tf.tensor2d(
            [encodeText(item.input, vocabulary)]
        );

        let prediction;

        try {
            prediction = model.predict(input);
            const probabilities = prediction.dataSync();

            let best = 0;

            for (let i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > probabilities[best]) {
                    best = i;
                }
            }

            if (best === target) correct++;
        } finally {
            input.dispose();
            if (prediction) prediction.dispose();
        }
    }

    return correct / data.length;
}

async function trainModel() {
    const allData = JSON.parse(
        fs.readFileSync(DATA_PATH, "utf8")
    );

    if (!Array.isArray(allData)) {
        throw new Error("Dataset harus berupa array JSON.");
    }

    const data = allData
        .filter(item =>
            item &&
            typeof item.input === "string" &&
            typeof item.output === "string" &&
            item.input.trim() &&
            item.output.trim()
        )
        .slice(-MAX_SAMPLES);

    if (data.length < 2) {
        throw new Error(
            "Minimal 2 contoh valid diperlukan untuk training."
        );
    }

    const shuffled = shuffle(data);
    const labels = [...new Set(data.map(item => item.output))];

    if (labels.length < 2) {
        throw new Error(
            "Dataset memerlukan minimal 2 jawaban unik."
        );
    }

    // Dataset kecil memakai semua contoh untuk training.
    // Dataset besar menyisihkan 20% untuk validasi.
    const canValidate = data.length >= 20;
    const splitIndex = canValidate
        ? Math.floor(shuffled.length * 0.8)
        : shuffled.length;

    const trainData = shuffled.slice(0, splitIndex);
    const validationData = shuffled.slice(splitIndex);

    const vocabulary = buildVocabulary(trainData);

    if (!vocabulary.length) {
        throw new Error("Kosakata training kosong.");
    }

    const labelIndex = new Map(
        labels.map((label, index) => [label, index])
    );

    const xs = tf.tensor2d(
        trainData.map(item => encodeText(item.input, vocabulary))
    );

    const ys = tf.tensor1d(
        trainData.map(item => labelIndex.get(item.output)),
        "int32"
    );

    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [vocabulary.length],
        units: HIDDEN_UNITS,
        activation: "relu"
    }));

    model.add(tf.layers.dense({
        units: labels.length,
        activation: "softmax"
    }));

    model.compile({
        optimizer: tf.train.adam(0.003),
        loss: "sparseCategoricalCrossentropy",
        metrics: ["accuracy"]
    });

    try {
        console.log(
            `🧠 Training ${trainData.length} contoh, ` +
            `${labels.length} kelas jawaban...`
        );

        await model.fit(xs, ys, {
            epochs: EPOCHS,
            batchSize: Math.min(8, trainData.length),
            shuffle: true,
            verbose: 0,
            callbacks: {
                onEpochEnd(epoch, logs) {
                    if ((epoch + 1) % 5 === 0) {
                        console.log(
                            `Epoch ${epoch + 1}/${EPOCHS} | ` +
                            `loss=${logs.loss.toFixed(4)}`
                        );
                    }
                }
            }
        });

        const validationAccuracy = canValidate
            ? await evaluateModel(
                model,
                validationData,
                vocabulary,
                labels
            )
            : null;

        fs.rmSync(CANDIDATE_DIR, {
            recursive: true,
            force: true
        });

        fs.mkdirSync(CANDIDATE_DIR, { recursive: true });

        await model.save(`file://${CANDIDATE_DIR}`);

        fs.writeFileSync(
            path.join(CANDIDATE_DIR, "metadata.json"),
            JSON.stringify({
                vocabulary,
                labels,
                sampleCount: data.length,
                validationAccuracy,
                trainedAt: new Date().toISOString()
            }, null, 2)
        );

        console.log(
            "✅ Model kandidat tersimpan.",
            "Validation accuracy:",
            validationAccuracy
        );

        return {
            sampleCount: data.length,
            validationAccuracy
        };
    } finally {
        xs.dispose();
        ys.dispose();
        model.dispose();
    }
}

module.exports = {
    trainModel,
    evaluateModel
};
