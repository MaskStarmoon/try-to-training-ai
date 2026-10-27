
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const natural = require("natural");

const tokenizer = new natural.WordTokenizer();

const DATA_PATH = path.join(__dirname, "training-data.json");
const MODEL_DIR = path.join(__dirname, "model");
const METADATA_PATH = path.join(MODEL_DIR, "metadata.json");

const MAX_VOCAB = 1000;
const HIDDEN_UNITS = 32;
const EPOCHS = 40;

function tokenize(text) {
    return tokenizer.tokenize(
        String(text).toLowerCase()
    );
}

function buildVocabulary(data) {
    const counts = new Map();

    for (const item of data) {
        for (const word of tokenize(item.input)) {
            counts.set(word, (counts.get(word) || 0) + 1);
        }
    }

    return [...counts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, MAX_VOCAB)
        .map(([word]) => word);
}

function encodeText(text, vocabulary) {
    const wordSet = new Set(tokenize(text));

    return vocabulary.map(word =>
        wordSet.has(word) ? 1 : 0
    );
}

async function trainModel() {
    const data = JSON.parse(
        fs.readFileSync(DATA_PATH, "utf8")
    );

    if (!Array.isArray(data) || data.length < 2) {
        throw new Error("Dataset minimal harus berisi 2 data.");
    }

    for (const item of data) {
        if (
            typeof item.input !== "string" ||
            !item.input.trim() ||
            typeof item.output !== "string" ||
            !item.output.trim()
        ) {
            throw new Error(
                "Setiap data harus memiliki input dan output berupa teks."
            );
        }
    }

    const vocabulary = buildVocabulary(data);
    const labels = [...new Set(data.map(item => item.output))];

    if (vocabulary.length === 0 || labels.length < 2) {
        throw new Error(
            "Dataset harus memiliki kosakata dan minimal 2 jawaban unik."
        );
    }

    const labelIndex = new Map(
        labels.map((label, index) => [label, index])
    );

    const xs = tf.tensor2d(
        data.map(item => encodeText(item.input, vocabulary)),
        [data.length, vocabulary.length]
    );

    const ys = tf.tensor1d(
        data.map(item => labelIndex.get(item.output)),
        "int32"
    );

    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [vocabulary.length],
        units: HIDDEN_UNITS,
        activation: "relu"
    }));

    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.dense({
        units: labels.length,
        activation: "softmax"
    }));

    model.compile({
        optimizer: tf.train.adam(0.005),
        loss: "sparseCategoricalCrossentropy",
        metrics: ["accuracy"]
    });

    try {
        console.log("🧠 Memulai pelatihan AI...");

        await model.fit(xs, ys, {
            epochs: EPOCHS,
            batchSize: Math.min(16, data.length),
            shuffle: true,
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if ((epoch + 1) % 10 === 0) {
                        console.log(
                            `Epoch ${epoch + 1}/${EPOCHS} | ` +
                            `loss: ${logs.loss.toFixed(4)} | ` +
                            `accuracy: ${logs.acc?.toFixed(4) ?? "N/A"}`
                        );
                    }
                }
            }
        });

        fs.mkdirSync(MODEL_DIR, { recursive: true });

        await model.save(`file://${MODEL_DIR}`);

        fs.writeFileSync(
            METADATA_PATH,
            JSON.stringify({ vocabulary, labels }, null, 2)
        );

        console.log("✅ Model dan metadata berhasil disimpan.");
    } finally {
        xs.dispose();
        ys.dispose();
        model.dispose();
    }
}

module.exports = { trainModel };
