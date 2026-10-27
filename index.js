
const express = require("express");
const cors = require("cors");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const natural = require("natural");
const { trainModel } = require("./train");

const app = express();
const tokenizer = new natural.WordTokenizer();

const PORT = process.env.PORT || 3000;
const DATA_PATH = path.join(__dirname, "training-data.json");
const MODEL_DIR = path.join(__dirname, "model");
const MODEL_PATH = path.join(MODEL_DIR, "model.json");
const METADATA_PATH = path.join(MODEL_DIR, "metadata.json");

const MAX_MESSAGE_LENGTH = 1000;
const MIN_CONFIDENCE = 0.25;

app.use(cors());
app.use(express.json({ limit: "16kb" }));

let model = null;
let vocabulary = [];
let labels = [];
let trainingData = [];
let ready = false;

function tokenize(text) {
    return tokenizer.tokenize(text.toLowerCase());
}

function encodeText(text) {
    const words = new Set(tokenize(text));

    return vocabulary.map(word =>
        words.has(word) ? 1 : 0
    );
}

async function loadModel() {
    trainingData = JSON.parse(
        fs.readFileSync(DATA_PATH, "utf8")
    );

    const modelExists = fs.existsSync(MODEL_PATH);
    const metadataExists = fs.existsSync(METADATA_PATH);

    if (!modelExists || !metadataExists) {
        console.log("⚠️ Model belum tersedia. Melatih AI...");
        await trainModel();
    }

    const metadata = JSON.parse(
        fs.readFileSync(METADATA_PATH, "utf8")
    );

    vocabulary = metadata.vocabulary;
    labels = metadata.labels;

    model = await tf.loadLayersModel(
        pathToFileURL(MODEL_PATH).href
    );

    ready = true;
    console.log("✅ AI siap digunakan.");
}

function generateReply(message) {
    const input = tf.tensor2d(
        [encodeText(message)],
        [1, vocabulary.length]
    );

    let output;

    try {
        output = model.predict(input);

        const probabilities = output.dataSync();
        let bestIndex = 0;

        for (let i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[bestIndex]) {
                bestIndex = i;
            }
        }

        const confidence = probabilities[bestIndex];

        if (confidence < MIN_CONFIDENCE) {
            return {
                reply: "Maaf, aku belum memahami pertanyaan itu. Bisa jelaskan dengan cara lain?",
                confidence: Number(confidence.toFixed(4))
            };
        }

        return {
            reply: labels[bestIndex],
            confidence: Number(confidence.toFixed(4))
        };
    } finally {
        input.dispose();

        if (output) {
            output.dispose();
        }
    }
}

function handleChat(req, res) {
    if (!ready || !model) {
        return res.status(503).json({
            error: "AI sedang disiapkan. Coba lagi sebentar."
        });
    }

    const message = req.body?.message;

    if (typeof message !== "string" || !message.trim()) {
        return res.status(400).json({
            error: "Pesan harus berupa teks dan tidak boleh kosong."
        });
    }

    if (message.length > MAX_MESSAGE_LENGTH) {
        return res.status(413).json({
            error: `Pesan maksimal ${MAX_MESSAGE_LENGTH} karakter.`
        });
    }

    try {
        const result = generateReply(message.trim());

        return res.json(result);
    } catch (error) {
        console.error("Kesalahan prediksi:", error.message);

        return res.status(500).json({
            error: "Gagal memproses pesan."
        });
    }
}

app.get("/", (req, res) => {
    res.json({
        name: "Lightweight ML Chatbot",
        status: ready ? "online" : "starting",
        endpoints: ["/chat", "/api/training-ai"]
    });
});

app.get("/api/status", (req, res) => {
    res.json({
        success: true,
        status: ready ? "online" : "starting",
        modelLoaded: Boolean(model),
        vocabularySize: vocabulary.length,
        answerClasses: labels.length
    });
});

app.post("/chat", handleChat);

app.get("/api/training-ai", (req, res) => {
    req.body = { message: req.query.text };
    handleChat(req, res);
});

async function startServer() {
    try {
        await loadModel();

        app.listen(PORT, "0.0.0.0", () => {
            console.log(`🤖 AI berjalan pada port ${PORT}`);
        });
    } catch (error) {
        console.error("❌ Gagal menyiapkan AI:", error);
        process.exit(1);
    }
}
startServer();
