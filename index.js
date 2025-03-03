const express = require("express");
const cors = require("cors");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const natural = require("natural");
const { trainModel } = require("./train");

const app = express();
app.use(express.json());
app.use(cors());

const tokenizer = new natural.WordTokenizer();
const trainingData = JSON.parse(fs.readFileSync("training-data.json", "utf-8"));
const modelPath = "file://./model/model.json";
let model;

const wordSet = new Set(trainingData.map(d => tokenizer.tokenize(d.input.toLowerCase())).flat());
const wordIndex = Array.from(wordSet).reduce((acc, word, i) => {
    acc[word] = i + 1;
    return acc;
}, {});

const encodeText = text => {
    const tokens = tokenizer.tokenize(text.toLowerCase());
    return Array.from({ length: 20 }, (_, i) => wordIndex[tokens[i]] || 0);
};

async function loadModel() {
    if (!fs.existsSync("./model/model.json")) {
        console.log("⚠️ Model tidak ditemukan! Melatih model...");
        await trainModel();
    }
    model = await tf.loadLayersModel(modelPath);
    console.log("✅ Model AI berhasil dimuat!");
}

app.post("/chat", async (req, res) => {
    const { message } = req.body;
    if (!model) return res.status(500).json({ reply: "Model AI belum siap!" });

    const inputTensor = tf.tensor2d([encodeText(message)]);
    const predictions = model.predict(inputTensor);
    const predictedIndex = predictions.argMax(1).dataSync()[0];

    res.json({ reply: trainingData[predictedIndex].output });
});

app.get("/api/training-ai", async (req, res) => {
    try {
        const userMessage = req.query.text;

        if (!userMessage) {
            return res.status(400).json({ error: "Parameter 'text' diperlukan." });
        }
      
        const response = await axios.post("http://localhost:3000/chat", { message: userMessage });

        res.json({ reply: response.data.reply });
    } catch (error) {
        res.status(500).json({ error: "Terjadi kesalahan saat memproses permintaan." });
    }
});

app.listen(3000, async () => {
    await loadModel();
    console.log("🤖 Chatbot ML berjalan di http://localhost:3000");
});
