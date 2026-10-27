
const express = require("express");
const cors = require("cors");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const natural = require("natural");
const { trainModel, evaluateModel } = require("./train");

const app = express();
const tokenizer = new natural.WordTokenizer();

const PORT = process.env.PORT || 3000;
const ROOT = __dirname;
const DATA_PATH = path.join(ROOT, "training-data.json");
const MODEL_DIR = path.join(ROOT, "model");
const CANDIDATE_DIR = path.join(ROOT, "model-candidate");
const BACKUP_DIR = path.join(ROOT, "model-backup");

const GEMINI_API =
    "https://api-rangestudio.vercel.app/api/gemini";

const MAX_MESSAGE = 1000;
const MAX_ANSWER = 4000;
const MAX_SAMPLES = 1000;
const MIN_SIMILARITY = 0.48;
const TRAIN_AFTER_NEW_SAMPLES = 10;
const MIN_VALIDATION_ACCURACY = 0.20;

let trainingData = [];
let model = null;
let vocabulary = [];
let labels = [];
let ready = false;
let newSamples = 0;
let teacherCalls = 0;
let training = false;
let saveQueue = Promise.resolve();

app.use(cors());
app.use(express.json({ limit: "16kb" }));

function normalize(text) {
    return String(text)
        .toLowerCase()
        .replace(/[^\p{L}\p{N}\s]/gu, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function tokenize(text) {
    return tokenizer.tokenize(normalize(text));
}

function encodeText(text) {
    const inputWords = new Set(tokenize(text));

    return vocabulary.map(word =>
        inputWords.has(word) ? 1 : 0
    );
}

function loadDataset() {
    if (!fs.existsSync(DATA_PATH)) {
        fs.writeFileSync(DATA_PATH, "[]", "utf8");
    }

    const data = JSON.parse(
        fs.readFileSync(DATA_PATH, "utf8")
    );

    if (!Array.isArray(data)) {
        throw new Error("training-data.json bukan array JSON.");
    }

    trainingData = data.filter(item =>
        item &&
        typeof item.input === "string" &&
        typeof item.output === "string" &&
        item.input.trim() &&
        item.output.trim()
    ).slice(-MAX_SAMPLES);
}

function similarity(a, b) {
    const wa = new Set(tokenize(a));
    const wb = new Set(tokenize(b));

    if (!wa.size || !wb.size) return 0;

    let intersection = 0;

    for (const word of wa) {
        if (wb.has(word)) intersection++;
    }

    return intersection / Math.sqrt(wa.size * wb.size);
}

function findLocalAnswer(message) {
    let best = null;
    let bestScore = 0;

    for (const item of trainingData) {
        const score = similarity(message, item.input);

        if (score > bestScore) {
            bestScore = score;
            best = item;
        }
    }

    if (!best || bestScore < MIN_SIMILARITY) {
        return null;
    }

    return {
        reply: best.output,
        confidence: Number(bestScore.toFixed(3))
    };
}

function predictLocal(message) {
    if (!model || !vocabulary.length || !labels.length) {
        return null;
    }

    const input = tfTensor(message);
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

        return {
            reply: labels[best],
            confidence: probabilities[best]
        };
    } finally {
        input.dispose();
        if (prediction) prediction.dispose();
    }
}

// TensorFlow diimpor secara terpisah untuk menjaga prediksi tetap sederhana.
function tfTensor(message) {
    return require("@tensorflow/tfjs-node").tensor2d(
        [encodeText(message)],
        [1, vocabulary.length]
    );
}

async function loadActiveModel() {
    const modelPath = path.join(MODEL_DIR, "model.json");
    const metadataPath = path.join(MODEL_DIR, "metadata.json");

    if (
        !fs.existsSync(modelPath) ||
        !fs.existsSync(metadataPath)
    ) {
        return false;
    }

    const metadata = JSON.parse(
        fs.readFileSync(metadataPath, "utf8")
    );

    const loaded = await require("@tensorflow/tfjs-node")
        .loadLayersModel(pathToFileURL(modelPath).href);

    const oldModel = model;

    model = loaded;
    vocabulary = metadata.vocabulary;
    labels = metadata.labels;
    ready = true;

    if (oldModel) oldModel.dispose();

    console.log("🧠 Model lokal aktif dimuat.");
    return true;
}

async function saveDataset() {
    saveQueue = saveQueue.then(async () => {
        const temp = DATA_PATH + ".tmp";

        await fs.promises.writeFile(
            temp,
            JSON.stringify(trainingData, null, 2),
            "utf8"
        );

        await fs.promises.rename(temp, DATA_PATH);
    });

    return saveQueue;
}

function parseTeacherAnswer(raw) {
    // Gemini mungkin mengembalikan JSON dalam teks biasa.
    const text = String(raw || "").trim();

    try {
        return JSON.parse(text);
    } catch {}

    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");

    if (start >= 0 && end > start) {
        try {
            return JSON.parse(text.slice(start, end + 1));
        } catch {}
    }

    return null;
}

async function askTeacher(message) {
    teacherCalls++;

    const prompt = `
Kamu adalah guru untuk AI chatbot berbahasa Indonesia.
Buat respons percakapan yang alami, ramah, tidak monoton,
dan sesuai pertanyaan pengguna.

Kembalikan JSON valid saja, tanpa markdown:
{
  "reply": "jawaban terbaik untuk pertanyaan pengguna",
  "examples": [
    {
      "input": "variasi pertanyaan dengan maksud sama",
      "output": "jawaban yang sesuai"
    }
  ]
}

Buat maksimal 3 variasi input yang berbeda tetapi memiliki
maksud sama dengan pertanyaan asli. Jangan membuat fakta
spesifik yang tidak diketahui. Output harus berupa teks.

Pertanyaan pengguna: ${JSON.stringify(message)}
`;

    const response = await axios.get(GEMINI_API, {
        params: { text: prompt },
        timeout: 15000,
        maxContentLength: 256 * 1024
    });

    const raw = response.data?.answer;

    if (typeof raw !== "string" || !raw.trim()) {
        throw new Error("Respons guru AI kosong.");
    }

    const parsed = parseTeacherAnswer(raw);

    // Jika endpoint tidak mengikuti format JSON, tetap gunakan
    // teks jawaban sebagai respons, tetapi jangan percaya contoh otomatis.
    if (
        !parsed ||
        typeof parsed.reply !== "string" ||
        !parsed.reply.trim()
    ) {
        return {
            reply: raw.trim().slice(0, MAX_ANSWER),
            examples: []
        };
    }

    const examples = Array.isArray(parsed.examples)
        ? parsed.examples.slice(0, 3).filter(item =>
            item &&
            typeof item.input === "string" &&
            typeof item.output === "string" &&
            item.input.trim() &&
            item.output.trim()
        )
        : [];

    return {
        reply: parsed.reply.trim().slice(0, MAX_ANSWER),
        examples
    };
}

async function learnExamples(message, teacherResult) {
    const examples = [
        { input: message, output: teacherResult.reply },
        ...teacherResult.examples
    ];

    const existing = new Set(
        trainingData.map(item => normalize(item.input))
    );

    let added = 0;

    for (const item of examples) {
        const input = item.input.trim();
        const output = item.output.trim();
        const key = normalize(input);

        if (!key || !output) continue;
        if (input.length > MAX_MESSAGE || output.length > MAX_ANSWER) {
            continue;
        }

        if (existing.has(key)) continue;

        // Hanya menerima contoh yang relevan dengan pertanyaan asli.
        if (similarity(message, input) < 0.25) continue;

        trainingData.push({ input, output });
        existing.add(key);
        added++;
    }

    if (!added) return 0;

    if (trainingData.length > MAX_SAMPLES) {
        trainingData = trainingData.slice(-MAX_SAMPLES);
    }

    try {
        await saveDataset();
        newSamples += added;
        console.log(`📚 ${added} contoh baru disimpan.`);

        return added;
    } catch (error) {
        console.error("Gagal menyimpan dataset:", error.message);
        throw error;
    }
}

async function retrainIfNeeded(force = false) {
    if (training) return;
    if (!force && newSamples < TRAIN_AFTER_NEW_SAMPLES) return;

    // Validasi kandidat membutuhkan jumlah data yang cukup.
    if (model && trainingData.length < 20) {
        console.log("⏳ Menunggu minimal 20 contoh sebelum retraining.");
        return;
    }

    training = true;

    try {
        const result = await trainModel();

        const candidateMetaPath = path.join(
            CANDIDATE_DIR,
            "metadata.json"
        );

        const candidateMeta = JSON.parse(
            fs.readFileSync(candidateMetaPath, "utf8")
        );

        const candidateAccuracy = result.validationAccuracy;

        if (model && candidateAccuracy === null) {
            console.log("⚠️ Kandidat belum memiliki validasi; model lama dipertahankan.");
            return;
        }

        if (
            model &&
            candidateAccuracy < MIN_VALIDATION_ACCURACY
        ) {
            console.log("⚠️ Akurasi kandidat terlalu rendah; model lama dipertahankan.");
            return;
        }

        const oldMetaPath = path.join(MODEL_DIR, "metadata.json");
        let oldAccuracy = null;

        if (fs.existsSync(oldMetaPath)) {
            const oldMeta = JSON.parse(
                fs.readFileSync(oldMetaPath, "utf8")
            );
            oldAccuracy = oldMeta.validationAccuracy;
        }

        // Tolak kandidat jika hasil validasinya jauh lebih buruk
        // daripada model aktif yang sebelumnya sudah tervalidasi.
        if (
            model &&
            oldAccuracy !== null &&
            candidateAccuracy < oldAccuracy - 0.05
        ) {
            console.log("⚠️ Kandidat lebih buruk daripada model lama.");
            return;
        }

        const tf = require("@tensorflow/tfjs-node");
        const oldModel = model;

        fs.rmSync(BACKUP_DIR, {
            recursive: true,
            force: true
        });

        if (fs.existsSync(MODEL_DIR)) {
            await fs.promises.rename(MODEL_DIR, BACKUP_DIR);
        }

        try {
            await fs.promises.rename(CANDIDATE_DIR, MODEL_DIR);
            await loadActiveModel();

            if (oldModel && oldModel !== model) {
                // loadActiveModel sudah membuang model sebelumnya.
            }

            fs.rmSync(BACKUP_DIR, {
                recursive: true,
                force: true
            });

            newSamples = 0;
            console.log("✅ Model baru berhasil diaktifkan.");
        } catch (error) {
            if (fs.existsSync(MODEL_DIR)) {
                fs.rmSync(MODEL_DIR, {
                    recursive: true,
                    force: true
                });
            }

            if (fs.existsSync(BACKUP_DIR)) {
                await fs.promises.rename(BACKUP_DIR, MODEL_DIR);
                await loadActiveModel();
            }

            throw error;
        }
    } catch (error) {
        console.error("Training gagal:", error.message);
    } finally {
        training = false;
    }
}

async function handleChat(req, res) {
    const message = req.body?.message ?? req.query?.text;

    if (typeof message !== "string" || !message.trim()) {
        return res.status(400).json({
            success: false,
            error: "Parameter message atau text diperlukan."
        });
    }

    if (message.length > MAX_MESSAGE) {
        return res.status(413).json({
            success: false,
            error: `Pesan maksimal ${MAX_MESSAGE} karakter.`
        });
    }

    const text = message.trim();

    // 1. Gunakan model neural network lokal.
    if (model) {
        const prediction = predictLocal(text);

        if (
            prediction &&
            prediction.confidence >= 0.65
        ) {
            return res.json({
                success: true,
                reply: prediction.reply,
                source: "local_model",
                confidence: Number(prediction.confidence.toFixed(3))
            });
        }
    }

    // 2. Gunakan pencarian memori sebagai cadangan.
    const memory = findLocalAnswer(text);

    if (memory) {
        return res.json({
            success: true,
            reply: memory.reply,
            source: "local_memory",
            confidence: memory.confidence
        });
    }

    // 3. Guru membantu pertanyaan yang belum dikenal.
    try {
        const teacherResult = await askTeacher(text);

        let added = 0;

        try {
            added = await learnExamples(text, teacherResult);
        } catch {}

        // Training berlangsung di background, bukan memblokir chat.
        if (newSamples >= TRAIN_AFTER_NEW_SAMPLES) {
            void retrainIfNeeded();
        }

        return res.json({
            success: true,
            reply: teacherResult.reply,
            source: "gemini_teacher",
            learnedExamples: added
        });
    } catch (error) {
        console.error("Guru AI gagal:", error.message);

        return res.status(502).json({
            success: false,
            error: "AI belum mengenal pertanyaan ini dan guru AI tidak tersedia."
        });
    }
}

app.get("/", (req, res) => {
    res.json({
        name: "My Own Learning AI",
        status: ready ? "online" : "starting"
    });
});

app.get("/api/status", (req, res) => {
    res.json({
        success: true,
        status: ready ? "online" : "starting",
        modelLoaded: Boolean(model),
        memorySize: trainingData.length,
        newSamplesSinceTraining: newSamples,
        training,
        teacherCalls
    });
});

app.post("/chat", handleChat);
app.get("/api/training-ai", handleChat);

async function start() {
    loadDataset();

    if (!(await loadActiveModel())) {
        console.log("🧠 Belum ada model aktif. Membuat model awal...");
        await retrainIfNeeded(true);

        if (!model) {
            throw new Error(
                "Model awal gagal dibuat. Periksa dataset dan log training."
            );
        }
    }

    ready = true;

    app.listen(PORT, "0.0.0.0", () => {
        console.log(`🤖 AI aktif pada port ${PORT}`);
    });
}

start().catch(error => {
    console.error("Startup gagal:", error);
    process.exit(1);
});
