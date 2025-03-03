const express = require("express");
const cors = require("cors");
const Chatbot = require("./nlp-bot");

const app = express();
app.use(express.json());
app.use(cors());

const bot = new Chatbot();

app.post("/chat", (req, res) => {
    const { message } = req.body;
    const reply = bot.getResponse(message);
    res.json({ reply });
});

app.listen(3000, () => console.log("🤖 Chatbot berjalan di http://localhost:3000"));
