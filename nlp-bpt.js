const natural = require("natural");

class Chatbot {
    constructor() {
        this.classifier = new natural.BayesClassifier();

        // Tambahkan data pelatihan
        this.train();
    }

    train() {
        this.classifier.addDocument("halo", "Halo! Apa kabar?");
        this.classifier.addDocument("hi", "Hai! Ada yang bisa saya bantu?");
        this.classifier.addDocument("siapa kamu", "Saya adalah chatbot AI buatanmu.");
        this.classifier.addDocument("apa kabar", "Saya baik, bagaimana denganmu?");
        this.classifier.addDocument("bagaimana cara kerja kamu", "Saya menggunakan Natural Language Processing untuk memahami pesanmu.");
        this.classifier.addDocument("terima kasih", "Sama-sama! 😊");
        this.classifier.addDocument("selamat tinggal", "Sampai jumpa lagi!");
        
        // Latih model
        this.classifier.train();
    }

    getResponse(input) {
        return this.classifier.classify(input) || "Maaf, saya belum mengerti.";
    }
}

module.exports = Chatbot;
