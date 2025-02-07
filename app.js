// BM25 Implementation
class BM25 {
    constructor(k1 = 1.5, b = 0.75) {
        this.k1 = k1;
        this.b = b;
        this.documentFrequencies = new Map();
        this.documentLengths = [];
        this.averageDocumentLength = 0.0;
        this.totalDocuments = 0;
    }

    addDocument(terms) {
        const uniqueTerms = new Set(terms);
        this.documentLengths.push(terms.length);
        this.totalDocuments++;

        for (const term of uniqueTerms) {
            this.documentFrequencies.set(term, (this.documentFrequencies.get(term) || 0) + 1);
        }

        this.averageDocumentLength = this.documentLengths.reduce((a, b) => a + b, 0) / this.totalDocuments;
    }

    score(query, document) {
        if (this.totalDocuments === 0) return 0.0;

        const termFrequencies = {};
        document.forEach(term => termFrequencies[term] = (termFrequencies[term] || 0) + 1);

        let score = 0.0;

        for (const term of new Set(query)) {
            const tf = termFrequencies[term] || 0;
            const df = this.documentFrequencies.get(term) || 0;
            if (df === 0) continue;

            const idf = Math.log((this.totalDocuments - df + 0.5) / (df + 0.5) + 1.0);
            const numerator = tf * (this.k1 + 1);
            const denominator = tf + this.k1 * (1 - this.b + this.b * document.length / this.averageDocumentLength);

            score += idf * (numerator / denominator);
        }

        return score;
    }
}

// Tokenizer
class Tokenizer {
    constructor(vocabSize) {
        if (vocabSize <= 0) throw new Error("Vocab size must be positive");
        this.vocabSize = vocabSize;
    }

    tokenize(text) {
        return this.preprocessText(text).map(word => Math.floor(Math.abs(this.hashCode(word)) % this.vocabSize));
    }

    tokenizeToWords(text) {
        return this.preprocessText(text);
    }

    preprocessText(text) {
        return text.toLowerCase()
            .replace(/[^\w\s-]/g, " ")
            .split(/\s+/)
            .filter(Boolean);
    }

    hashCode(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = (hash << 5) - hash + str.charCodeAt(i);
            hash |= 0; // Convert to 32bit integer
        }
        return hash;
    }
}

// Embedding Layer
class EmbeddingLayer {
    constructor(vocabSize, embeddingDim) {
        if (vocabSize <= 0 || embeddingDim <= 0) throw new Error("Vocab size and embedding dimension must be positive");
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.embeddings = Array.from({ length: vocabSize }, () =>
            Array(embeddingDim).fill(0).map(() => (Math.random() * 2 - 1) * Math.sqrt(6 / (vocabSize + embeddingDim)))
        );
    }

    getEmbedding(tokenId) {
        return [...this.embeddings[Math.floor(Math.abs(tokenId) % this.vocabSize)]];
    }
}

// Enhanced Document Store
class EnhancedDocumentStore {
    constructor(documents, tokenizer, embeddingLayer) {
        this.documents = documents;
        this.tokenizer = tokenizer;
        this.embeddingLayer = embeddingLayer;
        this.bm25 = new BM25();

        this.documentTokens = documents.map(doc => {
            const tokens = this.tokenizer.tokenizeToWords(doc);
            this.bm25.addDocument(tokens);
            return tokens;
        });

        this.documentEmbeddings = documents.map(sentence => {
            const tokens = this.tokenizer.tokenize(sentence);
            return tokens.map(token => this.embeddingLayer.getEmbedding(token))
                .reduce((acc, vec) => acc.map((a, i) => a + vec[i]), Array(this.embeddingLayer.embeddingDim).fill(0));
        });
    }

    retrieve(query, topK = 1) {
        const queryTokens = this.tokenizer.tokenizeToWords(query);

        return this.documents.map((doc, idx) => {
            const bm25Score = this.bm25.score(queryTokens, this.documentTokens[idx]);
            const embeddingScore = this.calculateEmbeddingScore(query, this.documentEmbeddings[idx]);

            const combinedScore = 0.8 * bm25Score + 0.2 * embeddingScore;
            return { doc, score: combinedScore };
        }).sort((a, b) => b.score - a.score).slice(0, topK);
    }

    calculateEmbeddingScore(query, docEmbedding) {
        const queryTokens = this.tokenizer.tokenize(query);
        const queryEmbedding = queryTokens.map(token => this.embeddingLayer.getEmbedding(token))
            .reduce((acc, vec) => acc.map((a, i) => a + vec[i]), Array(this.embeddingLayer.embeddingDim).fill(0));

        return this.cosineSimilarity(queryEmbedding, docEmbedding);
    }

    cosineSimilarity(vec1, vec2) {
        const dotProduct = vec1.reduce((sum, v, i) => sum + v * vec2[i], 0);
        const magnitude1 = Math.sqrt(vec1.reduce((sum, v) => sum + v * v, 0));
        const magnitude2 = Math.sqrt(vec2.reduce((sum, v) => sum + v * v, 0));
        return dotProduct / (magnitude1 * magnitude2);
    }
}

// Main Functionality
document.addEventListener("DOMContentLoaded", async () => {
    const vocabSize = 1000;
    const embeddingDim = 128;
    const tokenizer = new Tokenizer(vocabSize);
    const embeddingLayer = new EmbeddingLayer(vocabSize, embeddingDim);

    // Load dataset from data.txt
    const response = await fetch('data.txt');
    const dataset = (await response.text()).trim().split('\n');

    const documentStore = new EnhancedDocumentStore(dataset, tokenizer, embeddingLayer);

    const chatContainer = document.getElementById("chatContainer");
    const queryInput = document.getElementById("queryInput");
    const sendButton = document.getElementById("sendButton");

    sendButton.addEventListener("click", () => {
        const userInput = queryInput.value.trim();
        if (!userInput) return;

        // Add user message to chat
        const userMessageDiv = document.createElement("div");
        userMessageDiv.className = "message user-message";
        userMessageDiv.innerHTML = `<strong>You:</strong> ${userInput}`;
        chatContainer.appendChild(userMessageDiv);

        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Get bot response
        const retrievedDocuments = documentStore.retrieve(userInput, 1);
        const botResponse = retrievedDocuments.length > 0 ? retrievedDocuments[0].doc : "I don't have an answer for that.";

        // Add bot message to chat
        const botMessageDiv = document.createElement("div");
        botMessageDiv.className = "message bot-message";
        botMessageDiv.innerHTML = `<strong>Bot:</strong> ${botResponse}`;
        chatContainer.appendChild(botMessageDiv);

        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Clear input field
        queryInput.value = "";
    });
});
