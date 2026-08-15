import os
import sys
import json
import string

# ── Point NLTK at the bundled data directory ─────────────────────────────────
# On Vercel, __file__ is the api/index.py path; nltk_data lives one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")

import nltk
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# Fallback: download if somehow not bundled (shouldn't happen on Vercel)
_NLTK_PACKAGES = ["punkt_tab", "stopwords", "wordnet"]
for _pkg in _NLTK_PACKAGES:
    try:
        _kind = (
            "tokenizers" if _pkg.startswith("punkt") else
            "corpora"
        )
        nltk.data.find(f"{_kind}/{_pkg}")
    except LookupError:
        nltk.download(_pkg, download_dir=NLTK_DATA_DIR, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── NLP helpers ──────────────────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words and t.isalpha()]
    return " ".join(tokens)


# ── Load & index FAQs ────────────────────────────────────────────────────────
FAQ_PATH = os.path.join(BASE_DIR, "faqs.json")

with open(FAQ_PATH, "r", encoding="utf-8") as f:
    FAQS = json.load(f)

FAQ_QUESTIONS: list = [item["question"] for item in FAQS]
FAQ_ANSWERS: list   = [item["answer"]   for item in FAQS]
FAQ_CLEANED: list   = [preprocess(q) for q in FAQ_QUESTIONS]

_vectorizer   = TfidfVectorizer()
_tfidf_matrix = _vectorizer.fit_transform(FAQ_CLEANED)

CONFIDENCE_THRESHOLD = 0.28

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)
CORS(app)


@app.route("/")
def serve_index():
    public_dir = os.path.join(BASE_DIR, "public")
    return send_from_directory(public_dir, "index.html")


@app.route("/public/<path:filename>")
def serve_public(filename):
    public_dir = os.path.join(BASE_DIR, "public")
    return send_from_directory(public_dir, filename)


@app.route("/faqs", methods=["GET"])
def get_faqs():
    return jsonify({"questions": FAQ_QUESTIONS})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_message: str = data.get("message", "").strip()

    if not user_message:
        return jsonify({"answer": "Please type a message!", "matched_question": None, "confidence": 0.0})

    cleaned_msg = preprocess(user_message)

    if not cleaned_msg:
        return jsonify({
            "answer": "I couldn't understand that. Try asking about admissions, fees, scholarships, hostel, or exams!",
            "matched_question": None,
            "confidence": 0.0
        })

    user_vec    = _vectorizer.transform([cleaned_msg])
    similarities = cosine_similarity(user_vec, _tfidf_matrix).flatten()
    best_idx    = int(similarities.argmax())
    best_score  = float(similarities[best_idx])

    if best_score < CONFIDENCE_THRESHOLD:
        return jsonify({
            "answer": (
                "🤔 I'm not sure about that one! I'm best at answering questions about:\n\n"
                "• 📋 Admissions process & deadlines\n"
                "• 💰 Tuition fees & scholarships\n"
                "• 🏠 Hostel & accommodation\n"
                "• 📚 Programs & courses offered\n"
                "• 📅 Exam schedules & results\n"
                "• 🎓 Graduation & transcripts\n"
                "• 💼 Placements & internships\n"
                "• 🏥 Campus facilities & medical care\n\n"
                "Try asking something along those lines!"
            ),
            "matched_question": None,
            "confidence": round(best_score, 4)
        })

    return jsonify({
        "answer": FAQ_ANSWERS[best_idx],
        "matched_question": FAQ_QUESTIONS[best_idx],
        "confidence": round(best_score, 4)
    })


# ── Vercel / gunicorn entry point ─────────────────────────────────────────────
# Vercel looks for an `app` object in api/index.py
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
