import os
import json
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# NLTK data setup – runs at import time so it's ready before any request
# ---------------------------------------------------------------------------

NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]

def download_nltk_data():
    for pkg in NLTK_PACKAGES:
        try:
            # Check if already present
            nltk.data.find(
                f"tokenizers/{pkg}" if pkg.startswith("punkt") else
                f"corpora/{pkg}" if pkg in ("stopwords", "wordnet", "omw-1.4") else
                pkg
            )
        except LookupError:
            print(f"[NLTK] Downloading '{pkg}'…")
            nltk.download(pkg, quiet=True)

download_nltk_data()

# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Lowercase → remove punctuation → tokenize → remove stopwords → lemmatize."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words and t.isalpha()]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Load & index FAQs
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "faqs.json")

with open(FAQ_PATH, "r", encoding="utf-8") as f:
    FAQS = json.load(f)

FAQ_QUESTIONS: list[str] = [item["question"] for item in FAQS]
FAQ_ANSWERS: list[str] = [item["answer"] for item in FAQS]
FAQ_CLEANED: list[str] = [preprocess(q) for q in FAQ_QUESTIONS]

_vectorizer = TfidfVectorizer()
_tfidf_matrix = _vectorizer.fit_transform(FAQ_CLEANED)

CONFIDENCE_THRESHOLD = 0.28

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)
CORS(app)


@app.route("/")
def serve_index():
    return send_from_directory(os.path.join(BASE_DIR, "public"), "index.html")


@app.route("/public/<path:filename>")
def serve_public(filename):
    return send_from_directory(os.path.join(BASE_DIR, "public"), filename)


@app.route("/faqs", methods=["GET"])
def get_faqs():
    """Return all FAQ questions (used by frontend for suggestion chips)."""
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
            "answer": (
                "I couldn't understand that. Try asking about admissions, fees, "
                "scholarships, hostel, exams, or campus facilities!"
            ),
            "matched_question": None,
            "confidence": 0.0
        })

    user_vec = _vectorizer.transform([cleaned_msg])
    similarities = cosine_similarity(user_vec, _tfidf_matrix).flatten()

    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    if best_score < CONFIDENCE_THRESHOLD:
        fallback = (
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
        )
        return jsonify({"answer": fallback, "matched_question": None, "confidence": round(best_score, 4)})

    return jsonify({
        "answer": FAQ_ANSWERS[best_idx],
        "matched_question": FAQ_QUESTIONS[best_idx],
        "confidence": round(best_score, 4)
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
