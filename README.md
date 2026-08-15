# 🤖 Saood's Bot — College FAQ Chatbot

> A dark-themed, AI-powered college FAQ chatbot built with **Flask + NLTK + TF-IDF**, deployed on **Render**.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.3-black?style=flat-square&logo=flask)
![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange?style=flat-square&logo=scikit-learn)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)

---

## ✨ Features

- 💬 **Chat-style UI** — user bubbles (right) and bot bubbles (left)
- 🧠 **NLP pipeline** — NLTK tokenization, stopword removal, lemmatization → TF-IDF → cosine similarity
- 📊 **Confidence badge** — every bot reply shows a `% match` score
- 🔖 **Quick-reply chips** — 4 random sample questions auto-loaded from `/faqs`
- ⌨️ **Typing indicator** — animated 3-dot loader while waiting for a response
- 🛡️ **Fallback response** — politely redirects when confidence < 28%
- 🌈 **Aurora background** — animated magenta/violet/cyan orbs (respects `prefers-reduced-motion`)
- 🪟 **Glassmorphism card** — backdrop-blur chat window
- 📱 **Responsive** — works on mobile and desktop

---

## 📚 Knowledge Base

**32 FAQ pairs** covering:
| Topic | Topic |
|-------|-------|
| Admissions process & deadlines | Tuition fees |
| Required documents | Scholarships & financial aid |
| Eligibility criteria | Hostel / dormitory |
| Programs & courses | Exam schedules & results |
| Attendance policy | Academic advising |
| Leave of absence | Transcripts & certificates |
| Placements & internships | Campus facilities |
| Library hours | Student ID card |
| Medical & healthcare | Withdrawal process |
| Course registration | Grading system |
| Clubs & activities | Parking |

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11 + Flask |
| NLP | NLTK (punkt, stopwords, WordNetLemmatizer) |
| Matching | scikit-learn TF-IDF + cosine similarity |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| Fonts | Inter + JetBrains Mono (Google Fonts) |
| Containerization | Docker (multi-stage) |
| Deployment | Render (Docker web service) |

---

## 📁 Project Structure

```
saood-bot/
├── app.py              ← Flask backend (NLP + API endpoints)
├── faqs.json           ← 32 FAQ question/answer pairs
├── requirements.txt    ← Python dependencies
├── Dockerfile          ← Multi-stage Docker build
├── render.yaml         ← Render.com deployment config
├── .dockerignore
├── .gitignore
└── public/
    └── index.html      ← Complete frontend (dark futuristic UI)
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11+
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/saudcoder7/saood-bot.git
cd saood-bot

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
PORT=8080 python app.py
```

Open **http://localhost:8080** in your browser.

---

## 🐳 Running with Docker

```bash
# Build
docker build -t saood-bot .

# Run
docker run -p 8080:8080 saood-bot
```

Open **http://localhost:8080**.

---

## 🌐 API Reference

### `POST /chat`
Ask the chatbot a question.

**Request:**
```json
{ "message": "how do I apply for admission?" }
```

**Response:**
```json
{
  "answer": "To apply for admission, visit our official admissions portal...",
  "matched_question": "How do I apply for admission to the university?",
  "confidence": 0.8689
}
```

> If confidence < 0.28, `matched_question` will be `null` and `answer` will be a fallback message.

---

### `GET /faqs`
Get all FAQ questions (used by the frontend for suggestion chips).

**Response:**
```json
{
  "questions": [
    "How do I apply for admission to the university?",
    "What are the admission deadlines for the upcoming semester?",
    ...
  ]
}
```

---

## ✅ Test Results

| Query | Confidence | Result |
|-------|-----------|--------|
| `"how do I apply for admission"` | **86.9%** | ✅ Correct answer |
| `"is there a hostel on campus"` | **61.1%** | ✅ Correct answer |
| `"what's the weather today"` | **0.0%** | ✅ Fallback triggered |

---

## ☁️ Deploy to Render

1. Fork this repo
2. Go to [dashboard.render.com](https://dashboard.render.com)
3. Click **New → Web Service** → Connect your GitHub repo
4. Set **Environment: Docker**, **Branch: main**, **Plan: Free**
5. Click **Create Web Service**

The `render.yaml` file auto-configures everything.

---

## 📄 License

MIT © Saood — 2025
