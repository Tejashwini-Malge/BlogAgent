# 🚀 AI Blog Generator (Multi-Agent System)

An intelligent blog generation system powered by **CrewAI, LangChain, and OpenRouter**.
This project uses multiple AI agents to collaboratively research, write, and refine high-quality blog content from a single topic input.

---

## ✨ Features

* 🧠 **Multi-Agent Workflow**

  * Research Agent → gathers structured insights
  * Writer Agent → creates engaging blog content
  * Editor Agent → refines and polishes output

* ⚡ **LLM Integration**

  * Powered by OpenRouter (via LangChain)
  * Supports multiple models

* 📝 **Automated Blog Generation**

  * Input a topic → get a complete blog post

* 💾 **File Export**

  * Automatically saves output as `.md` file

* 🖥️ **CLI + UI Ready**

  * Run via command line
  * Optional Streamlit UI for interactive use

---

## 🏗️ Project Structure

```
BlogAgent/
│
├── app.py              # Streamlit UI (optional)
├── .env                # API keys
├── output/             # Generated blogs
│
└── src/
    ├── agents.py       # AI agents (Researcher, Writer, Editor)
    ├── tasks.py        # Task definitions
    ├── crew.py         # Execution pipeline
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone <your-repo-link>
cd BlogAgent
```

---

### 2. Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install crewai langchain langchain-openai python-dotenv streamlit
```

---

### 4. Add API Key

Create a `.env` file in root:

```
OPENAI_API_KEY=your_openrouter_key_here
```

> ⚠️ Use OpenRouter key (`sk-or-...`)

---

## ▶️ Usage

### 🔹 Run via CLI

```
python -m src.crew "Your topic here"
```

Example:

```
python -m src.crew "Future of AI in Education"
```

---

### 🔹 Run via Streamlit UI

```
streamlit run app.py
```

Then open browser → enter topic → generate blog 🚀

---

## 🧠 How It Works

1. User provides a topic
2. **Research Agent** creates structured insights
3. **Writer Agent** converts it into a blog
4. **Editor Agent** refines and improves readability
5. Final output is saved as a Markdown file

---

## 📌 Example Output

* Topic: *"How to Apply for Jobs"*
* Output: Structured, polished blog (500–800 words)
* Saved in `/output/` folder

---

## 🚀 Future Improvements

* 🔄 UI Agent for smarter input refinement
* 📊 SEO Optimization Agent
* 🔗 LinkedIn Post Generator
* 📚 RAG integration (for personalized knowledge)
* 🌐 Deployment (Streamlit Cloud / Vercel)

---

## 🧑‍💻 Author

**Tejashwini Malge**

* AI • Communication • Confidence
* Building projects at the intersection of AI & storytelling

---

## ⭐ Contribute

Feel free to fork, improve, and build on top of this project!

---

## 📜 License

This project is open-source and available under the MIT License.
