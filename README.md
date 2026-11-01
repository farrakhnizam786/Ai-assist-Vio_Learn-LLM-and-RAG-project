# Ai-assist-Vio_Learn-LLM-and-RAG-project

Ai-assist-Vio_Learn-LLM-and-RAG-project which able to answer using videos provided(converted to chunks and accuracy is more then 80%)

# 🎓 Ai-assist-Vio_Learn-LLM-and-RAG-project (Video RAG Pipeline)

An end-to-end Retrieval-Augmented Generation (RAG) system designed to process lecture videos/audio, transcribe them with precise timestamps, store embeddings in a persistent vector database, and answer student queries accurately using an advanced LLM.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Audio Extraction:** MoviePy
* **Transcription:** OpenAI-Whisper
* **Embeddings & Vector Store:** Sentence-Transformers (`all-MiniLM-L6-v2`) & ChromaDB
* **LLM API:** Groq API (`openai/gpt-oss-120b`)
* **Environment Management:** Python Dotenv

---

## 📂 Project Architecture
```text
AIassisTeacher/
│
├── input_videos/            # Place your source lecture videos (.mp4) here
├── output_data/             # Stores extracted audio (.mp3) and structured JSON transcripts
├── vector_db/               # Persistent ChromaDB vector database storage
├── transcribe.py            # Video-to-MP3 conversion & Whisper transcription + chunk optimization
├── rag_pipeline.py          # ChromaDB storage, semantic retrieval, and prompt engineering
├── main.py                  # Master pipeline orchestrator
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
🚀 Installation & Quick Start
Clone the repository / Navigate to project folder:

DOS
cd AIassisTeacher
Install dependencies:

DOS
pip install -r requirements.txt
Configure Environment Variables:
Create a .env file in the root directory and add your Groq API key:

Code snippet
GROQ_API_KEY=your_actual_groq_api_key_here
Add a Video:
Place any sample lecture video inside the input_videos/ folder and name it sample_video.mp4.

Run the Master Pipeline:

DOS
python main.py
🌟 Key Features
Automated Audio Pipeline: Extracts audio from video and generates accurate time-stamped text using OpenAI-Whisper.

Context Optimization: Groups short transcript segments into rich chunks to preserve core educational context.

Persistent Vector Search: Utilizes ChromaDB for fast and efficient local semantic similarity search.

Formatted Timestamps: Automatically converts raw seconds into a user-friendly MM:SS format.

Hallucination Guardrails: Strict RAG prompt constraints ensure the AI responds only to the provided lecture material, complete with a fallback mechanism.
