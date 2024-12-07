# Coba Langgraph

Coba Langgraph adalah project yang dirancang untuk membantu memahami dan mengimplementasikan Langgraph pada Retrieval Augmented Generation (RAG) ataupun kombinasi Multi-Agent LLM dari langchain, sebuah pendekatan yang menggabungkan beberapa agent untuk membangun aplikasi AI yang cerdas dan responsif. Project ini cocok untuk pengembang, peneliti, dan siapa saja yang ingin mempelajari cara mengintegrasikan Langgraph pada RAG ke dalam aplikasi berbasis model bahasa besar (LLM).

## Teknologi

- [Python](https://www.python.org/): Bahasa pemrograman untuk membuat Virtual Assistant.
- [Langchain](https://www.langchain.com/): Framework untuk mengelola alur kerja RAG.
- [Langgraph](https://www.langchain.com/langgraph): Framework untuk mengelola alur kerja multi-agent LLM.

## Instalasi Project

Clone project

```bash
  git clone https://github.com/odetv/coba-langgraph.git
```

Masuk ke direktori project

```bash
  cd coba-langgraph
```

Persiapkan environment python

```bash
  pip install virtualenv
  python -m venv venv
  venv/Scripts/activate (jika OS Windows)
  source venv/bin/activate (jika macOS/Linux)
```

Install packages requirements

```bash
  pip install -r requirements.txt
```

Buat dan lengkapi file environment variabel (.env)

```bash
  OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

Sesuaikan pertanyaan di `main.py`

```bash
  run("Sesuaikan Pertanyaan Disini")
```

Jalankan dengan CLI

```bash
  python main.py
```

## Authors

- [@odetv](https://www.github.com/odetv)
- [@DiarCode11](https://github.com/DiarCode11)
