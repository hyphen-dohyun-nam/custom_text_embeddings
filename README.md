# RAG Demo with QWEN3-Embedding-0.6B
![Demo Screenshot](assets/demo.png)

## Features
- **Retrieval-Augmented Generation (RAG):** Search and retrieve relevant information using text embeddings.
- **QWEN3 Embedding Model:** Uses QWEN3-Embedding-0.6B for high-quality text embeddings.
- **Candle Backend:** Efficient inference engine for running QWEN3 models in Rust.
- **Turso Database:** Fast, embedded SQLite-compatible database for storing and retrieving data.
- **Dioxus Frontend:** Modern, reactive desktop UI built with Dioxus for seamless user experience.
- **Semantic Search Results:** Results are ranked by cosine distance between query and database embeddings. Lower scores indicate higher similarity (more relevant results).
- **Live Demo:** Enter queries and instantly see the most relevant matches from the database, including their similarity scores and descriptions.

## Project Structure
```
text-embeddings-inference/
├── Cargo.toml           # Rust project manifest
├── src/                 # Source code
│   ├── main.rs          # App entry point
│   ├── lib.rs           # Shared library code
│   ├── embedder/        # QWEN3 embedder logic
│   │   ├── mod.rs
│   │   └── qwen3embedder.rs
│   ├── turso_db/        # Turso DB integration
│   │   ├── mod.rs
│   │   └── turso_db.rs
│   │   ├── data.db      # Database file
├── assets/              # Static assets (CSS, images)
│   ├── main.css
│   ├── favicon.ico
│   └── demo.png
├── README.md            # Project documentation
├── Dioxus.toml          # Dioxus config
├── clippy.toml          # Linting config
└── target/              # Build output
```

## Development Setup Guide

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install)
- [Dioxus-CLI](https://dioxuslabs.com/learn/0.6/getting_started/)

### Run the App
```fish
dx serve
```
This will start the Dioxus desktop app. Open your browser or desktop window to interact with the RAG demo.

---
For more details, see the source files in `src/` and the assets in `assets/`.
