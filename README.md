# Text Embeddings Inference

A Rust-based text embedding inference system using the Qwen3-Embedding-0.6B model with vector database storage and similarity search capabilities.

## Features

- **State-of-the-art Embeddings**: Uses Qwen3-Embedding-0.6B model for high-quality text embeddings
- **Vector Database**: Local SQLite database with vector search capabilities via Turso
- **Interactive CLI**: Real-time text input and similarity search
- **Cosine Similarity**: Efficient similarity computation for vector comparison
- **Async Support**: Built with Tokio for asynchronous operations

## Architecture

The project consists of three main components:

1. **QwenEmbedder**: Handles text-to-embedding conversion using the Qwen3 model
2. **TursoDB**: Manages vector storage and similarity search operations
3. **Interactive CLI**: Provides user interface for real-time embedding and search

## Prerequisites

- Rust 2024 edition or later
- Internet connection (for downloading the Qwen3 model on first run)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hyphen-dohyun-nam/custom_text_embeddings.git
cd text-embeddings-inference
```

2. Build the project:
```bash
cargo build --release
```

## Usage

### Basic Usage

Run the interactive demo:
```bash
cargo run --release
```

The application will:
1. Download the Qwen3-Embedding-0.6B model (first run only)
2. Initialize the local vector database
3. Start an interactive session for text input

### Interactive Session

Once running, you can:
- Enter any text to generate embeddings
- View similarity search results against stored candidates
- Type `quit` or `exit` to stop the session

Example interaction:
```
Enter your sentence: Software engineer with Python experience
✅ Your sentence: 'Software engineer with Python experience'
Embedding dimension: 1024
First 10 values: [0.1234, -0.5678, ...]

🔍 Top 5 most similar candidates from DB:
| Dist     | Description                                                  |
------------------------------------------------------------------------
| 0.8912   | Alice Smith is a Software Engineer at TechCorp...          |
| 0.8456   | Bob Johnson is a Data Scientist at DataWorks...            |
...
```

## API Reference

### QwenEmbedder

```rust
use text_embedding_inference::QwenEmbedder;

// Load the model
let embedder = QwenEmbedder::load()?;

// Generate embeddings
let embedding = embedder.forward("Your text here", &false)?;

// For search queries (with instruction prompting)
let query_embedding = embedder.forward("Search query", &true)?;

// Calculate cosine similarity
let similarity = QwenEmbedder::cosine_distance(&vec1, &vec2)?;
```

### TursoDB

```rust
use text_embedding_inference::TursoDB;

// Initialize database
let db = TursoDB::new("path/to/database.db").await?;

// Create tables
db.create().await?;

// Insert embeddings
db.insert("Text description", &embedding).await?;

// Search similar vectors
let results = db.vector_search(&query_embedding).await?;
```

## Configuration

### Model Configuration
- **Model**: Qwen3-Embedding-0.6B from Hugging Face
- **Embedding Dimension**: 1024
- **Device**: CPU (configurable)

### Database Configuration
- **Type**: Turso (SQLite re-write in rust with built-in vector search)
- **Storage**: Local file (`src/turso_db/data.db`)
- **Vector Type**: F32_BLOB(1024)

## Dependencies

Key dependencies include:
- `candle-core`: ML framework for model inference
- `tokenizers`: Text tokenization
- `hf-hub`: Hugging Face model hub integration
- `turso`: Vector database operations
- `tokio`: Async runtime
- `anyhow`: Error handling

## Performance

- **Model Size**: ~1.3GB (Qwen3-Embedding-0.6B)
  - model will be saved in ```~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/*```
- **Inference Speed**: CPU-optimized for real-time use
  - Optimization done via candle-core, supports ```cpu```
- **Memory Usage**: Efficient tensor operations with Candle
- **Vector Search**: Optimized cosine distance calculations via ```turso```

## Development

### Project Structure
```
src/
├── main.rs              # Interactive CLI application
├── lib.rs               # Library exports
├── embedder/
│   ├── mod.rs
│   └── qwen3embedder.rs # Qwen3 model implementation
└── turso_db/
    ├── mod.rs
    ├── turso_db.rs      # Vector database operations
    └── data.db*         # SQLite database file
```

### Building for Development
```bash
# Debug build
cargo build

# Run with debug output
RUST_LOG=debug cargo run

# Run tests
cargo test

# Clean build artifacts
cargo clean
```

### Adding Custom Examples

The codebase includes commented example data that can be uncommented in `main.rs` to populate the database with sample professional profiles for testing similarity search.

## Error Handling

The application uses `anyhow::Result` for comprehensive error handling:
- Model loading errors
- Tokenization failures
- Database connection issues
- Vector computation errors

## License

This project uses the Qwen3-Embedding-0.6B model, which is subject to its own license terms. Please refer to the [Qwen model repository](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) for licensing information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Ensure stable internet connection for first run
2. **Database Errors**: Check file permissions for `src/turso_db/data.db`
3. **Memory Issues**: Monitor system memory during model loading
4. **Compilation Errors**: Ensure Rust 2024 edition compatibility

### Debug Mode

For verbose logging:
```bash
RUST_LOG=debug cargo run --release
```