use anyhow::Result;
use text_embedding_inference::QwenEmbedder;
use std::io::{self, Write};

fn main() -> Result<()> {
    println!("=== QWEN3-Embedding-0.6B Interactive Demo ===\n");
    // Initialize the embedder
    let embedder = QwenEmbedder::new("models/QWEN3-Embedding-0.6B")?;
    
    // More example sentences for comparison
    // println!("=== Preloaded Example Sentences ===");
    let example_texts = vec![
        "Machine learning is fascinating",
        "I love programming and coding",
        "The weather is nice today",
        "Artificial intelligence will change the world",
        "I enjoy reading books in the evening",
        "The cat is sleeping on the couch",
        "Programming languages are tools for developers",
        "Natural language processing is a subfield of AI",
        "The sun is shining brightly",
        "I need to buy groceries from the store",
        "Deep learning models require large datasets",
        "Coffee helps me stay awake during work",
        "Traveling opens your mind to new cultures",
        "Music is a universal language",
        "The internet connects people globally",
        "Quantum computing is the future of technology",
        "Data science combines statistics and programming",
        "The stock market is volatile",
        "Healthy eating is important for well-being",
        "Exercise is beneficial for physical health",
        "Learning new skills can boost your career",
        "The ocean is vast and mysterious",
        "Space exploration expands our understanding of the universe",
        "History teaches us valuable lessons",
        "Art can express complex emotions",
        "Cooking is both a science and an art",
        "Gardening is a relaxing hobby",
        "Photography captures moments in time",
        "Writing helps clarify thoughts and ideas",
        "Volunteering can make a difference in the community",
        "Hiking is a great way to enjoy nature",
    ];
    
    let mut example_embeddings = Vec::new();
    
    for (_i, text) in example_texts.iter().enumerate() {
        let embedding = embedder.str2vec(text)?;
        let normalized = QwenEmbedder::normalize_l2(&embedding);
        example_embeddings.push((text, normalized));
        // println!("{}. '{}' -> embedding dim: {}", i+1, text, embedding.len());
    }
    
    // Interactive loop
    // println!("\n=== Interactive Mode ===");
    println!("Enter your own sentences to get embeddings and compare with examples!");
    println!("Type 'quit' or 'exit' to stop.\n");
    
    loop {
        print!("Enter your sentence: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        // Check for exit conditions
        if input.is_empty() || input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }
        
        // Process user input
        match embedder.str2vec(input) {
            Ok(embedding) => {
                let normalized = QwenEmbedder::normalize_l2(&embedding);
                
                println!("\n✅ Your sentence: '{}'", input);
                println!("Embedding dimension: {}", embedding.len());
                println!("First 10 values: {:?}", &embedding[..10.min(embedding.len())]);
                
                // Find most similar examples
                println!("\n🔍 Similarity with example sentences:");
                let mut similarities: Vec<(usize, f32)> = example_embeddings
                    .iter()
                    .enumerate()
                    .map(|(i, (_, example_emb))| {
                        let sim = QwenEmbedder::cosine_similarity(&normalized, example_emb);
                        (i, sim)
                    })
                    .collect();
                
                // Sort by similarity (highest first)
                similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                // Show top 5 most similar
                println!("Top 5 most similar:");
                for (i, (idx, similarity)) in similarities.iter().take(5).enumerate() {
                    println!("  {}. {:.4} - '{}'", i+1, similarity, example_texts[*idx]);
                }
                
                // Show top 3 least similar for contrast
                println!("\nLeast similar:");
                for (i, (idx, similarity)) in similarities.iter().rev().take(3).enumerate() {
                    println!("  {}. {:.4} - '{}'", i+1, similarity, example_texts[*idx]);
                }
                
                println!();
                println!("{}", "-".repeat(60));
            }
            Err(e) => {
                println!("❌ Error processing your sentence: {}", e);
            }
        }
    }
    
    Ok(())
}
