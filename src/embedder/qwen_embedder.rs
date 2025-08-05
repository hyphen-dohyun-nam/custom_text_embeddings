use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor, safetensors::load};
use candle_nn::{Embedding, Module};
use serde_json;
use serde::Deserialize;
use std::path::Path;
use std::collections::HashMap;

#[derive(Deserialize)]
struct QwenConfig {
    hidden_size: usize
}

struct SimpleQwenTokenizer {
    vocab: HashMap<String, u32>
}

impl SimpleQwenTokenizer {
    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;
        
        let mut vocab = HashMap::new();
        
        // Extract vocab from model section
        if let Some(model) = json.get("model") {
            if let Some(vocab_obj) = model.get("vocab") {
                if let Some(vocab_map) = vocab_obj.as_object() {
                    for (token, id) in vocab_map {
                        if let Some(id_num) = id.as_u64() {
                            let id_u32 = id_num as u32;
                            vocab.insert(token.clone(), id_u32);
                        }
                    }
                }
            }
        }
        
        // Extract special tokens
        if let Some(added_tokens) = json.get("added_tokens") {
            if let Some(tokens_array) = added_tokens.as_array() {
                for token_obj in tokens_array {
                    if let (Some(content), Some(id)) = (
                        token_obj.get("content").and_then(|v| v.as_str()),
                        token_obj.get("id").and_then(|v| v.as_u64())
                    ) {
                        let id_u32 = id as u32;
                        vocab.insert(content.to_string(), id_u32);
                    }
                }
            }
        }
        
        Ok(Self {
            vocab
        })
    }
    
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        
        if text.is_empty() {
            return tokens;
        }
        
        let mut remaining = text;
        
        while !remaining.is_empty() {
            let mut found = false;
            
            for len in (1..=remaining.len()).rev() {
                let substr = &remaining[..len];
                
                if let Some(&token_id) = self.vocab.get(substr) {
                    tokens.push(token_id);
                    remaining = &remaining[len..];
                    found = true;
                    break;
                }
            }
            
            if !found {
                let first_char = remaining.chars().next().unwrap();
                let char_str = first_char.to_string();
                
                if let Some(&token_id) = self.vocab.get(&char_str) {
                    tokens.push(token_id);
                } else {
                    // Use unknown token or a safe fallback
                    tokens.push(0); // Assuming 0 is a safe token
                }
                
                let char_len = first_char.len_utf8();
                remaining = &remaining[char_len..];
            }
        }
        
        tokens
    }
}

pub struct QwenEmbedder {
    embedding_layer: Embedding,
    tokenizer: SimpleQwenTokenizer,
    device: Device,
}

impl QwenEmbedder {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let device = Device::Cpu;
        let model_dir = model_dir.as_ref();
        
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(config_path)?;
        let qwen_config: QwenConfig = serde_json::from_str(&config_str)?;
        
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = SimpleQwenTokenizer::from_file(tokenizer_path)?;
        
        let weights_path = model_dir.join("model.safetensors");
        let tensors = load(&weights_path, &device)?;
        
        let embedding_weights = tensors.get("model.embed_tokens.weight")
            .or_else(|| tensors.get("embed_tokens.weight"))
            .or_else(|| tensors.get("embeddings.word_embeddings.weight"))
            .ok_or_else(|| {
                let available_keys: Vec<_> = tensors.keys().collect();
                E::msg(format!("Could not find embedding weights in model file. Available keys: {:?}", available_keys))
            })?
            .clone();
        
        // Create embedding layer
        let embedding_layer = Embedding::new(embedding_weights, qwen_config.hidden_size);
        
        Ok(Self {
            embedding_layer,
            tokenizer,
            device,
        })
    }
    
    /// Convert text to embedding vector
    pub fn str2vec(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(E::msg("Empty text provided"));
        }
        
        // Tokenize the text
        let tokens = self.tokenizer.encode(text);
        
        // Convert tokens to vector
        self.token2vec(&tokens)
    }
    
    /// Convert token IDs to embeddings using mean pooling
    pub fn token2vec(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(E::msg("Empty token sequence provided"));
        }
        
        // Convert token IDs to tensor
        let input_ids = Tensor::new(tokens, &self.device)?;
        
        // Get embeddings for each token
        let token_embeddings = self.embedding_layer.forward(&input_ids)?;
        
        // Mean pooling over sequence length
        let embeddings = token_embeddings.mean(0)?;
        
        // Convert to Vec<f32> (handle BF16 to F32 conversion)
        let embeddings_vec = embeddings.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
        
        Ok(embeddings_vec)
    }
    
    /// L2 normalize a vector
    pub fn normalize_l2(vector: &[f32]) -> Vec<f32> {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            vector.to_vec()
        } else {
            vector.iter().map(|x| x / norm).collect()
        }
    }
    
    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }
        
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
}
