use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor, safetensors::load};
use candle_nn::{Embedding, Module};
use serde_json;
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct QwenConfig {
    hidden_size: usize
}

pub struct QwenEmbedder {
    embedding_layer: Embedding,
    tokenizer: Tokenizer,
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
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
        
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
        
        let embedding_layer = Embedding::new(embedding_weights, qwen_config.hidden_size);
        
        Ok(Self {
            embedding_layer,
            tokenizer,
            device,
        })
    }
    
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(E::msg("Empty text provided"));
        }
        
        let tokens = self.tokenizer.encode(text, false)
            .map_err(|e| E::msg(format!("Tokenizer error: {}", e)))?;
        let ids: Vec<u32> = tokens.get_ids().to_vec();

        let ids_tensor = Tensor::new(ids.clone(), &self.device)?;

        let embeddings = self.embedding_layer.forward(&ids_tensor)?;

        let mean_embedding = embeddings.mean(0)?;
        let mean_embedding_f32 = mean_embedding.to_dtype(candle_core::DType::F32)?;
        Ok(mean_embedding_f32.to_vec1()?)
    }
    
    pub fn normalize_l2(vector: &[f32]) -> Vec<f32> {
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            vector.to_vec()
        } else {
            vector.iter().map(|x| x / norm).collect()
        }
    }
    
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
