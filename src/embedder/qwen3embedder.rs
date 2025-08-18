use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor, safetensors::load};
use tokenizers::Tokenizer;
use hf_hub::api::sync::Api;
use tokio::task;
use std::sync::Arc;

pub struct QwenEmbedder {
    embedding_weights: Arc<Tensor>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
}

impl QwenEmbedder {
    pub async fn load() -> Result<Self> {
        let device = Device::Cpu;

        // Using blocking API inside a spawn_blocking to make it async
        let model_data = task::spawn_blocking(move || -> Result<(std::path::PathBuf, std::path::PathBuf)> {
            let api = Api::new().map_err(E::msg)?;
            let repo = api.model("Qwen/Qwen3-Embedding-0.6B".to_string());

            let tokenizer_json = repo.get("tokenizer.json").map_err(E::msg)?;
            let weights_safetensor = repo.get("model.safetensors").map_err(E::msg)?;
            
            Ok((tokenizer_json, weights_safetensor))
        }).await??;
        
        let (tokenizer_json, weights_safetensor) = model_data;
        
        // Loading tokenizer is CPU-bound, so offload to a blocking task
        let tokenizer = task::spawn_blocking(move || {
            Tokenizer::from_file(tokenizer_json).map_err(E::msg)
        }).await??;

        // Model loading is CPU-bound, so offload to a blocking task
        let (weights_path, device_clone) = (weights_safetensor, device.clone());
        let tensors = task::spawn_blocking(move || {
            load(&weights_path, &device_clone).map_err(E::msg)
        }).await??;

        // Extract the embedding weights
        let embedding_weights = tensors.get("embed_tokens.weight")
            .ok_or_else(|| {
                let available_keys: Vec<_> = tensors.keys().collect();
                E::msg(format!("Could not find 'model.embed_tokens.weight' in model file. Available keys: {:?}", available_keys))
            })?
            .clone();

        Ok(Self {
            embedding_weights: Arc::new(embedding_weights),
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }
    
    pub async fn forward(&self, text: &str, is_query: &bool) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(E::msg("Empty text provided"));
        }

        let processed_text = if *is_query {
            Self::get_detailed_instruct("Given a candidate search query, retrieve relevant candidates that fits the description", text)
        } else {
            text.to_string()
        };

        // Tokenization and embedding generation can be CPU intensive, so offload to a blocking task
        let tokenizer = self.tokenizer.clone();
        let embedding_weights = self.embedding_weights.clone();
        let device = self.device.clone();
        let final_text = processed_text.clone();
        
        task::spawn_blocking(move || -> Result<Vec<f32>> {
            // Tokenize the text
            let tokens = tokenizer.encode(final_text, false)
                .map_err(|e| E::msg(format!("Tokenizer error: {}", e)))?;
            let ids: Vec<u32> = tokens.get_ids().to_vec();

            // Create tensor and get embeddings
            let ids_tensor = Tensor::new(ids, &device)
                .map_err(E::msg)?;
            let embeddings = embedding_weights.embedding(&ids_tensor)
                .map_err(E::msg)?;
            let mean_embedding = embeddings.mean(0)
                .map_err(E::msg)?;
            let mean_embedding_f32 = mean_embedding.to_dtype(candle_core::DType::F32)
                .map_err(E::msg)?;
            
            Ok(mean_embedding_f32.to_vec1().map_err(E::msg)?)
        }).await?
    }

    fn get_detailed_instruct(task_description: &str, query: &str) -> String {
        format!("Instruct: {task_description}\nQuery:{query}")
    }

    pub async fn cosine_distance(vec1: &[f32], vec2: &[f32]) -> Result<f64> {
        if vec1.len() != vec2.len() {
            return Err(E::msg("Invalid vector dimensions"));
        }

        if vec1.iter().any(|x| !x.is_finite()) || vec2.iter().any(|x| !x.is_finite()) {
            return Err(E::msg("Invalid vector value"));
        }
        
        // Vector operations can be CPU intensive for large vectors, so offload to a blocking task
        let v1 = vec1.to_vec();
        let v2 = vec2.to_vec();
        
        task::spawn_blocking(move || -> Result<f64> {
            let (mut dot, mut norm1, mut norm2) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..v1.len() {
                let e1 = v1[i] as f64;
                let e2 = v2[i] as f64;
                dot += e1 * e2;
                norm1 += e1 * e1;
                norm2 += e2 * e2;
            }

            if norm1 == 0.0 || norm2 == 0.0 {
                return Err(E::msg("Invalid vector value"));
            }

            Ok(dot / (norm1 * norm2).sqrt())
        }).await?
    }
}
