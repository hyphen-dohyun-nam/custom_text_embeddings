use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor, safetensors::load};
use tokenizers::Tokenizer;
use hf_hub::api::sync::Api;

pub struct QwenEmbedder {
    embedding_weights: Tensor,
    tokenizer: Tokenizer,
    device: Device,
}

impl QwenEmbedder {
    pub fn load() -> Result<Self> {
        let device = Device::Cpu;

        let api = Api::new().unwrap();
        let repo = api.model("Qwen/Qwen3-Embedding-0.6B".to_string());

        let tokenizer_json = repo.get("tokenizer.json").unwrap();
        let tokenizer = Tokenizer::from_file(tokenizer_json).map_err(E::msg)?;

        let weights_safetensor = repo.get("model.safetensors")?;
        let tensors = load(&weights_safetensor, &device)?;

        let embedding_weights = tensors.get("embed_tokens.weight")
            .ok_or_else(|| {
                let available_keys: Vec<_> = tensors.keys().collect();
                E::msg(format!("Could not find 'model.embed_tokens.weight' in model file. Available keys: {:?}", available_keys))
            })?
            .clone();

        Ok(Self {
            embedding_weights,
            tokenizer,
            device,
        })
    }
    
    pub fn forward(&self, text: &str, is_query: &bool) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(E::msg("Empty text provided"));
        }

        #[allow(unused_variables)]
        if *is_query {
            let text = Self::get_detailed_instruct("Given a candidate search query, retrieve relevant candidates that fits the description", text);
        }

        let tokens = self.tokenizer.encode(text, false)
            .map_err(|e| E::msg(format!("Tokenizer error: {}", e)))?;
        let ids: Vec<u32> = tokens.get_ids().to_vec();

    let ids_tensor = Tensor::new(ids.clone(), &self.device)?;

    let embeddings = self.embedding_weights.embedding(&ids_tensor)?;

    let mean_embedding = embeddings.mean(0)?;
    let mean_embedding_f32 = mean_embedding.to_dtype(candle_core::DType::F32)?;
    Ok(mean_embedding_f32.to_vec1()?)
    }

    fn get_detailed_instruct(task_description: &str, query: &str) -> String {
        format!("Instruct: {task_description}\nQuery:{query}")
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
