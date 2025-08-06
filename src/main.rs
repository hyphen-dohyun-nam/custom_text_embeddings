use anyhow::Result;
use text_embedding_inference::QwenEmbedder;
use std::io::{self, Write};

fn main() -> Result<()> {
    let embedder = QwenEmbedder::new("models/QWEN3-Embedding-0.6B")?;

    println!("=== QWEN3-Embedding-0.6B Interactive Demo ===\n");
    
    let example_texts = vec![
        r#"{"first_name": "Alice", "last_name": "Smith", "title": "Software Engineer", "company": "TechCorp", "skills": ["Rust", "Python", "Machine Learning"]}"#,
        r#"{"first_name": "Bob", "last_name": "Johnson", "title": "Data Scientist", "company": "DataWorks", "skills": ["Python", "Statistics", "Deep Learning"]}"#,
        r#"{"first_name": "Carol", "last_name": "Williams", "title": "Product Manager", "company": "InnovateX", "skills": ["Leadership", "Agile", "Communication"]}"#,
        r#"{"first_name": "David", "last_name": "Brown", "title": "UX Designer", "company": "DesignHub", "skills": ["Figma", "Sketch", "User Research"]}"#,
        r#"{"first_name": "Eve", "last_name": "Davis", "title": "DevOps Engineer", "company": "CloudOps", "skills": ["AWS", "Docker", "Kubernetes"]}"#,
        r#"{"first_name": "Frank", "last_name": "Miller", "title": "QA Analyst", "company": "QualityFirst", "skills": ["Testing", "Automation", "Selenium"]}"#,
        r#"{"first_name": "Grace", "last_name": "Wilson", "title": "Frontend Developer", "company": "Webify", "skills": ["JavaScript", "React", "CSS"]}"#,
        r#"{"first_name": "Henry", "last_name": "Moore", "title": "Backend Developer", "company": "ServerSide", "skills": ["Go", "Node.js", "Databases"]}"#,
        r#"{"first_name": "Ivy", "last_name": "Taylor", "title": "AI Researcher", "company": "FutureAI", "skills": ["NLP", "Transformers", "PyTorch"]}"#,
        r#"{"first_name": "Jack", "last_name": "Anderson", "title": "Mobile Developer", "company": "AppMakers", "skills": ["Swift", "Kotlin", "Flutter"]}"#,
        r#"{"first_name": "Kate", "last_name": "Thomas", "title": "Cloud Architect", "company": "SkyNet", "skills": ["Azure", "GCP", "Microservices"]}"#,
        r#"{"first_name": "Leo", "last_name": "Jackson", "title": "Security Specialist", "company": "SafeGuard", "skills": ["Penetration Testing", "Encryption", "Firewalls"]}"#,
        r#"{"first_name": "Mia", "last_name": "White", "title": "Business Analyst", "company": "BizInsights", "skills": ["Excel", "SQL", "Reporting"]}"#,
        r#"{"first_name": "Nate", "last_name": "Harris", "title": "Database Administrator", "company": "DataKeepers", "skills": ["PostgreSQL", "MySQL", "Backup"]}"#,
        r#"{"first_name": "Olivia", "last_name": "Martin", "title": "Content Writer", "company": "WriteRight", "skills": ["SEO", "Copywriting", "Editing"]}"#,
        r#"{"first_name": "Paul", "last_name": "Lee", "title": "IT Support Specialist", "company": "HelpDesk", "skills": ["Troubleshooting", "Customer Service", "Networking"]}"#,
        r#"{"first_name": "Quinn", "last_name": "Walker", "title": "Systems Engineer", "company": "SysTech", "skills": ["Linux", "Virtualization", "Scripting"]}"#,
        r#"{"first_name": "Ruby", "last_name": "Hall", "title": "Marketing Manager", "company": "MarketMinds", "skills": ["Strategy", "Branding", "Analytics"]}"#,
        r#"{"first_name": "Sam", "last_name": "Young", "title": "Network Engineer", "company": "NetWorks", "skills": ["Routing", "Switching", "Firewall"]}"#,
        r#"{"first_name": "Tina", "last_name": "King", "title": "HR Specialist", "company": "PeopleFirst", "skills": ["Recruitment", "Onboarding", "Payroll"]}"#,
        r#"{"first_name": "Uma", "last_name": "Scott", "title": "Operations Manager", "company": "OpsPro", "skills": ["Logistics", "Process Improvement", "Team Leadership"]}"#,
        r#"{"first_name": "Victor", "last_name": "Green", "title": "Sales Executive", "company": "SellWell", "skills": ["Negotiation", "CRM", "Lead Generation"]}"#,
        r#"{"first_name": "Wendy", "last_name": "Baker", "title": "Graphic Designer", "company": "Artify", "skills": ["Photoshop", "Illustrator", "Creativity"]}"#,
        r#"{"first_name": "Xander", "last_name": "Carter", "title": "Game Developer", "company": "PlayWorks", "skills": ["Unity", "C#", "Game Design"]}"#,
        r#"{"first_name": "Yara", "last_name": "Evans", "title": "Research Scientist", "company": "BioLabs", "skills": ["Biology", "Lab Work", "Data Analysis"]}"#,
        r#"{"first_name": "Zane", "last_name": "Perez", "title": "Financial Analyst", "company": "FinWise", "skills": ["Accounting", "Forecasting", "Excel"]}"#,
    ];
    
    let mut example_embeddings = Vec::new();
    
    for (_i, text) in example_texts.iter().enumerate() {
        let embedding = embedder.embed(text)?;
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
        match embedder.embed(input) {
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
                
                use serde_json::Value;
                // Helper to parse and extract fields from JSON string
                fn parse_fields(json_str: &str) -> (String, String, String, String, String) {
                    let v: Value = serde_json::from_str(json_str).unwrap_or(Value::Null);
                    let first_name = v.get("first_name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let last_name = v.get("last_name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let title = v.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let company = v.get("company").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let mut skills = v.get("skills")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", "))
                        .unwrap_or_else(|| "".to_string());
                    if skills.len() > 27 {
                        skills = format!("{}...", &skills[..27]);
                    }
                    (first_name, last_name, title, company, skills)
                }

                // Table header
                println!("\n| {:<12} | {:<12} | {:<22} | {:<14} | {:<30} | {:<8} |", "First Name", "Last Name", "Title", "Company", "Skills", "Similarity");
                println!("{}", "-".repeat(112));

                // Show top 5 most similar
                for (_i, (idx, similarity)) in similarities.iter().take(5).enumerate() {
                    let (first, last, title, company, skills) = parse_fields(example_texts[*idx]);
                    println!("| {:<12} | {:<12} | {:<22} | {:<14} | {:<30} | {:<8.4} |", first, last, title, company, skills, similarity);
                }

                // Table header for least similar
                println!("\nLeast similar:");
                println!("| {:<12} | {:<12} | {:<22} | {:<14} | {:<30} | {:<8} |", "First Name", "Last Name", "Title", "Company", "Skills", "Similarity");
                println!("{}", "-".repeat(112));
                for (_i, (idx, similarity)) in similarities.iter().rev().take(3).enumerate() {
                    let (first, last, title, company, skills) = parse_fields(example_texts[*idx]);
                    println!("| {:<12} | {:<12} | {:<22} | {:<14} | {:<30} | {:<8.4} |", first, last, title, company, skills, similarity);
                }

                println!();
                println!("{}", "-".repeat(112));
            }
            Err(e) => {
                println!("❌ Error processing your sentence: {}", e);
            }
        }
    }
    
    Ok(())
}
