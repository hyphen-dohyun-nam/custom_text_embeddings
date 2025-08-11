use anyhow::Result;
use text_embedding_inference::QwenEmbedder;
use std::io::{self, Write};

fn main() -> Result<()> {
    // let embedder = QwenEmbedder::new("models/QWEN3-Embedding-0.6B")?;
    let embedder = QwenEmbedder::load()?;

    println!("=== QWEN3-Embedding-0.6B Interactive Demo ===\n");
    
    let example_texts = vec![
        "Alice Smith is a Software Engineer at TechCorp. Skills: Rust, Python, Machine Learning.",
        "Bob Johnson is a Data Scientist at DataWorks. Skills: Python, Statistics, Deep Learning.",
        "Carol Williams is a Product Manager at InnovateX. Skills: Leadership, Agile, Communication.",
        "David Brown is a UX Designer at DesignHub. Skills: Figma, Sketch, User Research.",
        "Eve Davis is a DevOps Engineer at CloudOps. Skills: AWS, Docker, Kubernetes.",
        "Frank Miller is a QA Analyst at QualityFirst. Skills: Testing, Automation, Selenium.",
        "Grace Wilson is a Frontend Developer at Webify. Skills: JavaScript, React, CSS.",
        "Henry Moore is a Backend Developer at ServerSide. Skills: Go, Node.js, Databases.",
        "Ivy Taylor is an AI Researcher at FutureAI. Skills: NLP, Transformers, PyTorch.",
        "Jack Anderson is a Mobile Developer at AppMakers. Skills: Swift, Kotlin, Flutter.",
        "Kate Thomas is a Cloud Architect at SkyNet. Skills: Azure, GCP, Microservices.",
        "Leo Jackson is a Security Specialist at SafeGuard. Skills: Penetration Testing, Encryption, Firewalls.",
        "Mia White is a Business Analyst at BizInsights. Skills: Excel, SQL, Reporting.",
        "Nate Harris is a Database Administrator at DataKeepers. Skills: PostgreSQL, MySQL, Backup.",
        "Olivia Martin is a Content Writer at WriteRight. Skills: SEO, Copywriting, Editing.",
        "Paul Lee is an IT Support Specialist at HelpDesk. Skills: Troubleshooting, Customer Service, Networking.",
        "Quinn Walker is a Systems Engineer at SysTech. Skills: Linux, Virtualization, Scripting.",
        "Ruby Hall is a Marketing Manager at MarketMinds. Skills: Strategy, Branding, Analytics.",
        "Sam Young is a Network Engineer at NetWorks. Skills: Routing, Switching, Firewall.",
        "Tina King is an HR Specialist at PeopleFirst. Skills: Recruitment, Onboarding, Payroll.",
        "Uma Scott is an Operations Manager at OpsPro. Skills: Logistics, Process Improvement, Team Leadership.",
        "Victor Green is a Sales Executive at SellWell. Skills: Negotiation, CRM, Lead Generation.",
        "Wendy Baker is a Graphic Designer at Artify. Skills: Photoshop, Illustrator, Creativity.",
        "Xander Carter is a Game Developer at PlayWorks. Skills: Unity, C#, Game Design.",
        "Yara Evans is a Research Scientist at BioLabs. Skills: Biology, Lab Work, Data Analysis.",
        "Zane Perez is a Financial Analyst at FinWise. Skills: Accounting, Forecasting, Excel."
    ];
    
    let mut example_embeddings = Vec::new();
    
    for (_i, text) in example_texts.iter().enumerate() {
        let embedding = embedder.forward(text, &false)?;
        example_embeddings.push((text, embedding));
    }
    
    println!("Enter your own sentences to get embeddings and compare with examples!");
    println!("Type 'quit' or 'exit' to stop.\n");
    
    loop {
        print!("Enter your sentence: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() || input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }
        
        match embedder.forward(input, &true) {
            Ok(embedding) => {
                
                println!("\n✅ Your sentence: '{}'", input);
                println!("Embedding dimension: {}", embedding.len());
                println!("First 10 values: {:?}", &embedding[..10.min(embedding.len())]);
                
                // Find most similar examples
                println!("\n🔍 Similarity with example sentences:");
                let mut similarities: Vec<(usize, f32)> = example_embeddings
                    .iter()
                    .enumerate()
                    .map(|(i, (_, example_emb))| {
                        let sim = QwenEmbedder::cosine_similarity(&embedding, example_emb);
                        (i, sim)
                    })
                    .collect();
                
                // Sort by similarity (highest first)
                similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                
                // Table header
                println!("\n| {:<8} | {:<60} |", "Score", "Example Sentence");
                println!("{}", "-".repeat(72));

                // Show top 5 most similar
                for (_i, (idx, similarity)) in similarities.iter().take(5).enumerate() {
                    println!("| {:<8.4} | {:<60} |", similarity, example_texts[*idx]);
                }

                // Table header for least similar
                println!("\nLeast similar:");
                println!("| {:<8} | {:<60} |", "Score", "Example Sentence");
                println!("{}", "-".repeat(72));
                for (_i, (idx, similarity)) in similarities.iter().rev().take(3).enumerate() {
                    println!("| {:<8.4} | {:<60} |", similarity, example_texts[*idx]);
                }

                println!();
                println!("{}", "-".repeat(72));
            }
            Err(e) => {
                println!("❌ Error processing your sentence: {}", e);
            }
        }
    }
    
    Ok(())
}
