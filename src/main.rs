use anyhow::Result;
use text_embedding_inference::QwenEmbedder;
use text_embedding_inference::TursoDB;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<()> {
    let embedder = QwenEmbedder::load().await?;
    let db = TursoDB::new("src/turso_db/data.db").await?;
    db.delete().await?;
    db.create().await?;
    println!("=== QWEN3-Embedding-0.6B Interactive Demo ===\n");
    
    let example_texts = vec![
        "Alice Smith is a Software Engineer at TechCorp. Skills: Rust, Python, Machine Learning.",
        "Bob Johnson is a Data Scientist at DataWorks. Skills: Python, Statistics, Deep Learning.",
        "Carol Williamsc is a Product Manager at InnovateX. Skills: Leadership, Agile, Communication.",
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
        let embedding = embedder.forward(text, &false).await?;
        db.insert(text, &embedding.clone()).await?;
        example_embeddings.push((text, embedding));
    }
    let search_results = db.vector_search(&example_embeddings[0].1).await?;
    println!("Search results for '{}':", example_texts[0]);
    println!("Number of results: {}", search_results.len());
    for (i, (desc, dist)) in search_results.iter().enumerate() {
        println!("Result {}: {} (distance: {:.4})", i, desc, dist);
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
        
        match embedder.forward(input, &false).await {
            Ok(embedding) => {
                println!("\n✅ Your sentence: '{}'", input);
                println!("Embedding dimension: {}", embedding.len());
                println!("First 10 values: {:?}", &embedding[..10.min(embedding.len())]);

                // Use TursoDB to find most similar examples
                let db_results = db.vector_search(&embedding).await?;
                println!("\n🔍 Top 5 most similar candidates from DB:");
                println!("| {:<8} | {:<60} |", "Dist", "Description");
                println!("{}", "-".repeat(72));
                for (_i, (desc, dist)) in db_results.iter().take(5).enumerate() {
                    println!("| {:<8.4} | {:<60} |", dist, desc);
                }

                // Show least similar
                println!("\nLeast similar:");
                println!("| {:<8} | {:<60} |", "Dist", "Description");
                println!("{}", "-".repeat(72));
                let mut db_results_sorted = db_results.clone();
                db_results_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                for (_i, (desc, dist)) in db_results_sorted.iter().take(3).enumerate() {
                    println!("| {:<8.4} | {:<60} |", dist, desc);
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
