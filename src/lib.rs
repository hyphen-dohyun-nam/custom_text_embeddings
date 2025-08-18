pub mod embedder;
pub use embedder::QwenEmbedder;

pub mod turso_db;
pub use turso_db::TursoDB;

use anyhow::Result;

pub async fn load() -> Result<(QwenEmbedder, TursoDB)> {
    let embedder = QwenEmbedder::load().await?;
    
    let db_path = "src/turso_db/data.db";
    
    let db = TursoDB::new(db_path).await?;
    
    Ok((embedder, db))
}

pub async fn load_test_dataset(embedder: &QwenEmbedder, db: &TursoDB) -> Result<()> {
    // 1. Delete preexisting database
    db.delete().await?;
    
    // 2. Create a new one
    db.create().await?;
    
    // 3. Populate it with test dataset
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
    
    for text in example_texts {
        let embedding = embedder.forward(text, &false).await?;
        db.insert(text, &embedding).await?;
    }
    
    Ok(())
}

pub async fn search(db: &TursoDB, query: &str, embedder: &QwenEmbedder) -> Result<Vec<(String, f32)>> {
    // When searching with an exact match, we need to use the same parameters as we did when storing
    // Generate embedding for the query using is_query=false for consistency with stored embeddings
    let query_embedding = embedder.forward(query, &false).await?;
    
    // Perform semantic search in the database
    let search_results = db.vector_search(&query_embedding).await?;
    
    Ok(search_results)
}