use dioxus::prelude::*;
use text_embeddings_inference::{load, load_test_dataset, search};
use std::sync::Arc;
use tokio::sync::Mutex;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let mut query = use_signal(String::new);
    let mut search_result = use_signal(|| Vec::<(String, f32)>::new());
    let mut is_loading = use_signal(|| false);
    let mut is_initialized = use_signal(|| false);
    
    let embedder_db = use_resource(move || async move {
        let result = load().await;
        match result {
            Ok(loaded) => {
                println!("Embedder and DB loaded successfully");
                Some(Arc::new((
                    loaded.0, 
                    Mutex::new(loaded.1)
                )))
            },
            Err(e) => {
                println!("Failed to load embedder and DB: {:?}", e);
                None
            }
        }
    });

    // Initialize test dataset when embedder_db is loaded
    use_effect(move || {
        let embedder_db_val = embedder_db.read();
        
        // Only run this effect when the embedder_db resource changes
        if let Some(Some(data)) = embedder_db_val.as_ref() {
            let data = data.clone();
            spawn(async move {
                is_loading.set(true);
                println!("Starting to load test dataset...");
                let db = data.1.lock().await;
                if let Err(e) = load_test_dataset(&data.0, &db).await {
                    println!("Failed to load test dataset: {:?}", e);
                } else {
                    println!("Test dataset loaded successfully");
                    is_initialized.set(true);
                }
                is_loading.set(false);
            });
        }
        
        // Return nothing for cleanup (no cleanup needed)
        ()
    });

    let handle_search = move |e: KeyboardEvent| {
        // Only trigger search when Enter key is pressed
        if e.key() == Key::Enter && !query.read().is_empty() && is_initialized() && !is_loading() {
            let query_text = query.read().clone();
            
            // Clone the necessary resources
            let embedder_db_clone = embedder_db.clone();
            
            // Prevent multiple searches while one is in progress
            is_loading.set(true);
            println!("Search triggered for: {}", query_text);
            
            spawn(async move {
                if let Some(Some(data)) = &*embedder_db_clone.read() {
                    let db = data.1.lock().await;
                    match search(&db, &query_text, &data.0).await {
                        Ok(results) => {
                            let result_len = results.len();
                            search_result.set(results);
                            println!("Search completed with {} results", result_len);
                        },
                        Err(e) => {
                            println!("Search error: {:?}", e);
                        }
                    }
                    is_loading.set(false);
                } else {
                    println!("Embedder or DB not initialized");
                    is_loading.set(false);
                }
            });
        }
    };

    let status_message = if !is_initialized() {
        if is_loading() {
            "Loading test dataset... ⏳"
        } else if embedder_db.read().is_some() {
            "Preparing dataset... 🔄"
        } else {
            "Initializing embedder and DB... 🔄"
        }
    } else if is_loading() {
        "Searching... 🔍"
    } else {
        "Ready - Press Enter to search ✅"
    };

    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        div {
            id: "main",
            h1 { "RAG Demo with QWEN3-Embedding-0.6B" }
            p { "Status: {status_message}" }
            
            div {
                class: "search-box",
                input {
                    id: "QueryInput",
                    value: "{query}",
                    style: "width: 100%; max-width: 600px;",
                    oninput: move |e| query.set(e.value().clone()),
                    onkeydown: handle_search,
                    placeholder: if is_initialized() && !is_loading() { 
                        "Enter your search query and press Enter..." 
                    } else { 
                        "Wait for initialization to complete..." 
                    },
                    disabled: is_loading() || !is_initialized(),
                    class: if is_initialized() && !is_loading() { "ready" } else { "disabled" },
                }
            }
            
            if !search_result.read().is_empty() {
                div {
                    class: "results",
                    h2 { "Search Results:" }
                    table {
                        thead {
                            tr {
                                th { "Score" }
                                th { "Description" }
                            }
                        }
                        tbody {
                            for (_i, (desc, score)) in search_result.read().iter().enumerate() {
                                tr {
                                    td { "{score:.3}" }
                                    td { "{desc}" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}