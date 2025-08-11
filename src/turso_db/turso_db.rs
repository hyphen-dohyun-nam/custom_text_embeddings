use turso::{Builder, Connection};
use anyhow::Result;

pub struct TursoDB {
    conn: Connection,
}

impl TursoDB {
    pub async fn new(path: &str) -> Result<Self>{
        let db = Builder::new_local(path).build().await?;
        let conn = db.connect()?;
        Ok(Self {
            conn
        })
    }
    pub async fn create(&self) -> Result<()> {
        self.conn.execute(
            r#"CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                embedding F32_BLOB(1024)
            )"#,
            ()
        ).await?;
        Ok(())
    }

    pub async fn delete(&self) -> Result<()> {
        self.conn.execute(
            "DROP TABLE IF EXISTS candidates",
            ()
        ).await?;
        Ok(())
    }

    pub async fn insert(&self, description: &str, embedding: &[f32]) -> Result<()> {
        //convert &f[32] to Blob(Vec<u8>)
        let embedding_bytes = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                embedding.len() * std::mem::size_of::<f32>(),
            ).to_vec()
        };
        self.conn.execute(
            "INSERT INTO candidates (description, embedding) VALUES (?, ?)",
            (description, embedding_bytes)
        ).await?;
        Ok(())
    }
    pub async fn vector_search(&self, embedding: &[f32]) -> Result<Vec<(String, f32)>> {
        let embedding_str = embedding.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", ");
        let sql = format!(
            "SELECT description, 
                    vector_extract(embedding), 
                    vector_distance_cos(embedding, vector32('[{}]')) AS distance 
            FROM candidates
            ORDER BY distance ASC
            LIMIT 5",
            embedding_str
        );
        let mut rows = self.conn.query(&sql, ()).await?;
        let mut results = Vec::new();
        while let Some(row) = rows.next().await? {
            let description_val = row.get_value(0)?;
            let distance_val = row.get_value(2)?;
            let description = match description_val {
                turso::Value::Text(ref s) => s.clone(),
                _ => String::new(),
            };
            let distance = match distance_val {
                turso::Value::Real(f) => f as f32,
                turso::Value::Integer(i) => i as f32,
                _ => 0.0,
            };
            results.push((description, distance));
        }
        Ok(results)
    }

}