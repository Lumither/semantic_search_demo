use std::fs::File;
use std::io::Write;
use std::str::FromStr;
use std::{io, time};

use anyhow::{Error, Result};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo};
use indicatif::ProgressBar;
use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, Distance::Cosine, PointStruct, Query, QueryPointsBuilder,
        UpsertPointsBuilder, VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::{PaddingParams, Tokenizer};

const BATCH_SIZE: usize = 32;
const MODEL_DIM: u64 = 384;
const MODEL_ID: &str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Use CPU
    #[arg(short, long, default_value = "false")]
    cpu: bool,

    /// Qdrant database uri
    #[arg(short, long, default_value = "http://localhost:6334")]
    db_uri: String,

    /// Initialize Qdrant database
    #[arg(short, long, default_value = "false")]
    init_db: bool,

    /// Embed dataset
    #[arg(short, long, default_value = "false")]
    embed: bool,

    /// Query REPL
    #[arg(short, long, default_value = "false")]
    query: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct Record {
    id: u64,
    title: String,
    content: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let client = Qdrant::from_url(&args.db_uri).build()?;

    if args.init_db {
        client
            .create_collection(
                CreateCollectionBuilder::new("RMDaily")
                    .vectors_config(VectorParamsBuilder::new(MODEL_DIM, Cosine)),
            )
            .await?;
    }

    let device = if args.cpu {
        Device::Cpu
    } else if cuda_is_available() {
        Device::new_cuda(0)?
    } else if metal_is_available() {
        Device::new_metal(0)?
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!("Running on CPU, to run on GPU(metal), build with `--features metal`");
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build with `--features cuda`");
        }
        Device::Cpu
    };

    let (model, mut tokenizer) = build_model_and_tokenizer(&device, MODEL_ID.to_string())?;

    if args.embed {
        let mut recs: Vec<Record> = Vec::new();
        let csv_path = "./data/RMDaily.csv";
        let file = File::open(csv_path)?;
        let mut rdr = csv::Reader::from_reader(file);
        for result in rdr.records() {
            let record = result?;
            recs.push(Record {
                id: u64::from_str(&record.get(0).unwrap().replace("-", ""))?,
                title: record.get(2).unwrap().to_string(),
                content: record.get(3).unwrap().to_string(),
            });
        }
        embed(&model, &mut tokenizer, &recs, &client).await?;
    }

    if args.query {
        loop {
            print!("query> ");
            io::stdout().flush()?;

            let mut buffer = String::new();
            io::stdin().read_line(&mut buffer)?;
            let timer = time::Instant::now();

            let embedding = sentence2vec(&buffer, &model, &mut tokenizer)?
                .get(0)?
                .to_vec1::<f32>()?;

            let res = client
                .query(
                    QueryPointsBuilder::new("RMDaily")
                        .query(Query::new_nearest(embedding))
                        .limit(3)
                        .with_payload(true),
                )
                .await?;

            res.result.iter().for_each(|r| {
                println!(
                    "{:?} ({})\n<{}>\n{}\n",
                    r.clone().id.unwrap().point_id_options.unwrap(),
                    r.score,
                    r.payload.get("title").unwrap(),
                    r.payload.get("content").unwrap()
                );
            });

            println!("Elapsed: {:?}", timer.elapsed());
        }
    }

    Ok(())
}

fn build_model_and_tokenizer(device: &Device, model_id: String) -> Result<(BertModel, Tokenizer)> {
    let repo = Repo::model(model_id);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (config, tokenizer, weights)
    };
    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, device)? };
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

async fn embed(
    model: &BertModel,
    tokenizer: &mut Tokenizer,
    records: &[Record],
    db: &Qdrant,
) -> Result<()> {
    let timer = time::Instant::now();

    let bar = ProgressBar::new(records.len() as u64);

    for batch in records.chunks(BATCH_SIZE).collect::<Vec<_>>() {
        let embedding = batch2vec(
            &batch
                .iter()
                .map(|it| it.content.as_ref())
                .collect::<Vec<_>>(),
            model,
            tokenizer,
        )?;

        let points = (0..embedding.dim(0)?)
            .map(|idx| -> Result<_> {
                let vec = embedding.get(idx)?.to_vec1::<f32>()?;
                let rec = &batch[idx];
                Ok(PointStruct::new(
                    rec.id,
                    vec,
                    Payload::try_from(json!({
                        "title": rec.title,
                        "content": rec.content
                    }))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        db.upsert_points(UpsertPointsBuilder::new("RMDaily", points))
            .await?;

        bar.inc(batch.len() as u64);
    }
    bar.finish();

    println!("Elapsed: {:?}", timer.elapsed());
    Ok(())
}

fn sentence2vec(sentence: &str, model: &BertModel, tokenizer: &mut Tokenizer) -> Result<Tensor> {
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(Error::msg)?;
    let tokens = tokenizer
        .encode(sentence, true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], &model.device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let embedding = model.forward(&token_ids, &token_type_ids, None)?;

    let (_n_sentence, n_tokens, _hidden_size) = embedding.dims3()?;
    let embedding = (embedding.sum(1)? / (n_tokens as f64))?;
    Ok(embedding)
}

fn batch2vec(batch: &[&str], model: &BertModel, tokenizer: &mut Tokenizer) -> Result<Tensor> {
    if let Some(padding_params) = tokenizer.get_padding_mut() {
        padding_params.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let padding_params = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding_params));
    }
    let tokens = tokenizer
        .encode_batch(batch.to_vec(), true)
        .map_err(Error::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &model.device)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &model.device)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    let attention_mask = Tensor::stack(&attention_mask, 0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let embeddings = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    Ok(embeddings)
}
