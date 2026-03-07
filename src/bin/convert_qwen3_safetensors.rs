use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "convert-qwen3-safetensors",
    about = "Build a deterministic Qwen3 weight remap plan for Burn migration"
)]
struct Args {
    /// Source model directory containing *.safetensors.
    #[arg(long)]
    model: PathBuf,
    /// Output JSON file containing remap metadata and concat groups.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Serialize)]
struct TensorEntry {
    source_key: String,
    source_file: String,
    dtype: String,
    shape: Vec<usize>,
    target_key: String,
}

#[derive(Debug, Serialize)]
struct ConcatGroup {
    target_key: String,
    dim: usize,
    sources: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ConversionPlan {
    model_type: String,
    files_scanned: usize,
    tensor_count: usize,
    tensors: Vec<TensorEntry>,
    concat_groups: Vec<ConcatGroup>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let plan = build_plan(&args.model)?;
    let json = serde_json::to_string_pretty(&plan)?;
    fs::write(&args.out, json)
        .with_context(|| format!("failed to write plan file: {}", args.out.display()))?;
    println!(
        "Wrote conversion plan with {} tensors to {}",
        plan.tensor_count,
        args.out.display()
    );
    Ok(())
}

fn build_plan(model_dir: &Path) -> Result<ConversionPlan> {
    if !model_dir.is_dir() {
        bail!("model path is not a directory: {}", model_dir.display());
    }

    let mut files: Vec<PathBuf> = fs::read_dir(model_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();

    if files.is_empty() {
        bail!("no .safetensors files found in {}", model_dir.display());
    }

    let mut tensors = Vec::new();
    let mut seen = BTreeSet::new();

    for file in &files {
        let bytes = fs::read(file).with_context(|| format!("failed to read {}", file.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("invalid safetensors: {}", file.display()))?;
        for key in st.names() {
            if !seen.insert(key.to_string()) {
                continue;
            }
            let t = st
                .tensor(key)
                .with_context(|| format!("failed to load tensor key {key}"))?;
            tensors.push(TensorEntry {
                source_key: key.to_string(),
                source_file: file
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| file.display().to_string()),
                dtype: format!("{:?}", t.dtype()),
                shape: t.shape().to_vec(),
                target_key: remap_key(key),
            });
        }
    }

    tensors.sort_by(|a, b| a.source_key.cmp(&b.source_key));

    let mut by_layer_qkv: BTreeMap<String, (String, String, String)> = BTreeMap::new();
    let mut by_layer_mlp: BTreeMap<String, (String, String)> = BTreeMap::new();

    for t in &tensors {
        if let Some(layer) = t.source_key.strip_suffix(".self_attn.q_proj.weight") {
            let e = by_layer_qkv.entry(layer.to_string()).or_default();
            e.0 = t.source_key.clone();
        } else if let Some(layer) = t.source_key.strip_suffix(".self_attn.k_proj.weight") {
            let e = by_layer_qkv.entry(layer.to_string()).or_default();
            e.1 = t.source_key.clone();
        } else if let Some(layer) = t.source_key.strip_suffix(".self_attn.v_proj.weight") {
            let e = by_layer_qkv.entry(layer.to_string()).or_default();
            e.2 = t.source_key.clone();
        } else if let Some(layer) = t.source_key.strip_suffix(".mlp.gate_proj.weight") {
            let e = by_layer_mlp.entry(layer.to_string()).or_default();
            e.0 = t.source_key.clone();
        } else if let Some(layer) = t.source_key.strip_suffix(".mlp.up_proj.weight") {
            let e = by_layer_mlp.entry(layer.to_string()).or_default();
            e.1 = t.source_key.clone();
        }
    }

    let mut concat_groups = Vec::new();
    for (layer, (q, k, v)) in by_layer_qkv {
        if !(q.is_empty() || k.is_empty() || v.is_empty()) {
            concat_groups.push(ConcatGroup {
                target_key: format!("{layer}.self_attn.qkv_proj.weight"),
                dim: 0,
                sources: vec![q, k, v],
            });
        }
    }
    for (layer, (gate, up)) in by_layer_mlp {
        if !(gate.is_empty() || up.is_empty()) {
            concat_groups.push(ConcatGroup {
                target_key: format!("{layer}.mlp.gate_up_proj.weight"),
                dim: 0,
                sources: vec![gate, up],
            });
        }
    }
    concat_groups.sort_by(|a, b| a.target_key.cmp(&b.target_key));

    Ok(ConversionPlan {
        model_type: "qwen3".to_string(),
        files_scanned: files.len(),
        tensor_count: tensors.len(),
        tensors,
        concat_groups,
    })
}

fn remap_key(source: &str) -> String {
    source.to_string()
}
