use anyhow::{Context, Result, ensure};
use burn::tensor::{DType, Tensor, backend::Backend};
use rand::Rng;

/// Temperature-based token sampling using the Gumbel-max trick.
pub fn sample<B: Backend<IntElem = i32>>(
    logits: &Tensor<B, 2>,
    temperatures: Option<&Tensor<B, 1>>,
    do_sample: bool,
) -> Result<Vec<u32>> {
    let shape = logits.shape();
    let dims = shape.as_slice();
    ensure!(dims.len() == 2, "logits must be rank-2");
    let batch = dims[0];
    let vocab = dims[1];

    if !do_sample {
        if batch == 1 {
            let token_id = logits.clone().argmax(1).into_scalar();
            return Ok(vec![token_id as u32]);
        }

        let token_ids = logits.clone().argmax(1).to_data();
        return match token_ids.dtype {
            DType::I32 => Ok(token_ids
                .to_vec::<i32>()
                .context("failed to read argmax token ids")?
                .into_iter()
                .map(|id| id as u32)
                .collect()),
            DType::I64 => Ok(token_ids
                .to_vec::<i64>()
                .context("failed to read argmax token ids")?
                .into_iter()
                .map(|id| id as u32)
                .collect()),
            dtype => anyhow::bail!("unsupported argmax dtype: {dtype:?}"),
        };
    }

    let logits = logits
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .context("failed to read logits tensor data")?;
    let temperatures = temperatures
        .context("sampling requires temperatures")?
        .clone()
        .cast(DType::F32)
        .to_data()
        .to_vec::<f32>()
        .context("failed to read temperature tensor data")?;
    ensure!(temperatures.len() == batch, "temperature batch mismatch");

    let mut out = Vec::with_capacity(batch);
    let mut rng = rand::rng();

    for b in 0..batch {
        let row = &logits[b * vocab..(b + 1) * vocab];
        let t = temperatures[b].max(1e-6);

        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for (i, &logit) in row.iter().enumerate() {
            let score = if do_sample {
                let u: f32 = rng.random::<f32>().clamp(1e-10, 1.0 - 1e-10);
                let gumbel = -(-u.ln()).ln();
                logit / t + gumbel
            } else {
                logit
            };
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        out.push(best_idx as u32);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use burn::tensor::TensorData;

    #[test]
    fn test_greedy_mode() {
        let device = Default::default();
        let logits = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(vec![0.0f32, 1.0, 10.0, 2.0], [1, 4]),
            &device,
        );
        let temps = Tensor::<CpuBackend, 1>::from_data(TensorData::new(vec![1.0f32], [1]), &device);
        let tokens = sample(&logits, Some(&temps), false).unwrap();
        assert_eq!(tokens, vec![2]);
    }
}
