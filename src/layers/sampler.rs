use anyhow::Result;
use candle_core::{DType, Tensor};

/// Temperature-based token sampling using the Gumbel-max trick.
///
/// logits / temperature + Gumbel(0,1) -> argmax
pub fn sample(logits: &Tensor, temperatures: &Tensor, do_sample: bool) -> Result<Vec<u32>> {
    // Fast path for deterministic decode.
    if !do_sample {
        let token_ids = logits
            .to_dtype(DType::F32)?
            .argmax(candle_core::D::Minus1)?;
        return Ok(token_ids.to_vec1()?);
    }

    let device = logits.device();
    let shape = logits.dims().to_vec();

    // logits: (batch, vocab) -> f32
    let logits = logits.to_dtype(DType::F32)?;

    // temperatures: (batch,) -> (batch, 1)
    let temps = temperatures.to_dtype(DType::F32)?.unsqueeze(1)?;

    // Scale logits by temperature
    let scaled = logits.broadcast_div(&temps)?;

    // Gumbel-max trick: argmax(logits + gumbel_noise)
    // where gumbel_noise = -log(-log(U)), U ~ Uniform(0,1).
    let uniform = Tensor::rand(0.0f32, 1.0, shape.as_slice(), device)?;
    let gumbel_noise = uniform
        .clamp(1e-10, 1.0 - 1e-10)?
        .log()?
        .neg()?
        .log()?
        .neg()?;
    let scores = scaled.broadcast_add(&gumbel_noise)?;
    let token_ids = scores.argmax(candle_core::D::Minus1)?;

    let token_ids: Vec<u32> = token_ids.to_vec1()?;
    Ok(token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_sample_shape() {
        let device = Device::Cpu;
        let batch = 3;
        let vocab = 100;

        let logits = Tensor::randn(0.0f32, 1.0, (batch, vocab), &device).unwrap();
        let temps = Tensor::from_vec(vec![0.5f32, 1.0, 1.5], batch, &device).unwrap();

        let tokens = sample(&logits, &temps, true).unwrap();
        assert_eq!(tokens.len(), batch);
        for &t in &tokens {
            assert!((t as usize) < vocab);
        }
    }

    #[test]
    fn test_low_temperature_greedy() {
        let device = Device::Cpu;
        // With very low temperature, should act like argmax
        let logits = Tensor::from_vec(vec![0.0f32, 0.0, 10.0, 0.0], (1, 4), &device).unwrap();
        let temps = Tensor::from_vec(vec![0.001f32], 1, &device).unwrap();

        // Run multiple times — with temp near 0, should almost always pick index 2
        let mut picked_2 = 0;
        for _ in 0..10 {
            let tokens = sample(&logits, &temps, true).unwrap();
            if tokens[0] == 2 {
                picked_2 += 1;
            }
        }
        assert!(picked_2 >= 9, "low temp should be near-greedy");
    }

    #[test]
    fn test_greedy_mode() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 1.0, 10.0, 2.0], (1, 4), &device).unwrap();
        let temps = Tensor::from_vec(vec![1.0f32], 1, &device).unwrap();
        let tokens = sample(&logits, &temps, false).unwrap();
        assert_eq!(tokens, vec![2]);
    }
}
