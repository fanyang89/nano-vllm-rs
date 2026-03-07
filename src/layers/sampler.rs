use anyhow::Result;
use candle_core::{DType, Tensor};

/// Temperature-based token sampling using the Gumbel-max trick.
///
/// logits / temperature -> softmax -> divide by Exp(1) samples -> argmax
pub fn sample(logits: &Tensor, temperatures: &Tensor) -> Result<Vec<u32>> {
    let device = logits.device();
    let shape = logits.dims().to_vec();

    // logits: (batch, vocab) -> f32
    let logits = logits.to_dtype(DType::F32)?;

    // temperatures: (batch,) -> (batch, 1)
    let temps = temperatures
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;

    // Scale logits by temperature
    let scaled = logits.broadcast_div(&temps)?;

    // Softmax
    let probs = candle_nn::ops::softmax_last_dim(&scaled)?;

    // Gumbel-max trick: probs / Exp(1) samples, then argmax
    // Exp(1) = -log(Uniform(0,1))
    let uniform = Tensor::rand(0.0f32, 1.0, shape.as_slice(), device)?;
    let exp_samples = uniform.clamp(1e-10, 1.0)?.log()?.neg()?;
    let gumbel = probs.div(&exp_samples)?;
    let token_ids = gumbel.argmax(candle_core::D::Minus1)?;

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

        let tokens = sample(&logits, &temps).unwrap();
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
            let tokens = sample(&logits, &temps).unwrap();
            if tokens[0] == 2 {
                picked_2 += 1;
            }
        }
        assert!(picked_2 >= 9, "low temp should be near-greedy");
    }
}
