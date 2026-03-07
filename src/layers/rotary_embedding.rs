use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Rotary Position Embedding (RoPE).
///
/// Precomputes cos/sin cache and applies rotation to query and key tensors.
pub struct RotaryEmbedding {
    /// Shape: (max_position, head_dim), contains [cos | sin] concatenated
    cos_sin_cache: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_position: usize,
        rope_theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        // inv_freq = 1.0 / (theta ^ (arange(0, dim, 2) / dim))
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?;

        // t = arange(max_position)
        let t: Vec<f32> = (0..max_position).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, max_position, device)?;

        // freqs = outer(t, inv_freq) -> (max_position, half_dim)
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        // cos_sin_cache = cat(cos(freqs), sin(freqs), dim=-1) -> (max_position, head_dim)
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        let cos_sin_cache = Tensor::cat(&[&cos, &sin], 1)?;

        Ok(Self { cos_sin_cache })
    }

    /// Apply rotary embedding to query and key.
    ///
    /// - positions: (N,) i64
    /// - query: (N, num_heads, head_dim)
    /// - key: (N, num_kv_heads, head_dim)
    ///
    /// Returns (rotated_query, rotated_key).
    pub fn forward(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // cos_sin: (N, 1, head_dim)
        let cos_sin = self
            .cos_sin_cache
            .index_select(positions, 0)?
            .unsqueeze(1)?;
        let head_dim = cos_sin.dim(2)?;
        let half = head_dim / 2;

        let cos = cos_sin.narrow(2, 0, half)?;
        let sin = cos_sin.narrow(2, half, half)?;

        let q_rot = apply_rotary_emb(query, &cos, &sin)?;
        let k_rot = apply_rotary_emb(key, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }
}

/// Apply rotary embedding: split x into halves, rotate.
/// x: (..., head_dim), cos/sin: (..., head_dim/2) broadcast-compatible
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let half = x.dim(candle_core::D::Minus1)? / 2;

    let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
    let x2 = x.narrow(candle_core::D::Minus1, half, half)?;

    let cos = cos.to_dtype(DType::F32)?;
    let sin = sin.to_dtype(DType::F32)?;

    let y1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let y2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;

    let result = Tensor::cat(&[&y1, &y2], candle_core::D::Minus1)?.to_dtype(dtype)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rotary_embedding_shape() {
        let device = Device::Cpu;
        let head_dim = 64;
        let rope = RotaryEmbedding::new(head_dim, 128, 10000.0, &device).unwrap();

        let n = 5;
        let num_heads = 4;
        let num_kv_heads = 2;
        let positions = Tensor::arange(0u32, n as u32, &device)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap();
        let q = Tensor::zeros((n, num_heads, head_dim), DType::F32, &device).unwrap();
        let k = Tensor::zeros((n, num_kv_heads, head_dim), DType::F32, &device).unwrap();

        let (q_rot, k_rot) = rope.forward(&positions, &q, &k).unwrap();
        assert_eq!(q_rot.dims(), &[n, num_heads, head_dim]);
        assert_eq!(k_rot.dims(), &[n, num_kv_heads, head_dim]);
    }
}
