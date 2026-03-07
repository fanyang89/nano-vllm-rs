use anyhow::{Result, ensure};
use burn::tensor::{Int, Tensor, TensorData};
use burn_dispatch::{Dispatch, DispatchDevice};

pub struct RotaryEmbedding {
    head_dim: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        _device: &DispatchDevice,
    ) -> Result<Self> {
        ensure!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        Ok(Self {
            head_dim,
            max_position_embeddings,
            rope_theta,
        })
    }

    pub fn forward(
        &self,
        positions: &Tensor<Dispatch, 1, Int>,
        q: &Tensor<Dispatch, 3>,
        k: &Tensor<Dispatch, 3>,
    ) -> Result<(Tensor<Dispatch, 3>, Tensor<Dispatch, 3>)> {
        let q_shape = q.shape().as_slice().to_vec();
        let k_shape = k.shape().as_slice().to_vec();
        ensure!(q_shape.len() == 3 && k_shape.len() == 3, "q/k must be rank-3");
        ensure!(q_shape[0] == k_shape[0], "q/k token dim mismatch");
        ensure!(q_shape[2] == self.head_dim && k_shape[2] == self.head_dim, "head_dim mismatch");

        let n = q_shape[0];
        let qh = q_shape[1];
        let kh = k_shape[1];
        let hd = self.head_dim;
        let half = hd / 2;

        let pos = positions.to_data().to_vec::<i32>()?;
        ensure!(pos.len() == n, "positions length mismatch");

        let q_data = q.to_data().to_vec::<f32>()?;
        let k_data = k.to_data().to_vec::<f32>()?;
        let mut q_out = q_data.clone();
        let mut k_out = k_data.clone();

        for t in 0..n {
            let p = pos[t].max(0) as usize;
            let p = p.min(self.max_position_embeddings.saturating_sub(1)) as f64;
            for i in 0..half {
                let inv_freq = 1.0f64 / self.rope_theta.powf((2.0 * i as f64) / hd as f64);
                let angle = p * inv_freq;
                let cos = angle.cos() as f32;
                let sin = angle.sin() as f32;

                for h in 0..qh {
                    let base = (t * qh + h) * hd;
                    let x1 = q_data[base + i];
                    let x2 = q_data[base + i + half];
                    q_out[base + i] = x1 * cos - x2 * sin;
                    q_out[base + i + half] = x2 * cos + x1 * sin;
                }
                for h in 0..kh {
                    let base = (t * kh + h) * hd;
                    let x1 = k_data[base + i];
                    let x2 = k_data[base + i + half];
                    k_out[base + i] = x1 * cos - x2 * sin;
                    k_out[base + i + half] = x2 * cos + x1 * sin;
                }
            }
        }

        let device = q.device();
        Ok((
            Tensor::<Dispatch, 3>::from_data(TensorData::new(q_out, [n, qh, hd]), &device),
            Tensor::<Dispatch, 3>::from_data(TensorData::new(k_out, [n, kh, hd]), &device),
        ))
    }
}
