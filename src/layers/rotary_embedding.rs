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
        let hd = self.head_dim;
        let half = hd / 2;
        let device = q.device();
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let denom = self.rope_theta.powf((2.0 * i as f64) / hd as f64);
                (1.0 / denom) as f32
            })
            .collect();
        let inv_freq = Tensor::<Dispatch, 1>::from_data(TensorData::new(inv_freq, [half]), &device);

        // Clamp positions to configured max range before angle construction.
        let pos = positions
            .clone()
            .clamp(0, self.max_position_embeddings.saturating_sub(1) as i32)
            .float()
            .reshape([n, 1]);
        let angles = pos.matmul(inv_freq.unsqueeze_dim::<2>(0)); // [n, half]
        let cos = angles.clone().cos().unsqueeze_dim::<3>(1); // [n,1,half]
        let sin = angles.sin().unsqueeze_dim::<3>(1); // [n,1,half]

        let q1 = q.clone().slice([0..n, 0..q_shape[1], 0..half]);
        let q2 = q.clone().slice([0..n, 0..q_shape[1], half..hd]);
        let k1 = k.clone().slice([0..n, 0..k_shape[1], 0..half]);
        let k2 = k.clone().slice([0..n, 0..k_shape[1], half..hd]);

        let q_rot_1 = q1.clone() * cos.clone() - q2.clone() * sin.clone();
        let q_rot_2 = q2 * cos.clone() + q1 * sin.clone();
        let k_rot_1 = k1.clone() * cos.clone() - k2.clone() * sin.clone();
        let k_rot_2 = k2 * cos + k1 * sin;

        let q_out = Tensor::<Dispatch, 3>::cat(vec![q_rot_1, q_rot_2], 2);
        let k_out = Tensor::<Dispatch, 3>::cat(vec![k_rot_1, k_rot_2], 2);
        Ok((q_out, k_out))
    }
}
