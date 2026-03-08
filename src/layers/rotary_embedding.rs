use anyhow::{Result, ensure};
use burn::tensor::{DType, Element, Int, Tensor, TensorData, backend::Backend};

pub struct RotaryEmbedding<B: Backend> {
    head_dim: usize,
    max_position_embeddings: usize,
    inv_freq: Tensor<B, 1>,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        device: &B::Device,
    ) -> Result<Self> {
        ensure!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let denom = rope_theta.powf((2.0 * i as f64) / head_dim as f64);
                (1.0 / denom) as f32
            })
            .collect();
        Ok(Self {
            head_dim,
            max_position_embeddings,
            inv_freq: Tensor::<B, 1>::from_data(TensorData::new(inv_freq, [half]), device),
        })
    }

    pub fn forward(
        &self,
        positions: &Tensor<B, 1, Int>,
        q: &Tensor<B, 3>,
        k: &Tensor<B, 3>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        let q_shape = q.shape().as_slice().to_vec();
        let k_shape = k.shape().as_slice().to_vec();
        ensure!(
            q_shape.len() == 3 && k_shape.len() == 3,
            "q/k must be rank-3"
        );
        ensure!(q_shape[0] == k_shape[0], "q/k token dim mismatch");
        ensure!(
            q_shape[2] == self.head_dim && k_shape[2] == self.head_dim,
            "head_dim mismatch"
        );

        let n = q_shape[0];
        let hd = self.head_dim;
        let half = hd / 2;
        let native_dtype = <B::FloatElem as Element>::dtype();

        let pos = positions
            .clone()
            .clamp(0, self.max_position_embeddings.saturating_sub(1) as i32)
            .float()
            .cast(DType::F32)
            .reshape([n, 1]);
        let inv_freq = self.inv_freq.clone().cast(DType::F32);
        let angles = pos.matmul(inv_freq.unsqueeze_dim::<2>(0));
        let cos = angles.clone().cos().unsqueeze_dim::<3>(1); // [n,1,half]
        let sin = angles.sin().unsqueeze_dim::<3>(1); // [n,1,half]

        let q = q.clone().cast(DType::F32);
        let k = k.clone().cast(DType::F32);
        let q1 = q.clone().slice([0..n, 0..q_shape[1], 0..half]);
        let q2 = q.clone().slice([0..n, 0..q_shape[1], half..hd]);
        let k1 = k.clone().slice([0..n, 0..k_shape[1], 0..half]);
        let k2 = k.clone().slice([0..n, 0..k_shape[1], half..hd]);

        let q_rot_1 = q1.clone() * cos.clone() - q2.clone() * sin.clone();
        let q_rot_2 = q2 * cos.clone() + q1 * sin.clone();
        let k_rot_1 = k1.clone() * cos.clone() - k2.clone() * sin.clone();
        let k_rot_2 = k2 * cos + k1 * sin;

        let q_out = Tensor::<B, 3>::cat(vec![q_rot_1, q_rot_2], 2).cast(native_dtype);
        let k_out = Tensor::<B, 3>::cat(vec![k_rot_1, k_rot_2], 2).cast(native_dtype);
        Ok((q_out, k_out))
    }
}
