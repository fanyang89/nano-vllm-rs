use burn::tensor::backend::Backend;

#[cfg(feature = "cpu")]
pub type CpuBackend = burn_ndarray::NdArray<f32, i32, i8>;

#[cfg(feature = "rocm")]
pub type RocmBackend = burn_rocm::Rocm<half::bf16, i32, u8>;

pub trait NanoBackend: Backend<IntElem = i32> {}

impl<B> NanoBackend for B where B: Backend<IntElem = i32> {}
