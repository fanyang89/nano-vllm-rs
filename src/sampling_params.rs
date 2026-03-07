/// Parameters controlling token sampling during generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub max_tokens: usize,
    pub ignore_eos: bool,
    pub do_sample: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_tokens: 256,
            ignore_eos: false,
            do_sample: true,
        }
    }
}

impl SamplingParams {
    pub fn new(temperature: f32, max_tokens: usize, ignore_eos: bool, do_sample: bool) -> Self {
        assert!(temperature > 1e-10, "temperature must be > 0");
        Self {
            temperature,
            max_tokens,
            ignore_eos,
            do_sample,
        }
    }
}
