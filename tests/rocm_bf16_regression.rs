#[cfg(feature = "rocm")]
mod tests {
    use nano_vllm_rs::{LLMEngine, RuntimeDevice, SamplingParams};

    #[test]
    #[ignore = "requires local ROCm model files"]
    fn rocm_bf16_greedy_hello_regression() {
        let mut engine =
            LLMEngine::new("./models/Qwen3-0.6B", RuntimeDevice::Rocm).expect("create engine");
        let params = SamplingParams::new(1.0, 8, false, false);

        let outputs = engine
            .generate(&["Hello"], &params, false)
            .expect("generate text");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].text, "Hello! How can I assist you today");
    }
}
