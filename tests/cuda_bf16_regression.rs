#[cfg(feature = "cuda")]
mod tests {
    use nano_vllm_rs::{LLMEngine, RuntimeDevice, SamplingParams};

    #[test]
    #[ignore = "requires local CUDA model files and bf16-capable GPU"]
    fn cuda_bf16_greedy_hello_regression() {
        let mut engine =
            LLMEngine::new("./models/Qwen3-0.6B", RuntimeDevice::Cuda).expect("create engine");
        let params = SamplingParams::new(1.0, 8, false, false);

        let outputs = engine
            .generate(&["Hello"], &params, false)
            .expect("generate text");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].text, "Hello! How can I assist you today");
    }
}
