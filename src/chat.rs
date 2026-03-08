use crate::config::ModelConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    Plain,
    Qwen3,
}

impl ChatFormat {
    pub fn from_model_config(model_config: &ModelConfig) -> Self {
        if model_config.model_type.as_deref() == Some("qwen3") {
            Self::Qwen3
        } else {
            Self::Plain
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

pub fn format_single_prompt(prompt: &str, format: ChatFormat) -> String {
    match format {
        ChatFormat::Plain => prompt.to_string(),
        ChatFormat::Qwen3 => {
            let messages = vec![ChatMessage::new(ChatRole::User, prompt)];
            render_chat_prompt(&messages, format, true)
        }
    }
}

pub fn render_chat_prompt(
    messages: &[ChatMessage],
    format: ChatFormat,
    enable_thinking: bool,
) -> String {
    match format {
        ChatFormat::Plain => render_plain_chat_prompt(messages),
        ChatFormat::Qwen3 => render_qwen3_chat_prompt(messages, enable_thinking),
    }
}

pub fn strip_think_blocks(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::new();

    loop {
        let Some(start) = remaining.find("<think>") else {
            output.push_str(remaining);
            break;
        };

        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];
        let Some(end) = after_start.find("</think>") else {
            break;
        };
        remaining = &after_start[end + "</think>".len()..];
    }

    output.trim().to_string()
}

fn render_plain_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for message in messages {
        prompt.push_str(message.role.label());
        prompt.push_str(":\n");
        prompt.push_str(&message.content);
        prompt.push_str("\n\n");
    }

    prompt.push_str("assistant:\n");
    prompt
}

fn render_qwen3_chat_prompt(messages: &[ChatMessage], enable_thinking: bool) -> String {
    let mut prompt = String::new();

    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.label());
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }

    prompt.push_str("<|im_start|>assistant\n");
    if enable_thinking {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_think_block_from_reply() {
        let text = "<think>internal</think>\n\nVisible answer";
        assert_eq!(strip_think_blocks(text), "Visible answer");
    }

    #[test]
    fn renders_qwen3_prompt_with_generation_prefix() {
        let messages = vec![
            ChatMessage::new(ChatRole::System, "Be concise."),
            ChatMessage::new(ChatRole::User, "Hello"),
        ];
        let prompt = render_chat_prompt(&messages, ChatFormat::Qwen3, true);

        assert!(prompt.contains("<|im_start|>system\nBe concise.<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }
}
