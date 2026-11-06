package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Parity of every built-in named template of {@link ChatTemplateSource} with llama.cpp's
 * {@code llm_chat_apply_template} (src/llama-chat.cpp:250-939). The conversation is llama.cpp's test
 * conversation (tests/test-chat-template.cpp:355-362); the expected prompts of the styles its tests cover
 * are copied from its {@code expected_output} fields, and the others are written out from the C++ branch
 * of each style.
 */
public class LlamaCppNamedTemplatesTest {

    /** llama.cpp's {@code LLM_CHAT_TEMPLATES} names, in table order (src/llama-chat.cpp:28-83). */
    private static final List<String> LLAMA_CPP_NAMES = Arrays.asList("chatml", "llama2", "llama2-sys",
            "llama2-sys-bos", "llama2-sys-strip", "mistral-v1", "mistral-v3", "mistral-v3-tekken", "mistral-v7",
            "mistral-v7-tekken", "phi3", "phi4", "falcon3", "zephyr", "monarch", "gemma", "orion", "openchat",
            "vicuna", "vicuna-orca", "deepseek", "deepseek2", "deepseek3", "deepseek-ocr", "command-r", "llama3",
            "chatglm3", "chatglm4", "glmedge", "minicpm", "exaone3", "exaone4", "exaone-moe", "rwkv-world",
            "granite", "granite-4.0", "granite-4.1", "gigachat", "megrez", "yandex", "bailing", "bailing-think",
            "bailing2", "llama4", "smolvlm", "hunyuan-moe", "gpt-oss", "hunyuan-dense", "hunyuan-vl", "kimi-k2",
            "seed_oss", "grok-2", "pangu-embedded", "solar-open");

    private static Map<String, String> msg(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    private static final List<Map<String, String>> CONVERSATION = Collections.unmodifiableList(Arrays.asList(
            msg("system", "You are a helpful assistant"), msg("user", "Hello"), msg("assistant", "Hi there"),
            msg("user", "Who are you"), msg("assistant", "   I am an assistant   "), msg("user", "Another question")));

    private static String render(String name, List<Map<String, String>> messages, boolean addGenerationPrompt) {
        ChatFormatter formatter = new ChatFormatter(ChatTemplateSource.resolve(name), "", "", false, false, "llama");
        return formatter.format(ChatRequest.builder().messages(messages)
                .addGenerationPrompt(addGenerationPrompt).build());
    }

    /** Collects the mismatches of one table so a failure lists every style that differs. */
    private static final class Expectations {
        private final List<String> failures = new ArrayList<>();
        private int checked;

        void expect(String name, String expected) {
            withMessages(name, CONVERSATION, expected);
        }

        @SafeVarargs
        final void withExtra(String name, String expected, Map<String, String>... extra) {
            List<Map<String, String>> messages = new ArrayList<>(CONVERSATION);
            messages.addAll(Arrays.asList(extra));
            withMessages(name, messages, expected);
        }

        private void withMessages(String name, List<Map<String, String>> messages, String expected) {
            checked++;
            String actual = render(name, messages, true);
            if (!expected.equals(actual)) {
                failures.add(name + "\n  expected: " + escape(expected) + "\n  actual:   " + escape(actual));
            }
        }

        void verify(int expectedCount) {
            assertEquals("table size", expectedCount, checked);
            assertTrue(String.join("\n", failures), failures.isEmpty());
        }

        private static String escape(String s) {
            return s.replace("\n", "\\n");
        }
    }

    @Test
    public void stylesCoveredByLlamaCppTestsRenderItsExpectedPrompts() {
        Expectations t = new Expectations();
        t.expect("chatml", "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n"
                + "Hello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nWho are you<|im_end|>"
                + "\n<|im_start|>assistant\n   I am an assistant   <|im_end|>\n<|im_start|>user\nAnother "
                + "question<|im_end|>\n<|im_start|>assistant\n");
        t.expect("llama2", "[INST] You are a helpful assistant\nHello [/INST]Hi there</s>[INST] Who are you [/INST]   "
                + "I am an assistant   </s>[INST] Another question [/INST]");
        t.expect("llama2-sys-bos", "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST]Hi there</s>"
                + "<s>[INST] Who are you [/INST]   I am an assistant   </s><s>[INST] Another question [/INST]");
        t.expect("llama2-sys-strip", "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST]Hi "
                + "there</s>[INST] Who are you [/INST]I am an assistant</s>[INST] Another question [/INST]");
        t.expect("monarch", "system\nYou are a helpful assistant</s>\n<s>user\nHello</s>\n<s>assistant\nHi there</s>\n"
                + "<s>user\nWho are you</s>\n<s>assistant\n   I am an assistant   </s>\n<s>user\nAnother question</s>\n"
                + "<s>assistant\n");
        t.expect("gemma", "<start_of_turn>user\nYou are a helpful assistant\n\nHello<end_of_turn>\n<start_of_turn>"
                + "model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\n"
                + "I am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>"
                + "model\n");
        t.expect("orion", "Human: You are a helpful assistant\n\nHello\n\nAssistant: </s>Hi there</s>Human: Who are "
                + "you\n\nAssistant: </s>   I am an assistant   </s>Human: Another question\n\nAssistant: </s>");
        t.expect("openchat", "You are a helpful assistant<|end_of_turn|>GPT4 Correct User: Hello<|end_of_turn|>GPT4 "
                + "Correct Assistant: Hi there<|end_of_turn|>GPT4 Correct User: Who are you<|end_of_turn|>GPT4 "
                + "Correct Assistant:    I am an assistant   <|end_of_turn|>GPT4 Correct User: Another "
                + "question<|end_of_turn|>GPT4 Correct Assistant:");
        t.expect("deepseek", "You are a helpful assistant### Instruction:\nHello\n### Response:\nHi there\n<|EOT|>\n"
                + "### Instruction:\nWho are you\n### Response:\n   I am an assistant   \n<|EOT|>\n### Instruction:\n"
                + "Another question\n### Response:\n");
        t.expect("vicuna", "You are a helpful assistant\n\nUSER: Hello\nASSISTANT: Hi there</s>\nUSER: Who are you\n"
                + "ASSISTANT:    I am an assistant   </s>\nUSER: Another question\nASSISTANT:");
        t.expect("vicuna-orca", "SYSTEM: You are a helpful assistant\nUSER: Hello\nASSISTANT: Hi there</s>\nUSER: Who "
                + "are you\nASSISTANT:    I am an assistant   </s>\nUSER: Another question\nASSISTANT:");
        t.expect("command-r", "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful assistant<|END_OF_TURN_TOKEN|>"
                + "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|>"
                + "<|CHATBOT_TOKEN|>Hi there<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Who are "
                + "you<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>I am an "
                + "assistant<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Another "
                + "question<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
        t.expect("llama3", "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|>"
                + "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>"
                + "assistant<|end_header_id|>\n\nHi there<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWho "
                + "are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI am an assistant<|eot_id|>"
                + "<|start_header_id|>user<|end_header_id|>\n\nAnother question<|eot_id|><|start_header_id|>"
                + "assistant<|end_header_id|>\n\n");
        t.expect("phi3", "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\nHello<|end|>\n<|assistant|>\nHi "
                + "there<|end|>\n<|user|>\nWho are you<|end|>\n<|assistant|>\n   I am an assistant   <|end|>\n<|user|>"
                + "\nAnother question<|end|>\n<|assistant|>\n");
        t.expect("chatglm3", "[gMASK]sop<|system|>\n You are a helpful assistant<|user|>\n Hello<|assistant|>\n Hi "
                + "there<|user|>\n Who are you<|assistant|>\n    I am an assistant   <|user|>\n Another "
                + "question<|assistant|>");
        t.expect("chatglm4", "[gMASK]<sop><|system|>\nYou are a helpful assistant<|user|>\nHello<|assistant|>\nHi "
                + "there<|user|>\nWho are you<|assistant|>\n   I am an assistant   <|user|>\nAnother "
                + "question<|assistant|>\n");
        t.expect("glmedge", "<|system|>\nYou are a helpful assistant<|user|>\nHello<|assistant|>\nHi there<|user|>\n"
                + "Who are you<|assistant|>\n   I am an assistant   <|user|>\nAnother question<|assistant|>");
        t.expect("minicpm", "You are a helpful assistant<用户>Hello<AI>Hi there<用户>Who are you<AI>I am an assistant<用户>"
                + "Another question<AI>");
        t.expect("deepseek2", "You are a helpful assistant\n\nUser: Hello\n\nAssistant: Hi there<｜end▁of▁sentence｜>"
                + "User: Who are you\n\nAssistant:    I am an assistant   <｜end▁of▁sentence｜>User: Another question\n\n"
                + "Assistant:");
        t.expect("granite", "<|start_of_role|>system<|end_of_role|>You are a helpful assistant<|end_of_text|>\n"
                + "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
                + "Hi there<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Who are you<|end_of_text|>\n"
                + "<|start_of_role|>assistant<|end_of_role|>   I am an assistant   <|end_of_text|>\n<|start_of_role|>"
                + "user<|end_of_role|>Another question<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");
        t.expect("mistral-v1", " [INST] You are a helpful assistant\n\nHello [/INST] Hi there</s> [INST] Who are you "
                + "[/INST]    I am an assistant   </s> [INST] Another question [/INST]");
        t.expect("mistral-v3", "[INST] You are a helpful assistant\n\nHello[/INST] Hi there</s>[INST] Who are "
                + "you[/INST] I am an assistant</s>[INST] Another question[/INST]");
        t.expect("mistral-v3-tekken", "[INST]You are a helpful assistant\n\nHello[/INST]Hi there</s>[INST]Who are "
                + "you[/INST]   I am an assistant   </s>[INST]Another question[/INST]");
        t.expect("mistral-v7", "[SYSTEM_PROMPT] You are a helpful assistant[/SYSTEM_PROMPT][INST] Hello[/INST] Hi "
                + "there</s>[INST] Who are you[/INST]    I am an assistant   </s>[INST] Another question[/INST]");
        t.expect("gigachat", "<s>You are a helpful assistant<|message_sep|>user<|role_sep|>Hello<|message_sep|>"
                + "available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>Hi there<|message_sep|>"
                + "user<|role_sep|>Who are you<|message_sep|>available functions<|role_sep|>[]<|message_sep|>"
                + "assistant<|role_sep|>   I am an assistant   <|message_sep|>user<|role_sep|>Another "
                + "question<|message_sep|>available functions<|role_sep|>[]<|message_sep|>assistant<|role_sep|>");
        t.expect("megrez", "<|role_start|>system<|role_end|>You are a helpful assistant<|turn_end|><|role_start|>"
                + "user<|role_end|>Hello<|turn_end|><|role_start|>assistant<|role_end|>Hi there<|turn_end|>"
                + "<|role_start|>user<|role_end|>Who are you<|turn_end|><|role_start|>assistant<|role_end|>   I am an "
                + "assistant   <|turn_end|><|role_start|>user<|role_end|>Another question<|turn_end|><|role_start|>"
                + "assistant<|role_end|>");
        t.expect("phi4", "<|im_start|>system<|im_sep|>You are a helpful assistant<|im_end|><|im_start|>user<|im_sep|>"
                + "Hello<|im_end|><|im_start|>assistant<|im_sep|>Hi there<|im_end|><|im_start|>user<|im_sep|>Who are "
                + "you<|im_end|><|im_start|>assistant<|im_sep|>   I am an assistant   <|im_end|><|im_start|>"
                + "user<|im_sep|>Another question<|im_end|><|im_start|>assistant<|im_sep|>");
        t.expect("yandex", " Пользователь: Hello\n\n Ассистент: Hi there\n\n Пользователь: Who are you\n\n Ассистент: "
                + "   I am an assistant   \n\n Пользователь: Another question\n\n Ассистент:[SEP]");
        t.expect("bailing", "<role>SYSTEM</role>You are a helpful assistant<role>HUMAN</role>Hello<role>"
                + "ASSISTANT</role>Hi there<role>HUMAN</role>Who are you<role>ASSISTANT</role>   I am an assistant   "
                + "<role>HUMAN</role>Another question<role>ASSISTANT</role>");
        t.expect("seed_oss", "<seed:bos>system\nYou are a helpful assistant<seed:eos><seed:bos>user\nHello<seed:eos>"
                + "<seed:bos>assistant\nHi there<seed:eos><seed:bos>user\nWho are you<seed:eos><seed:bos>assistant\nI "
                + "am an assistant<seed:eos><seed:bos>user\nAnother question<seed:eos><seed:bos>assistant\n");
        t.withExtra("granite", "<|start_of_role|>system<|end_of_role|>You are a helpful assistant<|end_of_text|>\n"
                + "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
                + "Hi there<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Who are you<|end_of_text|>\n"
                + "<|start_of_role|>assistant<|end_of_role|>   I am an assistant   <|end_of_text|>\n<|start_of_role|>"
                + "user<|end_of_role|>Another question<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>What is "
                + "the weather?<|end_of_text|>\n<|start_of_role|>assistant_tool_call<|end_of_role|><|tool_call|>"
                + "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}]<|end_of_text|>\n"
                + "<|start_of_role|>tool_response<|end_of_role|>{\"temperature\": 72}<|end_of_text|>\n"
                + "<|start_of_role|>assistant<|end_of_role|>",
                msg("user", "What is the weather?"),
                msg("assistant_tool_call", "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}]"),
                msg("tool_response", "{\"temperature\": 72}"));
        t.withExtra("granite-4.0", "<|start_of_role|>system<|end_of_role|>You are a helpful assistant<|end_of_text|>\n"
                + "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
                + "Hi there<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Who are you<|end_of_text|>\n"
                + "<|start_of_role|>assistant<|end_of_role|>   I am an assistant   <|end_of_text|>\n<|start_of_role|>"
                + "user<|end_of_role|>Another question<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>What is "
                + "the weather?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|><|tool_call|><tool_call>\n"
                + "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}\n</tool_call><|end_of_text|>\n"
                + "<|start_of_role|>tool_response<|end_of_role|>{\"temperature\": 72}<|end_of_text|>\n"
                + "<|start_of_role|>assistant<|end_of_role|>",
                msg("user", "What is the weather?"),
                msg("assistant_tool_call", "<tool_call>\n{\"name\": \"get_weather\", "
                        + "\"arguments\": {\"location\": \"NYC\"}}\n</tool_call>"),
                msg("tool_response", "{\"temperature\": 72}"));
        t.withExtra("granite-4.1", "<|start_of_role|>system<|end_of_role|>You are a helpful assistant<|end_of_text|>\n"
                + "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
                + "Hi there<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Who are you<|end_of_text|>\n"
                + "<|start_of_role|>assistant<|end_of_role|>   I am an assistant   <|end_of_text|>\n<|start_of_role|>"
                + "user<|end_of_role|>Another question<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>What is "
                + "the weather?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|><|tool_call|><tool_call>\n"
                + "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}\n</tool_call><|end_of_text|>\n"
                + "<|start_of_role|>tool_response<|end_of_role|>{\"temperature\": 72}<|end_of_text|>\n"
                + "<|start_of_role|>assistant<|end_of_role|>",
                msg("user", "What is the weather?"),
                msg("assistant_tool_call", "<tool_call>\n{\"name\": \"get_weather\", "
                        + "\"arguments\": {\"location\": \"NYC\"}}\n</tool_call>"),
                msg("tool_response", "{\"temperature\": 72}"));
        t.verify(33);
    }

    @Test
    public void stylesWithoutLlamaCppTestCasesRenderTheirFormatterOutput() {
        Expectations t = new Expectations();
        t.expect("llama2-sys", "[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST]Hi there</s>"
                + "[INST] Who are you [/INST]   I am an assistant   </s>[INST] Another question [/INST]");
        t.expect("mistral-v7-tekken", "[SYSTEM_PROMPT]You are a helpful assistant[/SYSTEM_PROMPT][INST]Hello[/INST]Hi "
                + "there</s>[INST]Who are you[/INST]   I am an assistant   </s>[INST]Another question[/INST]");
        t.expect("falcon3", "<|system|>\nYou are a helpful assistant\n<|user|>\nHello\n<|assistant|>\nHi there\n"
                + "<|user|>\nWho are you\n<|assistant|>\n   I am an assistant   \n<|user|>\nAnother question\n"
                + "<|assistant|>\n");
        t.expect("zephyr", "<|system|>\nYou are a helpful assistant<|endoftext|>\n<|user|>\nHello<|endoftext|>\n"
                + "<|assistant|>\nHi there<|endoftext|>\n<|user|>\nWho are you<|endoftext|>\n<|assistant|>\n   I am "
                + "an assistant   <|endoftext|>\n<|user|>\nAnother question<|endoftext|>\n<|assistant|>\n");
        t.expect("deepseek3", "You are a helpful assistant\n\n<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜>"
                + "<｜User｜>Who are you<｜Assistant｜>   I am an assistant   <｜end▁of▁sentence｜><｜User｜>Another "
                + "question<｜Assistant｜>");
        t.expect("deepseek-ocr", "You are a helpful assistantHelloHi thereWho are you   I am an assistant   Another "
                + "question");
        t.expect("exaone3", "[|system|]You are a helpful assistant[|endofturn|]\n[|user|]Hello\n[|assistant|]Hi "
                + "there[|endofturn|]\n[|user|]Who are you\n[|assistant|]I am an assistant[|endofturn|]\n[|user|]"
                + "Another question\n[|assistant|]");
        t.expect("exaone4", "[|system|]You are a helpful assistant[|endofturn|]\n[|user|]Hello\n[|assistant|]Hi "
                + "there[|endofturn|]\n[|user|]Who are you\n[|assistant|]I am an assistant[|endofturn|]\n[|user|]"
                + "Another question\n[|assistant|]");
        t.expect("exaone-moe", "<|system|>\nYou are a helpful assistant<|endofturn|>\n<|user|>\nHello<|endofturn|>\n"
                + "<|assistant|>\nHi there<|endofturn|>\n<|user|>\nWho are you<|endofturn|>\n<|assistant|>\nI am an "
                + "assistant<|endofturn|>\n<|user|>\nAnother question<|endofturn|>\n<|assistant|>\n");
        t.expect("rwkv-world", "System: You are a helpful assistant\n\nUser: Hello\n\nAssistant: Hi there\n\nUser: "
                + "Who are you\n\nAssistant: I am an assistant\n\nUser: Another question\n\nAssistant:");
        t.expect("bailing-think", "<role>SYSTEM</role>You are a helpful assistant<role>HUMAN</role>Hello<role>"
                + "ASSISTANT</role>Hi there<role>HUMAN</role>Who are you<role>ASSISTANT</role>   I am an assistant   "
                + "<role>HUMAN</role>Another question<role>ASSISTANT</role><think>");
        t.expect("bailing2", "<role>SYSTEM</role>You are a helpful assistant<|role_end|><role>HUMAN</role>"
                + "Hello<|role_end|><role>ASSISTANT</role>Hi there<|role_end|><role>HUMAN</role>Who are "
                + "you<|role_end|><role>ASSISTANT</role>   I am an assistant   <|role_end|><role>HUMAN</role>Another "
                + "question<|role_end|><role>ASSISTANT</role>");
        t.expect("llama4", "<|header_start|>system<|header_end|>\n\nYou are a helpful assistant<|eot|><|header_start|>"
                + "user<|header_end|>\n\nHello<|eot|><|header_start|>assistant<|header_end|>\n\nHi there<|eot|>"
                + "<|header_start|>user<|header_end|>\n\nWho are you<|eot|><|header_start|>assistant<|header_end|>\n\n"
                + "I am an assistant<|eot|><|header_start|>user<|header_end|>\n\nAnother question<|eot|>"
                + "<|header_start|>assistant<|header_end|>\n\n");
        t.expect("smolvlm", "<|im_start|>You are a helpful assistant\n\nUser: Hello<end_of_utterance>\nAssistant: Hi "
                + "there<end_of_utterance>\nUser: Who are you<end_of_utterance>\nAssistant:    I am an assistant   "
                + "<end_of_utterance>\nUser: Another question<end_of_utterance>\nAssistant:");
        t.expect("hunyuan-moe", "<|startoftext|>You are a helpful assistant<|extra_4|><|startoftext|>Hello<|extra_0|>"
                + "Hi there<|eos|><|startoftext|>Who are you<|extra_0|>   I am an assistant   <|eos|><|startoftext|>"
                + "Another question<|extra_0|>");
        t.expect("gpt-oss", "<|start|>system<|message|>You are a helpful assistant<|end|><|start|>user<|message|>"
                + "Hello<|end|><|start|>assistant<|message|>Hi there<|return|><|start|>user<|message|>Who are "
                + "you<|end|><|start|>assistant<|message|>   I am an assistant   <|return|><|start|>user<|message|>"
                + "Another question<|end|><|start|>assistant");
        t.expect("hunyuan-dense", "You are a helpful assistant<｜hy_place▁holder▁no▁3｜><｜hy_User｜>Hello<｜hy_Assistant｜>"
                + "<｜hy_Assistant｜>Hi there<｜hy_place▁holder▁no▁2｜><｜hy_User｜>Who are you<｜hy_Assistant｜>"
                + "<｜hy_Assistant｜>   I am an assistant   <｜hy_place▁holder▁no▁2｜><｜hy_User｜>Another "
                + "question<｜hy_Assistant｜>");
        t.expect("hunyuan-vl", "<｜hy_begin▁of▁sentence｜>You are a helpful assistant<｜hy_place▁holder▁no▁3｜>"
                + "Hello<｜hy_User｜>Hi there<｜hy_Assistant｜>Who are you<｜hy_User｜>   I am an assistant   "
                + "<｜hy_Assistant｜>Another question<｜hy_User｜>");
        t.expect("kimi-k2", "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|><|im_user|>"
                + "user<|im_middle|>Hello<|im_end|><|im_assistant|>assistant<|im_middle|>Hi there<|im_end|><|im_user|>"
                + "user<|im_middle|>Who are you<|im_end|><|im_assistant|>assistant<|im_middle|>   I am an assistant   "
                + "<|im_end|><|im_user|>user<|im_middle|>Another question<|im_end|><|im_assistant|>"
                + "assistant<|im_middle|>");
        t.expect("grok-2", "System: You are a helpful assistant<|separator|>\n\nHuman: Hello<|separator|>\n\n"
                + "Assistant: Hi there<|separator|>\n\nHuman: Who are you<|separator|>\n\nAssistant:    I am an "
                + "assistant   <|separator|>\n\nHuman: Another question<|separator|>\n\nAssistant:");
        t.expect("pangu-embedded", "[unused9]系统：You are a helpful assistant[unused10][unused9]用户：Hello[unused10]"
                + "[unused9]助手：Hi there[unused10][unused9]用户：Who are you[unused10][unused9]助手：   I am an assistant   "
                + "[unused10][unused9]用户：Another question[unused10][unused9]助手：");
        t.expect("solar-open", "<|begin|>system<|content|>You are a helpful assistant<|end|><|begin|>user<|content|>"
                + "Hello<|end|><|begin|>assistant<|content|>Hi there<|end|><|begin|>user<|content|>Who are you<|end|>"
                + "<|begin|>assistant<|content|>   I am an assistant   <|end|><|begin|>user<|content|>Another "
                + "question<|end|><|begin|>assistant");
        t.verify(22);
    }

    @Test
    public void namesAreLlamaCppsTableAndEachIsABuiltIn() {
        assertEquals(LLAMA_CPP_NAMES, new ArrayList<>(ChatTemplateSource.names()));
        for (String name : LLAMA_CPP_NAMES) {
            String source = ChatTemplateSource.resolve(name);
            assertTrue(name, ChatTemplateSource.isName(name));
            assertTrue(name, ChatTemplateSource.isBuiltIn(source));
            assertTrue(name + " keeps the conversation as given",
                    new ChatFormatter(source, "", "", false, false, "llama").supportsSystemRole());
        }
        assertFalse(ChatTemplateSource.isBuiltIn("{{ messages }}"));
        assertFalse(ChatTemplateSource.isBuiltIn(null));
        assertFalse(ChatTemplateSource.isName("dots1"));
    }

    @Test
    public void conversationsWithoutASystemMessageTakeEachStylesDefault() {
        List<Map<String, String>> chat = Arrays.asList(msg("user", "Hi"), msg("assistant", "Yo"), msg("user", "Q"));
        assertEquals("[INST] Hi [/INST]Yo</s><s>[INST] Q [/INST]", render("llama2-sys-bos", chat, true));
        assertEquals("<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Hi<|role_end|>"
                + "<role>ASSISTANT</role>Yo<|role_end|><role>HUMAN</role>Q<|role_end|><role>ASSISTANT</role>",
                render("bailing2", chat, true));
        assertEquals("<s>user<|role_sep|>Hi<|message_sep|>available functions<|role_sep|>[]<|message_sep|>"
                + "assistant<|role_sep|>Yo<|message_sep|>user<|role_sep|>Q<|message_sep|>available functions"
                + "<|role_sep|>[]<|message_sep|>assistant<|role_sep|>", render("gigachat", chat, true));
        assertEquals("[unused9]系统：[unused10][unused9]用户：Hi[unused10][unused9]助手：Yo[unused10][unused9]用户：Q[unused10]",
                render("pangu-embedded", chat, false));
        assertEquals("<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nYo<end_of_turn>\n"
                + "<start_of_turn>user\nQ<end_of_turn>\n", render("gemma", chat, false));
    }

    @Test
    public void generationPromptFollowsAddGenerationPrompt() {
        List<Map<String, String>> chat = Collections.singletonList(msg("user", " Hi "));
        assertEquals("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>", render("llama3", chat, false));
        assertEquals("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n\n", render("llama3", chat, true));
        assertEquals("User: Hi\n\nAssistant:", render("rwkv-world", chat, false));
        assertEquals("<role>HUMAN</role> Hi <role>ASSISTANT</role>", render("bailing", chat, true));
    }

    @Test
    public void toolAndOtherRolesFollowEachStyle() {
        List<Map<String, String>> chat = Arrays.asList(msg("user", "U"), msg("tool", " R "), msg("function", "F"));
        assertEquals("[|user|]U\n", render("exaone3", chat, false));
        assertEquals("[|user|]U\n[|tool|]R[|endofturn|]\n", render("exaone4", chat, false));
        assertEquals("<|user|>\nU<|endofturn|>\n<|tool|>\nR<|endofturn|>\n", render("exaone-moe", chat, false));
        assertEquals("<|im_user|>user<|im_middle|>U<|im_end|><|im_system|>tool<|im_middle|> R <|im_end|>F<|im_end|>",
                render("kimi-k2", chat, false));
        assertEquals("[unused9]系统：[unused10][unused9]用户：U[unused10][unused9]工具： R [unused10]"
                + "[unused9]方法：F[unused10]", render("pangu-embedded", chat, false));
        assertEquals("GPT4 Correct User: U<|end_of_turn|>GPT4 Correct Tool:  R <|end_of_turn|>"
                + "GPT4 Correct Function: F<|end_of_turn|>", render("openchat", chat, false));
        assertEquals("<role>HUMAN</role>U<role>TOOL</role> R <role>FUNCTION</role>F", render("bailing", chat, false));
    }

    @Test
    public void aRequestChatTemplateMayNameABuiltIn() {
        ChatFormatter model = new ChatFormatter(ChatTemplateSource.CHATML, "", "", false, false, "llama");
        ChatRequest request = ChatRequest.builder().messages(CONVERSATION).chatTemplate("llama3").build();
        assertEquals(render("llama3", CONVERSATION, true), model.format(request));
        ChatRequest yandex = ChatRequest.builder().messages(CONVERSATION).chatTemplate("yandex").build();
        assertFalse("yandex leaves the system message out, as llama.cpp's formatter does",
                model.format(yandex).contains("helpful"));
    }
}
