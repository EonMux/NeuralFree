package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The interactive-console flags of {@link LLMDemoOptions}, spelled as llama.cpp common/arg.cpp spells them
 * ({@code -o/--output}, {@code -mli}, {@code -i}, {@code -if}, {@code --in-prefix}, {@code --in-suffix},
 * {@code --in-prefix-bos}, {@code -e/--no-escape}, {@code -st}, {@code -sys}, {@code -cnv/-no-cnv},
 * {@code --server-base}, {@code --image/--audio}), and the combinations refused by name.
 */
public class LLMDemoOptionsInteractiveTest {

    private static LLMDemoOptions parse(String... args) throws UsageException {
        return LLMDemoOptions.parse(args);
    }

    private static String refusal(String... args) {
        return assertThrows(UsageException.class, () -> parse(args)).getMessage();
    }

    @Test
    public void theChatFlagsParse() throws UsageException {
        LLMDemoOptions o = parse("m.gguf", "-o", "chat.txt", "-mli", "-st", "-sys", "short");
        assertEquals("chat.txt", o.outFile);
        assertTrue(o.multilineInput);
        assertTrue(o.singleTurn);
        assertEquals("short", o.systemPrompt);
        assertFalse(o.completionInteractive());
        assertEquals("chat.txt", parse("m.gguf", "--output-file", "chat.txt").outFile);
        assertEquals("chat.txt", parse("m.gguf", "--output", "chat.txt").outFile);
        assertEquals("s", parse("m.gguf", "--system-prompt", "s").systemPrompt);
    }

    @Test
    public void theCompletionFlagsSelectTheCompletionMode() throws UsageException {
        LLMDemoOptions o = parse("m.gguf", "-if", "--in-prefix", " ", "--in-suffix", "A:", "--in-prefix-bos",
                "--no-escape");
        assertTrue("-if implies -i", o.interactive);
        assertTrue(o.interactiveFirst);
        assertEquals(" ", o.inPrefix);
        assertEquals("A:", o.inSuffix);
        assertTrue(o.inPrefixBos);
        assertFalse(o.escape);
        assertTrue(o.completionInteractive());
        assertTrue("escapes are on by default", parse("m.gguf", "-i").escape);
        assertTrue(parse("m.gguf", "--in-suffix", "x").completionInteractive());
    }

    @Test
    public void conversationFlagsFollowTheReference() throws UsageException {
        assertTrue("-no-cnv turns the chat template off", parse("m.gguf", "-no-cnv", "-p", "x").raw);
        assertTrue(parse("m.gguf", "-no-cnv", "-i").completionInteractive());
        assertFalse(parse("m.gguf", "-cnv").completionInteractive());
        assertTrue(refusal("m.gguf", "-cnv", "-i").contains("-cnv runs the chat"));
        assertTrue(refusal("m.gguf", "-no-cnv").contains("-no-cnv completes the -p prompt"));
    }

    @Test
    public void clientModeTakesTheServerAndForwardsSampling() throws UsageException {
        LLMDemoOptions o = parse("--server-base", "http://localhost:8080/", "--api-key", "k", "--temp", "0.2",
                "-p", "hi", "--image", "a.png", "--audio", "b.wav", "-o", "t.txt", "--thinking");
        assertEquals("http://localhost:8080/", o.serverBase);
        assertEquals("k", o.apiKey);
        assertEquals(List.of(new LLMDemoOptions.MediaFile("image", "a.png"),
                new LLMDemoOptions.MediaFile("audio", "b.wav")), o.media);
        assertNull(o.modelPath);
    }

    @Test
    public void clientModeRefusesLocalSettingsByName() {
        assertTrue(refusal("--server-base", "http://h:1", "-m", "x.gguf").contains("--model"));
        assertTrue(refusal("--server-base", "http://h:1", "-c", "512").contains("--server-base and -c"));
        assertTrue(refusal("--server-base", "http://h:1", "--gpu").contains("--gpu"));
        assertTrue(refusal("--server-base", "http://h:1", "--draft", "d.gguf").contains("--draft"));
        assertTrue(refusal("--server-base", "http://h:1", "--lora", "a.gguf").contains("--lora"));
        assertTrue(refusal("--server-base", "http://h:1", "-i").contains("completion mode"));
        assertTrue(refusal("--server-base", "http://h:1", "--raw").contains("--raw"));
        assertTrue(refusal("--server-base", "http://h:1", "--debug-prompt").contains("--debug-prompt"));
        assertTrue(refusal("--server-base", "http://h:1", "--stats-file", "s").contains("--stats-file"));
        assertTrue(refusal("--server-base", "http://h:1", "--mode", "embed").contains("--mode embed"));
        assertTrue(refusal("--server-base", "http://h:1", "-no-cnv", "-p", "x").contains("-no-cnv"));
    }

    @Test
    public void clientModeRefusesTheSharedLoadFlagsByName() {
        assertTrue(refusal("--server-base", "http://h:1", "-fa", "on")
                .contains("--server-base and -fa exclude each other"));
        assertTrue(refusal("--server-base", "http://h:1", "-ctk", "q8_0")
                .contains("--server-base and -ctk exclude each other"));
        assertTrue(refusal("--server-base", "http://h:1", "--kv-type", "q8_0").contains("--kv-type"));
        assertTrue(refusal("--server-base", "http://h:1", "-ngl", "all").contains("-ngl"));
    }

    @Test
    public void clientModeRefusesTheLocalTemplateAndControlVectorFlags() {
        assertTrue(refusal("--server-base", "http://h:1", "--chat-template", "chatml")
                .contains("--server-base and --chat-template exclude each other"));
        assertTrue(refusal("--server-base", "http://h:1", "--chat-template-file", "t.jinja")
                .contains("--chat-template-file"));
        assertTrue(refusal("--server-base", "http://h:1", "--control-vector", "v.gguf")
                .contains("--control-vector"));
        assertTrue(refusal("--server-base", "http://h:1", "--control-vector-layer-range", "1", "2")
                .contains("--control-vector-layer-range"));
    }

    @Test
    public void clientModeTakesThePromptFilesAndChecksTheJsonSchema() throws Exception {
        java.nio.file.Path prompt = java.nio.file.Files.createTempFile("prompt", ".txt");
        java.nio.file.Path system = java.nio.file.Files.createTempFile("system", ".txt");
        try {
            java.nio.file.Files.writeString(prompt, "first turn\n");
            java.nio.file.Files.writeString(system, "be brief\n");
            LLMDemoOptions o = parse("--server-base", "http://h:1", "-f", prompt.toString(), "-sysf",
                    system.toString(), "--jinja", "-j", "{\"type\":\"object\"}");
            assertEquals("first turn", o.prompt);
            assertEquals("be brief", o.systemPrompt);
        } finally {
            java.nio.file.Files.delete(prompt);
            java.nio.file.Files.delete(system);
        }
        assertTrue(refusal("--server-base", "http://h:1", "-j", "{not json").contains("--json-schema is not valid"));
    }

    @Test
    public void consoleFlagsAreRefusedByModesWithoutAConversation() {
        assertTrue(refusal("m.gguf", "--mode", "embed", "-o", "x.txt").contains("-o/--output belongs to the"));
        assertTrue(refusal("m.gguf", "--mode", "rerank", "-mli").contains("-mli"));
        assertTrue(refusal("m.gguf", "--mode", "classify", "-st").contains("--mode classify runs no conversation"));
    }

    @Test
    public void mediaFlagsNeedThePromptTurn() {
        assertTrue(refusal("m.gguf", "--image", "a.png").contains("attach to the -p prompt's turn"));
        assertTrue(refusal("--server-base", "http://h:1", "--audio", "a.wav").contains("give -p"));
    }
}
