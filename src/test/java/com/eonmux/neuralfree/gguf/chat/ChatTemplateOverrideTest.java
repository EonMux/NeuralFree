package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@code neuralfree.chat.template} / {@code neuralfree.chat.template.file} overrides replace a model's
 * {@code tokenizer.chat_template} wherever a tokenizer is built (llama.cpp's {@code --chat-template} /
 * {@code --chat-template-file}); setting both properties is refused, and a named built-in selects it.
 */
public class ChatTemplateOverrideTest {

    private static final String SP = "▁";
    private static final String MODEL_TEMPLATE = "{{ bos_token }}{% for m in messages %}[{{ m['role'] }}]";
    private static final String OVERRIDE_TEMPLATE =
            "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}";

    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    private String savedInline;
    private String savedFile;

    @Before
    public void saveProperties() {
        savedInline = System.getProperty(ChatTemplateSource.OVERRIDE_PROPERTY);
        savedFile = System.getProperty(ChatTemplateSource.OVERRIDE_FILE_PROPERTY);
    }

    @After
    public void restoreProperties() {
        if (savedInline == null) {
            System.clearProperty(ChatTemplateSource.OVERRIDE_PROPERTY);
        } else {
            System.setProperty(ChatTemplateSource.OVERRIDE_PROPERTY, savedInline);
        }
        if (savedFile == null) {
            System.clearProperty(ChatTemplateSource.OVERRIDE_FILE_PROPERTY);
        } else {
            System.setProperty(ChatTemplateSource.OVERRIDE_FILE_PROPERTY, savedFile);
        }
    }

    private void setInline(String value) {
        System.setProperty(ChatTemplateSource.OVERRIDE_PROPERTY, value);
    }

    private void setFile(String value) {
        System.setProperty(ChatTemplateSource.OVERRIDE_FILE_PROPERTY, value);
    }

    /** A SentencePiece vocabulary carrying {@link #MODEL_TEMPLATE} as its chat template. */
    private static GGUFTokenizer tokenizer() {
        List<Object> tokens = new ArrayList<>();
        List<Object> scores = new ArrayList<>();
        List<Object> types = new ArrayList<>();
        add(tokens, scores, types, "<unk>", 2);
        add(tokens, scores, types, "<s>", 3);
        add(tokens, scores, types, "</s>", 3);
        for (int b = 0; b < 256; b++) {
            add(tokens, scores, types, String.format("<0x%02X>", b), 6);
        }
        for (String p : new String[]{SP, "a", "b", "u", "s", "e", "r", "<", ">", "[", "]"}) {
            add(tokens, scores, types, p, 1);
        }
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", "llama");
        md.put("tokenizer.ggml.tokens", tokens);
        md.put("tokenizer.ggml.scores", scores);
        md.put("tokenizer.ggml.token_type", types);
        md.put("tokenizer.ggml.bos_token_id", 1);
        md.put("tokenizer.ggml.eos_token_id", 2);
        md.put("tokenizer.ggml.unknown_token_id", 0);
        md.put("tokenizer.chat_template", MODEL_TEMPLATE);
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static void add(List<Object> tokens, List<Object> scores, List<Object> types, String t, int type) {
        tokens.add(t);
        scores.add(-1.0f);
        types.add(type);
    }

    private Path writeTemplate(String text) throws IOException {
        Path file = folder.newFile("template.jinja").toPath();
        Files.write(file, text.getBytes(StandardCharsets.UTF_8));
        return file;
    }

    @Test
    public void withoutPropertiesTheModelTemplateWins() {
        assertNull(ChatTemplateSource.fromProperties());
        assertNull(ChatTemplateSource.rawFromProperties());
        assertTrue(tokenizer().chatTemplate().contains("[{{ m['role'] }}]"));
    }

    @Test
    public void rawFromPropertiesIsTheTextBeforeAnyResolution() throws IOException {
        setInline("  " + OVERRIDE_TEMPLATE + "  ");
        assertEquals("the inline value, trimmed as the property route keeps it", OVERRIDE_TEMPLATE,
                ChatTemplateSource.rawFromProperties());
        System.clearProperty(ChatTemplateSource.OVERRIDE_PROPERTY);
        setFile(writeTemplate(OVERRIDE_TEMPLATE + "\n").toString());
        assertEquals("the file's text, resolution untouched", OVERRIDE_TEMPLATE + "\n",
                ChatTemplateSource.rawFromProperties());
        assertEquals("resolved source of the same text", ChatTemplateSource.resolve(OVERRIDE_TEMPLATE + "\n"),
                ChatTemplateSource.fromProperties());
    }

    @Test
    public void theInlinePropertyReplacesTheModelTemplate() {
        setInline(OVERRIDE_TEMPLATE);
        assertEquals(ChatTemplateSource.resolve(OVERRIDE_TEMPLATE), ChatTemplateSource.fromProperties());
        String rendered = tokenizer().formatChat(null, "ab", false);
        assertEquals("<user>ab", rendered);
    }

    @Test
    public void aBuiltInNameSelectsTheBuiltInTemplate() {
        setInline("chatml");
        assertEquals(ChatTemplateSource.CHATML, ChatTemplateSource.fromProperties());
        setInline("llama3");
        assertEquals(ChatTemplateSource.named("llama3"), ChatTemplateSource.fromProperties());
        assertEquals("<|start_header_id|>user<|end_header_id|>\n\nab<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n\n", tokenizer().formatChat(null, "ab", false));
    }

    @Test
    public void anInlinePropertyNamingAFileReadsThatFile() throws IOException {
        setInline(writeTemplate(OVERRIDE_TEMPLATE).toString());
        assertEquals("<user>ab", tokenizer().formatChat(null, "ab", false));
        setInline("missing-template.jinja");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, ChatTemplateSource::fromProperties);
        assertTrue(e.getMessage(), e.getMessage().startsWith(ChatTemplateSource.OVERRIDE_PROPERTY + ": "));
    }

    @Test
    public void theFilePropertyReplacesTheModelTemplate() throws IOException {
        setFile(writeTemplate(OVERRIDE_TEMPLATE).toString());
        assertEquals(ChatTemplateSource.resolve(OVERRIDE_TEMPLATE), ChatTemplateSource.fromProperties());
        assertEquals("<user>ab", tokenizer().formatChat(null, "ab", false));
    }

    @Test
    public void bothPropertiesAreRefused() throws IOException {
        setInline(OVERRIDE_TEMPLATE);
        setFile(writeTemplate(OVERRIDE_TEMPLATE).toString());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                ChatTemplateSource::fromProperties);
        assertTrue(e.getMessage(), e.getMessage().contains(ChatTemplateSource.OVERRIDE_PROPERTY)
                && e.getMessage().contains(ChatTemplateSource.OVERRIDE_FILE_PROPERTY));
    }

    @Test
    public void aMissingFileIsRefusedByThePropertyName() {
        Path missing = folder.getRoot().toPath().resolve("absent.jinja");
        setFile(missing.toString());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                ChatTemplateSource::fromProperties);
        assertTrue(e.getMessage(), e.getMessage().contains(ChatTemplateSource.OVERRIDE_FILE_PROPERTY));
    }

    @Test
    public void aNonUtf8FileIsRefusedByThePropertyName() throws IOException {
        Path file = folder.newFile("bad.jinja").toPath();
        Files.write(file, new byte[]{(byte) 0xFF, (byte) 0xFE, 'a'});
        setFile(file.toString());
        UncheckedIOException e = assertThrows(UncheckedIOException.class, ChatTemplateSource::fromProperties);
        assertTrue(e.getMessage(), e.getMessage().contains(ChatTemplateSource.OVERRIDE_FILE_PROPERTY));
    }
}
