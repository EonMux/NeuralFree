package com.eonmux.neuralfree.gguf.chat;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The non-Jinja route's pieces: {@link ChatTemplateSource#legacyTemplate} keeps llama.cpp's fallback
 * (empty or {@code chatml} default falls back to the tool-use template or ChatML), and
 * {@link ChatTemplateSource#detectedSource} selects what {@link ChatTemplateDetector} guessed —
 * including the detection-only {@code dots1}, which the name table itself never carries. The
 * {@link ChatTemplateSource#JINJA_PROPERTY} switch parses true/false or refuses.
 */
public class ChatTemplateNoJinjaRouteTest {

    private static Map<String, String> msg(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    @Rule
    public final TemporaryFolder temp = new TemporaryFolder();

    private String savedJinja;

    @Before
    public void saveJinjaProperty() {
        savedJinja = System.getProperty(ChatTemplateSource.JINJA_PROPERTY);
    }

    @After
    public void restoreJinjaProperty() {
        if (savedJinja == null) {
            System.clearProperty(ChatTemplateSource.JINJA_PROPERTY);
        } else {
            System.setProperty(ChatTemplateSource.JINJA_PROPERTY, savedJinja);
        }
    }

    // ------------------------------------------------------------------ legacyTemplate

    @Test
    public void theLegacyDefaultFallsBackLikeLlamaCppsInit() {
        assertEquals(ChatTemplateSource.CHATML, ChatTemplateSource.legacyTemplate(null, null));
        assertEquals(ChatTemplateSource.CHATML, ChatTemplateSource.legacyTemplate("", null));
        assertEquals(ChatTemplateSource.CHATML, ChatTemplateSource.legacyTemplate("chatml", null));
        assertEquals("TOOL", ChatTemplateSource.legacyTemplate("", "TOOL"));
        assertEquals("TOOL", ChatTemplateSource.legacyTemplate("chatml", "TOOL"));
        assertEquals("the model's template is kept as it reads", "{{ m['content'] }}",
                ChatTemplateSource.legacyTemplate("{{ m['content'] }}", "TOOL"));
    }

    @Test
    public void resolveRunsTheSameFallback() {
        assertEquals(ChatTemplateSource.resolve("chatml", null), ChatTemplateSource.resolve("", null));
        assertEquals("the tool-use template fills an empty default", ChatTemplateSource.resolve("TOOL", null),
                ChatTemplateSource.resolve("", "TOOL"));
    }

    // ------------------------------------------------------------------ detectedSource and dots1

    @Test
    public void detectedSourceSelectsTheNamedTemplate() {
        for (String name : Arrays.asList("chatml", "llama3", "zephyr", "bailing")) {
            assertEquals(name, ChatTemplateSource.resolve(name, null), ChatTemplateSource.detectedSource(name));
        }
    }

    @Test
    public void anUnknownDetectionNameIsRefusedByIt() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ChatTemplateSource.detectedSource("deepseek-v4"));
        assertEquals("no built-in chat template named 'deepseek-v4'", e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.detectedSource(null));
    }

    @Test
    public void dots1StaysOutOfTheNameTable() {
        assertFalse(ChatTemplateSource.names().contains("dots1"));
        assertFalse(ChatTemplateSource.isName("dots1"));
        assertNull(ChatTemplateSource.named("dots1"));
        assertTrue(ChatTemplateSource.isBuiltIn(ChatTemplateSource.detectedSource("dots1")));
    }

    @Test
    public void dots1RendersLlamaCppsBranch() {
        List<Map<String, String>> messages = Collections.unmodifiableList(Arrays.asList(
                msg("system", "You are a helpful assistant"), msg("user", "Hello"), msg("assistant", "Hi there"),
                msg("user", "Who are you")));
        ChatFormatter formatter = new ChatFormatter(ChatTemplateSource.detectedSource("dots1"), "", "", false,
                false, "llama");
        String rendered = formatter.format(ChatRequest.builder().messages(messages)
                .addGenerationPrompt(true).build());
        assertEquals("<|system|>You are a helpful assistant<|endofsystem|>"
                + "<|userprompt|>Hello<|endofuserprompt|>"
                + "<|response|>Hi there<|endofresponse|>"
                + "<|userprompt|>Who are you<|endofuserprompt|>"
                + "<|response|>", rendered);
    }

    // ------------------------------------------------------------------ the jinja switch

    @Test
    public void theJinjaSwitchDefaultsOnAndParsesItsValues() {
        System.clearProperty(ChatTemplateSource.JINJA_PROPERTY);
        assertTrue(ChatTemplateSource.jinjaFromProperties());
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "true");
        assertTrue(ChatTemplateSource.jinjaFromProperties());
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "TRUE");
        assertTrue(ChatTemplateSource.jinjaFromProperties());
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "false");
        assertFalse(ChatTemplateSource.jinjaFromProperties());
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, " False ");
        assertFalse(ChatTemplateSource.jinjaFromProperties());
    }

    @Test
    public void aBrokenJinjaSwitchIsRefusedByThePropertyName() {
        System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "yes");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                ChatTemplateSource::jinjaFromProperties);
        assertTrue(e.getMessage(), e.getMessage().contains(ChatTemplateSource.JINJA_PROPERTY));
        assertTrue(e.getMessage(), e.getMessage().contains("yes"));
    }

    // ------------------------------------------------------------------ raw override text

    @Test
    public void overrideTextStripsTheBomBeforeAnyResolution() throws Exception {
        File file = temp.newFile("raw.jinja");
        Files.writeString(file.toPath(), "﻿[{{ m['role'] }}]");
        assertEquals("[{{ m['role'] }}]", ChatTemplateSource.overrideText(file.toPath()));
        assertEquals("fromFile resolves what overrideText contributes",
                ChatTemplateSource.resolve("[{{ m['role'] }}]", null), ChatTemplateSource.fromFile(file.toPath()));
    }

    @Test
    public void theOverrideValueTextIsTheRawValueANameOrFileContributes() throws Exception {
        assertEquals("llama3", ChatTemplateSource.overrideValueText("llama3"));
        assertEquals("the value is kept as llama.cpp keeps it, no trim", "{{ m['content'] }}\n",
                ChatTemplateSource.overrideValueText("{{ m['content'] }}\n"));
        File file = temp.newFile("from.jinja");
        Files.writeString(file.toPath(), "User:\nAssistant:", StandardCharsets.UTF_8);
        assertEquals("User:\nAssistant:", ChatTemplateSource.overrideValueText(file.getPath()));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ChatTemplateSource.overrideValueText("no-such-named-template"));
        assertTrue(e.getMessage(), e.getMessage().contains("no-such-named-template"));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.overrideValueText(" "));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.overrideValueText(null));
    }

    @Test
    public void theDots1NameIsNoOverrideValue() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ChatTemplateSource.overrideValueText("dots1"));
        assertTrue(e.getMessage(), e.getMessage().contains("dots1"));
    }
}
