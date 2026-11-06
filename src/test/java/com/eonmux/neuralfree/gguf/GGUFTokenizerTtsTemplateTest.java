package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.speech.Lfm3aBackboneFixture;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The speech chat template of a tokenizer ({@code tokenizer.chat_template.tts}, llama.cpp's
 * {@code llama_model_chat_template(model, "tts")}): {@link GGUFTokenizer#ttsTemplate()} carries it and
 * {@link GGUFTokenizer#formatTtsPrompt} renders the speech text as one user message with the language and the
 * voice bound (common_chat_get_tts_prompt, common/chat.cpp:859-868), which the omni speech pipeline's fresh
 * turn speaks.
 */
public class GGUFTokenizerTtsTemplateTest {

    /** A template that shows every variable the render binds. */
    private static final String TTS_TEMPLATE = "{{- bos_token -}}{%- for m in messages -%}{{- m['content'] -}}"
            + "{%- endfor -%}[{{ language }}/{{ voice }}]" + Lfm3aBackboneFixture.AUDIO_START_PIECE;

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    @Test
    public void rendersTheSpeechTextWithTheLanguageAndTheVoice() throws IOException {
        Path backbone = new Lfm3aBackboneFixture(true, true).writer()
                .str(GGUFTokenizer.KEY_TTS_TEMPLATE, TTS_TEMPLATE).writeTempFile();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config())) {
            GGUFTokenizer tokenizer = model.tokenizer();
            assertTrue(tokenizer.ttsTemplate(), tokenizer.ttsTemplate().contains("audio_start"));
            assertEquals("the one user message, the language and the voice bound, one bos",
                    "<s>Hi[en/alloy]" + Lfm3aBackboneFixture.AUDIO_START_PIECE,
                    tokenizer.formatTtsPrompt("Hi", "en", "alloy"));
            assertEquals("an unnamed language and voice bind empty, llama.cpp's strings",
                    "<s>Hi[/]" + Lfm3aBackboneFixture.AUDIO_START_PIECE,
                    tokenizer.formatTtsPrompt("Hi", "", ""));
        }
    }

    @Test
    public void aModelWithoutTheSpeechTemplateCarriesNone() throws IOException {
        Path backbone = new Lfm3aBackboneFixture(true, true).write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config())) {
            GGUFTokenizer tokenizer = model.tokenizer();
            assertNull(tokenizer.ttsTemplate());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> tokenizer.formatTtsPrompt("Hi", "", ""));
            assertTrue(e.getMessage(), e.getMessage().contains(GGUFTokenizer.KEY_TTS_TEMPLATE));
        }
    }
}
