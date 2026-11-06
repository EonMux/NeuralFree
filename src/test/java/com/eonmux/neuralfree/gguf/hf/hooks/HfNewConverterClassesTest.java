package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The Hugging Face classes llama.cpp's converter gained after the a37cc4610 tables: the MiniCPM-OCR-Jev value head
 * ({@code MiniCPMOCRSeqClsModel}, qwen.py:712-757), the MiniCPM-OCR 1.0 text trunk ({@code MiniCPMOCR1_0TextModel},
 * minicpm.py:281), the Soprano Qwen3 backbone ({@code SopranoTextModel}, soprano.py:37) and the registration of the
 * LFM3, Limite, Mistral Large 4 and DeepSeek-V4.1 converter classes.
 */
public class HfNewConverterClassesTest {

    @Test
    public void theScoreHeadBecomesTheClassifierOutput() throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(dir(), Arch.QWEN35, "MiniCPMOCRForSequenceClassification", "");
        HfQwenHooksTest.RecordingSink sink = new HfQwenHooksTest.RecordingSink();
        HfDecisionHooks.miniCpmOcrJevTensors(ctx, "score.weight", HfQwenHooksTest.seq(2, 8), sink);
        assertEquals(Collections.singleton("cls.output.weight"), sink.tensors.keySet());
        assertEquals(List.of(2L, 8L), List.of(sink.tensors.get("cls.output.weight").shape()[0],
                sink.tensors.get("cls.output.weight").shape()[1]));
    }

    @Test
    public void aScoreHeadWithAnotherClassCountIsRefused() throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(dir(), Arch.QWEN35, "MiniCPMOCRForSequenceClassification", "");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfDecisionHooks.miniCpmOcrJevTensors(
                ctx, "score.weight", HfQwenHooksTest.seq(3, 8), new HfQwenHooksTest.RecordingSink()));
        assertTrue(e.getMessage(), e.getMessage().contains("score.weight must have 2 rows, got 3 rows"));
    }

    @Test
    public void aCheckpointWithoutTheScoreHeadIsRefused() throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(dir(), Arch.QWEN35, "MiniCPMOCRForSequenceClassification", "");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN35.ggufName());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfDecisionHooks.miniCpmOcrJevMeta(ctx, kv, plan()));
        assertTrue(e.getMessage(), e.getMessage().contains("no score.weight classification head"));
    }

    @Test
    public void theTemplateAndLabelsMatchTheConverter() {
        assertEquals(List.of("inconsistent", "consistent"), HfDecisionHooks.MINICPM_OCR_LABELS);
        assertEquals("openbmb-minicpm-ocr-jev", HfDecisionHooks.MINICPM_OCR_JEV_TYPE);
        String t = HfDecisionSupport.Templates.MINICPM_OCR_JEV;
        assertTrue(t, t.startsWith("<|im_start|>user\n{{ images[0] }}\nOCR文本：\n{{ state }}\n{{ images[1] }}\n"));
        assertTrue(t, t.endsWith("一致标为 1，不一致标为 0。"
                + "{{ '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"));
    }

    @Test
    public void theNewConverterClassesResolveToTheirArchitectures() throws IOException {
        Path dir = dir();
        assertSame(Arch.QWEN35, resolve(dir, "MiniCPMOcrForConditionalGeneration"));
        assertSame(Arch.QWEN35, resolve(dir, "MiniCPMOCRForSequenceClassification"));
        assertSame(Arch.QWEN3, resolve(dir, "SopranoTextModel"));
    }

    @Test
    public void theNewArchitecturesAreRegisteredTextModels() throws IOException {
        Path dir = dir();
        for (String name : List.of("Lfm3ForCausalLM", "LimiteForCausalLM", "MistralLarge4ForConditionalGeneration",
                "DeepseekV41ForCausalLM")) {
            assertFalse(name, HfArchRegistry.isRejected(name, HfArchEntry.Kind.TEXT));
            HfConfig cfg = config(dir, name);
            HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
            assertTrue(name, entry.hfNames().contains(name));
            assertSame(name, expected(name), entry.arch().resolve(cfg, HfTensorIndex.of(new LinkedHashMap<>())));
        }
        assertFalse(HfArchRegistry.isRejected("SopranoTextModel", HfArchEntry.Kind.TEXT));
    }

    private static Arch expected(String name) {
        switch (name) {
            case "Lfm3ForCausalLM":
                return Arch.LFM3;
            case "LimiteForCausalLM":
                return Arch.LIMITE;
            case "MistralLarge4ForConditionalGeneration":
                return Arch.MISTRAL_LARGE4;
            default:
                return Arch.DEEPSEEK41;
        }
    }

    private static Arch resolve(Path dir, String name) {
        HfConfig cfg = config(dir, name);
        return HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT).arch()
                .resolve(cfg, HfTensorIndex.of(new LinkedHashMap<>()));
    }

    private static HfConfig config(Path dir, String name) {
        return HfConfigLoader.fromJson(dir, "{\"architectures\":[\"" + name + "\"],\"num_hidden_layers\":1}");
    }

    /** A plan without tensors. */
    private static HfPlanView plan() {
        return new HfPlanView() {
            @Override
            public List<String> names() {
                return Collections.emptyList();
            }

            @Override
            public boolean has(String ggufName) {
                return false;
            }

            @Override
            public long[] ne(String ggufName) {
                throw new IllegalArgumentException(ggufName);
            }

            @Override
            public GGMLType type(String ggufName) {
                throw new IllegalArgumentException(ggufName);
            }

            @Override
            public float[] readF32(String ggufName, int maxElements) {
                throw new IllegalArgumentException(ggufName);
            }

            @Override
            public long[] readI64(String ggufName, int maxElements) {
                throw new IllegalArgumentException(ggufName);
            }
        };
    }

    private static Path dir() throws IOException {
        Path d = Files.createTempDirectory("hf-ocr-jev");
        d.toFile().deleteOnExit();
        return d;
    }
}
