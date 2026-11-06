package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link LLMDemoOptions}: every Hugging Face checkpoint flag and GGUF export flag parses into the matching
 * {@link ModelSourceOptions} or {@link EngineConfig} field, the {@code -Dneuralfree.*} property defaults still
 * apply where no flag is given, and the bad values are rejected with {@link UsageException}.
 */
public class LLMDemoOptionsHfTest {

    private final Map<String, String> saved = new HashMap<>();

    @Before
    public void clearHfProperties() {
        for (String key : new String[] {
                ModelSourceOptions.PROP_MATERIALIZE, ModelSourceOptions.PROP_MEMORY_MAX,
                ModelSourceOptions.PROP_CACHE_DIR, ModelSourceOptions.PROP_OUTTYPE,
                ModelSourceOptions.PROP_ONLINE_QUANT, ModelSourceOptions.PROP_ONLINE_OUTPUT,
                ModelSourceOptions.PROP_EXACT_QUANT,
                ModelSourceOptions.PROP_MTP, ModelSourceOptions.PROP_ST_DENSE,
                ModelSourceOptions.PROP_OVERRIDES, ModelSourceOptions.PROP_TOKENIZER,
                ModelSourceOptions.PROP_LORA_BASE_CONFIG, ModelSourceOptions.PROP_BASE_DIR,
                ActivationQuant.PROPERTY, EngineConfig.PROP_MODEL_SAMPLING_DEFAULTS}) {
            saved.put(key, System.getProperty(key));
            System.clearProperty(key);
        }
    }

    @After
    public void restoreHfProperties() {
        for (Map.Entry<String, String> e : saved.entrySet()) {
            if (e.getValue() == null) {
                System.clearProperty(e.getKey());
            } else {
                System.setProperty(e.getKey(), e.getValue());
            }
        }
    }

    private static LLMDemoOptions parse(String... args) throws UsageException {
        return LLMDemoOptions.parse(args);
    }

    @Test
    public void propertyDefaultsHoldWithoutFlags() throws UsageException {
        LLMDemoOptions o = parse();
        assertEquals(ModelSourceOptions.defaults(), o.engine.sourceOptions());
        assertEquals(ActivationQuant.WEIGHT_ONLY, o.engine.activationQuant());
        assertTrue(o.engine.modelSamplingDefaults());
        assertNull(o.exportGguf);
        assertEquals(GgufWriter.Mode.NATIVE, o.exportMode);
        assertFalse(o.exportOverwrite);
    }

    @Test
    public void checkpointFlagsReachTheSourceOptions() throws UsageException {
        ModelSourceOptions s = parse(
                "--hf-materialize", "disk",
                "--hf-memory-max", "512M",
                "--hf-cache-dir", "/tmp/nf/cache",
                "--hf-outtype", "q8_0",
                "--quant-online", "q4_0",
                "--leave-output-tensor",
                "--hf-quant-exact",
                "--hf-no-mtp",
                "--hf-st-dense", "false",
                "--hf-overrides", "{\"hidden_size\": 256}",
                "--hf-tokenizer", "tok.gguf",
                "--lora-base-config", "adapter",
                "--hf-base-dir", "base").engine.sourceOptions();
        assertEquals(ModelSourceOptions.Materialize.DISK, s.materialize());
        assertEquals(512L << 20, s.memoryBudget());
        assertEquals(Paths.get("/tmp/nf/cache"), s.cacheDir());
        assertEquals(ModelSourceOptions.OutType.Q8_0, s.outType());
        assertEquals(ModelSourceOptions.OnlineQuant.Q4_0, s.onlineQuant());
        assertFalse("--leave-output-tensor keeps the LM head unquantized", s.quantizeOutputTensor());
        assertTrue(s.exactQuant());
        assertFalse(s.includeMtp());
        assertFalse(s.stDenseModules());
        assertEquals(256L, s.configOverrides().get("hidden_size"));
        assertEquals(Paths.get("tok.gguf"), s.tokenizerDir());
        assertEquals(Paths.get("adapter"), s.loraBaseConfig());
        assertEquals(Paths.get("base"), s.baseDir());
    }

    @Test
    public void theDraftFlagReachesTheSourceOptions() throws UsageException {
        assertEquals(ModelSourceOptions.DraftMode.MTP_ONLY,
                parse("--hf-draft", "mtp").engine.sourceOptions().draftMode());
        assertEquals(ModelSourceOptions.DraftMode.DSPARK,
                parse("--hf-draft", "dspark").engine.sourceOptions().draftMode());
        assertTrue(parse("--hf-draft", "mtp").engine.sourceOptions().mtpOnly());
        assertFalse(parse("--hf-draft", "no-nextn").engine.sourceOptions().includeMtp());
        UsageException e = assertThrows(UsageException.class, () -> parse("--hf-draft", "eagle"));
        assertTrue(e.getMessage(), e.getMessage().contains("--hf-draft")
                && e.getMessage().contains("none, mtp, no-nextn or dspark"));
    }

    @Test
    public void engineFlagsReachTheEngineConfig() throws UsageException {
        LLMDemoOptions o = parse("--act-quant", "checkpoint", "--model-sampling-defaults", "false");
        assertEquals(ActivationQuant.CHECKPOINT, o.engine.activationQuant());
        assertFalse(o.engine.modelSamplingDefaults());
    }

    @Test
    public void flagsOverrideProperties() throws UsageException {
        System.setProperty(ModelSourceOptions.PROP_OUTTYPE, "f32");
        assertEquals(ModelSourceOptions.OutType.F32, parse().engine.sourceOptions().outType());
        assertEquals(ModelSourceOptions.OutType.F16, parse("--hf-outtype", "f16").engine.sourceOptions().outType());
    }

    @Test
    public void exportFlagsParseIntoTheExportOptions() throws UsageException {
        LLMDemoOptions o = parse("--export-gguf", "out.gguf");
        assertEquals("out.gguf", o.exportGguf);
        assertEquals(GgufWriter.Mode.NATIVE, o.exportMode);
        assertFalse(o.exportOverwrite);
        LLMDemoOptions full = parse("--export-gguf", "out.gguf", "--export-mode", "compat-exact",
                "--export-overwrite");
        assertEquals(GgufWriter.Mode.COMPAT_EXACT, full.exportMode);
        assertTrue(full.exportOverwrite);
        assertEquals(GgufWriter.Mode.COMPAT, parse("--export-mode", "compat").exportMode);
    }

    @Test
    public void quantOnlineAcceptsTheVllmOnlineMethods() throws UsageException {
        assertEquals(ModelSourceOptions.OnlineQuant.FP8_PER_BLOCK,
                parse("--quant-online", "fp8_per_block").engine.sourceOptions().onlineQuant());
        assertEquals(ModelSourceOptions.OnlineQuant.NVFP4_PER_TOKEN,
                parse("--quant-online", "nvfp4_per_token").engine.sourceOptions().onlineQuant());
        assertEquals(ModelSourceOptions.OnlineQuant.FP8_PER_TENSOR,
                parse("--quant-online", "fp8").engine.sourceOptions().onlineQuant());
        UsageException e = assertThrows(UsageException.class, () -> parse("--quant-online", "fp4"));
        assertTrue(e.getMessage(), e.getMessage().contains("--quant-online"));
        assertEquals(ModelSourceOptions.OnlineQuant.MXFP4,
                parse("--quant-online", "mxfp4").engine.sourceOptions().onlineQuant());
    }

    @Test
    public void quantizationConfigCarriesTheVllmLayerSelection() throws UsageException {
        ModelSourceOptions o = parse("--quant-online", "online", "--quantization-config",
                "{\"moe\": \"mxfp4\", \"ignore\": [\"re:.*shared_expert.*\"]}").engine.sourceOptions();
        assertEquals(ModelSourceOptions.OnlineQuant.ONLINE, o.onlineQuant());
        assertEquals(java.util.Collections.singletonList("re:.*shared_expert.*"), o.quantConfig().ignore());
        UsageException bad = assertThrows(UsageException.class,
                () -> parse("--quantization-config", "{\"linear\": \"fp4\"}"));
        assertTrue(bad.getMessage(), bad.getMessage().startsWith("--quantization-config: ")
                && bad.getMessage().contains("unknown quantization name 'fp4'"));
        UsageException legacy = assertThrows(UsageException.class,
                () -> parse("--quant-online", "q8_0", "--quantization-config", "{\"linear\": \"mxfp8\"}"));
        assertTrue(legacy.getMessage(), legacy.getMessage().contains("q8_0 is a llama-quantize type"));
        assertThrows(UsageException.class, () -> parse("--quantization-config"));
    }

    @Test
    public void leaveOutputTensorDefaultsToQuantizingTheLmHead() throws UsageException {
        assertTrue(parse("--quant-online", "q4_0").engine.sourceOptions().quantizeOutputTensor());
        assertFalse(parse("--leave-output-tensor").engine.sourceOptions().quantizeOutputTensor());
        System.setProperty(ModelSourceOptions.PROP_ONLINE_OUTPUT, "false");
        assertFalse(parse().engine.sourceOptions().quantizeOutputTensor());
    }

    @Test
    public void invalidCheckpointValuesAreRejected() {
        assertThrows(UsageException.class, () -> parse("--hf-materialize", "soon"));
        assertThrows(UsageException.class, () -> parse("--hf-materialize"));
        assertThrows(UsageException.class, () -> parse("--hf-memory-max", "10X"));
        assertThrows(UsageException.class, () -> parse("--hf-memory-max", "-1"));
        assertThrows(UsageException.class, () -> parse("--hf-outtype", "q5_1"));
        assertThrows(UsageException.class, () -> parse("--quant-online", "auto"));
        assertThrows(UsageException.class, () -> parse("--hf-quant-exact", "yes"));
        assertThrows(UsageException.class, () -> parse("--hf-st-dense", "maybe"));
        assertThrows(UsageException.class, () -> parse("--hf-overrides", "[1]"));
        assertThrows(UsageException.class, () -> parse("--hf-overrides", "null"));
        assertThrows(UsageException.class, () -> parse("--hf-overrides", "{\"a\" 1}"));
        assertThrows(UsageException.class, () -> parse("--hf-tokenizer"));
        assertThrows(UsageException.class, () -> parse("--act-quant", "strict"));
        assertThrows(UsageException.class, () -> parse("--model-sampling-defaults", "yes"));
    }

    @Test
    public void exportValueErrorsAreRejected() {
        assertThrows(UsageException.class, () -> parse("--export-gguf"));
        assertThrows(UsageException.class, () -> parse("--export-mode", "wave"));
        assertThrows(UsageException.class, () -> parse("--export-mode"));
        assertThrows(UsageException.class, () -> parse("--export-overwrite", "true"));
    }

    @Test
    public void unknownHfFlagIsStillRejected() {
        assertThrows(UsageException.class, () -> parse("--hf-mabel"));
    }
}
