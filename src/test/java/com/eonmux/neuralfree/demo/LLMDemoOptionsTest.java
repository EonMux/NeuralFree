package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateKwargs;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.TensorOverride;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link LLMDemoOptions}: sampler settings start from the {@code -Dneuralfree.*} properties with the demo defaults
 * filling unset ones, flags override both, the llama.cpp sampling flags reach the generation request, no short
 * alias contradicts llama.cpp's meaning, and {@code --reasoning-effort} and {@code --chat-template-kwargs} collect the
 * chat template variables in llama.cpp's flag order.
 */
public class LLMDemoOptionsTest {

    private static final String[] SAMPLER_PROPS = {
            SamplerConfig.PROP_TEMPERATURE, SamplerConfig.PROP_TOP_K, SamplerConfig.PROP_TOP_P,
            SamplerConfig.PROP_MIN_P, SamplerConfig.PROP_TYPICAL_P, SamplerConfig.PROP_REPEAT_PENALTY,
            SamplerConfig.PROP_PRESENCE_PENALTY, SamplerConfig.PROP_FREQUENCY_PENALTY,
            SamplerConfig.PROP_PENALTY_LAST_N, SamplerConfig.PROP_SEED, SamplerConfig.PROP_GREEDY,
            CpuParallel.THREADS_PROPERTY
    };

    @Rule
    public final TemporaryFolder temp = new TemporaryFolder();

    private final Map<String, String> saved = new HashMap<>();

    @Before
    public void clearSamplerProperties() {
        for (String key : SAMPLER_PROPS) {
            saved.put(key, System.getProperty(key));
            System.clearProperty(key);
        }
    }

    @After
    public void restoreSamplerProperties() {
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
    public void numaAndMlockFlagsReachTheConfigurationAndTheLazyModeRefusesAPin() throws UsageException {
        String priorMode = System.getProperty(TensorReadMode.PROP_READ_MODE);
        String priorNuma = System.getProperty(NumaStrategy.PROPERTY);
        try {
            System.clearProperty(TensorReadMode.PROP_READ_MODE);
            System.clearProperty(NumaStrategy.PROPERTY);

            LLMDemoOptions opts = parse("--numa", "distribute", "--mlock");
            assertEquals(NumaStrategy.DISTRIBUTE, opts.engine.numaStrategy());
            assertEquals("distribute", System.getProperty(NumaStrategy.PROPERTY));
            assertEquals("mmap+mlock", System.getProperty(TensorReadMode.PROP_READ_MODE));
            assertFalse("the no-mmap translation happens at run time", opts.noMmap);

            assertTrue(parse("--no-mmap").noMmap);
            System.clearProperty(TensorReadMode.PROP_READ_MODE);
            parse("--lazy-mode");
            assertEquals("lazy", System.getProperty(TensorReadMode.PROP_READ_MODE));

            System.setProperty(TensorReadMode.PROP_READ_MODE, "mlock");
            UsageException pinnedLazy = assertThrows(UsageException.class, () -> parse("--lazy-mode"));
            assertTrue(pinnedLazy.getMessage(), pinnedLazy.getMessage().contains("conflicts with --mlock"));
            System.clearProperty(TensorReadMode.PROP_READ_MODE);
            UsageException lazyThenMlock = assertThrows(UsageException.class, () -> parse("--lazy-mode", "--mlock"));
            assertTrue(lazyThenMlock.getMessage(), lazyThenMlock.getMessage().contains("lazily read tensors"));
        } finally {
            restoreProperty(TensorReadMode.PROP_READ_MODE, priorMode);
            restoreProperty(NumaStrategy.PROPERTY, priorNuma);
        }
    }

    private static void restoreProperty(String key, String prior) {
        if (prior == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, prior);
        }
    }

    @Test
    public void demoDefaultsApplyWithoutProperties() throws UsageException {
        GenerationRequest r = parse().generation;
        SamplerConfig s = r.sampler();
        assertEquals(LLMDemoOptions.DEFAULT_TEMPERATURE, s.temperature(), 0f);
        assertEquals(LLMDemoOptions.DEFAULT_TOP_K, s.topK());
        assertEquals(LLMDemoOptions.DEFAULT_TOP_P, s.topP(), 0f);
        assertEquals(LLMDemoOptions.DEFAULT_MIN_P, s.minP(), 0f);
        assertEquals(LLMDemoOptions.DEFAULT_REPEAT_WINDOW, s.penaltyLastN());
        assertEquals(1.0f, s.typicalP(), 0f);
        assertEquals(LLMDemoOptions.DEFAULT_MAX_TOKENS, r.maxNewTokens());
        assertTrue(r.stopStrings().isEmpty());
        assertFalse(r.ignoreEos());
    }

    @Test
    public void samplerPropertiesReplaceDemoDefaults() throws UsageException {
        System.setProperty(SamplerConfig.PROP_TEMPERATURE, "0.3");
        System.setProperty(SamplerConfig.PROP_TYPICAL_P, "0.9");
        System.setProperty(SamplerConfig.PROP_PRESENCE_PENALTY, "0.4");
        System.setProperty(SamplerConfig.PROP_FREQUENCY_PENALTY, "0.2");
        System.setProperty(SamplerConfig.PROP_PENALTY_LAST_N, "32");
        System.setProperty(SamplerConfig.PROP_GREEDY, "true");
        SamplerConfig s = parse().generation.sampler();
        assertEquals(0.3f, s.temperature(), 0f);
        assertEquals(0.9f, s.typicalP(), 0f);
        assertEquals(0.4f, s.presencePenalty(), 0f);
        assertEquals(0.2f, s.frequencyPenalty(), 0f);
        assertEquals(32, s.penaltyLastN());
        assertTrue(s.greedy());
        assertEquals(LLMDemoOptions.DEFAULT_TOP_K, s.topK());
    }

    @Test
    public void flagsOverrideProperties() throws UsageException {
        System.setProperty(SamplerConfig.PROP_TEMPERATURE, "0.3");
        SamplerConfig s = parse("--temp", "0.9").generation.sampler();
        assertEquals(0.9f, s.temperature(), 0f);
    }

    @Test
    public void audioSpeakerNamesAReferenceWaveform() throws UsageException {
        assertEquals("voice.wav", parse("--audio-speaker", "voice.wav").audioSpeaker);
        assertNull("no reference by default", parse().audioSpeaker);
        assertThrows(UsageException.class, () -> parse("--audio-speaker"));
    }

    @Test
    public void llamaCppSamplingFlagsReachTheRequest() throws UsageException {
        GenerationRequest r = parse("--typical", "0.5", "--presence-penalty", "0.25", "--frequency-penalty", "-0.5",
                "--ignore-eos", "-r", "User:", "--reverse-prompt", "###", "-n", "64").generation;
        SamplerConfig s = r.sampler();
        assertEquals(0.5f, s.typicalP(), 0f);
        assertEquals(0.25f, s.presencePenalty(), 0f);
        assertEquals(-0.5f, s.frequencyPenalty(), 0f);
        assertTrue(r.ignoreEos());
        assertEquals(Arrays.asList("User:", "###"), r.stopStrings());
        assertEquals(64, r.maxNewTokens());
        assertEquals(0.75f, parse("--typical-p", "0.75").generation.sampler().typicalP(), 0f);
    }

    @Test
    public void invalidSamplingValuesAreRejected() {
        assertThrows(UsageException.class, () -> parse("--typical", "1.5"));
        assertThrows(UsageException.class, () -> parse("--presence-penalty", "x"));
        assertThrows(UsageException.class, () -> parse("--reverse-prompt", ""));
        assertThrows(UsageException.class, () -> parse("-r"));
    }

    @Test
    public void shortAliasesWithOtherLlamaCppMeaningsAreUnknown() throws UsageException {
        // -t is llama.cpp's --threads and means that here too; -b is nothing in llama.cpp
        parse("-t", "8");
        assertEquals("8", System.getProperty(CpuParallel.THREADS_PROPERTY));
        UsageException b = assertThrows(UsageException.class, () -> parse("-b", "512"));
        assertTrue(b.getMessage(), b.getMessage().contains("unknown option -b"));
        assertEquals(8, parse("-n", "8").generation.maxNewTokens());
        assertEquals(16, parse("--max-tokens", "16").generation.maxNewTokens());
    }

    @Test
    public void modelPathAndStatsFileAreUnsetUnlessGiven() throws UsageException {
        LLMDemoOptions none = parse();
        assertNull(none.modelPath);
        assertNull(none.statsFile);
        assertEquals("model.gguf", parse("model.gguf").modelPath);
        assertEquals("model.gguf", parse("--model", "model.gguf").modelPath);
        assertEquals("stats.txt", parse("--stats-file", "stats.txt").statsFile);
    }

    @Test
    public void usageListsTheNewFlagsAndNoContradictingAliases() {
        String usage = String.join("\n", LLMDemoOptions.usage());
        assertTrue(usage.contains("--typical"));
        assertTrue(usage.contains("--presence-penalty"));
        assertTrue(usage.contains("--frequency-penalty"));
        assertTrue(usage.contains("--ignore-eos"));
        assertTrue(usage.contains("--reverse-prompt, -r"));
        assertTrue(usage.contains("--spec-draft-ubatch-size, -ubd"));
        assertTrue(usage.contains("-ctkd"));
        assertTrue(usage.contains("-ctvd"));
        assertFalse(usage.contains(", -t,"));
        assertFalse(usage.contains("--backend, -b"));
    }

    // ============================================================ llama.cpp sampling parity

    @Test
    public void extendedLlamaCppSamplingFlagsReachTheRequest() throws UsageException {
        SamplerConfig s = parse("--mirostat", "2", "--mirostat-ent", "4.0", "--mirostat-lr", "0.05",
                "--top-n-sigma", "2.0", "--xtc-probability", "0.5", "--xtc-threshold", "0.2",
                "--dry-multiplier", "0.8", "--dry-base", "1.9", "--dry-allowed-length", "3",
                "--dry-penalty-last-n", "128", "--dynatemp-range", "1.5", "--dynatemp-exp", "0.9",
                "--min-keep", "4").generation.sampler();
        assertEquals(2, s.mirostat());
        assertEquals(4.0f, s.mirostatTau(), 0f);
        assertEquals(0.05f, s.mirostatEta(), 0f);
        assertEquals(2.0f, s.topNSigma(), 0f);
        assertEquals(0.5f, s.xtcProbability(), 0f);
        assertEquals(0.2f, s.xtcThreshold(), 0f);
        assertEquals(0.8f, s.dryMultiplier(), 0f);
        assertEquals(1.9f, s.dryBase(), 0f);
        assertEquals(3, s.dryAllowedLength());
        assertEquals(128, s.dryPenaltyLastN());
        assertEquals(1.5f, s.dynatempRange(), 0f);
        assertEquals(0.9f, s.dynatempExponent(), 0f);
        assertEquals(4, s.minKeep());
        assertEquals("the llama.cpp alias spelling applies", 2.0f,
                parse("--top-nsigma", "2.0").generation.sampler().topNSigma(), 0f);
        assertEquals(0.9f, parse("--dynatemp-exponent", "0.9").generation.sampler().dynatempExponent(), 0f);
        assertEquals("a negative sigma disables the stage, as llama.cpp's -1 default does", -1.0f,
                parse("--top-n-sigma", "-1").generation.sampler().topNSigma(), 0f);
        assertEquals("llama.cpp's defaults hold without the flags", 0.10f,
                parse().generation.sampler().xtcThreshold(), 0f);
    }

    @Test
    public void drySequenceBreakersClearLlamaCppDefaultsOnFirstUse() throws UsageException {
        assertEquals(SamplerConfig.DRY_DEFAULT_BREAKERS, parse().generation.sampler().drySequenceBreakers());
        assertEquals(List.of("Foo"), parse("--dry-sequence-breaker", "Foo").generation.sampler()
                .drySequenceBreakers());
        assertEquals(List.of("A", "B", "\n"), parse("--dry-sequence-breaker", "A", "--dry-sequence-breaker",
                "B", "--dry-sequence-breaker", "\n").generation.sampler().drySequenceBreakers());
        assertEquals("none leaves no breakers", List.of(),
                parse("--dry-sequence-breaker", "none").generation.sampler().drySequenceBreakers());
        assertEquals("breakers added after none keep the cleared set", List.of("X"),
                parse("--dry-sequence-breaker", "none", "--dry-sequence-breaker", "X")
                        .generation.sampler().drySequenceBreakers());
    }

    @Test
    public void samplersFlagSetsTheChainOrder() throws UsageException {
        assertEquals(List.of(StageKind.PENALTIES, StageKind.TOP_K, StageKind.TEMPERATURE),
                parse("--samplers", "penalties;top_k;temp").generation.sampler().order());
        assertEquals("llama.cpp's name aliases apply", List.of(StageKind.TOP_K, StageKind.TOP_P),
                parse("--samplers", "top-k;nucleus").generation.sampler().order());
    }

    @Test
    public void samplerSeqFlagSetsTheChainOrderByCharacter() throws UsageException {
        assertEquals("llama.cpp common_sampler_types_from_chars",
                List.of(StageKind.PENALTIES, StageKind.DRY, StageKind.TOP_K, StageKind.TEMPERATURE),
                parse("--sampler-seq", "edkt").generation.sampler().order());
        assertEquals(List.of(StageKind.TOP_P, StageKind.MIN_P, StageKind.ADAPTIVE_P),
                parse("--sampling-seq", "pma").generation.sampler().order());
        UsageException e = assertThrows(UsageException.class, () -> parse("--sampler-seq", "kq"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--sampler-seq: unknown sampler character 'q'"));
        assertThrows(UsageException.class, () -> parse("--sampler-seq", "kk"));
        assertThrows(UsageException.class, () -> parse("--sampler-seq", ""));
    }

    @Test
    public void adaptivePFlagsReachTheSampler() throws UsageException {
        SamplerConfig s = parse("--adaptive-target", "0.4", "--adaptive-decay", "0.5").generation.sampler();
        assertEquals(0.4f, s.adaptiveTarget(), 0f);
        assertEquals(0.5f, s.adaptiveDecay(), 0f);
        assertEquals("negative disables adaptive-p, as llama.cpp's flag",
                -1f, parse("--adaptive-target", "-1").generation.sampler().adaptiveTarget(), 0f);
        assertThrows(UsageException.class, () -> parse("--adaptive-target", "1.5"));
        assertThrows(UsageException.class, () -> parse("--adaptive-decay", "0.995"));
        assertThrows(UsageException.class, () -> parse("--adaptive-decay", "-0.1"));
    }

    @Test
    public void reasoningBudgetFlagsReachTheRequest() throws UsageException {
        GenerationRequest r = parse("--reasoning-budget", "32", "--reasoning-budget-message", "Time is up.")
                .generation;
        assertEquals(32, r.reasoningBudget());
        assertEquals("Time is up.", r.reasoningBudgetMessage());
        assertEquals("0 ends the reasoning at once", 0, parse("--reasoning-budget", "0").generation.reasoningBudget());
        assertEquals("unrestricted by default", GenerationRequest.REASONING_BUDGET_DISABLED,
                parse().generation.reasoningBudget());
        assertEquals(GenerationRequest.REASONING_BUDGET_DISABLED,
                parse("--reasoning-budget", "-1").generation.reasoningBudget());
        UsageException e = assertThrows(UsageException.class, () -> parse("--reasoning-budget", "-2"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--reasoning-budget -2 is outside"));
        assertThrows(UsageException.class, () -> parse("--reasoning-budget-message"));
        IllegalArgumentException diffusion = assertThrows(IllegalArgumentException.class,
                () -> parse("--reasoning-budget", "8").requireNoDecodingConstraints("masked diffusion"));
        assertTrue(diffusion.getMessage(), diffusion.getMessage().startsWith("--reasoning-budget"));
        String usage = String.join("\n", LLMDemoOptions.usage());
        for (String flag : new String[]{"--sampler-seq", "--adaptive-target", "--adaptive-decay",
                "--reasoning-budget", "--reasoning-budget-message"}) {
            assertTrue("the usage lists " + flag, usage.contains(flag + " <"));
        }
    }

    @Test
    public void invalidExtendedSamplingValuesAreRejected() {
        assertThrows(UsageException.class, () -> parse("--mirostat", "3"));
        assertThrows(UsageException.class, () -> parse("--mirostat", "-1"));
        assertThrows(UsageException.class, () -> parse("--mirostat-ent", "0"));
        assertThrows(UsageException.class, () -> parse("--mirostat-lr", "0"));
        assertThrows(UsageException.class, () -> parse("--xtc-probability", "1.5"));
        assertThrows(UsageException.class, () -> parse("--xtc-threshold", "-0.1"));
        assertThrows(UsageException.class, () -> parse("--dry-penalty-last-n", "-1"));
        assertThrows(UsageException.class, () -> parse("--dry-allowed-length", "-1"));
        assertThrows(UsageException.class, () -> parse("--min-keep", "-1"));
        assertThrows(UsageException.class, () -> parse("--dynatemp-range", "-1"));
        assertThrows(UsageException.class, () -> parse("--samplers", "bogus"));
        assertThrows(UsageException.class, () -> parse("--samplers", "top_k;top_k"));
        assertThrows(UsageException.class, () -> parse("--samplers", ";"));
        assertThrows(UsageException.class, () -> parse("--samplers"));
    }

    // ============================================================ grammar constraints

    @Test
    public void grammarAndJsonSchemaFlagsBuildTheConstraint() throws Exception {
        ConstraintSpec g = parse("--grammar", "root ::= \"x\"").generation.constraint();
        assertEquals(ConstraintSpec.Kind.GBNF, g.kind());
        assertEquals("root ::= \"x\"", g.text());
        File grammarFile = temp.newFile("out.gbnf");
        Files.writeString(grammarFile.toPath(), "root ::= \"y\"\n");
        assertEquals("the file's one trailing newline is dropped", "root ::= \"y\"",
                parse("--grammar-file", grammarFile.getPath()).generation.constraint().text());
        ConstraintSpec schema = parse("-j", "{\"type\": \"object\"}").generation.constraint();
        assertEquals(ConstraintSpec.Kind.JSON_SCHEMA, schema.kind());
        assertEquals("{\"type\": \"object\"}", schema.text());
        File schemaFile = temp.newFile("schema.json");
        Files.writeString(schemaFile.toPath(), "{\"type\": \"string\"}");
        assertEquals(ConstraintSpec.Kind.JSON_SCHEMA,
                parse("--json-schema-file", schemaFile.getPath()).generation.constraint().kind());
        assertEquals("the last constraint flag wins, as llama.cpp's single grammar does",
                ConstraintSpec.Kind.GBNF,
                parse("-j", "{}", "--grammar", "root ::= \"x\"").generation.constraint().kind());
        assertEquals("no flag leaves the request unconstrained", null, parse().generation.constraint());
    }

    @Test
    public void grammarFlagsReadLarkWhenNoLineDefinesAGbnfRule() throws Exception {
        ConstraintSpec lark = parse("--grammar", "start: \"x\" NUMBER\n%import common.NUMBER").generation.constraint();
        assertEquals(ConstraintSpec.Kind.LARK, lark.kind());
        File grammarFile = temp.newFile("out.lark");
        Files.writeString(grammarFile.toPath(), "start: \"y\"\n");
        assertEquals(ConstraintSpec.Kind.LARK,
                parse("--grammar-file", grammarFile.getPath()).generation.constraint().kind());
    }

    @Test
    public void aModeWithoutATokenLoopRefusesTheConstraintByName() throws UsageException {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> parse("--grammar", "root ::= \"x\"").requireNoDecodingConstraints("masked diffusion"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--grammar/--json-schema"));
        assertTrue(e.getMessage(), e.getMessage().contains("masked diffusion"));
        parse("--temp", "0.5").requireNoDecodingConstraints("masked diffusion");
    }

    @Test
    public void missingGrammarFilesAreRefusedByName() {
        String missing = new File(temp.getRoot(), "absent.gbnf").getPath();
        UsageException e = assertThrows(UsageException.class, () -> parse("--grammar-file", missing));
        assertTrue(e.getMessage(), e.getMessage().contains("no such file"));
        UsageException s = assertThrows(UsageException.class,
                () -> parse("-jf", new File(temp.getRoot(), "absent.json").getPath()));
        assertTrue(s.getMessage(), s.getMessage().contains("no such file"));
    }

    // ============================================================ logit bias

    @Test
    public void logitBiasFlagParsesLlamaCppEntries() throws UsageException {
        Map<Integer, Float> bias = parse("-l", "15043+1", "--logit-bias", "5-0.5").generation.logitBias();
        assertEquals(2, bias.size());
        assertEquals(1.0f, bias.get(15043), 0f);
        assertEquals(-0.5f, bias.get(5), 0f);
    }

    @Test
    public void invalidLogitBiasEntriesAreRejected() {
        assertThrows(UsageException.class, () -> parse("-l", "15043"));
        assertThrows(UsageException.class, () -> parse("-l", "x+1"));
        assertThrows(UsageException.class, () -> parse("-l", "15043*1"));
        assertThrows(UsageException.class, () -> parse("-l", "15043+abc"));
        assertThrows(UsageException.class, () -> parse("-l", "-4+1"));
        assertThrows(UsageException.class, () -> parse("--logit-bias", "5+inf"));
    }

    // ============================================================ prompt and system prompt files

    @Test
    public void promptAndSystemPromptFilesFeedTheTexts() throws Exception {
        File promptFile = temp.newFile("prompt.txt");
        Files.writeString(promptFile.toPath(), "Hello prompt\n");
        assertEquals("Hello prompt", parse("-f", promptFile.getPath()).prompt);
        assertEquals("--file is the long form", "Hello prompt", parse("--file", promptFile.getPath()).prompt);
        File sysFile = temp.newFile("system.txt");
        Files.writeString(sysFile.toPath(), "Be terse.\n");
        assertEquals("Be terse.", parse("-sysf", sysFile.getPath()).systemPrompt);
        assertEquals("Be terse.", parse("--system-prompt-file", sysFile.getPath()).systemPrompt);
        assertEquals("-sys is the short --system", "Be terse.",
                parse("--system", "Be terse.").systemPrompt);
        UsageException e = assertThrows(UsageException.class,
                () -> parse("-f", new File(temp.getRoot(), "absent.txt").getPath()));
        assertTrue(e.getMessage(), e.getMessage().contains("no such file"));
    }

    // ============================================================ threads, parallel, offload

    @Test
    public void threadsSetsTheCpuThreadPoolProperty() throws UsageException {
        parse("--threads", "4");
        assertEquals("4", System.getProperty(CpuParallel.THREADS_PROPERTY));
        assertThrows(UsageException.class, () -> parse("-t", "0"));
    }

    @Test
    public void parallelSetsTheMemorySequenceCount() throws UsageException {
        assertEquals(4, parse("-np", "4").engine.memory().parallel());
        assertEquals(MemoryConfig.MAX_PARALLEL, parse("--parallel", "256").engine.memory().parallel());
        assertThrows(UsageException.class, () -> parse("-np", "0"));
        assertThrows(UsageException.class, () -> parse("-np", "257"));
    }

    @Test
    public void contextCheckpointFlagsReachTheMemoryConfig() throws UsageException {
        assertEquals(MemoryConfig.DEFAULT_CTX_CHECKPOINTS, parse().engine.memory().ctxCheckpoints());
        assertEquals(8, parse("--ctx-checkpoints", "8").engine.memory().ctxCheckpoints());
        assertEquals(0, parse("--swa-checkpoints", "0").engine.memory().ctxCheckpoints());
        assertEquals(2, parse("-ctxcp", "2").engine.memory().ctxCheckpoints());
        assertEquals(1024, parse("--checkpoint-min-step", "1024").engine.memory().checkpointMinStep());
        assertEquals(0, parse("-cms", "0").engine.memory().checkpointMinStep());
        assertThrows(UsageException.class, () -> parse("--ctx-checkpoints", "-1"));
        assertThrows(UsageException.class, () -> parse("-cms", "x"));
    }

    @Test
    public void gpuLayerFlagsReachTheGpuLayers() throws UsageException {
        assertEquals(12, parse("-ngl", "12").engine.gpuLayers());
        assertEquals(12, parse("--n-gpu-layers", "12").engine.gpuLayers());
        assertEquals("a layer count leaves the streaming depth automatic", EngineConfig.VRAM_DEPTH_AUTO,
                parse("-ngl", "12").engine.vramDepth());
        assertEquals(EngineConfig.GPU_LAYERS_AUTO, parse("--gpu-layers", "auto").engine.gpuLayers());
        assertEquals("-1 is llama.cpp's auto marker", EngineConfig.GPU_LAYERS_AUTO,
                parse("-ngl", "-1").engine.gpuLayers());
        assertEquals(EngineConfig.GPU_LAYERS_ALL, parse("-ngl", "all").engine.gpuLayers());
        assertThrows(UsageException.class, () -> parse("-ngl", "bogus"));
        assertThrows(UsageException.class, () -> parse("-ngl", "-2"));
    }

    @Test
    public void overrideTensorEntriesReachTheEngine() throws UsageException {
        TensorOverride o = parse("-ot", "ffn.*=cpu").engine.tensorOverrides().get(0);
        assertTrue(o.host());
        assertEquals("cpu", o.target());
        assertEquals("the target is case-insensitive", "gpu",
                parse("--override-tensor", "x=GPU").engine.tensorOverrides().get(0).target());
        UsageException e = assertThrows(UsageException.class, () -> parse("-ot", "bogus"));
        assertTrue(e.getMessage(), e.getMessage().contains("-ot"));
        assertThrows(UsageException.class, () -> parse("-ot", "ffn.*=nvme"));
    }

    @Test
    public void flashAttentionAndKvTypeFlagsAreTheSharedLoadFlags() throws UsageException {
        MemoryConfig m = parse("-fa", "on", "-ctk", "q8_0", "--cache-type-v", "q4_0").engine.memory();
        assertEquals(KvDType.Q8_0, m.typeK());
        assertEquals(KvDType.Q4_0, m.typeV());
        assertEquals(KvDType.Q5_1, parse("--kv-type", "q5_1", "--flash-attn", "auto").engine.memory().typeV());
        UsageException off = assertThrows(UsageException.class, () -> parse("-fa", "off"));
        assertTrue(off.getMessage(), off.getMessage().contains("-fa off is not supported"));
        UsageException badType = assertThrows(UsageException.class, () -> parse("-ctk", "q6_k"));
        assertTrue(badType.getMessage(), badType.getMessage().contains("-ctk"));
        assertTrue("the usage lists -fa", String.join("\n", LLMDemoOptions.usage()).contains("-fa, --flash-attn"));
    }

    @Test
    public void cpuMoeFlagsReachTheEngine() throws UsageException {
        assertTrue(parse("--cpu-moe").engine.cpuMoe());
        assertEquals(4, parse("--n-cpu-moe", "4").engine.nCpuMoe());
        assertThrows(UsageException.class, () -> parse("--n-cpu-moe", "-1"));
    }

    @Test
    public void moeCacheFlagsReachTheEngine() throws UsageException {
        assertEquals(MoeCacheConfig.Mode.AUTO, parse("--moe-cache", "auto").engine.moeCache().mode());
        assertEquals(512, parse("--moe-cache", "512").engine.moeCache().budgetMiB());
        MoeCacheConfig ranked = parse("--moe-cache-ranking", "r.bin", "--moe-cache", "on").engine.moeCache();
        assertEquals(MoeCacheConfig.Mode.ON, ranked.mode());
        assertEquals(MoeCacheRanking.PreloadMode.PATH, ranked.ranking().preloadMode());
        assertEquals(Paths.get("r.bin"), ranked.ranking().preloadPath());
        assertEquals(MoeCacheRanking.PreloadMode.OFF,
                parse("--moe-cache", "on", "--moe-cache-ranking", "off").engine.moeCache().ranking().preloadMode());
        UsageException e = assertThrows(UsageException.class, () -> parse("--moe-cache", "-3"));
        assertTrue(e.getMessage(), e.getMessage().contains("--moe-cache"));
        e = assertThrows(UsageException.class, () -> parse("--moe-cache-ranking", " "));
        assertTrue(e.getMessage(), e.getMessage().contains("--moe-cache-ranking"));
    }

    // ============================================================ chat templates

    @Test
    public void chatTemplateFlagsWriteTheMetadataOverride() throws Exception {
        LLMDemoOptions inline = parse("--chat-template", "<s>{{content}}</s>");
        assertEquals(ChatTemplateSource.KEY_DEFAULT,
                inline.engine.sourceOptions().metadataOverrides().get(0).key());
        assertEquals("<s>{{content}}</s>", inline.engine.sourceOptions().metadataOverrides().get(0).value());
        File templateFile = temp.newFile("t.jinja");
        Files.writeString(templateFile.toPath(), "{% for m in messages %}{{m.content}}{% endfor %}");
        LLMDemoOptions fromFile = parse("--chat-template-file", templateFile.getPath());
        assertEquals("{% for m in messages %}{{m.content}}{% endfor %}",
                fromFile.engine.sourceOptions().metadataOverrides().get(0).value());
        LLMDemoOptions viaPath = parse("--chat-template", templateFile.getPath());
        assertEquals("--chat-template reads a template file as vLLM's does",
                "{% for m in messages %}{{m.content}}{% endfor %}",
                viaPath.engine.sourceOptions().metadataOverrides().get(0).value());
        assertEquals("the metadata override keeps the name; it resolves when the model loads", "llama3",
                parse("--chat-template", "llama3").engine.sourceOptions().metadataOverrides().get(0).value());
        UsageException missing = assertThrows(UsageException.class, () -> parse("--chat-template", "absent.jinja"));
        assertTrue(missing.getMessage(), missing.getMessage().startsWith("--chat-template: "));
        assertFalse(parse("--jinja").noJinja);
    }

    @Test
    public void theJinjaSwitchFlagsSelectTheRoute() throws UsageException {
        assertFalse(parse().noJinja);
        assertTrue(parse("--no-jinja").noJinja);
        assertFalse("a later --jinja wins, as llama.cpp's last flag does", parse("--no-jinja", "--jinja").noJinja);
        assertTrue(parse("--jinja", "--no-jinja").noJinja);
        UsageException client = assertThrows(UsageException.class,
                () -> parse("--server-base", "http://h:1", "--no-jinja"));
        assertTrue(client.getMessage(), client.getMessage().contains("--no-jinja"));
        assertTrue(client.getMessage(), client.getMessage().contains("--server-base"));
    }

    @Test
    public void noJinjaRefusesAnOverrideNothingIsDetectedFrom() throws Exception {
        File templateFile = temp.newFile("plain.jinja");
        Files.writeString(templateFile.toPath(), "[{{ m['role'] }}]{{ m['content'] }}");
        UsageException e = assertThrows(UsageException.class,
                () -> parse("--no-jinja", "--chat-template", "[{{ m['role'] }}]{{ m['content'] }}"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--no-jinja: no built-in chat template was detected"));
        e = assertThrows(UsageException.class,
                () -> parse("--no-jinja", "--chat-template-file", templateFile.getPath()));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--no-jinja: "));
        UsageException named = assertThrows(UsageException.class,
                () -> parse("--no-jinja", "--chat-template", "dots1"));
        assertTrue(named.getMessage(), named.getMessage().contains("'dots1'"));
        assertEquals("a detectable override parses, its raw text stored for the model load", "chatml",
                parse("--no-jinja", "--chat-template", "chatml").engine.sourceOptions().metadataOverrides()
                        .get(0).value());
    }

    @Test
    public void chatTemplateKwargsMergeOverThePropertyAndReachClientMode() throws UsageException {
        assertTrue(parse().templateKwargs.isEmpty());
        LLMDemoOptions o = parse("--chat-template-kwargs", "{\"enable_thinking\": false, \"style\": \"a\"}",
                "--chat-template-kwargs", "{\"style\": \"b\"}");
        assertEquals(Boolean.FALSE, o.templateKwargs.get("enable_thinking"));
        assertEquals("a later flag replaces an earlier variable", "b", o.templateKwargs.get("style"));
        assertThrows(UnsupportedOperationException.class, () -> o.templateKwargs.put("x", 1));
        assertEquals("b", parse("--server-base", "http://h:1", "--chat-template-kwargs", "{\"style\": \"b\"}")
                .templateKwargs.get("style"));
        UsageException e = assertThrows(UsageException.class, () -> parse("--chat-template-kwargs", "[1]"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--chat-template-kwargs must be a JSON object"));
        String previous = System.getProperty(ChatTemplateKwargs.PROPERTY);
        try {
            System.setProperty(ChatTemplateKwargs.PROPERTY, "{\"style\": \"p\", \"n\": 1}");
            LLMDemoOptions over = parse("--chat-template-kwargs", "{\"style\": \"f\"}");
            assertEquals("f", over.templateKwargs.get("style"));
            assertEquals(1L, over.templateKwargs.get("n"));
            System.setProperty(ChatTemplateKwargs.PROPERTY, "nope");
            e = assertThrows(UsageException.class, () -> parse());
            assertTrue(e.getMessage(), e.getMessage().contains(ChatTemplateKwargs.PROPERTY));
        } finally {
            if (previous == null) {
                System.clearProperty(ChatTemplateKwargs.PROPERTY);
            } else {
                System.setProperty(ChatTemplateKwargs.PROPERTY, previous);
            }
        }
    }

    /**
     * llama.cpp's reasoning chat flags: {@code -rea/--reasoning} sets the thinking switch and, on or off, binds
     * {@code enable_thinking}; {@code auto} leaves the template's default; {@code --reasoning-effort} and
     * {@code --reasoning-preserve} bind their template variables, in flag order with {@code --chat-template-kwargs}.
     */
    @Test
    public void reasoningFlagsBindTheirTemplateVariables() throws UsageException {
        assertEquals("the switch defaults to off", Boolean.FALSE, parse().thinking);
        assertEquals(Boolean.TRUE, parse("--thinking").thinking);
        for (String on : new String[]{"on", "enabled", "true", "1"}) {
            LLMDemoOptions o = parse("-rea", on);
            assertEquals(on, Boolean.TRUE, o.thinking);
            assertEquals(on, Boolean.TRUE, o.templateKwargs.get("enable_thinking"));
        }
        for (String off : new String[]{"off", "disabled", "false", "0"}) {
            LLMDemoOptions o = parse("--thinking", "--reasoning", off);
            assertEquals(off, Boolean.FALSE, o.thinking);
            assertEquals(off, Boolean.FALSE, o.templateKwargs.get("enable_thinking"));
        }
        for (String auto : new String[]{"auto", "-1"}) {
            LLMDemoOptions o = parse("--reasoning", auto);
            assertNull(auto, o.thinking);
            assertFalse(auto, o.templateKwargs.containsKey("enable_thinking"));
        }
        assertEquals("a later kwarg replaces --reasoning's variable", Boolean.FALSE,
                parse("-rea", "on", "--chat-template-kwargs", "{\"enable_thinking\": false}")
                        .templateKwargs.get("enable_thinking"));
        UsageException e = assertThrows(UsageException.class, () -> parse("--reasoning", "maybe"));
        assertEquals("--reasoning must be on, off or auto, got 'maybe'", e.getMessage());

        LLMDemoOptions effort = parse("--reasoning-effort", "low", "--reasoning-preserve");
        assertEquals("low", effort.templateKwargs.get("reasoning_effort"));
        assertEquals(Boolean.TRUE, effort.templateKwargs.get("preserve_reasoning"));
        LLMDemoOptions reset = parse("--chat-template-kwargs", "{\"reasoning_effort\": \"high\"}",
                "--reasoning-effort", "default", "--reasoning-preserve", "--no-reasoning-preserve");
        assertFalse("default drops the variable", reset.templateKwargs.containsKey("reasoning_effort"));
        assertEquals(Boolean.FALSE, reset.templateKwargs.get("preserve_reasoning"));
        e = assertThrows(UsageException.class, () -> parse("--reasoning-effort", " "));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--reasoning-effort needs a level"));
        assertThrows(UsageException.class, () -> parse("--reasoning"));
    }

    // ============================================================ draft context types and ubatch

    @Test
    public void draftContextCacheTypeAndUbatchFlags() throws UsageException {
        LLMDemoOptions o = parse("-ctkd", "q8_0", "-ctvd", "q4_0", "-ubd", "256");
        MemoryConfig m = o.engine.memory();
        assertEquals(KvDType.Q8_0, m.draftTypeK());
        assertEquals(KvDType.Q4_0, m.draftTypeV());
        assertEquals(256, m.draftUbatch());

        LLMDemoOptions longForms = parse("--spec-draft-type-k", "q5_0", "--cache-type-v-draft", "iq4_nl",
                "--spec-draft-ubatch-size", "64", "--ubatch-size-draft", "32", "--spec-draft-n-ubatch", "16");
        assertEquals(KvDType.Q5_0, longForms.engine.memory().draftTypeK());
        assertEquals(KvDType.IQ4_NL, longForms.engine.memory().draftTypeV());
        assertEquals("the last -ubd wins", 16, longForms.engine.memory().draftUbatch());

        LLMDemoOptions shared = parse("--kv-type-draft", "q5_1");
        assertEquals(KvDType.Q5_1, shared.engine.memory().draftTypeK());
        assertEquals(KvDType.Q5_1, shared.engine.memory().draftTypeV());
        assertEquals("0 keeps the target's ubatch", 0, parse("-ubd", "0").engine.memory().draftUbatch());
        assertEquals("the main cache types stay untouched", KvDType.F16, o.engine.memory().typeK());
    }

    @Test
    public void invalidDraftContextFlagsAreRejectedNamingTheFlag() {
        UsageException badType = assertThrows(UsageException.class, () -> parse("-ctkd", "q6_k"));
        assertTrue(badType.getMessage(), badType.getMessage().contains("-ctkd"));
        UsageException badVType = assertThrows(UsageException.class, () -> parse("--kv-type-draft", "q6_k"));
        assertTrue(badVType.getMessage(), badVType.getMessage().contains("--kv-type-draft"));
        UsageException negative = assertThrows(UsageException.class, () -> parse("-ubd", "-4"));
        assertTrue(negative.getMessage(), negative.getMessage().contains("-ubd"));
        UsageException notANumber = assertThrows(UsageException.class, () -> parse("--spec-draft-ubatch-size", "few"));
        assertTrue(notANumber.getMessage(), notANumber.getMessage().contains("--spec-draft-ubatch-size"));
        assertThrows(UsageException.class, () -> parse("-ctvd"));
    }

    @Test
    public void reasoningEffortAndChatTemplateKwargsBindOverTheThinkingSwitchInFlagOrder() throws UsageException {
        LLMDemoOptions o = parse("--thinking", "--reasoning-effort", "high",
                "--chat-template-kwargs", "{\"enable_thinking\": false, \"depth\": {\"n\": 2}}");
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("reasoning_effort", "high");
        expected.put("enable_thinking", false);
        expected.put("depth", Collections.singletonMap("n", 2L));
        assertEquals(expected, o.templateKwargs);
        Map<String, Object> variables = LocalChat.chatRequest(Collections.<Map<String, Object>>emptyList(),
                o.thinking, o.templateKwargs).templateVariables();
        assertEquals("the kwargs bind over --thinking", Boolean.FALSE, variables.get("enable_thinking"));
        assertEquals("high", variables.get("reasoning_effort"));

        assertEquals("a later flag overrides an earlier one", "low", parse("--reasoning-effort", "high",
                "--chat-template-kwargs", "{\"reasoning_effort\": \"low\"}").templateKwargs.get("reasoning_effort"));
        assertTrue("default keeps the template's own", parse("--chat-template-kwargs", "{\"reasoning_effort\": \"low\"}",
                "--reasoning-effort", "default").templateKwargs.isEmpty());
        assertTrue(parse().templateKwargs.isEmpty());
        LLMDemoOptions thinking = parse("--thinking");
        assertEquals(Collections.singletonMap("enable_thinking", Boolean.TRUE), LocalChat.chatRequest(
                Collections.<Map<String, Object>>emptyList(), thinking.thinking, thinking.templateKwargs)
                .templateVariables());
    }

    @Test
    public void malformedTemplateVariablesAreRefusedNamingTheFlag() {
        for (String json : new String[]{"[1, 2]", "\"text\"", "{\"a\": ", "true"}) {
            UsageException e = assertThrows(json, UsageException.class, () -> parse("--chat-template-kwargs", json));
            assertTrue(e.getMessage(), e.getMessage().startsWith("--chat-template-kwargs "));
        }
        UsageException empty = assertThrows(UsageException.class, () -> parse("--reasoning-effort", ""));
        assertTrue(empty.getMessage(), empty.getMessage().startsWith("--reasoning-effort needs a level"));
        assertThrows(UsageException.class, () -> parse("--chat-template-kwargs"));
    }

    @Test
    public void templateVariablesTravelToAServerInClientMode() throws UsageException {
        LLMDemoOptions o = parse("--server-base", "http://127.0.0.1:1", "--reasoning-effort", "high",
                "--chat-template-kwargs", "{\"x\": 1}");
        assertEquals("high", o.templateKwargs.get("reasoning_effort"));
        assertEquals(1L, o.templateKwargs.get("x"));
    }

    @Test
    public void promptCacheFlagsParseLikeLlamaCpps() throws UsageException {
        LLMDemoOptions o = parse("--prompt-cache", "turns.session", "--prompt-cache-all");
        assertEquals("turns.session", o.promptCache);
        assertTrue(o.promptCacheAll);
        assertFalse(o.promptCacheRo);
        LLMDemoOptions ro = parse("--prompt-cache", "turns.session", "--prompt-cache-ro");
        assertFalse(ro.promptCacheAll);
        assertTrue(ro.promptCacheRo);
        assertNull("the flags stay unset unless given", parse().promptCache);
        assertFalse(parse().promptCacheAll);
        assertFalse(parse().promptCacheRo);
    }

    @Test
    public void promptCacheFlagsRefuseTheirUnsupportedCombinations() throws UsageException {
        LLMDemoOptions repl = parse("--prompt-cache", "s", "--prompt-cache-all", "-i");
        assertTrue("--prompt-cache-all saves the completion REPL's turns too", repl.promptCacheAll);
        UsageException ro = assertThrows(UsageException.class, () -> parse("--prompt-cache-ro"));
        assertTrue(ro.getMessage(), ro.getMessage().contains("--prompt-cache-ro"));
        UsageException all = assertThrows(UsageException.class, () -> parse("--prompt-cache-all"));
        assertTrue(all.getMessage(), all.getMessage().contains("--prompt-cache-all"));
        UsageException client = assertThrows("--prompt-cache configures a local model",
                UsageException.class, () -> parse("--server-base", "http://127.0.0.1:1",
                        "--prompt-cache", "s"));
        assertTrue(client.getMessage(), client.getMessage().contains("--prompt-cache"));
        UsageException empty = assertThrows("an empty path names no session file",
                UsageException.class, () -> parse("--prompt-cache", " "));
        assertTrue(empty.getMessage(), empty.getMessage().contains("--prompt-cache"));
        for (String[] args : new String[][]{
                {"--prompt-cache", "s", "--mode", "embed"},
                {"--prompt-cache", "s", "--embed"},
                {"--prompt-cache", "s", "--mode", "audio"},
                {"--prompt-cache", "s", "--benchmark"},
                {"--prompt-cache", "s", "--bench-streaming"},
                {"--prompt-cache", "s", "--vram-budget"},
                {"--prompt-cache", "s", "--export-gguf", "out.gguf"}}) {
            UsageException refused = assertThrows(String.join(" ", args),
                    UsageException.class, () -> parse(args));
            assertTrue(refused.getMessage(), refused.getMessage().contains("--prompt-cache"));
        }
    }

    /** llama.cpp's self-extend flags reach the generation request; a group size below 1 refuses naming the flag. */
    @Test
    public void theSelfExtendFlagsReachTheGenerationRequest() throws UsageException {
        assertEquals(GenerationRequest.GRP_ATTN_OFF, parse().generation.grpAttnN());
        assertEquals(GenerationRequest.DEFAULT_GRP_ATTN_W, parse().generation.grpAttnW());
        assertFalse(parse().generation.selfExtend());

        assertEquals(4, parse("--grp-attn-n", "4").generation.grpAttnN());
        assertEquals(4, parse("-gan", "4").generation.grpAttnN());
        assertEquals(64, parse("--grp-attn-w", "64").generation.grpAttnW());
        assertEquals(64, parse("-gaw", "64").generation.grpAttnW());
        assertTrue(parse("--grp-attn-n", "4").generation.selfExtend());

        UsageException zero = assertThrows(UsageException.class, () -> parse("--grp-attn-n", "0"));
        assertTrue(zero.getMessage(), zero.getMessage().contains("--grp-attn-n"));
        UsageException width = assertThrows(UsageException.class, () -> parse("--grp-attn-w", "0"));
        assertTrue(width.getMessage(), width.getMessage().contains("--grp-attn-w"));
    }
}
