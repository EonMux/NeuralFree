package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.engine.DraftHeadOptions;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.W12F16Mode;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The speculative decoding launch flags ({@link SpecLaunchFlags}): the drafter types, the draft model and its KV
 * settings, the n-gram sizes, the lookup cache files, the suffix limits, the verification mode and the draft model's
 * own placement map to the {@code neuralfree.*} properties the batch engine reads; {@code draft-mtp} and the Hugging
 * Face draft downloads are refused at launch naming the flag, and so is every unusable value. The draft thread flags
 * size the one CPU thread pool and must agree.
 */
class SpecLaunchFlagsTest {

    private static Map<String, String> properties(String... args) {
        return ServerLaunchArgs.parse(args).properties();
    }

    private static String refusalOf(String... args) {
        return assertThrows(IllegalArgumentException.class, () -> ServerLaunchArgs.parse(args)).getMessage();
    }

    @Test
    void heterogeneousVocabDraftingIsASwitchAndASpeculativeConfigKey(@TempDir Path dir) throws IOException {
        Path draft = Files.createFile(dir.resolve("draft.gguf"));
        assertEquals("true", properties("--spec-draft-heterogeneous-vocab")
                .get(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB));
        assertEquals("false", properties("--no-spec-draft-heterogeneous-vocab")
                .get(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB));
        assertEquals("true", properties("-md", draft.toString(), "--spec-draft-heterogeneous-vocab")
                .get(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB));
        String json = "{\"method\": \"draft_model\", \"model\": \"" + draft + "\","
                + " \"use_heterogeneous_vocab\": true}";
        Map<String, String> p = properties("--speculative-config", json);
        assertEquals("true", p.get(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB));
        assertEquals(draft.toString(), p.get(EngineConfig.PROP_DRAFT));
    }

    /** The lossless W12 switches set the weights properties the CLI carries too; the head needs the flag. */
    @Test
    void theW12SwitchesSetTheWeightsProperties() {
        assertEquals("true", properties("--w12").get(SpeculativeOptions.PROP_W12));
        assertEquals("false", properties("--no-w12").get(SpeculativeOptions.PROP_W12));
        assertEquals("true", properties("--w12", "--w12-head").get(SpeculativeOptions.PROP_W12_HEAD));
        assertEquals("false", properties("--w12", "--no-w12-head").get(SpeculativeOptions.PROP_W12_HEAD));
        assertTrue(refusalOf("--w12-head").contains(SpeculativeOptions.PROP_W12_HEAD));
        Properties launch = new Properties();
        launch.putAll(properties("--w12", "--w12-head"));
        assertTrue(EngineConfig.fromProperties(launch).speculativeOptions().w12());
        assertTrue(EngineConfig.fromProperties(launch).speculativeOptions().w12Head());
    }

    /** {@code --w12-f16} sets the F16 grid contract the W12 streaming holds an F16 tensor to. */
    @Test
    void theW12F16GridModeIsTakenFromItsFlag() {
        assertEquals("refuse", properties("--w12", "--w12-f16", "refuse").get(SpeculativeOptions.PROP_W12_F16));
        assertEquals("round", properties("--w12", "--w12-f16", "round").get(SpeculativeOptions.PROP_W12_F16));
        assertTrue(refusalOf("--w12-f16", "lossy").contains("--w12-f16"));
        assertTrue(refusalOf("--w12-f16", "lossy").contains("refuse or round"));
        Properties launch = new Properties();
        launch.putAll(properties("--w12", "--w12-f16", "round"));
        assertEquals(W12F16Mode.ROUND, EngineConfig.fromProperties(launch).speculativeOptions().w12F16());
    }

    /** The derived candidate head flags ({@code --draft-head-*}, {@code --draft-prec}) and the vLLM JSON key. */
    @Test
    void theDerivedDraftHeadFlagsMapToTheirProperties(@TempDir Path dir) throws IOException {
        Path ranks = Files.createFile(dir.resolve("ranks.json"));
        Map<String, String> p = properties("--draft-head-vocab", "98304", "--draft-head-freq", ranks.toString(),
                "--draft-head-keep", "128, 42,7", "--draft-head-prec", "FP8", "--draft-prec", "q8_0");
        assertEquals("98304", p.get(SpeculativeOptions.PROP_DRAFT_HEAD_VOCAB));
        assertEquals(ranks.toString(), p.get(SpeculativeOptions.PROP_DRAFT_HEAD_FREQ));
        assertEquals("128, 42,7", p.get(SpeculativeOptions.PROP_DRAFT_HEAD_KEEP));
        assertEquals("fp8", p.get(SpeculativeOptions.PROP_DRAFT_HEAD_PREC));
        assertEquals("q8_0", p.get(SpeculativeOptions.PROP_DRAFT_PREC));
        assertEquals("fp8", properties("--speculative-config",
                "{\"method\": \"dflash\", \"model\": \"" + ranks + "\", \"draft_lm_head_quantization\": \"fp8\"}")
                .get(SpeculativeOptions.PROP_DRAFT_HEAD_PREC));
        assertEquals("nvfp4", properties("--draft-head-prec", "nvfp4")
                .get(SpeculativeOptions.PROP_DRAFT_HEAD_PREC));
        assertEquals("nvfp4", properties("--speculative-config",
                "{\"method\": \"dflash\", \"model\": \"" + ranks + "\", \"draft_lm_head_quantization\": \"nvfp4\"}")
                .get(SpeculativeOptions.PROP_DRAFT_HEAD_PREC));
        Properties launch = new Properties();
        launch.putAll(p);
        assertEquals(DraftHeadOptions.Prec.FP8,
                EngineConfig.fromProperties(launch).speculativeOptions().draftHead().headPrec());
        assertTrue(refusalOf("--draft-head-prec", "int4").contains("int4"));
        assertTrue(refusalOf("--draft-prec", "int4").startsWith("--draft-prec"));
        String keep = refusalOf("--draft-head-keep", "4,x");
        assertTrue(keep.startsWith("--draft-head-keep"), keep);
        assertTrue(refusalOf("--draft-head-vocab", "0").startsWith("--draft-head-vocab"));
        String missing = refusalOf("--draft-head-freq", dir.resolve("absent.json").toString());
        assertTrue(missing.startsWith("--draft-head-freq"), missing);
    }

    @Test
    void llamaServerSpellingsMapToTheDraftProperties() {
        Map<String, String> p = properties("-m", "model.gguf", "--spec-type", "ngram-simple",
                "--spec-draft-n-max", "6", "--spec-draft-n-min", "2",
                "--spec-ngram-simple-size-n", "4", "--spec-ngram-simple-size-m", "12");
        assertEquals("ngram-simple", p.get(EngineConfig.PROP_DRAFT_TYPE));
        assertEquals("6", p.get(EngineConfig.PROP_DRAFT_MAX));
        assertEquals("2", p.get(EngineConfig.PROP_DRAFT_MIN));
        assertEquals("4", p.get(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n"));
        assertEquals("12", p.get(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.m"));

        Properties launch = new Properties();
        launch.putAll(p);
        EngineConfig cfg = EngineConfig.fromProperties(launch);
        assertEquals(List.of(EngineConfig.DraftType.NGRAM), cfg.lookupTypes());
        assertEquals(6, cfg.draftMax());
        assertEquals(2, cfg.draftMin());
        NgramDrafter.Options simple = cfg.speculativeOptions().ngramOptions(NgramDrafter.Kind.SIMPLE);
        assertEquals(4, simple.sizeN());
        assertEquals(12, simple.sizeM());
    }

    @Test
    void theOlderDraftLengthSpellingsSetTheSameProperties() {
        Map<String, String> p = properties("--draft-max", "5", "--draft-n-min", "1");
        assertEquals("5", p.get(EngineConfig.PROP_DRAFT_MAX));
        assertEquals("1", p.get(EngineConfig.PROP_DRAFT_MIN));
        assertEquals("7", properties("--draft-n-max", "7").get(EngineConfig.PROP_DRAFT_MAX));
        assertEquals("0", properties("--draft-min", "0").get(EngineConfig.PROP_DRAFT_MIN));
    }

    @Test
    void repeatedTypeFlagsAppend() {
        assertEquals("ngram-simple,ngram-simple",
                properties("--spec-type", "ngram-simple", "--spec-type", "ngram-simple")
                        .get(EngineConfig.PROP_DRAFT_TYPE));
    }

    @Test
    void everyLookupTypeAndTheDraftModelAreTakenWithTheirSettings(@TempDir Path dir) throws IOException {
        Path draft = Files.createFile(dir.resolve("draft.gguf"));
        Path statics = Files.createFile(dir.resolve("static.bin"));
        Path dynamic = dir.resolve("dynamic.bin");
        Map<String, String> p = properties("-m", "model.gguf", "--spec-type", "ngram-map-k,ngram-mod",
                "--spec-type", "ngram-cache,suffix", "--spec-type", "draft-simple", "-md", draft.toString(),
                "-ctkd", "q8_0", "-ctvd", "F16", "-ubd", "4", "--spec-draft-p-min", "0.25",
                "--spec-draft-sample-method", "probabilistic",
                "--spec-ngram-map-k-size-n", "2", "--spec-ngram-map-k-size-m", "6", "--spec-ngram-map-k-min-hits", "2",
                "--spec-ngram-mod-n-match", "4", "--spec-ngram-mod-n-max", "10", "--spec-ngram-mod-n-min", "1",
                "-lcs", statics.toString(), "-lcd", dynamic.toString(),
                "--spec-suffix-max-tree-depth", "16", "--spec-suffix-min-token-prob", "0.2",
                "--spec-draft-sampling", "probabilistic");

        Properties launch = new Properties();
        launch.putAll(p);
        EngineConfig cfg = EngineConfig.fromProperties(launch);
        assertEquals(List.of(EngineConfig.DraftType.NGRAM_MAP_K, EngineConfig.DraftType.NGRAM_MOD,
                EngineConfig.DraftType.NGRAM_CACHE, EngineConfig.DraftType.SUFFIX), cfg.lookupTypes());
        assertEquals(EngineConfig.DraftType.MODEL, cfg.draftType());
        assertEquals(draft.toString(), cfg.draftModelPath());
        assertEquals(0.25f, cfg.draftPMin());
        assertEquals(KvDType.Q8_0, cfg.memory().draftTypeK());
        assertEquals(KvDType.F16, cfg.memory().draftTypeV());
        assertEquals(4, cfg.memory().draftUbatch());
        SpeculativeOptions o = cfg.speculativeOptions();
        assertEquals(VerifyMode.REJECTION, o.verifyMode());
        assertTrue(o.probabilisticDraft());
        NgramDrafter.Options mapK = o.ngramOptions(NgramDrafter.Kind.MAP_K);
        assertEquals(List.of(2, 6, 2), List.of(mapK.sizeN(), mapK.sizeM(), mapK.minHits()));
        NgramDrafter.Options mod = o.ngramOptions(NgramDrafter.Kind.MOD);
        assertEquals(List.of(4, 10, 1), List.of(mod.sizeN(), mod.sizeM(), mod.draftMin()));
        assertEquals(statics.toString(), o.lookupCacheStatic());
        assertEquals(dynamic.toString(), o.lookupCacheDynamic());
        assertEquals(16, o.suffixOptions().maxTreeDepth());
        assertEquals(0.2, o.suffixOptions().minTokenProb(), 0.0);
    }

    @Test
    void theVerificationModeAndItsSettingsAreTaken() {
        Map<String, String> p = properties("--spec-type", "suffix", "--spec-verify", "FLY", "--spec-fly-window", "3",
                "--spec-fly-entropy-threshold", "0.5", "--spec-fly-entropy-top-k", "8");
        Properties launch = new Properties();
        launch.putAll(p);
        SpeculativeOptions o = EngineConfig.fromProperties(launch).speculativeOptions();
        assertEquals(VerifyMode.FLY, o.verifyMode());
        assertEquals(3, o.flyWindow());
        assertEquals(0.5, o.flyEntropyThreshold(), 0.0);
        assertEquals(8, o.flyEntropyTopK());
        assertEquals("rejection", properties("--rejection-sample-method", "rejection")
                .get(SpeculativeOptions.PROP_VERIFY));
        assertEquals("match", properties("--spec-draft-sampling", "greedy").get(SpeculativeOptions.PROP_VERIFY));
        assertEquals("0.5,0.25", properties("--spec-synth-rates", "0.5,0.25").get(SpeculativeOptions.PROP_SYNTH_RATES));
    }

    @Test
    void draftMtpIsRefusedNamingTheType() {
        for (String type : new String[]{"draft-mtp", "ngram-simple,draft-mtp"}) {
            String message = refusalOf("--spec-type", type);
            assertTrue(message.startsWith("--spec-type"), message);
            assertTrue(message.contains("draft-mtp runs in the single-sequence CLI"), message);
        }
        String appended = refusalOf("--spec-type", "suffix", "--spec-type", "draft-mtp");
        assertTrue(appended.contains("suffix,draft-mtp"), appended);
        String unknown = refusalOf("--spec-type", "lookahead");
        assertTrue(unknown.startsWith("--spec-type: "), unknown);
    }

    @Test
    void draftDownloadFlagsAreRefusedByName() {
        for (String flag : new String[]{"-hfd", "-hfrd", "--hf-repo-draft", "--spec-draft-hf"}) {
            String message = refusalOf("-m", "model.gguf", flag, "org/repo");
            assertTrue(message.startsWith(flag + " is not supported by the server"), message);
        }
        String hits = refusalOf("--spec-ngram-simple-min-hits", "2");
        assertTrue(hits.startsWith("--spec-ngram-simple-min-hits is not a setting"), hits);
    }

    @Test
    void theDraftPlacementFlagsMapToTheDraftProperties() {
        Map<String, String> p = properties("-m", "model.gguf", "-devd", "0, A100", "-ngld", "6",
                "-otd", "exps=cpu,blk\\.1\\..*=gpu", "-ncmoed", "2", "-cmoed");
        assertEquals("0,A100", p.get(EngineConfig.PROP_DRAFT_GPU_DEVICES));
        assertEquals("6", p.get(EngineConfig.PROP_DRAFT_N_GPU_LAYERS));
        assertEquals("exps=cpu,blk\\.1\\..*=gpu", p.get(EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES));
        assertEquals("2", p.get(EngineConfig.PROP_DRAFT_N_CPU_MOE));
        assertEquals("true", p.get(EngineConfig.PROP_DRAFT_CPU_MOE));

        Properties launch = new Properties();
        launch.putAll(p);
        launch.setProperty(EngineConfig.PROP_DRAFT, "draft.gguf");
        EngineConfig cfg = EngineConfig.fromProperties(launch);
        assertEquals(List.of("0", "A100"), cfg.draftDevices());
        assertEquals(6, cfg.draftGpuLayers());
        assertEquals(2, cfg.draftTensorOverrides().size());
        assertTrue(cfg.draftTensorOverrides().get(0).host());
        assertFalse(cfg.draftTensorOverrides().get(1).host());
        assertTrue(cfg.draftCpuMoe());
        assertEquals(2, cfg.draftNCpuMoe());
    }

    @Test
    void theDraftPlacementAliasesAndLlamaCppMarkersSetTheSameProperties() {
        assertEquals("auto", properties("-ngld", "auto").get(EngineConfig.PROP_DRAFT_N_GPU_LAYERS));
        assertEquals("auto", properties("--spec-draft-ngl", "-1").get(EngineConfig.PROP_DRAFT_N_GPU_LAYERS));
        assertEquals("all", properties("--gpu-layers-draft", "all").get(EngineConfig.PROP_DRAFT_N_GPU_LAYERS));
        assertEquals("all", properties("--n-gpu-layers-draft", "-2").get(EngineConfig.PROP_DRAFT_N_GPU_LAYERS));
        assertEquals("none", properties("--device-draft", "none").get(EngineConfig.PROP_DRAFT_GPU_DEVICES));
        assertEquals("3", properties("--spec-draft-ncmoe", "3").get(EngineConfig.PROP_DRAFT_N_CPU_MOE));
        assertEquals("7", properties("--n-cpu-moe-draft", "7").get(EngineConfig.PROP_DRAFT_N_CPU_MOE));
        assertEquals("exps=cpu", properties("--override-tensor-draft", "exps=cpu")
                .get(EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES));
    }

    @Test
    void unusableDraftPlacementValuesAreRefusedNamingTheFlag() {
        assertTrue(refusalOf("-ngld", "many").startsWith("-ngld must be a whole number"));
        assertTrue(refusalOf("-ngld", "-3").startsWith("-ngld must be between 0"));
        assertTrue(refusalOf("-devd", "none,0").startsWith("-devd none,0"));
        assertTrue(refusalOf("--spec-draft-device", "0,,1").startsWith("--spec-draft-device 0,,1 has an empty"));
        assertTrue(refusalOf("-otd", "exps").startsWith("-otd entry 'exps' is <tensor name pattern>"));
        assertTrue(refusalOf("-otd", "exps=vram").startsWith("-otd entry 'exps=vram'"));
        assertTrue(refusalOf("-ncmoed", "-1").startsWith("-ncmoed must be between"));
    }

    @Test
    void theDraftThreadFlagsSizeTheSharedCpuThreadPool() {
        assertEquals("4", properties("-td", "4").get(CpuParallel.THREADS_PROPERTY));
        assertEquals("5", properties("--threads-batch-draft", "5").get(CpuParallel.THREADS_PROPERTY));
        assertEquals("6", properties("--threads-draft", "6").get(CpuParallel.THREADS_PROPERTY));
        assertEquals("7", properties("--spec-draft-threads-batch", "7").get(CpuParallel.THREADS_PROPERTY));
        assertEquals("4", properties("-td", "4", "-tbd", "4").get(CpuParallel.THREADS_PROPERTY));
        String disagreement = refusalOf("-td", "4", "-tbd", "8");
        assertTrue(disagreement.startsWith("-tbd 8 conflicts with the " + CpuParallel.THREADS_PROPERTY), disagreement);
        assertTrue(refusalOf("-td", "0").startsWith("-td must be between 1"));
        assertTrue(refusalOf("-tbd", "x").startsWith("-tbd must be a whole number"));
    }

    @Test
    void unusableValuesAreRefusedNamingTheFlag(@TempDir Path dir) {
        assertTrue(refusalOf("-md", dir.resolve("absent.gguf").toString()).startsWith("-md is not a regular file"));
        assertTrue(refusalOf("-lcs", dir.toString()).startsWith("-lcs is not a regular file"));
        assertTrue(refusalOf("-lcd", dir.toString()).startsWith("-lcd is a directory"));
        assertTrue(refusalOf("-lcd", dir.resolve("missing/dynamic.bin").toString())
                .startsWith("-lcd names a file in a missing directory"));
        assertTrue(refusalOf("-ctkd", "q3").startsWith("-ctkd=q3 is not a supported KV cache type"));
        assertTrue(refusalOf("--spec-draft-p-min", "1.5").startsWith("--spec-draft-p-min must be between 0 and 1"));
        assertTrue(refusalOf("--spec-verify", "maybe").startsWith("--spec-verify: unknown verification mode"));
        assertTrue(refusalOf("--spec-draft-sampling", "top").startsWith("--spec-draft-sampling 'top' is invalid"));
        assertTrue(refusalOf("--spec-draft-sample-method", "beam")
                .startsWith("--spec-draft-sample-method=beam: expected greedy or probabilistic"));
        assertTrue(refusalOf("--spec-suffix-max-spec-factor", "-1")
                .startsWith("--spec-suffix-max-spec-factor must not be negative"));
    }

    @Test
    void contradictingSettingsAreRefusedOnceTheLineIsRead() {
        String message = refusalOf("--spec-type", "ngram-simple", "--spec-ngram-simple-size-n-min", "6",
                "--spec-ngram-simple-size-n", "4");
        assertTrue(message.startsWith("the speculative decoding flags contradict: "), message);
        assertTrue(message.contains("sizeNMin 6 exceeds sizeN 4"), message);
        // llama.cpp takes an ngram-mod floor above its longest draft: a full-length proposal is kept below it
        Properties launch = new Properties();
        launch.putAll(properties("--spec-type", "ngram-mod", "--spec-ngram-mod-n-max", "4",
                "--spec-ngram-mod-n-min", "6"));
        NgramDrafter.Options mod = EngineConfig.fromProperties(launch).speculativeOptions()
                .ngramOptions(NgramDrafter.Kind.MOD);
        assertEquals(List.of(4, 6), List.of(mod.sizeM(), mod.draftMin()));
    }

    @Test
    void thePromptLookupRangeMapsToItsProperty() {
        Map<String, String> p = properties("--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "6",
                "--spec-ngram-simple-size-n-min", "2");
        assertEquals("2", p.get(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n.min"));
        Properties launch = new Properties();
        launch.putAll(p);
        NgramDrafter.Options simple = EngineConfig.fromProperties(launch).speculativeOptions()
                .ngramOptions(NgramDrafter.Kind.SIMPLE);
        assertEquals(List.of(2, 6), List.of(simple.sizeNMin(), simple.sizeN()));
        Map<String, String> v0 = properties("--ngram-prompt-lookup-max", "5", "--ngram-prompt-lookup-min", "1");
        assertEquals("5", v0.get(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n"));
        assertEquals("1", v0.get(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n.min"));
        assertTrue(refusalOf("--spec-ngram-simple-size-n-min", "0")
                .startsWith("--spec-ngram-simple-size-n-min must be between 1 and 64"));
    }

    @Test
    void speculativeConfigJsonSetsTheSameProperties() {
        Map<String, String> p = properties("--speculative-config", "{\"method\": \"ngram\","
                + " \"num_speculative_tokens\": 5, \"prompt_lookup_max\": 4, \"prompt_lookup_min\": 2}");
        assertEquals("ngram-simple", p.get(EngineConfig.PROP_DRAFT_TYPE));
        assertEquals("5", p.get(EngineConfig.PROP_DRAFT_MAX));
        assertEquals("4", p.get(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n"));
        assertEquals("2", p.get(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n.min"));
        assertEquals("7", properties("--speculative-config",
                "{\"method\": \"ngram\", \"num_speculative_tokens\": 5}", "--spec-draft-n-max", "7")
                .get(EngineConfig.PROP_DRAFT_MAX), "a later flag overrides a key");
        assertEquals("3", properties("--speculative-config",
                "{\"model\": \"ngram\", \"prompt_lookup_max\": 3}")
                .get(SpeculativeOptions.PROP_NGRAM_SIMPLE + ".size.n.min"), "one bound sets both");
    }

    @Test
    void speculativeConfigJsonTakesTheSuffixDraftModelAndVerifyKeys(@TempDir Path dir) throws IOException {
        Map<String, String> suffix = properties("--speculative-config", "{\"method\": \"suffix\","
                + " \"suffix_decoding_max_tree_depth\": 16, \"suffix_decoding_max_cached_requests\": 4,"
                + " \"suffix_decoding_min_token_prob\": 0.2, \"rejection_sample_method\": \"block\"}");
        Properties launch = new Properties();
        launch.putAll(suffix);
        EngineConfig cfg = EngineConfig.fromProperties(launch);
        assertEquals(List.of(EngineConfig.DraftType.SUFFIX), cfg.lookupTypes());
        assertEquals(16, cfg.draftMax(), "suffix drafts up to its tree depth without num_speculative_tokens");
        assertEquals(16, cfg.speculativeOptions().suffixOptions().maxTreeDepth());
        assertEquals(4, cfg.speculativeOptions().suffixOptions().maxCachedRequests());
        assertEquals(0.2, cfg.speculativeOptions().suffixOptions().minTokenProb(), 0.0);
        assertEquals(VerifyMode.BLOCK, cfg.speculativeOptions().verifyMode());

        Path draft = Files.createFile(dir.resolve("draft.gguf"));
        Map<String, String> model = properties("--speculative-config", "{\"method\": \"draft_model\","
                + " \"model\": \"" + draft + "\", \"num_speculative_tokens\": 4}");
        assertEquals("draft-simple", model.get(EngineConfig.PROP_DRAFT_TYPE));
        assertEquals(draft.toString(), model.get(EngineConfig.PROP_DRAFT));
        assertEquals("4", model.get(EngineConfig.PROP_DRAFT_MAX));
        assertEquals("4", properties("--spec-suffix-max-cached-requests", "4")
                .get(SpeculativeOptions.PROP_SUFFIX_MAX_CACHED_REQUESTS));
    }

    @Test
    void speculativeConfigKeysTheServerDoesNotRunAreRefusedByKey(@TempDir Path dir) {
        String mtp = refusalOf("--speculative-config", "{\"method\": \"mtp\", \"num_speculative_tokens\": 1}");
        assertTrue(mtp.startsWith("--speculative-config method=mtp: --spec-type draft-mtp is not supported by the"
                + " server"), mtp);
        String absent = refusalOf("--speculative-config", "{\"method\": \"draft_model\", \"model\": \""
                + dir.resolve("absent.gguf") + "\"}");
        assertTrue(absent.startsWith("--speculative-config model="), absent);
        assertTrue(absent.contains("--spec-draft-model is not a regular file"), absent);
        String range = refusalOf("--speculative-config", "{\"method\": \"ngram\", \"prompt_lookup_max\": 80}");
        assertTrue(range.startsWith("--speculative-config prompt_lookup_max=80: --spec-ngram-simple-size-n must"
                + " be between 2 and 64"), range);
        String tp = refusalOf("--speculative-config", "{\"method\": \"ngram\", \"draft_tensor_parallel_size\": 2}");
        assertTrue(tp.startsWith("--speculative-config key 'draft_tensor_parallel_size' is not supported"), tp);
        assertTrue(refusalOf("--speculative-config", "{\"method\": \"ngram\", \"enforce_eager\": true}")
                .contains("'enforce_eager' is not supported"));
        assertTrue(refusalOf("--speculative-config").contains("needs a value"));
    }

    @Test
    void outOfRangeValuesAreRefusedNamingTheFlag() {
        assertTrue(refusalOf("--spec-draft-n-max", "0").startsWith("--spec-draft-n-max must be between 1"));
        assertTrue(refusalOf("--spec-draft-n-min", "-1").startsWith("--spec-draft-n-min must be between 0"));
        assertTrue(refusalOf("--spec-ngram-simple-size-n", "1").startsWith("--spec-ngram-simple-size-n must be"
                + " between 2 and 64"));
        assertTrue(refusalOf("--spec-ngram-simple-size-m", "129").startsWith("--spec-ngram-simple-size-m must be"
                + " between 1 and 128"));
        assertTrue(refusalOf("--spec-draft-n-max", "many").startsWith("--spec-draft-n-max must be a whole number"));
    }
}
