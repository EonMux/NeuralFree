package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfDraftFixtures;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfMtpFixtures;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Callable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool export} flag parity with llama.cpp's {@code convert_hf_to_gguf.py} and {@code convert_lora_to_gguf.py}:
 * {@code --outtype} including TQ1_0 / TQ2_0 (quantized matrices, F16 embeddings, the whole-block fallback) and its
 * refusals, {@code --dry-run} (the plan printed, nothing written), {@code --metadata} and {@code --model-name} in
 * the written file, {@code --print-supported-models}, {@code --mmproj}, {@code --lora} with {@code --base},
 * {@code --no-tensor-first-split}, the {@code {ftype}} output name and the usage errors naming their flags.
 */
public class ExportCommandTest {

    private static final int RANK = 4;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private String cacheBefore;

    @Before
    public void isolateTheCache() throws Exception {
        cacheBefore = System.getProperty(ModelSourceOptions.PROP_CACHE_DIR);
        System.setProperty(ModelSourceOptions.PROP_CACHE_DIR, tmp.newFolder("cache").toString());
    }

    @After
    public void restoreTheCache() {
        if (cacheBefore == null) {
            System.clearProperty(ModelSourceOptions.PROP_CACHE_DIR);
        } else {
            System.setProperty(ModelSourceOptions.PROP_CACHE_DIR, cacheBefore);
        }
    }

    // ================================================================== fixtures

    /** A synthetic BF16 Llama checkpoint of 256-wide rows (FFN 512), every matrix whole ternary blocks. */
    private Path checkpoint256() throws Exception {
        return SyntheticHf.llama(1, 256, 4, 2, 256).write(tmp.newFolder("hf256").toPath());
    }

    /** A synthetic Llama checkpoint of 128-wide rows, none of them whole ternary blocks. */
    private Path checkpoint128() throws Exception {
        return SyntheticHf.llama(1, 128, 2, 1, 256).write(tmp.newFolder("hf128").toPath());
    }

    /**
     * A PEFT adapter of rank 4 on the q projection of {@link #checkpoint256()}'s single layer; its
     * {@code base_model_name_or_path} is {@code baseName} when not null.
     */
    private Path adapter(String baseName) throws Exception {
        Path dir = tmp.newFolder().toPath();
        String q = "base_model.model.model.layers.0.self_attn.q_proj";
        SafetensorsWriter.Builder w = SafetensorsWriter.builder();
        w.add(q + ".lora_A.weight", StDType.F32, new long[]{RANK, 256},
                SyntheticHf.f32(SyntheticHf.values(1, RANK * 256)));
        w.add(q + ".lora_B.weight", StDType.F32, new long[]{256, RANK},
                SyntheticHf.f32(SyntheticHf.values(2, 256 * RANK)));
        w.metadata("format", "pt");
        w.build().write(dir.resolve("adapter_model.safetensors"));
        String config = "{'peft_type':'lora','r':" + RANK + ",'lora_alpha':8"
                + (baseName == null ? "" : ",'base_model_name_or_path':'" + baseName + "'") + "}";
        Files.write(dir.resolve("adapter_config.json"), SyntheticHf.json(config).getBytes(StandardCharsets.UTF_8));
        return dir;
    }

    private Path out(String name) {
        return tmp.getRoot().toPath().resolve(name);
    }

    private Path file(String name, String content) throws Exception {
        Path p = tmp.getRoot().toPath().resolve(name);
        Files.write(p, content.getBytes(StandardCharsets.UTF_8));
        return p;
    }

    private static List<Path> export(Object... args) throws Exception {
        String[] a = new String[args.length];
        for (int i = 0; i < args.length; i++) {
            a[i] = args[i].toString();
        }
        return ExportCommand.export(a);
    }

    private static ExportCommand.UsageException usage(Object... args) {
        return assertThrows(ExportCommand.UsageException.class, () -> export(args));
    }

    private static void assertMentions(Exception e, String... parts) {
        for (String part : parts) {
            assertTrue("'" + part + "' missing from: " + e.getMessage(), e.getMessage().contains(part));
        }
    }

    /** Runs {@code run} and returns everything it printed on stdout. */
    private static String printed(Callable<List<Path>> run) throws Exception {
        PrintStream old = System.out;
        ByteArrayOutputStream cap = new ByteArrayOutputStream();
        System.setOut(new PrintStream(cap, true, StandardCharsets.UTF_8.name()));
        try {
            run.call();
        } finally {
            System.out.flush();
            System.setOut(old);
        }
        return cap.toString(StandardCharsets.UTF_8.name());
    }

    // ================================================================== --outtype

    @Test
    public void outtypeTq1_0QuantizesTheMatricesAndKeepsTheEmbeddingsF16() throws Exception {
        Path out = out("tq1.gguf");
        List<Path> written = export(checkpoint256(), out, "--outtype", "tq1_0");
        assertEquals(List.of(out), written);
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals(GGMLType.TQ1_0, p.getTensor("blk.0.attn_q.weight").type());
            assertEquals(GGMLType.TQ1_0, p.getTensor("blk.0.ffn_down.weight").type());
            assertEquals(GGMLType.F16, p.getTensor("token_embd.weight").type());
            assertEquals(GGMLType.F16, p.getTensor("output.weight").type());
            assertEquals(GGMLType.F32, p.getTensor("blk.0.attn_norm.weight").type());
            assertEquals("MOSTLY_TQ1_0", 36, p.getInt(GgufKeys.GENERAL_FILE_TYPE).intValue());
            float[] q = p.readTensorDataAsFloat(p.getTensor("blk.0.attn_q.weight"));
            float scale = 0f;
            for (int i = 0; i < 256; i++) {
                scale = Math.max(scale, Math.abs(q[i]));
            }
            for (int i = 0; i < 256; i++) {
                assertTrue("a ternary row holds -d, 0 or d: " + q[i], q[i] == 0f || Math.abs(q[i]) == scale);
            }
        }
    }

    @Test
    public void outtypeTq2_0Q8_0AndF16SelectTheirTypes() throws Exception {
        Path model = checkpoint256();
        export(model, out("tq2.gguf"), "--outtype", "tq2_0");
        try (GGUFFileParser p = new GGUFFileParser(out("tq2.gguf").toString())) {
            assertEquals(GGMLType.TQ2_0, p.getTensor("blk.0.attn_k.weight").type());
            assertEquals(37, p.getInt(GgufKeys.GENERAL_FILE_TYPE).intValue());
        }
        export(model, out("q8.gguf"), "--outtype", "q8_0");
        try (GGUFFileParser p = new GGUFFileParser(out("q8.gguf").toString())) {
            assertEquals(GGMLType.Q8_0, p.getTensor("blk.0.attn_q.weight").type());
            assertEquals(GGMLType.Q8_0, p.getTensor("token_embd.weight").type());
        }
        export(model, out("f16.gguf"), "--outtype", "F16");
        try (GGUFFileParser p = new GGUFFileParser(out("f16.gguf").toString())) {
            assertEquals(GGMLType.F16, p.getTensor("blk.0.attn_q.weight").type());
        }
    }

    @Test
    public void aTernaryOuttypeFallsBackToF16WithoutWholeBlocks() throws Exception {
        Path out = out("tq2narrow.gguf");
        export(checkpoint128(), out, "--outtype", "tq2_0");
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals(GGMLType.F16, p.getTensor("blk.0.attn_q.weight").type());
            assertEquals(GGMLType.F16, p.getTensor("token_embd.weight").type());
        }
    }

    @Test
    public void outtypeIsRefusedForASourceThatIsAlreadyGguf() throws Exception {
        Path gguf = out("plain.gguf");
        export(checkpoint256(), gguf, "--outtype", "q8_0");
        assertMentions(usage(gguf, out("again.gguf"), "--outtype", "f32"), "--outtype f32", "keeps its tensor types");
        assertFalse(Files.exists(out("again.gguf")));
    }

    @Test
    public void anUnknownOuttypeNamesTheFlagAndTheChoices() throws Exception {
        Path model = checkpoint256();
        assertMentions(usage(model, out("x.gguf"), "--outtype", "q9_0"), "--outtype 'q9_0' is invalid", "tq1_0");
        assertFalse(Files.exists(out("x.gguf")));
    }

    // ================================================================== --dry-run and --no-tensor-first-split

    @Test
    public void dryRunPrintsThePlanAndWritesNothing() throws Exception {
        Path model = checkpoint256();
        Path out = out("dry.gguf");
        String printed = printed(() -> export(model, out, "--dry-run", "--split-max-tensors", "7"));
        assertMentions(new Exception(printed), "dry run, not writing files", "dry-00001-of-00002.gguf",
                "dry-00002-of-00002.gguf", "n_tensors = 7", "n_tensors = 5", "total_size = ");
        assertFalse(Files.exists(out));
        assertFalse(Files.exists(out("dry-00001-of-00002.gguf")));
    }

    @Test
    public void dryRunOfAVocabOnlyExportPlansItsSingleFile() throws Exception {
        Path model = checkpoint256();
        Path out = out("dryvocab.gguf");
        String printed = printed(() -> export(model, out, "--dry-run", "--vocab-only"));
        assertMentions(new Exception(printed), out + ": n_tensors = 0, total_size = 0");
        assertFalse(Files.exists(out));
    }

    @Test
    public void noTensorFirstSplitWritesAMetadataOnlyFirstShard() throws Exception {
        Path out = out("nt.gguf");
        List<Path> written = export(checkpoint256(), out, "--no-tensor-first-split");
        assertEquals(List.of(out("nt-00001-of-00002.gguf"), out("nt-00002-of-00002.gguf")), written);
        try (GGUFFileParser p = new GGUFFileParser(written.get(0).toString())) {
            assertEquals(0, p.getShards().get(0).tensorCount());
            assertEquals(12, p.getShards().get(1).tensorCount());
            assertEquals(GGMLType.BF16, p.getTensor("blk.0.attn_q.weight").type());
        }
    }

    // ================================================================== --metadata and --model-name

    @Test
    public void metadataFileOverridesTheExportedMetadata() throws Exception {
        Path meta = file("meta.json", "{\"general.description\": \"an export\", \"general.tags\":"
                + " [\"text-generation\", \"llama\"], \"general.sampling.temp\": 1,"
                + " \"general.base_models\": [{\"name\": \"Base\", \"organization\": \"Org\"}],"
                + " \"custom.ratio\": 0.5}");
        Path out = out("meta.gguf");
        export(checkpoint256(), out, "--metadata", meta);
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("an export", p.getString("general.description"));
            assertEquals(GGUFValueType.ARRAY, p.getValueType("general.tags"));
            assertEquals(GGUFValueType.STRING, p.getArrayElementType("general.tags"));
            assertEquals(List.of("text-generation", "llama"), p.getArray("general.tags"));
            assertEquals(GGUFValueType.FLOAT32, p.getValueType(GgufKeys.GENERAL_SAMPLING_TEMP));
            assertEquals(1.0f, p.getFloat(GgufKeys.GENERAL_SAMPLING_TEMP), 0f);
            assertEquals(GGUFValueType.UINT32, p.getValueType(GgufKeys.GENERAL_BASE_MODEL_COUNT));
            assertEquals(1L, p.getUInt32(GgufKeys.GENERAL_BASE_MODEL_COUNT).longValue());
            assertEquals("Base", p.getString("general.base_model.0.name"));
            assertEquals("Org", p.getString("general.base_model.0.organization"));
            assertEquals(0.5f, p.getFloat("custom.ratio"), 0f);
            assertEquals("llama", p.getString(GgufKeys.GENERAL_ARCHITECTURE));
        }
    }

    @Test
    public void overrideKvSetsScalarValuesOnTopOfTheMetadataFile() throws Exception {
        Path model = checkpoint256();
        Path meta = file("ov.json", "{\"general.description\": \"from file\"}");
        Path out = out("override.gguf");
        export(model, out, "--metadata", meta,
                "--override-kv", "general.description=str:from flag",
                "--override-kv", "general.sampling.top_k=uint32:5",
                "--override-kv", "custom.flag=bool:true");
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("from flag", p.getString("general.description"));
            assertEquals(GGUFValueType.UINT32, p.getValueType("general.sampling.top_k"));
            assertEquals(5L, p.getUInt32("general.sampling.top_k").longValue());
            assertEquals(Boolean.TRUE, p.getBool("custom.flag"));
        }
        assertMentions(usage(model, out("bad-override.gguf"), "--override-kv", "nope"), "--override-kv");
    }

    @Test
    public void modelNameWinsOverTheMetadataFile() throws Exception {
        Path meta = file("name.json", "{\"general.name\": \"From The File\"}");
        Path out = out("named.gguf");
        export(checkpoint256(), out, "--model-name", "My Renamed Model", "--metadata", meta);
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("My Renamed Model", p.getString(GgufKeys.GENERAL_NAME));
        }
    }

    @Test
    public void metadataRefusalsNameTheFlag() throws Exception {
        Path model = checkpoint256();
        assertMentions(usage(model, out("m1.gguf"), "--metadata", out("nope.json")), "--metadata",
                "not a readable file");
        assertMentions(usage(model, out("m2.gguf"), "--metadata", file("bad.json", "{oops")), "not valid JSON");
        assertMentions(usage(model, out("m3.gguf"), "--metadata", file("arr.json", "[1, 2]")), "not a JSON object");
        assertMentions(usage(model, out("m4.gguf"), "--metadata", file("nest.json", "{\"a\": {\"b\": 1}}")),
                "--metadata", "'a'", "JSON object");
        assertMentions(usage(model, out("m5.gguf"), "--metadata", file("k.json", "{\"general.sampling.top_k\": 0.5}")),
                "'general.sampling.top_k'", "not an integer");
        assertFalse(Files.exists(out("m4.gguf")));
    }

    // ================================================================== output name template

    @Test
    public void theOutputNameTakesTheExportedFileType() throws Exception {
        Path model = checkpoint256();
        List<Path> written = export(model, out("model-{ftype}.gguf"), "--outtype", "q8_0");
        assertEquals(List.of(out("model-q8_0.gguf")), written);
        assertEquals(List.of(out("model-BF16.gguf")), export(model, out("model-{FTYPE}.gguf")));
        assertTrue(Files.exists(out("model-q8_0.gguf")));
    }

    // ================================================================== --print-supported-models and --mmproj

    @Test
    public void printSupportedModelsListsTheConvertibleArchitectures() throws Exception {
        String printed = printed(() -> export("--print-supported-models"));
        assertMentions(new Exception(printed), "Supported models:", "TEXT models:", "  - LlamaForCausalLM",
                "MMPROJ models:");
        assertNull("printing needs no model", export("--print-supported-models"));
    }

    @Test
    public void mmprojOpensTheProjectorViewAndNamesAMissingConversion() throws Exception {
        Path model = checkpoint256();
        Path out = out("mmproj-out.gguf");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> export(model, out, "--mmproj"));
        assertMentions(e, "has no projector conversion");
        assertFalse(Files.exists(out));
        assertMentions(usage(model, out, "--mmproj", "--vocab-only"), "--vocab-only", "--mmproj");
    }

    // ================================================================== --lora

    @Test
    public void loraExportsThePeftAdapterForItsBase() throws Exception {
        Path base = checkpoint256();
        Path out = out("adapter.gguf");
        List<Path> written = export(adapter(null), out, "--lora", "--base", base, "--outtype", "f16");
        assertEquals(List.of(out), written);
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("adapter", p.getString(GgufKeys.GENERAL_TYPE));
            assertEquals("lora", p.getString(GgufKeys.ADAPTER_TYPE));
            assertEquals("llama", p.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertEquals(8f, p.getFloat(GgufKeys.ADAPTER_LORA_ALPHA), 0f);
            assertEquals(GGMLType.F16, p.getTensor("blk.0.attn_q.weight.lora_a").type());
            assertEquals(GGMLType.F16, p.getTensor("blk.0.attn_q.weight.lora_b").type());
            assertEquals(2, p.getTensorCount());
        }
    }

    @Test
    public void loraFindsALocalBaseNamedByTheAdapterConfig() throws Exception {
        Path base = checkpoint256();
        Path out = out("named-base.gguf");
        export(adapter(base.toAbsolutePath().toString()), out, "--lora");
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("lora", p.getString(GgufKeys.ADAPTER_TYPE));
            assertEquals(GGMLType.F32, p.getTensor("blk.0.attn_q.weight.lora_a").type());
        }
    }

    @Test
    public void loraBaseConfigOnlyDirectoryIsEnough() throws Exception {
        Path base = checkpoint256();
        Path configOnly = tmp.newFolder("config-only").toPath();
        Files.copy(base.resolve("config.json"), configOnly.resolve("config.json"));
        Path out = out("config-only.gguf");
        export(adapter(null), out, "--lora", "--base", configOnly);
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("llama", p.getString(GgufKeys.GENERAL_ARCHITECTURE));
        }
    }

    @Test
    public void loraRefusalsNameTheirFlags() throws Exception {
        Path base = checkpoint256();
        Path peft = adapter(null);
        assertMentions(usage(peft, out("a.gguf"), "--lora"), "--lora needs --base", "base_model_name_or_path");
        assertMentions(usage(adapter("no/such/base"), out("a.gguf"), "--lora"), "--lora needs --base",
                "not a local directory");
        assertMentions(usage(peft, out("a.gguf"), "--base", base), "--base", "add --lora");
        assertMentions(usage(peft, out("a.gguf"), "--lora", "--base", out("missing")), "--base", "does not exist");
        assertMentions(usage(peft, out("a.gguf"), "--lora", "--base", base, "--outtype", "tq2_0"),
                "--outtype tq2_0", "not an adapter type");
        assertMentions(usage(peft, out("a.gguf"), "--lora", "--mmproj"), "--lora", "--mmproj");
        assertMentions(usage(peft, out("a.gguf"), "--lora", "--vocab-only"), "--vocab-only");
        assertFalse(Files.exists(out("a.gguf")));
    }

    // ================================================================== --dspark, --mtp and --no-nextn

    @Test
    public void dsparkExportsTheDeepseekV41DraftOfItsTarget() throws Exception {
        Path model = HfDraftFixtures.deepseekV41TargetDspark().write(tmp.newFolder("v41-target").toPath());
        Path out = out("v41-dspark.gguf");
        assertEquals(List.of(out), export(model, out, "--dspark"));
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("dflash", p.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertEquals(Arch.DFLASH, Arch.fromGgufName(p.getString(GgufKeys.GENERAL_ARCHITECTURE)));
            assertEquals(2, p.getInt("dflash.block_count").intValue());
            assertEquals(8, p.getInt("dflash.block_size").intValue());
            assertEquals(1, p.getInt("dflash.attention.head_count_kv").intValue());
            assertEquals(Boolean.TRUE, p.getMetadata().get("dflash.has_confidence_head"));
            for (String name : new String[]{"fc.weight", "enc.output_norm.weight", "output_norm.weight",
                    "markov_w1.weight", "markov_w2.weight", "conf_proj.weight", "blk.0.attn_q_a.weight",
                    "blk.1.ffn_gate_exps.weight"}) {
                assertTrue(name + " is missing from " + names(p), p.getTensor(name) != null);
            }
            for (String name : names(p)) {
                assertFalse("an mtp tensor leaked: " + name, name.startsWith("mtp"));
            }
        }
    }

    @Test
    public void dsparkExportsTheMistralLarge4DraftOfItsTarget() throws Exception {
        Path model = HfDraftFixtures.mistralLarge4TargetDspark().write(tmp.newFolder("ml4-target").toPath());
        Path out = out("ml4-dspark.gguf");
        assertEquals(List.of(out), export(model, out, "--dspark"));
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("dflash", p.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertEquals(1, p.getInt("dflash.block_count").intValue());
            assertEquals(8, p.getInt("dflash.block_size").intValue());
            assertEquals(Boolean.FALSE, p.getMetadata().get("dflash.has_confidence_head"));
            for (String name : new String[]{"fc.weight", "enc.output_norm.weight", "output_norm.weight",
                    "mask_embd.weight", "markov_w1.weight", "markov_w2.weight", "joint_proj.weight",
                    "blk.0.attn_q.weight"}) {
                assertTrue(name + " is missing from " + names(p), p.getTensor(name) != null);
            }
            for (String name : names(p)) {
                assertFalse("an mtp_dspark tensor leaked: " + name, name.startsWith("mtp_dspark"));
            }
        }
    }

    @Test
    public void mtpExportsOnlyTheNextnHeadAndTheTensorsItShares() throws Exception {
        Path model = HfMtpFixtures.deepseekV4WithMtp().write(tmp.newFolder("v4-mtp").toPath());
        Path out = out("v4-mtp.gguf");
        assertEquals(List.of(out), export(model, out, "--mtp"));
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("deepseek4", p.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertEquals(1, p.getInt("deepseek4.nextn_predict_layers").intValue());
            assertTrue("the shared embedding", p.getTensor("token_embd.weight") != null);
            assertTrue("the shared output norm", p.getTensor("output_norm.weight") != null);
            assertTrue("the shared output head", p.getTensor("output.weight") != null);
            for (String name : names(p)) {
                assertTrue("a trunk tensor leaked: " + name, blockOf(name) >= 2 || blockOf(name) < 0);
            }
            assertTrue("the MTP block", p.getTensor("blk.2.attn_q_b.weight") != null);
            assertTrue("the MTP expert stack", p.getTensor("blk.2.ffn_gate_exps.weight") != null);
            assertTrue("the MTP projection", p.getTensor("blk.2.nextn.eh_proj.weight") != null);
        }
    }

    @Test
    public void noNextnDropsTheNextnTensorsAndItsKey() throws Exception {
        Path model = HfMtpFixtures.deepseekV4WithMtp().write(tmp.newFolder("v4-no-nextn").toPath());
        Path out = out("v4-no-nextn.gguf");
        assertEquals(List.of(out), export(model, out, "--no-nextn"));
        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals("deepseek4", p.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertNull("the NextN count", p.getInt("deepseek4.nextn_predict_layers"));
            for (String name : names(p)) {
                assertFalse("a NextN tensor leaked: " + name, name.startsWith("blk.2."));
            }
            assertTrue("the trunk experts", p.getTensor("blk.1.ffn_gate_exps.weight") != null);
        }
        assertEquals(List.of(out("v4-no-nextn-alias.gguf")),
                export(model, out("v4-no-nextn-alias.gguf"), "--no-mtp"));
    }

    @Test
    public void dsparkIsRefusedForATargetWithoutADSparkDraft() throws Exception {
        Path model = checkpoint256();
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> export(model, out("no-dspark.gguf"), "--dspark"));
        assertMentions(e, "--dspark is only supported for DeepseekV4ForCausalLM, DeepseekV41ForCausalLM",
                "Mistral Large 4");
        assertFalse(Files.exists(out("no-dspark.gguf")));
    }

    @Test
    public void mtpAndNoNextnAreRefusedForAClassWithoutTheMtpExport() throws Exception {
        Path v41 = HfDraftFixtures.deepseekV41TargetDspark().write(tmp.newFolder("v41-no-mtp").toPath());
        assertMentions(assertThrows(GgufFormatException.class,
                        () -> export(v41, out("v41-mtp.gguf"), "--mtp")),
                "--mtp / --no-nextn are not supported for DeepseekV41ForCausalLM");
        assertMentions(assertThrows(GgufFormatException.class,
                        () -> export(v41, out("v41-nonextn.gguf"), "--no-nextn")),
                "--mtp / --no-nextn are not supported for DeepseekV41ForCausalLM");
        assertFalse(Files.exists(out("v41-mtp.gguf")));
    }

    @Test
    public void theDraftFlagsAreMutuallyExclusiveAndNeedAConvertedCheckpoint() throws Exception {
        Path model = HfMtpFixtures.deepseekV4WithMtp().write(tmp.newFolder("v4-conflict").toPath());
        assertMentions(usage(model, out("c1.gguf"), "--mtp", "--dspark"),
                "--mtp, --no-nextn, and --dspark are mutually exclusive");
        assertMentions(usage(model, out("c2.gguf"), "--no-nextn", "--mtp"),
                "--mtp, --no-nextn, and --dspark are mutually exclusive");
        assertMentions(usage(model, out("c3.gguf"), "--dspark", "--mmproj"), "dspark", "--mmproj");
        Path gguf = out("plain-draft.gguf");
        export(checkpoint256(), gguf);
        assertMentions(usage(gguf, out("c4.gguf"), "--mtp"), "--mtp", "stored as GGUF");
        assertFalse(Files.exists(out("c1.gguf")));
    }

    /** The names of every tensor a parsed file holds, in order. */
    private static List<String> names(GGUFFileParser p) {
        List<String> out = new java.util.ArrayList<>();
        for (com.eonmux.neuralfree.gguf.GGUFTensorInfo t : p.getTensors()) {
            out.add(t.name());
        }
        return out;
    }

    /** The block number in a GGUF tensor name ({@code blk.<n>.}), or -1 for a model-level tensor. */
    private static int blockOf(String name) {
        if (!name.startsWith("blk.")) {
            return -1;
        }
        int dot = name.indexOf('.', 4);
        return Integer.parseInt(name.substring(4, dot));
    }

    // ================================================================== usage

    @Test
    public void helpPrintsEveryFlag() throws Exception {
        String printed = printed(() -> export("--help"));
        for (String flag : new String[]{"--outtype", "tq1_0", "tq2_0", "--mmproj", "--lora", "--base", "--dry-run",
                "--metadata", "--model-name", "--print-supported-models", "--no-tensor-first-split", "{ftype}",
                "--mtp", "--no-nextn", "--dspark"}) {
            assertTrue(flag, printed.contains(flag));
        }
        assertNull(export("--help"));
    }

    @Test
    public void optionConflictsAreUsageErrors() throws Exception {
        Path model = checkpoint256();
        assertMentions(usage(model, out("vs.gguf"), "--vocab-only", "--no-tensor-first-split"),
                "--no-tensor-first-split");
        assertMentions(usage(model, out("m.gguf"), "--mode", "fast"), "--mode is invalid");
        assertMentions(usage(model, out("m.gguf"), "--frobnicate"), "unknown option --frobnicate");
        assertFalse(Files.exists(out("vs.gguf")));
    }

    @Test
    public void missingArgumentsKeepTheirNamedErrors() {
        assertMentions(usage("--outtype"), "--outtype needs a value");
        assertMentions(usage("--base"), "--base needs a value");
        assertMentions(usage(), "export needs a model path");
        assertMentions(usage("m"), "export needs an output path after m");
    }
}
