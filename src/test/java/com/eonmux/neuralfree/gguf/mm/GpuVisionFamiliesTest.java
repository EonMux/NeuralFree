package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The GPU vision encoder against the CPU encoder on every runnable vision family beyond those of
 * {@link GpuVisionEncoderTest}: one small synthetic mmproj file per family, written with F32 tensors and again
 * with F16 matrices, opened through {@link MmprojModel#openVisionEncoder(ComputeBackend)} on a shared OpenCL
 * device. Both encoders run the shared family graphs of {@link VisionFamilies}, so the rows agree within F32
 * rounding (F16 files: both read the same F16 weights); a second GPU encode with the reused buffers repeats the
 * rows exactly. The CPU encoder itself is checked against the double-precision references by the
 * {@code VisionFamilies*Test} classes.
 */
public class GpuVisionFamiliesTest {

    /** Absolute plus relative tolerance of the F32 files. */
    private static final float F32_TOL = 1e-3f;
    /** Absolute plus relative tolerance of the F16 files. */
    private static final float F16_TOL = 2e-3f;

    /** The projector types {@link GpuVisionEncoderTest} covers. */
    private static final List<String> ENCODER_TEST_TYPES = Arrays.asList("mlp", "mlp_norm", "gemma3",
            "qwen2vl_merger", "qwen2.5vl_merger");

    /** The types the dedicated {@link #yasa2MatchesTheCpuEncoder()} and SAM tests cover. */
    private static final List<String> DEDICATED_TEST_TYPES = Arrays.asList("yasa2", "deepseekocr", "deepseekocr2");

    /** Two full-attention mimovl blocks. */
    private static final int[] MIMO_FULL = {-1, -1};

    @ClassRule
    public static final TemporaryFolder TMP = new TemporaryFolder();

    private static HWAcceleration acceleration;
    private static ComputeBackend backend;

    @BeforeClass
    public static void openDevice() {
        if (!GpuTestUtil.isOpenCLAvailable()) {
            return;
        }
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try {
            backend = acceleration.createBackend(Config.ACCELERATION_MODE.OPENCL, LlmKernels.requirement());
        } catch (RuntimeException e) {
            backend = null;
        }
    }

    @AfterClass
    public static void closeDevice() {
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static void assumeGpu() {
        Assume.assumeTrue("no OpenCL device running the LLM kernels", backend != null);
    }

    // ================================================================== harness

    private static Path write(MmprojFixture f, Map<String, Object> meta, boolean f16) throws IOException {
        Path file = TMP.newFolder().toPath().resolve("mmproj.gguf");
        VisionFamilyFixtures.gguf(f, meta, f16).writeTo(file);
        return file;
    }

    /**
     * Encodes a {@code width x height} image of the family with its separator {@code suffix} on the CPU and the
     * GPU, over the F32 file and the F16 file, and checks the row count, the CPU parity and the repeat.
     */
    private static void parity(MmprojFixture f, Map<String, Object> meta, int width, int height, ClipSuffix suffix,
                               int tokens) throws IOException {
        ClipImage image = ClipImage.of(width, height, MmprojFixture.pixels(width, height, 31 * width + height))
                .withSuffix(suffix);
        parityImage(f, meta, image, tokens);
    }

    /**
     * Encodes {@code image} of the family on the CPU and the GPU, over the F32 file and the F16 file, and checks
     * the row count, the CPU parity and the repeat.
     */
    private static void parityImage(MmprojFixture f, Map<String, Object> meta, ClipImage image, int tokens)
            throws IOException {
        assumeGpu();
        for (boolean f16 : new boolean[]{false, true}) {
            try (MmprojModel mm = MmprojModel.open(write(f, meta, f16).toString());
                 CpuVisionEncoder cpu = mm.openCpuVisionEncoder();
                 VisionEncoder gpu = mm.openVisionEncoder(backend)) {
                assertTrue(gpu.toString(), gpu instanceof GpuVisionEncoder);
                String what = mm.visionHparams().projectorType().typeName() + (f16 ? " F16" : " F32");
                float[][] want = cpu.encode(image);
                float[][] got = gpu.encode(image);
                assertEquals(what, tokens, got.length);
                assertClose(what, want, got, f16 ? F16_TOL : F32_TOL);
                assertRowsEqual(what, got, gpu.encode(image));
            }
        }
    }

    static void assertClose(String what, float[][] want, float[][] got, float tol) {
        assertEquals(what + ": rows", want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertEquals(what + ": row " + t + " width", want[t].length, got[t].length);
            for (int c = 0; c < want[t].length; c++) {
                float diff = Math.abs(want[t][c] - got[t][c]);
                if (!(diff <= tol + tol * Math.abs(want[t][c]))) {
                    fail(what + ": row " + t + " col " + c + ": cpu " + want[t][c] + ", gpu " + got[t][c]);
                }
            }
        }
    }

    private static void assertRowsEqual(String what, float[][] want, float[][] got) {
        assertEquals(what, want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertArrayEquals(what + ": repeat row " + t, want[t], got[t], 0f);
        }
    }

    // ================================================================== cases

    /** One family's fixture, metadata, image size, separator suffix and output row count. */
    private static final class Case {
        final String name;
        final Supplier<MmprojFixture> fixture;
        final Supplier<Map<String, Object>> meta;
        final int width;
        final int height;
        final ClipSuffix suffix;
        final int tokens;

        Case(String name, Supplier<MmprojFixture> fixture, Supplier<Map<String, Object>> meta, int width, int height,
             int tokens) {
            this(name, fixture, meta, width, height, ClipSuffix.NONE, tokens);
        }

        Case(String name, Supplier<MmprojFixture> fixture, Supplier<Map<String, Object>> meta, int width, int height,
             ClipSuffix suffix, int tokens) {
            this.name = name;
            this.fixture = fixture;
            this.meta = meta;
            this.width = width;
            this.height = height;
            this.suffix = suffix;
            this.tokens = tokens;
        }
    }

    private static final List<Case> CASES = Arrays.asList(
            new Case("minicpmv", () -> VisionFamilyFixtures.minicpmv(11), VisionFamilyFixtures::minicpmvMeta, 12, 8,
                    VisionFamilyFixtures.QUERIES),
            new Case("internvl.per_head", () -> VisionFamilyFixtures.internvl(12, "per_head"),
                    VisionFamilyFixtures::internvlMeta, 8, 8, 4),
            new Case("internvl", () -> VisionFamilyFixtures.internvl(13, null), VisionFamilyFixtures::internvlMeta,
                    8, 8, 4),
            new Case("pixtral.merger", () -> VisionFamilyFixtures.pixtral(13, true),
                    () -> VisionFamilyFixtures.pixtralMeta(true), 8, 12, 2 * 3 + 2),
            new Case("pixtral", () -> VisionFamilyFixtures.pixtral(14, false),
                    () -> VisionFamilyFixtures.pixtralMeta(false), 4, 4, 2 * 2 + 1),
            new Case("lightonocr", () -> VisionFamiliesCFixtures.lightonocr(101),
                    VisionFamiliesCFixtures::lightonocrMeta, 12, 8, 6),
            new Case("llama4", () -> VisionFamilyFixtures.llama4(16), VisionFamilyFixtures::llama4Meta, 8, 8, 4),
            new Case("qwen3vl", () -> VisionFamiliesCFixtures.qwen3vl(91), VisionFamiliesCFixtures::qwen3vlMeta, 12,
                    8, 6),
            new Case("paddleocr", () -> VisionFamiliesCFixtures.paddleocr(92), VisionFamiliesCFixtures::paddleocrMeta,
                    8, 12, 6),
            new Case("kimivl", () -> VisionFamilyFixturesB.kimivl(27), VisionFamilyFixturesB::kimivlMeta, 12, 8, 6),
            // a 5x4 patch grid: the merge permute zero-pads it to 6x4
            new Case("kimivl.padded", () -> VisionFamiliesCFixtures.kimivl(93), VisionFamiliesCFixtures::kimivlMeta,
                    10, 8, 6),
            // the bicubic-resized position grid, then the stored one
            new Case("kimik25", () -> VisionFamiliesDFixtures.kimik25(201), VisionFamiliesDFixtures::kimik25Meta, 8,
                    12, 6),
            new Case("kimik25.stored_grid", () -> VisionFamiliesDFixtures.kimik25(202),
                    VisionFamiliesDFixtures::kimik25Meta, 6, 4, 2),
            // clamped blocks, post norms, output scale, value norm, NEOX rope; the 2x2 and the default 3x3 pool
            new Case("gemma4v", () -> VisionFamiliesDFixtures.gemma4v(211),
                    () -> VisionFamiliesDFixtures.gemma4vMeta(2), 8, 12, 6),
            new Case("gemma4v.pool3", () -> VisionFamiliesDFixtures.gemma4v(212),
                    () -> VisionFamiliesDFixtures.gemma4vMeta(null), 12, 6, 2),
            new Case("gemma4uv", () -> VisionFamiliesDFixtures.gemma4uv(213), VisionFamiliesDFixtures::gemma4uvMeta,
                    16, 12, 12),
            // the ViT merger's window attention and downsample, the 2x merge without it, the separator rows
            new Case("minicpmv4_6", () -> VisionFamiliesDFixtures.minicpmv46(221, true, false),
                    () -> VisionFamiliesDFixtures.minicpmv46Meta("minicpmv4_6", null, null), 24, 16, 6),
            new Case("minicpmv4_7", () -> VisionFamiliesDFixtures.minicpmv46(222, true, false),
                    () -> VisionFamiliesDFixtures.minicpmv46Meta("minicpmv4_7", 4, 1), 16, 24,
                    ClipSuffix.MINICPMV_ROW_END, 6 + ClipSuffix.MINICPMV_ROW_END.size()),
            new Case("minicpm_ocr_1_0", () -> VisionFamiliesDFixtures.minicpmv46(224, false, true),
                    () -> VisionFamiliesDFixtures.minicpmv46Meta("minicpm_ocr_1_0", 2, null), 12, 8, 6),
            // the GPU tower runs mimovl's full-attention blocks of n_embd-wide heads; sinks and grouped heads stay
            // on the CPU
            new Case("mimovl.full", () -> VisionFamiliesDFixtures.mimovl(241, 2, 2, 4, MIMO_FULL),
                    () -> VisionFamiliesDFixtures.mimovlMeta(5, 2, 2, MIMO_FULL), 12, 8, 6),
            new Case("dots_ocr", () -> VisionFamilyFixturesB.dotsOcr(30), VisionFamilyFixturesB::dotsOcrMeta, 8, 12,
                    6),
            new Case("dots_ocr.square", () -> VisionFamiliesCFixtures.dotsOcr(94), VisionFamiliesCFixtures::dotsOcrMeta,
                    8, 8, 4),
            new Case("dots3note_v", () -> VisionFamiliesCFixtures.dots3note(95),
                    VisionFamiliesCFixtures::dots3noteMeta, 12, 8, 6),
            new Case("hunyuanvl", () -> VisionFamiliesCFixtures.hunyuanvl(96), VisionFamiliesCFixtures::hunyuanvlMeta,
                    8, 8, (2 + 1) * 2 + 2),
            new Case("cogvlm", () -> VisionFamilyFixturesB.cogvlm(25), VisionFamilyFixturesB::cogvlmMeta, 8, 8, 6),
            new Case("glm4v", () -> VisionFamilyFixturesB.glm4v(26, false), VisionFamilyFixturesB::glm4vMeta, 16, 8,
                    8),
            new Case("glm4v.ocr", () -> VisionFamilyFixturesB.glm4v(27, true), VisionFamilyFixturesB::glm4vMeta, 16,
                    8, 8),
            new Case("youtuvl", () -> VisionFamilyFixturesB.youtuvl(31), () -> VisionFamilyFixturesB.youtuvlMeta(1),
                    16, 8, 8),
            new Case("youtuvl.windowed", () -> VisionFamilyFixturesB.youtuvl(32), VisionFamilyFixturesB::youtuvlMeta,
                    16, 8, 8),
            new Case("granite4_vision", () -> VisionFamilyFixturesB.granite4(32), VisionFamilyFixturesB::granite4Meta,
                    16, 16, 16),
            new Case("llava.stacked", () -> VisionFamilyFixturesB.llavaStacked(21),
                    VisionFamilyFixturesB::llavaStackedMeta, 16, 16, 16),
            new Case("ldp", () -> VisionFamilyFixturesB.ldp(22), () -> VisionFamilyFixturesB.llavaMeta("ldp"), 16, 16,
                    4),
            new Case("ldpv2", () -> VisionFamilyFixturesB.ldpv2(23), () -> VisionFamilyFixturesB.llavaMeta("ldpv2"),
                    16, 16, 4),
            new Case("adapter", () -> VisionFamilyFixturesB.glmEdge(24),
                    () -> VisionFamilyFixturesB.llavaMeta("adapter"), 16, 16, 6),
            new Case("idefics3", () -> VisionFamilyFixtures.idefics3(15), VisionFamilyFixtures::idefics3Meta, 16, 16,
                    4),
            new Case("phi4", () -> VisionFamilyFixtures.phi4(17), VisionFamilyFixtures::phi4Meta, 20, 12, 15),
            new Case("lfm2", () -> VisionFamilyFixturesB.lfm2(28, false), () -> VisionFamilyFixturesB.lfm2Meta("lfm2"),
                    12, 8, 6),
            new Case("d1omni_v", () -> VisionFamilyFixturesB.lfm2(29, true),
                    () -> VisionFamilyFixturesB.lfm2Meta("d1omni_v"), 8, 8, 4),
            new Case("lfm2.input_norm", () -> VisionFamiliesCFixtures.lfm2(98, true),
                    () -> VisionFamiliesCFixtures.lfm2Meta("lfm2"), 8, 12, 6),
            new Case("d1omni_v.plain", () -> VisionFamiliesCFixtures.lfm2(99, false),
                    () -> VisionFamiliesCFixtures.lfm2Meta("d1omni_v"), 8, 8, 4),
            new Case("janus_pro", () -> VisionFamiliesCFixtures.janus(97), VisionFamiliesCFixtures::janusMeta, 8, 8,
                    16),
            new Case("cohere2v", () -> VisionFamiliesCFixtures.cohere2v(100), VisionFamiliesCFixtures::cohere2vMeta, 8,
                    8, 4),
            new Case("glm5v", () -> VisionFamiliesFFixtures.glm5v(901), VisionFamiliesFFixtures::glm5vMeta, 8, 12, 6),
            new Case("lfm3vl", () -> VisionFamiliesFFixtures.lfm3vl(902), VisionFamiliesFFixtures::lfm3vlMeta, 12, 8,
                    6),
            new Case("ling3vl", () -> VisionFamiliesFFixtures.ling3vl(903), VisionFamiliesFFixtures::ling3vlMeta, 8,
                    12, 6),
            new Case("minimax_m3", () -> VisionFamiliesFFixtures.minimaxM3(904),
                    VisionFamiliesFFixtures::minimaxM3Meta, 12, 8, 6),
            new Case("muse-glimmer", () -> VisionFamiliesFFixtures.museGlimmer(905),
                    VisionFamiliesFFixtures::museGlimmerMeta, 12, 8, 6),
            new Case("nemotron_v2_vl", () -> VisionFamiliesFFixtures.nemotronV2Vl(906),
                    VisionFamiliesFFixtures::nemotronV2VlMeta, 8, 8, 4),
            new Case("nemotron_omni", () -> VisionFamiliesFFixtures.nemotronOmni(907),
                    VisionFamiliesFFixtures::nemotronOmniMeta, 8, 12, 6),
            new Case("exaone4_5", () -> VisionFamiliesFFixtures.exaone45(908), VisionFamiliesFFixtures::exaone45Meta,
                    168, 112, 24),
            new Case("sensenova_u1", () -> VisionFamiliesEFixtures.sensenova(201),
                    VisionFamiliesEFixtures::sensenovaMeta, 12, 8, 6),
            new Case("step3vl", () -> VisionFamiliesEFixtures.step3vl(211, true), VisionFamiliesEFixtures::step3vlMeta,
                    16, 16, 4),
            new Case("step3vl.nobias", () -> VisionFamiliesEFixtures.step3vl(213, false),
                    VisionFamiliesEFixtures::step3vlMeta, 8, 16, 2),
            new Case("deepseek4v", () -> VisionFamiliesEFixtures.deepseekV4(221, true),
                    () -> VisionFamiliesEFixtures.deepseekV4Meta("deepseek4v"), 10, 6, 10),
            new Case("deepseek4v.oddrows", () -> VisionFamiliesEFixtures.deepseekV4(222, true),
                    () -> VisionFamiliesEFixtures.deepseekV4Meta("deepseek4v"), 8, 12, 14),
            new Case("deepseek41v", () -> VisionFamiliesEFixtures.deepseekV4(223, false),
                    () -> VisionFamiliesEFixtures.deepseekV4Meta("deepseek41v"), 10, 6, 10),
            // the MobileNetV5 conv stack: the fusion upscale (4x2 and 2x1 features), the 32x32 fusion map
            // pooled 2x2 to 16x16, and the three-stage layout whose last block alone feeds the adapter
            new Case("gemma3nv", () -> VisionFamiliesDFixtures.gemma3nv(251, true),
                    () -> VisionFamiliesDFixtures.gemma3nvMeta(16), 64, 32, 8),
            new Case("gemma3nv.pool", () -> VisionFamiliesDFixtures.gemma3nv(252, true),
                    () -> VisionFamiliesDFixtures.gemma3nvMeta(512), 512, 512, 256),
            new Case("gemma3nv.three_stages", () -> VisionFamiliesDFixtures.gemma3nv(253, false),
                    () -> VisionFamiliesDFixtures.gemma3nvMeta(8), 32, 32, 4));

    private static void run(String... names) throws IOException {
        for (String name : names) {
            Case c = null;
            for (Case k : CASES) {
                if (k.name.equals(name)) {
                    c = k;
                }
            }
            if (c == null) {
                throw new IllegalArgumentException("no case " + name);
            }
            parity(c.fixture.get(), c.meta.get(), c.width, c.height, c.suffix, c.tokens);
        }
    }

    // ================================================================== families with their own encoder graphs

    @Test
    public void minicpmvResamplerMatchesTheCpuEncoder() throws IOException {
        run("minicpmv");
    }

    @Test
    public void internvlMatchesTheCpuEncoder() throws IOException {
        run("internvl.per_head", "internvl");
    }

    @Test
    public void pixtralAndLightonocrMatchTheCpuEncoder() throws IOException {
        run("pixtral.merger", "pixtral", "lightonocr");
    }

    @Test
    public void llama4MatchesTheCpuEncoder() throws IOException {
        run("llama4");
    }

    @Test
    public void qwen3vlDeepstackMatchesTheCpuEncoder() throws IOException {
        run("qwen3vl");
    }

    @Test
    public void paddleocrMatchesTheCpuEncoder() throws IOException {
        run("paddleocr");
    }

    @Test
    public void kimivlMatchesTheCpuEncoder() throws IOException {
        run("kimivl", "kimivl.padded");
    }

    @Test
    public void kimik25MatchesTheCpuEncoder() throws IOException {
        run("kimik25", "kimik25.stored_grid");
    }

    @Test
    public void gemma4vAndGemma4uvMatchTheCpuEncoder() throws IOException {
        run("gemma4v", "gemma4v.pool3", "gemma4uv");
    }

    @Test
    public void minicpmvMergersMatchTheCpuEncoder() throws IOException {
        run("minicpmv4_6", "minicpmv4_7", "minicpm_ocr_1_0");
    }

    @Test
    public void mimovlFullAttentionBlocksMatchTheCpuEncoder() throws IOException {
        run("mimovl.full");
    }

    /** The GPU flash attention has no sinks and projects Q, K and V to {@code n_embd}: both refused by name. */
    @Test
    public void mimovlSinksAndGroupedHeadsAreRefusedOnTheGpu() {
        ClipVisionModel grouped = ClipVisionModel.from(
                VisionFamilyFixtures.hparams(VisionFamiliesDFixtures.mimovlMeta(5)),
                VisionFamiliesDFixtures.mimovl(242).catalog());
        String reason = GpuVisionEncoder.unsupportedReason(grouped, 4096, 8);
        assertTrue(reason, reason != null && reason.contains("mimovl") && reason.contains("key/value heads"));
        int[] windowed = {0, -1};
        ClipVisionModel sinks = ClipVisionModel.from(
                VisionFamilyFixtures.hparams(VisionFamiliesDFixtures.mimovlMeta(5, 2, 2, windowed)),
                VisionFamiliesDFixtures.mimovl(243, 2, 2, 4, windowed).catalog());
        reason = GpuVisionEncoder.unsupportedReason(sinks, 4096, 8);
        assertTrue(reason, reason != null && reason.contains("v.blk.0.attn_sinks"));
    }

    // ================================================================== convolutional encoders

    /**
     * The yasa2 ConvNeXt on the GPU: its depthwise 7x7 convolutions, response norms and 8x8 pool over the
     * {@link Yasa2Fixture} files (every optional tensor, then the sparse file whose final grid pools with
     * kernel 2), F32 and F16.
     */
    @Test
    public void yasa2MatchesTheCpuEncoder() throws IOException {
        assumeGpu();
        for (Yasa2Fixture fx : new Yasa2Fixture[]{Yasa2Fixture.full(), Yasa2Fixture.sparse()}) {
            ClipImage image = ClipImage.of(fx.imageSize, fx.imageSize, fx.pixels());
            for (boolean f16 : new boolean[]{false, true}) {
                Path file = TMP.newFolder().toPath().resolve("mmproj.gguf");
                GgufTestWriter w = f16 ? VisionFamilyFixtures.gguf(fx.tensors, fx.meta, true) : fx.gguf();
                w.writeTo(file);
                try (MmprojModel mm = MmprojModel.open(file.toString());
                     CpuVisionEncoder cpu = mm.openCpuVisionEncoder();
                     VisionEncoder gpu = mm.openVisionEncoder(backend)) {
                    assertTrue(gpu.toString(), gpu instanceof GpuVisionEncoder);
                    String what = "yasa2 " + fx.name + (f16 ? " F16" : " F32");
                    float[][] want = cpu.encode(image);
                    float[][] got = gpu.encode(image);
                    assertEquals(what, Yasa2Graph.POOLED_SIDE * Yasa2Graph.POOLED_SIDE, got.length);
                    assertClose(what, want, got, f16 ? F16_TOL : F32_TOL);
                    assertRowsEqual(what, got, gpu.encode(image));
                }
            }
        }
    }

    /**
     * The DeepSeek-OCR SAM backbone on the GPU: a 128-pixel global view (a 2x2 SAM grid, its newline rows and
     * the separator), the same tile as a row of one tile, and a row of two 128-pixel tiles over the padded
     * windows.
     */
    @Test
    public void deepseekOcrSamBackboneMatchesTheCpuEncoder() throws IOException {
        MmprojFixture f = VisionFamiliesEFixtures.deepseekOcr(231);
        Map<String, Object> meta = VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr");
        parityImage(f, meta, ClipImage.of(128, 128, MmprojFixture.pixels(128, 128, 31 * 128 + 128))
                .withViewSeparator(true), 7);
        parityImage(f, meta, ClipImage.of(128, 128, MmprojFixture.pixels(128, 128, 31 * 128 + 128)), 6);
        parityImage(f, meta, ClipImage.of(128, 256, MmprojFixture.pixels(128, 256, 31 * 128 + 256)), 10);
    }

    /**
     * The DeepSeek-OCR 2 SAM backbone and its query rows on the GPU: the 768-pixel image (144 rows) and the
     * 1024-pixel global view, whose whole-grid attention block runs 4096 SAM rows.
     */
    @Test
    public void deepseekOcr2QueriesMatchTheCpuEncoder() throws IOException {
        MmprojFixture f = VisionFamiliesEFixtures.deepseekOcr2(233);
        Map<String, Object> meta = VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr2");
        parityImage(f, meta, ClipImage.of(768, 768, MmprojFixture.pixels(768, 768, 31 * 768 + 768)), 144);
        parityImage(f, meta, ClipImage.of(1024, 1024, MmprojFixture.pixels(1024, 1024, 31 * 1024 + 1024))
                .withViewSeparator(true), DeepseekOcrGraph.QUERIES_1024 + 1);
    }

    @Test
    public void dotsOcrMatchesTheCpuEncoder() throws IOException {
        run("dots_ocr", "dots_ocr.square");
    }

    @Test
    public void dots3noteMixtureOfExpertsBlocksMatchTheCpuEncoder() throws IOException {
        MmprojFixture f = VisionFamiliesCFixtures.dots3note(95);
        ClipVisionModel model = ClipVisionModel.from(VisionFamilyFixtures.hparams(
                VisionFamiliesCFixtures.dots3noteMeta()), f.catalog());
        boolean dense = false;
        boolean moe = false;
        for (ClipLayer l : model.layers()) {
            moe |= l.moe();
            dense |= !l.moe();
        }
        assertTrue("the fixture mixes dense and MoE blocks", dense && moe);
        run("dots3note_v");
    }

    @Test
    public void hunyuanvlPerceiverMatchesTheCpuEncoder() throws IOException {
        run("hunyuanvl");
    }

    @Test
    public void cogvlmPostNormBlocksMatchTheCpuEncoder() throws IOException {
        run("cogvlm");
    }

    @Test
    public void glm4vMatchesTheCpuEncoder() throws IOException {
        run("glm4v", "glm4v.ocr");
    }

    @Test
    public void youtuvlWindowLayersMatchTheCpuEncoder() throws IOException {
        run("youtuvl", "youtuvl.windowed");
    }

    @Test
    public void granite4TileAndAnyresAssemblyMatchTheCpuEncoder() throws IOException {
        run("granite4_vision");
        ClipImage overview = ClipImage.of(16, 16, MmprojFixture.pixels(16, 16, 1016));
        List<ClipImage> tiles = Arrays.asList(ClipImage.of(16, 16, MmprojFixture.pixels(16, 16, 1017)),
                ClipImage.of(16, 16, MmprojFixture.pixels(16, 16, 1018)));
        Path file = write(VisionFamilyFixturesB.granite4(32), VisionFamilyFixturesB.granite4Meta(), false);
        try (MmprojModel mm = MmprojModel.open(file.toString());
             CpuVisionEncoder cpu = mm.openCpuVisionEncoder();
             VisionEncoder gpu = mm.openVisionEncoder(backend)) {
            assertTrue(gpu instanceof GpuVisionEncoder);
            float[][] want = cpu.encodeAnyres(overview, tiles, 2, 1, 30, 10);
            float[][] got = gpu.encodeAnyres(overview, tiles, 2, 1, 30, 10);
            assertEquals(16 + 2 * (8 + 1), got.length);
            assertClose("granite4_vision anyres", want, got, F32_TOL);
        }
    }

    // ================================================================== llava-graph projectors

    @Test
    public void llavaStackedFeatureLayersMatchTheCpuEncoder() throws IOException {
        run("llava.stacked");
    }

    @Test
    public void ldpAndLdpv2MatchTheCpuEncoder() throws IOException {
        run("ldp", "ldpv2");
    }

    @Test
    public void glmEdgeAdapterMatchesTheCpuEncoder() throws IOException {
        run("adapter");
    }

    // ================================================================== SigLIP-encoder projectors

    @Test
    public void idefics3AndPhi4MatchTheCpuEncoder() throws IOException {
        run("idefics3", "phi4");
    }

    @Test
    public void lfm2AndD1omniMatchTheCpuEncoder() throws IOException {
        run("lfm2", "d1omni_v", "lfm2.input_norm", "d1omni_v.plain");
    }

    @Test
    public void janusAndCohere2vMatchTheCpuEncoder() throws IOException {
        run("janus_pro", "cohere2v");
    }

    @Test
    public void glm5vClampedSwiGluMatchesTheCpuEncoder() throws IOException {
        run("glm5v");
    }

    @Test
    public void lfm3vlConvAndAttentionBlocksMatchTheCpuEncoder() throws IOException {
        run("lfm3vl");
    }

    @Test
    public void ling3vlMergerNormMatchesTheCpuEncoder() throws IOException {
        run("ling3vl");
    }

    @Test
    public void minimaxM3SplitRopeMatchesTheCpuEncoder() throws IOException {
        run("minimax_m3");
    }

    @Test
    public void museGlimmerWindowLayersMatchTheCpuEncoder() throws IOException {
        run("muse-glimmer");
    }

    @Test
    public void nemotronRegisterTowersMatchTheCpuEncoder() throws IOException {
        run("nemotron_v2_vl", "nemotron_omni");
    }

    @Test
    public void exaone45GqaWindowLayersMatchTheCpuEncoder() throws IOException {
        run("exaone4_5");
    }

    @Test
    public void sensenovaRotatedPatchConvMatchesTheCpuEncoder() throws IOException {
        run("sensenova_u1");
    }

    @Test
    public void step3vlDownsamplersMatchTheCpuEncoder() throws IOException {
        run("step3vl", "step3vl.nobias");
    }

    @Test
    public void deepseekV4AlignerBlocksMatchTheCpuEncoder() throws IOException {
        run("deepseek4v", "deepseek4v.oddrows", "deepseek41v");
    }

    // ================================================================== coverage

    /**
     * Every runnable vision type of the registry has a GPU parity case here, in {@link GpuVisionEncoderTest} or
     * in the yasa2 files of {@link #yasa2MatchesTheCpuEncoder()}, and the device limits admit each case
     * ({@link GpuVisionEncoder#unsupportedReason} at the 4096 shared floats and 8 compute units of a small
     * device). The convolutional encoders (yasa2, the DeepSeek-OCR SAM backbones, MobileNetV5) run their device
     * kernels, so nothing is CPU-only any more.
     */
    @Test
    public void everyRunnableVisionTypeHasAGpuParityCase() {
        Set<String> runnable = new TreeSet<>();
        for (ProjectorEntry e : ProjectorRegistry.entries()) {
            if (e.runnable() && e.modality() == ProjectorModality.VISION) {
                runnable.add(e.type().typeName());
            }
        }
        Set<String> covered = new HashSet<>(ENCODER_TEST_TYPES);
        covered.addAll(DEDICATED_TEST_TYPES);
        for (Case c : CASES) {
            Map<String, Object> meta = c.meta.get();
            covered.add((String) meta.get("clip.projector_type"));
            ClipVisionModel model = ClipVisionModel.from(VisionFamilyFixtures.hparams(meta), c.fixture.get().catalog());
            assertNull(c.name, GpuVisionEncoder.unsupportedReason(model, 4096, 8));
        }
        Yasa2Fixture fx = Yasa2Fixture.full();
        ClipVisionModel yasa2 = ClipVisionModel.from(fx.hparams(), fx.tensors.catalog());
        assertNull(GpuVisionEncoder.unsupportedReason(yasa2, 1 << 14, 64));
        ClipVisionModel ocr = ClipVisionModel.from(VisionFamilyFixtures.hparams(
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr")),
                VisionFamiliesEFixtures.deepseekOcr(231).catalog());
        assertNull(GpuVisionEncoder.unsupportedReason(ocr, 4096, 8));
        ClipVisionModel ocr2 = ClipVisionModel.from(VisionFamilyFixtures.hparams(
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr2")),
                VisionFamiliesEFixtures.deepseekOcr2(233).catalog());
        assertNull(GpuVisionEncoder.unsupportedReason(ocr2, 4096, 8));
        Set<String> missing = new TreeSet<>(runnable);
        missing.removeAll(covered);
        assertTrue("runnable vision types without a GPU parity case: " + missing, missing.isEmpty());
    }
}
