package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;
import com.eonmux.neuralfree.gguf.testutil.SyntheticMlaModels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticRoutedModels;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * The GPU collection of {@code nftool imatrix} ({@code -ngl}) against the CPU collection of the same synthetic
 * model: the device matmul tap reads every projection input back, the routed expert products report one row per
 * (token, slot) pair, the per-head products of the MLA families report one matrix per head, routed experts kept in
 * host memory ({@code --cpu-moe}, the MoE expert cache) reach the CPU tap, the layers a partial offload keeps on the
 * host are named through the GPU model's host weights, and the NextN head replays over the device trunk — each
 * giving the CPU run's counts and sums.
 */
public class ImatrixGpuParityTest {

    private static final int CTX = 32;
    private static final float TOLERANCE = 2e-3f;
    /** The absorbed per-head weights of both layers of the synthetic MLA models, four heads each. */
    private static final String[] MLA_HEADS = {"blk.0.attn_k_b.weight", "4", "blk.0.attn_v_b.weight", "4",
            "blk.1.attn_k_b.weight", "4", "blk.1.attn_v_b.weight", "4"};

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Before
    public void requireGpu() {
        Assume.assumeTrue("no GPU backend", GpuTestUtil.isOpenCLAvailable() || GpuTestUtil.isCudaAvailable());
    }

    @Test
    public void denseModelCollectsTheSameStatisticsOnTheGpu() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path corpus = corpus();
        requireGpuEngine(model, corpus);
        String[] batch = {"--batch-size", String.valueOf(CTX), "--process-output"};
        ImatrixFile.Read cpu = ImatrixFile.read(collect(model, corpus, "cpu.gguf", concat(batch, "-ngl", "0")));
        ImatrixFile.Read gpu = ImatrixFile.read(collect(model, corpus, "gpu.gguf", concat(batch, "-ngl", "all")));
        ImatrixFile.Read partial = ImatrixFile.read(collect(model, corpus, "partial.gguf",
                concat(batch, "-ngl", "1")));
        assertNotNull("the head ran on the device", ImatrixRunTest.entry(gpu, "output.weight"));
        ImatrixParity.assertSameStatistics("all layers on the GPU", cpu, gpu, TOLERANCE);
        ImatrixParity.assertSameStatistics("one layer on the GPU", cpu, partial, TOLERANCE);
    }

    @Test
    public void moeModelWithItsNextnHeadCollectsTheSameStatisticsOnTheGpu() throws Exception {
        Path model = SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe(), 1).writeTempFile();
        Path corpus = corpus();
        requireGpuEngine(model, corpus);
        String[] nextn = {"--nextn", "--batch-size", String.valueOf(CTX)};
        ImatrixFile.Read cpu = ImatrixFile.read(collect(model, corpus, "cpu.gguf", concat(nextn, "-ngl", "0")));
        ImatrixFile.Read gpu = ImatrixFile.read(collect(model, corpus, "gpu.gguf", concat(nextn, "-ngl", "all")));
        ImatrixFile.Entry experts = ImatrixRunTest.entry(gpu, "blk.1.ffn_up_exps.weight");
        assertNotNull("the routed experts were collected on the device", experts);
        long pairs = 0;
        for (long c : experts.counts()) {
            pairs += c;
        }
        int used = SyntheticModels.Config.qwen3moe().nExpertUsed();
        assertEquals("one row per routed (token, slot) pair", 2L * CTX * used, pairs);
        assertEquals(1, gpu.nLayerNextn());
        ImatrixParity.assertSameStatistics("glm4moe + NextN on the GPU", cpu, gpu, TOLERANCE);
    }

    @Test
    public void hostedAndCachedRoutedExpertsCollectThroughTheCpuTap() throws Exception {
        Path model = SyntheticModels.qwen3moe().writeTempFile();
        Path corpus = corpus();
        requireGpuEngine(model, corpus, m -> { }, "--cpu-moe");
        requireGpuEngine(model, corpus, m -> assertNotNull("the experts live in a MoE expert cache",
                m.moeExpertCache()), "--moe-cache", "on");
        String[] batch = {"--batch-size", String.valueOf(CTX)};
        ImatrixFile.Read cpu = ImatrixFile.read(collect(model, corpus, "cpu.gguf", concat(batch, "-ngl", "0")));
        ImatrixFile.Read hosted = ImatrixFile.read(collect(model, corpus, "hosted.gguf",
                concat(batch, "-ngl", "all", "--cpu-moe")));
        ImatrixFile.Read cached = ImatrixFile.read(collect(model, corpus, "cached.gguf",
                concat(batch, "-ngl", "all", "--moe-cache", "on")));
        assertNotNull("the hosted experts were collected", ImatrixRunTest.entry(hosted, "blk.1.ffn_down_exps.weight"));
        ImatrixParity.assertSameStatistics("--cpu-moe", cpu, hosted, TOLERANCE);
        ImatrixParity.assertSameStatistics("--moe-cache on", cpu, cached, TOLERANCE);
    }

    @Test
    public void absorbedMlaCollectsItsPerHeadProductsOnTheGpu() throws Exception {
        assertFamilyParity(SyntheticMlaModels.deepseek2().writeTempFile(), MLA_HEADS);
    }

    @Test
    public void sparseMlaAttentionCollectsItsPerHeadProductsOnTheGpu() throws Exception {
        assertFamilyParity(SyntheticMlaModels.deepseek32().writeTempFile(), MLA_HEADS);
    }

    @Test
    public void qsaAttentionCollectsItsPerHeadProductsOnTheGpu() throws Exception {
        assertFamilyParity(SyntheticMlaModels.glm5Next().writeTempFile(), MLA_HEADS);
    }

    @Test
    public void groupedDeepseekV4OutputCollectsOneMatrixPerGroupOnTheGpu() throws Exception {
        String groups = String.valueOf(SyntheticRoutedModels.DSV4_OUTPUT_GROUPS);
        assertFamilyParity(SyntheticRoutedModels.deepseek4().writeTempFile(),
                new String[] {"blk.0.attn_output_a.weight", groups, "blk.1.attn_output_a.weight", groups});
    }

    @Test
    public void routedValueExpertsCollectOneRowPerPairOnTheGpu() throws Exception {
        Path model = SyntheticRoutedModels.k2Horizon().writeTempFile();
        ImatrixFile.Read gpu = assertFamilyParity(model, new String[0]);
        ImatrixFile.Entry values = ImatrixRunTest.entry(gpu, "blk.1.attn_v_exps.weight");
        assertNotNull("the value experts were collected on the device", values);
        assertEquals(SyntheticRoutedModels.VALUE_EXPERTS, values.nMat());
        long pairs = 0;
        for (long c : values.counts()) {
            pairs += c;
        }
        assertEquals("one row per routed (token, slot) pair", 2L * CTX * SyntheticRoutedModels.VALUE_EXPERTS_USED,
                pairs);
    }

    /**
     * The CPU and the GPU collection of {@code model} hold the same statistics; each stacked weight of
     * {@code perHead} ({@code name, matrices} pairs) was collected on the device with one matrix per head or group,
     * each counting every token. Returns the GPU collection.
     */
    private ImatrixFile.Read assertFamilyParity(Path model, String[] perHead) throws Exception {
        Path corpus = corpus();
        requireGpuEngine(model, corpus);
        String[] batch = {"--batch-size", String.valueOf(CTX)};
        ImatrixFile.Read cpu = ImatrixFile.read(collect(model, corpus, "cpu.gguf", concat(batch, "-ngl", "0")));
        ImatrixFile.Read gpu = ImatrixFile.read(collect(model, corpus, "gpu.gguf", concat(batch, "-ngl", "all")));
        for (int i = 0; i < perHead.length; i += 2) {
            ImatrixFile.Entry heads = ImatrixRunTest.entry(gpu, perHead[i]);
            assertNotNull(perHead[i] + " was collected on the device", heads);
            int nMat = Integer.parseInt(perHead[i + 1]);
            assertEquals(perHead[i] + ": one matrix per head", nMat, heads.nMat());
            for (int h = 0; h < nMat; h++) {
                assertEquals(perHead[i] + ": matrix " + h + " counts every token", 2L * CTX, heads.count(h));
            }
        }
        ImatrixParity.assertSameStatistics(model.getFileName().toString(), cpu, gpu, TOLERANCE);
        return gpu;
    }

    /** Skips the test when {@code -ngl all} opens on the CPU engine (no GPU backend can run the model). */
    private static void requireGpuEngine(Path model, Path corpus) throws Exception {
        requireGpuEngine(model, corpus, m -> { });
    }

    /**
     * Opens {@code model} with {@code -ngl all} and {@code extra} and hands the GPU model to {@code check}; skips the
     * test when it opens on the CPU engine (no GPU backend can run the model).
     */
    private static void requireGpuEngine(Path model, Path corpus, Consumer<GGUFModelGPU> check, String... extra)
            throws Exception {
        String[] base = {model.toString(), "--file", corpus.toString(), "--ctx-size", String.valueOf(CTX),
                "--batch-size", String.valueOf(CTX), "-ngl", "all"};
        ImatrixCommand.Params p = ImatrixCommand.Params.parse(concat(base, extra));
        try (ImatrixRun run = ImatrixRun.open(p)) {
            Assume.assumeTrue("the model opened on " + run.model().getClass().getSimpleName(),
                    run.model() instanceof GGUFModelGPU);
            check.accept((GGUFModelGPU) run.model());
        }
    }

    private Path corpus() throws IOException {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < 80; i++) {
            text.append("the quick cat\n");
        }
        Path f = tmp.newFile("corpus.txt").toPath();
        Files.write(f, text.toString().getBytes(StandardCharsets.UTF_8));
        return f;
    }

    private Path collect(Path model, Path corpus, String name, String... extra) throws Exception {
        Path out = tmp.getRoot().toPath().resolve(name);
        List<String> args = new ArrayList<>(Arrays.asList(model.toString(), "--file", corpus.toString(),
                "--ctx-size", String.valueOf(CTX), "--chunks", "2", "--no-ppl", "--out-file", out.toString()));
        args.addAll(Arrays.asList(extra));
        PrintStream stdout = System.out;
        System.setOut(new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8.name()));
        try {
            ImatrixCommand.collect(args.toArray(new String[0]));
        } finally {
            System.setOut(stdout);
        }
        return out;
    }

    private static String[] concat(String[] a, String... b) {
        String[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
