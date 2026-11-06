package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.DraftHeadOptions;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashDrafter;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticDflash;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The derived draft head ({@code neuralfree.draft.head.*}, {@code neuralfree.draft.prec}): a DFlash/DSpark drafter
 * reading a reduced vocabulary through the derived head — its own restated, the target's borrowed, or at a lower
 * precision — commits exactly the tokens of the target's plain greedy loop, on the CPU and on the OpenCL device,
 * and the drafts of the device and host drafters still agree.
 */
public class DflashDerivedHeadTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final int[] CONTINUATION = {77, 3, 150, 9};
    private static final int GREEDY_TOKENS = 12;
    private static final int DRAFT = 4;
    private static final int KEPT = 64;
    private static final int WARM_ROUNDS = 50;
    private static final int MEASURED_ROUNDS = 500;

    private final List<Path> files = new ArrayList<>();
    private Path target;
    private Path draft;
    private Path borrowed;
    private Path dflash2;
    private Path dspark;

    @Before
    public void writeModels() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        target = written(SyntheticModels.llama(c.toBuilder().nLayer(4).build()));
        draft = written(SyntheticDflash.dflash(c, SyntheticDflash.Options.of(DRAFT + 1, 1, 2)));
        borrowed = written(SyntheticDflash.dflash(c,
                SyntheticDflash.Options.of(DRAFT + 1, 1, 2).shareTargetHead()));
        dflash2 = written(SyntheticDflash.dflash(c.toBuilder().nLayer(2).build(),
                SyntheticDflash.Options.of(DRAFT + 1, 1, 2).dflash2(2, 4, 8, 4)));
        dspark = written(SyntheticDflash.dflash(c, SyntheticDflash.Options.of(DRAFT + 1, 1, 2).dspark(8)));
    }

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    private static EngineConfig.Builder cpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(32).build());
    }

    private static EngineConfig.Builder gpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(32).build());
    }

    /** The session config of a drafter reading {@code head}, drafting on the device when {@code onDevice}. */
    private static EngineConfig cfg(Path draft, DraftHeadOptions head, boolean onDevice) {
        EngineConfig.Builder b = (onDevice ? gpu() : cpu()).draftModelPath(draft.toString()).draftMax(DRAFT);
        return b.speculativeOptions(SpeculativeOptions.builder().draftHead(head).dflashOnDevice(onDevice).build())
                .build();
    }

    private static DraftHeadOptions reduced() {
        return DraftHeadOptions.builder().vocabSize(KEPT).build();
    }

    /** The plain greedy tokens of the CPU target, the loop the sessions must reproduce. */
    private int[] cpuReference() throws IOException {
        EngineConfig plain = cpu().build();
        try (GGUFModel m = new GGUFModel(target.toString(), plain)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(plain.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            return m.generate(PROMPT, request, null).tokens();
        }
    }

    /** The plain greedy tokens of the OpenCL target. */
    private int[] gpuReference() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        EngineConfig plain = gpu().build();
        try (GGUFModelGPU m = new GGUFModelGPU(target.toString(), null, plain)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(plain.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            int[] tokens = m.generate(PROMPT, request, null).tokens();
            m.forwardBackend().reset();
            return tokens;
        }
    }

    /** A speculative session's greedy tokens and drafting stats. */
    private static int[] sessionGreedy(Path target, EngineConfig cfg, SpeculativeDecoder.Stats[] stats)
            throws IOException {
        try (SpeculativeSession s = LLMModelFactory.openSpeculative(target.toString(), null, cfg)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            int[] tokens = s.generate(PROMPT, request, null).tokens();
            stats[0] = s.lastStats();
            return tokens;
        }
    }

    /** The CPU session commits the reference tokens and drafts. */
    private void assertCpuParity(Path draft, DraftHeadOptions head, int[] reference) throws IOException {
        SpeculativeDecoder.Stats[] stats = new SpeculativeDecoder.Stats[1];
        int[] tokens = sessionGreedy(target, cfg(draft, head, false), stats);
        assertArrayEquals("speculative greedy with " + head, reference, tokens);
        assertTrue(String.valueOf(stats[0]), stats[0].drafted() > 0);
    }

    /** The OpenCL session commits the reference tokens, drafts, and its drafter runs on the device. */
    private void assertGpuParity(Path draft, DraftHeadOptions head, int[] reference, boolean wantsDevice)
            throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        EngineConfig config = cfg(draft, head, true);
        try (SpeculativeSession s = LLMModelFactory.openSpeculative(target.toString(), null, config)) {
            if (wantsDevice) {
                assertTrue("the drafter runs on the device", ((DflashDrafter) s.drafter()).onDevice());
            }
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(config.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            int[] tokens = s.generate(PROMPT, request, null).tokens();
            SpeculativeDecoder.Stats stats = s.lastStats();
            assertArrayEquals("speculative greedy on the device with " + head, reference, tokens);
            assertTrue(String.valueOf(stats), stats.drafted() > 0);
        }
    }

    // ------------------------------------------------------------------ CPU

    @Test
    public void aReducedVocabDraftKeepsThePlainGreedyTokens() throws IOException {
        assertCpuParity(draft, reduced(), cpuReference());
    }

    @Test
    public void aBorrowedTargetHeadKeepsThePlainGreedyTokens() throws IOException {
        assertCpuParity(borrowed, reduced(), cpuReference());
    }

    @Test
    public void anFp8DerivedHeadKeepsThePlainGreedyTokens() throws IOException {
        DraftHeadOptions head = DraftHeadOptions.builder().vocabSize(KEPT)
                .headPrec(DraftHeadOptions.Prec.FP8).build();
        assertCpuParity(draft, head, cpuReference());
    }

    @Test
    public void anNvfp4DerivedHeadKeepsThePlainGreedyTokens() throws IOException {
        DraftHeadOptions head = DraftHeadOptions.builder().vocabSize(KEPT)
                .headPrec(DraftHeadOptions.Prec.NVFP4).build();
        assertCpuParity(draft, head, cpuReference());
    }

    /** The borrowed target head restated at nvfp4: its global scale rides as the derived sibling. */
    @Test
    public void aBorrowedNvfp4HeadKeepsThePlainGreedyTokens() throws IOException {
        DraftHeadOptions head = DraftHeadOptions.builder().vocabSize(KEPT)
                .headPrec(DraftHeadOptions.Prec.NVFP4).build();
        assertCpuParity(borrowed, head, cpuReference());
    }

    @Test
    public void anNvfp4DerivedHeadAndMatricesKeepThePlainGreedyTokens() throws IOException {
        DraftHeadOptions head = DraftHeadOptions.builder().vocabSize(KEPT)
                .headPrec(DraftHeadOptions.Prec.NVFP4).matricesPrec(DraftHeadOptions.Prec.NVFP4).build();
        assertCpuParity(draft, head, cpuReference());
    }

    @Test
    public void aQ8ZeroDerivedHeadAndMatricesKeepThePlainGreedyTokens() throws IOException {
        DraftHeadOptions head = DraftHeadOptions.builder().vocabSize(KEPT)
                .headPrec(DraftHeadOptions.Prec.Q8_0).matricesPrec(DraftHeadOptions.Prec.Q8_0).build();
        assertCpuParity(draft, head, cpuReference());
    }

    @Test
    public void aLowerPrecisionDrafterKeepsThePlainGreedyTokens() throws IOException {
        DraftHeadOptions head = DraftHeadOptions.builder().matricesPrec(DraftHeadOptions.Prec.FP8).build();
        assertCpuParity(draft, head, cpuReference());
    }

    @Test
    public void aRankedFrequencyFileKeepsThePlainGreedyTokens() throws IOException {
        Path ranking = Files.createTempFile("ranked", ".json");
        files.add(ranking);
        Files.writeString(ranking, "{\"ranked_ids\": [5, 99, 17, 42, 1, 26]}");
        DraftHeadOptions head = DraftHeadOptions.builder().vocabSize(KEPT)
                .frequencyFile(ranking.toString()).build();
        assertCpuParity(draft, head, cpuReference());
    }

    /**
     * The strongest lossless check: with the ranking keeping every token the full-head drafter proposes along the
     * same forward sequence, the reduced drafter proposes the same tokens — the kept rows are exact and its argmax
     * over the kept rows is the full drafter's argmax — and the session accepts exactly what the full session does.
     */
    @Test
    public void aRankedReducedDrafterProposesTheFullDrafterTokens() throws IOException {
        List<int[]> full = drafts(cfg(draft, null, false));
        Set<Integer> proposed = new TreeSet<>();
        for (int[] d : full) {
            for (int token : d) {
                proposed.add(token);
            }
        }
        assertFalse("the full drafter proposes tokens", proposed.isEmpty());
        Path ranking = Files.createTempFile("ranked", ".json");
        files.add(ranking);
        StringBuilder sb = new StringBuilder("{\"ranked_ids\": [");
        for (int id : proposed) {
            sb.append(sb.charAt(sb.length() - 1) == '[' ? "" : ",").append(id);
        }
        Files.writeString(ranking, sb.append("]}").toString());
        DraftHeadOptions ranked = DraftHeadOptions.builder().vocabSize(KEPT)
                .frequencyFile(ranking.toString()).build();

        List<int[]> reduced = drafts(cfg(draft, ranked, false));
        assertEquals("the same rounds draft", full.size(), reduced.size());
        for (int i = 0; i < full.size(); i++) {
            assertArrayEquals("draft round " + i, full.get(i), reduced.get(i));
        }
        SpeculativeDecoder.Stats[] fullStats = new SpeculativeDecoder.Stats[1];
        int[] fullTokens = sessionGreedy(target, cfg(draft, null, false), fullStats);
        SpeculativeDecoder.Stats[] rankedStats = new SpeculativeDecoder.Stats[1];
        int[] rankedTokens = sessionGreedy(target, cfg(draft, ranked, false), rankedStats);
        assertArrayEquals("the same greedy tokens", fullTokens, rankedTokens);
        assertEquals("the same drafts are drafted", fullStats[0].drafted(), rankedStats[0].drafted());
        assertEquals("the same drafts are accepted", fullStats[0].accepted(), rankedStats[0].accepted());
        assertTrue("the full drafter's proposals are accepted: " + fullStats[0], fullStats[0].accepted() > 0);
    }

    /**
     * The drafts of a host drafter over {@code cfg}: after the prompt and after each continuation token, the
     * target's forward calls feeding the drafter as a session does.
     */
    private List<int[]> drafts(EngineConfig cfg) throws IOException {
        List<int[]> out = new ArrayList<>();
        try (GGUFModel t = new GGUFModel(target.toString(), cfg);
             SpeculativeSession s = LLMModelFactory.speculate(t, cfg)) {
            DraftModel d = s.drafter();
            ForwardBackend f = t.forwardBackend();
            f.reset();
            d.reset();
            f.prefill(PROMPT, 0);
            out.add(d.draft(PROMPT[PROMPT.length - 1], DRAFT));
            int pos = PROMPT.length;
            for (int token : CONTINUATION) {
                f.decode(token, pos++);
                out.add(d.draft(token, DRAFT));
            }
            f.reset();
            d.reset();
        }
        return out;
    }

    @Test
    public void aReducedDflash2DraftKeepsThePlainGreedyTokens() throws IOException {
        assertCpuParity(dflash2, reduced(), cpuReference());
    }

    @Test
    public void aReducedDsparkDraftKeepsThePlainGreedyTokens() throws IOException {
        assertCpuParity(dspark, reduced(), cpuReference());
    }

    /**
     * The device drafter's time per draft round with the full head, the reduced head verbatim and the reduced head
     * at fp8 and nvfp4, printed for the RTX 5080 numbers in RESULT.md: warm rounds first, then the mean of timed
     * {@code draft} calls over one prefilled target.
     */
    @Test
    public void theDeviceDraftRoundTimeIsMeasured() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        DraftHeadOptions fp8 = DraftHeadOptions.builder().vocabSize(KEPT)
                .headPrec(DraftHeadOptions.Prec.FP8).build();
        DraftHeadOptions nvfp4 = DraftHeadOptions.builder().vocabSize(KEPT)
                .headPrec(DraftHeadOptions.Prec.NVFP4).build();
        StringBuilder report = new StringBuilder("draft round time (us/round, draft=");
        report.append(DRAFT).append("):");
        for (String label : new String[] {"full-head", "reduced", "reduced-fp8", "reduced-nvfp4"}) {
            DraftHeadOptions head = label.equals("full-head") ? null : label.equals("reduced") ? reduced()
                    : label.equals("reduced-fp8") ? fp8 : nvfp4;
            EngineConfig config = cfg(draft, head, true);
            try (GGUFModelGPU t = new GGUFModelGPU(target.toString(), null, config);
                 SpeculativeSession s = LLMModelFactory.speculate(t, config)) {
                DraftModel d = s.drafter();
                ForwardBackend f = t.forwardBackend();
                f.reset();
                d.reset();
                f.prefill(PROMPT, 0);
                for (int i = 0; i < WARM_ROUNDS; i++) {
                    d.draft(PROMPT[PROMPT.length - 1], DRAFT);
                }
                long elapsed = 0;
                int rounds = 0;
                for (int i = 0; i < MEASURED_ROUNDS; i++) {
                    long start = System.nanoTime();
                    d.draft(PROMPT[PROMPT.length - 1], DRAFT);
                    elapsed += System.nanoTime() - start;
                    rounds++;
                }
                report.append(' ').append(label).append('=')
                        .append(String.format("%.1f", elapsed / 1000.0 / rounds));
                f.reset();
                d.reset();
            }
        }
        System.out.println(report);
    }

    // ------------------------------------------------------------------ OpenCL

    @Test
    public void aReducedHeadDraftsOnTheDeviceAndKeepsThePlainGreedyTokens() throws IOException {
        assertGpuParity(draft, reduced(), gpuReference(), true);
    }

    @Test
    public void aBorrowedTargetHeadDraftsOnTheDevice() throws IOException {
        assertGpuParity(borrowed, reduced(), gpuReference(), true);
    }

    @Test
    public void anFp8DerivedHeadOnTheDeviceKeepsThePlainGreedyTokens() throws IOException {
        DraftHeadOptions head = DraftHeadOptions.builder().vocabSize(KEPT)
                .headPrec(DraftHeadOptions.Prec.FP8).build();
        assertGpuParity(draft, head, gpuReference(), true);
    }

    @Test
    public void anNvfp4DerivedHeadOnTheDeviceKeepsThePlainGreedyTokens() throws IOException {
        DraftHeadOptions head = DraftHeadOptions.builder().vocabSize(KEPT)
                .headPrec(DraftHeadOptions.Prec.NVFP4).build();
        assertGpuParity(draft, head, gpuReference(), true);
    }

    @Test
    public void theDeviceAndHostDraftersAgreeWithAReducedHead() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        SpeculativeDecoder.Stats[] deviceStats = new SpeculativeDecoder.Stats[1];
        int[] fromDevice = sessionGreedy(target, cfg(draft, reduced(), true), deviceStats);
        assertTrue(deviceStats[0].drafted() > 0);
        SpeculativeDecoder.Stats[] hostStats = new SpeculativeDecoder.Stats[1];
        int[] fromHost = sessionGreedy(target, cfg(draft, reduced(), false), hostStats);
        assertTrue(hostStats[0].drafted() > 0);
        assertArrayEquals("the device and host drafters produce the same greedy tokens", fromHost, fromDevice);
    }
}
