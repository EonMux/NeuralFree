package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.testutil.RealModels;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in timing of the speculative families on one target ({@code NEURALFREE_BENCH_TLV=1} or the {@code
 * neuralfree.bench.tlv} property, which the test task forwards), on the GPU backends: the target alone, the
 * linear chain, the single-pass tree verify ({@code neuralfree.draft.tree}) and two-level verification ({@code
 * neuralfree.draft.tlv}) in both its shapes — the quantized copy beside the target and, on a target whose
 * projection matrices stream as W12 rows ({@code neuralfree.weights.w12}), the nested 8-bit Q view over the
 * target's own backend.
 *
 * <p>Three pairs: the synthetic llama target with its dflash drafter, in an F32 build (copy shape) and a BF16
 * build (nested W12 shape); the real Qwen3-Embedding 4B Q8_0 target with the 0.6B Q8_0 draft model, which no
 * TLV shape can serve (an 8-bit target streams no fewer bytes than the copy would, and Q8_0 is no W12
 * material) and which runs plain, chain and tree only; and the real pplx-embed v1 pair (two 0.6B qwen3 models
 * in F16 from the user's model directory, never downloaded), whose F16 target serves the copy and the nested
 * W12 shape alike. Each mode runs greedy over a warmup and three timed 48-token generations and reports the
 * best tokens/s, the tokens proposed, confirmed and emitted per verify step and the acceptance rate; every
 * TLV round runs one Q verify, one playback and one exact verify, so its emitted-per-step figure is the
 * tokens per exact verify. Asserts nothing; the numbers go to the result report.
 * {@code NEURALFREE_BENCH_BACKEND} or the {@code neuralfree.bench.backend} property picks one backend per run.
 */
@RunWith(Parameterized.class)
public class TlvBenchmarkTest {

    private static final String SWITCH = "NEURALFREE_BENCH_TLV";
    private static final String QWEN_TARGET =
            RealModels.path("Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf");
    private static final String QWEN_DRAFTER =
            RealModels.path("Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf");
    private static final String PPLX_TARGET =
            RealModels.path("pplx-embed-v1-0.6b-GGUF/pplx-embed-v1-0.6b-f16.gguf");
    private static final String PPLX_DRAFTER =
            RealModels.path("pplx-embed-context-0.6b-GGUF/pplx-embed-context-v1-0.6b-f16.gguf");
    private static final int N_CTX = 1024;
    private static final int UBATCH = 64;
    private static final int DRAFT_MAX = 8;
    private static final double P_SPLIT = 0.1;
    private static final int WARMUP_TOKENS = 16;
    private static final int TOKENS = 48;
    private static final int REPS = 3;
    private static final int PARALLEL = 2;
    private static final int BLOCK = 8;
    private static final int[] TARGET_LAYERS = {0, 2};
    private static final String PROMPT = "The solar system formed about 4.6 billion years ago from a giant cloud of "
            + "gas and dust. Gravity pulled the material together, and most of it gathered in the center to form "
            + "the Sun. The rest flattened into a disk from which the planets grew.";

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{EngineConfig.BackendMode.CUDA},
                new Object[]{EngineConfig.BackendMode.OPENCL});
    }

    private final EngineConfig.BackendMode backend;

    public TlvBenchmarkTest(EngineConfig.BackendMode backend) {
        this.backend = backend;
    }

    /** All pairs over one backend: plain, chain, tree and both TLV shapes, each timed greedily. */
    @Test
    public void plainChainTreeAndTwoLevelThroughput() throws IOException, InterruptedException {
        Assume.assumeTrue("set " + SWITCH + "=1 to run the TLV benchmark", enabled());
        switch (backend) {
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            default:
                Assume.assumeTrue("no device for backend " + backend, false);
        }
        // one backend per run: each variant's models hold hundreds of MB of direct buffers that free on
        // collection only, past what one test JVM's default direct-memory budget lets two backends coexist in.
        // a system property over the environment: the environment rides the Gradle daemon that answers the
        // invocation, while a neuralfree.* property rides the command line the test task forwards
        String only = System.getProperty("neuralfree.bench.backend", System.getenv("NEURALFREE_BENCH_BACKEND"));
        Assume.assumeTrue("neuralfree.bench.backend=" + only + " skips the " + backend + " run",
                only == null || only.isBlank() || only.equalsIgnoreCase(backend.name()));

        if (Files.exists(Path.of(QWEN_TARGET)) && Files.exists(Path.of(QWEN_DRAFTER))) {
            rest();
            int[] prompt = tokenize(QWEN_TARGET);
            System.out.println("== TLV benchmark backend " + backend + " pair real Qwen3-Embedding 4B Q8_0 target,"
                    + " 0.6B Q8_0 draft model (the 8-bit target serves no TLV shape: an 8-bit target streams no"
                    + " fewer bytes than the copy would, and Q8_0 is no W12 material)");
            timed("plain", () -> timedPlain(QWEN_TARGET, prompt));
            timed("chain", () -> timedModelChain(QWEN_TARGET, QWEN_DRAFTER, prompt));
            timed("mtree-w32", () -> timedModelTree(QWEN_TARGET, QWEN_DRAFTER, prompt, 32));
        } else {
            System.out.println("== TLV benchmark backend " + backend
                    + " pair real Qwen3-Embedding: the files are absent, the pair is skipped");
        }

        if (Files.exists(Path.of(PPLX_TARGET)) && Files.exists(Path.of(PPLX_DRAFTER))) {
            rest();
            int[] prompt = tokenize(PPLX_TARGET);
            System.out.println("== TLV benchmark backend " + backend + " pair real pplx-embed v1 0.6B F16 target,"
                    + " pplx-embed context v1 0.6B F16 draft model (the F16 target serves the copy and the nested"
                    + " W12 shape alike)");
            timed("plain", () -> timedPlain(PPLX_TARGET, prompt));
            timed("chain", () -> timedModelChain(PPLX_TARGET, PPLX_DRAFTER, prompt));
            timed("mtree-w32", () -> timedModelTree(PPLX_TARGET, PPLX_DRAFTER, prompt, 32));
            timed("tlv-copy-w32", () -> timedTlv(PPLX_TARGET, PPLX_DRAFTER, prompt, 32, false));
            timed("plain-w12", () -> timedPlain(PPLX_TARGET, prompt, true));
            timed("mtree-w12-w32", () -> timedModelTree(PPLX_TARGET, PPLX_DRAFTER, prompt, 32, true));
            timed("tlv-qview-w32", () -> timedTlv(PPLX_TARGET, PPLX_DRAFTER, prompt, 32, true));
        } else {
            System.out.println("== TLV benchmark backend " + backend
                    + " pair real pplx-embed: the files are absent, the pair is skipped");
        }

        rest();
        Path f32Target = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build())
                .writeTempFile();
        Path bf16Target = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4)
                .weightType(GGMLType.BF16).build()).writeTempFile();
        Path dflash = SyntheticModels.dflashDrafter(SyntheticModels.Config.llama().toBuilder().nLayer(2).build(),
                TARGET_LAYERS, BLOCK).writeTempFile();
        for (Path p : Arrays.asList(f32Target, bf16Target, dflash)) {
            p.toFile().deleteOnExit();
        }
        int[] prompt = {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};
        System.out.println("== TLV benchmark backend " + backend + " pair synthetic (llama target, dflash drafter"
                + " over its layers 0 and 2; the F32 target serves the copy shape, the BF16 build streams its"
                + " projections as W12 and serves the nested Q-view shape)");
        timed("plain", () -> timedPlain(f32Target.toString(), prompt));
        timed("chain", () -> timedDflashChain(f32Target.toString(), dflash.toString(), prompt));
        timed("tree-w16", () -> timedDflashTree(f32Target.toString(), dflash.toString(), prompt, 16, false));
        timed("tree-w32", () -> timedDflashTree(f32Target.toString(), dflash.toString(), prompt, 32, false));
        timed("tlv-copy-w16", () -> timedTlv(f32Target.toString(), dflash.toString(), prompt, 16, false));
        timed("tlv-copy-w32", () -> timedTlv(f32Target.toString(), dflash.toString(), prompt, 32, false));
        timed("plain-w12", () -> timedPlain(bf16Target.toString(), prompt, true));
        timed("tree-w12-w16", () -> timedDflashTree(bf16Target.toString(), dflash.toString(), prompt, 16, true));
        timed("tree-w12-w32", () -> timedDflashTree(bf16Target.toString(), dflash.toString(), prompt, 32, true));
        timed("tlv-qview-w16", () -> timedTlv(bf16Target.toString(), dflash.toString(), prompt, 16, true));
        timed("tlv-qview-w32", () -> timedTlv(bf16Target.toString(), dflash.toString(), prompt, 32, true));
    }

    /** One benchmark mode's body: the timing helpers' checked exceptions. */
    private interface BenchMode {
        void run() throws IOException, InterruptedException;
    }

    /** Runs one benchmark mode; a mode that cannot run reports its failure and the rest still time. */
    private void timed(String mode, BenchMode run) {
        try {
            run.run();
        } catch (IOException | InterruptedException | RuntimeException e) {
            report(mode + " backend " + backend + ": FAILED " + e);
            try {
                rest();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /** The opt-in switch: the environment's {@code NEURALFREE_BENCH_TLV=1} or the same-named neuralfree.* property. */
    private static boolean enabled() {
        String v = System.getProperty("neuralfree.bench.tlv", System.getenv(SWITCH));
        return "1".equals(v);
    }

    /** One report line to the console and to {@code build/tlv-bench.txt}, which outlives a lost test-report write. */
    private static void report(String line) {
        System.out.println(line);
        try (java.io.PrintWriter out = new java.io.PrintWriter(new java.io.FileWriter(
                Path.of("build", "tlv-bench.txt").toFile(), true))) {
            out.println(line);
        } catch (IOException e) {
            System.out.println("the benchmark report file write failed: " + e);
        }
    }

    /** Lets the previous mode's closed model release its direct buffers before the next opens. */
    private static void rest() throws InterruptedException {
        System.gc();
        Thread.sleep(300);
    }

    /** The real pair's prompt ids through the target's tokenizer, over a configuration without a drafter. */
    private int[] tokenize(String targetPath) throws IOException {
        try (EngineModel model = LLMModelFactory.open(targetPath, plainConfig(false))) {
            return model.tokenizer().encode(PROMPT);
        }
    }

    private EngineConfig plainConfig(boolean w12) {
        EngineConfig.Builder b = EngineConfig.builder().backend(backend)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).build());
        return w12 ? b.speculativeOptions(SpeculativeOptions.builder().w12(true).build()).build() : b.build();
    }

    /** The target alone through {@code GenerationLoop} over the plain contiguous memory. */
    private void timedPlain(String targetPath, int[] prompt) throws IOException {
        timedPlain(targetPath, prompt, false);
    }

    private void timedPlain(String targetPath, int[] prompt, boolean w12) throws IOException {
        try (EngineModel model = LLMModelFactory.open(targetPath, plainConfig(w12))) {
            GenerationLoop loop = GenerationLoop.forModel(model, plainConfig(w12));
            requireBudget("plain warmup", loop.generate(prompt, greedy(WARMUP_TOKENS), (TokenCallback) null),
                    WARMUP_TOKENS);
            List<Double> rates = new ArrayList<>();
            for (int rep = 0; rep < REPS; rep++) {
                long t0 = System.nanoTime();
                GenerationResult r = loop.generate(prompt, greedy(TOKENS), (TokenCallback) null);
                long dt = System.nanoTime() - t0;
                requireBudget("plain rep " + rep, r, TOKENS);
                rates.add(TOKENS / (dt / 1e9));
            }
            print("plain" + (w12 ? "-w12" : ""), 0, rates, null);
        }
    }

    /** The real pair's linear chain ({@code p_split 0} over the contiguous memory). */
    private void timedModelChain(String targetPath, String drafterPath, int[] prompt)
            throws IOException, InterruptedException {
        EngineConfig cfg = EngineConfig.builder().backend(backend)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).build())
                .draftModelPath(drafterPath).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().pSplit(0).treeVerify(false).build()).build();
        timedSession(targetPath, cfg, "chain", 0, prompt);
        rest();
    }

    private void timedModelTree(String targetPath, String drafterPath, int[] prompt, int width)
            throws IOException, InterruptedException {
        timedModelTree(targetPath, drafterPath, prompt, width, false);
    }

    /** The real pair's branching tree at the width over the paged memory's two draft sequences. */
    private void timedModelTree(String targetPath, String drafterPath, int[] prompt, int width, boolean w12)
            throws IOException, InterruptedException {
        EngineConfig.Builder b = EngineConfig.builder().backend(backend).memory(pagedMemory())
                .draftModelPath(drafterPath).draftMax(DRAFT_MAX);
        if (w12) {
            b.speculativeOptions(SpeculativeOptions.builder().pSplit(P_SPLIT).treeVerify(false)
                    .treeNodes(width).w12(true).build());
        } else {
            b.speculativeOptions(SpeculativeOptions.builder().pSplit(P_SPLIT).treeVerify(false)
                    .treeNodes(width).build());
        }
        timedSession(targetPath, b.build(), (w12 ? "mtree-w12-w" : "mtree-w") + width, width, prompt);
        rest();
    }

    /** The dflash drafter's linear chain ({@code neuralfree.draft.tree} off) over the contiguous memory. */
    private void timedDflashChain(String targetPath, String dflashPath, int[] prompt)
            throws IOException, InterruptedException {
        EngineConfig cfg = EngineConfig.builder().backend(backend)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).build())
                .draftModelPath(dflashPath).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(false).build()).build();
        timedSession(targetPath, cfg, "chain", 0, prompt);
        rest();
    }

    private void timedDflashTree(String targetPath, String dflashPath, int[] prompt, int width, boolean w12)
            throws IOException, InterruptedException {
        EngineConfig cfg = EngineConfig.builder().backend(backend).memory(pagedMemory())
                .draftModelPath(dflashPath).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(width)
                        .w12(w12).build()).build();
        timedSession(targetPath, cfg, (w12 ? "tree-w12-w" : "tree-w") + width, width, prompt);
        rest();
    }

    /** Two-level verification over the dflash drafter at the width: the copy shape, or the nested W12 shape. */
    private void timedTlv(String targetPath, String dflashPath, int[] prompt, int width, boolean w12)
            throws IOException, InterruptedException {
        EngineConfig cfg = EngineConfig.builder().backend(backend).memory(pagedMemory())
                .draftModelPath(dflashPath).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).twoLevelVerify(true)
                        .tlvBits(8).treeNodes(width).w12(w12).build()).build();
        timedSession(targetPath, cfg, (w12 ? "tlv-qview-w" : "tlv-copy-w") + width, width, prompt);
        rest();
    }

    private MemoryConfig pagedMemory() {
        return MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).parallel(PARALLEL).kvUnified(true)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    /** Opens the pair under {@code cfg} and times the warmup and the reps of one greedy generation. */
    private void timedSession(String targetPath, EngineConfig cfg, String mode, int width, int[] prompt)
            throws IOException {
        try (EngineModel model = LLMModelFactory.open(targetPath, cfg);
             SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
            GenerationResult w = session.generate(prompt, greedy(WARMUP_TOKENS), (TokenCallback) null);
            requireBudget(mode + " warmup", w, WARMUP_TOKENS);
            List<Double> rates = new ArrayList<>();
            SpeculativeDecoder.Stats stats = null;
            for (int rep = 0; rep < REPS; rep++) {
                long t0 = System.nanoTime();
                GenerationResult r = session.generate(prompt, greedy(TOKENS), (TokenCallback) null);
                long dt = System.nanoTime() - t0;
                requireBudget(mode + " rep " + rep, r, TOKENS);
                rates.add(TOKENS / (dt / 1e9));
                stats = session.lastStats();
            }
            print(mode, width, rates, stats);
        }
    }

    private static void requireBudget(String what, GenerationResult r, int expected) {
        if (r.stopReason() != StopReason.MAX_TOKENS || r.tokens().length != expected) {
            throw new AssertionError(what + " stopped at " + r.stopReason() + " after " + r.tokens().length
                    + " tokens; the benchmark needs the full budget of " + expected);
        }
    }

    /**
     * The mode's line: best tokens/s over the reps and, when speculated, the per-step proposal, confirmation and
     * emission — for the TLV lines the emitted figure is the tokens per exact verify, one per round.
     */
    private void print(String mode, int width, List<Double> rates, SpeculativeDecoder.Stats stats) {
        double best = rates.stream().mapToDouble(Double::doubleValue).max().orElse(0);
        if (stats == null) {
            report(String.format(Locale.ROOT,
                    "%s backend %s: best %.1f tok/s over %d tokens (target alone)", mode, backend, best, TOKENS));
            return;
        }
        double perRound = stats.rounds() == 0 ? 0 : (double) stats.accepted() / stats.rounds();
        double proposed = stats.rounds() == 0 ? 0 : (double) stats.drafted() / stats.rounds();
        report(String.format(Locale.ROOT,
                "%s backend %s width %d: best %.1f tok/s over %d tokens; per verify step proposed %.2f, confirmed "
                        + "%.2f, emitted %.2f; acceptance %.1f%% of drafted",
                mode, backend, width, best, TOKENS, proposed, perRound, perRound + 1,
                stats.acceptanceRate() * 100));
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding())
                .build();
    }
}
