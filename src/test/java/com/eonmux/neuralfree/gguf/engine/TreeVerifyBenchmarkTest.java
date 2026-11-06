package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.testutil.RealModels;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.TokenCallback;
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
import java.util.SplittableRandom;

/**
 * Opt-in timing of the speculative rounds ({@code NEURALFREE_BENCH_TREE=1} or the {@code neuralfree.bench.tree}
 * property, which the test task forwards), on the GPU backends with a device, over two target/drafter pairs.
 *
 * <p>The real pair comes from the user's model directory (never downloaded): Qwen3-Embedding-4B Q8_0 as the target
 * with Qwen3-Embedding-0.6B Q8_0 as the drafter, which share the Qwen3 tokenizer and architecture. Its drafter is a
 * separate generative draft model ({@link ModelDrafter}, draft route {@code draft-simple}), whose linear proposal
 * verifies as one chain and whose branching proposal verifies the single-pass masked forward ({@code TreeVerify}
 * over the paged memory) — the {@code mtree-w*} lines. The synthetic pair (the test llama target with a dflash
 * drafter over its first layers) runs the same single-pass masked verify, the {@code tree-w*} lines; it also runs
 * the perfect-drafter modes over a request whose logit bias pins the target's greedy pick to the drafters' forced
 * token, so the {@code forced-*} lines show the verify cost of a fully confirmed round. Each mode runs greedy over
 * a warmup and three timed 48-token generations and reports the best tokens/s, the mean proposed and confirmed
 * tokens per verify step, the emitted tokens per round and the acceptance rate. Asserts nothing; the numbers go to
 * the tree verify documentation. {@code NEURALFREE_BENCH_BACKEND} or the {@code neuralfree.bench.backend} property
 * picks one backend per run: each variant's models hold hundreds of MB of direct buffers that free on collection
 * only, past what one test JVM's default direct-memory budget lets two backends coexist in.
 */
@RunWith(Parameterized.class)
public class TreeVerifyBenchmarkTest {

    private static final String SWITCH = "NEURALFREE_BENCH_TREE";
    private static final String TARGET =
            RealModels.path("Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf");
    private static final String DRAFTER =
            RealModels.path("Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf");
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
                new Object[]{EngineConfig.BackendMode.OPENCL},
                new Object[]{EngineConfig.BackendMode.VULKAN});
    }

    private final EngineConfig.BackendMode backend;

    public TreeVerifyBenchmarkTest(EngineConfig.BackendMode backend) {
        this.backend = backend;
    }

    /** Both pairs over one backend: the target alone, the chain and the trees, each timed greedily. */
    @Test
    public void chainVersusTreeDecodeThroughput() throws IOException, InterruptedException {
        Assume.assumeTrue("set " + SWITCH + "=1 to run the tree verify benchmark", enabled());
        switch (backend) {
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            case VULKAN:
                GpuTestUtil.assumeVulkanAvailable();
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
        if (Files.exists(Path.of(TARGET)) && Files.exists(Path.of(DRAFTER))) {
            rest();
            int[] prompt = tokenize(TARGET);
            System.out.println("== tree verify benchmark backend " + backend + " pair real"
                    + " (Qwen3-Embedding 4B target, 0.6B draft model; the draft routes draft-simple,"
                    + " so its tree verify is the per-depth one)");
            timed("plain", () -> timedPlain(TARGET, prompt));
            timed("chain", () -> timedChain(TARGET, DRAFTER, prompt));
            timed("mtree-w16", () -> timedModelTree(TARGET, DRAFTER, prompt, 16));
            timed("mtree-w32", () -> timedModelTree(TARGET, DRAFTER, prompt, 32));
        } else {
            System.out.println("== tree verify benchmark backend " + backend
                    + " pair real: the Qwen3-Embedding files are absent, only the synthetic pair runs");
        }
        rest();
        Path target = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build())
                .writeTempFile();
        Path dflash = SyntheticModels.dflashDrafter(SyntheticModels.Config.llama().toBuilder().nLayer(2).build(),
                TARGET_LAYERS, BLOCK).writeTempFile();
        target.toFile().deleteOnExit();
        dflash.toFile().deleteOnExit();
        int[] prompt = {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};
        System.out.println("== tree verify benchmark backend " + backend
                + " pair synthetic (llama target, dflash drafter over its layers 0 and 2;"
                + " tree lines run the single-pass masked verify)");
        timed("plain", () -> timedPlain(target.toString(), prompt));
        timed("chain", () -> timedDflashChain(target.toString(), dflash.toString(), prompt));
        timed("tree-w16", () -> timedDflashTree(target.toString(), dflash.toString(), prompt, 16));
        timed("tree-w32", () -> timedDflashTree(target.toString(), dflash.toString(), prompt, 32));
        int nVocab = nVocab(target.toString());
        timed("forced-w16", () -> timedForced(target.toString(), prompt, nVocab, 16));
        timed("forced-w32", () -> timedForced(target.toString(), prompt, nVocab, 32));
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
            java.io.StringWriter sw = new java.io.StringWriter();
            e.printStackTrace(new java.io.PrintWriter(sw));
            try (java.io.PrintWriter out = new java.io.PrintWriter(new java.io.FileWriter(
                    Path.of("build", "tree-bench-stack.txt").toFile(), true))) {
                out.println("== " + mode + " backend " + backend);
                out.println(sw.toString());
            } catch (IOException io) {
                // the stack file is a convenience: the report line above carries the failure either way
            }
            try {
                rest();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /** The synthetic target's vocabulary size, over a plain configuration. */
    private int nVocab(String targetPath) throws IOException {
        try (EngineModel model = LLMModelFactory.open(targetPath, plainConfig())) {
            return model.spec().hparams().nVocab();
        }
    }

    /**
     * The perfect-drafter regime: the request's logit bias pins the target's sampling support to the drafter's
     * forced token (5, everything else at {@code -100}), so the chain confirms every drafted token and the tree
     * its whole forced path, and each line shows the verify cost a fully confirmed round carries. The chain runs
     * over the contiguous memory (the classic chain round needs it), the tree over the paged memory at the width.
     */
    private void timedForced(String targetPath, int[] prompt, int nVocab, int width)
            throws IOException, InterruptedException {
        EngineConfig chainCfg = EngineConfig.builder().backend(backend).draftMax(DRAFT_MAX)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).build()).build();
        try (EngineModel model = LLMModelFactory.open(targetPath, chainCfg)) {
            PerfectChainDrafter chain = new PerfectChainDrafter(DRAFT_MAX);
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, chain, chainCfg);
            List<Double> rates = timedDecoder(decoder, prompt, forced(nVocab, WARMUP_TOKENS),
                    forced(nVocab, TOKENS));
            print("forced-chain", 0, rates,
                    new SpeculativeDecoder.Stats(chain.rounds, chain.drafted, chain.accepted));
        }
        rest();
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).parallel(PARALLEL)
                .kvUnified(true).build();
        EngineConfig treeCfg = EngineConfig.builder().backend(backend).draftMax(DRAFT_MAX).memory(memory)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(width).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(targetPath, treeCfg)) {
            PerfectTreeDrafter tree = new PerfectTreeDrafter();
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, tree, treeCfg);
            List<Double> rates = timedDecoder(decoder, prompt, forced(nVocab, WARMUP_TOKENS),
                    forced(nVocab, TOKENS));
            print("forced-tree", width, rates,
                    new SpeculativeDecoder.Stats(tree.rounds, tree.drafted, tree.accepted));
        }
        rest();
    }

    /** The warmup and the timed reps of one generation over the prepared decoder. */
    private List<Double> timedDecoder(SpeculativeDecoder decoder, int[] prompt, GenerationRequest warmup,
                                      GenerationRequest request) {
        requireBudget("warmup", decoder.generate(prompt, warmup, (TokenCallback) null), WARMUP_TOKENS);
        List<Double> rates = new ArrayList<>();
        for (int rep = 0; rep < REPS; rep++) {
            long t0 = System.nanoTime();
            GenerationResult r = decoder.generate(prompt, request, (TokenCallback) null);
            long dt = System.nanoTime() - t0;
            requireBudget("rep " + rep, r, TOKENS);
            rates.add(TOKENS / (dt / 1e9));
        }
        return rates;
    }

    /** The greedy request whose bias pins the target's sampling support to the forced token 5. */
    private static GenerationRequest forced(int nVocab, int maxNewTokens) {
        GenerationRequest.Builder b = GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding());
        for (int t = 0; t < nVocab; t++) {
            b.logitBias(t, t == 5 ? 5.0f : -100.0f);
        }
        return b.build();
    }

    /** A drafter that always proposes {@code n} forced tokens: the biased target confirms every one. */
    private static final class PerfectChainDrafter implements DraftModel {
        private final int n;
        private int rounds;
        private int drafted;
        private int accepted;

        PerfectChainDrafter(int n) {
            this.n = n;
        }

        @Override
        public void bindTarget(EngineModel target) {
        }

        @Override
        public int[] draft(int lastToken, int nMax) {
            rounds++;
            int count = Math.min(nMax, n);
            drafted += count;
            int[] out = new int[count];
            Arrays.fill(out, 5);
            return out;
        }

        @Override
        public void accept(int nAccepted) {
            accepted += nAccepted;
        }
    }

    /**
     * A tree drafter whose every proposal is one forced path of up to four fives plus one sibling root: the
     * biased target confirms the path, so the single-pass masked verify runs each round at full acceptance.
     */
    private static final class PerfectTreeDrafter implements TreeDraftModel {
        private int rounds;
        private int drafted;
        private int accepted;

        @Override
        public void bindTarget(EngineModel target) {
        }

        @Override
        public int[] draft(int lastToken, int n) {
            return new int[0];
        }

        @Override
        public void accept(int nAccepted) {
        }

        @Override
        public int branches() {
            return 2;
        }

        @Override
        public boolean singlePassVerify() {
            return true;
        }

        @Override
        public void acceptTree(int nAccepted) {
            accepted += nAccepted;
        }

        @Override
        public DraftTree draftTree(int lastToken, int maxTokens, int branchCount, float temperature,
                                   SplittableRandom rng) {
            return draftTree(lastToken, maxTokens, branchCount, Integer.MAX_VALUE, temperature, rng);
        }

        @Override
        public DraftTree draftTree(int lastToken, int maxTokens, int branchCount, int maxRows, float temperature,
                                   SplittableRandom rng) {
            rounds++;
            List<DraftTree.Node> nodes = new ArrayList<>();
            int depth = Math.min(Math.min(maxTokens, maxRows) - 1, 4);
            for (int d = 1; d <= depth; d++) {
                nodes.add(new DraftTree.Node(5, d - 2, d, 0));
            }
            if (maxTokens > depth && maxRows > depth + 1) {
                nodes.add(new DraftTree.Node(6, -1, 1, 1));
            }
            drafted += nodes.size();
            return new DraftTree(nodes);
        }
    }

    /** The opt-in switch: the environment's {@code NEURALFREE_BENCH_TREE=1} or the same-named neuralfree.* property. */
    private static boolean enabled() {
        String v = System.getProperty("neuralfree.bench.tree", System.getenv(SWITCH));
        return "1".equals(v);
    }

    /** One report line to the console and to {@code build/tree-bench.txt}, which outlives a lost test-report write. */
    private static void report(String line) {
        System.out.println(line);
        try (java.io.PrintWriter out = new java.io.PrintWriter(new java.io.FileWriter(
                Path.of("build", "tree-bench.txt").toFile(), true))) {
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
        try (EngineModel model = LLMModelFactory.open(targetPath, plainConfig())) {
            return model.tokenizer().encode(PROMPT);
        }
    }

    private EngineConfig plainConfig() {
        return EngineConfig.builder().backend(backend)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).build()).build();
    }

    /** The target alone through {@code GenerationLoop} over the plain contiguous memory. */
    private void timedPlain(String targetPath, int[] prompt) throws IOException {
        try (EngineModel model = LLMModelFactory.open(targetPath, plainConfig())) {
            GenerationLoop loop = GenerationLoop.forModel(model, plainConfig());
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
            print("plain", 0, rates, null);
        }
    }

    /**
     * The real pair's linear chain ({@code p_split 0} over the contiguous memory, so the draft model proposes one
     * chain and the round verifies it the classic way), and its branching tree at the two widths (the paged
     * memory's two draft sequences, verified by the one single-pass masked forward per round).
     */
    private void timedChain(String targetPath, String drafterPath, int[] prompt)
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
        EngineConfig cfg = EngineConfig.builder().backend(backend).memory(pagedMemory())
                .draftModelPath(drafterPath).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().pSplit(P_SPLIT).treeVerify(false)
                        .treeNodes(width).build()).build();
        timedSession(targetPath, cfg, "mtree-w" + width, width, prompt);
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

    private void timedDflashTree(String targetPath, String dflashPath, int[] prompt, int width)
            throws IOException, InterruptedException {
        EngineConfig cfg = EngineConfig.builder().backend(backend).memory(pagedMemory())
                .draftModelPath(dflashPath).draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(width).build()).build();
        timedSession(targetPath, cfg, "tree-w" + width, width, prompt);
        rest();
    }

    private MemoryConfig pagedMemory() {
        return MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).parallel(PARALLEL).kvUnified(true).build();
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

    /** The mode's line: best tokens/s over the reps and, when speculated, the per-step proposal and acceptance. */
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
