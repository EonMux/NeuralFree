package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in timing of the batch engine's decode throughput by concurrency and speculation mode ({@code
 * NEURALFREE_BENCH_BATCH_TREE=1} or the {@code neuralfree.bench.batch.tree} property, which the test task
 * forwards), on CPU and on the OpenCL GPU. Every mode runs the same synthetic target with the same oracle drafter
 * knowledge — the target's own greedy continuation, pinned once against the fixture — as plain decodes, chain
 * verify and tree verify, at 4, 8 and 16 concurrent sequences; the oracle removes drafter quality from the
 * comparison, so the numbers show the scheduling and masking costs of the batch mechanisms (SpecPilot's
 * h100_batch_qwen38-27b-fp8_dflash2 measurement shape, arXiv 2611.02501). Each config runs a warmup and three
 * timed runs and reports the best tokens per second. Asserts the runs finish; the numbers go to the task report.
 */
@RunWith(Parameterized.class)
public class BatchTreeBudgetBenchmarkTest {

    private static final String SWITCH = "NEURALFREE_BENCH_BATCH_TREE";
    private static final int N_CTX = 256;
    private static final int UBATCH = 64;
    private static final int PROMPT_TOKENS = 24;
    private static final int NEW_TOKENS = 32;
    private static final int DRAFT_MAX = 4;
    private static final int WARMUP_RUNS = 1;
    private static final int TIMED_RUNS = 3;
    private static final int[] CONCURRENCY = {4, 8, 16};

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{EngineConfig.BackendMode.CPU},
                new Object[]{EngineConfig.BackendMode.OPENCL});
    }

    private final EngineConfig.BackendMode backend;

    public BatchTreeBudgetBenchmarkTest(EngineConfig.BackendMode backend) {
        this.backend = backend;
    }

    private static boolean enabled() {
        String v = System.getProperty("neuralfree.bench.batch.tree", System.getenv(SWITCH));
        return "1".equals(v) || "true".equalsIgnoreCase(v);
    }

    /** A prompt whose token pattern repeats, and the greedy request every sequence carries. */
    private static int[] prompt() {
        int[] t = new int[PROMPT_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (i % 5) * 7;
        }
        return t;
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(NEW_TOKENS)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static EngineConfig config(int parallel, EngineConfig.BackendMode mode) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(parallel).build();
        return EngineConfig.builder().memory(mc).backend(mode).build();
    }

    /** The reference stream one oracle needs: the prompt's greedy continuation through the single-sequence loop. */
    private static int[] continuation(Path model, int[] prompt) throws IOException {
        EngineConfig cfg = EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .backend(EngineConfig.BackendMode.CPU).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            GenerationResult r = com.eonmux.neuralfree.gguf.engine.GenerationLoop.forModel(one, cfg)
                    .generate(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            return r.tokens();
        }
    }

    /** One timed config: {@code runs} drains of {@code bs} concurrent sequences, the best wall-clock rate. */
    private static double timed(Path model, EngineConfig.BackendMode mode, int bs, BatchDrafter drafter,
                                BatchSpec spec, boolean warmup) throws IOException {
        EngineConfig cfg = config(bs, mode);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = drafter == null
                     ? BatchEngine.forModel(m, cfg)
                     : BatchEngine.forModel(m, cfg, drafter, spec)) {
            int[] prompt = prompt();
            GenerationRequest request = request();
            int runs = warmup ? WARMUP_RUNS : TIMED_RUNS;
            double best = 0.0;
            for (int run = 0; run < runs; run++) {
                List<BatchSequence> seqs = new ArrayList<>();
                long t0 = System.nanoTime();
                for (int i = 0; i < bs; i++) {
                    seqs.add(engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null));
                }
                engine.drain();
                long nanos = System.nanoTime() - t0;
                for (BatchSequence s : seqs) {
                    if (s.state() != BatchSequence.State.DONE) {
                        throw new IllegalStateException("a sequence finished " + s.state()
                                + (s.error() == null ? "" : ": " + s.error()));
                    }
                }
                double rate = (double) bs * NEW_TOKENS / (nanos / 1e9);
                best = Math.max(best, rate);
            }
            return best;
        }
    }

    /** Plain decodes, chain verify and tree verify at every concurrency, one line per config. */
    @Test
    public void batchThroughputByConcurrencyAndMode() throws IOException {
        Assume.assumeTrue("set " + SWITCH + "=1 to run the batch tree budget benchmark", enabled());
        if (backend == EngineConfig.BackendMode.OPENCL) {
            GpuTestUtil.assumeOpenCLAvailable();
        }
        Path model = SyntheticModels.llama().writeTempFile();
        int[] stream = continuation(model, prompt());

        for (int bs : CONCURRENCY) {
            double plain = timed(model, backend, bs, null, null, true);
            SpeculativeOptions chainOptions = SpeculativeOptions.builder().build();
            double chain = timed(model, backend, bs, new OracleChainDrafter(stream),
                    BatchSpec.of(DRAFT_MAX, 0, chainOptions), true);
            SpeculativeOptions treeOptions = SpeculativeOptions.builder().treeVerify(true).build();
            double tree = timed(model, backend, bs, new OracleTreeDrafter(stream),
                    BatchSpec.of(DRAFT_MAX, 0, treeOptions), true);
            System.out.printf(Locale.ROOT, "bench %s bs=%d plain=%.1f tok/s chain=%.1f tok/s tree=%.1f tok/s%n",
                    backend.name().toLowerCase(Locale.ROOT), bs, plain, chain, tree);
        }
    }

    /** The oracle every mode shares: the sequence's stream, re-anchored on the token the target just sampled. */
    private static int streamFrom(int[] stream, int[] cursor, int seq, int lastToken) {
        while (cursor[seq] < stream.length && stream[cursor[seq]] != lastToken) {
            cursor[seq]++;
        }
        return cursor[seq];
    }

    /** Drafts the reference's greedy continuation as a chain (the batch engine's chain verify mode). */
    private static final class OracleChainDrafter implements BatchDrafter {

        private final int[] stream;
        private final int[] cursor = new int[32];

        OracleChainDrafter(int[] stream) {
            this.stream = stream;
        }

        @Override
        public int[] draft(int seq, int lastToken, int n) {
            int at = streamFrom(stream, cursor, seq, lastToken);
            int left = Math.max(0, stream.length - at - 1);
            int[] out = new int[Math.min(n, left)];
            for (int i = 0; i < out.length; i++) {
                out[i] = stream[at + 1 + i];
            }
            return out;
        }

        @Override
        public void accept(int seq, int nAccepted) {
            cursor[seq] += 1 + nAccepted;
        }

        @Override
        public void reset(int seq) {
            cursor[seq] = 0;
        }
    }

    /**
     * Drafts the reference's greedy continuation as a tree: the chain as the best path plus a wrong decoy under
     * the root, within the request's node limit — the batch engine's tree verify mode.
     */
    private static final class OracleTreeDrafter implements BatchDrafter {

        private final int[] stream;
        private final int[] cursor = new int[32];

        OracleTreeDrafter(int[] stream) {
            this.stream = stream;
        }

        @Override
        public boolean proposesTrees() {
            return true;
        }

        @Override
        public int[] draft(int seq, int lastToken, int n) {
            return new int[0];
        }

        @Override
        public TreeProposal[] draftTreeBatch(List<Request> requests) {
            TreeProposal[] out = new TreeProposal[requests.size()];
            for (int i = 0; i < requests.size(); i++) {
                Request r = requests.get(i);
                int at = streamFrom(stream, cursor, r.seq(), r.lastToken());
                int t = Math.min(r.limit(), stream.length - at - 1);
                int chain = t >= 2 ? t - 1 : t;
                List<DraftTree.Node> nodes = new ArrayList<>();
                double[] scores = new double[t];
                int w = 0;
                for (int j = 0; j < chain; j++) {
                    int parent = j == 0 ? -1 : (j == 1 ? 0 : j);
                    nodes.add(new DraftTree.Node(stream[at + 1 + j], parent, j + 1, 0));
                    scores[w++] = -(j + 1) * 0.1;
                    if (j == 0 && t >= 2) {
                        nodes.add(new DraftTree.Node(stream[at + 1] == 4 ? 5 : 4, 0, 2, 0));
                        scores[w++] = -0.15;
                    }
                }
                out[i] = new TreeProposal(new DraftTree(nodes), scores);
            }
            return out;
        }

        @Override
        public void accept(int seq, int nAccepted) {
            cursor[seq] += 1 + nAccepted;
        }

        @Override
        public void reset(int seq) {
            cursor[seq] = 0;
        }
    }
}
