package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.testutil.RealModels;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.Assume;
import org.junit.Test;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in token parity of the decode replay on a real model ({@code neuralfree.bench.replayParity=1} or the
 * {@code NEURALFREE_BENCH_REPLAY_PARITY} environment variable, which the test task forwards): the same
 * greedy generation runs with the replay disabled and enabled — plain decode on the 4B target and a chain
 * speculation session over the real 4B target with its 0.6B drafter — and the report prints the token ids
 * and their first divergence. Asserts nothing; the verdict goes to the result report.
 */
public class GpuDecodeReplayParityTest {

    private static final String SWITCH = "NEURALFREE_BENCH_REPLAY_PARITY";
    private static final String MODEL =
            RealModels.path("Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf");
    private static final String DRAFTER =
            RealModels.path("Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf");
    private static final String PROMPT = "The solar system formed about 4.6 billion years ago from a giant "
            + "cloud of gas and dust. Gravity pulled the material together, and most of it gathered in the "
            + "center to form the Sun. The rest flattened into a disk from which the planets grew.";
    private static final int WARMUP = 4;
    /** The probe's decode steps ({@code neuralfree.bench.replayParitySteps}), long enough to cross the
     * flash decode's split-K boundaries. */
    private static final int STEPS = Integer.getInteger("neuralfree.bench.replayParitySteps", 160);
    private static final int TOKEN = 4;
    private static final int CHAIN_N_CTX = 1024;
    private static final int CHAIN_UBATCH = 64;
    private static final int CHAIN_DRAFT_MAX = 8;
    private static final int CHAIN_TOKENS = 48;
    private static final int CHAIN_REPS = 3;

    @Test
    public void replayedDecodeMatchesDirectDecode() throws IOException {
        Assume.assumeTrue("set " + SWITCH + "=1 to run the decode replay parity probe",
                "1".equals(System.getProperty("neuralfree.bench.replayParity",
                        System.getenv(SWITCH))));
        Assume.assumeTrue("model absent", Files.exists(Path.of(MODEL)));
        GpuTestUtil.assumeCudaAvailable();
        GpuDecodeReplayTimingTest.debugReplayLogs();
        List<Integer> direct = generate(MODEL, "false", null);
        List<Integer> replayed = generate(MODEL, "true", null);
        compare("plain decode " + Path.of(MODEL).getFileName(), direct, replayed);
        Assume.assumeTrue("drafter absent", Files.exists(Path.of(DRAFTER)));
        List<Integer> drafterDirect = generate(DRAFTER, "false", null);
        List<Integer> drafterReplayed = generate(DRAFTER, "true", null);
        compare("plain decode " + Path.of(DRAFTER).getFileName(), drafterDirect, drafterReplayed);
        int[] prompt = tokenize();
        List<Integer> prefilledDirect = generate(MODEL, "false", prompt);
        List<Integer> prefilledReplayed = generate(MODEL, "true", prompt);
        compare("decode after prefill " + Path.of(MODEL).getFileName(), prefilledDirect, prefilledReplayed);
        List<Integer> drafterPrefDirect = generate(DRAFTER, "false", prompt);
        List<Integer> drafterPrefReplayed = generate(DRAFTER, "true", prompt);
        compare("decode after prefill " + Path.of(DRAFTER).getFileName(), drafterPrefDirect, drafterPrefReplayed);
        List<Integer> rewindDirect = rewound(MODEL, "false", prompt);
        List<Integer> rewindReplayed = rewound(MODEL, "true", prompt);
        compare("decode after rollback " + Path.of(MODEL).getFileName(), rewindDirect, rewindReplayed);
        List<Integer> rewindDraftDirect = rewound(DRAFTER, "false", prompt);
        List<Integer> rewindDraftReplayed = rewound(DRAFTER, "true", prompt);
        compare("decode after rollback " + Path.of(DRAFTER).getFileName(), rewindDraftDirect, rewindDraftReplayed);
        int[][] chainDirect = chain("false", prompt);
        int[][] chainReplayed = chain("true", prompt);
        for (int gen = 0; gen < Math.min(chainDirect.length, chainReplayed.length); gen++) {
            compare("chain generation " + gen, toList(chainDirect[gen]), toList(chainReplayed[gen]));
        }
    }

    /** Compares the two runs' ids and reports them. */
    private static void compare(String mode, List<Integer> direct, List<Integer> replayed) {
        String verdict = "identical";
        for (int i = 0; i < Math.min(direct.size(), replayed.size()); i++) {
            if (!direct.get(i).equals(replayed.get(i))) {
                verdict = "diverges at step " + i + ": direct " + direct.get(i) + " vs replayed "
                        + replayed.get(i);
                break;
            }
        }
        String line = "== decode replay parity " + mode + ": " + verdict;
        System.out.println(line);
        report(line);
        report("  direct   " + ids(direct));
        report("  replayed " + ids(replayed));
    }

    /**
     * One greedy generation of {@code modelPath} with the decode replay property set from {@code replay};
     * a non-null {@code prompt} is prefilled first, so the decode steps run over a prefilled memory.
     */
    private static List<Integer> generate(String modelPath, String replay, int[] prompt) throws IOException {
        System.setProperty(GpuDecodeReplay.PROPERTY, replay);
        try {
            EngineConfig cfg = EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.CUDA)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .memory(MemoryConfig.builder().nCtx(256).build())
                    .build();
            try (EngineModel model = LLMModelFactory.open(modelPath, cfg)) {
                ForwardBackend f = model.forwardBackend();
                int pos0 = 0;
                if (prompt != null) {
                    f.prefill(prompt, 0);
                    pos0 = prompt.length;
                }
                List<Integer> ids = new ArrayList<>();
                for (int i = 0; i < WARMUP + STEPS; i++) {
                    int id = f.decodeGreedyOnDevice(TOKEN, pos0 + i);
                    if (i >= WARMUP) {
                        ids.add(id);
                    }
                }
                return ids;
            }
        } finally {
            System.clearProperty(GpuDecodeReplay.PROPERTY);
        }
    }

    /**
     * Decodes a few steps, rolls back and decodes again from the rewound position, with the decode replay
     * property set from {@code replay}: the ids from the rewind on.
     */
    private static List<Integer> rewound(String modelPath, String replay, int[] prompt) throws IOException {
        System.setProperty(GpuDecodeReplay.PROPERTY, replay);
        try {
            EngineConfig cfg = EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.CUDA)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .memory(MemoryConfig.builder().nCtx(256).build())
                    .build();
            try (EngineModel model = LLMModelFactory.open(modelPath, cfg)) {
                ForwardBackend f = model.forwardBackend();
                f.prefill(prompt, 0);
                int pos = prompt.length;
                for (int i = 0; i < 10; i++) {
                    f.decodeGreedyOnDevice(TOKEN, pos + i);
                }
                f.rollback(3);
                List<Integer> ids = new ArrayList<>();
                for (int i = 0; i < STEPS; i++) {
                    ids.add(f.decodeGreedyOnDevice(TOKEN, pos + 7 + i));
                }
                return ids;
            }
        } finally {
            System.clearProperty(GpuDecodeReplay.PROPERTY);
        }
    }

    /**
     * One chain speculation session over the real pair with the decode replay property set from
     * {@code replay}: the emitted token ids of each generation, warmup first, then the reps the benchmark
     * runs, over one session.
     */
    private static int[][] chain(String replay, int[] prompt) throws IOException {
        System.setProperty(GpuDecodeReplay.PROPERTY, replay);
        try {
            EngineConfig cfg = EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.CUDA)
                    .memory(MemoryConfig.builder().nCtx(CHAIN_N_CTX).ubatch(CHAIN_UBATCH).build())
                    .draftModelPath(DRAFTER).draftMax(CHAIN_DRAFT_MAX)
                    .speculativeOptions(SpeculativeOptions.builder().pSplit(0).treeVerify(false).build())
                    .build();
            try (EngineModel model = LLMModelFactory.open(MODEL, cfg);
                 SpeculativeSession session = LLMModelFactory.speculate(model, cfg)) {
                GenerationRequest request = GenerationRequest.builder().maxNewTokens(CHAIN_TOKENS)
                        .sampler(SamplerConfig.greedyDecoding()).build();
                int[][] all = new int[1 + CHAIN_REPS][];
                GenerationResult warm = session.generate(prompt, generation(16), (TokenCallback) null);
                all[0] = warm.tokens();
                for (int rep = 0; rep < CHAIN_REPS; rep++) {
                    GenerationResult result = session.generate(prompt, request, (TokenCallback) null);
                    all[1 + rep] = result.tokens();
                }
                String stats = String.valueOf(session.lastStats());
                report("  chain stats replay=" + replay + ": " + stats);
                return all;
            }
        } finally {
            System.clearProperty(GpuDecodeReplay.PROPERTY);
        }
    }

    /** A generation request of {@code maxNewTokens} greedy tokens. */
    private static GenerationRequest generation(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** The prompt's ids through the target's tokenizer. */
    private static int[] tokenize() throws IOException {
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CUDA)
                .memory(MemoryConfig.builder().nCtx(CHAIN_N_CTX).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(MODEL, cfg)) {
            return model.tokenizer().encode(PROMPT);
        }
    }

    private static List<Integer> toList(int[] tokens) {
        List<Integer> out = new ArrayList<>(tokens.length);
        for (int token : tokens) {
            out.add(token);
        }
        return out;
    }

    /** The ids as a compact list. */
    private static String ids(List<Integer> tokens) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < tokens.size(); i++) {
            if (i > 0) {
                out.append(' ');
            }
            out.append(tokens.get(i));
        }
        return out.toString();
    }

    /** One report line to the console and to {@code build/replay-parity.txt}. */
    private static void report(String line) {
        try (PrintWriter out = new PrintWriter(new FileWriter(
                Path.of("build", "replay-parity.txt").toFile(), true))) {
            out.println(line);
        } catch (IOException e) {
            System.out.println("the parity report write failed: " + e);
        }
    }
}
