package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.testutil.RealModels;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in probe of the chain drafter's round pattern on the small model alone ({@code
 * neuralfree.bench.replayRounds=1}): a prompt prefill, then rounds of eight decode steps and a
 * rollback of the drafted rows, with the replay disabled and enabled; the report prints the first
 * step whose top id diverges. The chain session's drafter runs exactly this pattern per round, so a
 * divergence here localizes the chain replay's staleness without the 4B target.
 */
public class GpuDecodeReplayRoundsTest {

    private static final String SWITCH = "NEURALFREE_BENCH_REPLAY_ROUNDS";
    private static final String MODEL =
            RealModels.path("Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf");
    private static final String PROMPT = "The solar system formed about 4.6 billion years ago from a giant "
            + "cloud of gas and dust. Gravity pulled the material together, and most of it gathered in the "
            + "center to form the Sun. The rest flattened into a disk from which the planets grew.";
    private static final int DRAFT = 8;
    private static final int ROUNDS = 40;

    @Test
    public void drafterRoundsMatch() throws IOException {
        Assume.assumeTrue("set " + SWITCH + "=1 to run the drafter rounds probe",
                "1".equals(System.getProperty("neuralfree.bench.replayRounds",
                        System.getenv(SWITCH))));
        Assume.assumeTrue("model absent", Files.exists(Path.of(MODEL)));
        GpuTestUtil.assumeCudaAvailable();
        GpuDecodeReplayTimingTest.debugReplayLogs();
        List<int[]> promptTokens = new ArrayList<>();
        int[] prompt = tokenize();
        List<Integer> direct = rounds("false", prompt);
        List<Integer> replayed = rounds("true", prompt);
        int diverge = -1;
        for (int i = 0; i < Math.min(direct.size(), replayed.size()); i++) {
            if (!direct.get(i).equals(replayed.get(i))) {
                diverge = i;
                break;
            }
        }
        String verdict = diverge < 0 ? "identical over " + direct.size() + " decodes"
                : "diverges at decode " + diverge + " (round " + (diverge / DRAFT) + ", step "
                + (diverge % DRAFT) + "): direct " + direct.get(diverge) + " vs replayed "
                + replayed.get(diverge);
        System.out.println("== drafter rounds: " + verdict);
        report("== drafter rounds: " + verdict);
        report("  direct   " + ids(direct));
        report("  replayed " + ids(replayed));
    }

    private static int[] tokenize() throws IOException {
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CUDA)
                .memory(MemoryConfig.builder().nCtx(1024).ubatch(64).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(MODEL, cfg)) {
            return model.tokenizer().encode(PROMPT);
        }
    }

    /**
     * The drafter's rounds with the decode replay property set from {@code replay}: one full prefill,
     * then per round eight decodes chained from each step's top id at consecutive positions and a
     * rollback of the seven drafted rows while the round advances one position, as an all-reject chain
     * session's drafter does; every decode's top id is collected.
     */
    private static List<Integer> rounds(String replay, int[] prompt) throws IOException {
        System.setProperty(GpuDecodeReplay.PROPERTY, replay);
        try {
            EngineConfig cfg = EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.CUDA)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .memory(MemoryConfig.builder().nCtx(1024).ubatch(64).build())
                    .build();
            try (EngineModel model = LLMModelFactory.open(MODEL, cfg)) {
                ForwardBackend f = model.forwardBackend();
                f.prefill(prompt, 0);
                int pos = prompt.length;
                int token = prompt[prompt.length - 1];
                List<Integer> ids = new ArrayList<>();
                org.apache.logging.log4j.Logger probe =
                        org.apache.logging.log4j.LogManager.getLogger("rounds-probe");
                for (int round = 0; round < ROUNDS; round++) {
                    for (int i = 0; i < DRAFT; i++) {
                        probe.debug("begin round {} step {} pos {}", round, i, pos + i);
                        token = f.decodeGreedyOnDevice(token, pos + i);
                        ids.add(token);
                    }
                    f.rollback(DRAFT - 1);
                    pos++;
                }
                return ids;
            }
        } finally {
            System.clearProperty(GpuDecodeReplay.PROPERTY);
        }
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

    /** One report line to the console and to {@code build/replay-rounds.txt}. */
    private static void report(String line) {
        try (java.io.PrintWriter out = new java.io.PrintWriter(new java.io.FileWriter(
                Path.of("build", "replay-rounds.txt").toFile(), true))) {
            out.println(String.format(Locale.ROOT, "%s", line));
        } catch (IOException e) {
            System.out.println("the rounds report write failed: " + e);
        }
    }
}
