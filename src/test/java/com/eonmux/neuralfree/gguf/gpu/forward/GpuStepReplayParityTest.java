package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.RealModels;
import org.junit.Assume;
import org.junit.Test;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in parity of the multi-row step replay ({@code neuralfree.bench.stepReplayParity=1} or the
 * {@code NEURALFREE_BENCH_STEP_REPLAY_PARITY} environment variable): the same sequence of contiguous
 * multi-row steps (the speculative verify shape, widths 2 to 16) runs with the step replay off, as a
 * body replay and as a patched replay, and the greedy continuation every run decodes after them must be
 * the same token ids. One run per model also feeds the same ids one row at a time, so the report shows
 * whether the multi-row rows' logits agree with the per-row decode's ({@code LLM_FlashAttnMma_*} against
 * {@code LLM_FlashDecode_*}, whose reduction orders differ in the low bits). Asserts the replay modes'
 * equality; the multi-row against per-row verdict goes to the result report.
 */
public class GpuStepReplayParityTest {

    private static final String SWITCH = "NEURALFREE_BENCH_STEP_REPLAY_PARITY";
    private static final String ENABLE = "neuralfree.bench.stepReplayParity";
    private static final String TARGET =
            RealModels.path("Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf");
    private static final String DRAFTER =
            RealModels.path("Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf");
    private static final String FED = "The solar system formed about 4.6 billion years ago from a giant "
            + "cloud of gas and dust. Gravity pulled the material together, and most of it gathered in the "
            + "center to form the Sun. The rest flattened into a disk from which the planets grew. The "
            + "planets then settled into the orbits we see today, and the leftovers became asteroids and "
            + "comets. Over time the Sun slowly converted its hydrogen into helium, and the lightest gas "
            + "drifted away from the inner worlds, leaving rocky worlds near the Sun and giants farther out.";
    private static final int PROMPT_TOKENS = 24;
    private static final int TOKEN = 4;
    /** The multi-row widths one run cycles through, the speculative verify's row counts. */
    private static final int[] WIDTHS = {2, 5, 9, 16};
    private static final int GREEDY_STEPS = 40;
    private static final int N_CTX = 256;

    @Test
    public void replayedMultiRowStepsMatchEagerSteps() throws IOException {
        Assume.assumeTrue("set " + ENABLE + "=1 to run the step replay parity probe",
                "1".equals(System.getProperty(ENABLE, System.getenv(SWITCH))));
        Assume.assumeTrue("target model absent", Files.exists(Path.of(TARGET)));
        GpuTestUtil.assumeCudaAvailable();
        GpuDecodeReplayTimingTest.debugReplayLogs();
        int[] fed = fedIds();
        String[] models = {TARGET, DRAFTER};
        for (String model : models) {
            if (!Files.exists(Path.of(model))) {
                report("skip (absent): " + model);
                continue;
            }
            String name = Path.of(model).getFileName().toString();
            List<Integer> eager = stepped(model, "off", fed);
            List<Integer> body = stepped(model, "body", fed);
            List<Integer> patched = stepped(model, "patched", fed);
            compare("multi-row " + name + " eager vs body", eager, body);
            compare("multi-row " + name + " eager vs patched", eager, patched);
            compare("multi-row " + name + " body vs patched", body, patched);
            List<Integer> perRow = perRow(model, fed);
            compare("row shape " + name + " multi-row (patched) vs per-row", patched, perRow);
        }
    }

    /** The fed ids: the fixed text through the target's tokenizer, past the prompt prefix. */
    private int[] fedIds() throws IOException {
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CUDA)
                .memory(MemoryConfig.builder().nCtx(N_CTX).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(TARGET, cfg)) {
            int[] ids = model.tokenizer().encode(FED);
            return ids.length > PROMPT_TOKENS
                    ? java.util.Arrays.copyOfRange(ids, PROMPT_TOKENS, ids.length) : ids;
        }
    }

    /**
     * Contiguous multi-row steps over {@code fed}'s ids at cycling widths, then the greedy continuation,
     * with the replay mode {@code mode}: {@code off} runs every dispatch directly, {@code body} captures
     * the trunk and replays it through the body, {@code patched} replays through the step patchers.
     */
    private static List<Integer> stepped(String modelPath, String mode, int[] fed) throws IOException {
        setMode(mode);
        try {
            EngineConfig cfg = EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.CUDA)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .memory(MemoryConfig.builder().nCtx(N_CTX).build())
                    .build();
            try (EngineModel model = LLMModelFactory.open(modelPath, cfg)) {
                ForwardBackend f = model.forwardBackend();
                int at = 0;
                int i = 0;
                while (at + WIDTHS[0] <= fed.length) {
                    int width = WIDTHS[i % WIDTHS.length];
                    if (at + width > fed.length) {
                        break;
                    }
                    f.prefill(java.util.Arrays.copyOfRange(fed, at, at + width), at);
                    at += width;
                    i++;
                }
                List<Integer> ids = new ArrayList<>();
                for (int s = 0; s < GREEDY_STEPS; s++) {
                    ids.add(f.decodeGreedyOnDevice(TOKEN, at + s));
                }
                report("  " + mode + " " + Path.of(modelPath).getFileName() + ": " + i + " steps to pos "
                        + at + ", continuation " + ids(ids));
                return ids;
            }
        } finally {
            clearMode();
        }
    }

    /**
     * The per-row reference of {@link #stepped}: the same fed ids one decode at a time at the same
     * positions, then the greedy continuation.
     */
    private static List<Integer> perRow(String modelPath, int[] fed) throws IOException {
        setMode("off");
        try {
            EngineConfig cfg = EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.CUDA)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .memory(MemoryConfig.builder().nCtx(N_CTX).build())
                    .build();
            try (EngineModel model = LLMModelFactory.open(modelPath, cfg)) {
                ForwardBackend f = model.forwardBackend();
                for (int at = 0; at < fed.length; at++) {
                    f.decodeGreedyOnDevice(fed[at], at);
                }
                List<Integer> ids = new ArrayList<>();
                for (int s = 0; s < GREEDY_STEPS; s++) {
                    ids.add(f.decodeGreedyOnDevice(TOKEN, fed.length + s));
                }
                return ids;
            }
        } finally {
            clearMode();
        }
    }

    private static void setMode(String mode) {
        if (mode.equals("off")) {
            System.setProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY, "false");
            return;
        }
        if (mode.equals("body")) {
            System.setProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY, "false");
            return;
        }
    }

    private static void clearMode() {
        System.clearProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY);
        System.clearProperty(GpuForwardBackend.CHAIN_REPLAY_PROPERTY);
    }

    /** Compares the two runs' ids and reports them. */
    private static void compare(String mode, List<Integer> a, List<Integer> b) {
        int diverge = -1;
        for (int i = 0; i < Math.min(a.size(), b.size()); i++) {
            if (!a.get(i).equals(b.get(i))) {
                diverge = i;
                break;
            }
        }
        String line = diverge < 0 ? "== step replay parity " + mode + ": identical"
                : "== step replay parity " + mode + ": diverges at " + diverge + ": " + a.get(diverge)
                + " vs " + b.get(diverge);
        System.out.println(line);
        report(line);
    }

    /** The ids as a compact list. */
    private static String ids(List<Integer> tokens) {
        StringBuilder out = new StringBuilder();
        for (int token : tokens) {
            if (out.length() > 0) {
                out.append(' ');
            }
            out.append(token);
        }
        return out.toString();
    }

    /** One report line to the console and to {@code build/step-replay-parity.txt}. */
    private static void report(String line) {
        System.out.println(line);
        try (PrintWriter out = new PrintWriter(new FileWriter(
                Path.of("build", "step-replay-parity.txt").toFile(), true))) {
            out.println(String.format(Locale.ROOT, "%s", line));
        } catch (IOException e) {
            System.out.println("the parity report write failed: " + e);
        }
    }
}
