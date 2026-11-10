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
 * Opt-in timing of the decode replay's states on a real model ({@code NEURALFREE_BENCH_REPLAY_TIMING=1} or the
 * {@code neuralfree.bench.replayTiming} property, which the test task forwards): one generation decodes with the
 * replay disabled and again enabled, and the report prints every step's wall time and the replay's state, so a
 * patched replay that never reaches the trusted state or recaptures shows up next to the direct steps. Asserts
 * nothing; the numbers go to the result report. The probe runs the 4B Q8_0 model on CUDA, or the model file
 * {@code neuralfree.bench.replayTimingModel} names on the backend {@code neuralfree.bench.replayTimingBackend}
 * names ({@code VULKAN}).
 */
public class GpuDecodeReplayTimingTest {

    private static final String SWITCH = "NEURALFREE_BENCH_REPLAY_TIMING";
    private static final String MODEL = System.getProperty("neuralfree.bench.replayTimingModel",
            RealModels.path("Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf"));
    private static final EngineConfig.BackendMode BACKEND = EngineConfig.BackendMode.valueOf(
            System.getProperty("neuralfree.bench.replayTimingBackend", "CUDA"));
    private static final int WARMUP = 8;
    private static final int STEPS = 24;

    @Test
    public void perStepTimingAndReplayState() throws IOException {
        Assume.assumeTrue("set " + SWITCH + "=1 to run the decode replay timing probe",
                "1".equals(System.getProperty("neuralfree.bench.replayTiming",
                        System.getenv(SWITCH))));
        Assume.assumeTrue("model absent", Files.exists(Path.of(MODEL)));
        if (BACKEND == EngineConfig.BackendMode.VULKAN) {
            GpuTestUtil.assumeVulkanAvailable();
        } else {
            GpuTestUtil.assumeCudaAvailable();
        }
        run("direct", "false");
        run("patched replay", "true");
    }

    /**
     * Routes the replay's debug logs to {@code build/replay-debug.log}: a probe that needs the capture's
     * per-step reasons (why a capture stays a body replay) calls this; the timing probe does not, because
     * the per-patch debug logging would land in its measurements.
     */
    static void debugReplayLogs() throws IOException {
        Path cfg = Path.of("build", "replay-log4j2.xml");
        String xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                + "<Configuration status=\"WARN\">\n"
                + "    <Appenders>\n"
                + "        <File name=\"ReplayDebug\" fileName=\"build/replay-debug.log\">\n"
                + "            <PatternLayout pattern=\"%-5level %logger{36} - %msg%n\"/>\n"
                + "        </File>\n"
                + "    </Appenders>\n"
                + "    <Loggers>\n"
                + "        <Logger name=\"com.eonmux.neuralfree.gguf.gpu.forward\" level=\"debug\"/>\n"
                + "        <Logger name=\"com.eonmux.neuralfree.gguf.engine\" level=\"debug\"/>\n"
                + "        <Logger name=\"com.eonmux.neuralfree.gguf.ops.attn\" level=\"debug\"/>\n"
                + "        <Logger name=\"com.eonmux.neuralfree.acceleration.backend.cuda\" level=\"debug\"/>\n"
                + "        <Logger name=\"rounds-probe\" level=\"debug\"/>\n"
                + "        <Root level=\"info\">\n"
                + "            <AppenderRef ref=\"ReplayDebug\"/>\n"
                + "        </Root>\n"
                + "    </Loggers>\n"
                + "</Configuration>\n";
        Files.writeString(cfg, xml);
        System.setProperty("log4j2.configurationFile", cfg.toAbsolutePath().toString());
        org.apache.logging.log4j.core.LoggerContext ctx =
                (org.apache.logging.log4j.core.LoggerContext) org.apache.logging.log4j.LogManager.getContext(false);
        ctx.reconfigure();
    }

    /** One report line to the console and to {@code build/replay-timing.txt}. */
    private static void report(String line) {
        try (java.io.PrintWriter out = new java.io.PrintWriter(new java.io.FileWriter(
                Path.of("build", "replay-timing.txt").toFile(), true))) {
            out.println(line);
        } catch (IOException e) {
            System.out.println("the timing report write failed: " + e);
        }
    }

    private static void run(String label, String replay) throws IOException {
        System.setProperty(GpuDecodeReplay.PROPERTY, replay);
        try {
            EngineConfig cfg = EngineConfig.builder()
                    .backend(BACKEND)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .memory(MemoryConfig.builder().nCtx(256).build())
                    .build();
            try (EngineModel model = LLMModelFactory.open(MODEL, cfg)) {
                ForwardBackend f = model.forwardBackend();
                int token = 4;
                for (int i = 0; i < WARMUP; i++) {
                    f.decode(token, i);
                }
                List<String> lines = new ArrayList<>();
                double sum = 0;
                for (int i = 0; i < STEPS; i++) {
                    int pos = WARMUP + i;
                    long t0 = System.nanoTime();
                    f.decode(token, pos);
                    double ms = (System.nanoTime() - t0) / 1e6;
                    sum += ms;
                    String state = f instanceof GpuForwardBackend
                            ? ((GpuForwardBackend) f).decodeReplayStats() : "";
                    lines.add(String.format(Locale.ROOT, "  %s step %d: %6.2f ms  %s", label, pos, ms, state));
                }
                System.out.println(String.format(Locale.ROOT,
                        "== decode replay timing %s: %.2f ms/step over %d steps", label, sum / STEPS, STEPS));
                lines.forEach(System.out::println);
                report(String.format(Locale.ROOT,
                        "== decode replay timing %s: %.2f ms/step over %d steps", label, sum / STEPS, STEPS));
                lines.forEach(GpuDecodeReplayTimingTest::report);
            }
        } finally {
            System.clearProperty(GpuDecodeReplay.PROPERTY);
        }
    }
}
