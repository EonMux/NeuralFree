package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.gpu.diag.GpuDiagnostics;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Engine-level combinations of a two-device layer split over the OpenCL devices of the machine: a LoRA adapter
 * across the split gives the logits of the one-device load with the adapter, each stage applying the pairs of its
 * own layers (the embedding and output pairs on the main device); {@code neuralfree.diag} and the decode profiler
 * report every stage under its device; and the layouts the split cannot run are refused by name, never swallowed by
 * the CPU fallback of {@link LLMModelFactory}. Host placement, control vectors, drafters, {@code --main-gpu} and the
 * VRAM cap over the split are covered by {@code GpuSplitHostPlacementTest}, {@code GpuSplitDeviceSettingsTest} and
 * {@code GpuSplitSpeculativeTest}.
 */
public class GpuSplitCombinationsTest {

    private static final int RANK = 4;
    private static final float ALPHA = 8.0f;
    private static final float USER_SCALE = 0.75f;
    private static final float TOL = 1e-3f;
    private static final String EMBD = "token_embd.weight";
    private static final String[] LAYER_PROJECTIONS = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate",
            "ffn_up", "ffn_down"};
    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final int NEXT = 77;

    private final List<Path> files = new ArrayList<>();
    private final Map<String, String> savedProperties = new LinkedHashMap<>();

    @Before
    public void requireTwoDevices() {
        GpuTestUtil.assumeOpenCLAvailable();
        int devices = openClDeviceCount();
        Assume.assumeTrue("the split needs two OpenCL devices, found " + devices, devices >= 2);
    }

    @After
    public void cleanUp() throws IOException {
        for (Map.Entry<String, String> e : savedProperties.entrySet()) {
            if (e.getValue() == null) {
                System.clearProperty(e.getKey());
            } else {
                System.setProperty(e.getKey(), e.getValue());
            }
        }
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    private void property(String key, String value) {
        if (!savedProperties.containsKey(key)) {
            savedProperties.put(key, System.getProperty(key));
        }
        System.setProperty(key, value);
    }

    private static EngineConfig.Builder oneDevice() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .addGpuDevice("0")
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(4).build());
    }

    private static EngineConfig.Builder split() {
        return oneDevice().splitMode(EngineConfig.SplitMode.LAYER).addGpuDevice("1");
    }

    // ------------------------------------------------------------------ LoRA

    @Test
    public void anAdapterAcrossTheSplitGivesTheLogitsOfTheOneDeviceLoad() throws IOException {
        GgufTestWriter baseWriter = SyntheticModels.llama();
        Path base = written(baseWriter);
        String adapter = written(adapter(baseWriter)).toString();

        float[][] reference = logits(base, oneDevice().addLora(adapter, USER_SCALE).build());
        float[][] adapted = logits(base, split().addLora(adapter, USER_SCALE).build());
        float[][] plain = logits(base, split().build());

        for (int i = 0; i < reference.length; i++) {
            assertArrayEquals("step " + i + " matches the one-device load", reference[i], adapted[i], TOL);
            assertTrue("the adapter changes the split logits of step " + i,
                    maxAbsDiff(plain[i], adapted[i]) > 100 * TOL);
        }
    }

    /** Prefill logits of {@link #PROMPT} (two ubatches) and decode logits of {@link #NEXT}. */
    private static float[][] logits(Path model, EngineConfig cfg) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            ForwardBackend f = m.forwardBackend();
            float[] prefill = f.prefill(PROMPT, 0).clone();
            float[] decode = f.decode(NEXT, PROMPT.length).clone();
            f.reset();
            return new float[][]{prefill, decode};
        }
    }

    /**
     * An adapter of every layer projection, the output head and the token table (llama-adapter.cpp:203-375): a
     * projection {@code W {in, out}} takes {@code A {in, r}} and {@code B {r, out}}, the token table
     * {@code E {n_embd, n_vocab}} takes {@code A {r, n_vocab}} and {@code B {r, n_embd}}.
     */
    private static GgufTestWriter adapter(GgufTestWriter base) {
        List<String> names = new ArrayList<>();
        for (int il = 0; il < SyntheticModels.Config.llama().nLayer(); il++) {
            for (String p : LAYER_PROJECTIONS) {
                names.add("blk." + il + "." + p + ".weight");
            }
        }
        names.add("output.weight");
        names.add(EMBD);
        GgufTestWriter w = new GgufTestWriter().architecture("llama");
        w.str("general.type", "adapter").str("adapter.type", "lora").f32("adapter.lora.alpha", ALPHA);
        long seed = 2000;
        for (String n : names) {
            long[] ne = base.tensorShape(n);
            int cols = Math.toIntExact(ne[0]);
            int rows = Math.toIntExact(ne[1]);
            if (EMBD.equals(n)) {
                w.tensorF32(n + ".lora_a", new long[]{RANK, ne[1]}, random(seed++, rows * RANK));
                w.tensorF32(n + ".lora_b", new long[]{RANK, ne[0]}, random(seed++, cols * RANK));
            } else {
                w.tensorF32(n + ".lora_a", new long[]{ne[0], RANK}, random(seed++, RANK * cols));
                w.tensorF32(n + ".lora_b", new long[]{RANK, ne[1]}, random(seed++, rows * RANK));
            }
        }
        return w;
    }

    // ------------------------------------------------------------------ diagnostics and profiler

    @Test
    public void diagnosticsReportEveryStageUnderItsDevice() throws IOException {
        Path base = written(SyntheticModels.llama());
        property(GpuDiagnostics.PROP_ENABLED, "true");
        property(GpuDiagnostics.PROP_STEPS, "1");
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream err = System.err;
        try (PrintStream sink = new PrintStream(captured, true, "UTF-8")) {
            System.setErr(sink);
            try (GGUFModelGPU m = new GGUFModelGPU(base.toString(), null, split().build())) {
                m.forwardBackend().decode(1, 0);
            } finally {
                System.setErr(err);
            }
        }
        String text = new String(captured.toByteArray(), StandardCharsets.UTF_8);
        Map<String, Set<String>> stagesByDevice = new LinkedHashMap<>();
        Matcher m = Pattern.compile("\\[DIAG\\] step=1 device=(\\S+ '[^']*') (\\S+) ").matcher(text);
        while (m.find()) {
            stagesByDevice.computeIfAbsent(m.group(1), k -> new LinkedHashSet<>()).add(m.group(2));
        }
        assertEquals(text, 2, stagesByDevice.size());
        List<Set<String>> stages = new ArrayList<>(stagesByDevice.values());
        assertTrue(text, stages.get(0).contains("embedding"));
        assertTrue(text, stages.get(0).contains("layer-0"));
        assertTrue(text, !stages.get(0).contains("layer-1"));
        assertTrue(text, stages.get(1).contains("layer-1"));
        assertTrue(text, !stages.get(1).contains("embedding"));
    }

    @Test
    public void theDecodeProfilerTimesEveryDevice() throws IOException {
        Path base = written(SyntheticModels.llama());
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream out = System.out;
        try (PrintStream sink = new PrintStream(captured, true, "UTF-8")) {
            System.setOut(sink);
            try (GGUFModelGPU m = new GGUFModelGPU(base.toString(), null, split().build())) {
                m.profileDecodeToken(1, 2, false);
            } finally {
                System.setOut(out);
            }
        }
        String text = new String(captured.toByteArray(), StandardCharsets.UTF_8);
        Matcher m = Pattern.compile("(?m)^  on (\\S+ '[^']*') \\(layers ([0-9, -]+)\\):$").matcher(text);
        List<String> devices = new ArrayList<>();
        List<String> layers = new ArrayList<>();
        while (m.find()) {
            devices.add(m.group(1));
            layers.add(m.group(2));
        }
        assertEquals(text, 2, devices.size());
        assertTrue(text, !devices.get(0).equals(devices.get(1)));
        assertEquals(text, "0", layers.get(0));
        assertEquals(text, "1", layers.get(1));
        assertTrue(text, text.contains("STEP:"));
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void layoutsTheSplitCannotRunAreRefusedByName() throws IOException {
        String base = written(SyntheticModels.llama()).toString();
        String layer = EngineConfig.PROP_SPLIT_MODE + "=layer";

        String noDeviceLayer = refusal(base, split().gpuLayers(1).build());
        assertTrue(noDeviceLayer, noDeviceLayer.startsWith(layer));
        assertTrue(noDeviceLayer, noDeviceLayer.contains(EngineConfig.PROP_N_GPU_LAYERS));

        String hostEverything = refusal(base, split().addTensorOverride("blk\\.1\\.ffn_.*", "CPU").build());
        assertTrue(hostEverything, hostEverything.startsWith(layer));
        assertTrue(hostEverything, hostEverything.contains("on the GPUs"));

        String shares = refusal(base, split().tensorSplit(new double[]{1, 1, 1}).build());
        assertTrue(shares, shares.contains(EngineConfig.PROP_TENSOR_SPLIT));
    }

    /** The message of the refusal, through the factory, which must not fall back to the CPU engine. */
    private static String refusal(String path, EngineConfig cfg) {
        return assertThrows(IllegalArgumentException.class, () -> LLMModelFactory.open(path, cfg).close())
                .getMessage();
    }

    // ------------------------------------------------------------------ helpers

    private static int openClDeviceCount() {
        HWAcceleration hw = new HWAcceleration(java.util.EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try {
            return hw.getCLDevices().size();
        } finally {
            hw.shutdown();
        }
    }

    private static float[] random(long seed, int n) {
        Random r = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = 0.2f * (r.nextFloat() - 0.5f);
        }
        return out;
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}
