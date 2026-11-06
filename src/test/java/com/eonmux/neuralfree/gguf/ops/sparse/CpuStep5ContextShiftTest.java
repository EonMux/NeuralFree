package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Context shift of a Step5 sequence: after the memory moves the KV rows down, the mixer moves the open-region ring to
 * the new positions, re-rotates the ring keys by the shift delta and rebuilds every closed-region summary from the
 * rows its region now holds.
 *
 * <p>The rows a shift does not move and every value row move unchanged. A moved key row is re-rotated from the
 * projection the cache holds, which the forward pass already rounded to BF16 (the normative cast point), so the
 * pre-rotation value is not recoverable: the re-rotated key stays within one BF16 step of the key a fresh run over
 * the kept tokens computes. A rebuilt summary can therefore land one E4M3 step from a fresh run's when its region
 * holds re-rotated rows; the summaries of regions made only of kept rows match exactly, and the next decode stays
 * close to the fresh run.
 */
public class CpuStep5ContextShiftTest {

    private static final int N_CTX = 40;
    /** Prompt length of the structural check: 24 kept positions close three regions (more than the top-k). */
    private static final int N_TOKENS = 32;
    private static final int N_KEEP = 4;
    private static final int N_DISCARD = 8;
    /** The step length and shift of the decode check: four rows move, so the re-rounding stays small. */
    private static final int SHORT_TOKENS = 16;
    private static final int SHORT_KEEP = 8;
    private static final int SHORT_DISCARD = 4;
    private static final float TOL = 5e-2f;
    private static final int PROXY = 256;
    private static final int REGIONS = N_CTX / 8;

    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static EngineConfig config(boolean contextShift) {
        return EngineConfig.builder().memory(MemoryConfig.builder()
                .nCtx(N_CTX).ubatch(8).contextShift(contextShift)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static int[] tokens(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = (i * 3 + i / 5 + 1) % 7;
        }
        return t;
    }

    /** The tokens a shift of {@code [nKeep, nKeep + nDiscard)} keeps, in order. */
    private static int[] kept(int[] prompt, int nKeep, int nDiscard) {
        int[] out = new int[prompt.length - nDiscard];
        System.arraycopy(prompt, 0, out, 0, nKeep);
        System.arraycopy(prompt, nKeep + nDiscard, out, nKeep, prompt.length - nKeep - nDiscard);
        return out;
    }

    @Test
    public void shiftedCacheRowsAndSummariesMatchAFreshRun() throws Exception {
        Path model = export("step5-shift-cache");
        int[] prompt = tokens(N_TOKENS);
        int[] kept = kept(prompt, N_KEEP, N_DISCARD);
        int n = kept.length;
        int width = 2 * 8;
        float[] shiftK = new float[n * width];
        float[] shiftV = new float[n * width];
        float[] shiftSummary;
        float[] freshK = new float[n * width];
        float[] freshV = new float[n * width];
        float[] freshSummary;
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            m.forwardBackend().prefill(prompt, 0);
            m.forwardBackend().shiftContext(N_KEEP, N_DISCARD);
            HeapModelMemory mem = (HeapModelMemory) m.forwardBackend().memory();
            assertEquals("the shift plan keeps the whole context in the ring", N_CTX * 2L * PROXY,
                    mem.recurrent(0).slotFloats(LayerMemoryPlan.STATE_STEP5_RING));
            readCache(mem, n, shiftK, shiftV);
            shiftSummary = summary(mem);
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            m.forwardBackend().prefill(kept, 0);
            HeapModelMemory mem = (HeapModelMemory) m.forwardBackend().memory();
            readCache(mem, n, freshK, freshV);
            freshSummary = summary(mem);
        }
        assertArrayEquals("the value rows move unchanged", freshV, shiftV, 0f);
        int keptFloats = N_KEEP * width;
        assertArrayEquals("the kept key rows stay as they were",
                Arrays.copyOf(freshK, keptFloats), Arrays.copyOf(shiftK, keptFloats), 0f);
        for (int i = keptFloats; i < freshK.length; i += 8) {
            float scale = 0f;
            for (int d = 0; d < 8; d++) {
                scale = Math.max(scale, Math.abs(freshK[i + d]));
            }
            for (int d = 0; d < 8; d++) {
                float tolerance = scale * 0.008f + 1e-6f;
                assertTrue("moved key " + (i + d) + " is within a BF16 step: fresh " + freshK[i + d] + " shifted "
                        + shiftK[i + d], Math.abs(freshK[i + d] - shiftK[i + d]) <= tolerance);
            }
        }
        assertEquals("the kept positions close " + (n / 8) + " regions", 3, n / 8);
        for (int i = 0; i < freshSummary.length; i++) {
            assertEquals("summary " + i + " is an E4M3 value", Step5Dtypes.e4m3(shiftSummary[i]), shiftSummary[i], 0f);
            assertTrue("summary " + i + " is within an E4M3 step: fresh " + freshSummary[i] + " shifted "
                            + shiftSummary[i],
                    Math.abs(freshSummary[i] - shiftSummary[i]) <= 0.02f);
        }
    }

    @Test
    public void shiftedSequenceDecodesTheKeptTokens() throws Exception {
        Path model = export("step5-shift");
        int[] prompt = tokens(SHORT_TOKENS);
        int[] kept = kept(prompt, SHORT_KEEP, SHORT_DISCARD);
        int next = 3;
        float[] shifted;
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            m.forwardBackend().prefill(prompt, 0);
            m.forwardBackend().shiftContext(SHORT_KEEP, SHORT_DISCARD);
            shifted = m.forwardBackend().decode(next, kept.length).clone();
        }
        float[] fresh;
        try (GGUFModel m = new GGUFModel(model.toString(), config(true))) {
            m.forwardBackend().prefill(kept, 0);
            fresh = m.forwardBackend().decode(next, kept.length).clone();
        }
        assertArrayEquals("the shifted sequence decodes the kept tokens", fresh, shifted, TOL);
    }

    @Test
    public void shiftWithoutTheContextShiftPlanRefusesByName() throws Exception {
        Path model = export("step5-shift-refused");
        try (GGUFModel m = new GGUFModel(model.toString(), config(false))) {
            m.forwardBackend().prefill(tokens(12), 0);
            HeapModelMemory mem = (HeapModelMemory) m.forwardBackend().memory();
            assertEquals("a memory without the shift plan keeps the open-region ring only",
                    (long) com.eonmux.neuralfree.gguf.spec.Step5Spec.ringCapacity(8, 7) * 2 * PROXY,
                    mem.recurrent(0).slotFloats(LayerMemoryPlan.STATE_STEP5_RING));
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.forwardBackend().shiftContext(2, 2));
            assertTrue(e.getMessage(), e.getMessage().contains("context.shift"));
        }
    }

    private static void readCache(HeapModelMemory mem, int n, float[] k, float[] v) {
        KvAccessor kv = mem.kv(0);
        int width = 2 * 8;
        float[] row = new float[width];
        for (int r = 0; r < n; r++) {
            for (int h = 0; h < kv.nHeadKv(); h++) {
                kv.readK(r, h, row, 0);
                System.arraycopy(row, 0, k, r * width + h * 8, 8);
                kv.readV(r, h, row, 0);
                System.arraycopy(row, 0, v, r * width + h * 8, 8);
            }
        }
    }

    private static float[] summary(HeapModelMemory mem) {
        RecurrentStateAccessor state = mem.recurrent(0);
        float[] all = state.tensor(LayerMemoryPlan.STATE_STEP5_SUMMARY);
        return Arrays.copyOf(all, REGIONS * PROXY);
    }

    /** Converts the synthetic Step5 checkpoint and exports it as a GGUF. */
    private Path export(String name) throws Exception {
        Path dir = CpuStep5ParityTest.builder().write(tmp.newFolder(name + "-hf").toPath());
        Path gguf = tmp.getRoot().toPath().resolve(name + ".gguf");
        try (HfModelSource src = HfModelSource.open(dir, ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder(name + "-cache").toPath()).build(), null)) {
            GgufWriter.write(src, gguf, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        return gguf;
    }
}
