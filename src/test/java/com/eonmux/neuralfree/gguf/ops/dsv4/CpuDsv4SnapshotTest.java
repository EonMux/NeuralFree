package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuCompressedKv;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link CpuDsv4Compressor#stateSnapshot} and the rollback it serves: plane {@code d} is the live plane after only the
 * first {@code nTokens - d} tokens of the step (llama.cpp {@code state_snapshot_src/dst_idxs},
 * llama-kv-cache-dsv4.cpp:684-713), and restoring it ({@link CpuCompressedKv#restorePlane}) leaves the compressor
 * exactly where a run that never stored the dropped positions would be (llama-kv-cache-dsv4.cpp:657-681, 1463-1510).
 */
public class CpuDsv4SnapshotTest {

    private static float[] random(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) rnd.nextGaussian();
        }
        return v;
    }

    private static CpuDsv4Compressor.Config config(int ratio, int width) {
        return CpuDsv4Compressor.Config.of(ratio, width, 1e-6f, Dsv4Kernels.RopeParams.none(), 0);
    }

    /** Plane {@code d} equals the live plane updated with the step's first {@code nTokens - d} tokens. */
    private static void assertSnapshotsArePrefixStates(int ratio, int width, int pos0, int nTokens, int snapshots) {
        CpuDsv4Compressor.Config cfg = config(ratio, width);
        Random rnd = new Random(31L * pos0 + nTokens);
        int sw = cfg.stateWidth();
        int plane = cfg.stateRows() * sw;
        float[] curKv = random(rnd, nTokens * sw + 3);
        float[] curScore = random(rnd, nTokens * sw + 5);
        float[] ape = random(rnd, ratio * sw);
        float[] liveKv = random(rnd, plane);
        float[] liveScore = random(rnd, plane);
        float[] kv = new float[(1 + snapshots) * plane];
        float[] score = new float[(1 + snapshots) * plane];
        System.arraycopy(liveKv, 0, kv, 0, plane);
        System.arraycopy(liveScore, 0, score, 0, plane);

        CpuDsv4Compressor.stateSnapshot(cfg, new CpuDsv4Compressor.Step(pos0, nTokens, curKv, 3, curScore, 5, ape),
                kv, score, snapshots);

        String what = "ratio " + ratio + ", pos0 " + pos0 + ", " + nTokens + " tokens";
        assertArrayEquals(what + ": the live kv plane is only read", liveKv, Arrays.copyOf(kv, plane), 0f);
        assertArrayEquals(what + ": the live score plane is only read", liveScore, Arrays.copyOf(score, plane), 0f);
        for (int d = 1; d <= snapshots; d++) {
            float[] wantKv = liveKv.clone();
            float[] wantScore = liveScore.clone();
            int prefix = nTokens - d;
            if (prefix > 0) {
                CpuDsv4Compressor.stateUpdate(cfg, new CpuDsv4Compressor.Step(pos0, prefix, curKv, 3, curScore, 5,
                        ape), wantKv, wantScore);
            }
            assertArrayEquals(what + ", kv plane " + d, wantKv,
                    Arrays.copyOfRange(kv, d * plane, (d + 1) * plane), 0f);
            assertArrayEquals(what + ", score plane " + d, wantScore,
                    Arrays.copyOfRange(score, d * plane, (d + 1) * plane), 0f);
        }
    }

    @Test
    public void csaSnapshotsAreTheStatesBeforeEachTrailingToken() {
        assertSnapshotsArePrefixStates(Dsv4Spec.RATIO_CSA, 4, 6, 5, 4);
        assertSnapshotsArePrefixStates(Dsv4Spec.RATIO_CSA, 4, 13, 1, 1);
        assertSnapshotsArePrefixStates(Dsv4Spec.RATIO_CSA, 4, 0, 3, 3);
    }

    /** A step longer than the state ring: a prefix token overwritten later in the prefix must not leak in. */
    @Test
    public void csaSnapshotsOfAStepLongerThanTheRingKeepTheLatestPrefixToken() {
        assertSnapshotsArePrefixStates(Dsv4Spec.RATIO_CSA, 4, 5, 19, 6);
    }

    @Test
    public void hcaSnapshotsAreTheStatesBeforeEachTrailingToken() {
        assertSnapshotsArePrefixStates(Dsv4Spec.RATIO_HCA, 4, 126, 4, 3);
        assertSnapshotsArePrefixStates(Dsv4Spec.RATIO_HCA, 4, 0, 2, 2);
    }

    @Test
    public void snapshotRejectsTooFewPlanes() {
        CpuDsv4Compressor.Config cfg = config(Dsv4Spec.RATIO_CSA, 4);
        int plane = cfg.stateRows() * cfg.stateWidth();
        float[] cur = new float[2 * cfg.stateWidth()];
        float[] ape = new float[Dsv4Spec.RATIO_CSA * cfg.stateWidth()];
        CpuDsv4Compressor.Step step = new CpuDsv4Compressor.Step(0, 2, cur, 0, cur, 0, ape);
        assertThrows(IllegalArgumentException.class,
                () -> CpuDsv4Compressor.stateSnapshot(cfg, step, new float[2 * plane], new float[2 * plane], 2));
        assertThrows(IllegalArgumentException.class,
                () -> CpuDsv4Compressor.stateSnapshot(cfg, step, new float[2 * plane], new float[2 * plane], 0));
    }

    /**
     * Rolling a CSA compressor back by {@code d} positions and continuing gives the compressed rows and the live
     * plane of a run that stored only the kept positions.
     */
    @Test
    public void restoringASnapshotMatchesARunWithoutTheDroppedPositions() {
        final int width = 8;
        final int snapshots = 4;
        final int p0 = 6;
        final int verify = 5;
        final int tail = 7;
        CpuDsv4Compressor.Config cfg = config(Dsv4Spec.RATIO_CSA, width);
        int sw = cfg.stateWidth();
        Random rnd = new Random(77);
        int total = p0 + verify + tail;
        float[] curKv = random(rnd, total * sw);
        float[] curScore = random(rnd, total * sw);
        float[] ape = random(rnd, Dsv4Spec.RATIO_CSA * sw);
        float[] normW = random(rnd, width);

        for (int d = 1; d <= snapshots; d++) {
            int kept = verify - d;
            CpuCompressedKv ref = CpuCompressedKv.allocate(0,
                    CompressedKvPlan.of(CompressedKvPlan.Kind.CSA, 64, width, 0, KvDType.F32));
            run(cfg, ref, 0, p0, curKv, curScore, ape, normW, 0);
            if (kept > 0) {
                run(cfg, ref, p0, kept, curKv, curScore, ape, normW, 0);
            }

            CpuCompressedKv got = CpuCompressedKv.allocate(0,
                    CompressedKvPlan.of(CompressedKvPlan.Kind.CSA, 64, width, 0, KvDType.F32, snapshots));
            run(cfg, got, 0, p0, curKv, curScore, ape, normW, snapshots);
            run(cfg, got, p0, verify, curKv, curScore, ape, normW, snapshots);
            got.restorePlane(d);

            int plane = cfg.stateRows() * sw;
            String what = "rollback of " + d;
            assertArrayEquals(what + ": live kv plane", ref.state().tensor(CpuMemory.COMPRESSOR_KV),
                    Arrays.copyOf(got.state().tensor(CpuMemory.COMPRESSOR_KV), plane), 0f);
            assertArrayEquals(what + ": live score plane", ref.state().tensor(CpuMemory.COMPRESSOR_SCORE),
                    Arrays.copyOf(got.state().tensor(CpuMemory.COMPRESSOR_SCORE), plane), 0f);

            int resume = p0 + kept;
            run(cfg, ref, resume, tail, curKv, curScore, ape, normW, 0);
            run(cfg, got, resume, tail, curKv, curScore, ape, normW, snapshots);
            float[] a = new float[width];
            float[] b = new float[width];
            for (int r = 0; r < (resume + tail) / Dsv4Spec.RATIO_CSA; r++) {
                ref.readK(r, 0, a, 0);
                got.readK(r, 0, b, 0);
                assertArrayEquals(what + ": compressed row " + r, a, b, 0f);
            }
        }
    }

    @Test
    public void restorePlaneRejectsPlanesOutsideTheSnapshots() {
        CpuCompressedKv c = CpuCompressedKv.allocate(0,
                CompressedKvPlan.of(CompressedKvPlan.Kind.HCA, 256, 8, 0, KvDType.F32, 2));
        assertThrows(IllegalArgumentException.class, () -> c.restorePlane(0));
        assertThrows(IllegalArgumentException.class, () -> c.restorePlane(3));
    }

    /** One step as the mixer runs it: compress, rollback snapshots, state update. */
    private static void run(CpuDsv4Compressor.Config cfg, CpuCompressedKv cache, int pos0, int n, float[] curKv,
                            float[] curScore, float[] ape, float[] normW, int snapshots) {
        int sw = cfg.stateWidth();
        float[] kv = cache.state().tensor(CpuMemory.COMPRESSOR_KV);
        float[] score = cache.state().tensor(CpuMemory.COMPRESSOR_SCORE);
        CpuDsv4Compressor.Step step = new CpuDsv4Compressor.Step(pos0, n, curKv, pos0 * sw, curScore, pos0 * sw, ape);
        if (Dsv4Kernels.blockCount(pos0, n, Dsv4Spec.RATIO_CSA) > 0) {
            CpuDsv4Compressor.compress(cfg, step, kv, score, normW, cache);
        }
        if (snapshots > 0) {
            CpuDsv4Compressor.stateSnapshot(cfg, step, kv, score, snapshots);
        }
        CpuDsv4Compressor.stateUpdate(cfg, step, kv, score);
    }
}
