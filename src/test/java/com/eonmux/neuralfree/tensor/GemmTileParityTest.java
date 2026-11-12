package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertNotNull;

/**
 * {@code TR_Gemm_F32} launched directly on each GPU API, against a double-precision reference within the GEMM
 * tolerance class: partial tiles and stages (the register-prefetched first and last stages and a partial last stage),
 * every operand layout the kernel distinguishes (unit stride along k or along m for A, along n or along k for B),
 * batches, both {@code beta} forms and an empty reduction.
 */
@RunWith(Parameterized.class)
public class GemmTileParityTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return TrainParity.apis();
    }

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend gpu;
    private final List<Tensor> owned = new ArrayList<>();
    private long seed = 11;

    public GemmTileParityTest(TrainParity.Api api) {
        this.api = api;
    }

    @Before
    public void setUp() {
        session = TrainParity.open(api);
        gpu = session.backend();
    }

    @After
    public void tearDown() {
        try {
            for (Tensor t : owned) {
                t.close();
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    /** Tails in every dimension: M and N past a tile edge, K past a stage, and a single partial stage. */
    @Test
    public void partialTilesAndStages() {
        check("70x100x203", 70, 100, 203, 1, false, false, 1.0f, 0.0f);
        check("33x65x48", 33, 65, 48, 1, false, false, 1.0f, 0.0f);
        check("1x31x7", 1, 31, 7, 1, false, false, 1.0f, 0.0f);
    }

    /** Products large enough that several groups stream the same operand slices over many stages. */
    @Test
    public void multiTileProducts() {
        check("128x192x96", 128, 192, 96, 1, false, false, 1.0f, 0.0f);
        check("64x96x1100", 64, 96, 1100, 1, false, false, 1.0f, 0.0f);
    }

    /** A read along m (column-major A) and B read along k (column-major B), alone and together. */
    @Test
    public void transposedOperands() {
        check("A^T", 45, 77, 61, 1, true, false, 1.0f, 0.0f);
        check("B^T", 45, 77, 61, 1, false, true, 1.0f, 0.0f);
        check("A^T B^T", 45, 77, 61, 1, true, true, 1.0f, 0.0f);
    }

    @Test
    public void batchedWithBeta() {
        check("batch 3, alpha 1.5, beta 0.5", 37, 41, 50, 3, false, false, 1.5f, 0.5f);
        check("batch 2 transposed, beta -1", 20, 70, 33, 2, true, true, 1.0f, -1.0f);
    }

    /** An empty reduction writes {@code alpha * 0 (+ beta * C)}. */
    @Test
    public void emptyReduction() {
        check("K = 0", 17, 19, 0, 1, false, false, 2.0f, 0.25f);
    }

    private void check(String what, int m, int n, int k, int batch, boolean aColMajor, boolean bColMajor,
                       float alpha, float beta) {
        TrainGpu g = TrainGpu.forBackend(gpu);
        assertNotNull(what + ": the train library is available on " + api, g);
        float[] a = TrainParity.uniform(seed++, Math.max(1, batch * m * k), -2.0f, 2.0f);
        float[] b = TrainParity.uniform(seed++, Math.max(1, batch * k * n), -2.0f, 2.0f);
        float[] c0 = TrainParity.uniform(seed++, batch * m * n, -1.0f, 1.0f);
        if (beta == 0.0f) {
            c0[0] = Float.NaN;
        }
        int aRs = aColMajor ? 1 : k;
        int aCs = aColMajor ? m : 1;
        int bRs = bColMajor ? 1 : n;
        int bCs = bColMajor ? k : 1;
        Tensor aDev = keep(Tensor.fromArray(gpu, a));
        Tensor bDev = keep(Tensor.fromArray(gpu, b));
        float[] got = run(g, m, n, k, batch, aDev, aRs, aCs, bDev, bRs, bCs, c0, alpha, beta);
        for (int bi = 0; bi < batch; bi++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double exact = 0.0;
                    double terms = 0.0;
                    for (int kk = 0; kk < k; kk++) {
                        double t = (double) a[bi * m * k + i * aRs + kk * aCs] * b[bi * k * n + kk * bRs + j * bCs];
                        exact += t;
                        terms += Math.abs(t);
                    }
                    int o = (bi * m + i) * n + j;
                    double ref = (double) alpha * exact + (beta != 0.0f ? (double) beta * c0[o] : 0.0);
                    double scaledTerms = Math.abs(alpha) * terms + (beta != 0.0f ? Math.abs(beta * c0[o]) : 0.0);
                    TrainParity.assertReduction(what + " on " + api + " output " + o, ref, got[o], Math.max(1, k),
                            scaledTerms);
                }
            }
        }
    }

    private float[] run(TrainGpu g, int m, int n, int k, int batch, Tensor a, int aRs, int aCs, Tensor b, int bRs,
                        int bCs, float[] c0, float alpha, float beta) {
        Tensor c = keep(Tensor.fromArray(gpu, c0));
        String kernel = Kernels.GEMM_F32;
        int tile = TrainOps.GEMM_TILE;
        long[] local = TrainOps.localSize(kernel);
        long[] global = {(long) ((n + tile - 1) / tile) * local[0], (long) ((m + tile - 1) / tile) * local[1], batch};
        g.exclusive(() -> {
            g.launch(kernel, global, local, m, n, k, batch,
                    0, aRs, aCs, m * k,
                    0, bRs, bCs, k * n,
                    0, n, m * n,
                    alpha, beta,
                    a.storage().gpuBuffer(), b.storage().gpuBuffer(), c.storage().gpuBuffer());
            return null;
        });
        return c.toFloatArray();
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }
}
