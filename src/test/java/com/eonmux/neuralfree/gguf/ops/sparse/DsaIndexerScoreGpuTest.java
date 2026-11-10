package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.testutil.LlmApiBackends;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * The lightning-indexer score kernel ({@code LLM_DSA_IndexerScore_F32}) masks exactly the rows the CPU reference
 * {@link CpuDsaIndexer#scores} masks, for every {@link SwaType} (CHUNKED_LOOKBACK included), causal and not, and
 * scores the visible rows alike, on every device of each API. The reference's visibility is itself checked against
 * {@link KvWindow}, the rule the attention kernels share.
 */
@RunWith(Parameterized.class)
public class DsaIndexerScoreGpuTest {

    private static final int N_TOKENS = 3;
    private static final int N_KV = 40;
    private static final int HEADS = 2;
    private static final int DIM = 8;
    private static final int N_SWA = 6;
    private static final int[] Q_POS = {9, 17, 31};

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> apis() {
        return Arrays.asList(new Object[]{BackendType.CUDA}, new Object[]{BackendType.OPENCL},
                new Object[]{BackendType.VULKAN});
    }

    private final BackendType api;
    private LlmApiBackends devices;

    public DsaIndexerScoreGpuTest(BackendType api) {
        this.api = api;
    }

    @Before
    public void open() {
        devices = LlmApiBackends.open(api);
    }

    @After
    public void close() {
        if (devices != null) {
            devices.close();
        }
    }

    @Test
    public void everyWindowTypeMasksAsTheReference() {
        for (SwaType type : SwaType.values()) {
            check(type, true);
            check(type, false);
        }
    }

    private void check(SwaType type, boolean causal) {
        Random rnd = new Random(31L * type.ordinal() + (causal ? 1 : 0));
        float[] iq = random(rnd, N_TOKENS * HEADS * DIM);
        float[] w = random(rnd, N_TOKENS * HEADS);
        float[] keys = random(rnd, N_KV * DIM);
        int[] kvPos = new int[N_KV];
        for (int s = 0; s < N_KV; s++) {
            kvPos[s] = s % 13 == 5 ? -1 : s;
        }
        int nSwa = type == SwaType.NONE ? 0 : N_SWA;
        float[] want = new float[N_TOKENS * N_KV];
        CpuDsaIndexer.scores(N_TOKENS, N_KV, HEADS, DIM, causal, type, nSwa, iq, 0, w, 0, keys, 0, Q_POS, 0, kvPos,
                0, want, 0);
        KvWindow window = KvWindow.of(type, nSwa, causal);
        for (int t = 0; t < N_TOKENS; t++) {
            for (int s = 0; s < N_KV; s++) {
                assertEquals(type + " causal=" + causal + " reference visibility t=" + t + " s=" + s,
                        window.visible(Q_POS[t], kvPos[s]), want[t * N_KV + s] != Float.NEGATIVE_INFINITY);
            }
        }
        String kernel = SparseKernels.indexerScore(KvDType.F32);
        for (ComputeBackend backend : devices.backends()) {
            String at = api + " " + backend.capabilities().getDeviceName() + " " + type + " causal=" + causal;
            LlmKernels kernels = LlmKernels.open(backend);
            Assume.assumeTrue("no " + kernel + " on " + at, kernels.has(kernel));
            GpuBuffer dScores = devices.upload(backend, new float[N_TOKENS * N_KV]);
            int local = Math.min(SparseKernels.INDEXER_LOCAL, kernels.rowReduceLocal());
            GpuEvent e = backend.computeQueue().enqueueKernel(kernels.get(kernel),
                    SparseKernels.indexerGlobal(N_TOKENS, N_KV, local), new long[]{local},
                    new Object[]{N_TOKENS, N_KV, HEADS, DIM, causal ? 1 : 0, type.code(), nSwa, DIM, 0,
                            devices.upload(backend, iq), devices.upload(backend, w), devices.upload(backend, keys),
                            devices.uploadInts(backend, Q_POS), devices.uploadInts(backend, kvPos), dScores});
            if (e != null) {
                e.close();
            }
            float[] got = LlmApiBackends.read(backend, dScores, N_TOKENS * N_KV);
            for (int i = 0; i < want.length; i++) {
                if (want[i] == Float.NEGATIVE_INFINITY) {
                    assertEquals(at + " masked score " + i, Float.NEGATIVE_INFINITY, got[i], 0f);
                } else {
                    assertEquals(at + " score " + i, want[i], got[i], 1e-5f * (1f + Math.abs(want[i])));
                }
            }
        }
    }

    private static float[] random(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2f - 1f;
        }
        return v;
    }
}
