package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuKvLayer;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;

public class HostKvAttentionTest {
    @Test
    public void prefillAndDecodeAttendToHostCache() {
        KvLayout layout = KvLayout.of(4, false, 1, 1, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        CpuKvLayer cache = CpuKvLayer.allocate(0, layout, 0, 1, 1, 1);
        int[] positions = {0, 1, 2, -1};
        HostKvAttention stage = new HostKvAttention(cache, row -> positions[row], 1, 1, 1, 1, 1, 1, null, null);
        CpuAttentionCore.Params params = CpuAttentionCore.Params.builder(1).build();
        float[] out = new float[2];
        stage.run(params, 0, 2, new int[]{0, 1}, new float[]{1, 1}, new float[]{1, 2},
                new float[]{10, 20}, out);
        float p = (float) (Math.exp(2) / (Math.exp(1) + Math.exp(2)));
        assertArrayEquals(new float[]{10, 10 + 10 * p}, out, 1e-5f);
        float[] decoded = new float[1];
        stage.run(params, 2, 1, new int[]{2}, new float[]{1}, new float[]{3}, new float[]{30}, decoded);
        float denom = (float) (Math.exp(1) + Math.exp(2) + Math.exp(3));
        assertArrayEquals(new float[]{(float) ((10 * Math.exp(1) + 20 * Math.exp(2) + 30 * Math.exp(3)) / denom)},
                decoded, 1e-5f);
    }

    @Test
    public void groupedQueriesUseTheSharedKvHead() {
        KvLayout layout = KvLayout.of(2, false, 1, 1, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        CpuKvLayer cache = CpuKvLayer.allocate(0, layout, 0, 1, 1, 1);
        int[] positions = {0, 1};
        HostKvAttention stage = new HostKvAttention(cache, row -> positions[row], 2, 1, 1, 2, 1, 1, null, null);
        float[] out = new float[4];
        stage.run(CpuAttentionCore.Params.builder(2).build(), 0, 2, positions, new float[4],
                new float[]{0, 0}, new float[]{10, 20}, out);
        assertArrayEquals(new float[]{10, 10, 15, 15}, out, 0f);
    }

    @Test
    public void ringCacheUsesStoredPositionsAndWindow() {
        KvWindow window = KvWindow.of(SwaType.STANDARD, 2, true);
        KvLayout layout = KvLayout.of(2, true, 1, 1, KvDType.F16, KvDType.F16, window);
        CpuKvLayer cache = CpuKvLayer.allocate(0, layout, 0, 1, 1, 1);
        int[] positions = {-1, -1};
        HostKvAttention stage = new HostKvAttention(cache, row -> positions[row], 1, 1, 1, 1, 1, 1, null, null);
        CpuAttentionCore.Params params = CpuAttentionCore.Params.builder(1).window(window).build();
        float[] out = new float[1];
        for (int pos = 0; pos < 4; pos++) {
            positions[layout.rowOf(pos)] = pos;
            stage.run(params, pos, 1, new int[]{pos}, new float[]{0}, new float[]{0},
                    new float[]{10 * (pos + 1)}, out);
            assertArrayEquals(new float[]{pos == 0 ? 10 : 10 * pos + 5}, out, 0f);
        }
        assertThrows(IllegalArgumentException.class,
                () -> stage.run(params, 3, 1, new int[]{3}, new float[]{0}, null, new float[]{1}, out));
    }

    @Test
    public void cacheScalesAndSharedCacheAreRespected() {
        KvLayout layout = KvLayout.of(2, false, 1, 1, KvDType.FP8_E4M3, KvDType.FP8_E5M2,
                KvWindow.FULL_CAUSAL);
        CpuKvLayer cache = CpuKvLayer.allocate(0, layout, 0, 1, 1, 1);
        int[] positions = {0, 1};
        HostKvAttention stage = new HostKvAttention(cache, row -> positions[row], 1, 1, 1, 1, 1, 1,
                new float[]{2}, new float[]{4});
        CpuAttentionCore.Params params = CpuAttentionCore.Params.builder(1)
                .kvScales(new float[]{2}, new float[]{4}).build();
        float[] out = new float[1];
        stage.run(params, 0, 1, new int[]{0}, new float[]{1}, new float[]{2}, new float[]{8}, out);
        assertArrayEquals(new float[]{8}, out, 0f);
        float[] stored = new float[1];
        cache.readK(0, 0, stored, 0);
        assertArrayEquals(new float[]{1}, stored, 0f);
        cache.readV(0, 0, stored, 0);
        assertArrayEquals(new float[]{2}, stored, 0f);
        stage.run(params, 0, 1, new int[]{0}, new float[]{1}, null, null, out);
        assertArrayEquals(new float[]{8}, out, 0f);
    }
}
