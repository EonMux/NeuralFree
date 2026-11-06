package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The host RAM cache sizing ({@link WeightDataProvider#ramCacheBytes}): an explicit
 * {@code neuralfree.ram.cache.mb} stands beside any host, a model larger than the physical RAM runs with the
 * automatic cache off (Avalon tools/server/server-context.cpp:1550-1566), and otherwise the automatic cache is
 * half the heap capped at half of the RAM the model leaves for its page cache.
 */
public class WeightDataProviderRamCacheTest {

    private static final long MIB = 1024L * 1024L;
    private static final long GIB = 1024L * 1024L * 1024L;

    private static EngineConfig auto() {
        return EngineConfig.builder().build();
    }

    private static EngineConfig explicitMb(long mb) {
        return EngineConfig.builder().ramCacheMb(mb).build();
    }

    private static long maxHeap() {
        return Runtime.getRuntime().maxMemory();
    }

    @Test
    public void anExplicitSizeStandsBesideAnyHost() {
        assertEquals(64 * MIB, WeightDataProvider.ramCacheBytes(explicitMb(64), 100 * GIB, 8 * GIB));
        assertEquals(64 * MIB, WeightDataProvider.ramCacheBytes(explicitMb(64), 0, 32 * MIB));
        assertEquals(0, WeightDataProvider.ramCacheBytes(explicitMb(0), 0, 8 * GIB));
        assertEquals(Long.MAX_VALUE, WeightDataProvider.ramCacheBytes(explicitMb(Long.MAX_VALUE), 0, -1));
    }

    @Test
    public void aModelLargerThanRamDisablesTheAutomaticCache() {
        assertEquals(0, WeightDataProvider.ramCacheBytes(auto(), 100 * GIB, 32 * GIB));
        assertEquals("a model exactly as large as RAM keeps no heap cache either",
                0, WeightDataProvider.ramCacheBytes(auto(), 32 * GIB, 32 * GIB));
    }

    @Test
    public void theAutomaticCacheCapsAtHalfTheRamBesideTheModel() {
        long model = 8 * GIB;
        long ram = 16 * GIB;
        long expected = Math.min(maxHeap() / 2, (ram - model) / 2);
        assertEquals(expected, WeightDataProvider.ramCacheBytes(auto(), model, ram));
        assertEquals("a tiny model keeps the heap-only figure",
                Math.min(maxHeap() / 2, ram / 2), WeightDataProvider.ramCacheBytes(auto(), 1 * MIB, ram));
    }

    @Test
    public void anUnknownRamFallsBackToTheHeapOnlyFigure() {
        long expected = Math.min(maxHeap() / 2, (32 * GIB - 1 * GIB) / 2);
        assertEquals(maxHeap() / 2, WeightDataProvider.ramCacheBytes(auto(), 1 * GIB, -1));
        assertEquals(expected, WeightDataProvider.ramCacheBytes(auto(), 1 * GIB, 0));
    }

    @Test
    public void theOverloadsDelegateToTheThreeArgForm() {
        long ram = HostMemory.totalBytes();
        assertEquals(WeightDataProvider.ramCacheBytes(auto(), 0, ram), WeightDataProvider.ramCacheBytes(auto()));
        assertEquals(WeightDataProvider.ramCacheBytes(auto(), 16 * MIB, ram),
                WeightDataProvider.ramCacheBytes(auto(), 16 * MIB));
        assertThrows(IllegalArgumentException.class,
                () -> WeightDataProvider.ramCacheBytes(auto(), -1, ram));
    }

    /** The provider builds its cache bound from the configuration and the model's tensor bytes. */
    @Test
    public void theProviderSizesItsCacheFromTheModelBytes() throws IOException {
        List<GGUFTensorInfo> tensors = Arrays.asList(
                new GGUFTensorInfo("token_embd.weight", new long[]{4}, GGMLType.F32, 0L),
                new GGUFTensorInfo("blk.0.attn_q.weight", new long[]{4}, GGMLType.F32, 16L));
        long modelBytes = 32;
        assertTrue("the test host is larger than the model", HostMemory.totalBytes() > modelBytes);

        long expected = WeightDataProvider.ramCacheBytes(auto(), modelBytes, HostMemory.totalBytes());
        assertTrue(expected > 0);
        try (WeightDataProvider data = new WeightDataProvider(new EmptyStore(), tensors,
                Collections.singletonList(Collections.singletonList("blk.0.attn_q.weight")), auto())) {
            assertEquals(expected, data.maxCacheBytes());
        }
        try (WeightDataProvider data = new WeightDataProvider(new EmptyStore(), tensors,
                Collections.singletonList(Collections.singletonList("blk.0.attn_q.weight")), explicitMb(7))) {
            assertEquals(7 * MIB, data.maxCacheBytes());
        }
    }

    /** Tensor data of nothing: the sizing tests never read through the store. */
    private static final class EmptyStore implements WeightStore {
        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) {
            throw new UnsupportedOperationException("no reads expected");
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
            throw new UnsupportedOperationException("no reads expected");
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
            throw new UnsupportedOperationException("no reads expected");
        }

        @Override
        public void close() {
        }
    }
}
