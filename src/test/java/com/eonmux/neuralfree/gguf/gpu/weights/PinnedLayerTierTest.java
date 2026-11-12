package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Streamed layers staged from page-locked host copies ({@link PinnedLayerTier}): the slots receive the stored bytes
 * of every layer through device copies without a host-tier read after load; the host tier stages them when the tier
 * is off, every layer has its own slot, or the page-locked copies cannot be made; and the bound follows {@value PinnedLayerTier#PROPERTY} and the RAM the host has available.
 */
public class PinnedLayerTierTest {

    private static final long MIB = 1024L * 1024L;

    @After
    public void clearProperty() {
        System.clearProperty(PinnedLayerTier.PROPERTY);
    }

    @Test
    public void pinnedCopiesStageEveryLayerWithoutHostReads() throws IOException {
        Streamed run = stream(null, 1, FakeComputeBackend.create());
        assertTrue(run.pinned);
        assertEquals("no host-tier read after load", 0, run.readsWhileStreaming);
    }

    @Test
    public void theHostTierStagesWhenTheTierIsOff() throws IOException {
        Streamed run = stream("0", 1, FakeComputeBackend.create());
        assertFalse(run.pinned);
        assertTrue(run.readsWhileStreaming > 0);
    }

    @Test
    public void layersWithASlotEachAreNotPinned() throws IOException {
        assertFalse(stream(null, Integer.MAX_VALUE, FakeComputeBackend.create()).pinned);
    }

    @Test
    public void copiesThatCannotBeFilledLeaveTheLayersOnTheHostTier() throws IOException {
        FakeComputeBackend fake = FakeComputeBackend.create();
        List<HostMappedBuffer> made = new ArrayList<>();
        ComputeBackend unfillable = (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                new Class<?>[]{ComputeBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("allocateHostStagingBuffer")) {
                        HostMappedBuffer buffer = fake.allocateHostStagingBuffer((Long) args[0]);
                        made.add(buffer);
                        return unmappable(buffer);
                    }
                    try {
                        return method.invoke(fake, args);
                    } catch (InvocationTargetException e) {
                        throw e.getCause();
                    }
                });
        Streamed run = stream(null, 1, fake, unfillable);
        assertFalse("a failed fill falls back to the host tier", run.pinned);
        assertTrue(run.readsWhileStreaming > 0);
        assertFalse("the copy was attempted", made.isEmpty());
        for (HostMappedBuffer b : made) {
            assertTrue("nothing stays allocated", ((FakeGpuBuffer) b).isClosed());
        }
    }

    /** {@code buffer} whose host mapping fails, as a mapping beyond what one host view holds does. */
    private static HostMappedBuffer unmappable(HostMappedBuffer buffer) {
        return (HostMappedBuffer) Proxy.newProxyInstance(HostMappedBuffer.class.getClassLoader(),
                new Class<?>[]{HostMappedBuffer.class}, (proxy, method, args) -> {
                    if (method.getName().equals("hostView")) {
                        throw new IllegalStateException("the mapping exceeds one host view");
                    }
                    try {
                        return method.invoke(buffer, args);
                    } catch (InvocationTargetException e) {
                        throw e.getCause();
                    }
                });
    }

    @Test
    public void theBoundIsHalfTheRamTheModelLeavesLessWhatOtherTiersHold() {
        long ram = 64L * 1024 * MIB;
        long unknown = -1L;
        assertEquals((ram - 10 * 1024 * MIB) / 2,
                PinnedLayerTier.budgetBytes(PinnedLayerTier.AUTO, 10 * 1024 * MIB, ram, unknown, 0L));
        assertEquals((ram - 10 * 1024 * MIB) / 2 - 1024 * MIB,
                PinnedLayerTier.budgetBytes(PinnedLayerTier.AUTO, 10 * 1024 * MIB, ram, unknown, 1024 * MIB));
        assertEquals("a model as large as RAM pins nothing", 0L,
                PinnedLayerTier.budgetBytes(PinnedLayerTier.AUTO, ram, ram, ram, 0L));
        assertEquals("unknown RAM pins nothing", 0L,
                PinnedLayerTier.budgetBytes(PinnedLayerTier.AUTO, 1L, 0L, unknown, 0L));
        assertEquals(0L, PinnedLayerTier.budgetBytes(0L, 1L, ram, ram, 0L));
        assertEquals(300 * MIB, PinnedLayerTier.budgetBytes(512L, 1L, ram, unknown, 212 * MIB));
        assertEquals(0L, PinnedLayerTier.budgetBytes(512L, 1L, ram, unknown, 600 * MIB));
    }

    @Test
    public void theAutomaticBoundIsAtMostHalfTheAvailableRam() {
        long ram = 64L * 1024 * MIB;
        long model = 10 * 1024 * MIB;
        assertEquals("other processes hold most of the RAM", 4 * 1024 * MIB,
                PinnedLayerTier.budgetBytes(PinnedLayerTier.AUTO, model, ram, 8 * 1024 * MIB, 0L));
        assertEquals("plenty available: the RAM the model leaves bounds it", (ram - model) / 2,
                PinnedLayerTier.budgetBytes(PinnedLayerTier.AUTO, model, ram, ram, 0L));
        assertEquals("an explicit bound ignores the available RAM", 512 * MIB,
                PinnedLayerTier.budgetBytes(512L, model, ram, MIB, 0L));
    }

    @Test
    public void thePropertyTakesMebibytesOrMinusOne() {
        assertEquals(PinnedLayerTier.AUTO, PinnedLayerTier.configuredMb());
        System.setProperty(PinnedLayerTier.PROPERTY, " 2048 ");
        assertEquals(2048L, PinnedLayerTier.configuredMb());
        System.setProperty(PinnedLayerTier.PROPERTY, "-2");
        assertThrows(IllegalArgumentException.class, PinnedLayerTier::configuredMb);
        System.setProperty(PinnedLayerTier.PROPERTY, "lots");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, PinnedLayerTier::configuredMb);
        assertTrue(e.getMessage(), e.getMessage().contains(PinnedLayerTier.PROPERTY));
    }

    /** What a streamed run saw. */
    private static final class Streamed {
        final boolean pinned;
        final int readsWhileStreaming;

        Streamed(boolean pinned, int readsWhileStreaming) {
            this.pinned = pinned;
            this.readsWhileStreaming = readsWhileStreaming;
        }
    }

    /**
     * Streams every layer of the synthetic llama trunk through {@code depth} slots twice over, checking that each
     * layer's slot holds the stored bytes of its tensors once awaited.
     */
    private static Streamed stream(String pinnedMb, int depth, FakeComputeBackend fake) throws IOException {
        return stream(pinnedMb, depth, fake, fake);
    }

    /** {@link #stream(String, int, FakeComputeBackend)} through {@code backend}, a view of {@code fake}. */
    private static Streamed stream(String pinnedMb, int depth, FakeComputeBackend fake, ComputeBackend backend)
            throws IOException {
        if (pinnedMb != null) {
            System.setProperty(PinnedLayerTier.PROPERTY, pinnedMb);
        }
        Path file = SyntheticModels.llama().writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(file.toString());
             WeightStore reference = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser),
                    MemoryConfig.defaults());
            List<GGUFTensorInfo> tensors = parser.getTensors();
            List<LayerWeightSet> sets = LayerWeightSet.forTrunk(spec,
                    LayerWeightSet.Options.builder(ref -> ref.type()).build());
            assertTrue("the fixture streams several layers", sets.size() >= 2);
            int nWeightLayers = 0;
            for (LayerWeightSet set : sets) {
                nWeightLayers = Math.max(nWeightLayers, set.weightLayer() + 1);
            }
            Set<String> streamed = new HashSet<>(LayerWeightSet.allNames(sets));
            CountingStore counting = new CountingStore(parser.createMappedStore(StoreUse.STREAMING));
            GpuWeightUploader uploader = new GpuWeightUploader(backend, backend.computeQueue(), tensors,
                    parser.createMappedStore(StoreUse.RESIDENT_UPLOAD), "llama",
                    GpuWeightUploader.config((type, use) -> true).build());
            try {
                uploader.uploadResident(t -> streamed.contains(t.name()));
                WeightDataProvider data = new WeightDataProvider(counting, tensors,
                        LayerWeightSet.namesByLayer(sets, nWeightLayers), 1, 0L);
                StreamingLayerWeightSource source = StreamingLayerWeightSource.open(backend,
                        backend.computeQueue(), uploader, data, spec, sets, Math.min(depth, sets.size()));
                try {
                    int readsAtLoad = counting.reads;
                    for (int pass = 0; pass < 2; pass++) {
                        for (LayerWeightSet set : sets) {
                            source.await(set.weightLayer());
                            backend.computeQueue().finish();
                            for (LayerWeightSet.Entry e : set.entries()) {
                                assertSlotHoldsStoredBytes(source.weightsFor(set.weightLayer()).find(e.name()), e,
                                        tensors, reference);
                            }
                        }
                    }
                    return new Streamed(source.pinned(), counting.reads - readsAtLoad);
                } finally {
                    source.close();
                }
            } finally {
                uploader.close();
                counting.close();
            }
        } finally {
            fake.close();
            Files.deleteIfExists(file);
        }
    }

    private static void assertSlotHoldsStoredBytes(GpuWeight w, LayerWeightSet.Entry e, List<GGUFTensorInfo> tensors,
                                                   WeightStore reference) throws IOException {
        assertEquals(e.name(), 1, e.shardCount());
        GGUFTensorInfo info = tensors.stream().filter(t -> t.name().equals(e.name())).findFirst().get();
        int bytes = (int) e.shardBytes(0);
        byte[] expected = new byte[bytes];
        reference.readTensor(info.offset(), bytes, expected);
        byte[] device = Arrays.copyOf(((FakeGpuBuffer) w.shard(0)).toByteArray(), bytes);
        assertArrayEquals(e.name(), expected, device);
    }

    /** A store counting its direct reads, the reads of the host tier. */
    private static final class CountingStore implements WeightStore {
        private final WeightStore delegate;
        volatile int reads;

        CountingStore(WeightStore delegate) {
            this.delegate = delegate;
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
            reads++;
            delegate.readTensor(tensorOffset, size, dst);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
            reads++;
            delegate.readTensor(tensorOffset, size, dst, dstOffset);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
            reads++;
            delegate.readTensorDirect(tensorOffset, size, dst);
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }
    }
}
