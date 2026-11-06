package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HostMappedWeightSource} over a synthetic llama on the host-only fake backend: the selected tensors are
 * copied whole into host-mapped buffers in their upload type, the unselected tensors fall back to the delegate,
 * prefetch and await reach the delegate, the F32 conversion of a stored F16 tensor decodes the same values as the
 * resident upload, and closing frees the buffers.
 */
public class HostMappedWeightSourceTest {

    /** A fifo fixture with the file, parser, uploader and spec one test needs. */
    private static final class Fixture implements AutoCloseable {
        final GgufTestWriter writer;
        final Path file;
        final GGUFFileParser parser;
        final FakeComputeBackend backend;
        final WeightStore store;
        final ModelSpec spec;
        final GpuWeightUploader uploader;
        final LayerWeightSet.Options options;
        WeightDataProvider data;

        Fixture(GgufTestWriter writer, GpuWeightUploader.TypePolicy policy) throws IOException {
            this.writer = writer;
            this.file = writer.writeTempFile();
            this.parser = new GGUFFileParser(file.toString());
            this.backend = FakeComputeBackend.create();
            this.store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
            this.spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser),
                    MemoryConfig.defaults());
            this.uploader = new GpuWeightUploader(backend, backend.computeQueue(), parser.getTensors(), store,
                    "llama", GpuWeightUploader.config(policy).build());
            this.options = LayerWeightSet.Options.builder(t -> uploader.deviceType(t.name()))
                    .maxShardBytes(LayerWeightSet.Options.maxShardBytes(backend.capabilities()))
                    .build();
        }

        WeightDataProvider hostTier() {
            if (data == null) {
                data = new WeightDataProvider(store, parser.getTensors(), Collections.<List<String>>emptyList(),
                        0, 0L);
            }
            return data;
        }

        @Override
        public void close() throws IOException {
            try {
                if (data != null) {
                    data.close();
                }
                uploader.close();
                store.close();
                parser.close();
                backend.close();
            } finally {
                Files.deleteIfExists(file);
            }
        }
    }

    private static Fixture llama() throws IOException {
        return new Fixture(SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2)
                .nEmbd(256).nFf(256).build()), (type, use) -> true);
    }

    private static HostMappedPlan plan(Fixture f, String... params) {
        EngineConfig cfg = EngineConfig.builder().cpuOffloadMb(4096).cpuOffloadParams(Arrays.asList(params)).build();
        return HostMappedPlan.plan(cfg, f.spec, f.options, name -> true);
    }

    @Test
    public void selectedTensorsHoldTheFileBytesInHostMappedBuffers() throws IOException {
        try (Fixture f = llama()) {
            HostMappedPlan plan = plan(f, "ffn_gate.weight", "attn_norm.weight");
            f.uploader.uploadResident(t -> plan.contains(t.name()));
            RecordingSource delegate = new RecordingSource(LayerWeightSource.resident(f.uploader));
            try (HostMappedWeightSource source = HostMappedWeightSource.open(f.backend, f.backend.computeQueue(),
                    f.hostTier(), "llama", plan, delegate)) {
                assertEquals(plan.tensorCount(), source.tensorCount());
                assertEquals(2, source.layerCount());
                assertEquals(plan.bytes(), source.hostBytes());

                List<FakeGpuBuffer> shards = new ArrayList<>();
                for (LayerWeightSet set : plan.sets()) {
                    for (LayerWeightSet.Entry e : set.entries()) {
                        GpuWeight w = source.weightsFor(set.weightLayer()).find(e.name());
                        assertNotNull(name(e), w);
                        assertEquals(e.uploadType(), w.type());
                        assertEquals(e.cols(), w.cols());
                        assertEquals(e.rows(), w.rows());
                        FakeGpuBuffer buffer = (FakeGpuBuffer) w.shard(0);
                        shards.add(buffer);
                        assertArrayEquals(e.name(), f.writer.tensorData(e.name()), buffer.toByteArray());
                    }
                }

                GpuWeight resident = f.uploader.find("blk.0.attn_q.weight");
                assertNotNull(resident);
                assertSame(resident, source.weightsFor(0).find("blk.0.attn_q.weight"));
                assertFalse(plan.contains("blk.0.attn_q.weight"));

                source.prefetch(1);
                source.await(0);
                assertEquals(Arrays.asList("prefetch:1", "await:0"), delegate.calls);

                for (FakeGpuBuffer buffer : shards) {
                    assertFalse(buffer.isClosed());
                }
                source.close();
                for (FakeGpuBuffer buffer : shards) {
                    assertTrue(buffer.isClosed());
                }
            }
        }
    }

    @Test
    public void aDequantizedTensorHoldsTheDecodedFloats() throws IOException {
        try (Fixture f = new Fixture(SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2)
                .nEmbd(256).nFf(256).weightType(GGMLType.F16).build()), (type, use) -> false)) {
            HostMappedPlan plan = plan(f, "ffn_gate.weight");
            f.uploader.uploadResident(t -> plan.contains(t.name()));
            RecordingSource delegate = new RecordingSource(LayerWeightSource.resident(f.uploader));
            try (HostMappedWeightSource source = HostMappedWeightSource.open(f.backend, f.backend.computeQueue(),
                    f.hostTier(), "llama", plan, delegate)) {
                LayerWeightSet.Entry entry = plan.sets().get(0).entries().get(0);
                assertEquals(GGMLType.F16, entry.source().type());
                assertTrue(entry.dequantize());
                GpuWeight w = source.weightsFor(plan.sets().get(0).weightLayer()).find(entry.name());
                assertNotNull(w);
                assertEquals(GGMLType.F32, w.type());

                float[] expected = decode(f, entry.source());
                byte[] bytes = ((FakeGpuBuffer) w.shard(0)).toByteArray();
                float[] actual = new float[expected.length];
                ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(actual);
                assertArrayEquals(expected, actual, 0.0f);
            }
        }
    }

    @Test
    public void anEmptyPlanIsRefused() throws IOException {
        try (Fixture f = llama()) {
            RecordingSource delegate = new RecordingSource(LayerWeightSource.resident(f.uploader));
            assertThrows(IllegalArgumentException.class, () -> HostMappedWeightSource.open(f.backend,
                    f.backend.computeQueue(), f.hostTier(), "llama", HostMappedPlan.empty(), delegate));
        }
    }

    private static String name(LayerWeightSet.Entry e) {
        return e.name();
    }

    /** The F32 values the host dequantization of {@code ref} produces, decided independently of the source. */
    private static float[] decode(Fixture f, TensorRef ref) {
        byte[] stored = f.writer.tensorData(ref.name());
        int elements = (int) ref.numElements();
        float[] out = new float[elements];
        CpuBlockDecoders.decodeElements(ref.type(), ByteBuffer.wrap(stored).order(ByteOrder.LITTLE_ENDIAN), 0, out,
                0, elements);
        return out;
    }

    /** A delegate recording the prefetch and await calls it receives. */
    private static final class RecordingSource implements LayerWeightSource {
        private final LayerWeightSource inner;
        final List<String> calls = new ArrayList<>();

        RecordingSource(LayerWeightSource inner) {
            this.inner = inner;
        }

        @Override
        public void prefetch(int il) {
            calls.add("prefetch:" + il);
            inner.prefetch(il);
        }

        @Override
        public void await(int il) {
            calls.add("await:" + il);
            inner.await(il);
        }

        @Override
        public GpuWeights weightsFor(int il) {
            return inner.weightsFor(il);
        }

        @Override
        public void close() {
            calls.add("close");
            inner.close();
        }
    }
}
