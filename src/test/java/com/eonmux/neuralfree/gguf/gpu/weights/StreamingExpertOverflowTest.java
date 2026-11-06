package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;

/**
 * The MoE overflow placement at runtime: a load whose expert stacks stream but whose dense tensors stay resident
 * binds every layer — the streamed experts arrive through the layer's slot view ({@code LayerView}) and the dense
 * tensors fall back to the resident weights, so a block reads both through one {@link GpuWeights}.
 */
public class StreamingExpertOverflowTest {

    private static ModelSpec spec(GGUFFileParser parser) {
        return ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
    }

    /** The expert-stack sets of weight layers {@code [from, nLayer)} of the fixture. */
    private static List<LayerWeightSet> expertSets(ModelSpec spec, int from) {
        LayerWeightSet.Options options = LayerWeightSet.Options.builder(ref -> ref.type())
                .filter(ref -> WeightShards.isExpert(ref.name(), ref.rank()))
                .build();
        List<LayerWeightSet> out = new ArrayList<>();
        for (LayerSpec layer : spec.layers()) {
            if (layer.weightLayer() >= from) {
                LayerWeightSet set = LayerWeightSet.of(spec.tensors(), layer.stack(), layer.weightLayer(), options);
                if (!set.entries().isEmpty()) {
                    out.add(set);
                }
            }
        }
        return out;
    }

    /** Tensor data of zeros. */
    private static final class ZeroStore implements WeightStore {
        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) {
            Arrays.fill(dst, 0, size, (byte) 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
            Arrays.fill(dst, dstOffset, dstOffset + size, (byte) 0);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
            for (int i = 0; i < size; i++) {
                dst.put((byte) 0);
            }
        }

        @Override
        public void close() {
        }
    }

    @Test
    public void expertOverflowBindsDenseTensorsResidentAndExpertsFromSlots() throws IOException {
        Path file = SyntheticModels.qwen3next().writeTempFile();
        GGUFFileParser parser = null;
        FakeComputeBackend backend = FakeComputeBackend.create();
        try {
            parser = new GGUFFileParser(file.toString());
            ModelSpec spec = spec(parser);
            List<GGUFTensorInfo> tensors = parser.getTensors();
            List<LayerWeightSet> sets = expertSets(spec, 1);
            assertEquals(3, sets.size());

            GpuWeightUploader uploader = new GpuWeightUploader(backend, backend.computeQueue(), tensors,
                    new ZeroStore(), "qwen3next", GpuWeightUploader.config((type, use) -> true).build());
            try {
                Set<String> streamed = LayerWeightSet.allNames(sets);
                uploader.uploadResident(t -> streamed.contains(t.name()));
                assertNull(uploader.find("blk.1.ffn_up_exps.weight"));
                assertNotNull(uploader.find("blk.1.ffn_gate_inp.weight"));

                WeightDataProvider data = new WeightDataProvider(new ZeroStore(), tensors,
                        LayerWeightSet.namesByLayer(sets, 4), 2, 0L);
                StreamingLayerWeightSource source = StreamingLayerWeightSource.open(backend,
                        backend.computeQueue(), uploader, data, spec, sets, 2);
                try {
                    assertEquals(2, source.depth());
                    assertSame(uploader, source.weightsFor(0));

                    source.await(1);
                    GpuWeights layer1 = source.weightsFor(1);
                    assertNotNull(layer1.find("blk.1.ffn_up_exps.weight"));
                    assertNotNull(layer1.find("blk.1.ffn_gate_inp.weight"));
                    assertNotNull(layer1.find("blk.1.ffn_up_shexp.weight"));

                    source.await(2);
                    assertNotNull(source.weightsFor(2).find("blk.2.ffn_down_exps.weight"));
                    assertNotNull(source.weightsFor(2).find("blk.2.attn_norm.weight"));
                } finally {
                    source.close();
                }
            } finally {
                uploader.close();
            }
        } finally {
            try {
                if (parser != null) {
                    parser.close();
                }
            } finally {
                backend.close();
                Files.deleteIfExists(file);
            }
        }
    }
}
