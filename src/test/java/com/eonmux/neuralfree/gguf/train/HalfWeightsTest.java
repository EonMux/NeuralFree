package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The half-precision path: an F16 or BF16 model decodes into float32 master weights, the run trains and steps those
 * masters, and the export writes them back in the tensor's own stored type rather than widening the file to F32.
 */
public class HalfWeightsTest {

    private static final int N_CTX = 16;
    private static final int[] TOKENS = {1, 7, 19, 42, 77, 100, 128, 201, 3, 55, 240, 17, 88, 129, 260, 301};
    private static final int[] LABELS = {7, 19, 42, 77, 100, 128, 201, 3, 55, 240, 17, 88, 129, 260, 301, 44};
    private static final String WEIGHT = "blk.0.attn_q.weight";
    private static final String NORM = "blk.0.ffn_norm.weight";

    @Rule
    public final TemporaryFolder folder = new TemporaryFolder();

    private static Path f16File;
    private static Path bf16File;

    @BeforeClass
    public static void write() throws IOException {
        f16File = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(GGMLType.F16).build()).writeTempFile();
        bf16File = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(GGMLType.BF16).build()).writeTempFile();
    }

    @AfterClass
    public static void close() throws IOException {
        if (f16File != null) {
            Files.deleteIfExists(f16File);
        }
        if (bf16File != null) {
            Files.deleteIfExists(bf16File);
        }
    }

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(N_CTX)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    @Test
    public void theStoredTypesSurviveTrainingAndTheWriteBack() throws IOException {
        checkStoredType(f16File, GGMLType.F16);
        checkStoredType(bf16File, GGMLType.BF16);
    }

    private void checkStoredType(Path file, GGMLType type) throws IOException {
        Path out = folder.getRoot().toPath().resolve(type + ".gguf");
        Map<String, float[]> masters = new LinkedHashMap<>();
        try (GGUFModel model = new GGUFModel(file.toString(), config());
             TrainableModel train = TrainableModel.open(model, N_CTX)) {
            for (TrainableModel.Parameter p : train.parameters()) {
                // A projection is stored in the model's weight type; a norm stays F32 in the fixture.
                if (p.name().equals(WEIGHT)) {
                    assertEquals(type + " " + p.name(), type, p.storedType());
                }
            }
            assertTrue("the fixture stored no " + type + " weight",
                    train.parameter(WEIGHT).value()[0] != 0f);
            train.zeroGrad();
            double loss = train.step(TOKENS, LABELS).loss();
            assertTrue(type + " loss is not finite: " + loss, Double.isFinite(loss));
            boolean grad = false;
            for (TrainableModel.Parameter p : train.parameters()) {
                for (float g : p.grad()) {
                    assertTrue(type + " gradient is not finite", Float.isFinite(g));
                    grad |= g != 0f;
                }
            }
            assertTrue(type + " produced no gradient", grad);
            for (TrainableModel.Parameter p : train.parameters()) {
                masters.put(p.name(), p.value().clone());
            }
        }
        ModelSource base = ModelSources.open(file.toString());
        try (TrainedSource source = TrainedSource.of(base, masters)) {
            GgufWriter.write(source, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (ModelSource written = ModelSources.open(out.toString())) {
            GGUFTensorInfo weight = tensor(written, WEIGHT);
            assertEquals(type + " was widened by the write-back", type, weight.type());
            assertEquals("a F32 norm was left alone", GGMLType.F32, tensor(written, NORM).type());
            float[] stored = storedValues(written, weight);
            float[] expected = masters.get(WEIGHT);
            for (int i = 0; i < stored.length; i++) {
                assertEquals(type + " value " + i, round(type, expected[i]), stored[i], 0f);
            }
        }
    }

    /** The value a float32 master is stored as in {@code type}. */
    private static float round(GGMLType type, float value) {
        return type == GGMLType.F16
                ? HalfFloats.halfToFloat(HalfFloats.floatBitsToHalf(Float.floatToRawIntBits(value)))
                : HalfFloats.bfloat16ToFloat(HalfFloats.floatBitsToBFloat16(Float.floatToRawIntBits(value)));
    }

    private static GGUFTensorInfo tensor(ModelSource source, String name) {
        for (GGUFTensorInfo info : source.getTensors()) {
            if (info.name().equals(name)) {
                return info;
            }
        }
        throw new AssertionError("no tensor " + name);
    }

    /** The decoded stored values of one F16/BF16 tensor. */
    private static float[] storedValues(ModelSource source, GGUFTensorInfo info) throws IOException {
        byte[] bytes = new byte[(int) info.sizeInBytes()];
        try (WeightStore store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            store.readTensor(info.offset(), bytes.length, bytes, 0);
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[(int) info.numElements()];
        for (int i = 0; i < values.length; i++) {
            int bits = buffer.getShort() & 0xffff;
            values[i] = info.type() == GGMLType.F16 ? HalfFloats.halfToFloat(bits)
                    : HalfFloats.bfloat16ToFloat(bits);
        }
        return values;
    }

    @Test
    public void theHalfModelForwardsLikeTheEngineAndTrainsTheMasters() throws IOException {
        for (Path file : new Path[]{f16File, bf16File}) {
            try (GGUFModel model = new GGUFModel(file.toString(), config());
                 TrainableModel train = TrainableModel.open(model, N_CTX)) {
                model.forwardBackend().reset();
                float[][] engine = model.forwardBackend().forwardRows(TOKENS, 0, null);
                float[][] rows = train.logits(TOKENS);
                double worst = 0.0;
                for (int t = 0; t < rows.length; t++) {
                    for (int v = 0; v < rows[t].length; v++) {
                        worst = Math.max(worst, Math.abs(rows[t][v] - engine[t][v]));
                    }
                }
                assertTrue("largest logit difference to the engine: " + worst, worst < 2e-3);
                train.zeroGrad();
                train.step(TOKENS, LABELS);
                TrainableModel.Parameter weight = train.parameter(WEIGHT);
                assertTrue("the masters did not move", weight.grad()[0] != 0f);
            }
        }
    }
}
