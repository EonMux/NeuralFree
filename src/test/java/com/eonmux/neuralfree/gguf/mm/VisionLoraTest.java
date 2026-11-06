package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapterLoader;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.heads.LoraFixtures;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * LoRA adapters on the vision encoder and its connector (vLLM {@code enable_tower_connector_lora}): the loader
 * classifies a pair against the text model first and the mmproj tensor table second, the text and vision
 * attachments each serve only their side's pairs, and a CPU/GPU encoder with a tower and connector adapter produces
 * the rows of the same encoder whose weights carry {@code W + scale * B A} (llama.cpp {@code build_lora_mm}).
 */
public class VisionLoraTest {

    private static final float TOL = 3e-4f;
    private static final float GPU_TOL = 2e-3f;
    private static final String TEXT_TARGET = "blk.0.attn_q.weight";
    private static final String Q_TARGET = "v.blk.0.attn_q.weight";
    private static final String CONNECTOR_TARGET = "mm.0.weight";

    @ClassRule
    public static final TemporaryFolder TMP = new TemporaryFolder();

    private static HWAcceleration acceleration;
    private static ComputeBackend backend;

    @BeforeClass
    public static void openDevice() {
        if (!GpuTestUtil.isOpenCLAvailable()) {
            return;
        }
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try {
            backend = acceleration.createBackend(Config.ACCELERATION_MODE.OPENCL, LlmKernels.requirement());
        } catch (RuntimeException e) {
            backend = null;
        }
    }

    @AfterClass
    public static void closeDevice() {
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static void assumeGpu() {
        Assume.assumeTrue("no OpenCL device running the LLM kernels", backend != null);
    }

    // ================================================================== loader classification

    /** An mmproj tensor table with one adapted tower matrix and one adapted connector matrix. */
    private static TensorCatalog visionCatalog() {
        MmprojFixture f = new MmprojFixture(7);
        f.weight(Q_TARGET, 8, 8);
        f.weight(CONNECTOR_TARGET, 8, 6);
        return f.catalog();
    }

    private static long[] textShape() {
        return SyntheticModels.llama().tensorShape(TEXT_TARGET);
    }

    /** A GGUF adapter with the factors of {@code targets} at {@code rank} and {@code alpha}. */
    private static Path adapterFile(String arch, float alpha, int rank, Map<String, long[]> targets)
            throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture(arch);
        w.str("general.type", "adapter").str("adapter.type", "lora").f32("adapter.lora.alpha", alpha);
        long seed = 1;
        for (Map.Entry<String, long[]> e : targets.entrySet()) {
            int in = Math.toIntExact(e.getValue()[0]);
            int out = Math.toIntExact(e.getValue()[1]);
            w.tensorF32(e.getKey() + ".lora_a", new long[]{in, rank}, LoraFixtures.random(seed++, rank * in));
            w.tensorF32(e.getKey() + ".lora_b", new long[]{rank, out}, LoraFixtures.random(seed++, out * rank));
        }
        Path file = TMP.newFolder().toPath().resolve("adapter.gguf");
        w.writeTo(file);
        return file;
    }

    private static Map<String, long[]> targets(Object... nameShapePairs) {
        Map<String, long[]> out = new LinkedHashMap<>();
        for (int i = 0; i < nameShapePairs.length; i += 2) {
            out.put((String) nameShapePairs[i], (long[]) nameShapePairs[i + 1]);
        }
        return out;
    }

    private static ModelSourceOptions withVisionCatalog() {
        return ModelSourceOptions.defaults().toBuilder().loraVisionCatalog(visionCatalog()).build();
    }

    private static LoraAdapter load(Path adapter, ModelSourceOptions options) throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (ModelSource source = ModelSources.open(model.toString(), options);
             CpuWeights base = CpuWeights.open(source, TensorCatalog.of(source), null)) {
            return LoraAdapterLoader.load(adapter.toString(), base, options);
        }
    }

    @Test
    public void aMixedAdapterResolvesTextPairsFirstAndVisionPairsAgainstTheMmprojTable() throws IOException {
        Path file = adapterFile("llama", 8.0f, 2, targets(
                TEXT_TARGET, textShape(),
                Q_TARGET, new long[]{8, 8},
                CONNECTOR_TARGET, new long[]{8, 6}));
        LoraAdapter adapter = load(file, withVisionCatalog());

        assertEquals(LoraAdapter.Side.TEXT, adapter.pair(TEXT_TARGET).side());
        assertEquals(LoraAdapter.Side.VISION, adapter.pair(Q_TARGET).side());
        assertEquals(LoraAdapter.Side.VISION, adapter.pair(CONNECTOR_TARGET).side());
        assertEquals(8, adapter.pair(Q_TARGET).inDim());
        assertEquals(8, adapter.pair(Q_TARGET).outDim());
        assertEquals(6, adapter.pair(CONNECTOR_TARGET).outDim());
        assertEquals(2, adapter.pair(Q_TARGET).rank());
        // scale = userScale * alpha / rank (llama-adapter.h:53-57)
        assertEquals(4.0f, adapter.scale(Q_TARGET, 1.0f), 0.0f);
    }

    @Test
    public void aTowerOnlyAdapterMayCarryTheMmprojArch() throws IOException {
        Path file = adapterFile("clip", 8.0f, 2, targets(
                Q_TARGET, new long[]{8, 8},
                CONNECTOR_TARGET, new long[]{8, 6}));
        LoraAdapter adapter = load(file, withVisionCatalog());

        assertEquals(2, adapter.targets().size());
        assertEquals(LoraAdapter.Side.VISION, adapter.pair(Q_TARGET).side());
        assertEquals(LoraAdapter.Side.VISION, adapter.pair(CONNECTOR_TARGET).side());
    }

    @Test
    public void aMixedAdapterMustCarryTheTextArch() throws IOException {
        Path file = adapterFile("clip", 8.0f, 2, targets(
                TEXT_TARGET, textShape(),
                Q_TARGET, new long[]{8, 8}));
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> load(file, withVisionCatalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("arch clip"));
    }

    @Test
    public void aVisionTargetNeedsTheMmprojTable() throws IOException {
        Path file = adapterFile("llama", 8.0f, 2, targets(Q_TARGET, new long[]{8, 8}));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> load(file, ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains(Q_TARGET));
    }

    @Test
    public void anUnknownVisionTargetIsRefusedByName() throws IOException {
        Path file = adapterFile("llama", 8.0f, 2, targets("v.blk.9.attn_q.weight", new long[]{8, 8}));
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> load(file, withVisionCatalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("v.blk.9.attn_q.weight"));
    }

    // ================================================================== side filtering

    @Test
    public void theVisionAttachmentServesOnlyTheVisionPairs() {
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(70, false, false);
        ClipHparams hp = llavaHparams();
        CpuVisionEncoder enc = bind(f, hp);
        LoraAdapter adapter = LoraFixtures.adapter("vision.gguf", 8.0f,
                LoraFixtures.projection(TEXT_TARGET, 8, 8, 2, 701),
                LoraFixtures.visionProjection(Q_TARGET, 8, 8, 2, 711),
                LoraFixtures.visionProjection(CONNECTOR_TARGET, 8, 6, 2, 721));

        LoraAttachment att = enc.attachLoras(Collections.singletonList(new LoraAttachment.Binding(adapter, 1.0f)));

        assertNotNull(att);
        assertEquals(2, att.deltaCount());
        assertEquals(3, adapter.targets().size());
    }

    @Test
    public void aTextOnlyAdapterAttachesNothingToTheVisionEncoder() {
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(71, false, false);
        CpuVisionEncoder enc = bind(f, llavaHparams());
        LoraAdapter adapter = LoraFixtures.adapter("text.gguf", 8.0f,
                LoraFixtures.projection(TEXT_TARGET, 8, 8, 2, 731));

        assertNull(enc.attachLoras(Collections.singletonList(new LoraAttachment.Binding(adapter, 1.0f))));
        assertTrue(enc.projectionDeltas().served(Q_TARGET).isEmpty());
    }

    // ================================================================== parity

    private static ClipHparams llavaHparams() {
        return CpuVisionEncoderTest.hparams(
                CpuVisionEncoderTest.clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null, null));
    }

    private static CpuVisionEncoder bind(MmprojFixture f, ClipHparams hp) {
        return CpuVisionEncoder.bind(ClipVisionModel.from(hp, f.catalog()), CpuMatVec.PLAIN, f::get, f::ref);
    }

    /** The adapter's tower and connector LoRA pair on the llava fixture of {@link #bind}. */
    private static LoraAdapter llavaAdapter() {
        return LoraFixtures.adapter("vision.gguf", 8.0f,
                LoraFixtures.visionProjection(Q_TARGET, 8, 8, 2, 741),
                LoraFixtures.visionProjection(CONNECTOR_TARGET, 8, 6, 2, 751));
    }

    @Test
    public void aCpuVisionEncoderWithATowerAndConnectorAdapterMatchesTheMergedWeights() {
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(72, false, false);
        ClipHparams hp = llavaHparams();
        float[] pixels = MmprojFixture.pixels(8, 8, 720);
        ClipImage image = ClipImage.of(8, 8, pixels);
        CpuVisionEncoder enc = bind(f, hp);
        float[][] base = enc.encode(image);

        LoraAdapter adapter = llavaAdapter();
        CpuVisionEncoder adapted = bind(f, hp);
        assertNotNull(adapted.attachLoras(
                Collections.singletonList(new LoraAttachment.Binding(adapter, 1.0f))));
        float[][] got = adapted.encode(image);

        // the adapter must change the rows: W x + s B (A x) differs from W x
        assertTrue(maxAbsDiff(base, got) > 1e-3f);

        // the same encoder over the weights merged with s * B A (build_lora_mm, llama-graph.cpp:1648-1664)
        merge(f, adapter, Q_TARGET, 1.0f);
        merge(f, adapter, CONNECTOR_TARGET, 1.0f);
        float[][] want = MmprojFixture.Reference.clip(f, hp, pixels, 8, 8);
        assertRows(want, got);
    }

    @Test
    public void aGpuVisionEncoderWithATowerAndConnectorAdapterMatchesTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(73, false, false);
        Map<String, Object> meta = CpuVisionEncoderTest.clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null,
                null, null);
        ClipHparams hp = CpuVisionEncoderTest.hparams(meta);
        Path file = write(f, meta);
        float[] pixels = MmprojFixture.pixels(8, 8, 730);
        ClipImage image = ClipImage.of(8, 8, pixels);
        LoraAdapter adapter = llavaAdapter();
        List<LoraAttachment.Binding> bindings =
                Collections.singletonList(new LoraAttachment.Binding(adapter, 1.0f));

        float[][] cpu;
        float[][] gpu;
        try (MmprojModel mm = MmprojModel.open(file.toString());
             CpuVisionEncoder cpuEnc = mm.openCpuVisionEncoder();
             VisionEncoder gpuEnc = mm.openVisionEncoder(backend)) {
            assertTrue(gpuEnc.toString(), gpuEnc instanceof GpuVisionEncoder);
            assertNotNull(cpuEnc.attachLoras(bindings));
            assertNotNull(gpuEnc.attachLoras(bindings));
            cpu = cpuEnc.encode(image);
            gpu = gpuEnc.encode(image);
        }
        assertRows(cpu, gpu, GPU_TOL);

        merge(f, adapter, Q_TARGET, 1.0f);
        merge(f, adapter, CONNECTOR_TARGET, 1.0f);
        assertRows(MmprojFixture.Reference.clip(f, hp, pixels, 8, 8), cpu);
    }

    // ================================================================== helpers

    /** Writes the fixture as an F32 mmproj GGUF file. */
    private static Path write(MmprojFixture f, Map<String, Object> meta) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("clip");
        for (Map.Entry<String, Object> e : meta.entrySet()) {
            Object v = e.getValue();
            if (e.getKey().equals("general.architecture")) {
                continue;
            }
            if (v instanceof Boolean) {
                w.bool(e.getKey(), (Boolean) v);
            } else if (v instanceof Integer) {
                w.u32(e.getKey(), (Integer) v);
            } else if (v instanceof Float) {
                w.f32(e.getKey(), (Float) v);
            } else if (v instanceof float[]) {
                w.f32Array(e.getKey(), (float[]) v);
            } else if (v instanceof int[]) {
                w.i32Array(e.getKey(), (int[]) v);
            } else {
                w.str(e.getKey(), (String) v);
            }
        }
        for (Map.Entry<String, long[]> e : f.shapes.entrySet()) {
            w.tensorFromFloats(e.getKey(), GGMLType.F32, e.getValue(), f.get(e.getKey()));
        }
        Path file = TMP.newFolder().toPath().resolve("mmproj.gguf");
        w.writeTo(file);
        return file;
    }

    /** Adds {@code scale * B A} of {@code target} to its fixture weight, in double (build_lora_mm). */
    private static void merge(MmprojFixture f, LoraAdapter adapter, String target, float userScale) {
        LoraAdapter.Pair pair = adapter.pair(target);
        LoraAdapter.Delta delta = adapter.delta(target, userScale);
        float[] a = delta.a();
        float[] b = delta.scaledB();
        float[] w = f.values.get(target);
        int in = pair.inDim();
        int out = pair.outDim();
        int rank = pair.rank();
        for (int o = 0; o < out; o++) {
            for (int i = 0; i < in; i++) {
                double acc = 0.0;
                for (int r = 0; r < rank; r++) {
                    acc += (double) b[o * rank + r] * a[r * in + i];
                }
                w[o * in + i] += (float) acc;
            }
        }
    }

    private static float maxAbsDiff(float[][] a, float[][] b) {
        assertEquals(a.length, b.length);
        float max = 0.0f;
        for (int r = 0; r < a.length; r++) {
            assertEquals(a[r].length, b[r].length);
            for (int c = 0; c < a[r].length; c++) {
                max = Math.max(max, Math.abs(a[r][c] - b[r][c]));
            }
        }
        return max;
    }

    private static void assertRows(float[][] want, float[][] got) {
        assertEquals(want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertEquals(want[t].length, got[t].length);
            assertArrayEquals("row " + t, want[t], got[t], TOL);
        }
    }

    private static void assertRows(float[][] want, float[][] got, float tol) {
        assertEquals(want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertEquals(want[t].length, got[t].length);
            assertArrayEquals("row " + t, want[t], got[t], tol);
        }
    }
}
