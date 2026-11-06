package com.eonmux.neuralfree.gguf.hf.torchao;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Scheme;
import com.eonmux.neuralfree.gguf.hf.quant.Fp8Scheme;
import com.eonmux.neuralfree.gguf.hf.quant.Int8Scheme;
import com.eonmux.neuralfree.gguf.hf.quant.IntGroupScheme;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The torchao checkpoints under {@code torchao/fixtures}: the safetensors and pickle readers decode every tensor
 * subclass into the ordinary checkpoint tensors, and each one, through its quantization scheme, dequantizes to the
 * values torchao itself produces. {@code torchao_fixtures.py} generated the fixtures and
 * {@code torchao-reference.json} (canonical tensor digests and the dequantized weight digest).
 */
public class TorchaoCheckpointTest {

    private static final String M = "model.layers.0.q_proj";
    private static final String W = M + ".weight";

    // ------------------------------------------------------------------ fixtures and reference

    private static Path fixture(String name) {
        URL url = TorchaoCheckpointTest.class.getResource(
                "/com/eonmux/neuralfree/gguf/hf/torchao/fixtures/" + name);
        assertNotNull("fixture " + name, url);
        try {
            return Paths.get(url.toURI());
        } catch (URISyntaxException e) {
            throw new AssertionError(e);
        }
    }

    private static Map<?, ?> reference(String fixture) throws IOException {
        Path file = fixture(fixture).getParent().resolve("torchao-reference.json");
        Object parsed = Json.parse(new String(Files.readAllBytes(file), StandardCharsets.UTF_8));
        Object entry = ((Map<?, ?>) parsed).get(fixture);
        assertNotNull("reference entry for " + fixture, entry);
        return (Map<?, ?>) entry;
    }

    private static CheckpointContainer open(String name) throws IOException {
        Path dir = fixture(name);
        return CheckpointContainers.open(dir, dir, false);
    }

    private static byte[] bytes(RowSource source) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(source.rows() * source.rowBytes()))
                .order(ByteOrder.LITTLE_ENDIAN);
        source.readRows(0, Math.toIntExact(source.rows()), b);
        assertEquals(b.capacity(), b.position());
        return b.array();
    }

    private static String sha256(byte[] bytes) {
        try {
            return hex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder s = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            s.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
        }
        return s.toString();
    }

    private static String digestF32(float[] values) {
        ByteBuffer b = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) {
            b.putFloat(v);
        }
        return sha256(b.array());
    }

    private static long numel(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        return n;
    }

    /** Every canonical tensor of the reference exists, with its dtype, shape and bytes, and no raw inner tensor. */
    private static void assertCanonical(String fixture, String... hidden) throws IOException {
        Map<?, ?> ref = reference(fixture);
        try (CheckpointContainer c = open(fixture)) {
            Map<?, ?> canonical = (Map<?, ?>) ref.get("canonical");
            assertFalse(fixture, canonical.isEmpty());
            for (Map.Entry<?, ?> e : canonical.entrySet()) {
                String name = (String) e.getKey();
                Map<?, ?> want = (Map<?, ?>) e.getValue();
                StTensor t = c.tensor(name);
                assertNotNull(fixture + " tensor " + name, t);
                StDType dtype = StDType.valueOf((String) want.get("dtype"));
                assertEquals(fixture + " " + name + " dtype", dtype, t.dtype());
                assertArrayEquals(fixture + " " + name + " shape", longs((List<?>) want.get("shape")), t.shape());
                assertEquals(fixture + " " + name + " bytes", want.get("sha256"), sha256(bytes(c.source(name))));
            }
            for (String name : hidden) {
                assertNull(fixture + " hides " + name, c.tensor(name));
                assertThrows(IllegalArgumentException.class, () -> c.source(name));
            }
        }
    }

    private static long[] longs(List<?> list) {
        long[] out = new long[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) list.get(i)).longValue();
        }
        return out;
    }

    // ------------------------------------------------------------------ safetensors

    @Test
    public void int4SafetensorsDecodesAndRewritesToTheReferenceValues() throws IOException {
        assertCanonical("int4", M + "._weight_qdata", M + "._weight_scale", M + "._weight_zero_point");
        assertQuantized("int4", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(8).build());
    }

    @Test
    public void tilePackedInt4SafetensorsDecodesThroughTheTileLayout() throws IOException {
        assertCanonical("int4-tile", M + "._weight_qdata", M + "._weight_scale_and_zero");
        assertQuantized("int4-tile", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(32).build());
    }

    @Test
    public void int8SafetensorsDecodesToTheChannelScheme() throws IOException {
        assertCanonical("int8", M + "._weight_qdata", M + "._weight_scale");
        assertQuantized("int8", QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(QuantArgs.CHANNEL).build());
    }

    @Test
    public void float8SafetensorsDecodesToTheFp8Scheme() throws IOException {
        assertCanonical("float8", M + "._weight_qdata", M + "._weight_scale");
        assertQuantized("float8", QuantArgs.builder(QuantFormat.FP8).group(QuantArgs.CHANNEL).build());
    }

    @Test
    public void nvfp4SafetensorsUnswizzlesItsScalesAndKeepsBothGlobalScales() throws IOException {
        assertCanonical("nvfp4", M + "._weight_qdata", M + "._weight_scale", M + "._weight_per_tensor_scale",
                M + "._weight_act_per_tensor_scale");
        assertQuantized("nvfp4", QuantArgs.builder(QuantFormat.NVFP4_MODELOPT).group(16)
                .actScheme(ActQuantScheme.NVFP4).build());
    }

    @Test
    public void swizzledMxfp4SafetensorsUnswizzlesItsScales() throws IOException {
        assertCanonical("mxfp4", M + "._weight_qdata", M + "._weight_scale");
        assertQuantized("mxfp4", QuantArgs.builder(QuantFormat.MXFP4_CT).actScheme(ActQuantScheme.MXFP4).build());
    }

    @Test
    public void mxfp8SafetensorsKeepsTheCompressedTensorsWeightName() throws IOException {
        assertCanonical("mxfp8", M + "._weight_qdata", M + "._weight_scale");
        assertQuantized("mxfp8", QuantArgs.builder(QuantFormat.MXFP8).actScheme(ActQuantScheme.MXFP8).build());
    }

    // ------------------------------------------------------------------ preshuffled int4

    @Test
    public void preshuffledBf16SafetensorsInvertsTheReorderAtom() throws IOException {
        assertCanonical("preshuffle-bf16", M + "._weight_qdata", M + "._weight_group_scale",
                M + "._weight_group_zero");
        assertQuantized("preshuffle-bf16", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(32).build(),
                GGMLType.Q4_1);
    }

    @Test
    public void preshuffledBf16ExpertsInvertTheReorderAtomPerExpert() throws IOException {
        assertCanonical("preshuffle-bf16-expert", M + "._weight_qdata", M + "._weight_group_scale",
                M + "._weight_group_zero");
        assertQuantized("preshuffle-bf16-expert", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(32).build(),
                GGMLType.Q4_1);
    }

    @Test
    public void preshuffledFp8SafetensorsRecoversThePackedGroupScale() throws IOException {
        assertCanonical("preshuffle-fp8", M + "._weight_qdata", M + "._weight_group_scale",
                M + "._weight_row_scale");
        assertQuantized("preshuffle-fp8", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(32)
                .actScheme(ActQuantScheme.FP8_TOKEN).build(), GGMLType.Q4_1);
    }

    @Test
    public void preshuffledFp8ExpertsRecoverThePackedScalePerExpert() throws IOException {
        assertCanonical("preshuffle-fp8-expert", M + "._weight_qdata", M + "._weight_group_scale",
                M + "._weight_row_scale");
        assertQuantized("preshuffle-fp8-expert", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(32)
                .actScheme(ActQuantScheme.FP8_TOKEN).build(), GGMLType.Q4_1);
    }

    // ------------------------------------------------------------------ XPU plain int32 and unpacked intx

    @Test
    public void plainInt32SafetensorsDecodesTheXPUWordLayout() throws IOException {
        assertCanonical("plain-int32", M + "._weight_qdata", M + "._weight_scale", M + "._weight_zero_point");
        assertQuantized("plain-int32", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(32).build(),
                GGMLType.Q4_1);
    }

    @Test
    public void unpackedInt4SafetensorsPacksNibbles() throws IOException {
        assertCanonical("intx4", M + "._weight_qdata", M + "._weight_scale", M + "._weight_zero_point");
        assertQuantized("intx4", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(32).build(), GGMLType.Q4_1);
    }

    @Test
    public void unpackedSymmetricInt8SafetensorsKeepsItsCodesInAQ8Block() throws IOException {
        assertCanonical("intx8-sym", M + "._weight_qdata", M + "._weight_scale", M + "._weight_zero_point");
        assertQuantized("intx8-sym", QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(32).build(), GGMLType.Q8_0);
    }

    @Test
    public void unpackedInt8SafetensorsKeepsItsCodes() throws IOException {
        assertCanonical("intx8", M + "._weight_qdata", M + "._weight_scale", M + "._weight_zero_point");
        // An asymmetric int8 zero point pushes q - z outside int8, so the exact F32 fallback carries it.
        assertQuantized("intx8", QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(32).build(), GGMLType.F32);
    }

    /** int1..int3 and int5..int7 keep their int8 byte codes, which the int8 channel scheme carries in Q8_0. */
    @Test
    public void unpackedNarrowAndWideIntxTargetsKeepTheirCodesInAQ8Block() throws IOException {
        for (int bits : new int[]{1, 2, 3, 5, 6, 7}) {
            String fixture = "intx" + bits;
            assertCanonical(fixture, M + "._weight_qdata", M + "._weight_scale", M + "._weight_zero_point");
            assertQuantized(fixture, QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(32).build(), GGMLType.Q8_0);
        }
    }

    // ------------------------------------------------------------------ per-expert tile packing and activation siblings

    @Test
    public void tilePackedInt4ExpertsDecodePerExpert() throws IOException {
        assertCanonical("int4-tile-expert", M + "._weight_qdata", M + "._weight_scale_and_zero");
        assertQuantized("int4-tile-expert", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(32).build(),
                GGMLType.Q4_1);
    }

    @Test
    public void tilePackedInt4Ntile16DecodesThroughThePaddedLayout() throws IOException {
        assertCanonical("int4-tile-ntile16", M + "._weight_qdata", M + "._weight_scale_and_zero");
        assertQuantized("int4-tile-ntile16", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(32).build(),
                GGMLType.Q4_1);
    }

    @Test
    public void activationPreScaleFoldsIntoTheInt4FloatFallback() throws IOException {
        assertCanonical("int4-pre-scale", M + "._weight_qdata", M + "._weight_scale",
                M + "._weight_zero_point");
        assertQuantized("int4-pre-scale", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(8).build(),
                GGMLType.F32);
    }

    @Test
    public void activationPreScaleFoldsIntoTheInt8FloatFallback() throws IOException {
        assertCanonical("int8-pre-scale", M + "._weight_qdata", M + "._weight_scale");
        assertQuantized("int8-pre-scale", QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(QuantArgs.CHANNEL)
                .build(), GGMLType.F32);
    }

    @Test
    public void staticActivationSiblingsBecomeTheInputTransformParameters() throws IOException {
        assertCanonical("int8-static-act", M + "._weight_qdata", M + "._weight_scale",
                M + "._weight_act_quant_scale", M + "._weight_act_quant_zero_point");
        assertQuantized("int8-static-act", QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(QuantArgs.CHANNEL)
                .actScheme(ActQuantScheme.INT8_STATIC_ASYM).build());
    }

    @Test
    public void activationPreScaleRidesAlongsideStaticActivation() throws IOException {
        assertCanonical("int8-pre-scale-act", M + "._weight_qdata", M + "._weight_scale");
        QuantArgs args = QuantArgs.builder(QuantFormat.INT8_CHANNEL).group(QuantArgs.CHANNEL)
                .actScheme(ActQuantScheme.INT8_STATIC_ASYM).build();
        // The weight keeps its own quantization; the pre-scale is not folded into it.
        assertQuantized("int8-pre-scale-act", args, GGMLType.F32);
        Path dir = fixture("int8-pre-scale-act");
        HfConfig cfg = HfConfigLoader.fromJson(dir, "{\"architectures\":[\"A\"]}");
        try (CheckpointContainer c = open("int8-pre-scale-act")) {
            HfTensorIndex out = Int8Scheme.rewriteModules(args, List.of(M), cfg, HfTensorIndex.of(c),
                    ModelSourceOptions.defaults());
            RowSource w = out.get(W);
            List<?> recorded = (List<?>) reference("int8-pre-scale-act").get("activation_pre_scale");
            float[] want = new float[recorded.size()];
            for (int i = 0; i < want.length; i++) {
                want[i] = ((Number) recorded.get(i)).floatValue();
            }
            assertArrayEquals(want, w.inputPreScale(), 0.0f);
            assertEquals(ActQuantScheme.INT8_STATIC_ASYM, w.actQuant());
            assertNotNull(w.inputScale());
            assertNotNull(w.inputZeroPoint());
        }
    }

    // ------------------------------------------------------------------ pickle

    @Test
    public void int4PickleWrapperSubclassDecodesAndRewritesToTheReferenceValues() throws IOException {
        assertCanonical("int4-pickle");
        assertQuantized("int4-pickle", QuantArgs.builder(QuantFormat.INT4_TORCHAO).group(8).build());
    }

    @Test
    public void float8PickleV3ViewsDecodeWithTheirDtypeMarker() throws IOException {
        assertCanonical("float8-pickle");
        assertQuantized("float8-pickle", QuantArgs.builder(QuantFormat.FP8).group(QuantArgs.CHANNEL).build());
    }

    @Test
    public void picklePrefixIsTheArchiveName() throws IOException {
        // torch 2.12 writes <stem>/data.pkl, not archive/data.pkl
        try (CheckpointContainer c = open("int4-pickle")) {
            assertNotNull(c.tensor(W));
        }
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void unknownSubclassIsRefusedByName() throws IOException {
        TorchaoWeight mystery = new TorchaoWeight(W, "MysteryTensor", null,
                new LinkedHashMap<>(), new LinkedHashMap<>());
        IOException e = assertThrows(IOException.class,
                () -> TorchaoDecoder.decode(List.of(mystery), "the test checkpoint"));
        assertTrue(e.getMessage(), e.getMessage().contains("MysteryTensor"));
    }

    @Test
    public void unknownMetadataTypeIsRefusedByName() throws IOException {
        Map<String, String> metadata = new LinkedHashMap<>();
        metadata.put(TorchaoSafetensors.TENSOR_NAMES, "[\"" + W + "\"]");
        metadata.put(W, "{\"_type\":\"MysteryTensor\",\"_data\":{},\"_tensor_data_names\":[]}");
        IOException e = assertThrows(IOException.class, () -> TorchaoSafetensors.read(List.of(metadata),
                new TorchaoSafetensors.Tensors() {
                    @Override
                    public StTensor tensor(String name) {
                        return null;
                    }

                    @Override
                    public RowSource source(String name) {
                        throw new IllegalArgumentException(name);
                    }
                }, "the test checkpoint"));
        assertTrue(e.getMessage(), e.getMessage().contains("MysteryTensor"));
    }

    // ------------------------------------------------------------------ schemes

    /** Rewrites the fixture's one module with {@code args} and digests the dequantized weight against torchao. */
    private static void assertQuantized(String fixture, QuantArgs args) throws IOException {
        assertQuantized(fixture, args, null);
    }

    /** As {@link #assertQuantized(String, QuantArgs)}, also requiring the ggml target the layout should choose. */
    private static void assertQuantized(String fixture, QuantArgs args, GGMLType target) throws IOException {
        Path dir = fixture(fixture);
        HfConfig cfg = HfConfigLoader.fromJson(dir, "{\"architectures\":[\"A\"]}");
        try (CheckpointContainer c = open(fixture)) {
            HfTensorIndex in = HfTensorIndex.of(c);
            HfTensorIndex out;
            switch (args.format()) {
                case INT4_TORCHAO:
                    out = IntGroupScheme.rewriteModules(args, List.of(M), cfg, in, ModelSourceOptions.defaults());
                    break;
                case INT8_CHANNEL:
                    out = Int8Scheme.rewriteModules(args, List.of(M), cfg, in, ModelSourceOptions.defaults());
                    break;
                case FP8: case MXFP8:
                    out = Fp8Scheme.rewriteModules(args, List.of(M), cfg, in, ModelSourceOptions.defaults());
                    break;
                default:
                    out = Fp4Scheme.rewriteModules(args, List.of(M), cfg, in, ModelSourceOptions.defaults());
                    break;
            }
            RowSource w = out.get(W);
            assertNotNull(fixture + " rewritten weight", w);
            if (target != null) {
                assertEquals(fixture + " target", target, w.format().ggmlType());
            }
            float[] values = CpuBlockDecoders.dequantize(bytes(w), w.format().ggmlType(), numel(w.shape()));
            float[] scale = w.outputScale();
            if (scale != null) {
                for (int i = 0; i < values.length; i++) {
                    values[i] *= scale[0];
                }
            }
            assertEquals(fixture + " dequantized weight", reference(fixture).get("dequantized"), digestF32(values));
        }
    }

    @Test
    public void everyFixtureHasTheSameBiasUntouched() throws IOException {
        for (String name : Arrays.asList("int4", "int4-tile", "int8", "float8", "nvfp4", "mxfp4", "mxfp8",
                "int4-pickle", "float8-pickle", "preshuffle-bf16", "preshuffle-bf16-expert", "preshuffle-fp8",
                "preshuffle-fp8-expert",
                "plain-int32", "intx1", "intx2", "intx3", "intx4", "intx5", "intx6", "intx7", "intx8", "intx8-sym",
                "int4-tile-expert", "int4-tile-ntile16", "int4-pre-scale",
                "int8-pre-scale", "int8-static-act", "int8-pre-scale-act")) {
            try (CheckpointContainer c = open(name)) {
                StTensor bias = c.tensor(M + ".bias");
                assertNotNull(name, bias);
                assertArrayEquals(name, new long[]{8}, bias.shape());
            }
        }
    }
}
