package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The Avalon packed-expert layout end to end: {@code nftool pack-experts}'s pipeline (the
 * {@link PackedExpertView} plus {@link GgufWriter}) rewrites the synthetic qwen3next MoE model into
 * one {@code blk.N.ffn_exps_packed} tensor per layer, the packed bytes match
 * {@code pack-experts.py} (role data per expert, pads of the next expert's first bytes), the file
 * loads through {@code ModelSources} as its unpacked form with byte-identical tensors and
 * {@link CpuWeights} views, and a packed model exports back to the standard layout unchanged.
 * Malformed packed files are refused with the layer or key named.
 */
public class PackExpertsRoundTripTest {

    /** The per-layer entry layout of the F32 n_embd-64 / n_ff_exp-32 / 4-expert fixture. */
    private static final long UP_OFFSET = 0;
    private static final long GATE_OFFSET = 9984;
    private static final long DOWN_OFFSET = 19968;
    private static final long STRIDE = 30208;
    private static final long ENTRY_BYTES = 8192;
    private static final int N_EXPERT = 4;
    private static final int N_LAYERS = 4;
    private static final String[] ROLES = {"up", "gate", "down"};
    private static final long[] ROLE_OFFSETS = {UP_OFFSET, GATE_OFFSET, DOWN_OFFSET};

    private static Path writeOriginalModel() throws IOException {
        return SyntheticModels.qwen3next().writeTempFile();
    }

    /** Packs {@code orig} the way {@code nftool pack-experts} does, and returns the file written. */
    private static Path pack(Path orig, Path packedPath) throws IOException {
        try (ModelSource src = ModelSources.open(orig.toString())) {
            ModelSource packed = PackedExpertView.packedViewIfMoe(src);
            assertTrue(packed instanceof PackedExpertView);
            return GgufWriter.write(packed, packedPath,
                    new GgufWriter.Options(GgufWriter.Mode.NATIVE).overwrite(true)).get(0);
        }
    }

    private static byte[] tensorBytes(GGUFFileParser parser, String name) throws IOException {
        GGUFTensorInfo info = parser.getTensor(name);
        assertNotNull(name, info);
        return parser.readTensorData(info);
    }

    private static long nextRoleOffset(int role) {
        return role + 1 < ROLES.length ? ROLE_OFFSETS[role + 1] : STRIDE;
    }

    @Test
    public void packedFileMatchesPackExpertsLayoutAndBytes() throws IOException {
        Path orig = writeOriginalModel();
        Path packedPath = Files.createTempFile("nf-packed-", ".gguf");
        packedPath.toFile().deleteOnExit();
        try (GGUFFileParser packed = ModelSources.parseGguf(pack(orig, packedPath).toString());
             GGUFFileParser original = ModelSources.parseGguf(orig.toString())) {
            for (int il = 0; il < N_LAYERS; il++) {
                for (String role : ROLES) {
                    assertNull(packed.getTensor("blk." + il + ".ffn_" + role + "_exps.weight"));
                }
                GGUFTensorInfo p = packed.getTensor("blk." + il + ".ffn_exps_packed");
                assertNotNull(p);
                assertEquals(GGMLType.I8, p.type());
                assertArrayEquals(new long[]{STRIDE, N_EXPERT}, p.dims());
                assertArrayEquals(new long[]{STRIDE, N_EXPERT,
                        GGMLType.F32.getId(), 64L, 32L, UP_OFFSET,
                        GGMLType.F32.getId(), 64L, 32L, GATE_OFFSET,
                        GGMLType.F32.getId(), 32L, 64L, DOWN_OFFSET},
                        packed.getUInt64Array("avalon.experts_packed.blk." + il));
            }
            // every non-expert tensor is unchanged
            for (GGUFTensorInfo t : original.getTensors()) {
                if (t.name().endsWith("_exps.weight")) {
                    continue;
                }
                assertArrayEquals(t.name(), tensorBytes(original, t.name()), tensorBytes(packed, t.name()));
            }
            // the packed bytes: every expert's role data at its offset, the pad filled with the
            // next expert's first bytes (the last expert repeats its own), as pack-experts.py writes
            byte[][] roleBytes = {
                    tensorBytes(original, "blk.0.ffn_up_exps.weight"),
                    tensorBytes(original, "blk.0.ffn_gate_exps.weight"),
                    tensorBytes(original, "blk.0.ffn_down_exps.weight")};
            byte[] packedBytes = tensorBytes(packed, "blk.0.ffn_exps_packed");
            assertEquals(STRIDE * N_EXPERT, packedBytes.length);
            for (int role = 0; role < ROLES.length; role++) {
                long offset = ROLE_OFFSETS[role];
                long bytes = roleBytes[role].length / N_EXPERT;
                assertEquals(ENTRY_BYTES, bytes);
                for (int e = 0; e < N_EXPERT; e++) {
                    for (int b = 0; b < bytes; b++) {
                        assertEquals("role " + role + " expert " + e + " byte " + b,
                                roleBytes[role][e * (int) bytes + b],
                                packedBytes[(int) (e * STRIDE + offset + b)]);
                    }
                    long padStart = offset + bytes;
                    long padBytes = nextRoleOffset(role) - padStart;
                    int padExpert = Math.min(e + 1, N_EXPERT - 1);
                    for (long b = 0; b < padBytes; b++) {
                        assertEquals("role " + role + " expert " + e + " pad byte " + b,
                                roleBytes[role][(int) (padExpert * bytes + b)],
                                packedBytes[(int) (e * STRIDE + padStart + b)]);
                    }
                }
            }
        } finally {
            Files.deleteIfExists(orig);
            Files.deleteIfExists(packedPath);
        }
    }

    @Test
    public void packedModelOpensAsItsUnpackedForm() throws IOException {
        Path orig = writeOriginalModel();
        Path packedPath = Files.createTempFile("nf-packed-", ".gguf");
        packedPath.toFile().deleteOnExit();
        try {
            pack(orig, packedPath);
            try (ModelSource view = ModelSources.open(packedPath.toString())) {
                TensorCatalog catalog = TensorCatalog.of(view);
                for (int il = 0; il < N_LAYERS; il++) {
                    for (String role : ROLES) {
                        String name = TensorNames.layerName(il, TensorNames.weight("ffn_" + role + "_exps"));
                        assertNotNull(name, catalog.get(name));
                    }
                    assertNull(catalog.get(TensorNames.layerName(il, "ffn_exps_packed")));
                }
                for (String key : view.getMetadata().keySet()) {
                    assertFalse(key, PackedExpertsIndex.isMetadataKey(key));
                }
                // the MoE gate/up pair resolves over the synthesized tensors, as over the unpacked model
                TensorCatalog.GateUpExps gu = catalog.gateUpExps(1);
                assertNotNull(gu);
                assertFalse(gu.fused());
                assertTrue(gu.gated());
                assertArrayEquals(new long[]{64, 32, N_EXPERT}, gu.gate().ne());
                assertArrayEquals(new long[]{64, 32, N_EXPERT}, gu.up().ne());
            }
        } finally {
            Files.deleteIfExists(orig);
            Files.deleteIfExists(packedPath);
        }
    }

    @Test
    public void packedModelTensorsAreByteIdenticalToTheOriginal() throws IOException {
        Path orig = writeOriginalModel();
        Path packedPath = Files.createTempFile("nf-packed-", ".gguf");
        packedPath.toFile().deleteOnExit();
        try (GGUFFileParser original = ModelSources.parseGguf(orig.toString())) {
            pack(orig, packedPath);
            try (ModelSource view = ModelSources.open(packedPath.toString())) {
                // host regions of every tensor, synthesized expert tensors included
                try (TensorRegions regions = view.openRegions(TensorReadMode.current())) {
                    for (GGUFTensorInfo t : view.getTensors()) {
                        byte[] bytes = new byte[(int) t.sizeInBytes()];
                        regions.region(t, 0, t.sizeInBytes()).get(bytes);
                        assertArrayEquals(t.name(), tensorBytes(original, t.name()), bytes);
                    }
                }
                // the weight store path the GPU staging and the weight provider read through
                try (WeightStore store = view.createMappedStore()) {
                    for (GGUFTensorInfo t : view.getTensors()) {
                        byte[] bytes = new byte[(int) t.sizeInBytes()];
                        store.readTensor(t.offset(), bytes.length, bytes, 0);
                        assertArrayEquals(t.name(), tensorBytes(original, t.name()), bytes);
                    }
                    // and one range per expert, as the per-expert readers do
                    GGUFTensorInfo up = view.getTensors().stream()
                            .filter(t -> t.name().endsWith("blk.2.ffn_up_exps.weight")
                                    || t.name().equals("blk.2.ffn_up_exps.weight"))
                            .findFirst().orElseThrow(AssertionError::new);
                    byte[] whole = tensorBytes(original, "blk.2.ffn_up_exps.weight");
                    int perExpert = (int) ENTRY_BYTES;
                    for (int e = 0; e < N_EXPERT; e++) {
                        byte[] slice = new byte[perExpert];
                        store.readTensor(up.offset() + e * perExpert, perExpert, slice, 0);
                        assertArrayEquals("expert " + e,
                                Arrays.copyOfRange(whole, e * perExpert, (e + 1) * perExpert), slice);
                    }
                }
            }
        } finally {
            Files.deleteIfExists(orig);
            Files.deleteIfExists(packedPath);
        }
    }

    @Test
    public void cpuWeightsDecodeThePackedExpertsToTheOriginalFloats() throws IOException {
        Path orig = writeOriginalModel();
        Path packedPath = Files.createTempFile("nf-packed-", ".gguf");
        packedPath.toFile().deleteOnExit();
        try {
            pack(orig, packedPath);
            try (ModelSource original = ModelSources.open(orig.toString());
                 ModelSource view = ModelSources.open(packedPath.toString());
                 CpuWeights packedWeights = CpuWeights.open(view);
                 CpuWeights originalWeights = CpuWeights.open(original)) {
                for (int il = 0; il < N_LAYERS; il++) {
                    for (String role : ROLES) {
                        String name = TensorNames.layerName(il, TensorNames.weight("ffn_" + role + "_exps"));
                        CpuTensorRef packedRef = packedWeights.get(name);
                        CpuTensorRef originalRef = originalWeights.get(name);
                        assertEquals(originalRef.rows(), packedRef.rows());
                        assertEquals(originalRef.cols(), packedRef.cols());
                        assertArrayEquals(decode(originalRef), decode(packedRef), 0f);
                    }
                }
            }
        } finally {
            Files.deleteIfExists(orig);
            Files.deleteIfExists(packedPath);
        }
    }

    /** Decodes a whole 3-D expert tensor row by row ({@code rows = ne1 * ne2}). */
    private static float[] decode(CpuTensorRef ref) {
        float[] out = new float[(int) (ref.rows() * ref.cols())];
        int rowFloats = (int) ref.cols();
        for (int r = 0; r < ref.rows(); r++) {
            ref.decodeRow(r, out, r * rowFloats);
        }
        return out;
    }

    @Test
    public void packedModelExportsBackToTheStandardLayout() throws IOException {
        Path orig = writeOriginalModel();
        Path packedPath = Files.createTempFile("nf-packed-", ".gguf");
        Path exported = Files.createTempFile("nf-unpacked-", ".gguf");
        packedPath.toFile().deleteOnExit();
        exported.toFile().deleteOnExit();
        try (GGUFFileParser original = ModelSources.parseGguf(orig.toString())) {
            pack(orig, packedPath);
            try (ModelSource view = ModelSources.open(packedPath.toString())) {
                GgufWriter.write(view, exported,
                        new GgufWriter.Options(GgufWriter.Mode.NATIVE).overwrite(true));
            }
            try (GGUFFileParser back = ModelSources.parseGguf(exported.toString())) {
                assertEquals(original.getTensors().size(), back.getTensors().size());
                for (GGUFTensorInfo t : original.getTensors()) {
                    assertArrayEquals(t.name(), tensorBytes(original, t.name()), tensorBytes(back, t.name()));
                    assertEquals(t.name() + " dims", Arrays.toString(t.dims()),
                            Arrays.toString(back.getTensor(t.name()).dims()));
                }
                for (int il = 0; il < N_LAYERS; il++) {
                    assertNull(back.getTensor("blk." + il + ".ffn_exps_packed"));
                    assertNull(back.getArrayElementType("avalon.experts_packed.blk." + il));
                }
            }
        } finally {
            Files.deleteIfExists(orig);
            Files.deleteIfExists(packedPath);
            Files.deleteIfExists(exported);
        }
    }

    // ------------------------------------------------------------------ malformed packed files

    /**
     * A one-layer GGUF with the fixture's packed tensor of shape {@code [stride, n_expert]} when
     * {@code withPackedTensor}, the given UINT64 and UINT32 metadata arrays, and optionally the
     * standard up tensor beside it.
     */
    private static Path packedFixture(long[][] u64Arrays, long[][] u32Arrays, boolean withPackedTensor,
                                      boolean withStandardTensor) throws IOException {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("qwen3next");
        for (long[] v : u64Arrays) {
            w.u64Array("avalon.experts_packed.blk.0", v);
        }
        for (long[] v : u32Arrays) {
            w.u32Array("avalon.experts_packed.blk.0", v);
        }
        if (withPackedTensor) {
            w.tensor("blk.0.ffn_exps_packed", GGMLType.I8, new long[]{STRIDE, N_EXPERT},
                    new byte[(int) (STRIDE * N_EXPERT)]);
        }
        if (withStandardTensor) {
            w.tensor("blk.0.ffn_up_exps.weight", GGMLType.F32, new long[]{16, 1, 1}, new byte[64]);
        }
        return w.writeTempFile();
    }

    private static long[] validKey() {
        return new long[]{STRIDE, N_EXPERT,
                GGMLType.F32.getId(), 16L, 1L, 0L,
                GGMLType.F32.getId(), 16L, 1L, 256L,
                GGMLType.F32.getId(), 16L, 1L, 512L};
    }

    @Test
    public void packedTensorWithoutMetadataKeyIsRefused() throws IOException {
        Path file = packedFixture(new long[0][], new long[0][], true, false);
        file.toFile().deleteOnExit();
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("avalon.experts_packed.blk.0"));
        Files.deleteIfExists(file);
    }

    @Test
    public void metadataKeyWithoutPackedTensorIsRefused() throws IOException {
        Path file = packedFixture(new long[][]{validKey()}, new long[0][], false, false);
        file.toFile().deleteOnExit();
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("describes no blk.0.ffn_exps_packed"));
        Files.deleteIfExists(file);
    }

    @Test
    public void nonUint64MetadataKeyIsRefused() throws IOException {
        Path file = packedFixture(new long[0][], new long[][]{validKey()}, true, false);
        file.toFile().deleteOnExit();
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("expected a UINT64 array"));
        Files.deleteIfExists(file);
    }

    @Test
    public void shortMetadataKeyIsRefused() throws IOException {
        Path file = packedFixture(new long[][]{Arrays.copyOf(validKey(), 13)}, new long[0][], true, false);
        file.toFile().deleteOnExit();
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("holds 13 values"));
        Files.deleteIfExists(file);
    }

    @Test
    public void standardTensorBesideThePackedTensorIsRefused() throws IOException {
        Path file = packedFixture(new long[][]{validKey()}, new long[0][], true, true);
        file.toFile().deleteOnExit();
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("cannot both be present"));
        Files.deleteIfExists(file);
    }

    @Test
    public void strideNotMatchingThePackedTensorIsRefused() throws IOException {
        long[] values = validKey();
        values[0] = STRIDE + 1;
        Path file = packedFixture(new long[][]{values}, new long[0][], true, false);
        file.toFile().deleteOnExit();
        IOException e = assertThrows(IOException.class, () -> ModelSources.open(file.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("expected stride " + (STRIDE + 1)));
        Files.deleteIfExists(file);
    }

    @Test
    public void unpackedModelsOpenUnchanged() throws IOException {
        Path file = writeOriginalModel();
        file.toFile().deleteOnExit();
        try (ModelSource src = ModelSources.open(file.toString())) {
            assertFalse(src instanceof StandardExpertView);
            TensorCatalog catalog = TensorCatalog.of(src);
            assertNotNull(catalog.get("blk.0.ffn_up_exps.weight"));
            assertNull(catalog.get("blk.0.ffn_exps_packed"));
        } finally {
            Files.deleteIfExists(file);
        }
    }
}
