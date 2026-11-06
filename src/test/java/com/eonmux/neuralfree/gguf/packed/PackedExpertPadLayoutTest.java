package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Packed entries whose role data is not a multiple of 256 bytes, against a byte-level model of
 * Avalon's {@code pack-experts.py}: a zeroed {@code [n_expert, stride]} buffer, every role's data at
 * its offset, the pad after it filled with the next expert's first bytes (the last expert its own),
 * and the alignment slack left zero. Layer 0 has F32 up/gate roles of 2000 bytes (pad 256, slack 48)
 * and an F32 down role of 192 bytes whose 2048-byte pad exceeds the role, which fills one role's
 * bytes and leaves the rest zero; layer 1 has Q8_0 roles of 4080 bytes. The packed file, the
 * packing view's regions, and the unpacking view's store and regions all agree with the model, and
 * the unpacking view maps a range inside one expert in place.
 */
public class PackedExpertPadLayoutTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final String[] ROLES = {"up", "gate", "down"};

    /** {@code [ne0, ne1, n_expert]} and type of each layer's roles, in storage order. */
    private static final long[][][] SHAPES = {
            {{500, 1, 3}, {500, 1, 3}, {16, 3, 3}},
            {{480, 8, 2}, {480, 8, 2}, {480, 8, 2}}};
    private static final GGMLType[] TYPES = {GGMLType.F32, GGMLType.Q8_0};

    /** One role's data of every expert, the source tensor's bytes. */
    private static byte[] randomBytes(long seed, int n) {
        byte[] b = new byte[n];
        new Random(seed).nextBytes(b);
        return b;
    }

    private static long bytesOf(GGMLType type, long[] shape) {
        long blockElems = type == GGMLType.F32 ? 1 : 32;
        long blockBytes = type == GGMLType.F32 ? 4 : 34;
        return shape[0] / blockElems * blockBytes * shape[1] * shape[2];
    }

    /** {@code pack-experts.py role_pad}, from the block sizes of F32 (1, 4) and Q8_0 (32, 34). */
    private static long refRolePad(GGMLType type, long ne0) {
        long blockElems = type == GGMLType.F32 ? 1 : 32;
        long blockBytes = type == GGMLType.F32 ? 4 : 34;
        long over = ((ne0 + 511) / 512 * 512 - ne0) / blockElems * blockBytes;
        return (over + 255) / 256 * 256;
    }

    private static Path source(byte[][][] data) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("qwen3moe");
        for (int il = 0; il < SHAPES.length; il++) {
            for (int r = 0; r < ROLES.length; r++) {
                w.tensor("blk." + il + ".ffn_" + ROLES[r] + "_exps.weight", TYPES[il], SHAPES[il][r], data[il][r]);
            }
        }
        w.tensor("output_norm.weight", GGMLType.F32, new long[]{4}, randomBytes(99, 16));
        return w.writeTempFile();
    }

    private static byte[][][] sourceData() {
        byte[][][] data = new byte[SHAPES.length][ROLES.length][];
        for (int il = 0; il < SHAPES.length; il++) {
            for (int r = 0; r < ROLES.length; r++) {
                data[il][r] = randomBytes(10L * il + r, (int) bytesOf(TYPES[il], SHAPES[il][r]));
            }
        }
        return data;
    }

    /** The packed tensor of one layer, laid out as {@code pack-experts.py} lays it out. */
    private static byte[] expectedPacked(int il, byte[][] roles, long[] metaOut) {
        int nExpert = (int) SHAPES[il][0][2];
        long[] offsets = new long[ROLES.length];
        long off = 0;
        for (int r = 0; r < ROLES.length; r++) {
            offsets[r] = off;
            long nb = roles[r].length / nExpert;
            off += (nb + 255) / 256 * 256 + refRolePad(TYPES[il], SHAPES[il][r][0]);
        }
        int stride = (int) off;
        metaOut[0] = stride;
        metaOut[1] = nExpert;
        byte[] packed = new byte[stride * nExpert];
        for (int r = 0; r < ROLES.length; r++) {
            int nb = roles[r].length / nExpert;
            int fill = (int) Math.min(refRolePad(TYPES[il], SHAPES[il][r][0]), nb);
            metaOut[2 + 4 * r] = TYPES[il].getId();
            metaOut[3 + 4 * r] = SHAPES[il][r][0];
            metaOut[4 + 4 * r] = SHAPES[il][r][1];
            metaOut[5 + 4 * r] = offsets[r];
            for (int e = 0; e < nExpert; e++) {
                int at = e * stride + (int) offsets[r];
                System.arraycopy(roles[r], e * nb, packed, at, nb);
                int padFrom = Math.min(e + 1, nExpert - 1);
                System.arraycopy(roles[r], padFrom * nb, packed, at + nb, fill);
            }
        }
        return packed;
    }

    private Path pack(Path source) throws IOException {
        Path out = tmp.getRoot().toPath().resolve("packed.gguf");
        try (ModelSource src = ModelSources.open(source.toString())) {
            return GgufWriter.write(PackedExpertView.packedViewIfMoe(src), out,
                    new GgufWriter.Options(GgufWriter.Mode.NATIVE).overwrite(true)).get(0);
        }
    }

    @Test
    public void packedFileMatchesThePackScriptLayout() throws IOException {
        byte[][][] data = sourceData();
        Path packed = pack(source(data));
        try (GGUFFileParser p = ModelSources.parseGguf(packed.toString())) {
            for (int il = 0; il < SHAPES.length; il++) {
                long[] meta = new long[PackedExpertEntry.METADATA_VALUES];
                byte[] expected = expectedPacked(il, data[il], meta);
                assertArrayEquals("metadata of layer " + il, meta,
                        p.getUInt64Array("avalon.experts_packed.blk." + il));
                GGUFTensorInfo t = p.getTensor("blk." + il + ".ffn_exps_packed");
                assertNotNull(t);
                assertArrayEquals("packed bytes of layer " + il, expected, p.readTensorData(t));
            }
            assertArrayEquals(randomBytes(99, 16), p.readTensorData(p.getTensor("output_norm.weight")));
        }
    }

    @Test
    public void packingViewRegionsServeTheSameBytesAsItsStore() throws IOException {
        byte[][][] data = sourceData();
        Path source = source(data);
        try (ModelSource src = ModelSources.open(source.toString())) {
            ModelSource view = PackedExpertView.packedViewIfMoe(src);
            try (TensorRegions regions = view.openRegions(TensorReadMode.current());
                 WeightStore store = view.createMappedStore()) {
                for (int il = 0; il < SHAPES.length; il++) {
                    long[] meta = new long[PackedExpertEntry.METADATA_VALUES];
                    byte[] expected = expectedPacked(il, data[il], meta);
                    GGUFTensorInfo t = tensor(view, "blk." + il + ".ffn_exps_packed");
                    assertArrayEquals(expected, bytes(regions.region(t, 0, expected.length)));
                    // a range that starts inside a pad and ends past the next role's start
                    int from = (int) (meta[5] + 1990);
                    int to = (int) meta[5 + 4] + 40;
                    assertArrayEquals(Arrays.copyOfRange(expected, from, to),
                            bytes(regions.region(t, from, to - from)));
                    byte[] direct = new byte[to - from];
                    ByteBuffer dst = ByteBuffer.allocateDirect(to - from);
                    store.readTensorDirect(t.offset() + from, to - from, dst);
                    dst.flip().get(direct);
                    assertArrayEquals(Arrays.copyOfRange(expected, from, to), direct);
                }
            }
        }
    }

    @Test
    public void unpackingViewMapsOneExpertInPlaceAndCopiesOnlyAcrossExperts() throws IOException {
        byte[][][] data = sourceData();
        Path packed = pack(source(data));
        try (ModelSource view = ModelSources.open(packed.toString());
             TensorRegions regions = view.openRegions(TensorReadMode.MMAP)) {
            GGUFTensorInfo t = tensor(view, "blk.1.ffn_gate_exps.weight");
            int nb = data[1][1].length / 2;
            // the mapped file is read-only; an assembled copy is a fresh writable buffer
            assertTrue(regions.region(t, nb, nb).isReadOnly());
            assertTrue(regions.region(t, 0, nb / 2).isReadOnly());
            assertFalse(regions.region(t, nb / 2, nb).isReadOnly());
        }
    }

    @Test
    public void unpackingViewServesTheOriginalExpertTensors() throws IOException {
        byte[][][] data = sourceData();
        Path packed = pack(source(data));
        try (ModelSource view = ModelSources.open(packed.toString());
             TensorRegions regions = view.openRegions(TensorReadMode.current());
             WeightStore store = view.createMappedStore()) {
            for (int il = 0; il < SHAPES.length; il++) {
                for (int r = 0; r < ROLES.length; r++) {
                    GGUFTensorInfo t = tensor(view, "blk." + il + ".ffn_" + ROLES[r] + "_exps.weight");
                    byte[] want = data[il][r];
                    int nb = want.length / (int) SHAPES[il][r][2];
                    assertEquals(nb, regions.contiguousRunBytes(t));
                    assertArrayEquals(want, bytes(regions.region(t, 0, want.length)));
                    // one expert in place, and a range across two experts
                    assertArrayEquals(Arrays.copyOfRange(want, nb, 2 * nb), bytes(regions.region(t, nb, nb)));
                    assertArrayEquals(Arrays.copyOfRange(want, nb / 2, nb + nb / 2),
                            bytes(regions.region(t, nb / 2, nb)));
                    byte[] read = new byte[want.length];
                    store.readTensor(t.offset(), read.length, read, 0);
                    assertArrayEquals(want, read);
                    ByteBuffer dst = ByteBuffer.allocateDirect(nb);
                    store.readTensorDirect(t.offset() + nb / 2, nb, dst);
                    byte[] direct = new byte[nb];
                    dst.flip().get(direct);
                    assertArrayEquals(Arrays.copyOfRange(want, nb / 2, nb + nb / 2), direct);
                }
            }
        }
    }

    private static GGUFTensorInfo tensor(ModelSource source, String name) {
        for (GGUFTensorInfo t : source.getTensors()) {
            if (t.name().equals(name)) {
                return t;
            }
        }
        throw new AssertionError("no tensor " + name);
    }

    private static byte[] bytes(ByteBuffer b) {
        byte[] out = new byte[b.remaining()];
        b.duplicate().get(out);
        return out;
    }
}
