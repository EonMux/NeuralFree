package com.eonmux.neuralfree.gguf.hf.store;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.Materialize;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicy.Reader;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicyTest.Counting;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlanBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link HfWeightStore}: reads at virtual offsets across checkpoint files, the zero gaps between them, and the derived
 * tensors after {@code P}.
 */
public class HfWeightStoreTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final byte[] A = SyntheticHf.f32(1, 2, 3);
    private static final byte[] B = SyntheticHf.f32(4, 5, 6, 7, 8);
    private static final byte[] C = SyntheticHf.f32(9, 10, 11, 12, 13, 14, 15);

    private CheckpointContainer ckpt;
    private Counting derivedSource;
    private VirtualLayout layout;
    private SidecarDirectory sidecars;

    @Before
    public void open() throws IOException {
        Path dir = SyntheticHf.builder().config("{}")
                .tensor("a", StDType.F32, new long[]{3}, A)
                .tensor("b", StDType.F32, new long[]{5}, B)
                .tensor("c", StDType.F32, new long[]{7}, C)
                .shards(2)
                .write(tmp.newFolder().toPath());
        ckpt = CheckpointContainers.open(dir, dir, false);
        derivedSource = new Counting(2, 2, 3, 4);
        TensorPlanBuilder b = VirtualLayoutTest.builder();
        b.add("a.weight", ckpt.source("a"));
        b.add("b.weight", ckpt.source("b"));
        b.add("c.weight", ckpt.source("c"));
        b.add("blk.0.ffn_down_exps.weight", ExpertTransforms.transposeLast2(derivedSource));
        b.add("blk.0.extra.weight", new Counting(7, 3, 5));
        TensorPlan plan = b.build();
        layout = VirtualLayout.of(ckpt, plan);
        sidecars = new SidecarDirectory(tmp.newFolder("cache").toPath());
    }

    @After
    public void close() throws IOException {
        sidecars.close();
        ckpt.close();
    }

    private HfWeightStore store(Reader reader, Materialize setting) throws IOException {
        ProducedSegments p = layout.produced();
        return new HfWeightStore(layout, p.view(MaterializePolicy.of(reader, setting, 1 << 20, p.tensors(),
                Long.MAX_VALUE), sidecars));
    }

    /** The expected bytes of the whole virtual address space. */
    private byte[] expected() throws IOException {
        byte[] all = new byte[(int) layout.totalBytes()];
        put(all, ckpt.directVirtualOffset("a"), A);
        put(all, ckpt.directVirtualOffset("b"), B);
        put(all, ckpt.directVirtualOffset("c"), C);
        ProducedSegments p = layout.produced();
        put(all, p.offset(0), MaterializePolicyTest.transposed(derivedSource));
        put(all, p.offset(1), new Counting(7, 3, 5).bytes());
        return all;
    }

    private static void put(byte[] all, long at, byte[] bytes) {
        System.arraycopy(bytes, 0, all, (int) at, bytes.length);
    }

    @Test
    public void everyRangeReadsTheFilesTheGapsAndTheDerivedTensors() throws IOException {
        byte[] expect = expected();
        List<DataSegment> segs = layout.segments();
        assertEquals(2, segs.size());
        for (Reader r : new Reader[]{Reader.RESIDENT_UPLOAD, Reader.STREAMING}) {
            for (Materialize m : Materialize.values()) {
                try (HfWeightStore s = store(r, m)) {
                    byte[] whole = new byte[expect.length];
                    s.readTensor(0, whole.length, whole);
                    assertArrayEquals(r + "/" + m, expect, whole);
                }
            }
        }
    }

    @Test
    public void readsCrossFileAndDerivedBoundaries() throws IOException {
        byte[] expect = expected();
        List<DataSegment> segs = layout.segments();
        long endA = segs.get(0).virtualEnd();
        long p = layout.producedBase();
        List<long[]> ranges = new ArrayList<>(Arrays.asList(
                new long[]{endA - 5, 10},
                new long[]{endA - 3, segs.get(1).virtualBase() - endA + 9},
                new long[]{segs.get(1).virtualEnd() - 6, p - segs.get(1).virtualEnd() + 20},
                new long[]{p + 40, layout.produced().offset(1) - p - 40 + 8},
                new long[]{1, layout.totalBytes() - 2}));
        try (HfWeightStore s = store(Reader.RESIDENT_UPLOAD, Materialize.AUTO)) {
            for (long[] r : ranges) {
                int len = (int) r[1];
                ByteBuffer direct = ByteBuffer.allocateDirect(len + 3);
                direct.position(3);
                s.readTensorDirect(r[0], len, direct);
                assertEquals(len + 3, direct.position());
                byte[] got = new byte[len];
                direct.position(3);
                direct.get(got);
                assertArrayEquals("range " + r[0] + "+" + len,
                        Arrays.copyOfRange(expect, (int) r[0], (int) (r[0] + len)), got);

                byte[] arr = new byte[len + 2];
                s.readTensor(r[0], len, arr, 2);
                assertArrayEquals(Arrays.copyOfRange(expect, (int) r[0], (int) (r[0] + len)),
                        Arrays.copyOfRange(arr, 2, len + 2));
            }
        }
    }

    @Test
    public void aNonRowLocalTensorIsProducedOnceForByteSizedReads() throws IOException {
        derivedSource.reset();
        ProducedSegments p = layout.produced();
        long start = p.offset(0);
        int size = (int) p.tensors().get(0).byteSize();
        try (HfWeightStore s = store(Reader.RESIDENT_UPLOAD, Materialize.AUTO)) {
            byte[] one = new byte[1];
            for (int i = 0; i < size; i++) {
                s.readTensor(start + i, 1, one);
            }
        }
        long sourceBytes = derivedSource.rows() * derivedSource.rowBytes();
        assertEquals(sourceBytes, derivedSource.bytesRead());
    }

    @Test
    public void rangesOutsideTheAddressSpaceFail() throws IOException {
        try (HfWeightStore s = store(Reader.STREAMING, Materialize.AUTO)) {
            assertThrows(IOException.class, () -> s.readTensor(layout.totalBytes() - 1, 2, new byte[2]));
            assertThrows(IOException.class, () -> s.readTensor(-1, 1, new byte[1]));
            assertThrows(IllegalArgumentException.class, () -> s.readTensor(0, 3, new byte[2]));
        }
    }

    @Test
    public void closeIsIdempotentAndClosesTheView() throws IOException {
        ProducedSegments p = layout.produced();
        ProducedSegments.View view = p.view(MaterializePolicy.of(Reader.STREAMING, Materialize.MEMORY, 1 << 20,
                p.tensors(), Long.MAX_VALUE), sidecars);
        HfWeightStore s = new HfWeightStore(layout, view);
        s.close();
        s.close();
        assertThrows(IOException.class, () -> view.read(p.base(), 1, ByteBuffer.allocate(1)));
    }

    @Test
    public void aViewOfAnotherLayoutIsRejected() throws IOException {
        TensorPlanBuilder b = VirtualLayoutTest.builder();
        b.add("x.weight", new Counting(1, 2, 2));
        VirtualLayout other = VirtualLayout.of(ckpt, b.build());
        ProducedSegments p = other.produced();
        ProducedSegments.View view = p.view(MaterializePolicy.of(Reader.STREAMING, Materialize.AUTO, 1 << 20,
                p.tensors(), Long.MAX_VALUE), sidecars);
        assertThrows(IllegalArgumentException.class, () -> new HfWeightStore(layout, view));
        view.close();
    }
}
