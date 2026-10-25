package com.eonmux.neuralfree.gguf.hf.store;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.Materialize;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicy.Reader;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicyTest.Counting;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
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
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfTensorRegions}: stored tensors come from the checkpoint files in either read mode, derived tensors from the
 * CPU reader's view.
 */
public class HfTensorRegionsTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final byte[] W = SyntheticHf.f32(1, 2, 3, 4, 5, 6);

    private CheckpointContainer ckpt;
    private Counting transposedSource;
    private Counting rowLocalSource;
    private VirtualLayout layout;
    private SidecarDirectory sidecars;

    @Before
    public void open() throws IOException {
        Path dir = SyntheticHf.builder().config("{}")
                .tensor("pad", StDType.U8, new long[]{3}, new byte[]{1, 2, 3})
                .tensor("w", StDType.F32, new long[]{2, 3}, W)
                .write(tmp.newFolder().toPath());
        ckpt = CheckpointContainers.open(dir, dir, false);
        transposedSource = new Counting(1, 2, 6, 4);
        rowLocalSource = new Counting(3, 4, 8);
        TensorPlanBuilder b = VirtualLayoutTest.builder();
        b.add("blk.0.attn_q.weight", ckpt.source("w"));
        b.add("blk.0.ffn_down_exps.weight", ExpertTransforms.transposeLast2(transposedSource));
        b.add("blk.0.attn_k.weight", rowLocalSource);
        layout = VirtualLayout.of(ckpt, b.build());
        sidecars = new SidecarDirectory(tmp.newFolder("cache").toPath());
    }

    @After
    public void close() throws IOException {
        sidecars.close();
        ckpt.close();
    }

    private HfTensorRegions regions(Materialize m, TensorReadMode mode) throws IOException {
        ProducedSegments p = layout.produced();
        return new HfTensorRegions(layout, p.view(MaterializePolicy.of(Reader.CPU_REGIONS, m, 1 << 20,
                p.tensors(), Long.MAX_VALUE), sidecars), mode, "test checkpoint");
    }

    private GGUFTensorInfo info(String name) {
        for (GGUFTensorInfo t : layout.tensors()) {
            if (t.name().equals(name)) {
                return t;
            }
        }
        throw new AssertionError(name);
    }

    private static byte[] bytes(ByteBuffer b) {
        byte[] out = new byte[b.remaining()];
        b.duplicate().get(out);
        return out;
    }

    @Test
    public void storedTensorsComeFromTheFilesInBothModes() throws IOException {
        for (TensorReadMode mode : TensorReadMode.values()) {
            try (HfTensorRegions r = regions(Materialize.AUTO, mode)) {
                GGUFTensorInfo q = info("blk.0.attn_q.weight");
                assertEquals(ckpt.directVirtualOffset("w"), q.offset());
                assertArrayEquals(mode.toString(), W, bytes(r.region(q, 0, W.length)));
                assertArrayEquals(Arrays.copyOfRange(W, 4, 16), bytes(r.region(q, 4, 12)));
                assertEquals("test checkpoint", r.description());
            }
        }
    }

    @Test
    public void derivedTensorsComeFromTheCpuView() throws IOException {
        for (Materialize m : new Materialize[]{Materialize.AUTO, Materialize.MEMORY, Materialize.DISK}) {
            transposedSource.reset();
            try (HfTensorRegions r = regions(m, TensorReadMode.MMAP)) {
                GGUFTensorInfo t = info("blk.0.ffn_down_exps.weight");
                assertEquals(GGMLType.F32, t.type());
                byte[] expect = MaterializePolicyTest.transposed(transposedSource);
                ByteBuffer first = r.region(t, 0, 40);
                ByteBuffer rest = r.region(t, 40, expect.length - 40);
                byte[] got = new byte[expect.length];
                System.arraycopy(bytes(first), 0, got, 0, 40);
                System.arraycopy(bytes(rest), 0, got, 40, expect.length - 40);
                assertArrayEquals(m.toString(), expect, got);
                long sourceBytes = transposedSource.rows() * transposedSource.rowBytes();
                assertTrue(m + ": " + transposedSource.bytesRead(), transposedSource.bytesRead()
                        <= sourceBytes * 11 / 10);

                GGUFTensorInfo k = info("blk.0.attn_k.weight");
                assertArrayEquals(Arrays.copyOfRange(rowLocalSource.bytes(), 32, 96), bytes(r.region(k, 32, 64)));
            }
        }
    }

    @Test
    public void regionsOutsideATensorFail() throws IOException {
        try (HfTensorRegions r = regions(Materialize.AUTO, TensorReadMode.READ)) {
            GGUFTensorInfo k = info("blk.0.attn_k.weight");
            assertThrows(IOException.class, () -> r.region(k, 100, 64));
            GGUFTensorInfo stray = new GGUFTensorInfo("stray.weight", new long[]{4}, GGMLType.F32,
                    layout.totalBytes() + 64);
            assertThrows(IOException.class, () -> r.region(stray, 0, 16));
        }
    }

    @Test
    public void noneIsRefusedForCpuRegions() {
        assertThrows(IllegalArgumentException.class, () -> regions(Materialize.NONE, TensorReadMode.MMAP));
    }

    @Test
    public void closeIsIdempotentAndClosesTheView() throws IOException {
        ProducedSegments p = layout.produced();
        ProducedSegments.View view = p.view(MaterializePolicy.of(Reader.CPU_REGIONS, Materialize.AUTO, 1 << 20,
                p.tensors(), Long.MAX_VALUE), sidecars);
        HfTensorRegions r = new HfTensorRegions(layout, view, TensorReadMode.MMAP, "x");
        r.close();
        r.close();
        assertThrows(IOException.class, () -> view.region(0, 0, 4));
    }
}
