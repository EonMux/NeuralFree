package com.eonmux.neuralfree.gguf.hf.store;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.Materialize;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicy.Decision;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicy.Mode;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicy.Reader;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The materialization table of the derived tensors: every reader and setting, the budget, and reads through a
 * {@link ProducedSegments.View} that produce a non-row-local tensor from at most 1.1 times its source bytes.
 */
public class MaterializePolicyTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final String NONE = "NONE/false";
    private static final String MEMORY_WINDOW = "MEMORY/false";
    private static final String MEMORY_KEPT = "MEMORY/true";
    private static final String DISK_WINDOW = "DISK/false";
    private static final String DISK_KEPT = "DISK/true";

    // ------------------------------------------------------------------ fixtures

    /**
     * A row-local F32 source of {@code shape} whose element {@code i} is {@code seed + i / 4}; it counts the bytes
     * read and can claim a stored virtual offset.
     */
    static final class Counting implements RowSource {
        private final long[] shape;
        private final float seed;
        private final long direct;
        private final AtomicLong bytesRead = new AtomicLong();
        private final AtomicLong calls = new AtomicLong();

        Counting(float seed, long... shape) {
            this(seed, -1L, shape);
        }

        private Counting(float seed, long direct, long[] shape) {
            this.shape = shape.clone();
            this.seed = seed;
            this.direct = direct;
        }

        /** A source that claims to be stored at virtual offset {@code direct}. */
        static Counting stored(long direct, float seed, long... shape) {
            return new Counting(seed, direct, shape);
        }

        static float value(float seed, long i) {
            return seed + i / 4f;
        }

        /** Every element as little-endian F32 bytes. */
        byte[] bytes() {
            ByteBuffer b = ByteBuffer.allocate((int) (rows() * rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
            for (long i = 0; i < rows() * cols(); i++) {
                b.putFloat(value(seed, i));
            }
            return b.array();
        }

        long cols() {
            return shape[shape.length - 1];
        }

        long bytesRead() {
            return bytesRead.get();
        }

        long calls() {
            return calls.get();
        }

        void reset() {
            bytesRead.set(0);
            calls.set(0);
        }

        @Override
        public String describe() {
            return "counting" + Arrays.toString(shape);
        }

        @Override
        public ElementFormat format() {
            return ElementFormat.of(StDType.F32);
        }

        @Override
        public long[] shape() {
            return shape.clone();
        }

        @Override
        public long rows() {
            long r = 1;
            for (int i = 0; i < shape.length - 1; i++) {
                r *= shape[i];
            }
            return r;
        }

        @Override
        public long rowBytes() {
            return 4 * cols();
        }

        @Override
        public boolean rowLocal() {
            return true;
        }

        @Override
        public long directVirtualOffset() {
            return direct;
        }

        @Override
        public void readRows(long first, int count, ByteBuffer dst) throws IOException {
            if (first < 0 || count < 0 || first + count > rows()) {
                throw new IOException("rows [" + first + ", +" + count + ") outside " + rows());
            }
            ByteBuffer out = dst.slice().order(ByteOrder.LITTLE_ENDIAN);
            long c = cols();
            for (long i = first * c; i < (first + count) * c; i++) {
                out.putFloat(value(seed, i));
            }
            dst.position(dst.position() + out.position());
            bytesRead.addAndGet(count * rowBytes());
            calls.incrementAndGet();
        }
    }

    /** A planned tensor of {@code src} (F32). */
    static PlannedTensor planned(String name, RowSource src) {
        return new PlannedTensor(name, src, GGMLType.F32, PlannedTensor.neOf(src.shape()),
                src.rows() * src.rowBytes());
    }

    /** The bytes of the transpose of the last two dimensions of {@code src}'s values. */
    static byte[] transposed(Counting src) {
        long[] s = src.shape();
        int batches = (int) (src.rows() / s[s.length - 2]);
        int r = (int) s[s.length - 2];
        int c = (int) s[s.length - 1];
        ByteBuffer b = ByteBuffer.allocate(batches * r * c * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int m = 0; m < batches; m++) {
            for (int j = 0; j < c; j++) {
                for (int i = 0; i < r; i++) {
                    b.putFloat(Counting.value(src.seed, (long) m * r * c + (long) i * c + j));
                }
            }
        }
        return b.array();
    }

    /** {@code d} as {@code MODE/keep}. */
    private static String code(Decision d) {
        return d.mode() + "/" + d.keep();
    }

    private static List<PlannedTensor> tensors(PlannedTensor... t) {
        return Arrays.asList(t);
    }

    private static PlannedTensor rowLocal(long rows, long cols) {
        return planned("blk.0.row_local.weight", new Counting(0, rows, cols));
    }

    private static PlannedTensor nonRowLocal(long batches, long rows, long cols) {
        return planned("blk.0.transposed.weight",
                ExpertTransforms.transposeLast2(new Counting(1, batches, rows, cols)));
    }

    private static List<String> decide(Reader reader, Materialize setting, long budget, long headroom,
                                         List<PlannedTensor> t) {
        MaterializePolicy p = MaterializePolicy.of(reader, setting, budget, t, headroom);
        List<String> out = new ArrayList<>();
        for (int i = 0; i < p.size(); i++) {
            out.add(code(p.decision(i)));
        }
        return out;
    }

    // ------------------------------------------------------------------ the table

    @Test
    public void residentUploadProducesRowLocalTensorsAndWindowsTheOthers() {
        PlannedTensor rl = rowLocal(4, 8);
        PlannedTensor small = nonRowLocal(1, 4, 8);
        PlannedTensor big = nonRowLocal(2, 64, 64);
        List<PlannedTensor> all = tensors(rl, small, big);
        long budget = small.byteSize();
        assertEquals(Arrays.asList(NONE, MEMORY_WINDOW, DISK_WINDOW),
                decide(Reader.RESIDENT_UPLOAD, Materialize.AUTO, budget, Long.MAX_VALUE, all));
        assertEquals("explicit NONE keeps the window of a non-row-local producer",
                Arrays.asList(NONE, MEMORY_WINDOW, DISK_WINDOW),
                decide(Reader.RESIDENT_UPLOAD, Materialize.NONE, budget, Long.MAX_VALUE, all));
        assertEquals(Arrays.asList(MEMORY_WINDOW, MEMORY_WINDOW, MEMORY_WINDOW),
                decide(Reader.RESIDENT_UPLOAD, Materialize.MEMORY, budget, Long.MAX_VALUE, all));
        assertEquals(Arrays.asList(DISK_WINDOW, DISK_WINDOW, DISK_WINDOW),
                decide(Reader.RESIDENT_UPLOAD, Materialize.DISK, budget, Long.MAX_VALUE, all));
    }

    @Test
    public void streamingKeepsDiskCopiesUnderAuto() {
        PlannedTensor rl = rowLocal(4, 8);
        PlannedTensor small = nonRowLocal(1, 4, 8);
        PlannedTensor big = nonRowLocal(2, 64, 64);
        List<PlannedTensor> all = tensors(rl, small, big);
        long budget = small.byteSize();
        assertEquals(Arrays.asList(DISK_KEPT, DISK_KEPT, DISK_KEPT),
                decide(Reader.STREAMING, Materialize.AUTO, budget, Long.MAX_VALUE, all));
        assertEquals(Arrays.asList(MEMORY_KEPT, MEMORY_KEPT, MEMORY_KEPT),
                decide(Reader.STREAMING, Materialize.MEMORY, budget, Long.MAX_VALUE, all));
        assertEquals(Arrays.asList(DISK_KEPT, DISK_KEPT, DISK_KEPT),
                decide(Reader.STREAMING, Materialize.DISK, budget, Long.MAX_VALUE, all));
        assertEquals(Arrays.asList(NONE, MEMORY_WINDOW, DISK_WINDOW),
                decide(Reader.STREAMING, Materialize.NONE, budget, Long.MAX_VALUE, all));
    }

    @Test
    public void cpuRegionsUseMemoryWhenEverythingFitsTheBudget() {
        PlannedTensor rl = rowLocal(4, 8);
        PlannedTensor small = nonRowLocal(1, 4, 8);
        List<PlannedTensor> all = tensors(rl, small);
        long total = rl.byteSize() + small.byteSize();
        assertEquals(Arrays.asList(MEMORY_WINDOW, MEMORY_WINDOW),
                decide(Reader.CPU_REGIONS, Materialize.AUTO, total, Long.MAX_VALUE, all));
        assertEquals(Arrays.asList(DISK_KEPT, DISK_KEPT),
                decide(Reader.CPU_REGIONS, Materialize.AUTO, total - 1, Long.MAX_VALUE, all));
        assertEquals(Arrays.asList(MEMORY_WINDOW, MEMORY_WINDOW),
                decide(Reader.CPU_REGIONS, Materialize.MEMORY, 0, Long.MAX_VALUE, all));
        assertEquals(Arrays.asList(DISK_KEPT, DISK_KEPT),
                decide(Reader.CPU_REGIONS, Materialize.DISK, total, Long.MAX_VALUE, all));
    }

    @Test
    public void noneIsInvalidForCpuRegions() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> MaterializePolicy.of(
                Reader.CPU_REGIONS, Materialize.NONE, 1 << 20, tensors(rowLocal(2, 8)), Long.MAX_VALUE));
        assertTrue(e.getMessage(), e.getMessage().contains(ModelSourceOptions.PROP_MATERIALIZE));
        ModelSourceOptions o = ModelSourceOptions.builder().materialize(Materialize.NONE).build();
        assertThrows(IllegalArgumentException.class,
                () -> MaterializePolicy.of(Reader.CPU_REGIONS, o, tensors(rowLocal(2, 8))));
    }

    @Test
    public void theBudgetIsTheSmallerOfTheSettingAndTheDirectHeadroom() {
        PlannedTensor small = nonRowLocal(1, 4, 8);
        MaterializePolicy p = MaterializePolicy.of(Reader.RESIDENT_UPLOAD, Materialize.AUTO, 1 << 20,
                tensors(small), small.byteSize() - 1);
        assertEquals(small.byteSize() - 1, p.budget());
        assertEquals("the headroom bounds the window", DISK_WINDOW, code(p.decision(0)));
        assertEquals(1 << 20, MaterializePolicy.of(Reader.RESIDENT_UPLOAD, Materialize.AUTO, 1 << 20,
                tensors(small), Long.MAX_VALUE).budget());
        assertTrue(MaterializePolicy.directHeadroom() >= 0);
        assertThrows(IllegalArgumentException.class, () -> MaterializePolicy.of(Reader.STREAMING, Materialize.AUTO,
                -1, tensors(small), 0));
    }

    @Test
    public void storeUsesMapToReaders() {
        assertEquals(Reader.RESIDENT_UPLOAD, MaterializePolicy.readerOf(StoreUse.RESIDENT_UPLOAD));
        assertEquals(Reader.STREAMING, MaterializePolicy.readerOf(StoreUse.STREAMING));
    }

    // ------------------------------------------------------------------ reads through a view

    /** Reads every byte of tensor 0 of {@code view} in unaligned pieces of {@code step} bytes. */
    private static byte[] readAll(ProducedSegments.View view, int step) throws IOException {
        ProducedSegments s = view.segments();
        int n = (int) s.tensors().get(0).byteSize();
        ByteBuffer out = ByteBuffer.allocate(n);
        for (int off = 0; off < n; off += step) {
            view.read(s.offset(0) + off, Math.min(step, n - off), out);
        }
        return out.array();
    }

    /** Reads every byte of tensor 0 of {@code view} as regions of {@code step} bytes. */
    private static byte[] regionsAll(ProducedSegments.View view, int step) throws IOException {
        int n = (int) view.segments().tensors().get(0).byteSize();
        ByteBuffer out = ByteBuffer.allocate(n);
        for (int off = 0; off < n; off += step) {
            ByteBuffer r = view.region(0, off, Math.min(step, n - off));
            assertEquals(ByteOrder.LITTLE_ENDIAN, r.order());
            out.put(r);
        }
        return out.array();
    }

    private ProducedSegments table(PlannedTensor t) {
        return ProducedSegments.of(Collections.singletonList(t), Collections.singletonList(0L), 0, t.byteSize());
    }

    private void assertBounded(Counting src, String what) {
        long sourceBytes = src.rows() * src.rowBytes();
        assertTrue(what + ": read " + src.bytesRead() + " source bytes for " + sourceBytes,
                src.bytesRead() <= sourceBytes * 11 / 10);
    }

    /** One non-row-local tensor read whole, in pieces, under {@code reader} and {@code setting}. */
    private void readsNonRowLocalOnce(Reader reader, Materialize setting, boolean regions) throws IOException {
        Counting src = new Counting(3, 3, 24, 40);
        PlannedTensor t = planned("blk.0.ffn_down_exps.weight", ExpertTransforms.transposeLast2(src));
        ProducedSegments s = table(t);
        try (SidecarDirectory dir = new SidecarDirectory(tmp.newFolder().toPath());
             ProducedSegments.View view = s.view(MaterializePolicy.of(reader, setting, 1 << 20, s.tensors(),
                     Long.MAX_VALUE), dir)) {
            byte[] got = regions ? regionsAll(view, 100) : readAll(view, 37);
            assertArrayEquals(reader + "/" + setting, transposed(src), got);
            assertBounded(src, reader + "/" + setting);
            if (view.policy().decision(0).mode() == Mode.DISK) {
                assertTrue("a DISK copy lives in the sidecar directory", dir.directory() != null);
            }
        }
    }

    @Test
    public void residentUploadReadsANonRowLocalProducerOnce() throws IOException {
        readsNonRowLocalOnce(Reader.RESIDENT_UPLOAD, Materialize.AUTO, false);
        readsNonRowLocalOnce(Reader.RESIDENT_UPLOAD, Materialize.NONE, false);
        readsNonRowLocalOnce(Reader.RESIDENT_UPLOAD, Materialize.MEMORY, false);
        readsNonRowLocalOnce(Reader.RESIDENT_UPLOAD, Materialize.DISK, false);
    }

    @Test
    public void streamingReadsANonRowLocalProducerOnce() throws IOException {
        readsNonRowLocalOnce(Reader.STREAMING, Materialize.AUTO, false);
        readsNonRowLocalOnce(Reader.STREAMING, Materialize.NONE, false);
        readsNonRowLocalOnce(Reader.STREAMING, Materialize.MEMORY, false);
        readsNonRowLocalOnce(Reader.STREAMING, Materialize.DISK, false);
    }

    @Test
    public void cpuRegionsReadANonRowLocalProducerOnce() throws IOException {
        readsNonRowLocalOnce(Reader.CPU_REGIONS, Materialize.AUTO, true);
        readsNonRowLocalOnce(Reader.CPU_REGIONS, Materialize.MEMORY, true);
        readsNonRowLocalOnce(Reader.CPU_REGIONS, Materialize.DISK, true);
    }

    @Test
    public void aDiskWindowReadsANonRowLocalProducerOnceWhenOverTheBudget() throws IOException {
        Counting src = new Counting(5, 2, 32, 16);
        PlannedTensor t = planned("blk.1.ffn_down_exps.weight", ExpertTransforms.transposeLast2(src));
        ProducedSegments s = table(t);
        MaterializePolicy p = MaterializePolicy.of(Reader.RESIDENT_UPLOAD, Materialize.AUTO, t.byteSize() - 1,
                s.tensors(), Long.MAX_VALUE);
        assertEquals(DISK_WINDOW, code(p.decision(0)));
        try (SidecarDirectory dir = new SidecarDirectory(tmp.newFolder().toPath());
             ProducedSegments.View view = s.view(p, dir)) {
            assertArrayEquals(transposed(src), readAll(view, 29));
            assertBounded(src, "DISK window");
        }
    }

    @Test
    public void rowLocalNoneReadsOnlyTheCoveringRows() throws IOException {
        Counting src = new Counting(2, 10, 16);
        PlannedTensor t = planned("blk.0.attn_q.weight", src);
        ProducedSegments s = table(t);
        try (SidecarDirectory dir = new SidecarDirectory(tmp.newFolder().toPath());
             ProducedSegments.View view = s.view(MaterializePolicy.of(Reader.RESIDENT_UPLOAD, Materialize.AUTO,
                     1 << 20, s.tensors(), Long.MAX_VALUE), dir)) {
            assertEquals(NONE, code(view.policy().decision(0)));
            ByteBuffer out = ByteBuffer.allocate(70);
            view.read(70, 70, out);
            assertArrayEquals(Arrays.copyOfRange(src.bytes(), 70, 140), out.array());
            assertEquals("rows 1 and 2 cover bytes [70, 140)", 2 * src.rowBytes(), src.bytesRead());
            assertTrue("nothing materializes", dir.directory() == null);
        }
    }

    @Test
    public void cpuMemoryRegionsOfARowLocalTensorAreProducedPerRegion() throws IOException {
        Counting src = new Counting(4, 12, 8);
        PlannedTensor t = planned("blk.0.attn_k.weight", src);
        ProducedSegments s = table(t);
        try (SidecarDirectory dir = new SidecarDirectory(tmp.newFolder().toPath());
             ProducedSegments.View view = s.view(MaterializePolicy.of(Reader.CPU_REGIONS, Materialize.AUTO, 1 << 20,
                     s.tensors(), Long.MAX_VALUE), dir)) {
            assertEquals(MEMORY_WINDOW, code(view.policy().decision(0)));
            ByteBuffer a = view.region(0, 64, 96);
            assertEquals("only rows 2..4 are produced", 3 * src.rowBytes(), src.bytesRead());
            ByteBuffer b = view.region(0, 64, 96);
            assertNotSame("every region is its own buffer", a, b);
            byte[] got = new byte[96];
            a.get(got);
            assertArrayEquals(Arrays.copyOfRange(src.bytes(), 64, 160), got);
            assertTrue(a.isDirect());
        }
    }

    @Test
    public void keptCopiesAreProducedOnceAcrossReads() throws IOException {
        Counting src = new Counting(6, 4, 16);
        PlannedTensor t = planned("blk.0.attn_v.weight", src);
        ProducedSegments s = table(t);
        try (SidecarDirectory dir = new SidecarDirectory(tmp.newFolder().toPath());
             ProducedSegments.View view = s.view(MaterializePolicy.of(Reader.STREAMING, Materialize.AUTO, 1 << 20,
                     s.tensors(), Long.MAX_VALUE), dir)) {
            assertEquals(DISK_KEPT, code(view.policy().decision(0)));
            for (int pass = 0; pass < 3; pass++) {
                assertArrayEquals(src.bytes(), readAll(view, 24));
            }
            assertEquals(src.rows() * src.rowBytes(), src.bytesRead());
            Path run = dir.directory();
            assertTrue(Files.isDirectory(run));
        }
    }

    @Test
    public void aClosedViewRefusesReads() throws IOException {
        Counting src = new Counting(0, 2, 8);
        ProducedSegments s = table(planned("blk.0.x.weight", src));
        SidecarDirectory dir = new SidecarDirectory(tmp.newFolder().toPath());
        ProducedSegments.View view = s.view(MaterializePolicy.of(Reader.STREAMING, Materialize.MEMORY, 1 << 20,
                s.tensors(), Long.MAX_VALUE), dir);
        view.close();
        view.close();
        assertThrows(IOException.class, () -> view.read(0, 4, ByteBuffer.allocate(4)));
        dir.close();
        assertFalse(view.toString().isEmpty());
    }
}
