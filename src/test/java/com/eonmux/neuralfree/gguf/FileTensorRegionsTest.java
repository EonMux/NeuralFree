package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link FileTensorRegions}: tensors found across the segments of several files, ranges outside a segment or its
 * file refused with the tensor named, and identical bytes in both {@link TensorReadMode}s.
 */
public class FileTensorRegionsTest {

    private static final TensorReadMode[] MODES = TensorReadMode.values();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A file of {@code header} filler bytes followed by {@code data} bytes valued {@code first, first + 1, ...}. */
    private Path file(String name, int header, int data, int first) throws IOException {
        byte[] bytes = new byte[header + data];
        Arrays.fill(bytes, 0, header, (byte) 0x7f);
        for (int i = 0; i < data; i++) {
            bytes[header + i] = (byte) (first + i);
        }
        Path p = tmp.getRoot().toPath().resolve(name);
        Files.write(p, bytes);
        return p;
    }

    /** Segment A: virtual [0, 32) at file offset 16; segment B: virtual [32, 56) at file offset 8. */
    private List<DataSegment> twoFiles() throws IOException {
        Path a = file("a.bin", 16, 32, 0);
        Path b = file("b.bin", 8, 24, 100);
        return Arrays.asList(new DataSegment(a.toString(), 16, 0, 32), new DataSegment(b.toString(), 8, 32, 24));
    }

    private static GGUFTensorInfo tensor(String name, long offset, long bytes) {
        return new GGUFTensorInfo(name, new long[]{bytes / 4}, GGMLType.F32, offset);
    }

    private static byte[] bytes(ByteBuffer buf) {
        assertEquals(ByteOrder.LITTLE_ENDIAN, buf.order());
        assertEquals(0, buf.position());
        byte[] out = new byte[buf.remaining()];
        buf.get(out);
        return out;
    }

    private static byte[] ramp(int first, int n) {
        byte[] out = new byte[n];
        for (int i = 0; i < n; i++) {
            out[i] = (byte) (first + i);
        }
        return out;
    }

    @Test
    public void findsTensorsInEverySegment() throws IOException {
        List<DataSegment> segments = twoFiles();
        for (TensorReadMode mode : MODES) {
            try (FileTensorRegions r = FileTensorRegions.open(segments, mode, "two-file model")) {
                assertEquals(mode, r.mode());
                assertEquals("two-file model", r.description());
                assertArrayEquals(mode.name(), ramp(0, 32), bytes(r.region(tensor("a", 0, 32), 0, 32)));
                assertArrayEquals(mode.name(), ramp(4, 8), bytes(r.region(tensor("a", 0, 32), 4, 8)));
                assertArrayEquals(mode.name(), ramp(100, 24), bytes(r.region(tensor("b", 32, 24), 0, 24)));
                assertArrayEquals(mode.name(), ramp(112, 8), bytes(r.region(tensor("b", 40, 16), 4, 8)));
            }
        }
    }

    @Test
    public void mappedRegionsTakeTheTensorAdviceAndReadRegionsNone() throws IOException {
        List<DataSegment> segments = twoFiles();
        String saved = System.getProperty(TensorAdvice.PROP_ADVICE);
        try {
            System.clearProperty(TensorAdvice.PROP_ADVICE);
            for (TensorReadMode mode : MODES) {
                try (FileTensorRegions r = FileTensorRegions.open(segments, mode, "two-file model")) {
                    assertEquals(mode.name(), mode.usesMmap() ? TensorAdvice.forMapping(System.getProperties(), 56)
                            : null, r.advice());
                    assertArrayEquals(mode.name(), ramp(0, 32), bytes(r.region(tensor("a", 0, 32), 0, 32)));
                }
            }
            System.setProperty(TensorAdvice.PROP_ADVICE, "sequential");
            try (FileTensorRegions r = FileTensorRegions.open(segments, TensorReadMode.MMAP, "two-file model")) {
                assertEquals(MemoryLock.Advice.SEQUENTIAL, r.advice());
                assertArrayEquals(ramp(100, 24), bytes(r.region(tensor("b", 32, 24), 0, 24)));
            }
            System.setProperty(TensorAdvice.PROP_ADVICE, "none");
            try (FileTensorRegions r = FileTensorRegions.open(segments, TensorReadMode.MMAP, "two-file model")) {
                assertEquals(null, r.advice());
            }
            System.setProperty(TensorAdvice.PROP_ADVICE, "bogus");
            assertThrows(IllegalArgumentException.class,
                    () -> FileTensorRegions.open(segments, TensorReadMode.MMAP, "two-file model"));
        } finally {
            if (saved == null) {
                System.clearProperty(TensorAdvice.PROP_ADVICE);
            } else {
                System.setProperty(TensorAdvice.PROP_ADVICE, saved);
            }
        }
    }

    /**
     * The kernel holds the advice of a mapped region: {@code random} and {@code sequential} show as the mapping's
     * {@code rr} and {@code sr} flags, {@code none} leaves the mapping without either.
     */
    @Test
    public void theKernelHoldsTheAdviceOfEveryMappedRegion() throws IOException {
        Assume.assumeTrue(com.sun.jna.Platform.isLinux() && MemoryLock.supported());
        Assume.assumeTrue(Files.isReadable(MappingFlags.SMAPS));
        String saved = System.getProperty(TensorAdvice.PROP_ADVICE);
        try {
            String[][] cases = {{"random", "rr", "sr"}, {"sequential", "sr", "rr"}, {"none", null, null}};
            for (int c = 0; c < cases.length; c++) {
                System.setProperty(TensorAdvice.PROP_ADVICE, cases[c][0]);
                Path a = file("advised" + c + ".bin", 16, 32, 0);
                List<DataSegment> segments = Collections.singletonList(new DataSegment(a.toString(), 16, 0, 32));
                for (TensorReadMode mode : new TensorReadMode[]{TensorReadMode.MMAP, TensorReadMode.MMAP_MLOCK}) {
                    try (FileTensorRegions r = FileTensorRegions.open(segments, mode, "advised model")) {
                        ByteBuffer region = r.region(tensor("a", 0, 32), 0, 32);
                        Set<String> flags = MappingFlags.of(a);
                        assertTrue(mode + " maps the file: " + flags, flags.contains("rd"));
                        if (cases[c][1] != null) {
                            assertTrue(cases[c][0] + " " + mode + ": " + flags, flags.contains(cases[c][1]));
                            assertFalse(cases[c][0] + " " + mode + ": " + flags, flags.contains(cases[c][2]));
                        } else {
                            assertFalse("none " + mode + ": " + flags, flags.contains("rr") || flags.contains("sr"));
                        }
                        assertArrayEquals(ramp(0, 32), bytes(region));
                    }
                }
            }
        } finally {
            if (saved == null) {
                System.clearProperty(TensorAdvice.PROP_ADVICE);
            } else {
                System.setProperty(TensorAdvice.PROP_ADVICE, saved);
            }
        }
    }

    @Test
    public void refusesRangesAcrossOrBeyondSegments() throws IOException {
        List<DataSegment> segments = twoFiles();
        for (TensorReadMode mode : MODES) {
            try (FileTensorRegions r = FileTensorRegions.open(segments, mode, "two-file model")) {
                IOException across = assertThrows(IOException.class, () -> r.region(tensor("x", 28, 8), 0, 8));
                assertEquals("tensor x bytes [28, +8) lie outside every data segment of two-file model",
                        across.getMessage());
                IOException beyond = assertThrows(IOException.class, () -> r.region(tensor("y", 56, 4), 0, 4));
                assertTrue(beyond.getMessage(), beyond.getMessage().startsWith("tensor y bytes [56, +4) lie outside"));
                IOException negative = assertThrows(IOException.class, () -> r.region(tensor("z", 0, 4), -1, 4));
                assertTrue(negative.getMessage(), negative.getMessage().contains("tensor z"));
                assertTrue(negative.getMessage(), negative.getMessage().contains("is not a valid byte range"));
            }
        }
    }

    @Test
    public void refusesSegmentsLongerThanTheirFile() throws IOException {
        Path a = file("short.bin", 4, 8, 0);
        List<DataSegment> segments = Collections.singletonList(new DataSegment(a.toString(), 4, 0, 64));
        for (TensorReadMode mode : MODES) {
            try (FileTensorRegions r = FileTensorRegions.open(segments, mode, "short model")) {
                assertArrayEquals(ramp(0, 8), bytes(r.region(tensor("ok", 0, 8), 0, 8)));
                IOException e = assertThrows(IOException.class, () -> r.region(tensor("t", 8, 16), 0, 16));
                assertEquals("tensor t bytes [12, +16) lie outside its file '" + a + "' of 12 bytes", e.getMessage());
            }
        }
    }

    @Test
    public void refusesMissingOrOverlappingSegments() throws IOException {
        assertThrows(IllegalArgumentException.class,
                () -> FileTensorRegions.open(Collections.emptyList(), TensorReadMode.MMAP, "m"));
        Path a = file("a.bin", 0, 32, 0);
        List<DataSegment> overlapping = Arrays.asList(new DataSegment(a.toString(), 0, 0, 32),
                new DataSegment(a.toString(), 0, 16, 16));
        assertThrows(IllegalArgumentException.class,
                () -> FileTensorRegions.open(overlapping, TensorReadMode.MMAP, "m"));
        List<DataSegment> missing = Collections.singletonList(
                new DataSegment(new File(tmp.getRoot(), "absent.bin").getPath(), 0, 0, 8));
        assertThrows(IOException.class, () -> FileTensorRegions.open(missing, TensorReadMode.READ, "m"));
    }

    @Test
    public void regionsFailAfterClose() throws IOException {
        FileTensorRegions r = FileTensorRegions.open(twoFiles(), TensorReadMode.READ, "m");
        r.close();
        assertThrows(IOException.class, () -> r.region(tensor("a", 0, 32), 0, 32));
    }

    @Test
    public void ggufRegionsMatchTheFileBytesOfEachTensor() throws IOException {
        GgufTestWriter w = new GgufTestWriter()
                .architecture("llama")
                .tensorRandom("token_embd.weight", GGMLType.Q8_0, 5L, -1f, 1f, 64, 3)
                .tensorRandom("output_norm.weight", GGMLType.F32, 6L, -1f, 1f, 64);
        Path path = tmp.getRoot().toPath().resolve("single.gguf");
        w.writeTo(path);
        byte[] file = Files.readAllBytes(path);
        try (GGUFFileParser parser = new GGUFFileParser(path.toString())) {
            assertEquals(Collections.singletonList(new DataSegment(path.toString(), w.dataSectionOffset(), 0,
                    parser.getDataLength())), parser.dataSegments());
            for (TensorReadMode mode : MODES) {
                try (TensorRegions r = parser.openRegions(mode)) {
                    for (String name : w.tensorNames()) {
                        GGUFTensorInfo info = parser.getTensor(name);
                        int start = (int) (w.dataSectionOffset() + w.tensorDataOffset(name));
                        int size = (int) info.sizeInBytes();
                        assertArrayEquals(name, Arrays.copyOfRange(file, start, start + size),
                                bytes(r.region(info, 0, size)));
                    }
                }
            }
        }
    }

    @Test
    public void splitGgufRegionsReadEachSplitFile() throws IOException {
        File dir = tmp.newFolder();
        for (int no = 0; no < 2; no++) {
            GGUFFileParserSplitTest.split(no).writeTo(Paths.get(GGUFFileParserSplitTest.splitName(dir, no)));
        }
        try (GGUFFileParser parser = new GGUFFileParser(GGUFFileParserSplitTest.splitName(dir, 0))) {
            List<DataSegment> segments = parser.dataSegments();
            assertEquals(2, segments.size());
            assertEquals(segments.get(0).virtualEnd(), segments.get(1).virtualBase());
            for (TensorReadMode mode : MODES) {
                try (TensorRegions r = parser.openRegions(mode)) {
                    GGUFTensorInfo q = parser.getTensor("blk.0.attn_q.weight");
                    ByteBuffer buf = r.region(q, 0, q.sizeInBytes());
                    float[] values = new float[GGUFFileParserSplitTest.Q.length];
                    buf.asFloatBuffer().get(values);
                    assertArrayEquals(mode.name(), GGUFFileParserSplitTest.Q, values, 0f);
                }
            }
        }
    }
}
