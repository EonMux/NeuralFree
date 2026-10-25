package com.eonmux.neuralfree.io.safetensors;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.io.safetensors.SafetensorsTestFiles.ramp;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SafetensorsWriter}: the reference header bytes and tensor order, space padding to 8 bytes, a round trip
 * through {@link SafetensorsFile}, the atomic rename, and builder validation.
 */
public class SafetensorsWriterTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path file(String name) {
        return tmp.getRoot().toPath().resolve(name);
    }

    @Test
    public void writesTheReferenceHeaderPaddedWithSpaces() throws IOException {
        Path p = file("ref.safetensors");
        byte[] data = ramp(1, 16);
        SafetensorsWriter.builder().add("test", StDType.I32, new long[]{2, 2}, data).build().write(p);
        byte[] bytes = Files.readAllBytes(p);
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        long n = buf.getLong(0);
        String json = "{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}";
        assertEquals(60, json.length());
        assertEquals(64, n);
        assertEquals(json + "    ", new String(bytes, 8, (int) n, StandardCharsets.UTF_8));
        assertArrayEquals(data, Arrays.copyOfRange(bytes, 8 + (int) n, bytes.length));
        assertEquals(8 + 64 + 16, bytes.length);
    }

    @Test
    public void ordersTensorsByDescendingDtypeAlignmentThenName() throws IOException {
        SafetensorsWriter w = SafetensorsWriter.builder()
                .add("b", StDType.F32, new long[]{1}, ramp(0, 4))
                .add("a", StDType.F16, new long[]{3}, ramp(10, 6))
                .add("c", StDType.F32, new long[]{2}, ramp(20, 8))
                .add("d", StDType.U8, new long[]{3}, ramp(30, 3))
                .add("e", StDType.I64, new long[]{1}, ramp(40, 8))
                .add("f", StDType.BF16, new long[]{1}, ramp(50, 2))
                .metadata("format", "pt")
                .metadata("author", "nf")
                .build();
        List<String> order = new ArrayList<>();
        List<Long> begins = new ArrayList<>();
        for (StTensor t : w.layout()) {
            order.add(t.name());
            begins.add(t.begin());
        }
        assertEquals(Arrays.asList("e", "b", "c", "f", "a", "d"), order);
        assertEquals(Arrays.asList(0L, 8L, 12L, 20L, 22L, 28L), begins);
        assertEquals(31, w.dataLength());
        assertTrue(w.headerJson().startsWith("{\"__metadata__\":{\"author\":\"nf\",\"format\":\"pt\"},\"e\":"));

        Path p = file("order.safetensors");
        w.write(p);
        try (SafetensorsFile f = SafetensorsFile.open(p)) {
            assertEquals(0, f.header().headerLength() % SafetensorsWriter.HEADER_ALIGNMENT);
            assertEquals(31, f.dataLength());
            assertEquals("nf", f.metadata().get("author"));
            for (StTensor t : w.layout()) {
                assertEquals(t, f.tensor(t.name()));
            }
            assertArrayEquals(ramp(20, 8), bytes(f.readAll(f.tensor("c"))));
            assertArrayEquals(ramp(30, 3), bytes(f.readAll(f.tensor("d"))));
        }
    }

    @Test
    public void ordersNamesByCodePoint() {
        String bmpHigh = "\uFFFF";
        String astral = "\uD83D\uDE00";
        List<String> names = new ArrayList<>(Arrays.asList(astral, bmpHigh, "a"));
        names.sort(SafetensorsHeader.NAME_ORDER);
        assertEquals(Arrays.asList("a", bmpHigh, astral), names);
        assertTrue(astral.compareTo(bmpHigh) < 0);
    }

    @Test
    public void suppliedBuffersAreWrittenWithoutMovingThem() throws IOException {
        ByteBuffer src = ByteBuffer.wrap(ramp(5, 12));
        Path p = file("supplied.safetensors");
        SafetensorsWriter.builder().add("x", StDType.F32, new long[]{3}, () -> src).build().write(p);
        assertEquals(0, src.position());
        try (SafetensorsFile f = SafetensorsFile.open(p)) {
            assertArrayEquals(ramp(5, 12), bytes(f.readAll(f.tensor("x"))));
        }
    }

    @Test
    public void replacesTheTargetAtomicallyAndLeavesNoTemporaryFile() throws IOException {
        Path p = file("model.safetensors");
        SafetensorsWriter.builder().add("old", StDType.U8, new long[]{1}, new byte[]{1}).build().write(p);
        SafetensorsWriter.builder().add("new", StDType.U8, new long[]{2}, new byte[]{2, 3}).build().write(p);
        try (SafetensorsFile f = SafetensorsFile.open(p)) {
            assertEquals(Collections.singletonList("new"), names(f));
        }
        assertFalse(Files.exists(file("model.safetensors" + SafetensorsWriter.TEMP_SUFFIX)));
        assertEquals(Collections.singletonList("model.safetensors"), listing());
    }

    @Test
    public void failedWriteKeepsTheOldFileAndRemovesTheTemporaryFile() throws IOException {
        Path p = file("keep.safetensors");
        SafetensorsWriter.builder().add("old", StDType.U8, new long[]{1}, new byte[]{9}).build().write(p);
        byte[] before = Files.readAllBytes(p);
        SafetensorsWriter bad = SafetensorsWriter.builder()
                .add("short", StDType.F32, new long[]{4}, () -> ByteBuffer.allocate(8))
                .build();
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> bad.write(p));
        assertTrue(e.getMessage(), e.getMessage().contains("tensor 'short'"));
        assertArrayEquals(before, Files.readAllBytes(p));
        assertEquals(Collections.singletonList("keep.safetensors"), listing());
    }

    @Test
    public void writesAnEmptyFile() throws IOException {
        Path p = file("empty.safetensors");
        SafetensorsWriter.builder().build().write(p);
        try (SafetensorsFile f = SafetensorsFile.open(p)) {
            assertEquals(0, f.tensors().size());
            assertEquals(8, f.header().headerLength());
            assertEquals(16, Files.size(p));
        }
    }

    @Test
    public void builderRejectsInvalidTensors() {
        SafetensorsWriter.Builder b = SafetensorsWriter.builder().add("x", StDType.U8, new long[]{1}, new byte[1]);
        assertThrows(IllegalArgumentException.class, () -> b.add("x", StDType.U8, new long[]{1}, new byte[1]));
        assertThrows(IllegalArgumentException.class,
                () -> b.add(SafetensorsHeader.METADATA_KEY, StDType.U8, new long[]{1}, new byte[1]));
        assertThrows(IllegalArgumentException.class, () -> b.add("y", StDType.F32, new long[]{2}, new byte[4]));
        assertThrows(IllegalArgumentException.class,
                () -> b.add("z", StDType.U8, new long[]{1, 1, 1, 1, 1, 1, 1, 1, 1}, new byte[1]));
        assertThrows(IllegalArgumentException.class, () -> b.add("w", StDType.U8, new long[]{-1}, new byte[0]));
        assertThrows(IllegalArgumentException.class, () -> b.add("v", StDType.F4, new long[]{3}, new byte[2]));
        assertThrows(IllegalArgumentException.class, () -> b.add("\uD800", StDType.U8, new long[]{1}, new byte[1]));
    }

    @Test
    public void addCopiesTheGivenBytes() throws IOException {
        byte[] data = {1, 2, 3, 4};
        SafetensorsWriter w = SafetensorsWriter.builder().add("x", StDType.U8, new long[]{4}, data).build();
        data[0] = 99;
        Path p = file("copy.safetensors");
        w.write(p);
        try (SafetensorsFile f = SafetensorsFile.open(p)) {
            assertArrayEquals(new byte[]{1, 2, 3, 4}, bytes(f.readAll(f.tensor("x"))));
        }
    }

    private List<String> listing() throws IOException {
        List<String> out = new ArrayList<>();
        try (java.util.stream.Stream<Path> s = Files.list(tmp.getRoot().toPath())) {
            s.forEach(x -> out.add(x.getFileName().toString()));
        }
        Collections.sort(out);
        return out;
    }

    private static List<String> names(SafetensorsFile f) {
        List<String> out = new ArrayList<>();
        for (StTensor t : f.tensors()) {
            out.add(t.name());
        }
        return out;
    }

    private static byte[] bytes(ByteBuffer b) {
        byte[] out = new byte[b.remaining()];
        b.get(out);
        return out;
    }
}
