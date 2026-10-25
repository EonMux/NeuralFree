package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.io.safetensors.SafetensorsTestFiles.ramp;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SafetensorsCheckpoint} discovery (index, globs, Mistral native, the unlisted MTP file), its index
 * cross-checks, virtual bases, side files, natural file order, {@link CheckpointContainers} dispatch, and the
 * {@link HfTensorIndex} built over a checkpoint.
 */
public class SafetensorsCheckpointTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path dir() {
        return tmp.getRoot().toPath();
    }

    /** Writes {@code name} with U8 tensors of the given names and byte lengths (values from {@code first}). */
    private Path shard(String name, int first, Object... nameAndLength) throws IOException {
        SafetensorsWriter.Builder b = SafetensorsWriter.builder();
        for (int i = 0; i < nameAndLength.length; i += 2) {
            int len = (Integer) nameAndLength[i + 1];
            b.add((String) nameAndLength[i], StDType.U8, new long[]{len}, ramp(first + 16 * i, len));
        }
        Path p = dir().resolve(name);
        Files.createDirectories(p.getParent());
        b.build().write(p);
        return p;
    }

    private void index(String name, String... tensorAndFile) throws IOException {
        StringBuilder sb = new StringBuilder("{\"metadata\":{\"total_size\":0},\"weight_map\":{");
        for (int i = 0; i < tensorAndFile.length; i += 2) {
            sb.append(i == 0 ? "" : ",").append('"').append(tensorAndFile[i]).append("\":\"")
                    .append(tensorAndFile[i + 1]).append('"');
        }
        Files.write(dir().resolve(name), sb.append("}}").toString().getBytes(StandardCharsets.UTF_8));
    }

    private static List<String> names(CheckpointContainer c) {
        List<String> out = new ArrayList<>();
        for (StTensor t : c.tensors()) {
            out.add(t.name());
        }
        return out;
    }

    private static void assertMessage(IOException e, String prefix, String fragment) {
        assertTrue(e.getMessage(), e.getMessage().startsWith(prefix));
        assertTrue(e.getMessage(), e.getMessage().contains(fragment));
    }

    // ------------------------------------------------------------------ files and virtual layout

    @Test
    public void singleFileIsOneSegmentAtZero() throws IOException {
        Path p = shard("model.safetensors", 0, "w", 5, "b", 3);
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(p, dir(), false)) {
            assertEquals(Arrays.asList("b", "w"), names(c));
            assertEquals(1, c.segments().size());
            DataSegment s = c.segments().get(0);
            assertEquals(0, s.virtualBase());
            assertEquals(8, s.dataLength());
            assertEquals(c.files().get(0).dataStart(), s.fileDataOffset());
            assertEquals(c.tensor("w").begin(), c.directVirtualOffset("w"));
            assertNull(c.tensor("missing"));
            assertThrows(IllegalArgumentException.class, () -> c.source("missing"));
            assertThrows(IllegalArgumentException.class, () -> c.directVirtualOffset("missing"));
            assertTrue(c.description().contains("1 file, 2 tensors"));
        }
    }

    @Test
    public void filesWithoutIndexAreNaturallyOrderedWith64ByteAlignedBases() throws IOException {
        shard("model-10.safetensors", 0, "t10", 3);
        shard("model-9.safetensors", 0, "t9", 70);
        shard("model-1.safetensors", 0, "t1", 5);
        shard("other.safetensors", 0, "ignored", 1);
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(dir(), dir(), false)) {
            assertEquals(Arrays.asList("t1", "t9", "t10"), names(c));
            List<DataSegment> s = c.segments();
            assertEquals(3, s.size());
            assertEquals(0, s.get(0).virtualBase());
            assertEquals(64, s.get(1).virtualBase());
            assertEquals(64 + 128, s.get(2).virtualBase());
            assertEquals(64 + c.tensor("t9").begin(), c.directVirtualOffset("t9"));
            assertEquals(192, c.source("t10").directVirtualOffset());
            assertTrue(s.get(0).path().endsWith("model-1.safetensors"));
        }
    }

    @Test
    public void withoutModelFilesEveryFileButAPeftAdapterIsUsed() throws IOException {
        shard("weights-b.safetensors", 0, "b", 1);
        shard("weights-a.safetensors", 0, "a", 1);
        shard(SafetensorsCheckpoint.ADAPTER_MODEL, 0, "lora", 1);
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(dir(), dir(), false)) {
            assertEquals(Arrays.asList("a", "b"), names(c));
        }
    }

    @Test
    public void mistralNativeUsesConsolidatedFiles() throws IOException {
        shard("model-00001-of-00001.safetensors", 0, "model.layers.0.w", 2);
        shard("consolidated.safetensors", 0, "layers.0.attention.wq.weight", 2);
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(dir(), dir(), true)) {
            assertEquals(Collections.singletonList("layers.0.attention.wq.weight"), names(c));
        }
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(dir(), dir(), false)) {
            assertEquals(Collections.singletonList("model.layers.0.w"), names(c));
        }
    }

    @Test
    public void tensorInTwoFilesIsRejected() throws IOException {
        shard("model-1.safetensors", 0, "dup", 1, "x", 1);
        shard("model-2.safetensors", 0, "dup", 1);
        IOException e = assertThrows(IOException.class, () -> SafetensorsCheckpoint.open(dir(), dir(), false));
        assertMessage(e, "Invalid Hugging Face checkpoint '" + dir() + "': ", "tensor 'dup' is stored in both");
    }

    @Test
    public void directoryWithoutSafetensorsIsRejected() throws IOException {
        Files.write(dir().resolve("config.json"), "{}".getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> SafetensorsCheckpoint.open(dir(), dir(), false));
        assertMessage(e, "Invalid Hugging Face checkpoint '", "no model *.safetensors files");
        assertThrows(FileNotFoundException.class,
                () -> SafetensorsCheckpoint.open(dir().resolve("absent"), dir(), false));
    }

    // ------------------------------------------------------------------ index

    @Test
    public void indexSelectsItsFilesAndIgnoresOthers() throws IOException {
        shard("model-00001-of-00002.safetensors", 0, "a", 2, "extra", 1);
        shard("model-00002-of-00002.safetensors", 0, "b", 2);
        shard("model-stale.safetensors", 0, "stale", 2);
        index(SafetensorsCheckpoint.INDEX, "a", "model-00001-of-00002.safetensors",
                "b", "model-00002-of-00002.safetensors");
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(dir(), dir(), false)) {
            assertEquals(Arrays.asList("a", "extra", "b"), names(c));
            assertEquals(2, c.files().size());
        }
    }

    @Test
    public void indexNamesOfOneFileOpenItOnce() throws IOException {
        shard("model-00001-of-00001.safetensors", 0, "a", 2, "b", 2);
        index(SafetensorsCheckpoint.INDEX, "a", "./model-00001-of-00001.safetensors",
                "b", "model-00001-of-00001.safetensors");
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(dir(), dir(), false)) {
            assertEquals(1, c.files().size());
            assertEquals(Arrays.asList("a", "b"), names(c));
        }
    }

    @Test
    public void indexNamingAMissingFileIsRejected() throws IOException {
        shard("model-00001-of-00002.safetensors", 0, "a", 2);
        index(SafetensorsCheckpoint.INDEX, "a", "model-00001-of-00002.safetensors",
                "b", "model-00002-of-00002.safetensors");
        IOException e = assertThrows(IOException.class, () -> SafetensorsCheckpoint.open(dir(), dir(), false));
        assertMessage(e, "Invalid safetensors file '" + dir().resolve(SafetensorsCheckpoint.INDEX) + "': ",
                "maps tensor 'b' to 'model-00002-of-00002.safetensors', which does not exist");
    }

    @Test
    public void indexMappingATensorToTheWrongFileIsRejected() throws IOException {
        shard("model-00001-of-00002.safetensors", 0, "a", 2);
        shard("model-00002-of-00002.safetensors", 0, "b", 2);
        index(SafetensorsCheckpoint.INDEX, "a", "model-00002-of-00002.safetensors",
                "b", "model-00002-of-00002.safetensors");
        IOException e = assertThrows(IOException.class, () -> SafetensorsCheckpoint.open(dir(), dir(), false));
        assertMessage(e, "Invalid safetensors file '", "maps tensor 'a'");
        assertTrue(e.getMessage(), e.getMessage().contains("which does not hold it"));
    }

    @Test
    public void indexWithATensorInTwoFilesIsRejected() throws IOException {
        shard("model-00001-of-00002.safetensors", 0, "a", 2, "dup", 1);
        shard("model-00002-of-00002.safetensors", 0, "b", 2, "dup", 1);
        index(SafetensorsCheckpoint.INDEX, "a", "model-00001-of-00002.safetensors",
                "b", "model-00002-of-00002.safetensors", "dup", "model-00001-of-00002.safetensors");
        IOException e = assertThrows(IOException.class, () -> SafetensorsCheckpoint.open(dir(), dir(), false));
        assertMessage(e, "Invalid Hugging Face checkpoint '", "tensor 'dup' is stored in both");
    }

    @Test
    public void unlistedMtpFileIsAdded() throws IOException {
        shard("model-00001-of-00001.safetensors", 0, "a", 2);
        shard(SafetensorsCheckpoint.MTP_FILE, 0, "model.layers.4.eh_proj.weight", 2);
        index(SafetensorsCheckpoint.INDEX, "a", "model-00001-of-00001.safetensors");
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(dir(), dir(), false)) {
            assertEquals(Arrays.asList("a", "model.layers.4.eh_proj.weight"), names(c));
            assertEquals(2, c.segments().size());
        }
    }

    @Test
    public void indexEntryOutsideTheModelDirectoryIsRejected() throws IOException {
        Path model = dir().resolve("model");
        Files.createDirectories(model);
        shard("x.safetensors", 0, "a", 2);
        shard("model/model-00001-of-00001.safetensors", 0, "b", 2);
        Files.write(model.resolve(SafetensorsCheckpoint.INDEX),
                "{\"weight_map\":{\"a\":\"../x.safetensors\",\"b\":\"model-00001-of-00001.safetensors\"}}"
                        .getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> SafetensorsCheckpoint.open(model, model, false));
        assertMessage(e, "Invalid safetensors file '" + model.resolve(SafetensorsCheckpoint.INDEX) + "': ",
                "'../x.safetensors', which is not a file inside the model directory");
    }

    @Test
    public void indexWithoutWeightMapIsRejected() throws IOException {
        shard("model.safetensors", 0, "a", 1);
        Files.write(dir().resolve(SafetensorsCheckpoint.INDEX), "{\"metadata\":{}}".getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> SafetensorsCheckpoint.open(dir(), dir(), false));
        assertMessage(e, "Invalid safetensors file '", "'weight_map' is missing or not an object");
        Files.write(dir().resolve(SafetensorsCheckpoint.INDEX), "{\"weight_map\":".getBytes(StandardCharsets.UTF_8));
        e = assertThrows(IOException.class, () -> SafetensorsCheckpoint.open(dir(), dir(), false));
        assertMessage(e, "Invalid safetensors file '", "not a valid index");
    }

    @Test
    public void contentNamesMustStayInsideTheModelDirectory() {
        Path d = dir();
        assertEquals(d.toAbsolutePath().normalize().resolve("sub/x.safetensors"),
                SafetensorsCheckpoint.resolveContentName(d, "sub/x.safetensors"));
        assertEquals(d.toAbsolutePath().normalize().resolve("x.safetensors"),
                SafetensorsCheckpoint.resolveContentName(d, "./x.safetensors"));
        for (String bad : Arrays.asList("../x.safetensors", "a/../../x", "a/..", "/etc/passwd", "\\x", "C:x",
                "a\0b", "", ".")) {
            assertThrows(bad, IllegalArgumentException.class, () -> SafetensorsCheckpoint.resolveContentName(d, bad));
        }
    }

    // ------------------------------------------------------------------ side files

    @Test
    public void opensSideFilesInsideTheModelDirectoryOnly() throws IOException {
        Path p = shard("model.safetensors", 0, "w", 2);
        shard("2_Dense/model.safetensors", 0, "linear.weight", 4);
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(p, dir(), false)) {
            SafetensorsFile side = c.openSideFile("2_Dense/model.safetensors");
            assertEquals(4, side.tensor("linear.weight").byteLength());
            IOException e = assertThrows(IOException.class, () -> c.openSideFile("../outside.safetensors"));
            assertMessage(e, "Invalid Hugging Face checkpoint '", "side file '../outside.safetensors'");
            assertThrows(FileNotFoundException.class, () -> c.openSideFile("readout.safetensors"));
        }
    }

    // ------------------------------------------------------------------ natural order

    @Test
    public void naturalOrderComparesDigitRunsAsNumbers() {
        List<String> names = new ArrayList<>(Arrays.asList("model-10.safetensors", "model-9.safetensors",
                "model-1.safetensors", "model-00002.safetensors", "model.safetensors", "model-2.safetensors",
                "a10b2", "a10b10", "a9b99"));
        names.sort(SafetensorsCheckpoint.NATURAL_ORDER);
        assertEquals(Arrays.asList("a9b99", "a10b2", "a10b10", "model-1.safetensors", "model-00002.safetensors",
                "model-2.safetensors", "model-9.safetensors", "model-10.safetensors", "model.safetensors"), names);
    }

    // ------------------------------------------------------------------ registry

    @Test
    public void registryDispatchesToTheFirstAcceptingFactory() throws IOException {
        Path p = shard("model.safetensors", 0, "w", 2);
        assertSame(SafetensorsCheckpoint.FACTORY, CheckpointContainers.forFile(dir()));
        assertSame(SafetensorsCheckpoint.FACTORY, CheckpointContainers.forFile(p));
        try (CheckpointContainer c = CheckpointContainers.open(p, dir(), false)) {
            assertTrue(c instanceof SafetensorsCheckpoint);
        }

        Path other = dir().resolve("other");
        Files.createDirectories(other);
        Path fake = other.resolve("weights.fake");
        Files.write(fake, new byte[]{1});
        assertNull(CheckpointContainers.forFile(other));
        IOException none = assertThrows(IOException.class, () -> CheckpointContainers.open(other, other, false));
        assertMessage(none, "Invalid checkpoint '" + other + "': ", "no checkpoint container reads it");

        FakeFactory factory = new FakeFactory();
        CheckpointContainers.register(factory);
        try {
            assertThrows(IllegalStateException.class, () -> CheckpointContainers.register(factory));
            assertThrows(IllegalStateException.class,
                    () -> CheckpointContainers.register(SafetensorsCheckpoint.FACTORY));
            List<CheckpointContainers.Factory> all = CheckpointContainers.factories();
            assertSame(SafetensorsCheckpoint.FACTORY, all.get(0));
            assertSame(factory, all.get(all.size() - 1));
            assertSame(factory, CheckpointContainers.forFile(other));
            assertSame(factory, CheckpointContainers.forFile(fake));
            assertSame(SafetensorsCheckpoint.FACTORY, CheckpointContainers.forFile(dir()));
            try (CheckpointContainer c = CheckpointContainers.open(fake, other, false)) {
                assertEquals("fake " + fake, c.description());
            }
        } finally {
            assertTrue(CheckpointContainers.unregister(factory));
        }
        assertFalse(CheckpointContainers.unregister(factory));
        assertNull(CheckpointContainers.forFile(other));
    }

    /** Accepts {@code *.fake} files and opens an empty container. */
    private static final class FakeFactory implements CheckpointContainers.Factory {
        @Override
        public boolean accepts(Path dir, List<Path> files) {
            for (Path f : files) {
                if (f.getFileName().toString().endsWith(".fake")) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public CheckpointContainer open(Path path, Path configDir, boolean mistralNative) {
            return new CheckpointContainer() {
                @Override public String description() { return "fake " + path; }
                @Override public List<StTensor> tensors() { return Collections.emptyList(); }
                @Override public StTensor tensor(String name) { return null; }
                @Override public RowSource source(String name) { throw new IllegalArgumentException(name); }
                @Override public List<DataSegment> segments() { return Collections.emptyList(); }
                @Override public long directVirtualOffset(String name) { throw new IllegalArgumentException(name); }
                @Override public SafetensorsFile openSideFile(String relPath) throws IOException {
                    throw new FileNotFoundException(relPath);
                }
                @Override public void close() { }
            };
        }
    }

    // ------------------------------------------------------------------ tensor index

    @Test
    public void tensorIndexKeepsCheckpointOrderAndReplacesInPlace() throws IOException {
        Path p = shard("model.safetensors", 0, "a.qweight", 2, "a.scales", 2, "b.weight", 2, "c.weight", 2);
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(p, dir(), false)) {
            HfTensorIndex index = HfTensorIndex.of(c);
            assertEquals(Arrays.asList("a.qweight", "a.scales", "b.weight", "c.weight"), index.names());
            assertTrue(index.has("b.weight"));
            assertEquals(c.directVirtualOffset("b.weight"), index.get("b.weight").directVirtualOffset());
            assertThrows(IllegalArgumentException.class, () -> index.get("missing"));

            RowSource repacked = c.source("c.weight");
            Map<String, RowSource> replace = new LinkedHashMap<>();
            replace.put("a.weight", repacked);
            HfTensorIndex rewritten = index.withReplaced(replace,
                    new HashSet<>(Arrays.asList("a.qweight", "a.scales")));
            assertEquals(Arrays.asList("a.weight", "b.weight", "c.weight"), rewritten.names());
            assertSame(repacked, rewritten.get("a.weight"));
            assertEquals(4, index.size());

            Map<String, RowSource> inPlace = new LinkedHashMap<>();
            inPlace.put("b.weight", repacked);
            inPlace.put("b.weight.extra", repacked);
            HfTensorIndex kept = index.withReplaced(inPlace, Collections.emptySet());
            assertEquals(Arrays.asList("a.qweight", "a.scales", "b.weight", "b.weight.extra", "c.weight"),
                    kept.names());

            HfTensorIndex appended = index.withReplaced(Collections.singletonMap("z", repacked),
                    Collections.emptySet());
            assertEquals("z", appended.names().get(4));
            assertThrows(IllegalArgumentException.class,
                    () -> index.withReplaced(Collections.emptyMap(), Collections.singleton("missing")));
            assertThrows(IllegalArgumentException.class,
                    () -> index.withReplaced(Collections.singletonMap("b.weight", repacked),
                            Collections.singleton("b.weight")));
        }
    }

    @Test
    public void sourcesReadTheirFileBytes() throws IOException {
        Path p = shard("model.safetensors", 3, "w", 6);
        try (SafetensorsCheckpoint c = SafetensorsCheckpoint.open(p, dir(), false)) {
            RowSource s = c.source("w");
            ByteBuffer dst = ByteBuffer.allocate(6);
            s.readRows(0, 1, dst);
            assertArrayEquals(ramp(3, 6), dst.array());
        }
    }
}
