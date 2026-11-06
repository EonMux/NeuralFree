package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link MappedWeightStore} under the pinned load modes: {@code mmap+mlock} pins the mapping but still serves
 * reads from it, {@code mlock} pins the file's pages while reads stay on the channel, and both return the same
 * tensor bytes as the plain modes.
 */
public class MappedWeightStoreMlockTest {

    private Path file;
    private String priorMode;

    @Before
    public void writeModel() throws IOException {
        file = SyntheticModels.llama().writeTempFile();
        priorMode = System.getProperty(TensorReadMode.PROP_READ_MODE);
    }

    @After
    public void cleanUp() throws IOException {
        if (priorMode == null) {
            System.clearProperty(TensorReadMode.PROP_READ_MODE);
        } else {
            System.setProperty(TensorReadMode.PROP_READ_MODE, priorMode);
        }
        Files.deleteIfExists(file);
    }

    /** Every tensor's bytes through a fresh parser and store in {@code mode}. */
    private Map<String, byte[]> readAll(TensorReadMode mode, boolean[] mmapAvailable) throws IOException {
        System.setProperty(TensorReadMode.PROP_READ_MODE, mode.flagName());
        Map<String, byte[]> bytes = new LinkedHashMap<>();
        GGUFFileParser parser = new GGUFFileParser(file.toString());
        try (WeightStore store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            assertTrue("a single-file model opens a MappedWeightStore", store instanceof MappedWeightStore);
            mmapAvailable[0] = ((MappedWeightStore) store).isMmapAvailable();
            for (GGUFTensorInfo tensor : parser.getTensors()) {
                byte[] data = new byte[Math.toIntExact(tensor.sizeInBytes())];
                store.readTensor(tensor.offset(), data.length, data);
                bytes.put(tensor.name(), data);
            }
        } finally {
            parser.close();
        }
        return bytes;
    }

    @Test
    public void pinnedModesReadTheSameBytesAsThePlainModes() throws IOException {
        boolean[] mapped = new boolean[1];
        Map<String, byte[]> baseline = readAll(TensorReadMode.MMAP, mapped);
        assertTrue("the default mode maps", mapped[0]);
        assertFalse(baseline.isEmpty());

        boolean[] pinnedMapping = new boolean[1];
        Map<String, byte[]> mmapMlock = readAll(TensorReadMode.MMAP_MLOCK, pinnedMapping);
        assertTrue("mmap+mlock still serves reads from the mapping", pinnedMapping[0]);
        assertFalse("mlock is available on Linux", mmapMlock.isEmpty());
        for (Map.Entry<String, byte[]> entry : baseline.entrySet()) {
            assertArrayEquals("mmap+mlock tensor " + entry.getKey(), entry.getValue(),
                    mmapMlock.get(entry.getKey()));
        }

        boolean[] readPinned = new boolean[1];
        Map<String, byte[]> mlock = readAll(TensorReadMode.MLOCK, readPinned);
        assertFalse("mlock reads through the channel", readPinned[0]);
        for (Map.Entry<String, byte[]> entry : baseline.entrySet()) {
            assertArrayEquals("mlock tensor " + entry.getKey(), entry.getValue(), mlock.get(entry.getKey()));
        }

        boolean[] read = new boolean[1];
        Map<String, byte[]> plain = readAll(TensorReadMode.READ, read);
        assertFalse(read[0]);
        for (Map.Entry<String, byte[]> entry : baseline.entrySet()) {
            assertArrayEquals("read tensor " + entry.getKey(), entry.getValue(), plain.get(entry.getKey()));
        }
    }

    @Test
    public void mmapMlockPinsTheModelsPages() throws IOException {
        Assume.assumeTrue(com.sun.jna.Platform.isLinux());
        Assume.assumeTrue("the host exposes mlock", MemoryLock.supported());
        Path smaps = Path.of("/proc/self/smaps");
        Assume.assumeTrue("/proc/self/smaps is readable", Files.isReadable(smaps));

        System.setProperty(TensorReadMode.PROP_READ_MODE, TensorReadMode.MMAP_MLOCK.flagName());
        String path = file.toRealPath().toString();
        GGUFFileParser parser = new GGUFFileParser(file.toString());
        try (WeightStore store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            assertTrue(store instanceof MappedWeightStore);
            assertTrue("the mapping stays readable while pinned", ((MappedWeightStore) store).isMmapAvailable());
            assertTrue("the model file's mapped pages are locked", lockedPages(smaps, path) > 0);
        } finally {
            parser.close();
        }
    }

    /** The locked kB the kernel reports for the mapping of {@code path}, summed over its blocks. */
    private static long lockedPages(Path smaps, String path) throws IOException {
        long lockedKb = 0;
        boolean ours = false;
        for (String line : Files.readAllLines(smaps)) {
            if (isHeader(line)) {
                String[] header = line.split("\\s+", 6);
                ours = header.length == 6 && header[5].equals(path);
            } else if (ours && line.startsWith("Locked:")) {
                lockedKb += Long.parseLong(line.trim().split("\\s+")[1]);
            }
        }
        return lockedKb;
    }

    /** Whether {@code line} is an {@code smaps} mapping header, {@code start-end perms ...}. */
    private static boolean isHeader(String line) {
        int dash = line.indexOf('-');
        int space = line.indexOf(' ');
        if (line.isEmpty() || Character.isWhitespace(line.charAt(0)) || dash <= 0 || space <= dash) {
            return false;
        }
        for (int i = 0; i < dash; i++) {
            if (Character.digit(line.charAt(i), 16) < 0) {
                return false;
            }
        }
        return true;
    }
}
