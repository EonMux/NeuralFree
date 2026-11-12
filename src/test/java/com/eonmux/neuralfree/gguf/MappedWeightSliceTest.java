package com.eonmux.neuralfree.gguf;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The mapped views of the weight stores ({@link WeightStore#mappedSlice}): byte-identical to a read of the same range,
 * read-only and little-endian, cut at a mapped segment's end and at a shard's end, and absent when the store reads
 * through its channel.
 */
public class MappedWeightSliceTest {

    private static final int HEADER = 37;
    private static final int DATA = 10_000;
    private static final int SEGMENT = 4096;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private byte[] data;

    private Path file(String name, long seed) throws IOException {
        data = new byte[DATA];
        new Random(seed).nextBytes(data);
        byte[] whole = new byte[HEADER + DATA];
        Arrays.fill(whole, 0, HEADER, (byte) 0x5A);
        System.arraycopy(data, 0, whole, HEADER, DATA);
        Path p = tmp.getRoot().toPath().resolve(name);
        Files.write(p, whole);
        return p;
    }

    private static byte[] bytes(ByteBuffer view) {
        byte[] out = new byte[view.capacity()];
        for (int i = 0; i < out.length; i++) {
            out[i] = view.get(i);
        }
        return out;
    }

    @Test
    public void aSliceHoldsTheStoredBytesUpToTheSegmentEnd() throws IOException {
        Path p = file("one.bin", 1);
        try (MappedWeightStore store = new MappedWeightStore(p.toString(), HEADER, DATA, TensorReadMode.MMAP,
                SEGMENT)) {
            ByteBuffer inside = store.mappedSlice(100, 500);
            assertEquals(500, inside.capacity());
            assertTrue(inside.isReadOnly());
            assertEquals(ByteOrder.LITTLE_ENDIAN, inside.order());
            assertArrayEquals(Arrays.copyOfRange(data, 100, 600), bytes(inside));

            ByteBuffer straddling = store.mappedSlice(SEGMENT - 10, 100);
            assertEquals(10, straddling.capacity());
            assertArrayEquals(Arrays.copyOfRange(data, SEGMENT - 10, SEGMENT), bytes(straddling));
            ByteBuffer next = store.mappedSlice(SEGMENT, 90);
            assertArrayEquals(Arrays.copyOfRange(data, SEGMENT, SEGMENT + 90), bytes(next));

            ByteBuffer tail = store.mappedSlice(DATA - 3, 1 << 20);
            assertArrayEquals(Arrays.copyOfRange(data, DATA - 3, DATA), bytes(tail));

            byte[] read = new byte[200];
            store.readTensor(SEGMENT - 100, 200, read);
            ByteBuffer a = store.mappedSlice(SEGMENT - 100, 200);
            ByteBuffer b = store.mappedSlice(SEGMENT - 100 + a.capacity(), 200 - a.capacity());
            byte[] joined = new byte[200];
            System.arraycopy(bytes(a), 0, joined, 0, a.capacity());
            System.arraycopy(bytes(b), 0, joined, a.capacity(), b.capacity());
            assertArrayEquals(read, joined);

            assertThrows(IOException.class, () -> store.mappedSlice(DATA, 1));
            assertThrows(IllegalArgumentException.class, () -> store.mappedSlice(0, 0));
        }
    }

    @Test
    public void aChannelReadStoreHasNoSlices() throws IOException {
        Path p = file("read.bin", 2);
        try (MappedWeightStore store = new MappedWeightStore(p.toString(), HEADER, DATA, TensorReadMode.READ,
                SEGMENT)) {
            assertNull(store.mappedSlice(0, 100));
            ByteBuffer dst = ByteBuffer.allocateDirect(100);
            store.readTensorDirect(50, 100, dst);
            dst.flip();
            assertArrayEquals(Arrays.copyOfRange(data, 50, 150), bytes(dst.slice()));
        }
    }

    @Test
    public void aShardedStoreCutsSlicesAtTheShardEnd() throws IOException {
        Path first = file("s1.bin", 3);
        byte[] firstData = data;
        Path second = file("s2.bin", 4);
        byte[] secondData = data;
        try (MultiShardMappedWeightStore store = new MultiShardMappedWeightStore(Arrays.asList(
                new DataSegment(first.toString(), HEADER, 0, DATA),
                new DataSegment(second.toString(), HEADER, DATA, DATA)))) {
            ByteBuffer end = store.mappedSlice(DATA - 20, 1000);
            assertEquals(20, end.capacity());
            assertArrayEquals(Arrays.copyOfRange(firstData, DATA - 20, DATA), bytes(end));
            ByteBuffer start = store.mappedSlice(DATA, 30);
            assertArrayEquals(Arrays.copyOfRange(secondData, 0, 30), bytes(start));
        }
    }
}
