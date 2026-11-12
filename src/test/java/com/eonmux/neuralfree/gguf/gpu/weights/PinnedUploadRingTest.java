package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Test;

import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

/**
 * The page-locked upload ring ({@link PinnedUploadRing}): chunks copied by several threads land byte for byte at their
 * device offsets through every slot of the ring, a chunk larger than a slot is refused, a backend without host-mapped
 * memory and a device that shares the host's memory get no ring, and a closed ring refuses uploads.
 */
public class PinnedUploadRingTest {

    private FakeComputeBackend backend;

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    private static byte[] blob(int n, long seed) {
        byte[] b = new byte[n];
        new Random(seed).nextBytes(b);
        return b;
    }

    @Test
    public void chunksLandAtTheirOffsetsThroughEverySlot() {
        backend = FakeComputeBackend.create();
        int slot = 6 * PinnedUploadRing.MIN_PART_BYTES + 12_345;
        byte[] data = blob(4 * slot + 777, 3);
        FakeGpuBuffer dst = (FakeGpuBuffer) backend.allocateBuffer(data.length + 64L);
        try (PinnedUploadRing ring = PinnedUploadRing.open(backend, slot)) {
            ByteBuffer src = ByteBuffer.allocateDirect(data.length);
            src.put(data).clear();
            int chunks = 0;
            for (int at = 0; at < data.length; at += slot, chunks++) {
                int n = Math.min(slot, data.length - at);
                ByteBuffer chunk = src.duplicate();
                chunk.limit(at + n).position(at);
                ring.upload(chunk, dst, 64L + at, backend.computeQueue());
                assertEquals("the source is not moved", at, chunk.position());
            }
            assertEquals("more chunks than slots, so every slot is reused", 5, chunks);
        }
        backend.computeQueue().finish();
        byte[] device = dst.toByteArray();
        assertArrayEquals(data, Arrays.copyOfRange(device, 64, 64 + data.length));
    }

    @Test
    public void aChunkLargerThanASlotIsRefused() {
        backend = FakeComputeBackend.create();
        try (PinnedUploadRing ring = PinnedUploadRing.open(backend, 1024)) {
            ring.upload(ByteBuffer.allocate(1025), backend.allocateBuffer(2048), 0L, backend.computeQueue());
            fail("a chunk past the slot size must be refused");
        } catch (IllegalArgumentException expected) {
            // refused before anything is copied
        }
    }

    @Test
    public void aClosedRingRefusesUploads() {
        backend = FakeComputeBackend.create();
        PinnedUploadRing ring = PinnedUploadRing.open(backend, 1024);
        ring.close();
        ring.close();
        try {
            ring.upload(ByteBuffer.allocate(8), backend.allocateBuffer(64), 0L, backend.computeQueue());
            fail("a closed ring must refuse uploads");
        } catch (IllegalStateException expected) {
            // closed
        }
    }

    @Test
    public void aBackendWithoutHostMappedMemoryGetsNoRing() {
        ComputeBackend plain = (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                new Class<?>[]{ComputeBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("supportsHostMappedBuffers")) {
                        return false;
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
        assertNull(PinnedUploadRing.open(plain, 1024));
    }

    @Test
    public void aBackendWhoseOwnWritesAreAsFastGetsNoRing() {
        try (FakeComputeBackend b = FakeComputeBackend.builder().parallelUploadStaging(false).build()) {
            assertNull(PinnedUploadRing.open(b, 1024));
        }
    }

    @Test
    public void onlyADiscreteDeviceGetsARing() {
        for (DeviceClass c : DeviceClass.values()) {
            DeviceCapabilities caps = new DeviceCapabilities.Builder()
                    .deviceName("Fake " + c)
                    .totalVramBytes(8L << 30)
                    .maxAllocSizeBytes(2L << 30)
                    .maxWorkGroupSize(1024)
                    .deviceClass(c)
                    .build();
            try (FakeComputeBackend b = FakeComputeBackend.builder().capabilities(caps).build();
                 PinnedUploadRing ring = PinnedUploadRing.open(b, 1024)) {
                assertEquals(c + " device", c == DeviceClass.DISCRETE, ring != null);
            }
        }
    }
}
