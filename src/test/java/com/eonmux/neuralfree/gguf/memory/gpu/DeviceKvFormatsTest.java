package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvVendorCodec;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The device KV allocation of the vendor formats: their side buffers and the geometries it refuses. */
public class DeviceKvFormatsTest {

    @Test
    public void vendorFormatsAllocateASideBuffer() {
        FakeComputeBackend backend = FakeComputeBackend.builder().build();
        DeviceKvLayer nvfp4 = null;
        DeviceKvLayer tq = null;
        DeviceKvLayer latent = null;
        try {
            KvLayout nvfp4Layout = KvLayout.of(8, false, 64, 64, KvDType.NVFP4, KvDType.NVFP4, KvWindow.FULL_CAUSAL)
                    .withHeads(4);
            nvfp4 = DeviceKvLayer.allocate(backend, backend.computeQueue(), 3, nvfp4Layout, 0, RotationPlan.NONE,
                    "test");
            assertEquals(nvfp4Layout.kBytes(), nvfp4.k().sizeInBytes());
            assertEquals(DeviceKvFormats.sideBytes(nvfp4Layout), nvfp4.vendorSide().sizeInBytes());
            assertEquals(nvfp4Layout.widthK() / KvVendorCodec.NVFP4_GROUP,
                    DeviceKvFormats.kPlaneRowBytes(nvfp4Layout));
            assertEquals(DeviceKvFormats.kPlaneRowBytes(nvfp4Layout) * 2,
                    DeviceKvFormats.sideRowBytes(nvfp4Layout));
            float[] header = header(nvfp4.vendorSide(), nvfp4Layout);
            for (int h = 0; h < 2 * nvfp4Layout.nHeadKv(); h++) {
                assertEquals("checkpoint scale " + h, 1.0f, header[h], 0.0f);
            }

            KvLayout tqLayout = KvLayout.of(8, false, 64, 64, KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES,
                    KvWindow.FULL_CAUSAL).withHeads(4);
            tq = DeviceKvLayer.allocate(backend, backend.computeQueue(), 4, tqLayout, 0, RotationPlan.NONE, "test");
            assertEquals(DeviceKvFormats.sideBytes(tqLayout), tq.vendorSide().sizeInBytes());
            float[] tqHeader = header(tq.vendorSide(), tqLayout);
            float[] table = KvVendorCodec.centroids(16, 4);
            for (int i = 0; i < table.length; i++) {
                assertEquals("centroid " + i, table[i], tqHeader[2 * tqLayout.nHeadKv() + i], 0.0f);
            }

            KvLayout latentLayout = KvLayout.latent(8, false, 576, 512, KvDType.FP8_DS_MLA, KvWindow.FULL_CAUSAL)
                    .withHeads(1);
            latent = DeviceKvLayer.allocate(backend, backend.computeQueue(), 5, latentLayout, 0, RotationPlan.NONE,
                    "test");
            assertEquals(0, DeviceKvFormats.vPlaneRowBytes(latentLayout));
            assertEquals(DeviceKvFormats.kPlaneRowBytes(latentLayout), DeviceKvFormats.sideRowBytes(latentLayout));
            assertEquals(DeviceKvFormats.sideBytes(latentLayout), latent.vendorSide().sizeInBytes());
        } finally {
            closeAll(nvfp4, tq, latent);
            backend.close();
        }
    }

    @Test
    public void kernelTypesKeepNoSideBuffer() {
        FakeComputeBackend backend = FakeComputeBackend.builder().build();
        DeviceKvLayer layer = null;
        try {
            KvLayout f16 = KvLayout.of(8, false, 64, 64, KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL).withHeads(4);
            layer = DeviceKvLayer.allocate(backend, backend.computeQueue(), 0, f16, 0, RotationPlan.NONE, "test");
            assertEquals(f16.kBytes(), layer.k().sizeInBytes());
            assertNull(layer.vendorSide());
        } finally {
            if (layer != null) {
                layer.close();
            }
            backend.close();
        }
    }

    @Test
    public void vendorGeometryTheKernelsCannotAddressIsRefusedByName() {
        FakeComputeBackend backend = FakeComputeBackend.builder().build();
        try {
            KvLayout oddHead = KvLayout.of(8, false, 48, 48, KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES,
                    KvWindow.FULL_CAUSAL).withHeads(4);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> DeviceKvLayer.allocate(backend, backend.computeQueue(), 1, oddHead, 0, RotationPlan.NONE,
                            "KV cache of layer 1"));
            assertTrue(e.getMessage(), e.getMessage().contains("tq_4bit_keys"));
            assertTrue(e.getMessage(), e.getMessage().contains("power of two"));

            KvLayout wideHead = KvLayout.of(8, false, 60, 60, KvDType.NVFP4, KvDType.NVFP4,
                    KvWindow.FULL_CAUSAL).withHeads(4);
            IllegalArgumentException wide = assertThrows(IllegalArgumentException.class,
                    () -> DeviceKvLayer.allocate(backend, backend.computeQueue(), 2, wideHead, 0, RotationPlan.NONE,
                            "KV cache of layer 2"));
            assertTrue(wide.getMessage(), wide.getMessage().contains("nvfp4"));
            assertTrue(wide.getMessage(), wide.getMessage().contains("multiple of the 16-value NVFP4"));
        } finally {
            backend.close();
        }
    }

    @Test
    public void draftCachesWithoutSidePlanesRefuseVendorTypes() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> DeviceKvFormats.requireKernel(KvDType.NVFP4, "the DFlash draft KV cache"));
        assertTrue(e.getMessage(), e.getMessage().contains("nvfp4"));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.kv.offload=false"));
        DeviceKvFormats.requireKernel(KvDType.F16, "the DFlash draft KV cache");
    }

    private static float[] header(com.eonmux.neuralfree.acceleration.backend.GpuBuffer buffer, KvLayout layout) {
        FakeGpuBuffer fake = (FakeGpuBuffer) buffer;
        int floats = 2 * layout.nHeadKv() + DeviceKvFormats.SIDE_CTR_FLOATS;
        ByteBuffer bb = ByteBuffer.wrap(fake.toByteArray(), 0, floats * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[floats];
        for (int i = 0; i < floats; i++) {
            out[i] = bb.getFloat();
        }
        return out;
    }

    private static void closeAll(DeviceKvLayer... layers) {
        for (DeviceKvLayer l : layers) {
            if (l != null) {
                l.close();
            }
        }
    }
}
