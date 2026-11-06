package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.hip.HipErrors;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The status-to-exception mapping of the HIP backend: out of memory to
 * {@code GpuOutOfMemoryException}, a code object the device cannot run to
 * {@code UnsupportedKernelException}, anything else to {@code GpuBackendException}.
 */
public class HipErrorsTest {

    @Test
    public void successPassesThrough() {
        assertEquals(0, HipErrors.call(HipErrors.SUCCESS, "hipMalloc"));
    }

    @Test
    public void outOfMemoryBecomesGpuOutOfMemoryException() {
        try {
            HipErrors.call(HipErrors.OUT_OF_MEMORY, "hipMalloc");
            fail("out of memory must throw");
        } catch (GpuOutOfMemoryException e) {
            // Without a requested size the exception names the backend in its message.
            assertTrue(e.getMessage().contains("backend=HIP"));
            assertTrue(e.getMessage().contains("hipMalloc"));
        }
    }

    @Test
    public void outOfMemoryReportsTheRequestedBytes() {
        RuntimeException e = HipErrors.translate(HipErrors.OUT_OF_MEMORY, "hipMalloc", 4096);
        assertTrue(e instanceof GpuOutOfMemoryException);
        assertEquals(4096, ((GpuOutOfMemoryException) e).getRequestedBytes());
    }

    @Test
    public void incompatibleCodeObjectsBecomeUnsupportedKernelException() {
        int[] codes = {HipErrors.NO_BINARY_FOR_GPU, HipErrors.INVALID_IMAGE, HipErrors.INVALID_KERNEL_FILE};
        for (int code : codes) {
            RuntimeException e = HipErrors.translate(code, "hipModuleLoadData", -1);
            assertTrue(code + " must be an unsupported-kernel case", e instanceof UnsupportedKernelException);
            assertTrue(e.getMessage().contains("code object"));
        }
    }

    @Test
    public void imageIncompatibilityFlagsMatchTheCodes() {
        assertTrue(HipErrors.isImageIncompatible(HipErrors.NO_BINARY_FOR_GPU));
        assertTrue(HipErrors.isImageIncompatible(HipErrors.INVALID_IMAGE));
        assertTrue(HipErrors.isImageIncompatible(HipErrors.INVALID_KERNEL_FILE));
        assertFalseForOtherCodes();
    }

    private void assertFalseForOtherCodes() {
        int[] others = {HipErrors.SUCCESS, HipErrors.OUT_OF_MEMORY, HipErrors.INVALID_DEVICE,
                HipErrors.INVALID_HANDLE, HipErrors.NOT_READY, HipErrors.INVALID_CONFIGURATION,
                HipErrors.INVALID_CONTEXT, HipErrors.LAUNCH_OUT_OF_RESOURCES, HipErrors.UNKNOWN};
        for (int code : others) {
            if (!HipErrors.isImageIncompatible(code)) {
                continue;
            }
            fail(code + " must not be flagged as image incompatibility");
        }
    }

    @Test
    public void otherCodesBecomeGpuBackendException() {
        RuntimeException e = HipErrors.translate(HipErrors.INVALID_DEVICE, "hipSetDevice", -1);
        assertTrue(e instanceof GpuBackendException);
        assertEquals("HIP", ((GpuBackendException) e).getBackend());
        assertEquals(HipErrors.INVALID_DEVICE, ((GpuBackendException) e).getCode());
    }

    @Test
    public void statusCodesMatchTheRocmHeader() {
        // hipError_t of the ROCm runtime: the shim returns these raw values.
        assertEquals(0, HipErrors.SUCCESS);
        assertEquals(1, HipErrors.INVALID_VALUE);
        assertEquals(2, HipErrors.OUT_OF_MEMORY);
        assertEquals(9, HipErrors.INVALID_CONFIGURATION);
        assertEquals(101, HipErrors.INVALID_DEVICE);
        assertEquals(200, HipErrors.INVALID_IMAGE);
        assertEquals(201, HipErrors.INVALID_CONTEXT);
        assertEquals(209, HipErrors.NO_BINARY_FOR_GPU);
        assertEquals(218, HipErrors.INVALID_KERNEL_FILE);
        assertEquals(400, HipErrors.INVALID_HANDLE);
        assertEquals(600, HipErrors.NOT_READY);
        assertEquals(701, HipErrors.LAUNCH_OUT_OF_RESOURCES);
        assertEquals(999, HipErrors.UNKNOWN);
    }
}
