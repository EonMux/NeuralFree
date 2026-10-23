package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import org.jocl.CL;
import org.jocl.CLException;
import org.junit.BeforeClass;
import org.junit.Test;

import static org.jocl.CL.*;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

/**
 * Status mapping of {@link OclErrors}: out-of-memory statuses become {@link GpuOutOfMemoryException},
 * every other failure a {@link GpuBackendException} with the status, in both JOCL exception modes.
 * Status names come from JOCL, so the tests need its native library.
 */
public class OclErrorsTest {

    @BeforeClass
    public static void requireJocl() {
        boolean loaded;
        try {
            loaded = CL.stringFor_errorCode(CL_SUCCESS) != null;
        } catch (Throwable e) {
            loaded = false;
        }
        assumeTrue("JOCL native library unavailable", loaded);
    }

    @Test
    public void outOfMemoryStatusesAreRecognised() {
        assertTrue(OclErrors.isOutOfMemory(CL_MEM_OBJECT_ALLOCATION_FAILURE));
        assertTrue(OclErrors.isOutOfMemory(CL_OUT_OF_RESOURCES));
        assertTrue(OclErrors.isOutOfMemory(CL_OUT_OF_HOST_MEMORY));
        assertFalse(OclErrors.isOutOfMemory(CL_INVALID_WORK_GROUP_SIZE));
        assertFalse(OclErrors.isOutOfMemory(CL_SUCCESS));
    }

    @Test
    public void returnedOutOfMemoryBecomesGpuOutOfMemoryException() {
        try {
            OclErrors.check("clEnqueueNDRangeKernel", "LLM_RmsNorm", () -> CL_MEM_OBJECT_ALLOCATION_FAILURE);
            fail("expected GpuOutOfMemoryException");
        } catch (GpuOutOfMemoryException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("clEnqueueNDRangeKernel (LLM_RmsNorm)"));
            assertTrue(e.getMessage(), e.getMessage().contains("CL_MEM_OBJECT_ALLOCATION_FAILURE"));
        }
    }

    @Test
    public void returnedFailureBecomesGpuBackendExceptionWithStatus() {
        try {
            OclErrors.check("clEnqueueNDRangeKernel", "Q4_K_MatVec", () -> CL_INVALID_WORK_GROUP_SIZE);
            fail("expected GpuBackendException");
        } catch (GpuBackendException e) {
            assertEquals(CL_INVALID_WORK_GROUP_SIZE, e.getCode());
            assertEquals(OclErrors.BACKEND, e.getBackend());
            assertTrue(e.getMessage(), e.getMessage().contains("CL_INVALID_WORK_GROUP_SIZE"));
        }
    }

    @Test
    public void thrownFailureIsTranslatedWithItsCause() {
        CLException thrown = new CLException("CL_OUT_OF_RESOURCES", CL_OUT_OF_RESOURCES);
        try {
            OclErrors.check("clFinish", null, () -> {
                throw thrown;
            });
            fail("expected GpuOutOfMemoryException");
        } catch (GpuOutOfMemoryException e) {
            assertSame(thrown, e.getCause());
        }
        CLException invalid = new CLException("CL_INVALID_EVENT", CL_INVALID_EVENT);
        try {
            OclErrors.check("clWaitForEvents", null, () -> {
                throw invalid;
            });
            fail("expected GpuBackendException");
        } catch (GpuBackendException e) {
            assertEquals(CL_INVALID_EVENT, e.getCode());
            assertSame(invalid, e.getCause());
        }
    }

    @Test
    public void successPasses() {
        OclErrors.check("clFlush", null, () -> CL_SUCCESS);
        OclErrors.check(CL_SUCCESS, "clFlush", null);
    }

    @Test
    public void allocationFailureReportsRequestedBytes() {
        RuntimeException e = OclErrors.translate(CL_MEM_OBJECT_ALLOCATION_FAILURE, "clCreateBuffer", "4096 bytes",
                4000, null);
        assertTrue(e instanceof GpuOutOfMemoryException);
        assertEquals(4000, ((GpuOutOfMemoryException) e).getRequestedBytes());
        assertEquals(OclErrors.BACKEND, ((GpuOutOfMemoryException) e).getBackendType());
    }

    @Test(expected = IllegalArgumentException.class)
    public void successIsNotAFailure() {
        OclErrors.translate(CL_SUCCESS, "clFinish", null, -1, null);
    }
}
