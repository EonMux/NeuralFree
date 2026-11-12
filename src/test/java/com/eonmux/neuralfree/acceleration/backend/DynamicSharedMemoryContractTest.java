package com.eonmux.neuralfree.acceleration.backend;

import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The dynamic shared memory contract of a backend that launches none: a reservation of 0 bytes is accepted, a
 * positive one is refused with the kernel named, and a negative one is refused as an argument.
 */
public class DynamicSharedMemoryContractTest {

    private final KernelHandle kernel = new KernelHandle() {
        @Override
        public String kernelName() {
            return "staged_tile";
        }

        @Override
        public void close() {
        }
    };

    private FakeComputeBackend backend;

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    @Test
    public void aBackendWithoutDynamicSharedMemoryAcceptsOnlyZero() {
        backend = FakeComputeBackend.builder().build();
        backend.reserveDynamicSharedMemory(kernel, 0);
        UnsupportedOperationException positive = assertThrows(UnsupportedOperationException.class,
                () -> backend.reserveDynamicSharedMemory(kernel, 1024));
        assertTrue(positive.getMessage(), positive.getMessage().contains("staged_tile"));
        assertThrows(IllegalArgumentException.class, () -> backend.reserveDynamicSharedMemory(kernel, -1));
    }
}
