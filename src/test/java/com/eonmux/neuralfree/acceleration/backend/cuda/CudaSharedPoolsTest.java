package com.eonmux.neuralfree.acceleration.backend.cuda;

import jcuda.driver.CUcontext;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The registry of buffer pools sharing a primary context: pooled bytes are summed over every
 * pool of the context, and an out-of-memory reclaim trims every pool except the caller's.
 * No device is needed.
 */
public class CudaSharedPoolsTest {

    private final CUcontext context = new CUcontext();
    private FakeCudaMemoryDriver ownDriver;
    private FakeCudaMemoryDriver siblingDriver;
    private CudaBufferPool own;
    private CudaBufferPool sibling;

    @Before
    public void registerPools() {
        ownDriver = new FakeCudaMemoryDriver();
        siblingDriver = new FakeCudaMemoryDriver();
        own = new CudaBufferPool(ownDriver);
        sibling = new CudaBufferPool(siblingDriver);
        CudaSharedPools.register(context, own);
        CudaSharedPools.register(context, sibling);
    }

    @After
    public void unregisterPools() {
        CudaSharedPools.unregister(context, own);
        CudaSharedPools.unregister(context, sibling);
    }

    @Test
    public void pooledBytesCountEveryPoolOfTheContext() {
        pool(own, 1024);
        pool(sibling, 4096);

        assertEquals(1024 + 4096, CudaSharedPools.pooledBytes(context));
    }

    @Test
    public void trimSiblingsFreesEveryPoolButTheCaller() {
        pool(own, 1024);
        pool(sibling, 4096);
        List<RuntimeException> failures = new ArrayList<>();

        assertEquals(4096, CudaSharedPools.trimSiblings(context, own, failures));

        assertTrue(failures.isEmpty());
        assertEquals(1024, own.pooledBytes());
        assertEquals(0, sibling.pooledBytes());
        assertEquals(0, siblingDriver.liveBytes());
        assertEquals(1, siblingDriver.syncCalls);
        assertEquals(0, ownDriver.freeCalls);
    }

    @Test
    public void unregisteredPoolIsNeitherCountedNorTrimmed() {
        pool(sibling, 4096);
        CudaSharedPools.unregister(context, sibling);
        List<RuntimeException> failures = new ArrayList<>();

        assertEquals(0, CudaSharedPools.pooledBytes(context));
        assertEquals(0, CudaSharedPools.trimSiblings(context, own, failures));
        assertEquals(4096, sibling.pooledBytes());
    }

    /** Acquires and releases one block of {@code bytes}, leaving it pooled with its fence in flight. */
    private static void pool(CudaBufferPool pool, long bytes) {
        CudaBufferPool.Allocation a = pool.acquire(bytes);
        pool.release(a.pointer(), a.bytes());
    }
}
