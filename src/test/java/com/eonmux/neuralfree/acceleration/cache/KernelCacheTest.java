package com.eonmux.neuralfree.acceleration.cache;

import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import org.junit.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

public class KernelCacheTest {

    @Test
    public void testCacheHitDoesNotRecompile() {
        KernelCache cache = new KernelCache();
        CacheKey key = new CacheKey("testKernel", "hash123", "-O3");
        AtomicInteger compileCount = new AtomicInteger(0);
        
        KernelHandle handle1 = cache.getOrCompile(key, k -> {
            compileCount.incrementAndGet();
            return new TestKernelHandle("testKernel");
        });
        
        KernelHandle handle2 = cache.getOrCompile(key, k -> {
            compileCount.incrementAndGet();
            return new TestKernelHandle("testKernel");
        });
        
        assertSame(handle1, handle2);
        assertEquals(1, compileCount.get());
    }

    @Test
    public void testDifferentKeysCompileSeparately() {
        KernelCache cache = new KernelCache();
        CacheKey key1 = new CacheKey("kernel1", "hash1", "");
        CacheKey key2 = new CacheKey("kernel2", "hash2", "");
        AtomicInteger compileCount = new AtomicInteger(0);
        
        KernelHandle handle1 = cache.getOrCompile(key1, k -> {
            compileCount.incrementAndGet();
            return new TestKernelHandle("kernel1");
        });
        
        KernelHandle handle2 = cache.getOrCompile(key2, k -> {
            compileCount.incrementAndGet();
            return new TestKernelHandle("kernel2");
        });
        
        assertNotSame(handle1, handle2);
        assertEquals(2, compileCount.get());
    }

    @Test
    public void testConcurrentGetOrCompile() throws InterruptedException {
        KernelCache cache = new KernelCache();
        CacheKey key = new CacheKey("concurrentKernel", "hash", "");
        AtomicInteger compileCount = new AtomicInteger(0);
        
        int threadCount = 10;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(threadCount);
        KernelHandle[] handles = new KernelHandle[threadCount];
        
        for (int i = 0; i < threadCount; i++) {
            final int idx = i;
            new Thread(() -> {
                try {
                    startLatch.await();
                    handles[idx] = cache.getOrCompile(key, k -> {
                        compileCount.incrementAndGet();
                        try {
                            Thread.sleep(10);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        return new TestKernelHandle("concurrentKernel");
                    });
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    doneLatch.countDown();
                }
            }).start();
        }
        
        startLatch.countDown();
        doneLatch.await();
        
        KernelHandle firstHandle = handles[0];
        assertNotNull(firstHandle);
        for (KernelHandle h : handles) {
            assertSame(firstHandle, h);
        }
        assertEquals(1, compileCount.get());
    }

    private static class TestKernelHandle implements KernelHandle {
        private final String name;

        TestKernelHandle(String name) {
            this.name = name;
        }

        @Override
        public String kernelName() {
            return name;
        }

        @Override
        public void close() {
        }
    }
}
