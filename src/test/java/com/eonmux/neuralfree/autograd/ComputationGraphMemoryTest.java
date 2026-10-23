package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue.QueueType;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.tensor.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.*;

public class ComputationGraphMemoryTest {

    private MockComputeBackend backend;

    @Before
    public void setUp() {
        backend = new MockComputeBackend();
        ComputationGraph.current().clear();
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        ComputationGraph.current().clear();
    }

    @Test
    public void testBackwardReleasesSavedTensors() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{4.0f, 5.0f, 6.0f}).withRequiresGrad(true);

        Tensor c = TensorOps.mul(a, b);
        assertEquals("Graph should have 1 node after mul", 1, ComputationGraph.current().size());

        ComputationGraph.GraphNode node = ComputationGraph.current().getNodes().get(0);
        GradFunctions.MulGrad mulGrad = (GradFunctions.MulGrad) node.gradFn();
        assertTrue("Should have saved tensors before backward", mulGrad.savedTensors().size() == 2);

        c.backward();

        assertTrue("Saved tensors should be cleared after backward", mulGrad.savedTensors().isEmpty());
    }

    @Test
    public void testNoGradScopePreventsGraphRecording() {
        ComputationGraph.current().clear();
        
        assertEquals("Graph should be empty initially", 0, ComputationGraph.current().size());
        
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            assertFalse("Grad should be disabled in noGrad scope", ComputationGraph.isGradEnabled());
            
            Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f}).withRequiresGrad(true);
            Tensor b = Tensor.fromArray(backend, new float[]{3.0f, 4.0f}).withRequiresGrad(true);
            
            Tensor c = TensorOps.add(a, b);
            Tensor d = TensorOps.mul(c, a);
            
            assertEquals("Graph should remain empty in noGrad scope", 0, ComputationGraph.current().size());
        }
        
        assertTrue("Grad should be re-enabled after noGrad scope", ComputationGraph.isGradEnabled());
    }

    @Test
    public void testNoGradDoesNotAffectGraphAfterScope() {
        ComputationGraph.current().clear();
        
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f}).withRequiresGrad(true);
            Tensor b = TensorOps.add(a, a);
        }
        
        assertEquals("Graph should be empty after noGrad scope", 0, ComputationGraph.current().size());
        
        Tensor x = Tensor.fromArray(backend, new float[]{1.0f, 2.0f}).withRequiresGrad(true);
        Tensor y = TensorOps.add(x, x);
        
        assertEquals("Graph should record after noGrad scope", 1, ComputationGraph.current().size());
    }

    @Test
    public void testTrain100StepsGraphMemoryConstant() {
        ComputationGraph.current().clear();
        
        int numSteps = 100;
        List<Integer> graphSizes = new ArrayList<>();
        
        for (int step = 0; step < numSteps; step++) {
            Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);
            Tensor b = Tensor.fromArray(backend, new float[]{4.0f, 5.0f, 6.0f}).withRequiresGrad(true);
            
            Tensor c = TensorOps.mul(a, b);
            Tensor d = TensorOps.add(c, a);
            
            d.backward();
            
            graphSizes.add(ComputationGraph.current().size());
            
            ComputationGraph.current().clear();
            ComputationGraph.current().zeroGrad();
        }
        
        for (int i = 1; i < graphSizes.size(); i++) {
            assertEquals("Graph size should be constant across steps", 
                graphSizes.get(0), graphSizes.get(i));
        }
        
        assertTrue("Graph size should be 2 (mul + add)", graphSizes.get(0) <= 2);
    }

    @Test
    public void testMemoryUsageWithRequiresGradTrueVsFalse() {
        ComputationGraph.current().clear();
        
        int tensorSize = 1000;
        int numOps = 50;
        
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException e) {}
        MemoryUsage beforeWithGrad = memoryBean.getHeapMemoryUsage();
        
        for (int i = 0; i < numOps; i++) {
            Tensor a = Tensor.fromArray(backend, new float[tensorSize]).withRequiresGrad(true);
            Tensor b = Tensor.fromArray(backend, new float[tensorSize]).withRequiresGrad(true);
            Tensor c = TensorOps.add(a, b);
            Tensor d = TensorOps.mul(c, a);
        }
        
        int graphSizeWithGrad = ComputationGraph.current().size();
        
        MemoryUsage afterWithGrad = memoryBean.getHeapMemoryUsage();
        long memoryWithGrad = afterWithGrad.getUsed() - beforeWithGrad.getUsed();
        
        ComputationGraph.current().clear();
        
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException e) {}
        MemoryUsage beforeNoGrad = memoryBean.getHeapMemoryUsage();
        
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            for (int i = 0; i < numOps; i++) {
                Tensor a = Tensor.fromArray(backend, new float[tensorSize]).withRequiresGrad(true);
                Tensor b = Tensor.fromArray(backend, new float[tensorSize]).withRequiresGrad(true);
                Tensor c = TensorOps.add(a, b);
                Tensor d = TensorOps.mul(c, a);
            }
        }
        
        int graphSizeNoGrad = ComputationGraph.current().size();
        
        MemoryUsage afterNoGrad = memoryBean.getHeapMemoryUsage();
        long memoryNoGrad = afterNoGrad.getUsed() - beforeNoGrad.getUsed();
        
        assertEquals("Graph should have recorded operations with grad enabled", 
            numOps * 2, graphSizeWithGrad);
        assertEquals("Graph should be empty with noGrad", 0, graphSizeNoGrad);
        
        assertTrue("Graph with grad should have more nodes than without", 
            graphSizeWithGrad > graphSizeNoGrad);
    }

    @Test
    public void testBackwardRetainGraphParameter() {
        ComputationGraph.current().clear();
        
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{4.0f, 5.0f, 6.0f}).withRequiresGrad(true);
        
        Tensor c = TensorOps.mul(a, b);
        
        int graphSizeBeforeBackward = ComputationGraph.current().size();
        
        ComputationGraph.current().backward(c, null, false);
        
        int graphSizeAfterBackwardNoRetain = ComputationGraph.current().size();
        
        assertEquals("Graph should be cleared when retainGraph=false", 0, graphSizeAfterBackwardNoRetain);
        
        ComputationGraph.current().clear();
        
        Tensor x = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);
        Tensor y = Tensor.fromArray(backend, new float[]{4.0f, 5.0f, 6.0f}).withRequiresGrad(true);
        
        Tensor z = TensorOps.mul(x, y);
        
        ComputationGraph.current().backward(z, null, true);
        
        int graphSizeAfterBackwardRetain = ComputationGraph.current().size();
        
        assertEquals("Graph should be retained when retainGraph=true", 
            graphSizeBeforeBackward, graphSizeAfterBackwardRetain);
    }

    @Test
    public void testReleaseSavedTensorsMethod() {
        ComputationGraph.current().clear();
        
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{4.0f, 5.0f, 6.0f}).withRequiresGrad(true);
        
        Tensor c = TensorOps.mul(a, b);
        
        ComputationGraph.GraphNode node = ComputationGraph.current().getNodes().get(0);
        GradFunctions.MulGrad mulGrad = (GradFunctions.MulGrad) node.gradFn();
        
        assertTrue("Should have saved tensors before release", mulGrad.savedTensors().size() == 2);
        
        ComputationGraph.current().releaseSavedTensors();
        
        assertTrue("Saved tensors should be cleared after releaseSavedTensors", 
            mulGrad.savedTensors().isEmpty());
        
        assertEquals("Graph structure should still exist after releaseSavedTensors", 
            1, ComputationGraph.current().size());
    }

    @Test
    public void testCheckpointMemorySavingsEstimate() {
        int numLayers = 10;
        long activationSizeBytes = 1024 * 1024;
        int checkpointEvery = 2;
        
        CheckpointFunction.MemoryStats stats = CheckpointFunction.estimateMemorySavings(
            numLayers, activationSizeBytes, checkpointEvery);
        
        long fullMemory = numLayers * activationSizeBytes;
        
        assertTrue("Peak memory with checkpoint should be less than full memory",
            stats.peakMemoryBytes() < fullMemory);
        
        assertTrue("Should have saved some memory",
            stats.savedActivationsBytes() > 0);
        
        assertEquals("Number of checkpoints should be ceil(numLayers/checkpointEvery)",
            5, stats.numCheckpoints());
    }

    @Test
    public void testClearResetsGraph() {
        Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f}).withRequiresGrad(true);
        Tensor b = Tensor.fromArray(backend, new float[]{3.0f, 4.0f}).withRequiresGrad(true);
        
        Tensor c = TensorOps.add(a, b);
        Tensor d = TensorOps.mul(c, a);
        
        assertTrue("Graph should have nodes before clear", ComputationGraph.current().size() > 0);
        
        ComputationGraph.current().clear();
        
        assertEquals("Graph should be empty after clear", 0, ComputationGraph.current().size());
        assertTrue("Graph should be empty after clear", ComputationGraph.current().isEmpty());
    }

    @Test
    public void testNestedNoGradScopes() {
        assertTrue("Grad should be enabled initially", ComputationGraph.isGradEnabled());
        
        try (ComputationGraph.NoGradScope scope1 = ComputationGraph.noGrad()) {
            assertFalse("Grad should be disabled in first scope", ComputationGraph.isGradEnabled());
            
            try (ComputationGraph.NoGradScope scope2 = ComputationGraph.noGrad()) {
                assertFalse("Grad should still be disabled in nested scope", ComputationGraph.isGradEnabled());
            }
            
            assertFalse("Grad should still be disabled after inner scope", ComputationGraph.isGradEnabled());
        }
        
        assertTrue("Grad should be re-enabled after all scopes", ComputationGraph.isGradEnabled());
    }

    @Test
    public void testMemoryLeakStress() {
        ComputationGraph.current().clear();
        
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException e) {}
        
        MemoryUsage before = memoryBean.getHeapMemoryUsage();
        long beforeUsed = before.getUsed();
        
        for (int i = 0; i < 100; i++) {
            Tensor a = Tensor.fromArray(backend, new float[]{1.0f, 2.0f, 3.0f}).withRequiresGrad(true);
            Tensor b = Tensor.fromArray(backend, new float[]{4.0f, 5.0f, 6.0f}).withRequiresGrad(true);
            
            Tensor c = TensorOps.add(a, b);
            Tensor d = TensorOps.mul(c, a);
            Tensor e = TensorOps.div(d, b);
            
            e.backward();
            
            ComputationGraph.current().clear();
            ComputationGraph.current().zeroGrad();
        }
        
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException e) {}
        
        MemoryUsage after = memoryBean.getHeapMemoryUsage();
        long afterUsed = after.getUsed();
        long heapGrowth = afterUsed - beforeUsed;
        
        long maxAcceptableGrowth = 10 * 1024 * 1024;
        
        assertTrue("Heap should not grow significantly after 100 iterations. Growth: " + 
            (heapGrowth / (1024 * 1024)) + " MB", heapGrowth < maxAcceptableGrowth);
    }

    /** Declines kernel artifacts so tensor ops take their CPU path. */
    private static class MockComputeBackend implements ComputeBackend {
        private final Map<Long, MockGpuBuffer> buffers = new HashMap<>();
        private final MockComputeQueue computeQueue = new MockComputeQueue(this);
        private final MockComputeQueue transferQueue = new MockComputeQueue(this);
        private volatile boolean closed = false;
        private long nextBufferId = 1;
        private static final AtomicLong aliveBufferCount = new AtomicLong(0);

        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                .deviceName("MockDevice")
                .totalVramBytes(1024 * 1024 * 1024)
                .computeUnits(8)
                .maxClockFrequencyMhz(1000)
                .maxWorkGroupSize(256)
                .supportsAsyncTransfer(true)
                .build();
        }

        @Override
        public ComputeQueue computeQueue() {
            return computeQueue;
        }

        @Override
        public ComputeQueue transferQueue() {
            return transferQueue;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            MockGpuBuffer buffer = new MockGpuBuffer(sizeInBytes, nextBufferId++);
            buffers.put(buffer.bufferId(), buffer);
            aliveBufferCount.incrementAndGet();
            return buffer;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return allocateBuffer(sizeInBytes);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String source, String buildOptions) {
            return new MockKernelHandle(kernelName);
        }

        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) {
            return false;
        }

        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            if (artifact == null) {
                throw new IllegalArgumentException("kernel artifact must not be null");
            }
            throw new UnsupportedKernelException("mock backend declines " + artifact.kind()
                    + " artifact for kernel " + artifact.kernelName());
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                for (MockGpuBuffer buffer : buffers.values()) {
                    buffer.close();
                }
                buffers.clear();
            }
        }

        MockGpuBuffer getBuffer(long id) {
            return buffers.get(id);
        }

        static long getAliveBufferCount() {
            return aliveBufferCount.get();
        }
    }

    private static class MockComputeQueue implements ComputeQueue {
        private final MockComputeBackend backend;
        private volatile boolean closed = false;

        MockComputeQueue(MockComputeBackend backend) {
            this.backend = backend;
        }

        @Override
        public QueueType getQueueType() {
            return QueueType.COMPUTE;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer buffer, long dstOffset, long sizeInBytes, byte[] data, GpuEvent... waitOnEvents) {
            MockGpuBuffer mockBuffer = backend.getBuffer(buffer.bufferId());
            if (mockBuffer != null) {
                mockBuffer.writeBytes(data, (int) dstOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer buffer, long dstOffset, long sizeInBytes, float[] data, GpuEvent... waitOnEvents) {
            MockGpuBuffer mockBuffer = backend.getBuffer(buffer.bufferId());
            if (mockBuffer != null) {
                mockBuffer.writeFloats(data, (int) dstOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer buffer, long srcOffset, long sizeInBytes, byte[] data, GpuEvent... waitOnEvents) {
            MockGpuBuffer mockBuffer = backend.getBuffer(buffer.bufferId());
            if (mockBuffer != null) {
                mockBuffer.readBytes(data, (int) srcOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer buffer, long srcOffset, long sizeInBytes, float[] data, GpuEvent... waitOnEvents) {
            MockGpuBuffer mockBuffer = backend.getBuffer(buffer.bufferId());
            if (mockBuffer != null) {
                mockBuffer.readFloats(data, (int) srcOffset);
            }
            return new MockGpuEvent();
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args, GpuEvent... waitOnEvents) {
            return new MockGpuEvent();
        }

        @Override
        public void flush() {
        }

        @Override
        public void finish() {
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockGpuBuffer implements GpuBuffer {
        private final byte[] data;
        private final long sizeInBytes;
        private final long bufferId;
        private volatile boolean closed = false;

        MockGpuBuffer(long sizeInBytes, long bufferId) {
            this.sizeInBytes = sizeInBytes;
            this.bufferId = bufferId;
            this.data = new byte[(int) sizeInBytes];
        }

        void writeBytes(byte[] src, int offset) {
            System.arraycopy(src, 0, data, offset, src.length);
        }

        void writeFloats(float[] src, int offset) {
            ByteBuffer buffer = ByteBuffer.wrap(data, offset, src.length * 4).order(ByteOrder.nativeOrder());
            for (float f : src) {
                buffer.putFloat(f);
            }
        }

        void readBytes(byte[] dst, int offset) {
            System.arraycopy(data, offset, dst, 0, dst.length);
        }

        void readFloats(float[] dst, int offset) {
            ByteBuffer buffer = ByteBuffer.wrap(data, offset, dst.length * 4).order(ByteOrder.nativeOrder());
            for (int i = 0; i < dst.length; i++) {
                dst[i] = buffer.getFloat();
            }
        }

        @Override
        public long sizeInBytes() {
            return sizeInBytes;
        }

        @Override
        public long bufferId() {
            return bufferId;
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                MockComputeBackend.aliveBufferCount.decrementAndGet();
            }
        }
    }

    private static class MockGpuEvent implements GpuEvent {
        private volatile boolean complete = true;
        private volatile boolean closed = false;

        @Override
        public void awaitCompletion() {
            complete = true;
        }

        @Override
        public boolean isComplete() {
            return complete;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockKernelHandle implements KernelHandle {
        private final String kernelName;

        MockKernelHandle(String kernelName) {
            this.kernelName = kernelName;
        }

        @Override
        public String kernelName() {
            return kernelName;
        }

        @Override
        public void close() {
        }
    }
}
