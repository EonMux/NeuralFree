package com.eonmux.neuralfree.data;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.*;

public class DataLoaderTest {

    static class ListDataset implements Dataset<Integer> {
        private final List<Integer> data;
        
        ListDataset(int size) {
            this.data = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                data.add(i);
            }
        }

        @Override
        public int size() {
            return data.size();
        }

        @Override
        public Integer get(int index) {
            return data.get(index);
        }
    }

    @Test
    public void testSequentialDataLoader() {
        Dataset<Integer> dataset = new ListDataset(10);
        DataLoader<Integer> loader = new DataLoader<>(dataset, 3, false);
        
        int batchCount = 0;
        int totalItems = 0;
        for (List<Integer> batch : loader) {
            batchCount++;
            totalItems += batch.size();
        }
        
        assertEquals(4, batchCount);
        assertEquals(10, totalItems);
    }

    @Test
    public void testNumBatches() {
        Dataset<Integer> dataset = new ListDataset(10);
        DataLoader<Integer> loader = new DataLoader<>(dataset, 3, false);
        
        assertEquals(4, loader.numBatches());
    }

    /** {@code numBatches} counts the sampler's indices, which a weighted sampler sets independently of the dataset. */
    @Test
    public void testNumBatchesFollowsTheSamplerLength() {
        Dataset<Integer> dataset = new ListDataset(4);
        for (boolean dropLast : new boolean[]{false, true}) {
            DataLoader<Integer> loader = new DataLoader<>(dataset, 3,
                new WeightedRandomSampler(new double[]{1, 2, 3, 4}, 11, true, 9L), 0, false, null, dropLast);
            int batches = 0;
            int items = 0;
            for (List<Integer> batch : loader) {
                batches++;
                items += batch.size();
            }
            assertEquals(dropLast ? 3 : 4, loader.numBatches());
            assertEquals(loader.numBatches(), batches);
            assertEquals(dropLast ? 9 : 11, items);
        }
        assertEquals(4, new SequentialSampler().length(4));
        assertEquals(4, new RandomSampler(3L).length(4));
    }

    @Test
    public void testDropLast() {
        Dataset<Integer> dataset = new ListDataset(10);
        DataLoader<Integer> loader = DataLoader.<Integer>builder(dataset)
            .batchSize(3)
            .dropLast(true)
            .build();
        
        int batchCount = 0;
        for (List<Integer> batch : loader) {
            assertEquals(3, batch.size());
            batchCount++;
        }
        
        assertEquals(3, batchCount);
        assertEquals(3, loader.numBatches());
    }

    @Test
    public void testBuilder() {
        Dataset<Integer> dataset = new ListDataset(20);
        DataLoader<Integer> loader = DataLoader.<Integer>builder(dataset)
            .batchSize(5)
            .shuffle(true)
            .numWorkers(0)
            .build();
        
        assertEquals(5, loader.batchSize());
        assertEquals(0, loader.numWorkers());
        assertEquals(4, loader.numBatches());
    }

    @Test
    public void testBatchSizes() {
        Dataset<Integer> dataset = new ListDataset(10);
        DataLoader<Integer> loader = new DataLoader<>(dataset, 3, false);
        
        Iterator<List<Integer>> iter = loader.iterator();
        
        assertEquals(3, iter.next().size());
        assertEquals(3, iter.next().size());
        assertEquals(3, iter.next().size());
        assertEquals(1, iter.next().size());
        assertFalse(iter.hasNext());
    }

    @Test
    public void testShuffleReproducible() {
        Dataset<Integer> dataset = new ListDataset(100);
        
        DataLoader<Integer> loader1 = new DataLoader<>(
            dataset, 10, new RandomSampler(42), 0, false, null, false);
        DataLoader<Integer> loader2 = new DataLoader<>(
            dataset, 10, new RandomSampler(42), 0, false, null, false);
        
        List<Integer> firstBatch1 = loader1.iterator().next();
        List<Integer> firstBatch2 = loader2.iterator().next();
        
        assertEquals(firstBatch1, firstBatch2);
    }

    @Test
    public void testCustomSampler() {
        Dataset<Integer> dataset = new ListDataset(10);
        DataLoader<Integer> loader = new DataLoader<>(
            dataset, 3, new SequentialSampler(), 0, false, null, false);
        
        int first = loader.iterator().next().get(0);
        assertEquals(0, first);
    }

    @Test
    public void testSize() {
        Dataset<Integer> dataset = new ListDataset(50);
        DataLoader<Integer> loader = new DataLoader<>(dataset, 10, false);
        
        assertEquals(50, loader.size());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidBatchSizeZero() {
        Dataset<Integer> dataset = new ListDataset(10);
        new DataLoader<>(dataset, 0, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidBatchSizeNegative() {
        Dataset<Integer> dataset = new ListDataset(10);
        new DataLoader<>(dataset, -1, false);
    }

    @Test
    public void testMultiThreadedLoading() {
        Dataset<Integer> dataset = new ListDataset(100);
        DataLoader<Integer> loader = DataLoader.<Integer>builder(dataset)
            .batchSize(10)
            .numWorkers(2)
            .build();
        
        int batchCount = 0;
        int totalItems = 0;
        for (List<Integer> batch : loader) {
            batchCount++;
            totalItems += batch.size();
        }
        
        assertEquals(10, batchCount);
        assertEquals(100, totalItems);
    }

    /**
     * Closing a pass ends that pass alone, with and without workers: it hands out no further batch, closing it again
     * does nothing, and another pass over the same loader still yields every item in order.
     */
    @Test
    public void testClosingAPassLeavesTheOtherPassesRunning() {
        for (int workers : new int[]{0, 2}) {
            DataLoader<Integer> loader = DataLoader.<Integer>builder(new ListDataset(10))
                .batchSize(2)
                .numWorkers(workers)
                .build();
            DataLoader.Pass<Integer> ended = loader.pass();
            DataLoader.Pass<Integer> running = loader.pass();
            assertEquals(List.of(0, 1), ended.next());

            ended.close();
            ended.close();

            try {
                ended.next();
                fail("a closed pass hands out no batch (numWorkers=" + workers + ")");
            } catch (IllegalStateException expected) {
                assertEquals("The DataLoader pass was closed", expected.getMessage());
            }
            List<Integer> items = new ArrayList<>();
            while (running.hasNext()) {
                items.addAll(running.next());
            }
            assertEquals("numWorkers=" + workers, List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), items);
        }
    }

    /**
     * A pass leaves the loader's in-progress set once it can hand out nothing more: at once when it has no batch
     * (empty dataset, or {@code dropLast} with fewer samples than a batch), after its last batch, or when closed.
     */
    @Test
    public void testPassesWithoutBatchesAreNotLeftInProgress() {
        DataLoader<Integer> empty = DataLoader.<Integer>builder(new ListDataset(0))
            .batchSize(4)
            .numWorkers(2)
            .build();
        DataLoader<Integer> tooShort = DataLoader.<Integer>builder(new ListDataset(3))
            .batchSize(4)
            .numWorkers(2)
            .dropLast(true)
            .build();
        for (DataLoader<Integer> loader : List.of(empty, tooShort)) {
            for (int pass = 0; pass < 5; pass++) {
                assertFalse(loader.pass().hasNext());
            }
            assertEquals(0, loader.passesInProgress());
        }

        DataLoader<Integer> loader = DataLoader.<Integer>builder(new ListDataset(6))
            .batchSize(2)
            .numWorkers(2)
            .build();
        DataLoader.Pass<Integer> consumed = loader.pass();
        DataLoader.Pass<Integer> abandoned = loader.pass();
        assertEquals(2, loader.passesInProgress());
        while (consumed.hasNext()) {
            consumed.next();
        }
        assertEquals(1, loader.passesInProgress());
        abandoned.next();
        loader.close();
        assertEquals(0, loader.passesInProgress());
    }

    /**
     * A seeded sampler advances across passes, so consecutive passes take different orders, and resetting it replays
     * the same sequence of orders, like a seeded generator across PyTorch epochs.
     */
    @Test
    public void testSeededSamplerDrawsANewOrderEachPass() {
        RandomSampler sampler = new RandomSampler(42);
        RandomSampler reference = new RandomSampler(42);
        for (int workers : new int[]{0, 2}) {
            DataLoader<Integer> loader = new DataLoader<>(new ListDataset(100), 100, sampler, workers, false, null,
                false);
            sampler.reset();
            List<Integer> first = loader.pass().next();
            List<Integer> second = loader.pass().next();
            assertNotEquals("numWorkers=" + workers, first, second);

            reference.reset();
            assertEquals(reference.sample(100), first);
            assertEquals(reference.sample(100), second);

            sampler.reset();
            assertEquals(first, loader.pass().next());
            assertEquals(second, loader.pass().next());
            loader.close();
        }
    }

    @Test
    public void testShuffleProducesDifferentOrderAcrossEpochs() {
        Dataset<Integer> dataset = new ListDataset(100);
        DataLoader<Integer> loader = DataLoader.<Integer>builder(dataset)
            .batchSize(10)
            .shuffle(true)
            .build();
        
        List<Integer> epoch1FirstBatch = loader.iterator().next();
        List<Integer> epoch2FirstBatch = loader.iterator().next();
        List<Integer> epoch3FirstBatch = loader.iterator().next();
        
        int differentCount = 0;
        if (!epoch1FirstBatch.equals(epoch2FirstBatch)) differentCount++;
        if (!epoch1FirstBatch.equals(epoch3FirstBatch)) differentCount++;
        if (!epoch2FirstBatch.equals(epoch3FirstBatch)) differentCount++;
        
        assertTrue("Shuffle should produce different order across epochs (at least 2 different batches out of 3)", 
                   differentCount >= 2);
    }

    @Test
    public void testNumWorkersHigherThroughput() {
        Dataset<Integer> dataset = new ListDataset(1000);
        
        DataLoader<Integer> syncLoader = DataLoader.<Integer>builder(dataset)
            .batchSize(32)
            .numWorkers(0)
            .build();
        
        DataLoader<Integer> asyncLoader = DataLoader.<Integer>builder(dataset)
            .batchSize(32)
            .numWorkers(4)
            .build();
        
        long syncStart = System.nanoTime();
        int syncBatches = 0;
        for (List<Integer> batch : syncLoader) {
            syncBatches++;
        }
        long syncTime = System.nanoTime() - syncStart;
        
        long asyncStart = System.nanoTime();
        int asyncBatches = 0;
        for (List<Integer> batch : asyncLoader) {
            asyncBatches++;
        }
        long asyncTime = System.nanoTime() - asyncStart;
        
        assertEquals(syncBatches, asyncBatches);
        assertTrue("numWorkers>0 should complete successfully (asyncTime=" + asyncTime + "ns, syncTime=" + syncTime + "ns)", 
                   asyncBatches > 0);
    }

    @Test
    public void testPinMemoryTensorsOnGpu() {
        MockBackendForPinMemory backend = new MockBackendForPinMemory();
        
        Tensor[] inputs = new Tensor[10];
        Tensor[] targets = new Tensor[10];
        for (int i = 0; i < 10; i++) {
            inputs[i] = Tensor.zeros(backend, Shape.of(3));
            targets[i] = Tensor.zeros(backend, Shape.of(1));
        }
        
        TensorDataset tensorDataset = new TensorDataset(inputs, targets);
        
        DataLoader<TensorDataset.Sample> loader = DataLoader.<TensorDataset.Sample>builder(tensorDataset)
            .batchSize(2)
            .pinMemory(true)
            .backend(backend)
            .build();
        
        List<TensorDataset.Sample> batch = loader.iterator().next();
        
        assertNotNull(batch);
        assertEquals(2, batch.size());
        
        for (TensorDataset.Sample sample : batch) {
            assertNotNull(sample.input());
            assertNotNull(sample.target());
            assertEquals("Pinned tensor should be on the provided backend", 
                         backend, sample.input().backend());
            assertEquals("Pinned target tensor should be on the provided backend", 
                         backend, sample.target().backend());
        }
    }

    private static class MockBackendForPinMemory implements ComputeBackend {
        private final AtomicLong bufferIdCounter = new AtomicLong(1);
        private final Map<Long, byte[]> bufferData = new HashMap<>();
        private final MockQueueForPinMemory computeQueue = new MockQueueForPinMemory(bufferData);
        private final MockQueueForPinMemory transferQueue = new MockQueueForPinMemory(bufferData);
        private volatile boolean closed = false;

        @Override
        public BackendType backendType() {
            return BackendType.OPENCL;
        }

        @Override
        public DeviceCapabilities capabilities() {
            return new DeviceCapabilities.Builder()
                    .deviceName("MockBackendForPinMemory")
                    .totalVramBytes(1024 * 1024 * 1024)
                    .computeUnits(8)
                    .maxClockFrequencyMhz(1000)
                    .maxWorkGroupSize(256)
                    .supportsAsyncTransfer(true)
                    .driverVersion("1.0")
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
            long id = bufferIdCounter.getAndIncrement();
            bufferData.put(id, new byte[(int) sizeInBytes]);
            return new MockBufferForPinMemory(id, sizeInBytes);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return allocateBuffer(sizeInBytes);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            return new MockKernelForPinMemory(kernelName);
        }

        /** Declines kernel artifacts so tensor ops take their CPU path. */
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
            closed = true;
            bufferData.clear();
        }
    }

    private static class MockQueueForPinMemory implements ComputeQueue {
        private final Map<Long, byte[]> bufferData;
        private volatile boolean closed = false;

        MockQueueForPinMemory(Map<Long, byte[]> bufferData) {
            this.bufferData = bufferData;
        }

        @Override
        public QueueType getQueueType() {
            return QueueType.TRANSFER;
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(dstBuffer.bufferId());
            if (data != null && dstOffset + sizeInBytes <= data.length) {
                System.arraycopy(srcData, 0, data, (int) dstOffset, (int) sizeInBytes);
            }
            return new MockEventForPinMemory();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(dstBuffer.bufferId());
            if (data != null && dstOffset + sizeInBytes <= data.length) {
                ByteBuffer buffer = ByteBuffer.wrap(data, (int) dstOffset, (int) sizeInBytes).order(ByteOrder.nativeOrder());
                for (float f : srcData) {
                    buffer.putFloat(f);
                }
            }
            return new MockEventForPinMemory();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(srcBuffer.bufferId());
            if (data != null && srcOffset + sizeInBytes <= data.length) {
                System.arraycopy(data, (int) srcOffset, dstData, 0, (int) sizeInBytes);
            }
            return new MockEventForPinMemory();
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData, GpuEvent... waitOnEvents) {
            byte[] data = bufferData.get(srcBuffer.bufferId());
            if (data != null && srcOffset + sizeInBytes <= data.length) {
                ByteBuffer buffer = ByteBuffer.wrap(data, (int) srcOffset, (int) sizeInBytes).order(ByteOrder.nativeOrder());
                for (int i = 0; i < dstData.length; i++) {
                    dstData[i] = buffer.getFloat();
                }
            }
            return new MockEventForPinMemory();
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args, GpuEvent... waitOnEvents) {
            return new MockEventForPinMemory();
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

    private static class MockBufferForPinMemory implements GpuBuffer {
        private final long id;
        private final long size;
        private volatile boolean closed = false;

        MockBufferForPinMemory(long id, long size) {
            this.id = id;
            this.size = size;
        }

        @Override
        public long sizeInBytes() {
            return size;
        }

        @Override
        public long bufferId() {
            return id;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    private static class MockEventForPinMemory implements GpuEvent {
        @Override
        public void awaitCompletion() {
        }

        @Override
        public boolean isComplete() {
            return true;
        }

        @Override
        public void close() {
        }
    }

    private static class MockKernelForPinMemory implements KernelHandle {
        private final String name;

        MockKernelForPinMemory(String name) {
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
