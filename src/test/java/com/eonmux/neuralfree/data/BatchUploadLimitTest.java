package com.eonmux.neuralfree.data;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * A pinning {@link DataLoader} uploads one role of a batch as one buffer only while its bytes fit one upload
 * ({@link DataLoader#MAX_BATCH_UPLOAD_BYTES}, a heap buffer's capacity); a role above the limit is pinned sample by
 * sample with the same values. The limit is lowered through the package's builder seam so the test needs no 2 GiB
 * batch, and the size rule itself is checked at the real limit without allocating.
 */
public class BatchUploadLimitTest {

    private ComputeBackend source;
    private ComputeBackend target;
    private final List<AutoCloseable> owned = new ArrayList<>();

    @Before
    public void setUp() {
        source = TrainParity.hostBackend();
        target = TrainParity.hostBackend();
    }

    @After
    public void tearDown() throws Exception {
        for (int i = owned.size() - 1; i >= 0; i--) {
            owned.get(i).close();
        }
        target.close();
        source.close();
    }

    @Test
    public void aRoleAboveTheUploadLimitIsPinnedSampleBySample() {
        float[][] rows = {{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}, {10f, 11f, 12f}};
        List<TensorDataset.Sample> batch = firstBatch(dataset(rows), 4, 2L * 3 * Float.BYTES);

        for (int i = 0; i < rows.length; i++) {
            TensorDataset.Sample sample = batch.get(i);
            assertSame("the loader's backend", target, sample.input().backend());
            assertArrayEquals("input " + i, rows[i], sample.input().toFloatArray(), 0f);
            assertArrayEquals("target " + i, new float[]{i}, sample.target().toFloatArray(), 0f);
            for (int j = 0; j < i; j++) {
                assertNotSame("inputs " + j + " and " + i + " are separate uploads",
                    batch.get(j).input().storage().gpuBuffer(), sample.input().storage().gpuBuffer());
            }
            assertSame("the targets fit the limit and share one upload", batch.get(0).target().storage().gpuBuffer(),
                sample.target().storage().gpuBuffer());
        }
        closeAll(batch);
    }

    @Test
    public void aRoleAtTheUploadLimitGoesUpAsOneBuffer() {
        float[][] rows = {{1f, 2f}, {3f, 4f}, {5f, 6f}};
        List<TensorDataset.Sample> batch = firstBatch(dataset(rows), 3, 3L * 2 * Float.BYTES);
        for (TensorDataset.Sample sample : batch) {
            assertSame(batch.get(0).input().storage().gpuBuffer(), sample.input().storage().gpuBuffer());
        }
        closeAll(batch);
    }

    @Test
    public void theSizeRuleHoldsAtTheRealLimitWithoutOverflow() {
        long max = DataLoader.MAX_BATCH_UPLOAD_BYTES;
        assertTrue(DataLoader.fitsOneUpload(max / 2, 2, max));
        assertFalse("2 GiB of one role", DataLoader.fitsOneUpload(1L << 30, 2, max));
        assertFalse("a product that overflows a long", DataLoader.fitsOneUpload(Long.MAX_VALUE / 2, 3, max));
        assertFalse(DataLoader.fitsOneUpload(1, 0, max));
    }

    @Test
    public void aLimitOutsideTheBufferCapacityIsRefused() {
        TensorDataset data = dataset(new float[][]{{1f}});
        assertThrows(IllegalArgumentException.class, () -> DataLoader.<TensorDataset.Sample>builder(data)
            .pinMemory(true).backend(target).maxBatchUploadBytes(0).build());
        assertThrows(IllegalArgumentException.class, () -> DataLoader.<TensorDataset.Sample>builder(data)
            .pinMemory(true).backend(target).maxBatchUploadBytes(DataLoader.MAX_BATCH_UPLOAD_BYTES + 1).build());
    }

    private TensorDataset dataset(float[][] rows) {
        Tensor[] inputs = new Tensor[rows.length];
        Tensor[] targets = new Tensor[rows.length];
        for (int i = 0; i < rows.length; i++) {
            inputs[i] = keep(Tensor.fromArray(source, rows[i]));
            targets[i] = keep(Tensor.fromArray(source, new float[]{i}));
        }
        return new TensorDataset(inputs, targets);
    }

    private List<TensorDataset.Sample> firstBatch(TensorDataset data, int size, long limit) {
        DataLoader<TensorDataset.Sample> loader = DataLoader.<TensorDataset.Sample>builder(data)
            .batchSize(size).pinMemory(true).backend(target).maxBatchUploadBytes(limit).build();
        owned.add(loader);
        return loader.iterator().next();
    }

    private static void closeAll(List<TensorDataset.Sample> batch) {
        for (TensorDataset.Sample sample : batch) {
            sample.close();
        }
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }
}
