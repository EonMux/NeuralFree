package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;

/**
 * A pinning {@link DataLoader} uploads a batch's inputs and targets as one buffer each, its samples are slices of it
 * laid back to back, and {@link Trainer#collateInputs} views such samples without a copy; samples of differing shapes
 * are pinned one by one and collated by a copy, with the same values either way. The loader's backend and the
 * dataset's are two host backends, so every sample needs the upload.
 */
public class BatchPinningTest {

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
    public void aBatchIsUploadedAsOneBufferAndCollatedWithoutACopy() {
        float[][] rows = {{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}, {10f, 11f, 12f}};
        List<TensorDataset.Sample> batch = firstBatch(dataset(rows, 3), 4);

        Tensor firstInput = batch.get(0).input();
        for (TensorDataset.Sample sample : batch) {
            assertSame("the loader's backend", target, sample.input().backend());
            assertSame("one input buffer for the batch", firstInput.storage().gpuBuffer(),
                sample.input().storage().gpuBuffer());
        }
        assertNotSame("inputs and targets are separate uploads", firstInput.storage().gpuBuffer(),
            batch.get(0).target().storage().gpuBuffer());

        Tensor inputs = keep(Trainer.collateInputs(batch));
        assertEquals(Shape.of(4, 3), inputs.shape());
        assertSame("collated as a view of the batch buffer", firstInput.storage().gpuBuffer(),
            inputs.storage().gpuBuffer());
        assertArrayEquals(flat(rows), inputs.toFloatArray(), 0f);
        Tensor targets = keep(Trainer.collateTargets(batch));
        assertArrayEquals(new float[]{0f, 1f, 2f, 3f}, targets.toFloatArray(), 0f);

        for (TensorDataset.Sample sample : batch) {
            sample.close();
        }
        assertArrayEquals("the view outlives the samples", flat(rows), inputs.toFloatArray(), 0f);
    }

    @Test
    public void samplesOutOfOrderAreCollatedByACopy() {
        float[][] rows = {{1f, 2f}, {3f, 4f}, {5f, 6f}};
        List<TensorDataset.Sample> batch = firstBatch(dataset(rows, 2), 3);
        List<TensorDataset.Sample> reversed = new ArrayList<>(batch);
        Collections.reverse(reversed);

        Tensor inputs = keep(Trainer.collateInputs(reversed));
        assertNotSame("not adjacent in this order", batch.get(0).input().storage().gpuBuffer(),
            inputs.storage().gpuBuffer());
        assertArrayEquals(new float[]{5f, 6f, 3f, 4f, 1f, 2f}, inputs.toFloatArray(), 0f);
        for (TensorDataset.Sample sample : batch) {
            sample.close();
        }
    }

    @Test
    public void samplesOfDifferentShapesArePinnedOneByOne() {
        Tensor[] inputs = {
            keep(Tensor.fromArray(source, new float[]{1f, 2f})),
            keep(Tensor.fromArray(source, new float[]{3f, 4f, 5f}))};
        Tensor[] targets = {
            keep(Tensor.fromArray(source, new float[]{0f})),
            keep(Tensor.fromArray(source, new float[]{1f}))};
        List<TensorDataset.Sample> batch = firstBatch(new TensorDataset(inputs, targets), 2);

        assertNotSame(batch.get(0).input().storage().gpuBuffer(), batch.get(1).input().storage().gpuBuffer());
        assertArrayEquals(new float[]{1f, 2f}, batch.get(0).input().toFloatArray(), 0f);
        assertArrayEquals(new float[]{3f, 4f, 5f}, batch.get(1).input().toFloatArray(), 0f);
        assertSame("the targets still share one shape and one upload", batch.get(0).target().storage().gpuBuffer(),
            batch.get(1).target().storage().gpuBuffer());
        for (TensorDataset.Sample sample : batch) {
            sample.close();
        }
    }

    private TensorDataset dataset(float[][] rows, int width) {
        Tensor[] inputs = new Tensor[rows.length];
        Tensor[] targets = new Tensor[rows.length];
        for (int i = 0; i < rows.length; i++) {
            assertEquals(width, rows[i].length);
            inputs[i] = keep(Tensor.fromArray(source, rows[i]));
            targets[i] = keep(Tensor.fromArray(source, new float[]{i}));
        }
        return new TensorDataset(inputs, targets);
    }

    private List<TensorDataset.Sample> firstBatch(TensorDataset data, int size) {
        DataLoader<TensorDataset.Sample> loader = DataLoader.<TensorDataset.Sample>builder(data)
            .batchSize(size).pinMemory(true).backend(target).build();
        owned.add(loader);
        return loader.iterator().next();
    }

    private static float[] flat(float[][] rows) {
        int n = 0;
        for (float[] r : rows) {
            n += r.length;
        }
        float[] out = new float[n];
        int k = 0;
        for (float[] r : rows) {
            System.arraycopy(r, 0, out, k, r.length);
            k += r.length;
        }
        return out;
    }

    private Tensor keep(Tensor t) {
        owned.add(t);
        return t;
    }
}
