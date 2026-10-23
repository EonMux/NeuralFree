package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.acceleration.backend.BackendResources;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.training.Callback;
import com.eonmux.neuralfree.training.Trainer;
import com.eonmux.neuralfree.training.TrainingContext;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

/**
 * Tensor lifetime in autograd and the trainer: training iterations do not grow the number of live device buffers and
 * leave no node in the graph, an epoch left early releases the batches its loader pinned ahead, a loss that reaches no
 * parameter is refused without leaking, a retained graph supports a further backward pass, released nodes close their
 * outputs, and gradients are closed when replaced or zeroed. Runs on {@link TrainParity#hostBackend()} behind a
 * wrapper that counts the buffers not yet closed. The recorded
 * operations are defined here ({@link Autograd#record} over results computed without recording), so the counts depend
 * only on the graph and the trainer.
 */
public class TensorLifetimeTest {

    private CountingBackend backend;

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        backend = new CountingBackend(TrainParity.hostBackend());
    }

    @After
    public void tearDown() {
        ComputationGraph.current().clear();
        backend.close();
    }

    // ------------------------------------------------------------------ trainer

    @Test
    public void trainingDoesNotGrowLiveBuffers() {
        ScaleModel model = new ScaleModel(backend, new float[]{0.5f, -1f, 2f});
        Trainer trainer = trainer(model);

        trainer.train(1);
        int live = backend.liveBuffers();
        trainer.train(3);

        assertEquals("live device buffers after three more epochs", live, backend.liveBuffers());
        assertEquals("graph released after every batch", 0, ComputationGraph.current().size());
        assertNull("gradients zeroed at the accumulation boundary", model.weight().grad());
    }

    @Test
    public void unconnectedLossIsRefusedAndLeavesNothingBehind() {
        ScaleModel model = new ScaleModel(backend, new float[]{0.5f, -1f, 2f});
        Trainer trainer = Trainer.builder()
            .model(model)
            .optimizer(new SGD(model.parameters(), 0.1f))
            .lossFunction(new UnrecordedSumLoss())
            .trainLoader(loader())
            .build();

        trainRefused(trainer);
        int live = backend.liveBuffers();
        for (int i = 0; i < 3; i++) {
            trainRefused(trainer);
        }

        assertEquals("live device buffers after three more refused epochs", live, backend.liveBuffers());
        assertEquals("the forward graph is removed with the batch", 0, ComputationGraph.current().size());
        assertNull("no gradient reaches the model", model.weight().grad());
    }

    /** {@link Trainer#train} refuses a loss whose backward pass gives the model's parameter no gradient. */
    private static void trainRefused(Trainer trainer) {
        try {
            trainer.train(1);
            fail("a loss that reaches no trainable parameter must be refused");
        } catch (IllegalStateException expected) {
            assertEquals(Trainer.NOT_CONNECTED, expected.getMessage());
        }
    }

    @Test
    public void evaluationRecordsNothingAndReleasesItsBatches() {
        ScaleModel model = new ScaleModel(backend, new float[]{0.5f, -1f, 2f});
        Trainer trainer = trainer(model);
        DataLoader<TensorDataset.Sample> loader = loader();

        trainer.evaluate(loader);
        int live = backend.liveBuffers();
        for (int i = 0; i < 3; i++) {
            trainer.evaluate(loader);
        }

        assertEquals("live device buffers after three more evaluations", live, backend.liveBuffers());
        assertEquals("evaluation records no node", 0, ComputationGraph.current().size());
    }

    @Test
    public void anEpochLeftEarlyReleasesTheBatchesLoadedAhead() throws InterruptedException {
        ScaleModel model = new ScaleModel(backend, new float[]{0.5f, -1f, 2f});
        trainer(model).train(1);
        int live = backend.liveBuffers();

        ComputeBackend source = TrainParity.hostBackend();
        try (TensorDataset data = sourceDataset(source, 8);
             DataLoader<TensorDataset.Sample> pinning = DataLoader.builder(data)
                 .batchSize(1).numWorkers(2).pinMemory(true).backend(backend).build()) {
            Trainer stopping = Trainer.builder()
                .model(model)
                .optimizer(new SGD(model.parameters(), 0.1f))
                .lossFunction(new SumLoss())
                .trainLoader(pinning)
                .addCallback(new Callback() {
                    private TrainingContext owner;

                    @Override
                    public void onTrainBegin(TrainingContext t) {
                        owner = t;
                    }

                    @Override
                    public void onBatchEnd(int batch, float loss) {
                        owner.requestStop();
                    }
                })
                .build();

            stopping.train(1);

            assertEquals("one step ran", 1, stopping.currentStep());
            awaitLiveBuffers(live);
            assertEquals("the pinned copies loaded ahead are released", live, backend.liveBuffers());
            assertEquals(0, ComputationGraph.current().size());
        } finally {
            source.close();
        }
    }

    /**
     * {@link Trainer#endPass} closes only the pass it is given: another pass over the same loader, in progress at the
     * same time, still yields every batch, and the pinned copies of both passes are released.
     */
    @Test
    public void endingOnePassLeavesTheOtherPassesOfItsLoaderRunning() throws InterruptedException {
        int baseline = backend.liveBuffers();
        int samples = 8;
        ComputeBackend source = TrainParity.hostBackend();
        try (TensorDataset data = sourceDataset(source, samples);
             DataLoader<TensorDataset.Sample> pinning = DataLoader.builder(data)
                 .batchSize(1).numWorkers(2).pinMemory(true).backend(backend).build()) {
            DataLoader.Pass<TensorDataset.Sample> left = pinning.pass();
            DataLoader.Pass<TensorDataset.Sample> kept = pinning.pass();
            closeSamples(left.next());
            closeSamples(kept.next());

            Trainer.endPass(left, null);

            int batches = 1;
            while (kept.hasNext()) {
                closeSamples(kept.next());
                batches++;
            }
            assertEquals("the pass still in progress yields every batch", samples, batches);
            try {
                left.next();
                fail("the ended pass hands out no further batch");
            } catch (IllegalStateException expected) {
                assertEquals("The DataLoader pass was closed", expected.getMessage());
            }
            awaitLiveBuffers(baseline);
            assertEquals("the pinned copies of both passes are released", baseline, backend.liveBuffers());
        } finally {
            source.close();
        }
    }

    // ------------------------------------------------------------------ graph

    @Test
    public void retainedGraphSupportsASecondBackward() {
        Tensor x = leaf(1f, 2f, 3f);
        Tensor w = leaf(4f, 5f, 6f);
        Tensor product = mul(x, w);
        Tensor loss = sum(product);

        ComputationGraph.current().backward(loss, true);
        assertEquals("retained graph keeps both nodes", 2, ComputationGraph.current().size());
        assertArrayEquals(new float[]{4f, 5f, 6f}, x.grad().toFloatArray(), 0f);
        Tensor firstGrad = x.grad();

        loss.backward();
        assertEquals("the default pass releases the graph", 0, ComputationGraph.current().size());
        assertArrayEquals(new float[]{8f, 10f, 12f}, x.grad().toFloatArray(), 0f);
        assertArrayEquals(new float[]{2f, 4f, 6f}, w.grad().toFloatArray(), 0f);

        assertClosed("the gradient replaced by accumulation", firstGrad);
        assertClosed("the released intermediate", product);
        assertArrayEquals("the root stays open", new float[]{32f}, loss.toFloatArray(), 0f);
    }

    @Test
    public void nodesARemainingNodeDependsOnAreKept() {
        Tensor x = leaf(1f, 2f, 3f);
        Tensor w = leaf(4f, 5f, 6f);
        Tensor shared = mul(x, w);
        Tensor first = sum(shared);
        Tensor second = sum(shared);

        first.backward();
        assertEquals("the shared product node stays for the second loss", 2, ComputationGraph.current().size());
        assertArrayEquals("the shared product stays open", new float[]{4f, 10f, 18f}, shared.toFloatArray(), 0f);

        second.backward();
        assertEquals(0, ComputationGraph.current().size());
        assertArrayEquals(new float[]{8f, 10f, 12f}, x.grad().toFloatArray(), 0f);
        assertClosed("the shared product once nothing uses it", shared);
    }

    @Test
    public void zeroGradClosesLeafGradients() {
        Tensor x = leaf(1f, 2f, 3f);
        Tensor w = leaf(4f, 5f, 6f);
        Tensor loss = sum(mul(x, w));
        loss.backward();
        int withGradients = backend.liveBuffers();

        ComputationGraph.current().zeroGrad();

        assertNull(x.grad());
        assertNull(w.grad());
        assertEquals("both gradient buffers released", withGradients - 2, backend.liveBuffers());
    }

    @Test
    public void everyGradientSlotGetsItsOwnReference() {
        int baseline = backend.liveBuffers();
        Tensor x = leaf(1f, 2f, 3f);
        Tensor y = leaf(4f, 5f, 6f);
        Tensor loss = sum(passThroughAdd(x, y));

        loss.backward();
        Tensor gx = x.grad();
        Tensor gy = y.grad();
        assertNotSame(gx, gy);
        gx.close();
        assertArrayEquals("closing one slot's gradient leaves the other intact", new float[]{1f, 1f, 1f},
            gy.toFloatArray(), 0f);

        gy.close();
        loss.close();
        x.close();
        y.close();
        assertEquals("every buffer of the pass released", baseline, backend.liveBuffers());
    }

    // ------------------------------------------------------------------ fixtures

    private Trainer trainer(ScaleModel model) {
        return Trainer.builder()
            .model(model)
            .optimizer(new SGD(model.parameters(), 0.1f))
            .lossFunction(new SumLoss())
            .trainLoader(loader())
            .build();
    }

    private DataLoader<TensorDataset.Sample> loader() {
        float[][] inputs = {{1f, 2f, 3f}, {-1f, 0.5f, 2f}, {0f, 1f, -2f}, {3f, -3f, 1f}};
        Tensor[] in = new Tensor[inputs.length];
        Tensor[] out = new Tensor[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            in[i] = Tensor.fromArray(backend, inputs[i]);
            out[i] = Tensor.fromArray(backend, new float[]{0f});
        }
        return new DataLoader<>(new TensorDataset(in, out), 1, false);
    }

    /**
     * {@code samples} samples on {@code source}, a backend other than the counting one, so a pinning loader copies
     * every sample and the copies are counted.
     */
    private static TensorDataset sourceDataset(ComputeBackend source, int samples) {
        Tensor[] in = new Tensor[samples];
        Tensor[] out = new Tensor[samples];
        for (int i = 0; i < samples; i++) {
            in[i] = Tensor.fromArray(source, new float[]{i, 1f, -1f});
            out[i] = Tensor.fromArray(source, new float[]{0f});
        }
        return new TensorDataset(in, out);
    }

    private static void closeSamples(List<TensorDataset.Sample> batch) {
        for (TensorDataset.Sample sample : batch) {
            sample.close();
        }
    }

    /**
     * Waits up to ten seconds for the live buffer count to reach {@code expected}: loader workers of a closed pass
     * release the batches they finish loading after the close.
     */
    private void awaitLiveBuffers(int expected) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (backend.liveBuffers() != expected && System.nanoTime() < deadline) {
            Thread.sleep(10);
        }
    }

    private Tensor leaf(float... values) {
        Tensor raw = Tensor.fromArray(backend, values);
        Tensor leaf = raw.withRequiresGrad(true);
        raw.close();
        return leaf;
    }

    private static void assertClosed(String what, Tensor t) {
        try {
            t.toFloatArray();
            fail(what + " should be closed");
        } catch (IllegalStateException expected) {
            // closed tensors refuse reads
        }
    }

    /** {@code input * w} for inputs of shape {@code [1, n]}, with one parameter {@code w} of shape {@code [1, n]}. */
    static final class ScaleModel extends Module {
        private final Parameter w;

        ScaleModel(ComputeBackend backend, float[] init) {
            Tensor flat = Tensor.fromArray(backend, init);
            try {
                this.w = registerParameter("w", new Parameter(flat.reshape(1, init.length)));
            } finally {
                flat.close();
            }
        }

        Parameter weight() {
            return w;
        }

        @Override
        public Tensor forward(Tensor input) {
            return mul(input, w.tensor());
        }
    }

    /** The sum of the prediction; the target is not used. */
    static final class SumLoss implements LossFunction {
        @Override
        public Tensor compute(Tensor prediction, Tensor target) {
            return sum(prediction);
        }

        @Override
        public String name() {
            return "sum";
        }
    }

    /**
     * The sum of the prediction computed without recording, made a leaf that requires grad: the backward pass from it
     * fills its own gradient slot and reaches no parameter.
     */
    static final class UnrecordedSumLoss implements LossFunction {
        @Override
        public Tensor compute(Tensor prediction, Tensor target) {
            Tensor raw = Autograd.untracked(() -> TensorOps.sum(prediction));
            Tensor loss = raw.withRequiresGrad(true);
            if (loss != raw) {
                raw.close();
            }
            return loss;
        }

        @Override
        public String name() {
            return "unrecorded sum";
        }
    }

    // ------------------------------------------------------------------ recorded operations

    /**
     * The result {@code compute} gives without recording, recorded as the output of {@code fn} over {@code inputs}
     * when an input requires grad in grad mode (the result then requires grad and carries {@code fn}); otherwise the
     * unrecorded result.
     */
    private static Tensor recorded(Supplier<Tensor> compute, GradientFunction fn, Tensor... inputs) {
        List<Tensor> in = Arrays.asList(inputs);
        Tensor raw = Autograd.untracked(compute);
        if (!Autograd.needsGrad(in)) {
            return raw;
        }
        return Autograd.record(raw, in, out -> fn);
    }

    /** Elementwise product of two tensors of one shape. */
    static Tensor mul(Tensor a, Tensor b) {
        return recorded(() -> TensorOps.mul(a, b), new MulFn(a, b), a, b);
    }

    /** Sum of all elements. */
    static Tensor sum(Tensor a) {
        return recorded(() -> TensorOps.sum(a), new SumFn(a.shape()), a);
    }

    /** {@code a + b} whose backward returns the incoming gradient itself for both inputs. */
    static Tensor passThroughAdd(Tensor a, Tensor b) {
        return recorded(() -> TensorOps.add(a, b), new PassThroughFn(), a, b);
    }

    private static final class MulFn implements GradientFunction {
        private List<Tensor> saved;

        MulFn(Tensor a, Tensor b) {
            this.saved = Arrays.asList(a, b);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Arrays.asList(TensorOps.mul(gradOutput, saved.get(1)), TensorOps.mul(gradOutput, saved.get(0)));
        }

        @Override
        public List<Tensor> savedTensors() {
            return saved;
        }

        @Override
        public void clearSavedTensors() {
            saved = Collections.emptyList();
        }
    }

    private static final class SumFn implements GradientFunction {
        private final Shape inputShape;

        SumFn(Shape inputShape) {
            this.inputShape = inputShape;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            try (Tensor ones = Tensor.ones(gradOutput.backend(), inputShape)) {
                return Collections.singletonList(TensorOps.mul(ones, gradOutput));
            }
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }
    }

    private static final class PassThroughFn implements GradientFunction {
        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Arrays.asList(gradOutput, gradOutput);
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }
    }

    // ------------------------------------------------------------------ counting backend

    /** Delegates to a backend and counts the buffers allocated through it that are not closed yet. */
    private static final class CountingBackend implements ComputeBackend {
        private final ComputeBackend delegate;
        private final AtomicInteger live = new AtomicInteger();
        private final CountingQueue compute;
        private final CountingQueue transfer;

        CountingBackend(ComputeBackend delegate) {
            this.delegate = delegate;
            this.compute = new CountingQueue(delegate.computeQueue());
            this.transfer = new CountingQueue(delegate.transferQueue());
        }

        int liveBuffers() {
            return live.get();
        }

        @Override
        public BackendType backendType() {
            return delegate.backendType();
        }

        @Override
        public DeviceCapabilities capabilities() {
            return delegate.capabilities();
        }

        @Override
        public ComputeQueue computeQueue() {
            return compute;
        }

        @Override
        public ComputeQueue transferQueue() {
            return transfer;
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
            return new CountedBuffer(delegate.allocateBuffer(sizeInBytes, readOnly), live);
        }

        @Override
        public GpuBuffer allocateBuffer(long sizeInBytes) {
            return new CountedBuffer(delegate.allocateBuffer(sizeInBytes), live);
        }

        @Override
        public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
            return delegate.compileKernel(kernelName, kernelSource, buildOptions);
        }

        @Override
        public KernelHandle loadKernel(KernelArtifact artifact) {
            return delegate.loadKernel(artifact);
        }

        @Override
        public boolean supportsArtifact(KernelArtifact.Kind kind) {
            return delegate.supportsArtifact(kind);
        }

        /** Releases the per-backend train state first, as the real backends do. */
        @Override
        public void close() {
            try {
                BackendResources.closeAll(this);
            } finally {
                delegate.close();
            }
        }
    }

    private static final class CountedBuffer implements GpuBuffer {
        private final GpuBuffer inner;
        private final AtomicInteger live;
        private boolean closed;

        CountedBuffer(GpuBuffer inner, AtomicInteger live) {
            this.inner = inner;
            this.live = live;
            live.incrementAndGet();
        }

        @Override
        public long sizeInBytes() {
            return inner.sizeInBytes();
        }

        @Override
        public long bufferId() {
            return inner.bufferId();
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                live.decrementAndGet();
                inner.close();
            }
        }
    }

    /** Unwraps counted buffers before handing each command to the delegate queue. */
    private static final class CountingQueue implements ComputeQueue {
        private final ComputeQueue delegate;

        CountingQueue(ComputeQueue delegate) {
            this.delegate = delegate;
        }

        private static GpuBuffer unwrap(GpuBuffer b) {
            return b instanceof CountedBuffer ? ((CountedBuffer) b).inner : b;
        }

        @Override
        public QueueType getQueueType() {
            return delegate.getQueueType();
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long dstOffset, long size, byte[] src, GpuEvent... waits) {
            return delegate.enqueueWriteBuffer(unwrap(dst), dstOffset, size, src, waits);
        }

        @Override
        public GpuEvent enqueueWriteBuffer(GpuBuffer dst, long dstOffset, long size, float[] src, GpuEvent... waits) {
            return delegate.enqueueWriteBuffer(unwrap(dst), dstOffset, size, src, waits);
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long srcOffset, long size, byte[] dst, GpuEvent... waits) {
            return delegate.enqueueReadBuffer(unwrap(src), srcOffset, size, dst, waits);
        }

        @Override
        public GpuEvent enqueueReadBuffer(GpuBuffer src, long srcOffset, long size, float[] dst, GpuEvent... waits) {
            return delegate.enqueueReadBuffer(unwrap(src), srcOffset, size, dst, waits);
        }

        @Override
        public GpuEvent enqueueKernel(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                      GpuEvent... waits) {
            Object[] unwrapped = args == null ? null : args.clone();
            if (unwrapped != null) {
                for (int i = 0; i < unwrapped.length; i++) {
                    if (unwrapped[i] instanceof GpuBuffer) {
                        unwrapped[i] = unwrap((GpuBuffer) unwrapped[i]);
                    }
                }
            }
            return delegate.enqueueKernel(kernel, global, local, unwrapped, waits);
        }

        @Override
        public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset, long size,
                                      GpuEvent... waits) {
            delegate.enqueueCopyBuffer(unwrap(src), srcOffset, unwrap(dst), dstOffset, size, waits);
        }

        @Override
        public GpuEvent enqueueMarker(GpuEvent... waits) {
            return delegate.enqueueMarker(waits);
        }

        @Override
        public void flush() {
            delegate.flush();
        }

        @Override
        public void finish() {
            delegate.finish();
        }

        @Override
        public void close() {
            delegate.close();
        }
    }
}
