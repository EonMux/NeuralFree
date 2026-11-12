package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.tensor.gpu.TransferCountingBackend;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The device trainer's residency on every GPU API: the optimizer steps on the device with the arithmetic of the host
 * update, a run trains to the host trainer's losses, windows move no parameter or gradient between host and device,
 * the host arrays follow the device copy when read and are uploaded when written, and gradients accumulate across
 * steps as the host trainer's do. Skipped where an API or the {@code train} kernels are unavailable.
 */
@RunWith(Parameterized.class)
public class GpuFinetuneResidencyTest {

    private static final int N_CTX = 16;
    private static final int[] TOKENS = {1, 7, 19, 42, 77, 100, 128, 201, 3, 55, 240, 17, 88, 129, 260, 301};
    private static final int[] LABELS = {7, 19, 42, 77, 100, 128, 201, 3, 55, 240, 17, 88, 129, 260, 301, 44};

    private static Path file;
    private static GGUFModel model;

    private final TrainParity.Api api;
    private TrainParity.Session session;
    private ComputeBackend backend;

    public GpuFinetuneResidencyTest(TrainParity.Api api) {
        this.api = api;
    }

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> apis() {
        return TrainParity.apis();
    }

    @BeforeClass
    public static void openModel() throws IOException {
        ComputationGraph.current().clear();
        file = SyntheticModels.llama().writeTempFile();
        model = new GGUFModel(file.toString(), EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(N_CTX)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build());
    }

    @AfterClass
    public static void closeModel() throws IOException {
        if (model != null) {
            model.close();
        }
        if (file != null) {
            Files.deleteIfExists(file);
        }
    }

    @Before
    public void openBackend() {
        session = TrainParity.open(api);
        backend = session.backend();
    }

    @After
    public void closeBackend() {
        ComputationGraph.current().clear();
        if (session != null) {
            session.close();
        }
    }

    /**
     * The device AdamW against the host element loop on the gradients the device computed. {@code TR_AdamStep_F32}
     * runs the loop's operations in its order, so on CUDA, whose train library is built with IEEE division and square
     * root, no contraction and no flush to zero, every parameter has the loop's bits; the other APIs may round the
     * division and square root inexactly (the division tolerance class), so they are held to a relative bound.
     */
    @Test
    public void theDeviceAdamWStepsAsTheHostLoopOnTheSameGradients() throws IOException {
        float lr = 1e-3f;
        float wd = 0.1f;
        try (GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, backend);
             FinetuneOptimizer optimizer = new FinetuneOptimizer(gpu, FinetuneOptimizer.Kind.ADAMW, wd)) {
            assertEquals(FinetuneOptimizer.Placement.DEVICE, optimizer.placement());
            List<TrainableModel.Parameter> params = gpu.parameters();
            List<float[]> w = new ArrayList<>();
            List<float[]> m = new ArrayList<>();
            List<float[]> v = new ArrayList<>();
            for (TrainableModel.Parameter p : params) {
                w.add(p.values());
                m.add(new float[p.size()]);
                v.add(new float[p.size()]);
            }
            for (int step = 1; step <= 3; step++) {
                gpu.zeroGrad();
                gpu.step(TOKENS, LABELS);
                List<float[]> grads = new ArrayList<>();
                for (TrainableModel.Parameter p : params) {
                    grads.add(p.grad().clone());
                }
                optimizer.step(lr);
                float bc1 = 1f - (float) Math.pow(FinetuneOptimizer.BETA1, step);
                float bc2 = 1f - (float) Math.pow(FinetuneOptimizer.BETA2, step);
                for (int i = 0; i < params.size(); i++) {
                    float[] expected = w.get(i);
                    referenceAdamW(expected, grads.get(i), m.get(i), v.get(i), lr, wd, bc1, bc2);
                    float[] actual = params.get(i).values();
                    String what = params.get(i).name() + " after step " + step;
                    if (api == TrainParity.Api.CUDA) {
                        TrainParity.assertBitIdentical(what, expected, actual);
                    } else {
                        for (int k = 0; k < expected.length; k++) {
                            double allowed = 1e-6 * (Math.abs(expected[k]) + lr);
                            assertEquals(what + " [" + k + "]", expected[k], actual[k], allowed);
                        }
                    }
                    System.arraycopy(actual, 0, expected, 0, expected.length);
                }
            }
        }
    }

    /** {@link FinetuneOptimizer}'s AdamW element loop, in place on {@code w}, {@code m} and {@code v}. */
    private static void referenceAdamW(float[] w, float[] g, float[] m, float[] v, float lr, float wd, float bc1,
                                       float bc2) {
        float stepSize = lr / bc1;
        for (int i = 0; i < w.length; i++) {
            w[i] -= lr * wd * w[i];
            m[i] = FinetuneOptimizer.BETA1 * m[i] + (1f - FinetuneOptimizer.BETA1) * g[i];
            v[i] = FinetuneOptimizer.BETA2 * v[i] + (1f - FinetuneOptimizer.BETA2) * g[i] * g[i];
            w[i] -= stepSize * m[i] / ((float) Math.sqrt(v[i] / bc2) + FinetuneOptimizer.EPS);
        }
    }

    @Test
    public void aRunTrainsToTheHostTrainersLosses() throws IOException {
        for (FinetuneOptimizer.Kind kind : FinetuneOptimizer.Kind.values()) {
            FinetuneDataset data = FinetuneDataset.of(corpus(4 * N_CTX), N_CTX, 0.25);
            FinetuneRunner.Config config = new FinetuneRunner.Config(2, 1e-3f, 1e-4f, 4f, 0.1f, kind);
            List<FinetuneRunner.EpochStats> expected;
            try (TrainableModel host = TrainableModel.open(model, N_CTX)) {
                expected = FinetuneRunner.run(host, data, config, null);
            }
            List<FinetuneRunner.EpochStats> actual;
            try (GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, backend)) {
                actual = FinetuneRunner.run(gpu, data, config, null);
            }
            assertEquals(expected.size(), actual.size());
            for (int e = 0; e < expected.size(); e++) {
                String where = kind + " epoch " + (e + 1);
                assertEquals(where + " training loss", expected.get(e).trainLoss(), actual.get(e).trainLoss(), 2e-3);
                assertEquals(where + " validation loss", expected.get(e).validationLoss(),
                        actual.get(e).validationLoss(), 2e-3);
            }
        }
    }

    /**
     * Counts every byte the backend's queues move ({@link TransferCountingBackend}), not only the parameter mirror's:
     * after a warm-up run, an epoch of {@code W} windows of {@code n} positions writes exactly the token ids and the
     * labels ({@code 2 n} ints per window, uploaded together before the window's first kernel) and reads exactly the
     * loss, the {@code n} predicted ids and one check of the deferred index-error slots per window. Any parameter,
     * gradient or optimizer-state transfer, or an upload after an index-validating kernel (which adds an error-slot
     * check and a host synchronization), changes these totals.
     */
    @Test
    public void windowsSteppedOnTheDeviceMoveNoParameterOrGradient() throws IOException {
        FinetuneDataset data = FinetuneDataset.of(corpus(3 * N_CTX), N_CTX, 0.0);
        FinetuneRunner.Config config = new FinetuneRunner.Config(2, 1e-3f, 1e-4f, 4f, 0.1f,
                FinetuneOptimizer.Kind.ADAMW);
        try (TransferCountingBackend counting = new TransferCountingBackend(backend);
             GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, counting)) {
            FinetuneRunner.run(gpu, data, config, null);
            assertEquals("parameter bytes moved by the mirror during the run", 0L, gpu.hostTransferBytes());

            long written = counting.bytesWritten();
            long read = counting.bytesRead();
            FinetuneRunner.run(gpu, data, new FinetuneRunner.Config(1, 1e-3f, 1e-4f, 4f, 0.1f,
                    FinetuneOptimizer.Kind.ADAMW), null);
            int windows = data.training().size();
            long n = data.training().get(0).positions();
            assertEquals("bytes written by an epoch: the token ids and labels of each window",
                    windows * 2L * n * Integer.BYTES, counting.bytesWritten() - written);
            assertEquals("bytes read by an epoch: each window's loss, predicted ids and one index-error check",
                    windows * (n * Integer.BYTES + Float.BYTES + TrainGpu.errorSlotBytes()),
                    counting.bytesRead() - read);
            assertEquals(0L, gpu.hostTransferBytes());

            TrainableModel.Parameter p = gpu.parameter("blk.0.attn_q.weight");
            long before = counting.bytesRead();
            float[] values = p.values();
            assertEquals("a read moves the tensor once", (long) p.size() * Float.BYTES, gpu.hostTransferBytes());
            assertEquals("and the backend moves exactly those bytes", (long) p.size() * Float.BYTES,
                    counting.bytesRead() - before);
            assertArrayEquals(values, p.values(), 0f);
            assertEquals("a second read moves nothing", (long) p.size() * Float.BYTES, gpu.hostTransferBytes());
        }
    }

    @Test
    public void aWrittenHostArrayIsWhatTheNextWindowComputesWith() throws IOException {
        try (TrainableModel host = TrainableModel.open(model, N_CTX);
             GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, backend)) {
            for (Trainer t : new Trainer[]{host, gpu}) {
                float[] w = t.parameter("output_norm.weight").value();
                for (int i = 0; i < w.length; i++) {
                    w[i] = 0.25f + 0.5f * (i % 3);
                }
            }
            long before = gpu.hostTransferBytes();
            float[][] expected = host.logits(TOKENS);
            float[][] actual = gpu.logits(TOKENS);
            assertEquals("the written tensor is uploaded once",
                    before + (long) gpu.parameter("output_norm.weight").size() * Float.BYTES,
                    gpu.hostTransferBytes());
            double worst = 0;
            for (int t = 0; t < expected.length; t++) {
                for (int k = 0; k < expected[t].length; k++) {
                    worst = Math.max(worst, Math.abs(expected[t][k] - actual[t][k]));
                }
            }
            assertTrue("largest logit difference after the host write: " + worst, worst < 2e-3);
        }
    }

    @Test
    public void gradientsAccumulateAcrossStepsAsOnTheHostTrainer() throws IOException {
        try (GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, backend)) {
            TrainableModel.Parameter p = gpu.parameter("blk.1.ffn_up.weight");
            gpu.zeroGrad();
            gpu.step(TOKENS, LABELS);
            float[] once = p.grad().clone();
            gpu.step(TOKENS, LABELS);
            float[] twice = p.grad();
            float[] doubled = new float[once.length];
            for (int i = 0; i < once.length; i++) {
                doubled[i] = once[i] + once[i];
            }
            assertArrayEquals("the second step adds its gradient to the first", doubled, twice, 0f);
            gpu.zeroGrad();
            assertArrayEquals("zeroGrad clears the gradient", new float[once.length], p.grad(), 0f);
        }
    }

    @Test
    public void withoutTrainKernelsTheHostUpdateStepsTheMirroredArrays() throws IOException {
        ComputeBackend reference = TrainParity.hostBackend();
        try (TrainableModel host = TrainableModel.open(model, N_CTX);
             GpuTrainableModel cpuPath = GpuTrainableModel.open(model, N_CTX, reference);
             FinetuneOptimizer hostOptimizer = new FinetuneOptimizer(host, FinetuneOptimizer.Kind.ADAMW, 0.1f);
             FinetuneOptimizer mirrored = new FinetuneOptimizer(cpuPath, FinetuneOptimizer.Kind.ADAMW, 0.1f)) {
            assertEquals(FinetuneOptimizer.Placement.HOST, mirrored.placement());
            for (int step = 0; step < 2; step++) {
                host.zeroGrad();
                double expected = host.step(TOKENS, LABELS).loss();
                hostOptimizer.step(1e-3f);
                cpuPath.zeroGrad();
                double actual = cpuPath.step(TOKENS, LABELS).loss();
                mirrored.step(1e-3f);
                assertEquals("loss of step " + step, expected, actual, 2e-3);
            }
            float[] a = host.parameter("blk.0.attn_q.weight").values();
            float[] b = cpuPath.parameter("blk.0.attn_q.weight").values();
            double worst = 0;
            for (int i = 0; i < a.length; i++) {
                worst = Math.max(worst, Math.abs(a[i] - b[i]));
            }
            assertTrue("largest weight difference after two host-updated steps: " + worst, worst < 1e-4);
        } finally {
            reference.close();
        }
    }

    /** A deterministic token stream of {@code n} tokens inside the synthetic vocabulary. */
    private static int[] corpus(int n) {
        Random random = new Random(11);
        int[] tokens = new int[n];
        for (int i = 0; i < n; i++) {
            tokens[i] = random.nextInt(300);
        }
        return tokens;
    }
}
