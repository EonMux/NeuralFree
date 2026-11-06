package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.testutil.RealModels;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.quant.W12Codec;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in timing of the lossless W12 row kernels on real model weights (environment
 * {@code NEURALFREE_BENCH_W12=1}): the 2-D projections of a bfloat16-representable GGUF from the model
 * directories on this machine (no downloads), encoded per row and timed against the BF16 matvec of the
 * same weights on CUDA and OpenCL. Prints each tensor's real escape rate and bytes per weight next to
 * the decode tok/s; asserts nothing.
 */
@RunWith(Parameterized.class)
public class W12RealGgufBenchmarkTest {

    private static final String SWITCH = "NEURALFREE_BENCH_W12";
    private static final int WARMUP = 20;
    private static final int RUNS = 60;
    private static final int MAX_TENSORS = 5;
    private static final long MAX_TENSOR_BYTES = 96L * 1024 * 1024;
    /** The GGUFs tried in order; the first present one is measured. All are F16 embedding models. */
    private static final String[] CANDIDATES = {
        RealModels.path("pplx-embed-v1-0.6b-GGUF/pplx-embed-v1-0.6b-f16.gguf"),
        RealModels.path("jina-v5-small-retrieval-GGUF/v5-small-retrieval-F16.gguf"),
        RealModels.path("Qwen3-VL-Embedding-2B-GGUF/Qwen.Qwen3-VL-Embedding-2B.f16.gguf"),
    };

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private boolean previousCuda;

    public W12RealGgufBenchmarkTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + SWITCH + "=1 to run", "1".equals(System.getenv(SWITCH)));
        previousCuda = HWAcceleration.enableCuda;
        try {
            backend = openBackend();
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
            scratch = GpuScratch.builder(backend, 1).slot(ScratchSlots.X, 10240L * 4).build();
            placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    private ComputeBackend openBackend() {
        if (api.equals("cuda")) {
            HWAcceleration.enableCuda = true;
            HWAcceleration accel = new HWAcceleration();
            hwAccel = accel;
            Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
            CUDevice device = accel.getCUDevices().get(0);
            return new CudaComputeBackend(device);
        }
        HWAcceleration accel = new HWAcceleration();
        hwAccel = accel;
        Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
        return new OclComputeBackend(accel.getCLDevices().get(0));
    }

    @After
    public void closeDevice() {
        if (ops != null) {
            ops.close();
            ops = null;
        }
        if (placeholders != null) {
            placeholders.close();
            placeholders = null;
        }
        if (scratch != null) {
            scratch.close();
            scratch = null;
        }
        if (backend != null) {
            try {
                backend.close();
            } catch (RuntimeException ignored) {
            }
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
        HWAcceleration.enableCuda = previousCuda;
    }

    /** One-token decode of every selected projection: W12 exact, the W12 Q view, and the BF16 matvec. */
    @Test
    public void decodeRealProjectionsAgainstTheBf16Matvec() throws Exception {
        File file = modelFile();
        List<GGUFTensorInfo> tensors = selectProjections(file);
        System.out.println("real " + api + " " + file.getName() + ": " + tensors.size() + " projections");
        try (GGUFFileParser parser = new GGUFFileParser(file.getPath())) {
            for (GGUFTensorInfo info : tensors) {
                measure(parser, info);
            }
        }
    }

    private static File modelFile() {
        for (String path : CANDIDATES) {
            File f = new File(path);
            if (f.isFile()) {
                return f;
            }
        }
        throw new AssertionError("no candidate GGUF under " + Arrays.toString(CANDIDATES));
    }

    /** The largest F16/BF16 2-D projections whose rows are whole W12 blocks, one per distinct shape. */
    private static List<GGUFTensorInfo> selectProjections(File file) throws Exception {
        try (GGUFFileParser parser = new GGUFFileParser(file.getPath())) {
            List<GGUFTensorInfo> all = new ArrayList<>();
            for (GGUFTensorInfo info : parser.getTensors()) {
                if (info.rank() == 2 && (info.type() == GGMLType.F16 || info.type() == GGMLType.BF16)
                        && info.dim(0) % W12Codec.BLOCK_ELEMENTS == 0 && info.sizeInBytes() <= MAX_TENSOR_BYTES) {
                    all.add(info);
                }
            }
            all.sort((a, b) -> Long.compare(b.numElements(), a.numElements()));
            List<GGUFTensorInfo> picked = new ArrayList<>();
            List<String> shapes = new ArrayList<>();
            for (GGUFTensorInfo info : all) {
                String shape = info.dim(0) + "x" + info.dim(1);
                if (!shapes.contains(shape)) {
                    picked.add(info);
                    shapes.add(shape);
                }
                if (picked.size() == MAX_TENSORS) {
                    break;
                }
            }
            return picked;
        }
    }

    private void measure(GGUFFileParser parser, GGUFTensorInfo info) throws Exception {
        int cols = (int) info.dim(0);
        int rows = (int) info.dim(1);
        byte[] raw = parser.readTensorData(info);
        // W12 streams bfloat16-representable weights, so the stored values are brought onto the bf16 grid
        // first (round-to-nearest-even, as the loader's upload does)
        float[] values = new float[rows * cols];
        if (info.type() == GGMLType.F16) {
            for (int i = 0; i < values.length; i++) {
                int h = (raw[2 * i] & 0xFF) | ((raw[2 * i + 1] & 0xFF) << 8);
                values[i] = W12Codec.bf16Grid(GGUFFileParser.halfToFloat(h));
            }
        } else {
            for (int i = 0; i < values.length; i++) {
                int b = (raw[2 * i] & 0xFF) | ((raw[2 * i + 1] & 0xFF) << 8);
                values[i] = Float.intBitsToFloat(b << 16);
            }
        }
        int escapes = 0;
        int qEscapes = 0;
        for (int row = 0; row < rows; row++) {
            W12Codec.RowPlan plan = W12Codec.scanRow(info.name(), values, row * cols, cols, row);
            escapes += plan.nEscapes();
            qEscapes += plan.nQEscapes();
        }
        byte[] record = W12Codec.encode(info.name(), values, rows, cols);
        double bitsPerWeight = record.length * 8.0 / ((double) rows * cols);
        // the mirror the loader would decode to: the values rounded to the bfloat16 grid
        ByteBuffer mirrorBits = ByteBuffer.allocate(values.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) {
            mirrorBits.putShort((short) (W12Codec.bf16GridBits(value) >>> 16));
        }
        float[] mirror = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            mirror[i] = Float.intBitsToFloat((mirrorBits.getShort(2 * i) & 0xFFFF) << 16);
        }
        GpuBuffer in = backend.allocateBuffer((long) cols * 4);
        backend.computeQueue().enqueueWriteBuffer(in, 0, (long) cols * 4, input(cols)).awaitCompletion();
        GpuBuffer out = backend.allocateBuffer((long) rows * 4);
        GpuWeight w12 = upload(GGMLType.W12, record, cols, rows);
        GpuWeight bf16 = upload(GGMLType.BF16, mirrorBits.array(), cols, rows);
        try {
            long w12Nanos = time(() -> ops.matvec(w12, in, 0, out, 0, 1));
            long bf16Nanos = time(() -> ops.matvec(bf16, in, 0, out, 0, 1));
            long qNanos;
            try (GpuOps.W12QView scope = ops.openW12QView()) {
                qNanos = time(() -> ops.matvec(w12, in, 0, out, 0, 1));
            }
            double escapeShare = (escapes + (double) qEscapes) / ((double) rows * cols);
            System.out.println(String.format(Locale.ROOT,
                    "real %-5s %-28s %6dx%-6d escapes %5.2f%%  %6.3f bits/weight  "
                            + "w12 %8.3f us (%9.2f tok/s)  q-view %8.3f us  bf16 %8.3f us (%9.2f tok/s)"
                            + "  ratio %5.2fx",
                    api, info.name(), rows, cols, escapeShare * 100.0, bitsPerWeight,
                    w12Nanos / 1e3, 1e9 / w12Nanos, qNanos / 1e3, bf16Nanos / 1e3, 1e9 / bf16Nanos,
                    (double) w12Nanos / (double) bf16Nanos));
        } finally {
            w12.shard(0).close();
            bf16.shard(0).close();
            in.close();
            out.close();
        }
    }

    private GpuWeight upload(GGMLType type, byte[] data, int cols, int rows) throws Exception {
        GpuBuffer buf = backend.allocateBuffer(data.length);
        backend.computeQueue().enqueueWriteBuffer(buf, 0, data.length, data).awaitCompletion();
        return GpuWeight.of("real.weight", type, new long[]{cols, rows}, buf);
    }

    private long time(Runnable launches) {
        for (int i = 0; i < WARMUP; i++) {
            launches.run();
        }
        backend.computeQueue().finish();
        long start = System.nanoTime();
        for (int i = 0; i < RUNS; i++) {
            launches.run();
        }
        backend.computeQueue().finish();
        return (System.nanoTime() - start) / RUNS;
    }

    private static float[] input(int n) {
        float[] x = new float[n];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) Math.sin(i * 0.01) * 0.5f;
        }
        return x;
    }
}
