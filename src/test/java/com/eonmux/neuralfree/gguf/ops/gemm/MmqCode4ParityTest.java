package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.FaSmem;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelManifest;
import com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.SplittableRandom;

import static org.junit.Assert.assertEquals;

/**
 * The word-wise code decoders of the MMQ weight types against their per-element decoders: for every MMQ type
 * T, {@code T_MMQ_CODE4(bp, e0)} must equal the four codes {@code T_MMQ_CODE(bp, e0 .. e0 + 3)} packed as int8
 * lanes ({@code NF_MMQ_CODE4_PACK}), bit for bit, at every group start of every block. A probe kernel compiled
 * on the OpenCL device from the LLM library's headers (common/matmul_mmq_types.h and its requirements) evaluates
 * both over {@value #BLOCKS} blocks of random bytes per type. Random index and sign bytes are valid for every
 * type (each grid has 2^bits entries for its index bits), so the draw reaches every grid entry, sign pattern,
 * high-bit and scale-select combination of the grouped i-quant decoders many times over (IQ1_S's 2048-entry
 * grid about 64 times).
 */
@RunWith(Parameterized.class)
public class MmqCode4ParityTest {

    private static final int BLOCKS = 4096;
    private static final int LOCAL = 64;
    private static final String PROBE = "#define NF_CODE4_PROBE(T) \\\n"
            + "NF_KERNEL void T##_Code4Probe(const int nblocks, NF_GLOBAL const uchar* blocks, NF_GLOBAL uint* got, \\\n"
            + "                              NF_GLOBAL uint* want) { \\\n"
            + "    const int i = (int)get_global_id(0); \\\n"
            + "    if (i >= nblocks * (T##_BLOCK_SIZE / 4)) { \\\n"
            + "        return; \\\n"
            + "    } \\\n"
            + "    const int e0 = (i % (T##_BLOCK_SIZE / 4)) * 4; \\\n"
            + "    NF_GLOBAL const uchar* bp = blocks + (long)(i / (T##_BLOCK_SIZE / 4)) * T##_BLOCK_BYTES; \\\n"
            + "    got[i] = (uint)(T##_MMQ_CODE4(bp, e0)); \\\n"
            + "    want[i] = NF_MMQ_CODE4_PACK(T##_MMQ_CODE(bp, e0), T##_MMQ_CODE(bp, e0 + 1), \\\n"
            + "                                T##_MMQ_CODE(bp, e0 + 2), T##_MMQ_CODE(bp, e0 + 3)); \\\n"
            + "}\n";

    @Parameterized.Parameters(name = "{0}")
    public static Collection<GGMLType> types() {
        Collection<GGMLType> out = new ArrayList<>();
        for (GGMLType t : GemmKernels.MMQ_TYPES) {
            if (GemmKernels.isInt8Mmq(t)) {
                out.add(t);
            }
        }
        return out;
    }

    private final GGMLType type;

    private HWAcceleration hwAccel;
    private OclComputeBackend backend;
    private ComputeQueue queue;

    public MmqCode4ParityTest(GGMLType type) {
        this.type = type;
    }

    @Before
    public void setUp() {
        HWAcceleration hw = null;
        try {
            hw = new HWAcceleration();
            Assume.assumeTrue("no OpenCL device", !hw.getCLDevices().isEmpty());
            CLDevice device = hw.getCLDevices().get(0);
            backend = new OclComputeBackend(device);
            hwAccel = hw;
            queue = backend.computeQueue();
        } catch (Throwable t) {
            if (hw != null) {
                hw.shutdown();
            }
            Assume.assumeTrue("OpenCL init failed: " + t.getMessage(), false);
        }
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
        }
    }

    @Test
    public void wordDecoderEqualsThePackedElementDecoder() throws InterruptedException {
        String kernelName = type.name() + "_Code4Probe";
        String source = headers() + PROBE + "NF_CODE4_PROBE(" + type.name() + ")\n";
        String options = OpenClKernelLibrary.buildOptions(KernelLibraries.LLM_CONFIG.openClOptions(), false,
                FaSmem.faSmemFloats(ComputeBackend.BackendType.OPENCL, backend.capabilities()),
                MmqConfig.PACKED_DOT.kernelDefines(false));
        KernelHandle probe = backend.compileKernel(kernelName, source, options);
        int blockBytes = type.getTypeSize();
        int items = BLOCKS * (type.getBlockSize() / 4);
        byte[] blocks = new byte[BLOCKS * blockBytes];
        new SplittableRandom(0x5EEDL + type.ordinal()).nextBytes(blocks);
        GpuBuffer bbuf = backend.allocateBuffer(blocks.length, true);
        GpuBuffer gbuf = backend.allocateBuffer((long) items * Integer.BYTES);
        GpuBuffer wbuf = backend.allocateBuffer((long) items * Integer.BYTES);
        try {
            byte[] got = new byte[items * Integer.BYTES];
            byte[] want = new byte[items * Integer.BYTES];
            try (GpuEvent written = queue.enqueueWriteBuffer(bbuf, 0, blocks.length, blocks)) {
                queue.enqueueKernelNoEvent(probe, new long[]{(items + LOCAL - 1L) / LOCAL * LOCAL},
                        new long[]{LOCAL}, new Object[]{BLOCKS, bbuf, gbuf, wbuf}, written);
                try (GpuEvent g = queue.enqueueReadBuffer(gbuf, 0, got.length, got);
                     GpuEvent w = queue.enqueueReadBuffer(wbuf, 0, want.length, want, g)) {
                    w.awaitCompletion();
                }
            }
            ByteBuffer gb = ByteBuffer.wrap(got).order(ByteOrder.LITTLE_ENDIAN);
            ByteBuffer wb = ByteBuffer.wrap(want).order(ByteOrder.LITTLE_ENDIAN);
            int perBlock = type.getBlockSize() / 4;
            for (int i = 0; i < items; i++) {
                int g = gb.getInt(i * Integer.BYTES);
                int w = wb.getInt(i * Integer.BYTES);
                if (g != w) {
                    assertEquals(type + " block " + i / perBlock + " codes " + 4 * (i % perBlock) + "..+3",
                            Integer.toHexString(w), Integer.toHexString(g));
                }
            }
        } finally {
            bbuf.close();
            gbuf.close();
            wbuf.close();
        }
    }

    /** The LLM library's headers in manifest order, as the OpenCL program sees them before its kernel files. */
    private static String headers() {
        ClassLoader cl = MmqCode4ParityTest.class.getClassLoader();
        String root = KernelLibraries.LLM_CONFIG.sourceRoot();
        KernelManifest manifest = KernelManifest.load(cl, root);
        StringBuilder sb = new StringBuilder();
        for (KernelManifest.Entry e : manifest.entries()) {
            if (e.cudaOnly() || !e.path().endsWith(".h")) {
                continue;
            }
            sb.append("#line 1 \"").append(e.path()).append("\"\n").append(read(cl, e.resourcePath(root)))
                    .append('\n');
        }
        return sb.toString();
    }

    private static String read(ClassLoader cl, String path) {
        try (InputStream in = cl.getResourceAsStream(path)) {
            if (in == null) {
                throw new IllegalStateException("kernel source " + path + " is not on the classpath");
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
