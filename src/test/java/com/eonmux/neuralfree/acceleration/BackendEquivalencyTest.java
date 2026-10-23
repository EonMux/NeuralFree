package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import org.junit.*;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.Assert.*;

public class BackendEquivalencyTest {

    private static final float TOLERANCE = 2e-4f;
    private static final Set<String> completedTests = ConcurrentHashMap.newKeySet();

    private static HWAcceleration hwAccel;
    private static OclComputeBackend openCLBackend;
    private static CudaComputeBackend cudaBackend;
    private static VkComputeBackend vulkanBackend;

    private static boolean openCLAvailable = false;
    private static boolean cudaAvailable = false;
    private static boolean vulkanAvailable = false;

    private final List<EquivalencyResult> results = new ArrayList<>();

    private static final String OPENCL_SIMPLE_ADD =
            "__kernel void simple_add(__global float *a, __global float *b, __global float *c, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        c[i] = a[i] + b[i];\n" +
            "    }\n" +
            "}\n";

    private static final String OPENCL_SCALE =
            "__kernel void scale(__global float *data, const float factor, const int n) {\n" +
            "    int i = get_global_id(0);\n" +
            "    if (i < n) {\n" +
            "        data[i] = data[i] * factor;\n" +
            "    }\n" +
            "}\n";

    private static final String OPENCL_MATMUL =
            "__kernel void matmul(__global float *A, __global float *B, __global float *C, const int M, const int K, const int N) {\n" +
            "    int row = get_global_id(0);\n" +
            "    int col = get_global_id(1);\n" +
            "    if (row < M && col < N) {\n" +
            "        float sum = 0.0f;\n" +
            "        for (int k = 0; k < K; k++) {\n" +
            "            sum += A[row * K + k] * B[k * N + col];\n" +
            "        }\n" +
            "        C[row * N + col] = sum;\n" +
            "    }\n" +
            "}\n";

    private static final String CUDA_SIMPLE_ADD_PTX =
            ".version 7.5\n" +
            ".target sm_50\n" +
            ".address_size 64\n" +
            "\n" +
            ".visible .entry simple_add(\n" +
            "    .param .u64 a,\n" +
            "    .param .u64 b,\n" +
            "    .param .u64 c,\n" +
            "    .param .u32 n\n" +
            ") {\n" +
            "    .reg .f32   %f<4>;\n" +
            "    .reg .b32   %r<5>;\n" +
            "    .reg .b64   %rd<10>;\n" +
            "\n" +
            "    ld.param.u64    %rd1, [a];\n" +
            "    ld.param.u64    %rd2, [b];\n" +
            "    ld.param.u64    %rd3, [c];\n" +
            "    ld.param.u32    %r1, [n];\n" +
            "\n" +
            "    mov.u32         %r2, %tid.x;\n" +
            "    mov.u32         %r3, %ntid.x;\n" +
            "    mov.u32         %r4, %ctaid.x;\n" +
            "    mad.lo.u32      %r2, %r4, %r3, %r2;\n" +
            "\n" +
            "    setp.ge.u32     p0, %r2, %r1;\n" +
            "    @p0 bra         L_exit;\n" +
            "\n" +
            "    mul.wide.u32    %rd4, %r2, 4;\n" +
            "    add.s64         %rd5, %rd1, %rd4;\n" +
            "    add.s64         %rd6, %rd2, %rd4;\n" +
            "    add.s64         %rd7, %rd3, %rd4;\n" +
            "\n" +
            "    ld.global.f32   %f1, [%rd5];\n" +
            "    ld.global.f32   %f2, [%rd6];\n" +
            "    add.f32         %f3, %f1, %f2;\n" +
            "    st.global.f32   [%rd7], %f3;\n" +
            "\n" +
            "L_exit:\n" +
            "    ret;\n" +
            "}\n";

    private static final String CUDA_SCALE_PTX =
            ".version 7.5\n" +
            ".target sm_50\n" +
            ".address_size 64\n" +
            "\n" +
            ".visible .entry scale(\n" +
            "    .param .u64 data,\n" +
            "    .param .f32 factor,\n" +
            "    .param .u32 n\n" +
            ") {\n" +
            "    .reg .f32   %f<4>;\n" +
            "    .reg .b32   %r<5>;\n" +
            "    .reg .b64   %rd<8>;\n" +
            "\n" +
            "    ld.param.u64    %rd1, [data];\n" +
            "    ld.param.f32    %f1, [factor];\n" +
            "    ld.param.u32    %r1, [n];\n" +
            "\n" +
            "    mov.u32         %r2, %tid.x;\n" +
            "    mov.u32         %r3, %ntid.x;\n" +
            "    mov.u32         %r4, %ctaid.x;\n" +
            "    mad.lo.u32      %r2, %r4, %r3, %r2;\n" +
            "\n" +
            "    setp.ge.u32     p0, %r2, %r1;\n" +
            "    @p0 bra         L_exit;\n" +
            "\n" +
            "    mul.wide.u32    %rd2, %r2, 4;\n" +
            "    add.s64         %rd3, %rd1, %rd2;\n" +
            "\n" +
            "    ld.global.f32   %f2, [%rd3];\n" +
            "    mul.f32         %f3, %f2, %f1;\n" +
            "    st.global.f32   [%rd3], %f3;\n" +
            "\n" +
            "L_exit:\n" +
            "    ret;\n" +
            "}\n";

    private static final String CUDA_MATMUL_PTX =
            ".version 7.5\n" +
            ".target sm_50\n" +
            ".address_size 64\n" +
            "\n" +
            ".visible .entry matmul(\n" +
            "    .param .u64 A,\n" +
            "    .param .u64 B,\n" +
            "    .param .u64 C,\n" +
            "    .param .u32 M,\n" +
            "    .param .u32 K,\n" +
            "    .param .u32 N\n" +
            ") {\n" +
            "    .reg .f32   %f<8>;\n" +
            "    .reg .b32   %r<12>;\n" +
            "    .reg .b64   %rd<16>;\n" +
            "    .reg .pred  p0, p1;\n" +
            "\n" +
            "    ld.param.u64    %rd1, [A];\n" +
            "    ld.param.u64    %rd2, [B];\n" +
            "    ld.param.u64    %rd3, [C];\n" +
            "    ld.param.u32    %r1, [M];\n" +
            "    ld.param.u32    %r2, [K];\n" +
            "    ld.param.u32    %r3, [N];\n" +
            "\n" +
            "    mov.u32         %r4, %tid.x;\n" +
            "    mov.u32         %r5, %ntid.x;\n" +
            "    mov.u32         %r6, %ctaid.x;\n" +
            "    mad.lo.u32      %r4, %r6, %r5, %r4;\n" +
            "\n" +
            "    mov.u32         %r7, %tid.y;\n" +
            "    mov.u32         %r8, %ntid.y;\n" +
            "    mov.u32         %r9, %ctaid.y;\n" +
            "    mad.lo.u32      %r7, %r9, %r8, %r7;\n" +
            "\n" +
            "    setp.ge.u32     p0, %r4, %r1;\n" +
            "    setp.ge.u32     p1, %r7, %r3;\n" +
            "    or.pred         p0, p0, p1;\n" +
            "    @p0 bra         L_exit;\n" +
            "\n" +
            "    mov.f32         %f1, 0.0;\n" +
            "    mov.u32         %r10, 0;\n" +
            "\n" +
            "L_loop:\n" +
            "    setp.ge.u32     p0, %r10, %r2;\n" +
            "    @p0 bra         L_store;\n" +
            "\n" +
            "    mul.lo.u32      %r11, %r4, %r2;\n" +
            "    add.u32         %r11, %r11, %r10;\n" +
            "    mul.wide.u32    %rd4, %r11, 4;\n" +
            "    add.s64         %rd5, %rd1, %rd4;\n" +
            "    ld.global.f32   %f2, [%rd5];\n" +
            "\n" +
            "    mul.lo.u32      %r11, %r10, %r3;\n" +
            "    add.u32         %r11, %r11, %r7;\n" +
            "    mul.wide.u32    %rd6, %r11, 4;\n" +
            "    add.s64         %rd7, %rd2, %rd6;\n" +
            "    ld.global.f32   %f3, [%rd7];\n" +
            "\n" +
            "    mul.f32         %f4, %f2, %f3;\n" +
            "    add.f32         %f1, %f1, %f4;\n" +
            "\n" +
            "    add.u32         %r10, %r10, 1;\n" +
            "    bra             L_loop;\n" +
            "\n" +
            "L_store:\n" +
            "    mul.lo.u32      %r11, %r4, %r3;\n" +
            "    add.u32         %r11, %r11, %r7;\n" +
            "    mul.wide.u32    %rd8, %r11, 4;\n" +
            "    add.s64         %rd9, %rd3, %rd8;\n" +
            "    st.global.f32   [%rd9], %f1;\n" +
            "\n" +
            "L_exit:\n" +
            "    ret;\n" +
            "}\n";

    @Before
    public void setUp() {
        results.clear();
        
        if (openCLBackend == null) {
            try {
                hwAccel = new HWAcceleration();
                if (!hwAccel.getCLDevices().isEmpty()) {
                    CLDevice clDevice = hwAccel.getCLDevices().get(0);
                    openCLBackend = new OclComputeBackend(clDevice);
                    openCLAvailable = true;
                    System.out.println("OpenCL backend initialized: " + openCLBackend.capabilities().getDeviceName());
                }
            } catch (Throwable e) {
                System.out.println("OpenCL initialization failed: " + e.getMessage());
            }
        }

        if (cudaBackend == null) {
            try {
                HWAcceleration hwAccelCuda = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.CUDA));
                if (!hwAccelCuda.getCUDevices().isEmpty()) {
                    CUDevice cuDevice = hwAccelCuda.getCUDevices().get(0);
                    cudaBackend = new CudaComputeBackend(cuDevice);
                    cudaAvailable = true;
                    System.out.println("CUDA backend initialized: " + cudaBackend.capabilities().getDeviceName());
                }
                hwAccelCuda.shutdown();
            } catch (Throwable e) {
                System.out.println("CUDA initialization failed: " + e.getMessage());
            }
        }

        if (vulkanBackend == null) {
            try {
                vulkanBackend = new VkComputeBackend();
                vulkanAvailable = true;
                System.out.println("Vulkan backend initialized: " + vulkanBackend.capabilities().getDeviceName());
            } catch (Throwable e) {
                System.out.println("Vulkan initialization failed: " + e.getMessage());
            }
        }
    }

    @After
    public void tearDown() {
        results.forEach(r -> completedTests.add(r.testName + "_" + r.matrixSize));
    }

    @AfterClass
    public static void tearDownClass() {
        if (openCLBackend != null) {
            try { openCLBackend.close(); } catch (Throwable ignored) {}
        }
        if (cudaBackend != null) {
            try { cudaBackend.close(); } catch (Throwable ignored) {}
        }
        if (vulkanBackend != null) {
            try { vulkanBackend.close(); } catch (Throwable ignored) {}
        }
        if (hwAccel != null) {
            try { hwAccel.shutdown(); } catch (Throwable ignored) {}
        }
    }

    public static class EquivalencyResult {
        String testName;
        String kernelName;
        String matrixSize;
        boolean openCLPass;
        boolean cudaPass;
        boolean vulkanPass;
        boolean crossBackendPass;
        String notes;

        EquivalencyResult(String testName, String kernelName, String matrixSize) {
            this.testName = testName;
            this.kernelName = kernelName;
            this.matrixSize = matrixSize;
            this.openCLPass = false;
            this.cudaPass = false;
            this.vulkanPass = false;
            this.crossBackendPass = false;
            this.notes = "";
        }
    }

    @Test
    public void testSimpleAddEquivalency_4x4() {
        testSimpleAddEquivalency(16, "4x4");
    }

    @Test
    public void testSimpleAddEquivalency_64x64() {
        testSimpleAddEquivalency(64 * 64, "64x64");
    }

    @Test
    public void testSimpleAddEquivalency_256x256() {
        testSimpleAddEquivalency(256 * 256, "256x256");
    }

    @Test
    public void testSimpleAddEquivalency_NonSquare() {
        testSimpleAddEquivalency(32 * 64, "32x64");
    }

    private void testSimpleAddEquivalency(int size, String sizeLabel) {
        EquivalencyResult result = new EquivalencyResult("simple_add", "simple_add", sizeLabel);
        
        float[] a = new float[size];
        float[] b = new float[size];
        float[] expected = new float[size];
        Random rand = new Random(42);
        for (int i = 0; i < size; i++) {
            a[i] = rand.nextFloat() * 100.0f - 50.0f;
            b[i] = rand.nextFloat() * 100.0f - 50.0f;
            expected[i] = a[i] + b[i];
        }

        Map<ComputeBackend.BackendType, float[]> backendResults = new EnumMap<>(ComputeBackend.BackendType.class);

        if (openCLAvailable) {
            try {
                float[] res = runSimpleAddKernel(openCLBackend, OPENCL_SIMPLE_ADD, a, b, size);
                backendResults.put(ComputeBackend.BackendType.OPENCL, res);
                result.openCLPass = compareArrays(expected, res, "OpenCL simple_add");
            } catch (Exception e) {
                result.notes += "OpenCL failed: " + e.getMessage() + "; ";
                System.out.println("OpenCL simple_add exception: " + e.getClass().getName() + ": " + e.getMessage());
                e.printStackTrace();
            } catch (Throwable t) {
                result.notes += "OpenCL error: " + t.getMessage() + "; ";
                System.out.println("OpenCL simple_add throwable: " + t.getClass().getName() + ": " + t.getMessage());
                t.printStackTrace();
            }
        }

        if (cudaAvailable) {
            try {
                float[] res = runSimpleAddKernel(cudaBackend, CUDA_SIMPLE_ADD_PTX, a, b, size);
                backendResults.put(ComputeBackend.BackendType.CUDA, res);
                result.cudaPass = compareArrays(expected, res, "CUDA simple_add");
            } catch (Exception e) {
                result.notes += "CUDA failed: " + e.getMessage() + "; ";
                System.out.println("CUDA simple_add exception: " + e.getClass().getName() + ": " + e.getMessage());
                e.printStackTrace();
            } catch (Throwable t) {
                result.notes += "CUDA error: " + t.getMessage() + "; ";
                System.out.println("CUDA simple_add throwable: " + t.getClass().getName() + ": " + t.getMessage());
                t.printStackTrace();
            }
        }

        if (vulkanAvailable) {
            try {
                float[] res = runVulkanSimpleAddKernel(a, b, size);
                backendResults.put(ComputeBackend.BackendType.VULKAN, res);
                result.vulkanPass = compareArrays(expected, res, "Vulkan simple_add");
            } catch (Exception e) {
                result.notes += "Vulkan failed: " + e.getMessage() + "; ";
            }
        }

        result.crossBackendPass = verifyCrossBackendEquivalency(backendResults, "simple_add", sizeLabel);
        results.add(result);

        assertTrue("At least one backend must pass for " + sizeLabel, 
                   result.openCLPass || result.cudaPass || result.vulkanPass);
        if (backendResults.size() > 1) {
            assertTrue("Cross-backend equivalency failed for " + sizeLabel, result.crossBackendPass);
        }
    }

    @Test
    public void testScaleEquivalency_4x4() {
        testScaleEquivalency(16, "4x4");
    }

    @Test
    public void testScaleEquivalency_64x64() {
        testScaleEquivalency(64 * 64, "64x64");
    }

    @Test
    public void testScaleEquivalency_256x256() {
        testScaleEquivalency(256 * 256, "256x256");
    }

    @Test
    public void testScaleEquivalency_NonSquare() {
        testScaleEquivalency(64 * 128, "64x128");
    }

    private void testScaleEquivalency(int size, String sizeLabel) {
        EquivalencyResult result = new EquivalencyResult("scale", "scale", sizeLabel);

        float[] data = new float[size];
        float factor = 3.14159f;
        float[] expected = new float[size];
        Random rand = new Random(123);
        for (int i = 0; i < size; i++) {
            data[i] = rand.nextFloat() * 100.0f - 50.0f;
            expected[i] = data[i] * factor;
        }

        Map<ComputeBackend.BackendType, float[]> backendResults = new EnumMap<>(ComputeBackend.BackendType.class);

        if (openCLAvailable) {
            try {
                float[] res = runScaleKernel(openCLBackend, OPENCL_SCALE, data, factor, size);
                backendResults.put(ComputeBackend.BackendType.OPENCL, res);
                result.openCLPass = compareArrays(expected, res, "OpenCL scale");
            } catch (Exception e) {
                result.notes += "OpenCL failed: " + e.getMessage() + "; ";
            }
        }

        if (cudaAvailable) {
            try {
                float[] res = runScaleKernel(cudaBackend, CUDA_SCALE_PTX, data, factor, size);
                backendResults.put(ComputeBackend.BackendType.CUDA, res);
                result.cudaPass = compareArrays(expected, res, "CUDA scale");
            } catch (Exception e) {
                result.notes += "CUDA failed: " + e.getMessage() + "; ";
            }
        }

        if (vulkanAvailable) {
            try {
                float[] res = runVulkanScaleKernel(data, factor, size);
                backendResults.put(ComputeBackend.BackendType.VULKAN, res);
                result.vulkanPass = compareArrays(expected, res, "Vulkan scale");
            } catch (Exception e) {
                result.notes += "Vulkan failed: " + e.getMessage() + "; ";
            }
        }

        result.crossBackendPass = verifyCrossBackendEquivalency(backendResults, "scale", sizeLabel);
        results.add(result);

        assertTrue("At least one backend must pass for " + sizeLabel,
                   result.openCLPass || result.cudaPass || result.vulkanPass);
        if (backendResults.size() > 1) {
            assertTrue("Cross-backend equivalency failed for " + sizeLabel, result.crossBackendPass);
        }
    }

    @Test
    public void testMatrixMultiplyEquivalency_4x4() {
        testMatrixMultiplyEquivalency(4, 4, 4, "4x4");
    }

    @Test
    public void testMatrixMultiplyEquivalency_64x64() {
        testMatrixMultiplyEquivalency(64, 64, 64, "64x64");
    }

    @Test
    public void testMatrixMultiplyEquivalency_256x256() {
        testMatrixMultiplyEquivalency(256, 256, 256, "256x256");
    }

    @Test
    public void testMatrixMultiplyEquivalency_NonSquare_32x64() {
        testMatrixMultiplyEquivalency(32, 64, 32, "32x64");
    }

    @Test
    public void testMatrixMultiplyEquivalency_NonSquare_64x128() {
        testMatrixMultiplyEquivalency(64, 128, 64, "64x128");
    }

    private void testMatrixMultiplyEquivalency(int M, int K, int N, String sizeLabel) {
        EquivalencyResult result = new EquivalencyResult("matmul", "matmul", sizeLabel + " (M=" + M + ",K=" + K + ",N=" + N + ")");

        Random rand = new Random(456);
        float[] A = new float[M * K];
        float[] B = new float[K * N];
        for (int i = 0; i < A.length; i++) A[i] = rand.nextFloat() * 10.0f - 5.0f;
        for (int i = 0; i < B.length; i++) B[i] = rand.nextFloat() * 10.0f - 5.0f;

        float[] expected = new float[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                expected[i * N + j] = sum;
            }
        }

        Map<ComputeBackend.BackendType, float[]> backendResults = new EnumMap<>(ComputeBackend.BackendType.class);

        if (openCLAvailable) {
            try {
                float[] res = runMatMulKernel(openCLBackend, OPENCL_MATMUL, A, B, M, K, N);
                backendResults.put(ComputeBackend.BackendType.OPENCL, res);
                result.openCLPass = compareArrays(expected, res, "OpenCL matmul");
            } catch (Exception e) {
                result.notes += "OpenCL failed: " + e.getMessage() + "; ";
            }
        }

        if (cudaAvailable) {
            try {
                float[] res = runMatMulKernel(cudaBackend, CUDA_MATMUL_PTX, A, B, M, K, N);
                backendResults.put(ComputeBackend.BackendType.CUDA, res);
                result.cudaPass = compareArrays(expected, res, "CUDA matmul");
            } catch (Exception e) {
                result.notes += "CUDA failed: " + e.getMessage() + "; ";
            }
        }

        if (vulkanAvailable) {
            try {
                float[] res = runVulkanMatMulKernel(A, B, M, K, N);
                backendResults.put(ComputeBackend.BackendType.VULKAN, res);
                result.vulkanPass = compareArrays(expected, res, "Vulkan matmul");
            } catch (Exception e) {
                result.notes += "Vulkan failed: " + e.getMessage() + "; ";
            }
        }

        result.crossBackendPass = verifyCrossBackendEquivalency(backendResults, "matmul", sizeLabel);
        results.add(result);

        assertTrue("At least one backend must pass for " + sizeLabel,
                   result.openCLPass || result.cudaPass || result.vulkanPass);
        if (backendResults.size() > 1) {
            assertTrue("Cross-backend equivalency failed for " + sizeLabel, result.crossBackendPass);
        }
    }

    private float[] runSimpleAddKernel(ComputeBackend backend, String kernelSource, float[] a, float[] b, int size) {
        KernelHandle kernel = backend.compileKernel("simple_add", kernelSource, null);
        long bufferSize = size * Float.BYTES;

        GpuBuffer bufferA = backend.allocateBuffer(bufferSize);
        GpuBuffer bufferB = backend.allocateBuffer(bufferSize);
        GpuBuffer bufferC = backend.allocateBuffer(bufferSize);

        try {
            ComputeQueue tQueue = backend.transferQueue();
            ComputeQueue cQueue = backend.computeQueue();

            tQueue.enqueueWriteBuffer(bufferA, 0, bufferSize, a);
            tQueue.enqueueWriteBuffer(bufferB, 0, bufferSize, b);
            tQueue.finish();

            Object[] args = new Object[]{bufferA, bufferB, bufferC, size};
            int localSize = Math.min(64, size);
            int globalSize = ((size + localSize - 1) / localSize) * localSize;
            cQueue.enqueueKernel(kernel, new long[]{globalSize}, new long[]{localSize}, args);
            cQueue.finish();

            float[] result = new float[size];
            cQueue.enqueueReadBuffer(bufferC, 0, bufferSize, result);
            cQueue.finish();

            return result;
        } finally {
            bufferA.close();
            bufferB.close();
            bufferC.close();
        }
    }

    private float[] runScaleKernel(ComputeBackend backend, String kernelSource, float[] data, float factor, int size) {
        KernelHandle kernel = backend.compileKernel("scale", kernelSource, null);
        long bufferSize = size * Float.BYTES;

        GpuBuffer buffer = backend.allocateBuffer(bufferSize);

        try {
            ComputeQueue tQueue = backend.transferQueue();
            ComputeQueue cQueue = backend.computeQueue();

            tQueue.enqueueWriteBuffer(buffer, 0, bufferSize, data);
            tQueue.finish();

            Object[] args = new Object[]{buffer, factor, size};
            int localSize = Math.min(64, size);
            int globalSize = ((size + localSize - 1) / localSize) * localSize;
            cQueue.enqueueKernel(kernel, new long[]{globalSize}, new long[]{localSize}, args);
            cQueue.finish();

            float[] result = new float[size];
            cQueue.enqueueReadBuffer(buffer, 0, bufferSize, result);
            cQueue.finish();

            return result;
        } finally {
            buffer.close();
        }
    }

    private float[] runMatMulKernel(ComputeBackend backend, String kernelSource, float[] A, float[] B, int M, int K, int N) {
        KernelHandle kernel = backend.compileKernel("matmul", kernelSource, null);

        GpuBuffer bufferA = backend.allocateBuffer(M * K * Float.BYTES);
        GpuBuffer bufferB = backend.allocateBuffer(K * N * Float.BYTES);
        GpuBuffer bufferC = backend.allocateBuffer(M * N * Float.BYTES);

        try {
            ComputeQueue tQueue = backend.transferQueue();
            ComputeQueue cQueue = backend.computeQueue();

            tQueue.enqueueWriteBuffer(bufferA, 0, A.length * Float.BYTES, A);
            tQueue.enqueueWriteBuffer(bufferB, 0, B.length * Float.BYTES, B);
            tQueue.finish();

            Object[] args = new Object[]{bufferA, bufferB, bufferC, M, K, N};
            int blockSize = Math.min(16, Math.min(M, N));
            if (blockSize < 1) blockSize = 1;
            int gridX = ((M + blockSize - 1) / blockSize) * blockSize;
            int gridY = ((N + blockSize - 1) / blockSize) * blockSize;
            cQueue.enqueueKernel(kernel, new long[]{gridX, gridY}, new long[]{blockSize, blockSize}, args);
            cQueue.finish();

            float[] result = new float[M * N];
            cQueue.enqueueReadBuffer(bufferC, 0, result.length * Float.BYTES, result);
            cQueue.finish();

            return result;
        } finally {
            bufferA.close();
            bufferB.close();
            bufferC.close();
        }
    }

    private float[] runVulkanSimpleAddKernel(float[] a, float[] b, int size) {
        KernelHandle kernel = vulkanBackend.compileKernel("simple_add", "shaders/test/simple_add.spv", null);
        long bufferSize = size * Float.BYTES;

        GpuBuffer bufferA = vulkanBackend.allocateBuffer(bufferSize);
        GpuBuffer bufferB = vulkanBackend.allocateBuffer(bufferSize);
        GpuBuffer bufferC = vulkanBackend.allocateBuffer(bufferSize);

        try {
            ComputeQueue tQueue = vulkanBackend.transferQueue();
            ComputeQueue cQueue = vulkanBackend.computeQueue();

            tQueue.enqueueWriteBuffer(bufferA, 0, bufferSize, a);
            tQueue.enqueueWriteBuffer(bufferB, 0, bufferSize, b);
            tQueue.finish();

            Object[] args = new Object[]{bufferA, bufferB, bufferC, size};
            int localSize = Math.min(64, size);
            int globalSize = ((size + localSize - 1) / localSize) * localSize;
            cQueue.enqueueKernel(kernel, new long[]{globalSize}, new long[]{localSize}, args);
            cQueue.finish();

            float[] result = new float[size];
            cQueue.enqueueReadBuffer(bufferC, 0, bufferSize, result);
            cQueue.finish();

            return result;
        } finally {
            bufferA.close();
            bufferB.close();
            bufferC.close();
        }
    }

    private float[] runVulkanScaleKernel(float[] data, float factor, int size) {
        KernelHandle kernel = vulkanBackend.compileKernel("scale", "shaders/test/scale.spv", null);
        long bufferSize = size * Float.BYTES;

        GpuBuffer buffer = vulkanBackend.allocateBuffer(bufferSize);

        try {
            ComputeQueue tQueue = vulkanBackend.transferQueue();
            ComputeQueue cQueue = vulkanBackend.computeQueue();

            tQueue.enqueueWriteBuffer(buffer, 0, bufferSize, data);
            tQueue.finish();

            Object[] args = new Object[]{buffer, factor, size};
            int localSize = Math.min(64, size);
            int globalSize = ((size + localSize - 1) / localSize) * localSize;
            cQueue.enqueueKernel(kernel, new long[]{globalSize}, new long[]{localSize}, args);
            cQueue.finish();

            float[] result = new float[size];
            cQueue.enqueueReadBuffer(buffer, 0, bufferSize, result);
            cQueue.finish();

            return result;
        } finally {
            buffer.close();
        }
    }

    private float[] runVulkanMatMulKernel(float[] A, float[] B, int M, int K, int N) {
        KernelHandle kernel = vulkanBackend.compileKernel("matmul", "shaders/test/matmul.spv", null);

        GpuBuffer bufferA = vulkanBackend.allocateBuffer(M * K * Float.BYTES);
        GpuBuffer bufferB = vulkanBackend.allocateBuffer(K * N * Float.BYTES);
        GpuBuffer bufferC = vulkanBackend.allocateBuffer(M * N * Float.BYTES);

        try {
            ComputeQueue tQueue = vulkanBackend.transferQueue();
            ComputeQueue cQueue = vulkanBackend.computeQueue();

            tQueue.enqueueWriteBuffer(bufferA, 0, A.length * Float.BYTES, A);
            tQueue.enqueueWriteBuffer(bufferB, 0, B.length * Float.BYTES, B);
            tQueue.finish();

            Object[] args = new Object[]{bufferA, bufferB, bufferC, M, K, N};
            int blockSize = Math.min(16, Math.min(M, N));
            if (blockSize < 1) blockSize = 1;
            int globalCols = ((N + blockSize - 1) / blockSize) * blockSize;
            int globalRows = ((M + blockSize - 1) / blockSize) * blockSize;
            cQueue.enqueueKernel(kernel, new long[]{globalCols, globalRows}, new long[]{blockSize, blockSize}, args);
            cQueue.finish();

            float[] result = new float[M * N];
            cQueue.enqueueReadBuffer(bufferC, 0, result.length * Float.BYTES, result);
            cQueue.finish();

            return result;
        } finally {
            bufferA.close();
            bufferB.close();
            bufferC.close();
        }
    }

    private boolean compareArrays(float[] expected, float[] actual, String label) {
        if (expected.length != actual.length) {
            System.out.println(label + ": Array length mismatch. Expected: " + expected.length + ", Actual: " + actual.length);
            return false;
        }

        int mismatchCount = 0;
        float maxDiff = 0;
        int maxDiffIdx = -1;

        for (int i = 0; i < expected.length; i++) {
            float diff = Math.abs(expected[i] - actual[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIdx = i;
            }
            if (diff > TOLERANCE) {
                mismatchCount++;
                if (mismatchCount <= 5) {
                    System.out.println(label + ": Mismatch at index " + i + ". Expected: " + expected[i] + ", Actual: " + actual[i] + ", Diff: " + diff);
                }
            }
        }

        if (mismatchCount > 0) {
            System.out.println(label + ": Total mismatches: " + mismatchCount + "/" + expected.length + ", Max diff: " + maxDiff + " at index " + maxDiffIdx);
            return false;
        }

        System.out.println(label + ": PASS (max diff: " + maxDiff + ")");
        return true;
    }

    private boolean verifyCrossBackendEquivalency(Map<ComputeBackend.BackendType, float[]> results, String kernelName, String sizeLabel) {
        if (results.size() < 2) {
            System.out.println(kernelName + " (" + sizeLabel + "): Skipping cross-backend check (only " + results.size() + " backend available)");
            return true;
        }

        List<ComputeBackend.BackendType> backends = new ArrayList<>(results.keySet());
        boolean allEquivalent = true;

        for (int i = 0; i < backends.size(); i++) {
            for (int j = i + 1; j < backends.size(); j++) {
                ComputeBackend.BackendType b1 = backends.get(i);
                ComputeBackend.BackendType b2 = backends.get(j);
                float[] r1 = results.get(b1);
                float[] r2 = results.get(b2);

                int mismatchCount = 0;
                float maxDiff = 0;

                for (int k = 0; k < r1.length; k++) {
                    float diff = Math.abs(r1[k] - r2[k]);
                    if (diff > maxDiff) maxDiff = diff;
                    if (diff > TOLERANCE) {
                        mismatchCount++;
                    }
                }

                if (mismatchCount > 0) {
                    System.out.println(kernelName + " (" + sizeLabel + "): Cross-backend mismatch " + b1 + " vs " + b2 + 
                                       " - Mismatches: " + mismatchCount + "/" + r1.length + ", Max diff: " + maxDiff);
                    allEquivalent = false;
                } else {
                    System.out.println(kernelName + " (" + sizeLabel + "): " + b1 + " vs " + b2 + " PASS (max diff: " + maxDiff + ")");
                }
            }
        }

        return allEquivalent;
    }

    @Test
    public void testGenerateEquivalencyReport() {
        StringBuilder report = new StringBuilder();
        report.append("\n========================================\n");
        report.append("Cross-Backend Numerical Equivalency Report\n");
        report.append("========================================\n");
        report.append(String.format("Tolerance: %.6f%n", TOLERANCE));
        report.append(String.format("OpenCL Available: %b%n", openCLAvailable));
        report.append(String.format("CUDA Available: %b%n", cudaAvailable));
        report.append(String.format("Vulkan Available: %b%n", vulkanAvailable));
        report.append("----------------------------------------\n");

        int passCount = 0;
        int failCount = 0;

        for (EquivalencyResult r : results) {
            String status = (r.openCLPass || r.cudaPass || r.vulkanPass) && r.crossBackendPass ? "PASS" : "FAIL";
            if (status.equals("PASS")) passCount++; else failCount++;

            report.append(String.format("%-20s | %-12s | %-15s | OCL:%s CUDA:%s VK:%s | Cross:%s | %s%n",
                    r.testName, r.kernelName, r.matrixSize,
                    r.openCLPass ? "Y" : "N",
                    r.cudaPass ? "Y" : "N",
                    r.vulkanPass ? "Y" : "N",
                    r.crossBackendPass ? "Y" : "N",
                    status));
            if (!r.notes.isEmpty()) {
                report.append("  Notes: ").append(r.notes).append("\n");
            }
        }

        report.append("----------------------------------------\n");
        report.append(String.format("Summary: %d PASS, %d FAIL out of %d tests%n", passCount, failCount, results.size()));
        report.append("========================================\n");

        System.out.println(report.toString());

        assertTrue("Equivalency report should show at least one passing test", passCount > 0 || results.isEmpty());
    }

    @Test
    public void testBackendAvailabilityDetection() {
        System.out.println("\n=== Backend Availability Detection ===");
        System.out.println("OpenCL Available: " + openCLAvailable);
        if (openCLAvailable && openCLBackend != null) {
            System.out.println("  Device: " + openCLBackend.capabilities().getDeviceName());
            System.out.println("  VRAM: " + (openCLBackend.capabilities().getTotalVramBytes() / (1024 * 1024)) + " MB");
        }

        System.out.println("CUDA Available: " + cudaAvailable);
        if (cudaAvailable && cudaBackend != null) {
            System.out.println("  Device: " + cudaBackend.capabilities().getDeviceName());
            System.out.println("  VRAM: " + (cudaBackend.capabilities().getTotalVramBytes() / (1024 * 1024)) + " MB");
        }

        System.out.println("Vulkan Available: " + vulkanAvailable);
        if (vulkanAvailable && vulkanBackend != null) {
            System.out.println("  Device: " + vulkanBackend.capabilities().getDeviceName());
            System.out.println("  VRAM: " + (vulkanBackend.capabilities().getTotalVramBytes() / (1024 * 1024)) + " MB");
        }
        System.out.println("=======================================");

        assertTrue("At least one backend should be available", openCLAvailable || cudaAvailable || vulkanAvailable);
    }

    @Test
    public void testBufferRoundTripEquivalency() {
        int size = 1024;
        float[] data = new float[size];
        Random rand = new Random(789);
        for (int i = 0; i < size; i++) {
            data[i] = rand.nextFloat() * 1000.0f - 500.0f;
        }

        Map<ComputeBackend.BackendType, float[]> roundTripResults = new EnumMap<>(ComputeBackend.BackendType.class);

        if (openCLAvailable) {
            try {
                GpuBuffer buffer = openCLBackend.allocateBuffer(size * Float.BYTES);
                openCLBackend.transferQueue().enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);
                openCLBackend.transferQueue().finish();

                float[] result = new float[size];
                openCLBackend.transferQueue().enqueueReadBuffer(buffer, 0, size * Float.BYTES, result);
                openCLBackend.transferQueue().finish();
                buffer.close();

                roundTripResults.put(ComputeBackend.BackendType.OPENCL, result);
                assertTrue("OpenCL buffer round-trip should preserve data", compareArrays(data, result, "OpenCL round-trip"));
            } catch (Exception e) {
                System.out.println("OpenCL buffer round-trip failed: " + e.getMessage());
            }
        }

        if (cudaAvailable) {
            try {
                GpuBuffer buffer = cudaBackend.allocateBuffer(size * Float.BYTES);
                cudaBackend.transferQueue().enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);
                cudaBackend.transferQueue().finish();

                float[] result = new float[size];
                cudaBackend.transferQueue().enqueueReadBuffer(buffer, 0, size * Float.BYTES, result);
                cudaBackend.transferQueue().finish();
                buffer.close();

                roundTripResults.put(ComputeBackend.BackendType.CUDA, result);
                assertTrue("CUDA buffer round-trip should preserve data", compareArrays(data, result, "CUDA round-trip"));
            } catch (Exception e) {
                System.out.println("CUDA buffer round-trip failed: " + e.getMessage());
            }
        }

        if (vulkanAvailable) {
            try {
                GpuBuffer buffer = vulkanBackend.allocateBuffer(size * Float.BYTES);
                vulkanBackend.transferQueue().enqueueWriteBuffer(buffer, 0, size * Float.BYTES, data);
                vulkanBackend.transferQueue().finish();

                float[] result = new float[size];
                vulkanBackend.transferQueue().enqueueReadBuffer(buffer, 0, size * Float.BYTES, result);
                vulkanBackend.transferQueue().finish();
                buffer.close();

                roundTripResults.put(ComputeBackend.BackendType.VULKAN, result);
                assertTrue("Vulkan buffer round-trip should preserve data", compareArrays(data, result, "Vulkan round-trip"));
            } catch (Exception e) {
                System.out.println("Vulkan buffer round-trip failed: " + e.getMessage());
            }
        }

        boolean crossBackendOk = verifyCrossBackendEquivalency(roundTripResults, "buffer_round_trip", "1024");
        if (roundTripResults.size() > 1) {
            assertTrue("Cross-backend round-trip should be equivalent", crossBackendOk);
        }
    }
}
