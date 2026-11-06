package com.eonmux.neuralfree.acceleration.hip;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * The HIP driver operations over {@link HipNative}, the production {@link HipDriver}.
 *
 * <p>Device-scoped methods first make {@code device} the calling thread's device
 * ({@code hipSetDevice}) — the HIP runtime has no explicit context, its implicit per-device
 * context follows the thread's device. The check-and-translate of every status is
 * {@link HipErrors}'.</p>
 */
public final class JniHipDriver implements HipDriver {

    private static final int PCI_BUS_ID_CHARS = 64;
    private static final int UUID_BYTES = 16;

    private final int device;

    /**
     * @param device the HIP device ordinal every device-scoped call runs on
     */
    public JniHipDriver(int device) {
        HipNative.ensureLoaded();
        this.device = device;
    }

    private void enter() {
        HipErrors.check("hipSetDevice", () -> HipNative.setDevice(device));
    }

    @Override
    public int deviceCount() {
        int result = HipNative.getDeviceCount();
        if (result < 0) {
            throw HipErrors.translate(-result, "hipGetDeviceCount", -1);
        }
        return result;
    }

    @Override
    public void setDevice(int index) {
        HipErrors.check("hipSetDevice", () -> HipNative.setDevice(index));
    }

    @Override
    public String deviceName(int index) {
        return HipNative.getDeviceName(index);
    }

    @Override
    public String gcnArchName(int index) {
        return HipNative.getGcnArchName(index);
    }

    @Override
    public long totalMem(int index) {
        long bytes = HipNative.deviceTotalMem(index);
        if (bytes < 0) {
            throw HipErrors.translate(HipErrors.INVALID_DEVICE, "hipDeviceTotalMem", -1);
        }
        return bytes;
    }

    @Override
    public int deviceAttribute(int index, int selector) {
        int[] out = new int[1];
        HipErrors.call(HipNative.deviceAttribute(index, selector, out), "hipGetDeviceProperties");
        return out[0];
    }

    @Override
    public int driverVersion() {
        int[] out = new int[1];
        HipErrors.check("hipDriverGetVersion", () -> HipNative.getDriverVersion(out));
        return out[0];
    }

    @Override
    public byte[] deviceUuid(int index) {
        byte[] out = new byte[UUID_BYTES];
        int status = HipNative.getDeviceUuid(index, out);
        if (status != HipErrors.SUCCESS) {
            return null;
        }
        return out;
    }

    @Override
    public String pciBusId(int index) {
        byte[] buf = new byte[PCI_BUS_ID_CHARS];
        int status = HipNative.getPciBusId(index, buf);
        if (status != HipErrors.SUCCESS) {
            return null;
        }
        int end = 0;
        while (end < buf.length && buf[end] != 0) {
            end++;
        }
        String id = new String(buf, 0, end, StandardCharsets.US_ASCII).trim();
        try {
            // Validate through DeviceIdentity's parser so a malformed answer counts as unknown.
            new DeviceIdentity("", DeviceIdentity.UNKNOWN_VENDOR, id, null);
        } catch (RuntimeException e) {
            return null;
        }
        return id.toLowerCase(Locale.ROOT);
    }

    @Override
    public long memAlloc(long bytes) {
        long[] out = new long[1];
        HipErrors.check("hipMalloc", bytes, () -> HipNative.memAlloc(bytes, out));
        return out[0];
    }

    @Override
    public void memFree(long pointer) {
        HipErrors.check("hipFree", () -> HipNative.memFree(pointer));
    }

    @Override
    public long[] memGetInfo() {
        long[] free = new long[1];
        long[] total = new long[1];
        HipErrors.check("hipMemGetInfo", () -> HipNative.memGetInfo(free, total));
        return new long[]{free[0], total[0]};
    }

    @Override
    public long hostAlloc(long bytes) {
        long[] out = new long[1];
        HipErrors.check("hipHostMalloc", bytes, () -> HipNative.hostAlloc(bytes, out));
        return out[0];
    }

    @Override
    public void hostFree(long pointer) {
        HipErrors.check("hipHostFree", () -> HipNative.hostFree(pointer));
    }

    @Override
    public ByteBuffer pinnedView(long pointer, int bytes) {
        Object view = HipNative.pinnedView(pointer, bytes);
        if (!(view instanceof ByteBuffer)) {
            throw new IllegalStateException("the HIP shim returned no ByteBuffer view of the pinned memory");
        }
        return ((ByteBuffer) view).order(ByteOrder.nativeOrder());
    }

    @Override
    public void memcpyHtoDAsync(long dst, long srcHost, long bytes, long stream) {
        enter();
        HipErrors.check("hipMemcpyAsync(HtoD)", () -> HipNative.memcpyHtoDAsync(dst, srcHost, bytes, stream));
    }

    @Override
    public void memcpyDtoHAsync(long dstHost, long src, long bytes, long stream) {
        enter();
        HipErrors.check("hipMemcpyAsync(DtoH)", () -> HipNative.memcpyDtoHAsync(dstHost, src, bytes, stream));
    }

    @Override
    public void memcpyDtoDAsync(long dst, long src, long bytes, long stream) {
        enter();
        HipErrors.check("hipMemcpyAsync(DtoD)", () -> HipNative.memcpyDtoDAsync(dst, src, bytes, stream));
    }

    @Override
    public void memsetD32Async(long dst, int value, long count, long stream) {
        enter();
        HipErrors.check("hipMemsetD32Async", () -> HipNative.memsetD32Async(dst, value, count, stream));
    }

    @Override
    public long streamCreate() {
        long[] out = new long[1];
        HipErrors.check("hipStreamCreateWithFlags", () -> HipNative.streamCreate(out));
        return out[0];
    }

    @Override
    public void streamDestroy(long stream) {
        HipErrors.check("hipStreamDestroy", () -> HipNative.streamDestroy(stream));
    }

    @Override
    public void streamSynchronize(long stream) {
        HipErrors.check("hipStreamSynchronize", () -> HipNative.streamSynchronize(stream));
    }

    @Override
    public void streamWaitEvent(long stream, long event) {
        HipErrors.check("hipStreamWaitEvent", () -> HipNative.streamWaitEvent(stream, event));
    }

    @Override
    public long eventCreate() {
        long[] out = new long[1];
        HipErrors.check("hipEventCreateWithFlags", () -> HipNative.eventCreate(out));
        return out[0];
    }

    @Override
    public void eventDestroy(long event) {
        HipErrors.check("hipEventDestroy", () -> HipNative.eventDestroy(event));
    }

    @Override
    public void eventRecord(long event, long stream) {
        HipErrors.check("hipEventRecord", () -> HipNative.eventRecord(event, stream));
    }

    @Override
    public boolean eventQuery(long event) {
        return HipNative.eventQuery(event) == HipErrors.SUCCESS;
    }

    @Override
    public void eventSynchronize(long event) {
        HipErrors.check("hipEventSynchronize", () -> HipNative.eventSynchronize(event));
    }

    @Override
    public long moduleLoadData(byte[] image) {
        long[] out = new long[1];
        int status = HipNative.moduleLoadData(out, image);
        if (status != HipErrors.SUCCESS) {
            throw HipErrors.translate(status, "hipModuleLoadData", -1);
        }
        return out[0];
    }

    @Override
    public void moduleUnload(long module) {
        HipErrors.check("hipModuleUnload", () -> HipNative.moduleUnload(module));
    }

    @Override
    public long moduleGetFunction(long module, String name) {
        long[] out = new long[1];
        int status = HipNative.moduleGetFunction(out, module, name);
        if (status == HipErrors.INVALID_HANDLE) {
            throw new UnsupportedKernelException("hipModuleGetFunction(" + name + ") failed: the function is not in "
                    + "the module", null);
        }
        HipErrors.call(status, "hipModuleGetFunction(" + name + ")");
        return out[0];
    }

    @Override
    public int funcGetAttribute(long function, int attribute) {
        int[] out = new int[1];
        HipErrors.call(HipNative.funcGetAttribute(out, attribute, function), "hipFuncGetAttribute");
        return out[0];
    }

    @Override
    public void launchKernel(long function, int gridDimX, int gridDimY, int gridDimZ,
                             int blockDimX, int blockDimY, int blockDimZ,
                             int sharedMemBytes, long stream, Object[] params) {
        enter();
        HipErrors.call(HipNative.launchKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                sharedMemBytes, stream, params), "hipModuleLaunchKernel");
    }

    @Override
    public void synchronize() {
        enter();
        HipErrors.check("hipDeviceSynchronize", () -> HipNative.synchronize());
    }

    @Override
    public void streamBeginCapture(long stream) {
        enter();
        HipErrors.check("hipStreamBeginCapture", () -> HipNative.streamBeginCapture(stream));
    }

    @Override
    public long streamEndCapture(long stream) {
        enter();
        long[] out = new long[1];
        HipErrors.check("hipStreamEndCapture", () -> HipNative.streamEndCapture(stream, out));
        return out[0];
    }

    @Override
    public long graphInstantiate(long graph) {
        enter();
        long[] out = new long[1];
        HipErrors.check("hipGraphInstantiate", () -> HipNative.graphInstantiate(out, graph));
        return out[0];
    }

    @Override
    public void graphLaunch(long exec, long stream) {
        enter();
        HipErrors.check("hipGraphLaunch", () -> HipNative.graphLaunch(exec, stream));
    }

    @Override
    public void graphDestroy(long graph) {
        HipErrors.check("hipGraphDestroy", () -> HipNative.graphDestroy(graph));
    }

    @Override
    public void graphExecDestroy(long exec) {
        HipErrors.check("hipGraphExecDestroy", () -> HipNative.graphExecDestroy(exec));
    }

    @Override
    public int graphNodeCount(long graph) {
        int[] out = new int[1];
        HipErrors.check("hipGraphGetNodes", () -> HipNative.graphNodeCount(graph, out));
        return out[0];
    }

    @Override
    public long[] graphNodesInOrder(long graph) {
        int count = graphNodeCount(graph);
        long[] out = new long[count];
        HipErrors.check("hipGraphGetNodes", () -> HipNative.graphNodesInOrder(graph, out));
        return out;
    }

    @Override
    public int graphNodeType(long node) {
        int[] out = new int[1];
        HipErrors.check("hipGraphNodeGetType", () -> HipNative.graphNodeType(node, out));
        switch (out[0]) {
            case HipNative.GRAPH_NODE_KERNEL:
            case HipNative.GRAPH_NODE_MEMCPY:
            case HipNative.GRAPH_NODE_MEMSET:
                return out[0];
            default:
                return -1;
        }
    }

    @Override
    public void graphExecKernelNodeSetParams(long exec, long node, long function, int gridDimX, int gridDimY,
                                             int gridDimZ, int blockDimX, int blockDimY, int blockDimZ,
                                             Object[] params) {
        enter();
        HipErrors.check("hipGraphExecKernelNodeSetParams", () -> HipNative.graphExecKernelNodeSetParams(
                exec, node, function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, params));
    }

    @Override
    public void graphExecMemcpyNodeSetParams(long exec, long node, long dst, long src, long bytes) {
        enter();
        HipErrors.check("hipGraphExecMemcpyNodeSetParams",
                () -> HipNative.graphExecMemcpyNodeSetParams(exec, node, dst, src, bytes));
    }

    @Override
    public void graphExecMemsetNodeSetParams(long exec, long node, long dst, int value, long count) {
        enter();
        HipErrors.check("hipGraphExecMemsetNodeSetParams",
                () -> HipNative.graphExecMemsetNodeSetParams(exec, node, dst, value, count));
    }

    @Override
    public String errorString(int code) {
        return HipNative.errorString(code);
    }
}
