package com.eonmux.neuralfree.acceleration.hip;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * A {@link HipDriver} over a simulated AMD device (gfx1100 by default), for HIP backend tests
 * that need no GPU and no JNI shim.
 *
 * <ul>
 *   <li><b>Memory.</b> Device pointers address private byte blocks of a fixed capacity;
 *       page-locked host pointers address host blocks backed by heap arrays. Transfers copy
 *       between them.</li>
 *   <li><b>Kernels.</b> {@link #moduleLoadData} accepts any non-empty byte array that carries the
 *       fake code-object magic; the module exposes exactly the functions of {@link #functions}.
 *       The built-in {@link #SCALE_KERNEL} reads {@code (data, factor, n)} from its parameter
 *       snapshot and multiplies {@code n} floats of the device block by {@code factor}, so a
 *       dispatch produces real data a test reads back.</li>
 *   <li><b>Events.</b> An event completes when the test sets {@link FakeEvent#complete} or the
 *       stream or device is synchronized. Out-of-memory failures can be injected with
 *       {@link #failNextAllocs}.</li>
 * </ul>
 */
public final class FakeHipDriver implements HipDriver {

    /** Magic every fake code object starts with. */
    public static final byte[] CODE_OBJECT_MAGIC = "NFHSACO1".getBytes(StandardCharsets.US_ASCII);

    /** The one kernel the fake device provides: {@code (data, factor, n)} scaling floats in place. */
    public static final String SCALE_KERNEL = "nf_test_scale";

    private final Map<Long, byte[]> deviceBlocks = new HashMap<>();
    private final Map<Long, byte[]> hostBlocks = new HashMap<>();
    private final TreeMap<Long, Long> modules = new TreeMap<>();
    private final List<FakeEvent> events = new ArrayList<>();
    private final Map<Long, FakeEvent> eventById = new HashMap<>();
    /** Launches since the last {@link #launches()} read, newest last. */
    private final List<Launch> launchLog = new ArrayList<>();
    /** Streams whose commands record into a capture instead of executing. */
    private final Map<Long, List<FakeNode>> captures = new HashMap<>();
    /** Captured graphs by id. */
    private final Map<Long, List<FakeNode>> graphs = new HashMap<>();
    /** Executable graphs by id; their nodes are updated by the replay and run by a launch. */
    private final Map<Long, List<FakeNode>> execs = new HashMap<>();
    /** Every node ever created, so a node handle resolves to its kind. */
    private final Map<Long, FakeNode> nodesById = new HashMap<>();
    private long nextGraph = 1;
    private long nextGraphExec = 1;
    private long nextNode = 1;

    public long capacity = Long.MAX_VALUE;
    public long totalMem = 16L << 30;
    public String gcnArchName = "gfx1100";
    public String pciBusId = "0000:03:00.0";
    public byte[] uuidBytes = new byte[16];
    public long[] maxGridDims = {2147483647L, 65535L, 65535L};
    public long maxSharedPerBlock = 64L << 10;
    public int warpSize = 32;
    public int asyncEngineCount = 1;
    public int driverVersion = 6000;
    /** Number of allocations that fail with out-of-memory before the next one succeeds. */
    public int failNextAllocs;
    /** Functions a loaded module provides; every loaded module carries the same set. */
    public List<String> functions = new ArrayList<>(Arrays.asList(SCALE_KERNEL));

    private long used;
    private long nextDevicePointer = 0x1000_0000_0000L;
    private long nextHostPointer = 0x7000_0000_0000L;
    private long nextModule = 1;
    private long nextEvent = 0xE000_0000L;
    private long nextStream = 0x9000_0000_0000L;

    @Override
    public int deviceCount() {
        return 1;
    }

    @Override
    public void setDevice(int device) {
        if (device != 0) {
            throw HipErrors.translate(HipErrors.INVALID_DEVICE, "hipSetDevice(" + device + ")", -1);
        }
    }

    @Override
    public String deviceName(int device) {
        return device == 0 ? "AMD Radeon Test GPU" : null;
    }

    @Override
    public String gcnArchName(int device) {
        return device == 0 ? gcnArchName : null;
    }

    @Override
    public long totalMem(int device) {
        return device == 0 ? totalMem : -1;
    }

    @Override
    public int deviceAttribute(int device, int selector) {
        switch (selector) {
            case HipNative.ATTR_WARP_SIZE:
                return warpSize;
            case HipNative.ATTR_MAX_THREADS_PER_BLOCK:
                return 1024;
            case HipNative.ATTR_MAX_GRID_DIM_X:
                return (int) maxGridDims[0];
            case HipNative.ATTR_MAX_GRID_DIM_Y:
                return (int) maxGridDims[1];
            case HipNative.ATTR_MAX_GRID_DIM_Z:
                return (int) maxGridDims[2];
            case HipNative.ATTR_MULTIPROCESSOR_COUNT:
                return 60;
            case HipNative.ATTR_CLOCK_RATE:
                return 2_300_000;
            case HipNative.ATTR_SHARED_PER_BLOCK:
                return (int) maxSharedPerBlock;
            case HipNative.ATTR_SHARED_PER_BLOCK_OPTIN:
                return (int) maxSharedPerBlock;
            case HipNative.ATTR_INTEGRATED:
                return 0;
            case HipNative.ATTR_ASYNC_ENGINE_COUNT:
                return asyncEngineCount;
            default:
                throw HipErrors.translate(HipErrors.INVALID_VALUE, "deviceAttribute(" + selector + ")", -1);
        }
    }

    @Override
    public int driverVersion() {
        return driverVersion;
    }

    @Override
    public byte[] deviceUuid(int device) {
        return uuidBytes.clone();
    }

    @Override
    public String pciBusId(int device) {
        return pciBusId;
    }

    @Override
    public long memAlloc(long bytes) {
        if (failNextAllocs > 0) {
            failNextAllocs--;
            throw new GpuOutOfMemoryException("fake HIP device out of memory", bytes, "HIP");
        }
        if (bytes > capacity - used) {
            throw new GpuOutOfMemoryException("fake HIP device out of memory", bytes, "HIP");
        }
        byte[] block = new byte[(int) bytes];
        long pointer = nextDevicePointer;
        nextDevicePointer += align(bytes, 256);
        deviceBlocks.put(pointer, block);
        used += bytes;
        return pointer;
    }

    @Override
    public void memFree(long pointer) {
        byte[] block = deviceBlocks.remove(pointer);
        if (block == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipFree", -1);
        }
        used -= block.length;
    }

    @Override
    public long[] memGetInfo() {
        return new long[]{capacity - used, capacity};
    }

    @Override
    public long hostAlloc(long bytes) {
        byte[] block = new byte[(int) bytes];
        long pointer = nextHostPointer;
        nextHostPointer += align(bytes, 256);
        hostBlocks.put(pointer, block);
        return pointer;
    }

    @Override
    public void hostFree(long pointer) {
        hostBlocks.remove(pointer);
    }

    @Override
    public ByteBuffer pinnedView(long pointer, int bytes) {
        byte[] block = hostBlock(pointer);
        return ByteBuffer.wrap(block, (int) (pointer - hostBase(pointer)), bytes).slice()
                .order(ByteOrder.nativeOrder());
    }

    @Override
    public void memcpyHtoDAsync(long dst, long srcHost, long bytes, long stream) {
        byte[] device = deviceBlock(dst);
        byte[] host = hostBlock(srcHost);
        int deviceOffset = (int) (dst - deviceBase(dst));
        int hostOffset = (int) (srcHost - hostBase(srcHost));
        System.arraycopy(host, hostOffset, device, deviceOffset, (int) bytes);
    }

    @Override
    public void memcpyDtoHAsync(long dstHost, long src, long bytes, long stream) {
        byte[] device = deviceBlock(src);
        byte[] host = hostBlock(dstHost);
        int deviceOffset = (int) (src - deviceBase(src));
        int hostOffset = (int) (dstHost - hostBase(dstHost));
        System.arraycopy(device, deviceOffset, host, hostOffset, (int) bytes);
    }

    @Override
    public void memcpyDtoDAsync(long dst, long src, long bytes, long stream) {
        if (recording(stream)) {
            captures.get(stream).add(copyNode(dst, src, bytes));
            return;
        }
        copyDeviceBytes(dst, src, bytes);
    }

    @Override
    public void memsetD32Async(long dst, int value, long count, long stream) {
        if (recording(stream)) {
            captures.get(stream).add(fillNode(dst, value, count));
            return;
        }
        fillDeviceWords(dst, value, count);
    }

    @Override
    public long streamCreate() {
        return nextStream += 0x100;
    }

    @Override
    public void streamDestroy(long stream) {
        // Streams are plain ids here; nothing to free.
    }

    @Override
    public void streamSynchronize(long stream) {
        completeAll();
    }

    @Override
    public void streamWaitEvent(long stream, long event) {
        FakeEvent e = eventById.get(event);
        if (e == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipStreamWaitEvent", -1);
        }
    }

    @Override
    public long eventCreate() {
        FakeEvent event = new FakeEvent(nextEvent++);
        events.add(event);
        eventById.put(event.id, event);
        return event.id;
    }

    @Override
    public void eventDestroy(long event) {
        eventById.remove(event);
    }

    @Override
    public void eventRecord(long event, long stream) {
        // Recording keeps the event pending until a test completes it or a sync runs.
    }

    @Override
    public boolean eventQuery(long event) {
        FakeEvent e = eventById.get(event);
        if (e == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipEventQuery", -1);
        }
        return e.complete;
    }

    @Override
    public void eventSynchronize(long event) {
        FakeEvent e = eventById.get(event);
        if (e == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipEventSynchronize", -1);
        }
        e.complete = true;
    }

    @Override
    public long moduleLoadData(byte[] image) {
        if (image.length < CODE_OBJECT_MAGIC.length
                || !Arrays.equals(Arrays.copyOf(image, CODE_OBJECT_MAGIC.length), CODE_OBJECT_MAGIC)) {
            throw HipErrors.translate(HipErrors.INVALID_KERNEL_FILE, "hipModuleLoadData", -1);
        }
        long module = nextModule++;
        modules.put(module, (long) image.length);
        return module;
    }

    @Override
    public void moduleUnload(long module) {
        if (modules.remove(module) == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipModuleUnload", -1);
        }
    }

    @Override
    public long moduleGetFunction(long module, String name) {
        if (!modules.containsKey(module) || !functions.contains(name)) {
            throw new UnsupportedKernelException("hipModuleGetFunction(" + name + ") failed: the function is not in "
                    + "the module", null);
        }
        // Function ids share the module id space with a fake offset; the launch only logs the name.
        return module * 1000 + functions.indexOf(name);
    }

    @Override
    public int funcGetAttribute(long function, int attribute) {
        if (!functions.contains(functionName(function))) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipFuncGetAttribute", -1);
        }
        switch (attribute) {
            case HipNative.FUNC_MAX_THREADS_PER_BLOCK:
                return 1024;
            case HipNative.FUNC_SHARED_SIZE_BYTES:
                return 0;
            default:
                throw HipErrors.translate(HipErrors.INVALID_VALUE, "hipFuncGetAttribute(" + attribute + ")", -1);
        }
    }

    @Override
    public void launchKernel(long function, int gridDimX, int gridDimY, int gridDimZ,
                             int blockDimX, int blockDimY, int blockDimZ,
                             int sharedMemBytes, long stream, Object[] params) {
        String kernel = functionName(function);
        if (recording(stream)) {
            FakeNode node = kernelNode(kernel, function, gridDimX, gridDimY, gridDimZ,
                    blockDimX, blockDimY, blockDimZ, params);
            captures.get(stream).add(node);
            return;
        }
        execute(kernel, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, params);
    }

    /** Runs one kernel of a graph launch or a direct dispatch and logs it. */
    private void execute(String kernel, int gx, int gy, int gz, int bx, int by, int bz, Object[] params) {
        launchLog.add(new Launch(kernel, gx, gy, gz, bx, by, bz, params));
        if (SCALE_KERNEL.equals(kernel)) {
            runScaleKernel(params);
        }
    }

    @Override
    public void streamBeginCapture(long stream) {
        if (captures.containsKey(stream)) {
            throw HipErrors.translate(HipErrors.INVALID_VALUE, "hipStreamBeginCapture", -1);
        }
        captures.put(stream, new ArrayList<>());
    }

    @Override
    public long streamEndCapture(long stream) {
        List<FakeNode> nodes = captures.remove(stream);
        if (nodes == null) {
            throw HipErrors.translate(HipErrors.INVALID_VALUE, "hipStreamEndCapture", -1);
        }
        long graph = nextGraph++;
        graphs.put(graph, nodes);
        return graph;
    }

    @Override
    public long graphInstantiate(long graph) {
        List<FakeNode> nodes = graphs.get(graph);
        if (nodes == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipGraphInstantiate", -1);
        }
        List<FakeNode> copies = new ArrayList<>();
        for (FakeNode node : nodes) {
            copies.add(node.copy());
        }
        long exec = nextGraphExec++;
        execs.put(exec, copies);
        return exec;
    }

    @Override
    public void graphLaunch(long exec, long stream) {
        List<FakeNode> nodes = execs.get(exec);
        if (nodes == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipGraphLaunch", -1);
        }
        for (FakeNode node : nodes) {
            executeNode(node);
        }
    }

    @Override
    public void graphDestroy(long graph) {
        if (graphs.remove(graph) == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipGraphDestroy", -1);
        }
    }

    @Override
    public void graphExecDestroy(long exec) {
        if (execs.remove(exec) == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipGraphExecDestroy", -1);
        }
    }

    @Override
    public int graphNodeCount(long graph) {
        List<FakeNode> nodes = graphs.get(graph);
        if (nodes == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipGraphGetNodes", -1);
        }
        return nodes.size();
    }

    @Override
    public long[] graphNodesInOrder(long graph) {
        List<FakeNode> nodes = graphs.get(graph);
        if (nodes == null) {
            throw HipErrors.translate(HipErrors.INVALID_HANDLE, "hipGraphGetNodes", -1);
        }
        long[] ids = new long[nodes.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = nodes.get(i).id;
        }
        return ids;
    }

    @Override
    public int graphNodeType(long node) {
        FakeNode found = nodesById.get(node);
        return found == null ? -1 : found.kind;
    }

    @Override
    public void graphExecKernelNodeSetParams(long exec, long node, long function, int gx, int gy, int gz,
                                             int bx, int by, int bz, Object[] params) {
        FakeNode target = execNode(exec, node, "hipGraphExecKernelNodeSetParams");
        target.kernel = functionName(function);
        target.function = function;
        target.gx = gx;
        target.gy = gy;
        target.gz = gz;
        target.bx = bx;
        target.by = by;
        target.bz = bz;
        target.params = params.clone();
    }

    @Override
    public void graphExecMemcpyNodeSetParams(long exec, long node, long dst, long src, long bytes) {
        FakeNode target = execNode(exec, node, "hipGraphExecMemcpyNodeSetParams");
        target.dst = dst;
        target.src = src;
        target.bytes = bytes;
    }

    @Override
    public void graphExecMemsetNodeSetParams(long exec, long node, long dst, int value, long count) {
        FakeNode target = execNode(exec, node, "hipGraphExecMemsetNodeSetParams");
        target.dst = dst;
        target.value = value;
        target.count = count;
    }

    private FakeNode execNode(long exec, long node, String operation) {
        List<FakeNode> nodes = execs.get(exec);
        if (nodes != null) {
            for (FakeNode candidate : nodes) {
                if (candidate.id == node) {
                    return candidate;
                }
            }
        }
        throw HipErrors.translate(HipErrors.INVALID_HANDLE, operation, -1);
    }

    private void executeNode(FakeNode node) {
        switch (node.kind) {
            case HipNative.GRAPH_NODE_KERNEL:
                execute(node.kernel, node.gx, node.gy, node.gz, node.bx, node.by, node.bz, node.params);
                break;
            case HipNative.GRAPH_NODE_MEMCPY:
                copyDeviceBytes(node.dst, node.src, node.bytes);
                break;
            default:
                fillDeviceWords(node.dst, node.value, node.count);
                break;
        }
    }

    private boolean recording(long stream) {
        return captures.containsKey(stream);
    }

    private FakeNode kernelNode(String kernel, long function, int gx, int gy, int gz,
                                int bx, int by, int bz, Object[] params) {
        FakeNode node = FakeNode.kernel(nextNode++, kernel, function, gx, gy, gz, bx, by, bz, params);
        nodesById.put(node.id, node);
        return node;
    }

    private FakeNode copyNode(long dst, long src, long bytes) {
        FakeNode node = FakeNode.copy(nextNode++, dst, src, bytes);
        nodesById.put(node.id, node);
        return node;
    }

    private FakeNode fillNode(long dst, int value, long count) {
        FakeNode node = FakeNode.fill(nextNode++, dst, value, count);
        nodesById.put(node.id, node);
        return node;
    }

    private void copyDeviceBytes(long dst, long src, long bytes) {
        byte[] from = deviceBlock(src);
        byte[] to = deviceBlock(dst);
        System.arraycopy(from, (int) (src - deviceBase(src)), to, (int) (dst - deviceBase(dst)), (int) bytes);
    }

    private void fillDeviceWords(long dst, int value, long count) {
        byte[] device = deviceBlock(dst);
        int offset = (int) (dst - deviceBase(dst));
        ByteBuffer words = ByteBuffer.wrap(device).order(ByteOrder.LITTLE_ENDIAN);
        for (long i = 0; i < count; i++) {
            words.putInt(offset + 4 * (int) i, value);
        }
    }

    @Override
    public void synchronize() {
        completeAll();
    }

    @Override
    public String errorString(int code) {
        return "hipError " + code;
    }

    // ------------------------------------------------------------------------------------------
    // Test control
    // ------------------------------------------------------------------------------------------

    /** Marks every recorded event complete. */
    public void completeAll() {
        for (FakeEvent event : events) {
            event.complete = true;
        }
    }

    /** The bytes of a device block, for direct assertions. */
    public byte[] deviceBytes(long pointer) {
        return deviceBlocks.get(deviceBase(pointer));
    }

    /** The launches since the last call, oldest first. */
    public List<Launch> launches() {
        List<Launch> copy = new ArrayList<>(launchLog);
        launchLog.clear();
        return copy;
    }

    /** Events still registered. */
    public List<FakeEvent> events() {
        return events;
    }

    /** The name of a fake function id ({@code module * 1000 + index}). */
    public String functionName(long function) {
        return functions.get((int) (function % 1000));
    }

    /** Runs the fake scale kernel: {@code data[i] *= factor} for {@code i < n}. */
    private void runScaleKernel(Object[] params) {
        if (params.length != 3) {
            throw new IllegalArgumentException("the fake scale kernel takes 3 parameters, got " + params.length);
        }
        long data = (Long) params[0];
        float factor = (Float) params[1];
        int n = (Integer) params[2];
        byte[] block = deviceBlock(data);
        int offset = (int) (data - deviceBase(data));
        ByteBuffer floats = ByteBuffer.wrap(block).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            int at = offset + 4 * i;
            floats.putFloat(at, floats.getFloat(at) * factor);
        }
    }

    private byte[] deviceBlock(long pointer) {
        byte[] block = deviceBlocks.get(deviceBase(pointer));
        if (block == null) {
            throw new IllegalStateException("fake HIP device: no device block at 0x" + Long.toHexString(pointer));
        }
        return block;
    }

    private long deviceBase(long pointer) {
        // Device blocks are 256-byte aligned, so the base is the greatest allocated start <= pointer.
        long best = -1;
        for (Long start : deviceBlocks.keySet()) {
            if (start <= pointer && start > best) {
                best = start;
            }
        }
        return best;
    }

    private byte[] hostBlock(long pointer) {
        byte[] block = hostBlocks.get(hostBase(pointer));
        if (block == null) {
            throw new IllegalStateException("fake HIP device: no page-locked block at 0x" + Long.toHexString(pointer));
        }
        return block;
    }

    private long hostBase(long pointer) {
        long best = -1;
        for (Long start : hostBlocks.keySet()) {
            if (start <= pointer && start > best) {
                best = start;
            }
        }
        return best;
    }

    private static long align(long n, long alignment) {
        return (n + alignment - 1) / alignment * alignment;
    }

    /** One recorded command of a fake capture: a kernel dispatch, a device copy or a fill. */
    public static final class FakeNode {
        public final long id;
        public final int kind;
        public String kernel;
        public long function;
        public int gx;
        public int gy;
        public int gz;
        public int bx;
        public int by;
        public int bz;
        public Object[] params;
        public long src;
        public long dst;
        public long bytes;
        public int value;
        public long count;

        private FakeNode(long id, int kind) {
            this.id = id;
            this.kind = kind;
        }

        static FakeNode kernel(long id, String kernel, long function, int gx, int gy, int gz,
                               int bx, int by, int bz, Object[] params) {
            FakeNode node = new FakeNode(id, HipNative.GRAPH_NODE_KERNEL);
            node.kernel = kernel;
            node.function = function;
            node.gx = gx;
            node.gy = gy;
            node.gz = gz;
            node.bx = bx;
            node.by = by;
            node.bz = bz;
            node.params = params.clone();
            return node;
        }

        static FakeNode copy(long id, long dst, long src, long bytes) {
            FakeNode node = new FakeNode(id, HipNative.GRAPH_NODE_MEMCPY);
            node.dst = dst;
            node.src = src;
            node.bytes = bytes;
            return node;
        }

        static FakeNode fill(long id, long dst, int value, long count) {
            FakeNode node = new FakeNode(id, HipNative.GRAPH_NODE_MEMSET);
            node.dst = dst;
            node.value = value;
            node.count = count;
            return node;
        }

        /** An executable-graph copy of this node: same identity, independently updatable fields. */
        FakeNode copy() {
            FakeNode node = new FakeNode(id, kind);
            node.kernel = kernel;
            node.function = function;
            node.gx = gx;
            node.gy = gy;
            node.gz = gz;
            node.bx = bx;
            node.by = by;
            node.bz = bz;
            node.params = params == null ? null : params.clone();
            node.src = src;
            node.dst = dst;
            node.bytes = bytes;
            node.value = value;
            node.count = count;
            return node;
        }
    }

    /** An event that completes when a test or a synchronization says so. */
    public static final class FakeEvent {        public final long id;
        public boolean complete;

        FakeEvent(long id) {
            this.id = id;
        }
    }

    /** One recorded kernel launch, for assertions about grids, blocks and arguments. */
    public static final class Launch {
        public final String kernel;
        public final int gridX;
        public final int gridY;
        public final int gridZ;
        public final int blockX;
        public final int blockY;
        public final int blockZ;
        public final Object[] params;

        Launch(String kernel, int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, Object[] params) {
            this.kernel = kernel;
            this.gridX = gridX;
            this.gridY = gridY;
            this.gridZ = gridZ;
            this.blockX = blockX;
            this.blockY = blockY;
            this.blockZ = blockZ;
            this.params = params.clone();
        }
    }
}
