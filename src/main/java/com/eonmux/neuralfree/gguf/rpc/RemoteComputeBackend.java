package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A compute device of a remote machine, opened over the authenticated remote-compute protocol: the client half of
 * {@code nftool rpc-server}. It implements {@link ComputeBackend} by proxying every device operation to the server,
 * which runs it on one of its devices through the same backend APIs, so a layer split can treat a remote device as
 * one more layer target (llama.cpp's RPC device and {@code --rpc}). {@link #openAll} returns one view per device a
 * server advertises ({@code rpc-server -d list}), all sharing one connection and one pipelined command batch;
 * {@link #connect} returns the first. Dispatches, fills, copies and writes are deferred into the batch and flush
 * together, so a decode step costs one round trip; reads and other answered commands flush first. The device's
 * backend type and capabilities come from the server's handshake, so kernel libraries, work sizes and memory
 * planning are chosen for the hardware that actually runs them. Host-mapped buffers are not available over the
 * protocol and are refused. A view does not own the server's device: closing the last view of its endpoint closes
 * the connection, and the server releases the connection's buffers and cloned handles.
 */
public final class RemoteComputeBackend implements ComputeBackend {

    private final RemoteEndpoint endpoint;
    private final int device;
    private final RemoteDeviceInfo info;
    private final Set<Long> liveBuffers = Collections.synchronizedSet(new HashSet<>());
    private final AtomicLong allocatedBytes = new AtomicLong();
    private final RemoteComputeQueue computeQueue;
    private final RemoteComputeQueue transferQueue;
    private final AtomicBoolean closed = new AtomicBoolean();

    private RemoteComputeBackend(RemoteEndpoint endpoint, int device) {
        this.endpoint = endpoint;
        this.device = device;
        this.info = endpoint.devices().get(device);
        this.computeQueue = new RemoteComputeQueue(this, ComputeQueue.QueueType.COMPUTE);
        this.transferQueue = new RemoteComputeQueue(this, ComputeQueue.QueueType.TRANSFER);
        endpoint.retain();
    }

    /**
     * Connects to and authenticates with {@code spec}, returning its first device. Servers that serve several
     * devices expose the others through {@link #openAll}.
     *
     * @throws IOException           when the endpoint cannot be reached
     * @throws IllegalStateException when the server refuses the key, speaks another version, or is closed
     */
    public static RemoteComputeBackend connect(RpcServerSpec spec) throws IOException {
        return openAll(spec).get(0);
    }

    /**
     * Connects to and authenticates with {@code spec}, returning one backend per device the server advertises, in
     * the server's {@code -d} order. The views share one connection and release it together.
     *
     * @throws IOException           when the endpoint cannot be reached
     * @throws IllegalStateException when the server refuses the key, speaks another version, or is closed
     */
    public static List<RemoteComputeBackend> openAll(RpcServerSpec spec) throws IOException {
        RemoteEndpoint endpoint = RemoteEndpoint.open(spec);
        List<RemoteComputeBackend> out = new ArrayList<>(endpoint.devices().size());
        for (int i = 0; i < endpoint.devices().size(); i++) {
            out.add(new RemoteComputeBackend(endpoint, i));
        }
        return out;
    }

    @Override
    public BackendType backendType() {
        return info.backendType();
    }

    @Override
    public DeviceCapabilities capabilities() {
        return info.capabilities();
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
    public boolean supportsHostMappedBuffers() {
        return false;
    }

    @Override
    public HostMappedBuffer allocateHostMappedBuffer(long sizeInBytes) {
        throw new UnsupportedOperationException("a remote device does not expose host-mapped buffers; keep the"
                + " offloaded weights in the streamed tier instead");
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes, boolean readOnly) {
        if (sizeInBytes <= 0) {
            throw new IllegalArgumentException("buffer size must be positive, got " + sizeInBytes);
        }
        long id = request(RpcWire.OP_ALLOC, w -> w.i32(device).i64(sizeInBytes).bool(readOnly)).i64();
        liveBuffers.add(id);
        allocatedBytes.addAndGet(sizeInBytes);
        return new RemoteBuffer(this, id, sizeInBytes);
    }

    @Override
    public GpuBuffer allocateBuffer(long sizeInBytes) {
        return allocateBuffer(sizeInBytes, false);
    }

    /** Releases {@code buffer} on the server once; called by {@link RemoteBuffer#close()}. */
    void releaseBuffer(RemoteBuffer buffer) {
        if (!liveBuffers.remove(buffer.id())) {
            return;
        }
        allocatedBytes.addAndGet(-buffer.sizeInBytes());
        if (closed.get() || endpoint.isClosed()) {
            return;
        }
        endpoint.submit(new RpcWire.Writer(RpcWire.OP_FREE).i64(buffer.id()), false);
    }

    boolean ownsBuffer(RemoteBuffer buffer) {
        return liveBuffers.contains(buffer.id());
    }

    @Override
    public KernelHandle compileKernel(String kernelName, String kernelSource, String buildOptions) {
        Objects.requireNonNull(kernelName, "kernelName");
        Objects.requireNonNull(kernelSource, "kernelSource");
        long id = request(RpcWire.OP_COMPILE,
                w -> w.i32(device).utf(kernelName).rawString(kernelSource).nullableUtf(buildOptions)).i64();
        return new RemoteKernelHandle(this, id, kernelName, false);
    }

    @Override
    public KernelHandle loadKernel(KernelArtifact artifact) {
        Objects.requireNonNull(artifact, "artifact");
        if (!supportsArtifact(artifact.kind())) {
            throw new UnsupportedKernelException(backendType()
                    + " remote backend cannot load " + artifact.kind() + " artifact for " + artifact.kernelName());
        }
        long id = request(RpcWire.OP_LOAD, w -> {
            w.i32(device);
            RpcWire.writeArtifact(w, artifact);
        }).i64();
        return new RemoteKernelHandle(this, id, artifact.kernelName(), false);
    }

    @Override
    public boolean supportsArtifact(KernelArtifact.Kind kind) {
        switch (backendType()) {
            case OPENCL:
                return kind == KernelArtifact.Kind.OPENCL_SOURCE;
            case CUDA:
                return kind == KernelArtifact.Kind.CUDA_IMAGE;
            case HIP:
                return kind == KernelArtifact.Kind.HIP_IMAGE;
            case VULKAN:
                return kind == KernelArtifact.Kind.SPIRV;
            default:
                return false;
        }
    }

    @Override
    public KernelHandle cloneKernel(KernelHandle template) {
        if (!(template instanceof RemoteKernelHandle)
                || !((RemoteKernelHandle) template).belongsTo(this)) {
            throw new IllegalArgumentException("kernel is not a handle of this remote backend");
        }
        long source = ((RemoteKernelHandle) template).id();
        long id = request(RpcWire.OP_CLONE, w -> w.i64(source)).i64();
        return new RemoteKernelHandle(this, id, template.kernelName(), true);
    }

    @Override
    public void prepareKernel(KernelHandle kernel, long[] localWorkSize) {
        if (!(kernel instanceof RemoteKernelHandle) || !((RemoteKernelHandle) kernel).belongsTo(this)) {
            throw new IllegalArgumentException("kernel is not a handle of this remote backend");
        }
        long id = ((RemoteKernelHandle) kernel).id();
        endpoint.submit(new RpcWire.Writer(RpcWire.OP_PREPARE).i64(id).longArray(localWorkSize), false);
    }

    /** Releases a caller-owned clone on the server; called by {@link RemoteKernelHandle#close()}. */
    void releaseKernel(RemoteKernelHandle kernel) {
        if (closed.get() || endpoint.isClosed()) {
            return;
        }
        endpoint.submit(new RpcWire.Writer(RpcWire.OP_RELEASE_KERNEL).i64(kernel.id()), false);
    }

    @Override
    public ComputeQueue createComputeQueue() {
        return new RemoteComputeQueue(this, ComputeQueue.QueueType.COMPUTE);
    }

    @Override
    public long getAllocatedBytes() {
        return allocatedBytes.get();
    }

    @Override
    public long getFreeVramBytes() {
        return request(RpcWire.OP_FREE_VRAM, w -> w.i32(device)).i64();
    }

    /** The endpoint this backend is connected to, for messages. */
    public String endpoint() {
        return info.endpoint();
    }

    /** The device the server reported at the handshake. */
    public RemoteDeviceInfo info() {
        return info;
    }

    /** The device index of this view in the server's {@code -d} order. */
    int deviceIndex() {
        return device;
    }

    /** The pipelined command batch the queues of this backend append to. */
    RemoteEndpoint batch() {
        return endpoint;
    }

    /** Whether {@link #close()} ran. */
    boolean closed() {
        return closed.get();
    }

    /**
     * A request that needs an answer, prefixed by this view's device index for the device-addressed opcodes. The
     * buffered commands flush first.
     */
    private RpcWire.Reader request(byte opcode, java.util.function.Consumer<RpcWire.Writer> body) {
        RpcWire.Writer w = new RpcWire.Writer(opcode);
        body.accept(w);
        return endpoint.request(w);
    }

    /**
     * Appends one command to the endpoint's batch and returns the sequence an event of it carries.
     *
     * @throws IllegalStateException when the backend or the connection is closed
     */
    long submit(RpcWire.Writer member, boolean syncAfter) {
        if (closed.get()) {
            throw new IllegalStateException("remote device " + info.endpoint() + " is closed");
        }
        return endpoint.submit(member, syncAfter);
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        endpoint.release();
    }

    @Override
    public String toString() {
        return "RemoteComputeBackend{" + info.endpoint() + "[" + device + "], " + backendType() + " '"
                + capabilities().getDeviceName() + "'}";
    }
}
