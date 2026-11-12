package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import javax.net.ssl.SSLContext;

/**
 * The remote compute server of a layer split ({@code nftool rpc-server}): it serves the GPU backends of this machine
 * over TCP so a client on another machine uses them as layer-split devices (llama.cpp's {@code rpc-server} with its
 * {@code -d} device list). The client allocates buffers, loads kernels, uploads weights and dispatches exactly as it
 * would on a local backend; the server runs every command on the device it names and answers in order. Commands that
 * need no answer arrive as one {@link RpcWire#OP_BATCH} frame the server executes member by member on its device
 * queues without finishing, and answers once after the last one, so a decode step is one pass over the queues
 * instead of one finish per command (the async dispatcher of llama.cpp's transport); the queue finishes on the
 * batch's {@code FINISH} member, on a read, and on a cache-hit write, which are the points whose results the client
 * observes.
 *
 * <p>Every connection authenticates with the shared key: the server sends a random nonce, the client answers with
 * {@code HMAC-SHA256(key, magic || version || nonce)}, and the server drops the connection on a mismatch before any
 * device command runs. With {@code tls://} transport ({@link RpcTls}), the TLS layer wraps the listener and every
 * accepted socket before the handshake. Buffer and kernel ids are scoped to their connection and each carries the
 * device it was allocated or loaded on, so one client can never touch another connection's allocations and a buffer
 * is only used on its device. Commands of one connection run in wire order; a failing batch member answers an error
 * naming how many members ran, and the rest are skipped. A {@code --cache} ({@link RpcWeightCache}) stores uploads
 * by content hash and answers {@code WRITE_HASH} probes from it, so a reconnecting client skips the re-upload. The
 * server does not own the backends; the caller that opened it closes it.</p>
 */
public final class RpcComputeServer implements AutoCloseable {

    private final List<ComputeBackend> backends;
    private final List<Object> backendLocks;
    private final byte[] key;
    private final String bindHost;
    private final int requestedPort;
    /** The weight cache the {@code WRITE} cache flag stores into and {@code WRITE_HASH} probes; null when off. */
    private final RpcWeightCache cache;
    /** The TLS context of the listener, or null for a plain socket. */
    private final SSLContext tlsContext;
    private final SecureRandom random = new SecureRandom();
    private final AtomicLong ids = new AtomicLong();
    private final Set<Session> sessions = Collections.synchronizedSet(new HashSet<>());
    private ServerSocket serverSocket;
    private Thread acceptor;
    private volatile int port;
    private volatile boolean started;
    private volatile boolean closed;

    /**
     * A server over one backend, listening on {@code host}:{@code port} for clients holding {@code key}, without a
     * weight cache and over a plain socket.
     *
     * @param port the TCP port, 0 for an ephemeral one ({@link #port()} reports it)
     * @throws IllegalArgumentException when the key is blank, the host is blank or the port is out of range
     */
    public RpcComputeServer(ComputeBackend backend, String key, String host, int port) {
        this(Collections.singletonList(backend), key, host, port, null, null);
    }

    /**
     * A server over {@code backends} in {@code -d} order. Each device answers by its index in the list, which the
     * client names per request; commands against device {@code i} run under {@code i}'s own lock, so independent
     * devices make progress without waiting on each other.
     *
     * @param cache      the weight cache, or null when the server runs without one ({@code rpc-server --cache})
     * @param tlsContext the server-side TLS context (from {@link RpcTls#server(Path, Path)}), or null for a plain
     *                   socket
     * @throws IllegalArgumentException when the key is blank, the host is blank, the port is out of range, or the
     *                                  backend list is empty
     */
    public RpcComputeServer(List<ComputeBackend> backends, String key, String host, int port,
                            RpcWeightCache cache, SSLContext tlsContext) {
        Objects.requireNonNull(backends, "backends");
        if (backends.isEmpty()) {
            throw new IllegalArgumentException("rpc-server needs at least one device to serve");
        }
        for (ComputeBackend backend : backends) {
            Objects.requireNonNull(backend, "backend");
        }
        if (key == null || key.trim().isEmpty()) {
            throw new IllegalArgumentException("rpc-server needs a shared key; the transport is authenticated");
        }
        if (host == null || host.trim().isEmpty()) {
            throw new IllegalArgumentException("rpc-server needs a bind host");
        }
        if (port < 0 || port > 65535) {
            throw new IllegalArgumentException("rpc-server port " + port + " is outside 0..65535");
        }
        this.backends = new ArrayList<>(backends);
        this.backendLocks = new ArrayList<>(backends.size());
        for (int i = 0; i < backends.size(); i++) {
            backendLocks.add(new Object());
        }
        this.key = key.trim().getBytes(StandardCharsets.UTF_8);
        this.bindHost = host.trim();
        this.requestedPort = port;
        this.cache = cache;
        this.tlsContext = tlsContext;
    }

    /**
     * Binds the listening socket and starts accepting connections.
     *
     * @throws IOException           when the endpoint cannot be bound
     * @throws IllegalStateException when the server was started or closed already
     */
    public void start() throws IOException {
        if (started || closed) {
            throw new IllegalStateException("rpc-server was already started or closed");
        }
        ServerSocket socket = tlsContext != null
                ? RpcTls.serverSocket(tlsContext)
                : new ServerSocket();
        try {
            socket.setReuseAddress(true);
            socket.bind(new InetSocketAddress(bindHost, requestedPort));
        } catch (IOException e) {
            closeQuietly(socket);
            throw e;
        }
        serverSocket = socket;
        port = socket.getLocalPort();
        started = true;
        acceptor = new Thread(this::acceptLoop, "neuralfree-rpc-server");
        acceptor.setDaemon(true);
        acceptor.start();
    }

    /** The bound TCP port (the ephemeral one when 0 was requested); 0 before {@link #start()}. */
    public int port() {
        return port;
    }

    /** The devices the server serves, in {@code -d} order, for messages. */
    public List<ComputeBackend> backends() {
        return Collections.unmodifiableList(backends);
    }

    /** Blocks until the server is closed. */
    public void awaitTermination() throws InterruptedException {
        Thread thread = acceptor;
        if (thread != null) {
            thread.join();
        }
    }

    private void acceptLoop() {
        while (!closed) {
            Socket socket;
            try {
                socket = serverSocket.accept();
            } catch (IOException e) {
                if (closed) {
                    return;
                }
                continue;
            }
            Session session = new Session(socket);
            sessions.add(session);
            Thread worker = new Thread(session::run, "neuralfree-rpc-session");
            worker.setDaemon(true);
            worker.start();
        }
    }

    /** One authenticated client connection and the resources it owns on this server. */
    private final class Session implements AutoCloseable {
        private final Socket socket;
        /** This connection's buffers by id, each with the device it was allocated on. */
        private final Map<Long, Owned<GpuBuffer>> buffers = new LinkedHashMap<>();
        /** This connection's kernels by id, each with its device and whether the connection owns (cloned) it. */
        private final Map<Long, Owned<KernelHandle>> kernels = new LinkedHashMap<>();

        Session(Socket socket) {
            this.socket = socket;
        }

        void run() {
            try {
                socket.setTcpNoDelay(true);
                socket.setKeepAlive(true);
                DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
                DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
                RpcWire wire = new RpcWire(in, out);
                if (!authenticate(wire, in, out)) {
                    return;
                }
                writeHandshake(wire);
                serve(wire);
            } catch (IOException ignored) {
                // the peer closed the connection
            } finally {
                close();
                sessions.remove(this);
            }
        }

        private boolean authenticate(RpcWire wire, DataInputStream in, DataOutputStream out) throws IOException {
            byte[] nonce = new byte[RpcWire.NONCE_BYTES];
            random.nextBytes(nonce);
            out.writeInt(RpcWire.MAGIC);
            out.writeInt(RpcWire.VERSION);
            out.write(nonce);
            out.flush();
            byte[] got = new byte[32];
            try {
                in.readFully(got);
            } catch (EOFException e) {
                return false;
            }
            byte[] expected = RpcWire.hmac(key, RpcWire.authMessage(nonce));
            if (!RpcWire.sameMac(expected, got)) {
                RpcWire.Writer error = new RpcWire.Writer(RpcWire.STATUS_ERROR);
                error.u8(RpcWire.ERR_ILLEGAL_STATE).utf("authentication failed");
                wire.writeFrame(error);
                return false;
            }
            return true;
        }

        /** The handshake body: every device's type, capabilities and host-mapped support, then the cache flag. */
        private void writeHandshake(RpcWire wire) throws IOException {
            RpcWire.Writer info = new RpcWire.Writer(RpcWire.STATUS_OK);
            List<RemoteDeviceInfo> infos = new ArrayList<>(backends.size());
            for (ComputeBackend backend : backends) {
                infos.add(new RemoteDeviceInfo(backend.backendType(), backend.capabilities(),
                        backend.supportsHostMappedBuffers(), bindHost + ":" + port));
            }
            RpcWire.writeHandshake(info, infos, cache != null);
            wire.writeFrame(info);
        }

        private void serve(RpcWire wire) throws IOException {
            while (!closed) {
                byte[] payload;
                try {
                    payload = wire.readFrame();
                } catch (EOFException e) {
                    return;
                }
                RpcWire.Reader request = new RpcWire.Reader(payload);
                RpcWire.Writer response;
                try {
                    response = executeFrame(request);
                } catch (RuntimeException e) {
                    response = error(e);
                }
                wire.writeFrame(response);
            }
        }

        /**
         * Executes one request frame: a single command, or an {@code OP_BATCH} whose members run in order on their
         * devices and whose answer comes once, carrying how many members ran (a failing member skips the rest and
         * answers the error instead).
         */
        private RpcWire.Writer executeFrame(RpcWire.Reader request) {
            byte opcode = request.opcode();
            if (opcode != RpcWire.OP_BATCH) {
                RpcWire.Writer ok = new RpcWire.Writer(RpcWire.STATUS_OK);
                execute(opcode, request, ok);
                return ok;
            }
            int count = request.i32();
            int executed = 0;
            try {
                for (int i = 0; i < count; i++) {
                    execute(request.opcode(), request, null);
                    executed++;
                }
            } catch (RuntimeException e) {
                return error(e, "[" + executed + " of " + count + " batch members executed] "
                        + (e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage()));
            }
            RpcWire.Writer ok = new RpcWire.Writer(RpcWire.STATUS_OK);
            ok.i32(executed);
            return ok;
        }

        /** Dispatches one command; {@code ok} carries the response of the round-trip commands and is null in a batch. */
        private void execute(byte opcode, RpcWire.Reader r, RpcWire.Writer ok) {
            switch (opcode) {
                case RpcWire.OP_ALLOC:
                    alloc(r, ok);
                    return;
                case RpcWire.OP_FREE:
                    free(r);
                    return;
                case RpcWire.OP_WRITE:
                    write(r);
                    return;
                case RpcWire.OP_WRITE_HASH:
                    writeHash(r, ok);
                    return;
                case RpcWire.OP_READ:
                    read(r, ok);
                    return;
                case RpcWire.OP_FILL:
                    fill(r);
                    return;
                case RpcWire.OP_COPY:
                    copy(r);
                    return;
                case RpcWire.OP_COMPILE:
                    compile(r, ok);
                    return;
                case RpcWire.OP_LOAD:
                    load(r, ok);
                    return;
                case RpcWire.OP_CLONE:
                    clone(r, ok);
                    return;
                case RpcWire.OP_PREPARE:
                    prepare(r);
                    return;
                case RpcWire.OP_ENQUEUE:
                    enqueue(r);
                    return;
                case RpcWire.OP_FINISH:
                    finishAll();
                    return;
                case RpcWire.OP_FLUSH:
                    flushAll();
                    return;
                case RpcWire.OP_ALLOCATED:
                    allocated(ok);
                    return;
                case RpcWire.OP_FREE_VRAM:
                    freeVram(r, ok);
                    return;
                case RpcWire.OP_RELEASE_KERNEL:
                    releaseKernel(r);
                    return;
                default:
                    throw new IllegalArgumentException("unknown remote-compute opcode " + RpcWire.opName(opcode));
            }
        }

        // ------------------------------------------------------------------ device commands

        private ComputeBackend backend(int device) {
            if (device < 0 || device >= backends.size()) {
                throw new IllegalArgumentException("device index " + device + " is outside the "
                        + backends.size() + " device(s) this server serves");
            }
            return backends.get(device);
        }

        private <T> T onDevice(int device, Supplier<T> command) {
            synchronized (backendLocks.get(device)) {
                return command.get();
            }
        }

        private void onDevice(int device, Runnable command) {
            synchronized (backendLocks.get(device)) {
                command.run();
            }
        }

        private void alloc(RpcWire.Reader r, RpcWire.Writer ok) {
            int device = r.i32();
            long size = r.i64();
            boolean readOnly = r.bool();
            GpuBuffer buffer = onDevice(device, () -> backend(device).allocateBuffer(size, readOnly));
            long id = ids.incrementAndGet();
            buffers.put(id, new Owned<>(buffer, device));
            ok.i64(id);
        }

        private void free(RpcWire.Reader r) {
            long id = r.i64();
            Owned<GpuBuffer> owned = owned(buffers, id, "buffer");
            onDevice(owned.device, () -> owned.value.close());
            buffers.remove(id);
        }

        /** One write; the cache flag stores the chunk in the weight cache once the device queued it. */
        private void write(RpcWire.Reader r) {
            Owned<GpuBuffer> target = owned(buffers, r.i64(), "buffer");
            long offset = r.i64();
            byte[] data = r.rawBytes();
            boolean cacheFlag = r.bool();
            onDevice(target.device, () -> {
                closeEvent(queue(target.device).enqueueWriteBuffer(target.value, offset, data.length, data));
            });
            if (cacheFlag && cache != null && data.length >= RpcWire.HASH_THRESHOLD) {
                cache.store(RpcWire.fnv1a(data), data);
            }
        }

        /**
         * A cache probe: answers whether the weight cache holds {@code size} bytes under the hash and wrote them to
         * the buffer when it does, finishing the device first so the data is durable before the answer. A stored
         * entry of another size is stale and is dropped, so the next upload re-stores it.
         */
        private void writeHash(RpcWire.Reader r, RpcWire.Writer ok) {
            Owned<GpuBuffer> target = owned(buffers, r.i64(), "buffer");
            long offset = r.i64();
            long size = r.i64();
            byte[] hashBytes = r.rawBytes();
            if (hashBytes.length != 8) {
                throw new IllegalArgumentException("a write hash probe carries 8 hash bytes, got "
                        + hashBytes.length);
            }
            long hash = java.nio.ByteBuffer.wrap(hashBytes).getLong();
            if (offset < 0 || size < 0 || offset + size > target.value.sizeInBytes()) {
                throw new IllegalArgumentException("cached write range [" + offset + ", " + (offset + size)
                        + ") exceeds the buffer size of " + target.value.sizeInBytes() + " bytes");
            }
            byte[] data = cache == null ? null : cache.load(hash);
            if (data == null || data.length != size) {
                if (data != null) {
                    cache.dropIfStale(hash, size);
                }
                ok.bool(false);
                return;
            }
            onDevice(target.device, () -> {
                closeEvent(queue(target.device).enqueueWriteBuffer(target.value, offset, data.length, data));
                queue(target.device).finish();
            });
            ok.bool(true);
        }

        private void read(RpcWire.Reader r, RpcWire.Writer ok) {
            Owned<GpuBuffer> source = owned(buffers, r.i64(), "buffer");
            long offset = r.i64();
            int length = r.i32();
            if (length < 0 || length > RpcWire.MAX_FRAME) {
                throw new IllegalArgumentException("read of " + length + " bytes is outside 0.."
                        + RpcWire.MAX_FRAME);
            }
            byte[] data = new byte[length];
            onDevice(source.device, () -> {
                closeEvent(queue(source.device).enqueueReadBuffer(source.value, offset, length, data));
                queue(source.device).finish();
            });
            ok.rawBytes(data);
        }

        private void fill(RpcWire.Reader r) {
            Owned<GpuBuffer> target = owned(buffers, r.i64(), "buffer");
            long offset = r.i64();
            long length = r.i64();
            int pattern = r.i32();
            onDevice(target.device, () -> {
                queue(target.device).enqueueFillBuffer(target.value, offset, length, pattern);
            });
        }

        private void copy(RpcWire.Reader r) {
            Owned<GpuBuffer> from = owned(buffers, r.i64(), "copy source buffer");
            long fromOffset = r.i64();
            Owned<GpuBuffer> to = owned(buffers, r.i64(), "copy destination buffer");
            long toOffset = r.i64();
            long length = r.i64();
            if (from.device != to.device) {
                throw new IllegalArgumentException("a copy runs on one device; source is on device " + from.device
                        + " and destination on device " + to.device);
            }
            onDevice(from.device, () -> {
                queue(from.device).enqueueCopyBuffer(from.value, fromOffset, to.value, toOffset, length);
            });
        }

        private void compile(RpcWire.Reader r, RpcWire.Writer ok) {
            int device = r.i32();
            String name = r.utf();
            String source = r.rawString();
            String options = r.nullableUtf();
            KernelHandle handle = onDevice(device, () -> backend(device).compileKernel(name, source, options));
            ok.i64(recordKernel(handle, device, false));
        }

        private void load(RpcWire.Reader r, RpcWire.Writer ok) {
            int device = r.i32();
            KernelArtifact artifact = RpcWire.readArtifact(r);
            KernelHandle handle = onDevice(device, () -> backend(device).loadKernel(artifact));
            ok.i64(recordKernel(handle, device, false));
        }

        private void clone(RpcWire.Reader r, RpcWire.Writer ok) {
            Owned<KernelHandle> source = owned(kernels, r.i64(), "kernel");
            KernelHandle handle = onDevice(source.device, () -> backend(source.device).cloneKernel(source.value));
            ok.i64(recordKernel(handle, source.device, true));
        }

        private void prepare(RpcWire.Reader r) {
            Owned<KernelHandle> kernel = owned(kernels, r.i64(), "kernel");
            long[] local = r.longArray();
            onDevice(kernel.device, () -> backend(kernel.device).prepareKernel(kernel.value, local));
        }

        private void enqueue(RpcWire.Reader r) {
            Owned<KernelHandle> kernel = owned(kernels, r.i64(), "kernel");
            long[] global = r.longArray();
            long[] local = r.longArray();
            DispatchBuffers scope = new DispatchBuffers(buffers, kernel.device);
            Object[] args = RpcWire.readArgs(r, scope);
            if (global == null) {
                throw new IllegalArgumentException("dispatch without global work sizes");
            }
            scope.checkDevice();
            onDevice(kernel.device, () -> {
                closeEvent(queue(kernel.device).enqueueKernel(kernel.value, global, local, args));
            });
        }

        private void releaseKernel(RpcWire.Reader r) {
            long id = r.i64();
            Owned<KernelHandle> kernel = kernels.get(id);
            if (kernel == null) {
                throw new IllegalArgumentException("kernel id " + id + " was not loaded by this connection");
            }
            if (!kernel.owned) {
                throw new IllegalArgumentException("kernel id " + id + " is backend-owned and cannot be released");
            }
            onDevice(kernel.device, () -> kernel.value.close());
            kernels.remove(id);
        }

        private void finishAll() {
            for (int device = 0; device < backends.size(); device++) {
                final int d = device;
                onDevice(d, () -> queue(d).finish());
            }
        }

        private void flushAll() {
            for (int device = 0; device < backends.size(); device++) {
                final int d = device;
                onDevice(d, () -> queue(d).flush());
            }
        }

        /** The bytes the server's devices have allocated, across every device. */
        private void allocated(RpcWire.Writer ok) {
            long sum = 0;
            for (int device = 0; device < backends.size(); device++) {
                final int d = device;
                sum += onDevice(d, () -> backend(d).getAllocatedBytes());
            }
            ok.i64(sum);
        }

        private void freeVram(RpcWire.Reader r, RpcWire.Writer ok) {
            int device = r.i32();
            ok.i64(onDevice(device, () -> backend(device).getFreeVramBytes()));
        }

        private ComputeQueue queue(int device) {
            return backend(device).computeQueue();
        }

        private <T> Owned<T> owned(Map<Long, Owned<T>> map, long id, String what) {
            Owned<T> value = map.get(id);
            if (value == null) {
                throw new IllegalArgumentException(what + " id " + id + " was not allocated by this connection");
            }
            return value;
        }

        private long recordKernel(KernelHandle handle, int device, boolean owned) {
            long id = ids.incrementAndGet();
            kernels.put(id, new Owned<>(handle, device, owned));
            return id;
        }

        private void closeEvent(GpuEvent event) {
            if (event != null) {
                event.close();
            }
        }

        @Override
        public void close() {
            List<Owned<KernelHandle>> liveKernels;
            List<Owned<GpuBuffer>> liveBuffers;
            synchronized (this) {
                liveKernels = new ArrayList<>(kernels.values());
                liveBuffers = new ArrayList<>(buffers.values());
                kernels.clear();
                buffers.clear();
            }
            for (Owned<KernelHandle> kernel : liveKernels) {
                if (kernel.owned) {
                    try {
                        onDevice(kernel.device, () -> kernel.value.close());
                    } catch (RuntimeException ignored) {
                        // the connection is going away
                    }
                }
            }
            for (Owned<GpuBuffer> buffer : liveBuffers) {
                try {
                    onDevice(buffer.device, () -> buffer.value.close());
                } catch (RuntimeException ignored) {
                    // the connection is going away
                }
            }
            closeQuietly(socket);
        }
    }

    /**
     * The buffer arguments of one dispatch, resolved by id against the connection's buffer table as the arguments
     * are read: each lookup is one table probe, and the first argument living on another device than the kernel is
     * remembered, so the dispatch is refused after the whole argument list proved well-formed.
     */
    static final class DispatchBuffers implements RpcWire.BufferResolver {
        private final Map<Long, Owned<GpuBuffer>> buffers;
        private final int device;
        private int foreignDevice = -1;

        DispatchBuffers(Map<Long, Owned<GpuBuffer>> buffers, int device) {
            this.buffers = buffers;
            this.device = device;
        }

        @Override
        public GpuBuffer buffer(long id) {
            Owned<GpuBuffer> owned = buffers.get(id);
            if (owned == null) {
                return null;
            }
            if (owned.device != device && foreignDevice < 0) {
                foreignDevice = owned.device;
            }
            return owned.value;
        }

        /**
         * @throws IllegalArgumentException when an argument buffer lives on another device than the kernel
         */
        void checkDevice() {
            if (foreignDevice >= 0) {
                throw new IllegalArgumentException("a kernel on device " + device + " takes a buffer of device "
                        + foreignDevice);
            }
        }
    }

    /** One id's server-side value, the device it lives on, and whether this connection owns it. */
    static final class Owned<T> {
        final T value;
        final int device;
        final boolean owned;

        Owned(T value, int device) {
            this(value, device, false);
        }

        Owned(T value, int device, boolean owned) {
            this.value = value;
            this.device = device;
            this.owned = owned;
        }
    }

    /** The frame answering a failed request, with the exception kind the client raises. */
    private static RpcWire.Writer error(RuntimeException e) {
        return error(e, e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage());
    }

    private static RpcWire.Writer error(RuntimeException e, String message) {
        RpcWire.Writer writer = new RpcWire.Writer(RpcWire.STATUS_ERROR);
        byte kind;
        if (e instanceof GpuOutOfMemoryException) {
            kind = RpcWire.ERR_OUT_OF_MEMORY;
        } else if (e instanceof UnsupportedKernelException) {
            kind = RpcWire.ERR_UNSUPPORTED_KERNEL;
        } else if (e instanceof UnsupportedOperationException) {
            kind = RpcWire.ERR_UNSUPPORTED;
        } else if (e instanceof IllegalArgumentException) {
            kind = RpcWire.ERR_ILLEGAL_ARGUMENT;
        } else {
            kind = RpcWire.ERR_ILLEGAL_STATE;
        }
        writer.u8(kind).utf(message);
        return writer;
    }

    private static void closeQuietly(ServerSocket socket) {
        try {
            socket.close();
        } catch (IOException ignored) {
            // nothing left to release
        }
    }

    private static void closeQuietly(Socket socket) {
        try {
            socket.close();
        } catch (IOException ignored) {
            // nothing left to release
        }
    }

    /** Closes the listening socket and every connection, releasing their server-side resources. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        ServerSocket listening = serverSocket;
        serverSocket = null;
        if (listening != null) {
            closeQuietly(listening);
        }
        List<Session> open = new ArrayList<>(sessions);
        for (Session session : open) {
            session.close();
        }
        sessions.clear();
    }

    @Override
    public String toString() {
        return "RpcComputeServer{" + bindHost + ":" + port + ", " + backends.size() + " device(s)"
                + (cache != null ? ", weight cache" : "") + (tlsContext != null ? ", tls" : "") + "}";
    }
}
