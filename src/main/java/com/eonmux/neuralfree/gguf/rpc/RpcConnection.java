package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * One authenticated connection to a remote compute server: the HMAC handshake, the request/response exchange of
 * {@link RpcWire} frames, and the translation of the server's error frames back into the exception kinds the device
 * contract uses. The handshake reports every device the server serves ({@code rpc-server -d} list) and whether it
 * keeps a weight cache. Requests are serialized on the connection, so every remote backend of one endpoint shares
 * one ordered stream. Closing the connection is idempotent.
 */
final class RpcConnection implements AutoCloseable {

    private static final int CONNECT_TIMEOUT_MILLIS = 10_000;

    private final RpcServerSpec spec;
    private final Socket socket;
    private final RpcWire wire;
    private final List<RemoteDeviceInfo> devices;
    private final boolean cacheEnabled;
    private final Object ioLock = new Object();
    private volatile boolean closed;

    private RpcConnection(RpcServerSpec spec, Socket socket, RpcWire wire, List<RemoteDeviceInfo> devices,
                          boolean cacheEnabled) {
        this.spec = spec;
        this.socket = socket;
        this.wire = wire;
        this.devices = Collections.unmodifiableList(devices);
        this.cacheEnabled = cacheEnabled;
    }

    /**
     * Connects to {@code spec} and authenticates with its key. A {@code tls://} spec wraps the socket in TLS with
     * the client trust configuration ({@link RpcTls#clientContext()}) before the handshake.
     *
     * @throws IOException           when the endpoint cannot be reached or ends the handshake
     * @throws IllegalStateException when the server refuses the key, speaks another protocol version, or serves no
     *                               device
     */
    static RpcConnection open(RpcServerSpec spec) throws IOException {
        Objects.requireNonNull(spec, "spec");
        Socket socket = spec.tls()
                ? RpcTls.socket(RpcTls.clientContext(), spec.host(), spec.port())
                : new Socket();
        boolean ok = false;
        try {
            if (!spec.tls()) {
                socket.connect(new InetSocketAddress(spec.host(), spec.port()), CONNECT_TIMEOUT_MILLIS);
            }
            socket.setTcpNoDelay(true);
            socket.setKeepAlive(true);
            DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
            DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
            RpcWire wire = new RpcWire(in, out);
            Handshake handshake = handshake(spec, wire, in, out);
            ok = true;
            return new RpcConnection(spec, socket, wire, handshake.devices, handshake.cache);
        } finally {
            if (!ok) {
                closeQuietly(socket);
            }
        }
    }

    /** What the server's handshake answered: its devices and whether it keeps a weight cache. */
    private static final class Handshake {
        final List<RemoteDeviceInfo> devices;
        final boolean cache;

        Handshake(List<RemoteDeviceInfo> devices, boolean cache) {
            this.devices = devices;
            this.cache = cache;
        }
    }

    private static Handshake handshake(RpcServerSpec spec, RpcWire wire, DataInputStream in,
                                       DataOutputStream out) throws IOException {
        int magic = in.readInt();
        int version = in.readInt();
        if (magic != RpcWire.MAGIC) {
            throw new IllegalStateException("neuralfree.rpc entry " + spec.describe()
                    + " did not answer the remote-compute protocol magic");
        }
        if (version != RpcWire.VERSION) {
            throw new IllegalStateException("neuralfree.rpc entry " + spec.describe() + " speaks protocol version "
                    + version + ", this client speaks " + RpcWire.VERSION);
        }
        byte[] nonce = new byte[RpcWire.NONCE_BYTES];
        in.readFully(nonce);
        byte[] mac = RpcWire.hmac(spec.key().getBytes(StandardCharsets.UTF_8), RpcWire.authMessage(nonce));
        out.write(mac);
        out.flush();
        RpcWire.Reader r = new RpcWire.Reader(wire.readFrame());
        byte status = r.u8();
        if (status != RpcWire.STATUS_OK) {
            stringify(status, r, spec.describe());
        }
        return new Handshake(RpcWire.readHandshake(r, spec.describe()), r.bool());
    }

    /** The devices the handshake advertised, in the server's {@code -d} order. */
    List<RemoteDeviceInfo> devices() {
        return devices;
    }

    /** Whether the server keeps a weight cache ({@code rpc-server --cache}). */
    boolean cacheEnabled() {
        return cacheEnabled;
    }

    String describe() {
        return spec.describe();
    }

    /**
     * Sends one request and returns its response body, positioned after the status byte.
     *
     * @throws IllegalStateException          when the connection failed, or the server rejected the request
     * @throws UnsupportedOperationException  when the server's backend cannot perform the request
     * @throws IllegalArgumentException       when the server rejected an argument
     */
    RpcWire.Reader call(RpcWire.Writer request) {
        Objects.requireNonNull(request, "request");
        byte[] response;
        synchronized (ioLock) {
            ensureOpen();
            try {
                wire.writeFrame(request.toBytes());
                response = wire.readFrame();
            } catch (IOException e) {
                throw failed(e);
            }
        }
        return answer(response, spec.describe());
    }

    /**
     * Sends two requests back to back in one flush and reads both responses in order: one round trip where two
     * {@link #call}s take two. The server executes the frames of a connection strictly in order, so the second
     * request observes everything the first did. Both responses are read before either is examined, so the stream
     * stays in step whatever they answer; the caller examines the first answer before the second ({@link
     * Answers}). Only a request without side effects may follow a request that can fail: it still executes when
     * the first one failed, and its answer is then dropped.
     *
     * @throws IllegalStateException when the connection failed
     */
    Answers callPipelined(RpcWire.Writer first, RpcWire.Writer second) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        byte[] firstResponse;
        byte[] secondResponse;
        synchronized (ioLock) {
            ensureOpen();
            try {
                wire.writeFrame(first.toBytes(), false);
                wire.writeFrame(second.toBytes());
                firstResponse = wire.readFrame();
                secondResponse = wire.readFrame();
            } catch (IOException e) {
                throw failed(e);
            }
        }
        return new Answers(firstResponse, secondResponse, spec.describe());
    }

    /**
     * The two responses of a {@link #callPipelined} exchange, each examined on demand so a caller can account for
     * the first before the second raises its error.
     */
    static final class Answers {
        private final byte[] first;
        private final byte[] second;
        private final String endpoint;

        private Answers(byte[] first, byte[] second, String endpoint) {
            this.first = first;
            this.second = second;
            this.endpoint = endpoint;
        }

        /** The first response body, positioned after the status byte; raises the error it carries. */
        RpcWire.Reader first() {
            return answer(first, endpoint);
        }

        /** The second response body, positioned after the status byte; raises the error it carries. */
        RpcWire.Reader second() {
            return answer(second, endpoint);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("remote device " + spec.describe() + " is closed");
        }
    }

    /** Marks the connection failed after an I/O error and returns the exception to raise. */
    private IllegalStateException failed(IOException e) {
        closed = true;
        closeQuietly(socket);
        return new IllegalStateException("remote device " + spec.describe() + " connection failed: "
                + e.getMessage(), e);
    }

    /** A response body positioned after its status byte; an error frame raises the exception it carries. */
    private static RpcWire.Reader answer(byte[] response, String endpoint) {
        RpcWire.Reader r = new RpcWire.Reader(response);
        byte status = r.u8();
        if (status != RpcWire.STATUS_OK) {
            stringify(status, r, endpoint);
        }
        return r;
    }

    /** Raises the exception an error frame carries; never returns. */
    private static void stringify(byte status, RpcWire.Reader r, String endpoint) {
        if (status != RpcWire.STATUS_ERROR) {
            throw new IllegalStateException("remote device " + endpoint + " answered status " + status);
        }
        byte kind = r.u8();
        String message = r.utf();
        String text = "remote device " + endpoint + ": " + message;
        switch (kind) {
            case RpcWire.ERR_UNSUPPORTED:
                throw new UnsupportedOperationException(text);
            case RpcWire.ERR_UNSUPPORTED_KERNEL:
                throw new UnsupportedKernelException(text);
            case RpcWire.ERR_OUT_OF_MEMORY:
                throw new GpuOutOfMemoryException(text);
            case RpcWire.ERR_ILLEGAL_ARGUMENT:
                throw new IllegalArgumentException(text);
            default:
                throw new IllegalStateException(text);
        }
    }

    private static void closeQuietly(Socket socket) {
        try {
            socket.close();
        } catch (IOException ignored) {
            // the connection is already unusable
        }
    }

    @Override
    public void close() {
        synchronized (ioLock) {
            if (closed) {
                return;
            }
            closed = true;
            closeQuietly(socket);
        }
    }
}
