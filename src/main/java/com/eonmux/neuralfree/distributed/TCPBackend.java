package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Collectives between ranks over TCP: one socket per pair of ranks, the lower rank accepting and the higher rank
 * connecting. Construction binds this rank's port on its bind address ({@code addresses[rank]} unless another is
 * given) and connects to every lower rank; connections from higher ranks are accepted in the background, and a
 * collective waits up to {@value #CONNECT_TIMEOUT_SECONDS} seconds for a connection it needs. All ranks call the same
 * collectives in the same order, with tensors of the same element count, as with PyTorch's process groups.
 *
 * <p><b>Security.</b> The protocol is neither authenticated nor encrypted: any host that can reach a rank's port can
 * join as a rank, read the exchanged values and inject its own. Run it only on a trusted, private network, and
 * firewall the rank ports from other hosts.</p>
 *
 * <p><b>Connections.</b> A connecting rank opens with a handshake (a magic word, the world size and its rank), which
 * the accepting rank reads within {@value #HANDSHAKE_TIMEOUT_MILLIS} ms. A connection whose handshake is late,
 * malformed, from another world size, or announces a rank that is not above the accepting rank or that is already
 * connected is closed and logged, and accepting goes on; an established connection is never replaced.</p>
 *
 * <p><b>Agreement.</b> Every collective starts with the ranks sending their {@link CollectiveDescriptor} (kind,
 * reduce op or source rank, element count) to rank 0, which returns the table of all of them; every rank checks it,
 * and when the ranks' calls differ every rank throws the same {@link IllegalArgumentException} and the backend stays
 * usable. An all-reduce sends its values along with its descriptor, so it costs one round trip; a barrier is the
 * agreement alone.</p>
 *
 * <p><b>Failures.</b> A rank waits at most {@value CollectiveDescriptor#TIMEOUT_MINUTES} minutes for a peer's data.
 * A collective that fails for another reason than the ranks disagreeing (a timeout, a lost connection, a peer that
 * breaks the protocol) leaves the streams out of step, so it closes the backend before it throws: the other ranks then
 * fail at once on the closed connections instead of waiting for the timeout.</p>
 *
 * <p><b>Results.</b> {@link #allReduce}: rank 0 reduces every rank's values in rank order
 * ({@link CollectiveTensors#reduce}) and sends the result back, so all ranks hold bit-identical results, written into
 * the caller's FLOAT32 tensor in place (under {@code noGrad}). {@link #broadcast} writes the source rank's values into
 * every other rank's FLOAT32 tensor. {@link #allGather} exchanges values pairwise in a fixed order (the lower rank of
 * each pair sends first, pairs in ascending order), which cannot deadlock, and replaces each entry of the output list
 * with a new tensor (the entries it replaces are not closed). Values cross through the host.</p>
 */
public final class TCPBackend implements DistributedBackend {
    private static final Logger LOGGER = LoggingUtil.getLogger(TCPBackend.class);
    private static final long CONNECT_TIMEOUT_SECONDS = 60;
    private static final int CONNECT_RETRIES = 30;
    private static final int CONNECT_ATTEMPT_MILLIS = 5000;
    private static final int HANDSHAKE_TIMEOUT_MILLIS = 10_000;
    /** The first int a connecting rank sends. */
    static final int HANDSHAKE_MAGIC = 0x4E465443;
    private static final int COLLECTIVE_TIMEOUT_MILLIS =
        Math.toIntExact(TimeUnit.MINUTES.toMillis(CollectiveDescriptor.TIMEOUT_MINUTES));
    private static final int DISCARD_CHUNK_BYTES = 64 * 1024;

    private final int rank;
    private final int worldSize;
    private final String[] addresses;
    private final int[] ports;
    private final String bindAddress;
    private final Map<Integer, Socket> connections;
    private final AtomicBoolean initialized;
    private final AtomicBoolean closed;
    private volatile IOException acceptFailure;
    /** The accepted connection whose handshake is being read, so that {@link #close()} can interrupt it. */
    private volatile Socket handshaking;
    private ServerSocket serverSocket;

    /**
     * Joins the job as {@code rank}, listening on {@code addresses[rank]:ports[rank]}.
     *
     * @param addresses the host name or IP address of every rank, indexed by rank
     * @param ports     the port of every rank, indexed by rank
     */
    public TCPBackend(int rank, int worldSize, String[] addresses, int[] ports) {
        this(rank, worldSize, addresses, ports, null);
    }

    /**
     * Joins the job as {@code rank}, listening on {@code bindAddress:ports[rank]}: for example {@code 0.0.0.0} to
     * accept connections on every interface when {@code addresses[rank]}, the address the other ranks connect to, is
     * not a local interface (behind NAT or a port forward).
     *
     * @param addresses   the host name or IP address of every rank, indexed by rank
     * @param ports       the port of every rank, indexed by rank
     * @param bindAddress the local address this rank listens on; null means {@code addresses[rank]}
     */
    public TCPBackend(int rank, int worldSize, String[] addresses, int[] ports, String bindAddress) {
        if (rank < 0 || rank >= worldSize) {
            throw new IllegalArgumentException("Rank must be in range [0, " + (worldSize - 1) + "]");
        }
        if (addresses.length != worldSize || ports.length != worldSize) {
            throw new IllegalArgumentException("Addresses and ports arrays must have length worldSize");
        }

        this.rank = rank;
        this.worldSize = worldSize;
        this.addresses = addresses.clone();
        this.ports = ports.clone();
        this.bindAddress = bindAddress != null ? bindAddress : this.addresses[rank];
        this.connections = new ConcurrentHashMap<>();
        this.initialized = new AtomicBoolean(false);
        this.closed = new AtomicBoolean(false);

        initialize();
    }

    private void initialize() {
        try {
            serverSocket = new ServerSocket();
            serverSocket.setReuseAddress(true);
            serverSocket.bind(new InetSocketAddress(bindAddress, ports[rank]));
        } catch (IOException e) {
            IOException failure = closeServerSocket(e);
            throw new UncheckedIOException("Failed to bind rank " + rank + " to " + bindAddress + ":" + ports[rank],
                failure);
        }
        Thread acceptThread = new Thread(this::acceptConnections, "TCPBackend-accept-" + rank);
        acceptThread.setDaemon(true);
        acceptThread.start();
        try {
            for (int i = 0; i < rank; i++) {
                connectToRank(i);
            }
        } catch (RuntimeException e) {
            try {
                close();
            } catch (RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
        initialized.set(true);
    }

    /**
     * Accepts the connections of the higher ranks until the backend closes. Only a failure of the listening socket
     * itself ends accepting; it is reported by {@link #connection}.
     */
    private void acceptConnections() {
        while (!closed.get()) {
            Socket socket;
            try {
                socket = serverSocket.accept();
            } catch (IOException e) {
                if (!closed.get()) {
                    acceptFailure = e;
                }
                return;
            }
            register(socket);
        }
    }

    /**
     * Reads the handshake of an accepted connection and registers it under the rank it announces. A connection that
     * fails the handshake, or announces a rank that is already connected, is closed and logged.
     */
    private void register(Socket socket) {
        handshaking = socket;
        try {
            if (closed.get()) {
                socket.close();
                return;
            }
            int remoteRank = readHandshake(socket);
            if (connections.putIfAbsent(remoteRank, socket) != null) {
                throw new IOException("rank " + remoteRank + " is already connected");
            }
            if (closed.get() && connections.remove(remoteRank, socket)) {
                socket.close();
            }
        } catch (IOException e) {
            refuse(socket, e);
        } finally {
            handshaking = null;
        }
    }

    /**
     * The rank a connecting peer announces, after checking its handshake; the socket then waits up to the collective
     * timeout on reads.
     *
     * @throws IOException when the handshake is late, malformed, from another world size or announces a rank that does
     *                     not connect to this one
     */
    private int readHandshake(Socket socket) throws IOException {
        socket.setSoTimeout(HANDSHAKE_TIMEOUT_MILLIS);
        socket.setTcpNoDelay(true);
        DataInputStream in = new DataInputStream(socket.getInputStream());
        if (in.readInt() != HANDSHAKE_MAGIC) {
            throw new IOException("the connection did not open with a TCPBackend handshake");
        }
        int remoteWorldSize = in.readInt();
        int remoteRank = in.readInt();
        if (remoteWorldSize != worldSize) {
            throw new IOException("the peer belongs to a job of " + remoteWorldSize + " ranks, this job has "
                + worldSize);
        }
        if (remoteRank <= rank || remoteRank >= worldSize) {
            throw new IOException("the peer announced rank " + remoteRank + "; only ranks above " + rank
                + " and below " + worldSize + " connect to it");
        }
        socket.setSoTimeout(COLLECTIVE_TIMEOUT_MILLIS);
        return remoteRank;
    }

    /** Closes a connection that failed its handshake and logs why (unless the backend is closing). */
    private void refuse(Socket socket, IOException reason) {
        try {
            socket.close();
        } catch (IOException closeFailure) {
            reason.addSuppressed(closeFailure);
        }
        if (!closed.get()) {
            LOGGER.warn("Rank {} refused the connection from {}: {}", rank, socket.getRemoteSocketAddress(),
                reason.getMessage(), reason);
        }
    }

    private void connectToRank(int targetRank) {
        for (int attempt = 1; ; attempt++) {
            Socket socket = new Socket();
            try {
                socket.connect(new InetSocketAddress(addresses[targetRank], ports[targetRank]),
                    CONNECT_ATTEMPT_MILLIS);
                socket.setTcpNoDelay(true);
                socket.setSoTimeout(COLLECTIVE_TIMEOUT_MILLIS);
                ByteBuffer handshake = ByteBuffer.allocate(3 * Integer.BYTES);
                handshake.putInt(HANDSHAKE_MAGIC).putInt(worldSize).putInt(rank);
                OutputStream out = socket.getOutputStream();
                out.write(handshake.array());
                out.flush();
                connections.put(targetRank, socket);
                return;
            } catch (IOException e) {
                try {
                    socket.close();
                } catch (IOException closeFailure) {
                    e.addSuppressed(closeFailure);
                }
                if (attempt >= CONNECT_RETRIES) {
                    throw new UncheckedIOException("Failed to connect to rank " + targetRank + " after "
                        + CONNECT_RETRIES + " attempts", e);
                }
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("Interrupted while connecting to rank " + targetRank, ie);
                }
            }
        }
    }

    @Override
    public int rank() {
        ensureOpen();
        return rank;
    }

    @Override
    public int worldSize() {
        ensureOpen();
        return worldSize;
    }

    /**
     * Reduces {@code tensor} across the ranks with {@code op} and writes the result into it in place.
     *
     * @throws IllegalArgumentException when {@code tensor} is not FLOAT32, or the ranks' calls differ (another
     *                                  collective, another op, or a tensor of another size), on every rank
     * @throws IllegalStateException    when the backend is closed, or a peer does not answer in time (which closes
     *                                  the backend)
     */
    @Override
    public synchronized void allReduce(Tensor tensor, ReduceOp op) {
        ensureOpen();
        Objects.requireNonNull(op, "ReduceOp cannot be null");
        CollectiveTensors.requireWritable(tensor);
        float[] data = CollectiveTensors.read(tensor);
        float[][] perRank = agree(CollectiveDescriptor.allReduce(op, data.length), data);
        float[] reduced = guarded(() -> {
            if (rank != 0) {
                return receiveFloats(0, data.length);
            }
            float[] result = CollectiveTensors.reduce(perRank, op);
            for (int r = 1; r < worldSize; r++) {
                sendFloats(r, result);
            }
            return result;
        });
        CollectiveTensors.writeBack(tensor, reduced);
    }

    /**
     * Copies the values of {@code srcRank}'s tensor into {@code tensor} on every other rank, in place.
     *
     * @throws IllegalArgumentException when {@code srcRank} is out of range or {@code tensor} is not FLOAT32, or the
     *                                  ranks' calls differ (another collective, another source rank, or a tensor of
     *                                  another size), on every rank
     * @throws IllegalStateException    when the backend is closed, or a peer does not answer in time (which closes
     *                                  the backend)
     */
    @Override
    public synchronized void broadcast(Tensor tensor, int srcRank) {
        ensureOpen();
        if (srcRank < 0 || srcRank >= worldSize) {
            throw new IllegalArgumentException("srcRank must be in range [0, " + (worldSize - 1) + "]");
        }
        CollectiveTensors.requireWritable(tensor);
        int count = Math.toIntExact(tensor.numElements());
        agree(CollectiveDescriptor.broadcast(srcRank, count), null);
        if (rank == srcRank) {
            float[] data = CollectiveTensors.read(tensor);
            guarded(() -> {
                for (int r = 0; r < worldSize; r++) {
                    if (r != rank) {
                        sendFloats(r, data);
                    }
                }
                return null;
            });
        } else {
            CollectiveTensors.writeBack(tensor, guarded(() -> receiveFloats(srcRank, count)));
        }
    }

    /**
     * Sets entry {@code r} of {@code output} to a new tensor holding rank {@code r}'s {@code input}, with the shape,
     * backend and dtype of this rank's {@code input}; the entries replaced are not closed.
     *
     * @throws IllegalArgumentException when {@code output} does not have {@code worldSize} entries, or the ranks'
     *                                  calls differ (another collective or a tensor of another size), on every rank
     * @throws IllegalStateException    when the backend is closed, or a peer does not answer in time (which closes
     *                                  the backend)
     */
    @Override
    public synchronized void allGather(List<Tensor> output, Tensor input) {
        ensureOpen();
        Objects.requireNonNull(output, "Output list cannot be null");
        if (output.size() != worldSize) {
            throw new IllegalArgumentException("Output list must have size equal to worldSize");
        }
        float[] inputData = CollectiveTensors.read(input);
        agree(CollectiveDescriptor.allGather(inputData.length), null);
        float[][] gathered = guarded(() -> {
            float[][] values = new float[worldSize][];
            values[rank] = inputData;
            for (int peer = 0; peer < worldSize; peer++) {
                if (peer == rank) {
                    continue;
                }
                if (rank < peer) {
                    sendFloats(peer, inputData);
                    values[peer] = receiveFloats(peer, inputData.length);
                } else {
                    values[peer] = receiveFloats(peer, inputData.length);
                    sendFloats(peer, inputData);
                }
            }
            return values;
        });
        for (int r = 0; r < worldSize; r++) {
            output.set(r, CollectiveTensors.shapedLike(input, gathered[r]));
        }
    }

    /**
     * Returns once every rank has called {@code barrier} the same number of times.
     *
     * @throws IllegalArgumentException when another rank called another collective, on every rank
     * @throws IllegalStateException    when the backend is closed, or a peer does not answer in time (which closes
     *                                  the backend)
     */
    @Override
    public synchronized void barrier() {
        ensureOpen();
        agree(CollectiveDescriptor.barrier(), null);
    }

    /**
     * The agreement round of a collective (see the class description): every rank sends {@code mine} to rank 0, an
     * all-reduce with its {@code values}; rank 0 returns the table of all descriptors, and every rank checks it.
     * Returns, on rank 0 of an all-reduce, every rank's values (entry {@code r} for rank {@code r}); null otherwise.
     *
     * @throws IllegalArgumentException when the ranks' descriptors differ, on every rank; the streams stay in step
     */
    private float[][] agree(CollectiveDescriptor mine, float[] values) {
        CollectiveDescriptor[] table = new CollectiveDescriptor[worldSize];
        float[][] perRank = new float[worldSize][];
        table[0] = mine;
        perRank[0] = values;
        if (worldSize > 1) {
            guarded(() -> rank == 0 ? gatherDescriptors(mine, table, perRank) : sendDescriptor(mine, values, table));
        }
        CollectiveDescriptor.requireAgreement(table);
        return rank == 0 && mine.kind() == CollectiveDescriptor.Kind.ALL_REDUCE ? perRank : null;
    }

    /**
     * Rank 0's side of the agreement: reads every peer's descriptor (and an all-reduce's values, kept when they can
     * be reduced with {@code mine}, else discarded), then sends the table to every peer.
     */
    private Void gatherDescriptors(CollectiveDescriptor mine, CollectiveDescriptor[] table, float[][] perRank) {
        for (int r = 1; r < worldSize; r++) {
            DataInputStream in = input(r);
            CollectiveDescriptor theirs = readDescriptor(r, in, 0);
            table[r] = theirs;
            if (theirs.kind() == CollectiveDescriptor.Kind.ALL_REDUCE) {
                if (theirs.count() == mine.count() && mine.kind() == CollectiveDescriptor.Kind.ALL_REDUCE) {
                    perRank[r] = readPayload(r, in, theirs.count());
                } else {
                    discard(r, in, (long) Float.BYTES * theirs.count());
                }
            }
        }
        ByteBuffer message = ByteBuffer.allocate(Integer.BYTES * CollectiveDescriptor.INTS * worldSize);
        for (CollectiveDescriptor descriptor : table) {
            for (int field : descriptor.toInts()) {
                message.putInt(field);
            }
        }
        for (int r = 1; r < worldSize; r++) {
            send(r, message.array());
        }
        return null;
    }

    /** A peer's side of the agreement: sends {@code mine} (with an all-reduce's values) and reads rank 0's table. */
    private Void sendDescriptor(CollectiveDescriptor mine, float[] values, CollectiveDescriptor[] table) {
        boolean withValues = mine.kind() == CollectiveDescriptor.Kind.ALL_REDUCE;
        ByteBuffer message = ByteBuffer.allocate(Integer.BYTES * CollectiveDescriptor.INTS
            + (withValues ? Float.BYTES * values.length : 0));
        for (int field : mine.toInts()) {
            message.putInt(field);
        }
        if (withValues) {
            message.asFloatBuffer().put(values);
        }
        send(0, message.array());
        DataInputStream in = input(0);
        for (int r = 0; r < worldSize; r++) {
            table[r] = readDescriptor(0, in, r);
        }
        return null;
    }

    /**
     * Reads one descriptor from {@code peer}; {@code of} is the rank it describes.
     *
     * @throws IllegalStateException when the ints do not encode a descriptor (the peer broke the protocol)
     */
    private CollectiveDescriptor readDescriptor(int peer, DataInputStream in, int of) {
        int[] fields = new int[CollectiveDescriptor.INTS];
        try {
            for (int i = 0; i < fields.length; i++) {
                fields[i] = in.readInt();
            }
        } catch (IOException e) {
            throw peerFailure(peer, e);
        }
        try {
            return CollectiveDescriptor.fromInts(fields, 0);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException("Rank " + peer + " sent rank " + rank + " a corrupt descriptor of rank "
                + of, e);
        }
    }

    /**
     * Runs one step of a collective's exchange. A failure leaves the streams out of step, so it closes the backend
     * (see the class description) before it is rethrown.
     */
    private <T> T guarded(Exchange<T> exchange) {
        try {
            return exchange.run();
        } catch (RuntimeException e) {
            try {
                close();
            } catch (RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /** The socket to {@code peer}, waiting for a higher rank's connection to be accepted. */
    private Socket connection(int peer) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(CONNECT_TIMEOUT_SECONDS);
        while (true) {
            ensureOpen();
            Socket socket = connections.get(peer);
            if (socket != null) {
                return socket;
            }
            IOException failure = acceptFailure;
            if (failure != null) {
                throw new UncheckedIOException("Rank " + rank + " stopped accepting connections", failure);
            }
            if (System.nanoTime() - deadline > 0) {
                throw new IllegalStateException("Rank " + rank + " has no connection to rank " + peer + " after "
                    + CONNECT_TIMEOUT_SECONDS + " seconds");
            }
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted while waiting for the connection to rank " + peer, e);
            }
        }
    }

    private DataInputStream input(int peer) {
        try {
            return new DataInputStream(connection(peer).getInputStream());
        } catch (IOException e) {
            throw peerFailure(peer, e);
        }
    }

    private void send(int peer, byte[] message) {
        Socket socket = connection(peer);
        try {
            OutputStream out = socket.getOutputStream();
            out.write(message);
            out.flush();
        } catch (IOException e) {
            throw peerFailure(peer, e);
        }
    }

    /** Sends {@code data} as one message: its length, then its floats (big-endian). */
    private void sendFloats(int peer, float[] data) {
        ByteBuffer message = ByteBuffer.allocate(Integer.BYTES + Float.BYTES * data.length);
        message.putInt(data.length);
        message.asFloatBuffer().put(data);
        send(peer, message.array());
    }

    /**
     * Receives one message of {@link #sendFloats} from {@code peer}.
     *
     * @throws IllegalStateException when it does not hold {@code expectedLength} values (the peer broke the protocol)
     */
    private float[] receiveFloats(int peer, int expectedLength) {
        DataInputStream in = input(peer);
        int length;
        try {
            length = in.readInt();
        } catch (IOException e) {
            throw peerFailure(peer, e);
        }
        if (length != expectedLength) {
            throw new IllegalStateException("Rank " + peer + " sent " + length + " values where rank " + rank
                + " expected " + expectedLength + " after the ranks agreed on the collective");
        }
        return readPayload(peer, in, length);
    }

    private float[] readPayload(int peer, DataInputStream in, int count) {
        byte[] payload = new byte[Float.BYTES * count];
        try {
            in.readFully(payload);
        } catch (IOException e) {
            throw peerFailure(peer, e);
        }
        float[] data = new float[count];
        ByteBuffer.wrap(payload).asFloatBuffer().get(data);
        return data;
    }

    /** Reads and drops {@code bytes} bytes from {@code peer}, keeping the stream in step. */
    private void discard(int peer, InputStream in, long bytes) {
        byte[] chunk = new byte[(int) Math.min(DISCARD_CHUNK_BYTES, Math.max(1, bytes))];
        long remaining = bytes;
        try {
            while (remaining > 0) {
                int read = in.read(chunk, 0, (int) Math.min(chunk.length, remaining));
                if (read < 0) {
                    throw new IOException("the connection ended " + remaining + " bytes before the message did");
                }
                remaining -= read;
            }
        } catch (IOException e) {
            throw peerFailure(peer, e);
        }
    }

    /** The failure of an exchange with {@code peer}: a timeout, or a lost connection. */
    private RuntimeException peerFailure(int peer, IOException e) {
        if (e instanceof SocketTimeoutException) {
            return new IllegalStateException("Rank " + rank + " waited more than "
                + CollectiveDescriptor.TIMEOUT_MINUTES + " minutes for rank " + peer + " in a collective", e);
        }
        return new UncheckedIOException("Rank " + rank + " lost its connection to rank " + peer, e);
    }

    @Override
    public boolean isInitialized() {
        return initialized.get() && !closed.get();
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("TCPBackend has been closed");
        }
    }

    /**
     * Closes every connection, a connection still in its handshake and the listening socket; idempotent. Everything
     * is tried; the first failure is thrown afterwards with later ones suppressed.
     */
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        IOException failure = null;
        for (Socket socket : connections.values()) {
            try {
                socket.close();
            } catch (IOException e) {
                failure = collect(failure, e);
            }
        }
        connections.clear();
        Socket pending = handshaking;
        if (pending != null) {
            try {
                pending.close();
            } catch (IOException e) {
                failure = collect(failure, e);
            }
        }
        failure = closeServerSocket(failure);
        if (failure != null) {
            throw new UncheckedIOException("Failed to close the connections of rank " + rank, failure);
        }
    }

    /** Closes the listening socket, if any; returns {@code failure} with a close failure collected. */
    private IOException closeServerSocket(IOException failure) {
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException e) {
                return collect(failure, e);
            }
        }
        return failure;
    }

    private static IOException collect(IOException failure, IOException e) {
        if (failure == null) {
            return e;
        }
        failure.addSuppressed(e);
        return failure;
    }

    /** One step of a collective's exchange; it throws unchecked exceptions only. */
    @FunctionalInterface
    private interface Exchange<T> {
        T run();
    }
}
