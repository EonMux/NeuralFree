package com.eonmux.neuralfree.gguf.engine.kvevents;

import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.zeromq.ZMQException;
import org.zeromq.ZMsg;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.channels.ClosedChannelException;
import java.nio.channels.Pipe;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The KV cache event publisher on vLLM's ZMQ transport ({@code vllm/distributed/kv_events.py}'s
 * {@code ZmqEventPublisher}), wire-compatible with vLLM's KV-aware routers (llm-d, Dynamo): a background thread
 * msgpack-encodes every {@link KvEventBatch} ({@link KvEventBatchMsgpack}) and sends it on a PUB socket as the
 * multipart message {@code [topic, sequence, payload]}, the sequence an 8-byte big-endian counter from 0. An
 * optional ROUTER replay socket keeps the last {@code buffer_steps} payloads: a client (DEALER or REQ) sends the
 * first sequence it missed as 8 big-endian bytes and receives {@code ["", topic, sequence, payload]} for every
 * buffered batch from there on, then the end marker {@code ["", "", -1, ""]}.
 *
 * <p>The engine's thread only queues the batch ({@code max_queue_size} deep, waiting while it is full, as vLLM's
 * queue does), stamping the configured data-parallel rank on a batch that has none. The publisher thread owns the
 * sockets: a batch whose encoding or send fails is logged and lost, its sequence number skipped, as vLLM's is.
 * {@link #close()} lets the thread flush the queue for up to a second, as vLLM's shutdown does, then drops what is
 * left and releases the sockets.
 */
public final class ZmqEventPublisher implements KvEventPublisher {

    private static final Logger LOGGER = Logger.getLogger(ZmqEventPublisher.class.getName());

    /** Milliseconds the publisher thread waits for a batch, or for a batch or a replay request, before it re-checks. */
    private static final int POLL_MILLIS = 100;
    /** Milliseconds {@link #close()} lets the publisher thread flush the queue, vLLM's {@code SHUTDOWN_TIMEOUT}. */
    private static final long SHUTDOWN_TIMEOUT_MILLIS = 1_000;
    /** The sequence of the replay's end marker, vLLM's {@code END_SEQ}. */
    private static final long END_SEQ = -1L;
    private static final byte[] EMPTY = new byte[0];
    /** The one byte a wake-up signal writes. */
    private static final byte[] WAKE_BYTE = {1};
    /** The queue entry {@link #close()} appends so the publisher thread stops once it has sent the rest. */
    private static final Object STOP = new Object();

    private final int dataParallelRank;
    private final int bufferSteps;
    private final String endpoint;
    private final String replayEndpoint;
    private final byte[] topic;
    private final BlockingQueue<Object> queue;
    private final ZContext context;
    private final ZMQ.Socket pub;
    private final ZMQ.Socket replay;
    /**
     * With a replay socket, the pipe a queued batch or the stop entry signals, so the publisher thread waits on the
     * replay socket and the queue at once and answers either at once; null without one.
     */
    private final Pipe wake;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Thread thread;
    /** Set when the flush timed out: the publisher thread drops the queued batches and stops. */
    private volatile boolean abandon;

    /** The payloads the replay socket serves, oldest first; the publisher thread's alone. */
    private final ArrayDeque<Buffered> buffer = new ArrayDeque<>();
    /** The sequence number of the next batch; the publisher thread's alone. */
    private long nextSeq;

    /** A published payload held for replay. */
    private static final class Buffered {
        final long seq;
        final byte[] payload;

        Buffered(long seq, byte[] payload) {
            this.seq = seq;
            this.payload = payload;
        }
    }

    /** The publisher of vLLM's default configuration: PUB bound on {@code tcp://*:5557}, no replay socket. */
    public ZmqEventPublisher() {
        this(ZmqEventPublisherConfig.defaults());
    }

    /**
     * The publisher {@code config} describes, its endpoints' ports offset by the data-parallel rank and its sockets
     * opened straight away, as vLLM's constructor does: the PUB socket binds a wildcard-host endpoint
     * ({@code tcp://*:5557}, {@code tcp://[::]:5557}) and connects to any other; the replay socket binds.
     *
     * @throws IllegalStateException when an endpoint cannot be bound or connected
     */
    public ZmqEventPublisher(ZmqEventPublisherConfig config) {
        Objects.requireNonNull(config, "config");
        this.dataParallelRank = config.dataParallelRank();
        this.bufferSteps = config.bufferSteps();
        this.topic = config.topic().getBytes(StandardCharsets.UTF_8);
        this.queue = config.maxQueueSize() == 0 ? new LinkedBlockingQueue<>()
                : new ArrayBlockingQueue<>(config.maxQueueSize());
        String pubAt = ZmqEventPublisherConfig.offsetEndpointPort(config.endpoint(), dataParallelRank);
        String replayAt = ZmqEventPublisherConfig.offsetEndpointPort(config.replayEndpoint(), dataParallelRank);
        this.context = new ZContext();
        try {
            this.pub = context.createSocket(SocketType.PUB);
            pub.setLinger(0);
            pub.setHWM(config.hwm());
            this.endpoint = isBindEndpoint(pubAt) ? bind(pub, "endpoint", pubAt) : connect(pub, pubAt);
            if (replayAt == null) {
                this.replay = null;
                this.replayEndpoint = null;
                this.wake = null;
            } else {
                this.replay = context.createSocket(SocketType.ROUTER);
                replay.setLinger(0);
                // a full replay plus its end marker fits even when ZMQ sees half the HWM as unacknowledged
                replay.setSndHWM(2 * (bufferSteps + 1));
                this.replayEndpoint = bind(replay, "replay_endpoint", replayAt);
                this.wake = openWake();
            }
        } catch (RuntimeException e) {
            context.close();
            throw e;
        }
        LOGGER.info("publishing KV cache events on " + endpoint
                + (replayEndpoint == null ? "" : ", replaying on " + replayEndpoint));
        this.thread = new Thread(this::run, "zmq-publisher");
        thread.setDaemon(true);
        thread.start();
    }

    /**
     * The PUB address in use: the configured one offset by the rank, with the port the system picked for an
     * ephemeral ({@code :0}) bind.
     */
    public String endpoint() {
        return endpoint;
    }

    /** The replay address in use (as {@link #endpoint()} resolves it), or null without a replay socket. */
    public String replayEndpoint() {
        return replayEndpoint;
    }

    /**
     * Queues {@code batch} for the publisher thread, stamping the configured data-parallel rank when the batch has
     * none. Waits while {@code max_queue_size} batches wait, as vLLM's publisher does.
     *
     * @throws IllegalStateException when the publisher is closed, or the wait was interrupted
     */
    @Override
    public void publish(KvEventBatch batch) {
        Objects.requireNonNull(batch, "batch");
        KvEventBatch stamped = batch.dataParallelRank() == null ? batch.withRank(dataParallelRank) : batch;
        try {
            while (!closed.get()) {
                if (queue.offer(stamped, POLL_MILLIS, TimeUnit.MILLISECONDS)) {
                    signal();
                    return;
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted queueing the KV event batch", e);
        }
        throw new IllegalStateException("the KV events ZMQ publisher on " + endpoint + " is closed");
    }

    /**
     * Lets the publisher thread send the queued batches for up to a second, then drops the rest and waits for the
     * thread to release the sockets. Idempotent.
     */
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        queue.offer(STOP);
        signal();
        boolean interrupted = false;
        try {
            thread.join(SHUTDOWN_TIMEOUT_MILLIS);
        } catch (InterruptedException e) {
            interrupted = true;
        }
        if (thread.isAlive()) {
            LOGGER.warning("the KV events ZMQ publisher on " + endpoint + " drops " + queue.size()
                    + " queued batches after " + SHUTDOWN_TIMEOUT_MILLIS + " ms");
            abandon = true;
            try {
                thread.join();
            } catch (InterruptedException e) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public String toString() {
        return "zmq:" + endpoint;
    }

    /** vLLM's {@code _is_bind_endpoint}: a wildcard host binds; any other host connects. */
    private static boolean isBindEndpoint(String endpoint) {
        return endpoint.contains("*") || endpoint.contains("::");
    }

    /** The PUB endpoint that binds this node's IP and picks a port, vLLM's ephemeral wildcard. */
    static final String EPHEMERAL_WILDCARD = "tcp://*:0";

    /** Binds {@code socket}; answers the bound address, with the node IP and picked port of the ephemeral wildcard. */
    private String bind(ZMQ.Socket socket, String field, String endpoint) {
        try {
            if (endpoint.equals(EPHEMERAL_WILDCARD)) {
                String ip = ZmqEventPublisherConfig.nodeIp();
                if (ip.equals("0.0.0.0") || ip.equals("::")) {
                    throw new IllegalStateException("the KV events " + field + " " + endpoint + " needs a node IP;"
                            + " set " + ZmqEventPublisherConfig.PROP_HOST_IP);
                }
                if (ZmqEventPublisherConfig.ipv6(ip)) {
                    socket.setIPv6(true);
                }
                String at = ZmqEventPublisherConfig.tcpUri(ip, 0);
                if (!socket.bind(at)) {
                    throw new IllegalStateException("the KV events " + field + " " + at + " cannot be bound");
                }
                return reportBound(socket, ip);
            }
            if (!socket.bind(endpoint)) {
                throw new IllegalStateException("the KV events " + field + " " + endpoint + " cannot be bound");
            }
        } catch (ZMQException e) {
            throw new IllegalStateException("the KV events " + field + " " + endpoint + " cannot be bound: "
                    + e.getMessage(), e);
        }
        return endpoint.endsWith(":0") ? socket.getLastEndpoint() : endpoint;
    }

    /** The bound endpoint as {@code tcp://<node ip>:<port>}, the port the system picked off the socket. */
    private static String reportBound(ZMQ.Socket socket, String ip) {
        String last = socket.getLastEndpoint();
        if (last == null) {
            throw new IllegalStateException("the KV events ephemeral bind reports no endpoint");
        }
        int port = Integer.parseInt(last.substring(last.lastIndexOf(':') + 1));
        return ZmqEventPublisherConfig.tcpUri(ip, port);
    }

    private static String connect(ZMQ.Socket socket, String endpoint) {
        try {
            if (!socket.connect(endpoint)) {
                throw new IllegalStateException("the KV events endpoint " + endpoint + " cannot be connected");
            }
        } catch (ZMQException e) {
            throw new IllegalStateException("the KV events endpoint " + endpoint + " cannot be connected: "
                    + e.getMessage(), e);
        }
        return endpoint;
    }

    /**
     * The publisher thread: serves a waiting replay, then sends the next queued batch, until the stop entry (or,
     * when a full queue left no room for it, the drained queue of a closed publisher).
     */
    private void run() {
        ZMQ.Poller poller = null;
        try {
            if (replay != null) {
                poller = context.createPoller(2);
                poller.register(replay, ZMQ.Poller.POLLIN);
                poller.register(wake.source(), ZMQ.Poller.POLLIN);
            }
            while (!abandon) {
                if (replay != null) {
                    serviceReplaySafely();
                }
                Object item = poller == null ? queue.poll(POLL_MILLIS, TimeUnit.MILLISECONDS) : queue.poll();
                if (item == STOP || item == null && closed.get() && queue.isEmpty()) {
                    return;
                }
                if (item != null) {
                    publishOne((KvEventBatch) item);
                } else if (poller != null) {
                    awaitWork(poller);
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            if (poller != null) {
                poller.close();
            }
            closeWake();
            context.close();
        }
    }

    /** A non-blocking pipe for {@link #wake}. */
    private static Pipe openWake() {
        try {
            Pipe pipe = Pipe.open();
            pipe.source().configureBlocking(false);
            pipe.sink().configureBlocking(false);
            return pipe;
        } catch (IOException e) {
            throw new UncheckedIOException("the KV events publisher cannot open its wake-up pipe", e);
        }
    }

    /** Wakes the publisher thread waiting in {@link #awaitWork}; a full pipe already holds a pending wake-up. */
    private void signal() {
        if (wake == null) {
            return;
        }
        try {
            wake.sink().write(ByteBuffer.wrap(WAKE_BYTE));
        } catch (ClosedChannelException e) {
            // the publisher thread has ended and closed the pipe: nothing waits for the signal
        } catch (IOException e) {
            LOGGER.log(Level.WARNING, "waking the KV events publisher failed; it re-checks within "
                    + POLL_MILLIS + " ms", e);
        }
    }

    /**
     * Waits up to {@link #POLL_MILLIS} for a replay request or a queued batch, then empties the wake-up pipe, whose
     * bytes only signal: the loop re-reads the queue and the replay socket itself.
     */
    private void awaitWork(ZMQ.Poller poller) {
        poller.poll(POLL_MILLIS);
        ByteBuffer drain = ByteBuffer.allocate(64);
        try {
            while (wake.source().read(drain) > 0) {
                drain.clear();
            }
        } catch (IOException e) {
            LOGGER.log(Level.WARNING, "reading the KV events publisher's wake-up pipe failed", e);
            pause();
        }
    }

    private void closeWake() {
        if (wake == null) {
            return;
        }
        try {
            wake.sink().close();
            wake.source().close();
        } catch (IOException e) {
            LOGGER.log(Level.WARNING, "closing the KV events publisher's wake-up pipe failed", e);
        }
    }

    /** Sends one batch and buffers its payload for replay; a failure loses the batch and its sequence number. */
    private void publishOne(KvEventBatch batch) {
        long seq = nextSeq++;
        try {
            byte[] payload = KvEventBatchMsgpack.encode(batch);
            pub.sendMore(topic);
            pub.sendMore(sequence(seq));
            pub.send(payload);
            if (bufferSteps > 0) {
                if (buffer.size() == bufferSteps) {
                    buffer.removeFirst();
                }
                buffer.addLast(new Buffered(seq, payload));
            }
        } catch (RuntimeException e) {
            LOGGER.log(Level.WARNING, "publishing the KV event batch " + seq + " failed; the batch is lost", e);
            pause();
        }
    }

    private void serviceReplaySafely() {
        try {
            serviceReplay();
        } catch (RuntimeException e) {
            LOGGER.log(Level.WARNING, "the KV events replay failed", e);
        }
    }

    /**
     * Answers one waiting replay request {@code [client, "", start]}: every buffered batch from sequence
     * {@code start} on, then the end marker. A request of another shape is logged and dropped, as vLLM's is.
     */
    private void serviceReplay() {
        ZMsg request = ZMsg.recvMsg(replay, ZMQ.DONTWAIT);
        if (request == null) {
            return;
        }
        try {
            if (request.size() != 3) {
                LOGGER.warning("ignoring the KV events replay request of " + request.size()
                        + " frames (a client sends one frame: the 8-byte start sequence)");
                return;
            }
            byte[] client = request.pop().getData();
            request.pop();
            long start = startSequence(request.pop().getData());
            for (Buffered buffered : buffer) {
                if (Long.compareUnsigned(buffered.seq, start) >= 0) {
                    sendReplay(client, topic, buffered.seq, buffered.payload);
                }
            }
            sendReplay(client, EMPTY, END_SEQ, EMPTY);
        } finally {
            request.destroy();
        }
    }

    /**
     * The unsigned big-endian start sequence of a replay request (vLLM's {@code int.from_bytes(..., "big")}); a
     * number past 64 bits saturates, so it replays nothing.
     */
    private static long startSequence(byte[] bytes) {
        long start = 0;
        for (byte b : bytes) {
            if (Long.compareUnsigned(start, -1L >>> 8) > 0) {
                return -1L;
            }
            start = (start << 8) | (b & 0xff);
        }
        return start;
    }

    private void sendReplay(byte[] client, byte[] frameTopic, long seq, byte[] payload) {
        replay.sendMore(client);
        replay.sendMore(EMPTY);
        replay.sendMore(frameTopic);
        replay.sendMore(sequence(seq));
        replay.send(payload);
    }

    /** {@code seq} as the 8 big-endian bytes of the wire (two's complement, so {@link #END_SEQ} is all ones). */
    private static byte[] sequence(long seq) {
        byte[] bytes = new byte[Long.BYTES];
        for (int i = Long.BYTES - 1; i >= 0; i--) {
            bytes[i] = (byte) (seq >>> ((Long.BYTES - 1 - i) * 8));
        }
        return bytes;
    }

    /** Backs off after a failed send, so a broken socket does not spin the thread. */
    private static void pause() {
        try {
            TimeUnit.MILLISECONDS.sleep(POLL_MILLIS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
