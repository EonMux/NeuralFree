package com.eonmux.neuralfree.gguf.engine.kvevents;

import org.msgpack.core.MessagePack;
import org.msgpack.value.ArrayValue;
import org.msgpack.value.MapValue;
import org.msgpack.value.Value;
import org.msgpack.value.ValueFactory;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;
import org.zeromq.ZMsg;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.ServerSocket;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * A vLLM-style KV events subscriber for the tests of both modules: SUB and DEALER sockets speaking the frames of
 * vLLM's {@code ZmqEventPublisher}, and msgpack decoding of the {@code KVEventBatch} payloads. A malformed frame
 * fails with an {@link AssertionError}, whatever the test framework.
 */
public final class ZmqKvEventsTestSupport {

    /** Milliseconds a test waits for a frame before it fails. */
    public static final int TIMEOUT_MILLIS = 5_000;

    private ZmqKvEventsTestSupport() {
    }

    /** One message as a subscriber receives it: the topic, the sequence number and the msgpack payload. */
    public static final class Frame {
        public final String topic;
        public final long seq;
        public final byte[] payload;

        Frame(String topic, long seq, byte[] payload) {
            this.topic = topic;
            this.seq = seq;
            this.payload = payload;
        }

        /** The decoded {@code KVEventBatch} array {@code [ts, events, data_parallel_rank]}. */
        public ArrayValue batch() {
            return decode(payload);
        }
    }

    /** A TCP port no socket listens on right now. */
    public static int freePort() {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** The loopback address of a bound {@code endpoint}'s port. */
    public static String loopback(String endpoint) {
        return "tcp://127.0.0.1:" + endpoint.substring(endpoint.lastIndexOf(':') + 1);
    }

    /** A SUB socket on {@code endpoint}, subscribed to {@code topicPrefix}, with the test's receive timeout. */
    public static ZMQ.Socket subscriber(ZContext context, String endpoint, String topicPrefix) {
        ZMQ.Socket sub = context.createSocket(SocketType.SUB);
        sub.setReceiveTimeOut(200);
        sub.subscribe(topicPrefix.getBytes(StandardCharsets.UTF_8));
        sub.connect(endpoint);
        return sub;
    }

    /** A DEALER socket on a replay {@code endpoint}, as vLLM's replay clients connect. */
    public static ZMQ.Socket replayClient(ZContext context, String endpoint) {
        ZMQ.Socket dealer = context.createSocket(SocketType.DEALER);
        dealer.setReceiveTimeOut(TIMEOUT_MILLIS);
        dealer.connect(endpoint);
        return dealer;
    }

    /** The next PUB message on {@code sub}, or null when none arrives within its receive timeout. */
    public static Frame receive(ZMQ.Socket sub) {
        ZMsg msg = ZMsg.recvMsg(sub);
        if (msg == null) {
            return null;
        }
        check(msg.size() == 3, "a PUB message is [topic, sequence, payload], got " + msg.size() + " frames");
        String topic = new String(msg.pop().getData(), StandardCharsets.UTF_8);
        long seq = sequence(msg.pop().getData());
        return new Frame(topic, seq, msg.pop().getData());
    }

    /**
     * Requests a replay from {@code start} and reads every replayed batch up to the end marker
     * {@code ["", "", -1, ""]}.
     */
    public static List<Frame> replay(ZMQ.Socket dealer, long start) {
        dealer.sendMore(new byte[0]);
        dealer.send(ByteBuffer.allocate(Long.BYTES).putLong(start).array());
        List<Frame> frames = new ArrayList<>();
        while (true) {
            ZMsg msg = ZMsg.recvMsg(dealer);
            check(msg != null, "the replay answers before the timeout");
            check(msg.size() == 4, "a replay message is [\"\", topic, sequence, payload], got " + msg.size()
                    + " frames");
            check(msg.pop().getData().length == 0, "the replay's first frame is the empty delimiter");
            String topic = new String(msg.pop().getData(), StandardCharsets.UTF_8);
            byte[] seqBytes = msg.pop().getData();
            byte[] payload = msg.pop().getData();
            if (sequence(seqBytes) == -1L) {
                check(topic.isEmpty(), "the end marker carries no topic, got " + topic);
                check(payload.length == 0, "the end marker carries no payload");
                return frames;
            }
            frames.add(new Frame(topic, sequence(seqBytes), payload));
        }
    }

    /**
     * Requests replays from {@code start} until one reaches sequence {@code last}, the publisher sending meanwhile.
     */
    public static List<Frame> replayThrough(ZMQ.Socket dealer, long start, long last) {
        long deadline = System.currentTimeMillis() + TIMEOUT_MILLIS;
        List<Frame> frames = replay(dealer, start);
        while (!reaches(frames, last) && System.currentTimeMillis() < deadline) {
            frames = replay(dealer, start);
        }
        check(reaches(frames, last), "the replay from " + start + " never reached sequence " + last + ", got "
                + frames.size() + " batches");
        return frames;
    }

    private static boolean reaches(List<Frame> frames, long last) {
        return !frames.isEmpty() && frames.get(frames.size() - 1).seq >= last;
    }

    /** The 8-byte big-endian sequence number of a frame. */
    public static long sequence(byte[] bytes) {
        check(bytes.length == Long.BYTES, "a sequence frame is 8 bytes, got " + bytes.length);
        return ByteBuffer.wrap(bytes).getLong();
    }

    /** The {@code KVEventBatch} array a payload holds. */
    public static ArrayValue decode(byte[] payload) {
        try {
            return MessagePack.newDefaultUnpacker(payload).unpackValue().asArrayValue();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** The events of a decoded batch. */
    public static List<MapValue> events(ArrayValue batch) {
        List<MapValue> out = new ArrayList<>();
        for (Value event : batch.get(1).asArrayValue()) {
            out.add(event.asMapValue());
        }
        return out;
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    /** The {@code field} of a decoded event, null when the event leaves it out. */
    public static Value field(MapValue event, String field) {
        return event.map().get(ValueFactory.newString(field));
    }
}
