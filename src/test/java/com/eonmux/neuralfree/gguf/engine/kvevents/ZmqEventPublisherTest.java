package com.eonmux.neuralfree.gguf.engine.kvevents;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.msgpack.value.ArrayValue;
import org.msgpack.value.MapValue;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.Frame;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.TIMEOUT_MILLIS;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.events;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.field;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.freePort;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.loopback;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.receive;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.replay;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.replayClient;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.replayThrough;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.subscriber;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The ZMQ transport against a subscriber speaking vLLM's protocol: the PUB socket sends
 * {@code [topic, sequence, payload]} with consecutive sequence numbers and vLLM's msgpack payload, stamped with the
 * publisher's data-parallel rank; the ROUTER replay socket answers a start sequence with the buffered batches from
 * there (the last {@code buffer_steps}) and the end marker; endpoints offset by the rank as vLLM's do; a closed
 * publisher refuses batches and frees its port.
 */
public class ZmqEventPublisherTest {

    /** The node IP of the ephemeral wildcard binds: loopback, so the tests' subscribers reach the publisher. */
    private String previousHostIp;

    @Before
    public void pinTheNodeIpToLoopback() {
        previousHostIp = System.getProperty(ZmqEventPublisherConfig.PROP_HOST_IP);
        System.setProperty(ZmqEventPublisherConfig.PROP_HOST_IP, "127.0.0.1");
    }

    @After
    public void restoreTheNodeIp() {
        if (previousHostIp == null) {
            System.clearProperty(ZmqEventPublisherConfig.PROP_HOST_IP);
        } else {
            System.setProperty(ZmqEventPublisherConfig.PROP_HOST_IP, previousHostIp);
        }
    }

    private static KvEventBatch removal(long hash, Integer rank) {
        return new KvEventBatch(1.0, Collections.singletonList(
                new KvCacheEvent.BlockRemoved(hashes(hash), KvCacheEvent.MEDIUM_GPU)), rank);
    }

    /** One block hash in the wire's integer form, an 8-byte digest. */
    private static byte[][] hashes(long... values) {
        byte[][] out = new byte[values.length][];
        for (int i = 0; i < values.length; i++) {
            byte[] bytes = new byte[8];
            for (int j = 0; j < 8; j++) {
                bytes[j] = (byte) (values[i] >>> ((7 - j) * 8));
            }
            out[i] = bytes;
        }
        return out;
    }

    private static ZmqEventPublisherConfig config(String replayEndpoint, int bufferSteps, String topic, int rank) {
        return new ZmqEventPublisherConfig("tcp://*:0", replayEndpoint, bufferSteps, 1_000, 100, topic, rank);
    }

    /** The removed hash of a decoded single-removal batch. */
    private static long removedHash(Frame frame) {
        MapValue event = events(frame.batch()).get(0);
        assertEquals("BlockRemoved", field(event, "type").asStringValue().asString());
        return field(event, "block_hashes").asArrayValue().get(0).asIntegerValue().toLong();
    }

    /** Publishes probes until the subscription has reached the publisher; answers the last sequence received. */
    private static long awaitSubscription(ZmqEventPublisher publisher, ZMQ.Socket sub) {
        long deadline = System.currentTimeMillis() + TIMEOUT_MILLIS;
        Frame probe = null;
        while (probe == null && System.currentTimeMillis() < deadline) {
            publisher.publish(removal(1L, null));
            probe = receive(sub);
        }
        assertNotNull("a probe reached the subscriber", probe);
        long last = probe.seq;
        for (Frame more = receive(sub); more != null; more = receive(sub)) {
            last = more.seq;
        }
        return last;
    }

    @Test
    public void aSubscriberDecodesEveryPublishedBatchInSequence() {
        try (ZContext context = new ZContext();
             ZmqEventPublisher publisher = new ZmqEventPublisher(config(null, 0, "kv@rank0", 0))) {
            ZMQ.Socket sub = subscriber(context, loopback(publisher.endpoint()), "kv@");
            long last = awaitSubscription(publisher, sub);

            KvEventBatch stored = new KvEventBatch(2.5, Arrays.asList(
                    new KvCacheEvent.BlockStored(hashes(-2L, 9L), hashes(5L)[0], new int[] {11, 12, 13, 14}, 2,
                            KvCacheEvent.MEDIUM_GPU, null, null, null, null,
                            Arrays.asList(Collections.singletonList(BlockExtraKey.salt("tenant")), null)),
                    KvCacheEvent.AllBlocksCleared.INSTANCE), null);
            publisher.publish(stored);
            publisher.publish(removal(42L, 7));

            Frame first = receive(sub);
            assertNotNull(first);
            assertEquals("kv@rank0", first.topic);
            assertEquals("the sequence follows the probes", last + 1, first.seq);
            assertArrayEquals("the payload is vLLM's msgpack of the rank-stamped batch",
                    KvEventBatchMsgpack.encode(stored.withRank(0)), first.payload);
            ArrayValue batch = first.batch();
            assertEquals(3, batch.size());
            assertEquals(2.5, batch.get(0).asFloatValue().toDouble(), 0.0);
            assertEquals("a batch without a rank carries the publisher's", 0, batch.get(2).asIntegerValue().toInt());
            MapValue head = events(batch).get(0);
            assertEquals("BlockStored", field(head, "type").asStringValue().asString());
            assertEquals("hashes are unsigned 64-bit", BigInteger.ONE.shiftLeft(64).subtract(BigInteger.valueOf(2)),
                    field(head, "block_hashes").asArrayValue().get(0).asIntegerValue().toBigInteger());
            assertEquals(5L, field(head, "parent_block_hash").asIntegerValue().toLong());
            assertEquals(4, field(head, "token_ids").asArrayValue().size());
            assertEquals(2, field(head, "block_size").asIntegerValue().toInt());
            assertEquals("GPU", field(head, "medium").asStringValue().asString());
            assertEquals("tenant", field(head, "extra_keys").asArrayValue().get(0).asArrayValue().get(0)
                    .asStringValue().asString());
            assertTrue(field(head, "extra_keys").asArrayValue().get(1).isNilValue());
            assertEquals("AllBlocksCleared", field(events(batch).get(1), "type").asStringValue().asString());

            Frame second = receive(sub);
            assertNotNull(second);
            assertEquals(first.seq + 1, second.seq);
            assertEquals(42L, removedHash(second));
            assertEquals("a batch's own rank stays", 7, second.batch().get(2).asIntegerValue().toInt());
        }
    }

    @Test
    public void theTopicFiltersSubscribers() {
        try (ZContext context = new ZContext();
             ZmqEventPublisher publisher = new ZmqEventPublisher(config(null, 0, "kv-events", 0))) {
            ZMQ.Socket matching = subscriber(context, loopback(publisher.endpoint()), "kv");
            ZMQ.Socket other = subscriber(context, loopback(publisher.endpoint()), "metrics");
            awaitSubscription(publisher, matching);
            assertNull("a subscriber of another topic hears nothing", receive(other));
        }
    }

    @Test
    public void theReplayServesTheBufferedBatchesFromTheRequestedSequence() {
        try (ZContext context = new ZContext();
             ZmqEventPublisher publisher = new ZmqEventPublisher(config("tcp://*:0", 3, "kv", 3))) {
            assertNotNull(publisher.replayEndpoint());
            for (long hash = 10; hash < 15; hash++) {
                publisher.publish(removal(hash, null));
            }
            ZMQ.Socket dealer = replayClient(context, loopback(publisher.replayEndpoint()));
            List<Frame> all = replayThrough(dealer, 0, 4);
            List<Long> seqs = new ArrayList<>();
            List<Long> hashes = new ArrayList<>();
            for (Frame frame : all) {
                assertEquals("the replay carries the topic, as the PUB does", "kv", frame.topic);
                assertEquals("the publisher stamps its rank", 3, frame.batch().get(2).asIntegerValue().toInt());
                seqs.add(frame.seq);
                hashes.add(removedHash(frame));
            }
            assertEquals("the buffer keeps the last buffer_steps batches", Arrays.asList(2L, 3L, 4L), seqs);
            assertEquals(Arrays.asList(12L, 13L, 14L), hashes);

            List<Frame> missed = replay(dealer, 4);
            assertEquals(1, missed.size());
            assertEquals(4L, missed.get(0).seq);
            assertTrue("a start past the newest batch answers only the end marker", replay(dealer, 99).isEmpty());

            dealer.sendMore(new byte[0]);
            dealer.sendMore(new byte[8]);
            dealer.send(new byte[8]);
            assertEquals("a malformed request is dropped and the next one served", 3, replay(dealer, 0).size());
        }
    }

    /**
     * An idle publisher answers a replay request at once instead of after its queue wait: ten replays one after
     * another take a fraction of the ten 100 ms waits a queue-bound thread would add, one per request.
     */
    @Test
    public void anIdlePublisherAnswersReplaysWithoutWaitingOutItsQueuePoll() {
        try (ZContext context = new ZContext();
             ZmqEventPublisher publisher = new ZmqEventPublisher(config("tcp://*:0", 3, "kv", 3))) {
            publisher.publish(removal(10, null));
            ZMQ.Socket dealer = replayClient(context, loopback(publisher.replayEndpoint()));
            replayThrough(dealer, 0, 0);
            long started = System.nanoTime();
            for (int i = 0; i < 10; i++) {
                assertEquals(1, replay(dealer, 0).size());
            }
            long millis = (System.nanoTime() - started) / 1_000_000;
            assertTrue("ten replays took " + millis + " ms", millis < 700);
        }
    }

    @Test
    public void noBufferStepsReplaysNothing() {
        try (ZContext context = new ZContext();
             ZmqEventPublisher publisher = new ZmqEventPublisher(config("tcp://*:0", 0, "", 0))) {
            publisher.publish(removal(1L, null));
            ZMQ.Socket dealer = replayClient(context, loopback(publisher.replayEndpoint()));
            assertTrue(replay(dealer, 0).isEmpty());
        }
    }

    @Test
    public void theEndpointsOffsetByTheDataParallelRank() {
        assertEquals("tcp://*:5559", ZmqEventPublisherConfig.offsetEndpointPort("tcp://*:5557", 2));
        assertEquals("tcp://10.0.0.1:5558", ZmqEventPublisherConfig.offsetEndpointPort("tcp://10.0.0.1:5557", 1));
        assertEquals("tcp://[::]:5560", ZmqEventPublisherConfig.offsetEndpointPort("tcp://[::]:5557", 3));
        assertEquals("rank 0 keeps the port", "tcp://*:5557",
                ZmqEventPublisherConfig.offsetEndpointPort("tcp://*:5557", 0));
        assertEquals("an ephemeral port stays ephemeral", "tcp://*:0",
                ZmqEventPublisherConfig.offsetEndpointPort("tcp://*:0", 5));
        assertNull(ZmqEventPublisherConfig.offsetEndpointPort(null, 2));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ZmqEventPublisherConfig.offsetEndpointPort("tcp://*:65535", 1)).getMessage().contains("65535"));

        // rank 1 binds the configured port plus one, so the port it binds is the one probed free
        int bound = freePort();
        try (ZmqEventPublisher ranked = new ZmqEventPublisher(new ZmqEventPublisherConfig("tcp://*:" + (bound - 1),
                null, 0, 0, 0, "", 1))) {
            assertEquals("tcp://*:" + bound, ranked.endpoint());
        }
    }

    @Test
    public void aConnectingPublisherReachesABoundSubscriber() {
        try (ZContext context = new ZContext()) {
            ZMQ.Socket sub = context.createSocket(SocketType.SUB);
            sub.setReceiveTimeOut(200);
            sub.subscribe(new byte[0]);
            sub.bind("tcp://127.0.0.1:0");
            String at = sub.getLastEndpoint();
            try (ZmqEventPublisher publisher = new ZmqEventPublisher(new ZmqEventPublisherConfig(at, null, 0,
                    1_000, 0, "", 0))) {
                assertEquals("a host endpoint connects", at, publisher.endpoint());
                awaitSubscription(publisher, sub);
            }
        }
    }

    @Test
    public void aClosedPublisherRefusesBatchesAndFreesItsPort() {
        int port = freePort();
        String endpoint = "tcp://*:" + port;
        ZmqEventPublisher publisher = new ZmqEventPublisher(new ZmqEventPublisherConfig(endpoint, null, 0, 0, 1, "",
                0));
        publisher.publish(removal(1L, null));
        publisher.close();
        publisher.close();
        assertTrue(assertThrows(IllegalStateException.class, () -> publisher.publish(removal(2L, null)))
                .getMessage().contains("closed"));
        try (ZmqEventPublisher again = new ZmqEventPublisher(new ZmqEventPublisherConfig(endpoint, null, 0, 0, 1,
                "", 0))) {
            assertEquals(endpoint, again.endpoint());
            assertTrue(assertThrows(IllegalStateException.class, () -> new ZmqEventPublisher(
                    new ZmqEventPublisherConfig(endpoint, null, 0, 0, 1, "", 0))).getMessage().contains(endpoint));
        }
    }

    @Test
    public void theConfigRefusesUnusableValuesByName() {
        assertTrue(assertThrows(IllegalArgumentException.class, () -> new ZmqEventPublisherConfig("ipc:///tmp/kv",
                null, 0, 0, 0, "", 0)).getMessage().contains("endpoint"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> new ZmqEventPublisherConfig("tcp://*:5557",
                "inproc://replay", 0, 0, 0, "", 0)).getMessage().contains("replay_endpoint"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> new ZmqEventPublisherConfig("tcp://*:x",
                null, 0, 0, 0, "", 0)).getMessage().contains("port"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> new ZmqEventPublisherConfig("tcp://*:70000",
                null, 0, 0, 0, "", 0)).getMessage().contains("65535"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> new ZmqEventPublisherConfig("tcp://*:5557",
                null, -1, 0, 0, "", 0)).getMessage().contains("buffer_steps"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> new ZmqEventPublisherConfig("tcp://*:5557",
                null, 0, -1, 0, "", 0)).getMessage().contains("hwm"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> new ZmqEventPublisherConfig("tcp://*:5557",
                null, 0, 0, -1, "", 0)).getMessage().contains("max_queue_size"));
        assertFalse(ZmqEventPublisherConfig.defaults().equals(ZmqEventPublisherConfig.defaults()
                .withDataParallelRank(1)));
        assertEquals("tcp://*:5557", ZmqEventPublisherConfig.defaults().endpoint());
        assertNull(ZmqEventPublisherConfig.defaults().replayEndpoint());
    }

    @Test
    public void anEphemeralWildcardBindsTheNodeIpAndReportsIt() {
        assertEquals("the pinned node IP", "127.0.0.1", ZmqEventPublisherConfig.nodeIp());
        try (ZmqEventPublisher publisher = new ZmqEventPublisher(config(null, 0, "", 0))) {
            assertTrue("the publisher reports the node IP, got " + publisher.endpoint(),
                    publisher.endpoint().matches("tcp://127\\.0\\.0\\.1:\\d+"));
            try (ZContext context = new ZContext()) {
                ZMQ.Socket sub = subscriber(context, publisher.endpoint(), "");
                publisher.publish(removal(7L, null));
                assertNotNull("a subscriber on the reported endpoint receives the batch",
                        receive(sub) != null);
                sub.close();
            }
        }
    }

    @Test
    public void anEphemeralWildcardOverAnIpv6NodeIpReportsTheBracketedUri() {
        System.setProperty(ZmqEventPublisherConfig.PROP_HOST_IP, "::1");
        try (ZmqEventPublisher publisher = new ZmqEventPublisher(config(null, 0, "", 0))) {
            assertTrue("the publisher reports the bracketed IPv6 URI, got " + publisher.endpoint(),
                    publisher.endpoint().matches("tcp://\\[::1\\]:\\d+"));
        }
    }

    @Test
    public void anEphemeralWildcardWithoutANodeIpRefusesByName() {
        System.setProperty(ZmqEventPublisherConfig.PROP_HOST_IP, "0.0.0.0");
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> new ZmqEventPublisher(config(null, 0, "", 0)));
        assertTrue(e.getMessage(), e.getMessage().contains(ZmqEventPublisherConfig.PROP_HOST_IP));
    }
}
