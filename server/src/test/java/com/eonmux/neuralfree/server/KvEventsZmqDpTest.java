package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.ZmqEventPublisherConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport;
import com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.Frame;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.msgpack.value.MapValue;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.events;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.field;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * vLLM's ZMQ KV cache events across a data-parallel pool: with the {@code --kv-events-config} JSON on
 * {@code neuralfree.kv.events}, each replica publishes from its own publisher, its replay port offset by its
 * global rank as vLLM's per-rank publishers are, and every batch it serves carries that rank.
 */
class KvEventsZmqDpTest {

    private static final TokenCallback IGNORED = (token, text) -> { };

    /** The node IP of the replicas' ephemeral wildcard PUB binds: loopback, so the test's subscriber reaches them. */
    private String previousHostIp;

    @BeforeEach
    void pinTheNodeIpToLoopback() {
        previousHostIp = System.getProperty(ZmqEventPublisherConfig.PROP_HOST_IP);
        System.setProperty(ZmqEventPublisherConfig.PROP_HOST_IP, "127.0.0.1");
    }

    @AfterEach
    void restoreTheNodeIp() {
        if (previousHostIp == null) {
            System.clearProperty(ZmqEventPublisherConfig.PROP_HOST_IP);
        } else {
            System.setProperty(ZmqEventPublisherConfig.PROP_HOST_IP, previousHostIp);
        }
    }

    @Test
    void everyReplicaPublishesOnItsOwnPortWithItsRank() throws IOException, InterruptedException {
        Path model = SyntheticModels.llama().writeTempFile();
        int port = freePortPair();
        String json = "{\"enable_kv_cache_events\": true, \"endpoint\": \"tcp://*:0\", \"replay_endpoint\":"
                + " \"tcp://*:" + port + "\", \"topic\": \"kv\"}";
        EngineConfig config = ServerTestSupport.config().memory(MemoryConfig.builder().nCtx(512).ubatch(8)
                .kvPaged(true).kvPageSize(4).parallel(2).prefixReuse(true).typeK(KvDType.F32).typeV(KvDType.F32)
                .kvEvents(json).build()).build();
        EngineService pool = new EngineService(model.toString(), config,
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
        try (ZContext context = new ZContext()) {
            GenerationRequest busy = GenerationRequest.builder().maxNewTokens(200).ignoreEos(true)
                    .sampler(SamplerConfig.greedyDecoding()).build();
            EngineService.Submission first = pool.submit(pool.encode("the first replica's prompt", true, false),
                    busy, IGNORED);
            awaitRunning(pool, 0);
            EngineService.Submission second = pool.submit(pool.encode("a prompt for the other replica", true,
                    false), busy, IGNORED);
            first.await(60, TimeUnit.SECONDS);
            second.await(60, TimeUnit.SECONDS);
            for (int rank = 0; rank < 2; rank++) {
                ZMQ.Socket dealer = ZmqKvEventsTestSupport.replayClient(context, "tcp://127.0.0.1:" + (port + rank));
                List<Frame> frames = storedReplay(dealer);
                for (Frame frame : frames) {
                    assertEquals(rank, frame.batch().get(2).asIntegerValue().toInt(),
                            "replica " + rank + " stamps its rank");
                    assertEquals("kv", frame.topic);
                }
            }
        } finally {
            pool.close();
        }
    }

    /** The replay from sequence 0, re-requested until it holds a store. */
    private static List<Frame> storedReplay(ZMQ.Socket dealer) {
        long deadline = System.currentTimeMillis() + ZmqKvEventsTestSupport.TIMEOUT_MILLIS;
        while (true) {
            List<Frame> frames = ZmqKvEventsTestSupport.replay(dealer, 0);
            for (Frame frame : frames) {
                for (MapValue event : events(frame.batch())) {
                    if ("BlockStored".equals(field(event, "type").asStringValue().asString())) {
                        return frames;
                    }
                }
            }
            if (System.currentTimeMillis() > deadline) {
                fail("the replay holds no store, got " + frames.size() + " batches");
            }
        }
    }

    private static void awaitRunning(EngineService pool, int replica) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (pool.replicaStats().get(replica).running() == 0) {
            assertTrue(System.nanoTime() < deadline, "replica " + replica + " never started");
            Thread.sleep(2);
        }
    }

    /** A port {@code p} with {@code p} and {@code p + 1} both free, for the two ranks' replay sockets. */
    private static int freePortPair() throws IOException {
        for (int attempt = 0; attempt < 20; attempt++) {
            int port = ZmqKvEventsTestSupport.freePort();
            if (port < 65_535) {
                try (ServerSocket next = new ServerSocket(port + 1)) {
                    return port;
                } catch (IOException taken) {
                    // the neighbour port is taken: pick another pair
                }
            }
        }
        throw new IOException("no two consecutive free ports");
    }
}
