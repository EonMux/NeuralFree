package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport;
import com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.Frame;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.msgpack.value.MapValue;
import org.msgpack.value.Value;
import com.eonmux.neuralfree.gguf.engine.kvevents.ZmqEventPublisherConfig;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.events;
import static com.eonmux.neuralfree.gguf.engine.kvevents.ZmqKvEventsTestSupport.field;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine publishing its prefix cache's KV cache events on vLLM's ZMQ transport, configured by the
 * {@code KVEventsConfig} JSON of {@code neuralfree.kv.events}: a replay client decodes the stores of a salted
 * prompt's block chain (tokens, block size, parent links, the salt as the first block's extra key, rank 0) and the
 * clear of a prefix-cache reset, in consecutive sequence order.
 */
public class BatchZmqKvEventsTest {

    private static final int PAGE = 4;

    /** The node IP of the ephemeral wildcard PUB bind: loopback, so the test's subscriber reaches it. */
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

    @Test
    public void aReplayClientDecodesTheEnginesStoresAndClears() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int replayPort = ZmqKvEventsTestSupport.freePort();
        String json = "{\"enable_kv_cache_events\": true, \"endpoint\": \"tcp://*:0\", \"replay_endpoint\":"
                + " \"tcp://*:" + replayPort + "\", \"topic\": \"kv\"}";
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).kvPaged(true)
                .kvPageSize(PAGE).parallel(1).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true)
                .kvEvents(json).build()).build();
        int[] prompt = {5, 9, 13, 17, 21, 25, 29, 33, 37, 41};
        try (ZContext context = new ZContext();
             GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence seq = engine.submit(prompt, GenerationRequest.builder().maxNewTokens(2).ignoreEos(true)
                    .sampler(SamplerConfig.greedyDecoding()).cacheSalt("tenant-z").build(), (TokenCallback) null);
            for (int step = 0; step < 2_000 && !seq.finished(); step++) {
                engine.pump();
            }
            assertEquals(BatchSequence.State.DONE, seq.state());
            engine.clearPrefixCache();

            ZMQ.Socket dealer = ZmqKvEventsTestSupport.replayClient(context, "tcp://127.0.0.1:" + replayPort);
            List<Frame> frames = untilCleared(dealer);
            List<MapValue> stored = new ArrayList<>();
            for (int i = 0; i < frames.size(); i++) {
                Frame frame = frames.get(i);
                assertEquals("kv", frame.topic);
                assertEquals("the sequence counts every batch", i, frame.seq);
                assertEquals("a single engine publishes rank 0", 0,
                        frame.batch().get(2).asIntegerValue().toInt());
                for (MapValue event : events(frame.batch())) {
                    if ("BlockStored".equals(type(event))) {
                        stored.add(event);
                    }
                }
            }
            assertFalse("the run stored blocks", stored.isEmpty());
            MapValue head = stored.get(0);
            assertTrue("the chain's first block has no parent", field(head, "parent_block_hash").isNilValue());
            assertEquals(PAGE, field(head, "block_size").asIntegerValue().toInt());
            List<Value> tokens = field(head, "token_ids").asArrayValue().list();
            for (int i = 0; i < PAGE; i++) {
                assertEquals(prompt[i], tokens.get(i).asIntegerValue().toInt());
            }
            assertEquals("the salt keys the first block", "tenant-z", field(head, "extra_keys").asArrayValue()
                    .get(0).asArrayValue().get(0).asStringValue().asString());
            for (int i = 1; i < stored.size(); i++) {
                List<Value> before = field(stored.get(i - 1), "block_hashes").asArrayValue().list();
                assertEquals("each store extends the previous one", before.get(before.size() - 1),
                        field(stored.get(i), "parent_block_hash"));
            }
            List<MapValue> last = events(frames.get(frames.size() - 1).batch());
            assertEquals("AllBlocksCleared", type(last.get(last.size() - 1)));
        }
    }

    /** The replay from sequence 0, re-requested until the publisher thread has sent the clear. */
    private static List<Frame> untilCleared(ZMQ.Socket dealer) {
        long deadline = System.currentTimeMillis() + ZmqKvEventsTestSupport.TIMEOUT_MILLIS;
        List<Frame> frames = ZmqKvEventsTestSupport.replay(dealer, 0);
        while (!cleared(frames) && System.currentTimeMillis() < deadline) {
            frames = ZmqKvEventsTestSupport.replay(dealer, 0);
        }
        assertTrue("the replay reached the clear, got " + frames.size() + " batches", cleared(frames));
        return frames;
    }

    private static boolean cleared(List<Frame> frames) {
        if (frames.isEmpty()) {
            return false;
        }
        List<MapValue> last = events(frames.get(frames.size() - 1).batch());
        return "AllBlocksCleared".equals(type(last.get(last.size() - 1)));
    }

    private static String type(MapValue event) {
        return field(event, "type").asStringValue().asString();
    }
}
