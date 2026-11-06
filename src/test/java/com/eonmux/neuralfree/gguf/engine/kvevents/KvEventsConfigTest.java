package com.eonmux.neuralfree.gguf.engine.kvevents;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * vLLM's {@code KVEventsConfig} JSON: an enabled config without a publisher resolves to {@code zmq} with vLLM's
 * defaults, every transport field reads, a disabled or {@code null} config publishes nothing, and unusable values
 * refuse naming the flag and the field.
 */
public class KvEventsConfigTest {

    private static final String FLAG = "--kv-events-config";

    private static String refusal(String json) {
        return assertThrows(IllegalArgumentException.class, () -> KvEventsConfig.parse(json, FLAG)).getMessage();
    }

    @Test
    public void anEnabledConfigDefaultsToVllmsZmqTransport() {
        KvEventsConfig config = KvEventsConfig.parse("{\"enable_kv_cache_events\": true}", FLAG);
        assertTrue(config.enabled());
        assertEquals("zmq", config.publisher());
        assertEquals(ZmqEventPublisherConfig.defaults(), config.zmq());
        assertFalse(config.hasTransportFields());
        assertEquals("a JSON null publisher is vLLM's unset one", "zmq", KvEventsConfig.parse(
                "{\"enable_kv_cache_events\": true, \"publisher\": null}", FLAG).publisher());
    }

    @Test
    public void everyTransportFieldReads() {
        KvEventsConfig config = KvEventsConfig.parse("{\"enable_kv_cache_events\": true, \"publisher\": \"zmq\","
                + " \"endpoint\": \"tcp://*:6000\", \"replay_endpoint\": \"tcp://*:6001\", \"buffer_steps\": 50,"
                + " \"hwm\": 0, \"max_queue_size\": 0, \"topic\": \"kv@a\"}", FLAG);
        assertTrue(config.hasTransportFields());
        assertEquals(new ZmqEventPublisherConfig("tcp://*:6000", "tcp://*:6001", 50, 0, 0, "kv@a", 0),
                config.zmq());
    }

    @Test
    public void aDisabledOrNullConfigPublishesNothing() {
        assertFalse(KvEventsConfig.parse("{}", FLAG).enabled());
        assertFalse(KvEventsConfig.parse("{\"enable_kv_cache_events\": false, \"publisher\": \"zmq\","
                + " \"endpoint\": \"tcp://*:6000\"}", FLAG).enabled());
        KvEventsConfig nulled = KvEventsConfig.parse("{\"enable_kv_cache_events\": true, \"publisher\": \"null\","
                + " \"hwm\": 5}", FLAG);
        assertFalse(nulled.enabled());
        assertNull(nulled.publisher());
        assertNull(nulled.zmq());
    }

    @Test
    public void theEngineExtensionsCarryNoTransport() {
        KvEventsConfig log = KvEventsConfig.parse("{\"enable_kv_cache_events\": true, \"publisher\": \"log\"}", FLAG);
        assertEquals("log", log.publisher());
        assertNull(log.zmq());
        assertEquals("com.example.Sink", KvEventsConfig.parse(
                "{\"enable_kv_cache_events\": true, \"publisher\": \" com.example.Sink \"}", FLAG).publisher());
        assertTrue(refusal("{\"enable_kv_cache_events\": true, \"publisher\": \"log\", \"topic\": \"kv\"}")
                .contains("topic"));
    }

    @Test
    public void unusableValuesRefuseByFlagAndField() {
        String on = "{\"enable_kv_cache_events\": true, ";
        assertTrue(refusal("not json").contains(FLAG));
        assertTrue(refusal("[1]").contains("JSON object"));
        assertTrue(refusal("{\"publish\": \"zmq\"}").contains("publish"));
        assertTrue(refusal("{\"enable_kv_cache_events\": \"yes\"}").contains("enable_kv_cache_events"));
        assertTrue(refusal(on + "\"publisher\": 3}").contains("publisher"));
        String[][] bad = {
                {"endpoint", "\"ipc:///tmp/kv\""}, {"endpoint", "\"tcp://*\""}, {"endpoint", "5557"},
                {"replay_endpoint", "\"inproc://replay\""}, {"buffer_steps", "-1"}, {"buffer_steps", "1.5"},
                {"hwm", "\"many\""}, {"hwm", "4294967296"}, {"max_queue_size", "-3"}, {"topic", "7"},
        };
        for (String[] field : bad) {
            String message = refusal(on + "\"" + field[0] + "\": " + field[1] + "}");
            assertTrue(field[0] + "=" + field[1] + ": " + message,
                    message.contains(FLAG) && message.contains(field[0]));
        }
    }
}
