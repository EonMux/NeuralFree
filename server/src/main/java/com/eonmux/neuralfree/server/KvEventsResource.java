package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventsConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.ZmqEventPublisherConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code GET /kv_event_sources}: the KV cache event publisher of every ready replica keyed by its data-parallel
 * rank, vLLM's instrumentator route over {@code EngineClient.get_kv_event_sources}. The map is empty while the
 * service is unloaded or when {@code neuralfree.kv.events} publishes nothing; a ZMQ publisher reports its resolved
 * transport fields with the rank's port offset, as vLLM's per-rank publisher does. A publisher this engine adds
 * beyond vLLM's {@code zmq} (its {@code log} and class-name publishers) reports its name alone, since it has no ZMQ
 * transport.
 */
@Path("/")
public class KvEventsResource {

    @Inject
    EngineService service;

    /** Test and manual wiring. */
    KvEventsResource(EngineService service) {
        this.service = service;
    }

    public KvEventsResource() {
    }

    /** The publisher configuration per data-parallel rank. */
    @GET
    @Path("/kv_event_sources")
    @Produces(MediaType.APPLICATION_JSON)
    public Response sources() {
        List<EngineReplica> replicas = service.replicas();
        Map<String, Object> out = new LinkedHashMap<>();
        Source source = Source.resolve(service.config().memory().kvEvents());
        if (source != null) {
            for (EngineReplica replica : replicas) {
                out.put(Integer.toString(replica.rank()), source.describe(replica.rank()));
            }
        }
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * The publisher {@code neuralfree.kv.events} selects: a ZMQ transport, a named publisher, or null when nothing
     * publishes. Resolving never opens a socket or instantiates a publisher class, unlike
     * {@link KvEventPublishers#named(String, int)}, which the engine calls when it builds.
     */
    private record Source(String publisher, ZmqEventPublisherConfig zmq) {

        /** The source {@code spec} names, or null when it publishes nothing. */
        private static Source resolve(String spec) {
            String name = spec == null ? "" : spec.trim();
            if (name.isEmpty() || name.equalsIgnoreCase("null") || name.equalsIgnoreCase("none")) {
                return null;
            }
            if (name.startsWith("{")) {
                KvEventsConfig config;
                try {
                    config = KvEventsConfig.parse(name, MemoryConfig.PROP_KV_EVENTS);
                } catch (IllegalArgumentException e) {
                    throw ServerJson.error(500, e.getMessage(), "InternalServerError");
                }
                if (!config.enabled()) {
                    return null;
                }
                return config.zmq() != null ? new Source(KvEventsConfig.ZMQ, config.zmq())
                        : new Source(config.publisher(), null);
            }
            if (name.equalsIgnoreCase(KvEventsConfig.ZMQ)) {
                return new Source(KvEventsConfig.ZMQ, ZmqEventPublisherConfig.defaults());
            }
            return new Source(name, null);
        }

        /** One rank's configuration, vLLM's {@code KVEventsConfig} fields. */
        private Map<String, Object> describe(int rank) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("enable_kv_cache_events", true);
            out.put("publisher", publisher);
            if (zmq != null) {
                out.put("endpoint", ZmqEventPublisherConfig.offsetEndpointPort(zmq.endpoint(), rank));
                out.put("replay_endpoint", ZmqEventPublisherConfig.offsetEndpointPort(zmq.replayEndpoint(), rank));
                out.put("buffer_steps", zmq.bufferSteps());
                out.put("hwm", zmq.hwm());
                out.put("max_queue_size", zmq.maxQueueSize());
                out.put("topic", zmq.topic());
            }
            return out;
        }
    }
}
