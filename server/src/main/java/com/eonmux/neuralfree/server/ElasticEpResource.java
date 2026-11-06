package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeoutException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The elastic data-parallel scaling of the replica pool, vLLM's {@code /scale_elastic_ep} and
 * {@code /is_scaling_elastic_ep}:
 *
 * <ul>
 *   <li>{@code POST /scale_elastic_ep} with {@code {"new_data_parallel_size": N, "drain_timeout": S}} resizes the
 *       pool to {@code N} replicas: a shrink stops routing to the removed replicas and waits up to {@code S}
 *       seconds (default 120) for their requests, then closes them; a grow opens the new replicas before routing to
 *       them. The answer is vLLM's {@code {"message": "Scaled to N data parallel engines"}}; an unusable body is a
 *       400 naming the field, a drain timeout a 408, and any other failure a 500.</li>
 *   <li>{@code POST /is_scaling_elastic_ep} reports the scale lifecycle: {@code is_scaling_elastic_ep} is true
 *       while the phase is {@code preparing} or {@code committing}, beside the phase, epoch, error and the
 *       requested data-parallel size, as vLLM's external elastic EP status.</li>
 * </ul>
 *
 * <p>The routes serve only with {@code --enable-elastic-ep} ({@link ClusterSwitches}); without it they answer 404.
 * A server that holds only some of the deployment's ranks leaves the topology to its external balancer and refuses
 * the scale by name.</p>
 */
@Path("/")
public class ElasticEpResource {

    private static final Logger LOGGER = Logger.getLogger(ElasticEpResource.class.getName());

    /** vLLM's default drain timeout for a scale. */
    static final int DEFAULT_DRAIN_TIMEOUT_SECONDS = 120;

    @Inject
    EngineService service;

    @Inject
    ClusterSwitches switches;

    /** Test and manual wiring. */
    ElasticEpResource(EngineService service, ClusterSwitches switches) {
        this.service = service;
        this.switches = switches;
    }

    public ElasticEpResource() {
    }

    /** Resizes the replica pool to the requested data-parallel size. */
    @POST
    @Path("/scale_elastic_ep")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response scale(String bodyText) {
        ClusterSwitches.require(switches.elasticEp(), ClusterSwitches.ELASTIC_FLAG);
        JsonView body;
        try {
            body = ServerJson.parse(bodyText);
        } catch (JsonException e) {
            throw ServerJson.badRequest("Request body must be a JSON object.");
        }
        int size = positiveInt(body, "new_data_parallel_size", true);
        int drainSeconds = body.raw("drain_timeout") == null ? DEFAULT_DRAIN_TIMEOUT_SECONDS
                : positiveInt(body, "drain_timeout", false);
        try {
            service.scaleElasticEp(size, drainSeconds * 1000L);
        } catch (TimeoutException e) {
            throw ServerJson.error(408, "Scale failed due to request drain timeout after " + drainSeconds
                    + " seconds", "TimeoutError");
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            LOGGER.log(Level.WARNING, "the elastic scale was interrupted", e);
            throw ServerJson.error(500, "Scale failed", "InternalServerError");
        } catch (RuntimeException e) {
            LOGGER.log(Level.WARNING, "the elastic scale failed", e);
            throw ServerJson.error(500, "Scale failed", "InternalServerError");
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("message", "Scaled to " + size + " data parallel engines");
        return Response.ok(ServerJson.write(out)).build();
    }

    /** The elastic-scale lifecycle (vLLM's external elastic EP status). */
    @POST
    @Path("/is_scaling_elastic_ep")
    @Produces(MediaType.APPLICATION_JSON)
    public Response status() {
        ClusterSwitches.require(switches.elasticEp(), ClusterSwitches.ELASTIC_FLAG);
        EngineService.ElasticStatus status = service.elasticStatus();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("is_scaling_elastic_ep", status.scaling());
        out.put("phase", status.phase());
        out.put("epoch", status.epoch());
        out.put("error", status.error());
        out.put("requested_data_parallel_size", status.requestedDataParallelSize());
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * The positive whole number {@code field} holds.
     *
     * @param required whether a missing field refuses instead of answering 0
     * @throws jakarta.ws.rs.WebApplicationException 400 naming the field when it is missing, not a whole number,
     *                                               or not positive
     */
    private static int positiveInt(JsonView body, String field, boolean required) {
        Object raw = body.raw(field);
        if (raw == null) {
            if (required) {
                throw ServerJson.badRequest(field + " is required");
            }
            return 0;
        }
        if (!(raw instanceof Number number) || number.doubleValue() != number.intValue()) {
            throw ServerJson.badRequest(field + " must be a positive integer");
        }
        long value = number.longValue();
        if (value <= 0 || value > Integer.MAX_VALUE) {
            throw ServerJson.badRequest(field + " must be a positive integer");
        }
        return (int) value;
    }
}
