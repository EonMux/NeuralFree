package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The fault-tolerance control routes of vLLM's engine, served only with {@code --enable-fault-tolerance}:
 *
 * <ul>
 *   <li>{@code GET /v1/fault_tolerance/status} reports the pool as vLLM's DP engine status:
 *       {@code schema_version}, {@code total_engines} and one entry per replica with its data-parallel {@code id},
 *       {@code status} ({@code healthy}, {@code unhealthy} or {@code dead}), {@code fault_info} when it failed, and
 *       the state of the instruction that last touched it ({@code ft_state}, {@code last_ft_request_id},
 *       {@code ft_error}).</li>
 *   <li>{@code POST /v1/fault_tolerance/apply} takes {@code {"instruction", "params", "request_id"}}, answers 202,
 *       and runs the instruction off the request thread so the orchestrator can dispatch to every rank; the status
 *       route observes the outcome. {@code retry} recovers every unhealthy or dead replica (a failed model re-opens
 *       from its path, a stopped replica is replaced). {@code scale_down} removes the ranks
 *       {@code params.removed_dp_ranks} names from the pool; only failed ranks may be removed, and the deployment's
 *       global size keeps their slots.</li>
 * </ul>
 *
 * <p>Without the switch both routes answer 404 as an unregistered vLLM route does.</p>
 */
@Path("/")
public class FaultToleranceResource {

    /** The instructions vLLM's engine accepts. */
    static final List<String> INSTRUCTIONS = List.of("retry", "scale_down");

    @Inject
    EngineService service;

    @Inject
    ClusterSwitches switches;

    /** Test and manual wiring. */
    FaultToleranceResource(EngineService service, ClusterSwitches switches) {
        this.service = service;
        this.switches = switches;
    }

    public FaultToleranceResource() {
    }

    /** The pool's fault-tolerance status. */
    @GET
    @Path("/v1/fault_tolerance/status")
    @Produces(MediaType.APPLICATION_JSON)
    public Response status() {
        ClusterSwitches.require(switches.faultTolerance(), ClusterSwitches.FAULT_FLAG);
        List<EngineScaling.ReplicaHealth> engines = service.scaling().faultToleranceEngines();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("schema_version", 1);
        out.put("total_engines", engines.size());
        List<Object> rows = new java.util.ArrayList<>(engines.size());
        for (EngineScaling.ReplicaHealth engine : engines) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("id", engine.id());
            row.put("status", engine.status());
            if (engine.faultInfo() != null) {
                row.put("fault_info", engine.faultInfo());
            }
            if (engine.ftState() != null) {
                row.put("ft_state", engine.ftState());
            }
            if (engine.lastFtRequestId() != null) {
                row.put("last_ft_request_id", engine.lastFtRequestId());
            }
            if (engine.ftError() != null) {
                row.put("ft_error", engine.ftError());
            }
            rows.add(row);
        }
        out.put("engines", rows);
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * Validates and dispatches one fault-tolerance instruction; vLLM answers 202 before the recovery runs.
     */
    @POST
    @Path("/v1/fault_tolerance/apply")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response apply(String bodyText) {
        ClusterSwitches.require(switches.faultTolerance(), ClusterSwitches.FAULT_FLAG);
        JsonView body;
        try {
            body = ServerJson.parse(bodyText);
        } catch (JsonException e) {
            throw ServerJson.badRequest("Request body must be a JSON object.");
        }
        String instruction = requiredString(body, "instruction");
        Map<?, ?> params = params(body);
        String requestId = requiredString(body, "request_id");
        if (!INSTRUCTIONS.contains(instruction)) {
            throw ServerJson.badRequest("Invalid instruction: '" + instruction + "'.");
        }
        try {
            service.scaling().submitFaultTolerance(instruction, params, requestId);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("message", "Request accepted; poll /v1/fault_tolerance/status for updates.");
        out.put("request_id", requestId);
        return Response.status(Response.Status.ACCEPTED).entity(ServerJson.write(out)).build();
    }

    /** A required non-empty string field. */
    private static String requiredString(JsonView body, String field) {
        Object raw = body.raw(field);
        if (!(raw instanceof String text) || text.isEmpty()) {
            throw ServerJson.badRequest("'" + field + "' must be a non-empty string.");
        }
        return text;
    }

    /** The optional {@code params} object, empty when absent. */
    private static Map<?, ?> params(JsonView body) {
        Object raw = body.raw("params");
        if (raw == null) {
            return Map.of();
        }
        if (!(raw instanceof Map<?, ?> map)) {
            throw ServerJson.badRequest("'params' must be an object.");
        }
        return map;
    }
}
