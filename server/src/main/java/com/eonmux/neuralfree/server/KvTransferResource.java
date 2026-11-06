package com.eonmux.neuralfree.server;

import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The disaggregated prefill/decode status: the instance's {@code kv_role}, engine id, connector and counters. The
 * transfer itself runs through the ordinary generation endpoints' {@code kv_transfer_params}, as vLLM's does; this
 * route only reports what the connector holds.
 */
@Path("/")
public class KvTransferResource {

    @Inject
    EngineService service;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public KvTransferResource(EngineService service) {
        this.service = service;
    }

    public KvTransferResource() {
    }

    /** The connector's role, address and counters; {@code enabled} false when no transfer is configured. */
    @GET
    @Path("/v1/kv_transfer/status")
    @Produces(MediaType.APPLICATION_JSON)
    public Response status() {
        Map<String, Object> out = new LinkedHashMap<>();
        KvTransferService transfer = service.kvTransfer();
        out.put("enabled", transfer != null);
        if (transfer != null) {
            out.putAll(transfer.status());
        }
        return Response.ok(ServerJson.write(out)).build();
    }
}
