package com.eonmux.neuralfree.server;

import io.quarkus.websockets.next.HttpUpgradeCheck;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Singleton;
import jakarta.ws.rs.core.HttpHeaders;

import java.util.List;
import java.util.Map;

/**
 * Applies the HTTP API key to WebSocket upgrades: a JAX-RS filter ({@link ApiKeyFilter}) never sees an upgrade
 * request, so without this check the realtime endpoint would answer clients the keyed HTTP routes refuse. The
 * same key forms are accepted ({@code Authorization: Bearer <key>} and {@code x-api-key}), and the upgrade is
 * refused with 401 when keys are configured and the presented one is not.
 */
@Singleton
public class ApiKeyUpgradeCheck implements HttpUpgradeCheck {

    private final ApiKeys keys;

    public ApiKeyUpgradeCheck() {
        this(ApiKeys.from(System.getenv(), System.getProperties()));
    }

    ApiKeyUpgradeCheck(ApiKeys keys) {
        this.keys = keys;
    }

    @Override
    public Uni<CheckResult> perform(HttpUpgradeContext context) {
        if (!keys.enabled()) {
            return Uni.createFrom().item(CheckResult.permitUpgradeSync());
        }
        String presented = ApiKeys.presented(context.httpRequest().headers().get(HttpHeaders.AUTHORIZATION),
                context.httpRequest().headers().get("x-api-key"));
        if (keys.accepts(presented)) {
            return Uni.createFrom().item(CheckResult.permitUpgradeSync());
        }
        return Uni.createFrom().item(CheckResult.rejectUpgradeSync(401,
                Map.of(HttpHeaders.WWW_AUTHENTICATE, List.of("Bearer"))));
    }
}
