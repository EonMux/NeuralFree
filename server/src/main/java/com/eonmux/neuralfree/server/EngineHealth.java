package com.eonmux.neuralfree.server;

import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.Readiness;

/**
 * The Kubernetes readiness probe over the model ({@code /q/health/ready}): up while the service holds an open
 * model or sleeps (a request wakes it), down before a model is open — the same fact {@code /ready} answers
 * ({@link EngineService#ready}), in MicroProfile Health form. In router mode the probe is up while the router
 * serves, as the router's own {@code /health} and {@code /ready} are. Liveness stays the extension's default: the
 * process answering is that fact.
 */
@Readiness
@Singleton
public class EngineHealth implements HealthCheck {

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where the check reads {@code service} alone. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: the check over an explicit service. */
    EngineHealth(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    EngineHealth(EngineService service, ModelRouter router) {
        this.service = service;
        this.router = router;
    }

    public EngineHealth() {
    }

    @Override
    public HealthCheckResponse call() {
        boolean ready = (router != null && router.active()) || service.ready();
        return HealthCheckResponse.named("engine-model").status(ready).build();
    }
}
