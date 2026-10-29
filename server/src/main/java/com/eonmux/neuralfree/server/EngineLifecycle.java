package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import io.quarkus.runtime.ShutdownEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Singleton;

import java.io.IOException;

/**
 * The serving lifecycle wiring (PARITY plan P2): one {@link EngineService} per application, opened on first use
 * from the {@code neuralfree.server.model} property (the engine's remaining settings read their
 * {@code neuralfree.*} properties as everywhere else) and closed on shutdown. Without the property the service
 * starts unloaded: the health and metrics endpoints answer and every request refuses by name.
 */
@ApplicationScoped
public class EngineLifecycle {

    private volatile EngineService service;

    @Produces
    @Singleton
    synchronized EngineService engineService() {
        if (service == null) {
            try {
                service = new EngineService(System.getProperty("neuralfree.server.model"),
                        EngineConfig.fromSystemProperties());
            } catch (IOException e) {
                throw new IllegalStateException("the model at neuralfree.server.model does not open: "
                        + e.getMessage(), e);
            }
        }
        return service;
    }

    void onShutdown(@Observes ShutdownEvent event) {
        EngineService current = service;
        service = null;
        if (current != null) {
            current.close();
        }
    }
}
