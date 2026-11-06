package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import io.quarkus.runtime.ShutdownEvent;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.enterprise.inject.Produces;
import jakarta.inject.Singleton;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The serving lifecycle wiring: one {@link EngineService} per application, opened on first use
 * from the {@code neuralfree.server.model} property (the engine's remaining settings read their
 * {@code neuralfree.*} properties as everywhere else) and closed on shutdown. Without the property the service
 * starts unloaded: the health and metrics endpoints answer and every request refuses by name. Readers that bind to
 * the open service's shape — the per-replica meters, whose replicas exist only once the model opened — register an
 * open hook ({@link #onOpen}), which runs whenever a service opens.
 *
 * <p>In the router mode ({@code neuralfree.server.models.dir} or {@code .preset} set; the single-model property
 * refuses there) this class also produces the {@link ModelRouter}, whose entries open their own services; the
 * produced {@code EngineService} stays unloaded, since the routed requests serve on the entries' services and the
 * hooks run for every opened entry through {@link ModelRouter#onOpen}. A preset's {@code load_on_startup} models
 * load when the lifecycle first produces ({@link ModelRouter#loadStartupModels}).
 */
@ApplicationScoped
public class EngineLifecycle {

    private static final Logger LOGGER = Logger.getLogger(EngineLifecycle.class.getName());

    private volatile EngineService service;
    private volatile ModelRouter router;
    /** The resumable stream sessions every streaming response can record into. */
    private final StreamSessions streams = new StreamSessions();
    /** The open hooks, in registration order; guarded by this lifecycle's monitor. */
    private final List<Consumer<EngineService>> openHooks = new ArrayList<>();

    /** The resumable stream sessions ({@code GET}/{@code DELETE /v1/stream}, {@code POST /v1/streams/lookup}). */
    @Produces
    @Singleton
    StreamSessions streamSessions() {
        return streams;
    }

    /** The multi-model router; inactive outside the router mode ({@code active()} false). */
    @Produces
    @Singleton
    synchronized ModelRouter modelRouter() throws IOException {
        if (router == null) {
            router = ModelRouter.fromSystemProperties(EngineConfig.fromSystemProperties(),
                    ServerOptions.fromSystemProperties());
            if (router.active()) {
                for (Consumer<EngineService> hook : openHooks) {
                    router.onOpen(hook);
                }
                router.loadStartupModels();
            }
        }
        return router;
    }

    @Produces
    @Singleton
    synchronized EngineService engineService() {
        if (service == null) {
            boolean routerMode;
            try {
                routerMode = modelRouter().active();
            } catch (IOException e) {
                throw new IllegalStateException("the router sources do not read: " + e.getMessage(), e);
            }
            try {
                service = new EngineService(routerMode ? null : System.getProperty("neuralfree.server.model"),
                        EngineConfig.fromSystemProperties(), ServerOptions.fromSystemProperties(),
                        routerMode ? MediaOptions.none() : MediaOptions.fromSystemProperties(),
                        routerMode ? ServerLoras.NONE : ServerLoras.fromSystemProperties());
            } catch (IOException e) {
                throw new IllegalStateException("the model at neuralfree.server.model does not open: "
                        + e.getMessage(), e);
            }
            if (!routerMode) {
                // the router mode's services open through the router, whose onOpen runs these hooks for them
                for (Consumer<EngineService> hook : openHooks) {
                    runHook(hook, service);
                }
            }
        }
        return service;
    }

    /**
     * Runs {@code hook} with the open service now if one is open ({@link #current}), and again each time a service
     * opens; in the router mode the hook runs for every entry's service instead, through {@link ModelRouter#onOpen}.
     * A hook that fails is logged and leaves the service serving.
     */
    synchronized void onOpen(Consumer<EngineService> hook) {
        openHooks.add(hook);
        ModelRouter active = router;
        if (active != null && active.active()) {
            active.onOpen(hook);
            return;
        }
        EngineService open = current();
        if (open != null) {
            runHook(hook, open);
        }
    }

    private static void runHook(Consumer<EngineService> hook, EngineService open) {
        try {
            hook.accept(open);
        } catch (RuntimeException e) {
            LOGGER.log(Level.WARNING, "an open hook of the engine service failed", e);
        }
    }

    /** The service if one was produced already, else null: readers that must not load the model use it. */
    EngineService current() {
        return service;
    }

    void onShutdown(@Observes ShutdownEvent event) {
        streams.close();
        EngineService current = service;
        service = null;
        if (current != null) {
            current.close();
        }
        ModelRouter active = router;
        if (active != null) {
            active.close();
        }
    }
}
