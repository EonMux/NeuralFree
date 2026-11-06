package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The per-request LoRA adapters of the replica pool: loads and unloads apply to every replica's
 * {@code LoraRegistry} on its pump thread (a replica a full sleep released picks them up when it wakes), a change
 * that fails on one replica rolls back on the others, and the {@link LoraCatalog} records the result. Requests
 * select adapters through {@link #model} and {@link #apply} (a pooling request through {@link #selection}).
 *
 * <p>Thread-safe: the changes are serialized; request resolution reads the catalog's snapshot.</p>
 */
final class LoraAdmin {

    private static final Logger LOGGER = Logger.getLogger(LoraAdmin.class.getName());

    private final Supplier<List<EngineReplica>> replicas;
    private final ServedModels served;
    private final LoraCatalog catalog;
    private final boolean runtimeUpdating;

    LoraAdmin(Supplier<List<EngineReplica>> replicas, ServedModels served, LoraCatalog catalog,
              boolean runtimeUpdating) {
        this.replicas = Objects.requireNonNull(replicas, "replicas");
        this.served = Objects.requireNonNull(served, "served");
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.runtimeUpdating = runtimeUpdating;
    }

    /** The served adapters in load order. */
    List<LoraCatalog.Adapter> list() {
        return catalog.list();
    }

    /** Whether the runtime load and unload endpoints serve ({@link ServerLoras#runtimeUpdating}). */
    boolean runtimeUpdating() {
        return runtimeUpdating;
    }

    /**
     * Refuses a runtime adapter change while the switch is off with the 404 an unregistered vLLM route answers; the
     * HTTP load and unload routes and the gRPC {@code LoadLora} and {@code UnloadLora} calls share this policy.
     */
    void requireRuntimeUpdating() {
        if (!runtimeUpdating) {
            throw ServerJson.error(404, "runtime LoRA adapter updating is off; start the server with -D"
                    + ServerLoras.RUNTIME_PROPERTY + "=true (vLLM's " + ServerLoras.RUNTIME_ENV + "=True)",
                    "NotFoundError");
        }
    }

    /**
     * Loads the adapter at {@code path} as {@code name} on every replica (vLLM's {@code /v1/load_lora_adapter});
     * {@code inPlace} replaces an adapter of that name, keeping its global scale.
     *
     * @throws NoSuchFileException           when {@code path} does not exist
     * @throws IOException                   when the adapter does not read or validate
     * @throws IllegalArgumentException      for a blank or reserved name, a name a served model or a startup adapter
     *                                       takes, or a loaded name without {@code inPlace}
     * @throws IllegalStateException         when the adapter limit is reached by adapters in use, or the replaced
     *                                       adapter serves running requests
     * @throws UnsupportedOperationException when the model runs no per-request adapters
     */
    synchronized void load(String name, String path, boolean inPlace) throws IOException, InterruptedException {
        ServerLoras.requireName(name);
        if (path == null || path.isBlank()) {
            throw new IllegalArgumentException("Both 'lora_name' and 'lora_path' must be provided.");
        }
        if (served.names().contains(name)) {
            throw new IllegalArgumentException("The lora adapter '" + name + "' conflicts with a served base model.");
        }
        if (!Files.exists(Path.of(path))) {
            throw new NoSuchFileException(path, null, "Loading lora " + name + " failed: No adapter found for "
                    + path);
        }
        LoraCatalog.Adapter previous = catalog.find(name);
        if (previous != null && previous.startup()) {
            throw new IllegalArgumentException("The lora adapter '" + name + "' is a startup adapter (--lora "
                    + previous.path() + "), which stays loaded; choose another name.");
        }
        if (previous != null && !inPlace) {
            throw new IllegalArgumentException("The lora adapter '" + name + "' has already been loaded. If you"
                    + " want to load the adapter in place, set 'load_inplace' to True.");
        }
        catalog.put(new LoraCatalog.Adapter(name, path, previous == null ? 0.0f : previous.scale()));
        List<EngineReplica> pool = replicas.get();
        List<EngineReplica> changed = new ArrayList<>();
        try {
            for (EngineReplica replica : pool) {
                replica.adapterOp("the LoRA adapter load", r -> ReplicaLoras.ensureLoaded(r, name, path));
                changed.add(replica);
            }
        } catch (IOException | RuntimeException e) {
            if (previous == null) {
                catalog.remove(name);
                rollback(changed, e, r -> ReplicaLoras.ensureUnloaded(r, name));
            } else {
                catalog.put(previous);
                rollback(changed, e, r -> ReplicaLoras.ensureLoaded(r, name, previous.path()));
            }
            throw e;
        }
        if (!pool.isEmpty()) {
            catalog.metadata(name, pool.get(0).adapterMetadata(name));
        }
        LOGGER.info("loaded LoRA adapter '" + name + "' from " + path);
    }

    /**
     * Unloads the adapter {@code name} from every replica (vLLM's {@code /v1/unload_lora_adapter}).
     *
     * @throws NoSuchElementException   when no adapter is named so
     * @throws IllegalArgumentException for a startup adapter ({@code --lora}), which stays loaded
     * @throws IllegalStateException    when running requests use the adapter
     */
    synchronized void unload(String name) throws InterruptedException {
        LoraCatalog.Adapter adapter = catalog.find(name);
        if (adapter == null) {
            throw new NoSuchElementException("The lora adapter '" + name + "' cannot be found.");
        }
        if (adapter.startup()) {
            throw new IllegalArgumentException("The lora adapter '" + name + "' is a startup adapter (--lora "
                    + adapter.path() + "), which stays loaded; set its scale to 0 (POST /lora-adapters) instead.");
        }
        int at = catalog.remove(name);
        List<EngineReplica> changed = new ArrayList<>();
        try {
            for (EngineReplica replica : replicas.get()) {
                replica.adapterOp("the LoRA adapter unload", r -> ReplicaLoras.ensureUnloaded(r, name));
                changed.add(replica);
            }
        } catch (IOException | RuntimeException e) {
            catalog.restore(at, adapter);
            rollback(changed, e, r -> ReplicaLoras.ensureLoaded(r, name, adapter.path()));
            if (e instanceof IOException io) {
                throw new IllegalStateException(io.getMessage(), io);
            }
            throw (RuntimeException) e;
        }
        LOGGER.info("unloaded LoRA adapter '" + name + "'");
    }

    /**
     * Sets the global scales (llama.cpp's {@code POST /lora-adapters}): the listed ids take their scale, the
     * others 0.
     *
     * @throws IllegalArgumentException naming an invalid entry
     */
    synchronized void setScales(Object body) {
        catalog.setScales(body);
    }

    /**
     * The model name a generation response to {@code body} carries: an adapter its {@code model} names (vLLM's
     * adapter-as-model), else the served-model resolution ({@link ServedModels#resolve}).
     */
    String model(JsonView body) {
        Object raw = body.raw("model");
        if (raw instanceof String requested && catalog.contains(requested)) {
            return requested;
        }
        return served.resolve(body);
    }

    /**
     * {@code request} running the adapters {@code body} selects ({@link LoraCatalog#select}) given its resolved
     * {@code model}; unchanged when it runs the base model.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming the field of an invalid selection
     */
    GenerationRequest apply(JsonView body, String model, GenerationRequest request) {
        LoraSelection selection = selection(body, model);
        return selection == null ? request : request.toBuilder().lora(selection).build();
    }

    /**
     * The adapters {@code body} selects ({@link LoraCatalog#select}) given its resolved {@code model}, the form a
     * pooling request carries; null runs slot 0.
     *
     * @throws jakarta.ws.rs.WebApplicationException 400 naming the field of an invalid selection
     */
    LoraSelection selection(JsonView body, String model) {
        try {
            return catalog.select(body, model);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** Re-applies the catalog's state to the replicas a failed change already changed, logging what fails. */
    private static void rollback(List<EngineReplica> changed, Exception cause, EngineReplica.RegistryCall undo) {
        for (EngineReplica replica : changed) {
            try {
                replica.adapterOp("the LoRA adapter rollback", undo);
            } catch (IOException | RuntimeException e) {
                cause.addSuppressed(e);
                LOGGER.log(Level.WARNING, "the LoRA adapter rollback failed on replica " + replica.device(), e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                cause.addSuppressed(e);
                return;
            }
        }
    }
}
