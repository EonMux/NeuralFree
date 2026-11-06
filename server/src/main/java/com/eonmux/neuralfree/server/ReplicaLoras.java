package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.lora.LoraHandle;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The per-request LoRA adapters of one {@link EngineReplica}, and the model open that binds them: opening the model
 * checks the configured parsers, records the startup adapters in the service's {@link LoraCatalog}, loads the
 * catalog's adapters and records their metadata; the runtime load, unload and metadata calls run on the replica's
 * pump thread between engine steps. The replica owns the pump thread and the open model.
 */
final class ReplicaLoras {

    /** A call on the open model, run on the pump thread between engine steps. */
    @FunctionalInterface
    private interface OpenModelCall<T> {
        T run(EngineModel open) throws IOException;
    }

    private final EngineReplica owner;

    ReplicaLoras(EngineReplica owner) {
        this.owner = owner;
    }

    /**
     * Opens the model, checks that the configured parsers can serve it, records its startup adapters first in the
     * catalog and loads the catalog's per-request adapters into it; a failing load closes it again.
     */
    EngineModel open() throws IOException {
        EngineModel open = LLMModelFactory.open(owner.modelPath(), owner.config());
        try {
            checkParsers(open);
            LoraRegistry registry = open.loraRegistry();
            if (registry != null) {
                List<LoraCatalog.Adapter> startup = new ArrayList<>();
                for (LoraHandle h : registry.list()) {
                    if (h.startup()) {
                        startup.add(LoraCatalog.Adapter.startup(h, open.tokenizer()));
                    }
                }
                owner.adapters().startup(startup);
            }
            for (LoraCatalog.Adapter a : owner.adapters().list()) {
                if (!a.startup()) {
                    ensureLoaded(registryOf(open), a.name(), a.path());
                    owner.adapters().metadata(a.name(), metadataOf(open, a.name()));
                }
            }
        } catch (IOException | RuntimeException e) {
            try {
                open.close();
            } catch (IOException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
        return open;
    }

    /**
     * Fails the load when a configured reasoning or tool parser cannot serve the model.
     *
     * @throws IllegalArgumentException naming the parser and what the model lacks
     */
    private void checkParsers(EngineModel open) {
        GGUFTokenizer tokenizer = open.tokenizer();
        if (!ReasoningParsers.isAuto(owner.config().reasoningParser())) {
            ReasoningParsers.forModel(owner.config().reasoningParser(), tokenizer);
        }
        ToolRouting.checkFormat(owner.config().toolParser(), tokenizer);
    }

    /**
     * Runs {@code call} on the model's adapter registry on the pump thread, between engine steps, and waits for it.
     * A replica whose model a full sleep released skips it: the wake loads the catalog's adapters.
     *
     * @throws IOException                   when the call fails reading an adapter
     * @throws UnsupportedOperationException when the model runs no per-request adapters
     * @throws IllegalStateException         when the pump did not answer within the request budget, or the call
     *                                       refused (an adapter in use, the adapter limit)
     */
    void adapterOp(String what, EngineReplica.RegistryCall call) throws IOException, InterruptedException {
        modelOp(what, open -> {
            call.run(registryOf(open));
            return null;
        });
    }

    /**
     * The PEFT metadata of the resident adapter {@code name}, read on the pump thread;
     * {@link LoraCatalog.Metadata#NONE} while a full sleep released the model.
     */
    LoraCatalog.Metadata adapterMetadata(String name) throws IOException, InterruptedException {
        LoraCatalog.Metadata read = modelOp("the LoRA adapter metadata", open -> metadataOf(open, name));
        return read == null ? LoraCatalog.Metadata.NONE : read;
    }

    /**
     * The media-session adapter key of {@code selection}, resolved on the pump thread (the registry resolves its
     * slots there): null when the selection names no adapter beyond the startup set, else the key whose identity the
     * projector's media caches fold into their item keys and whose parts the vision encoder's session applies.
     * A replica whose model a full sleep released resolves nothing: the encode then refuses on the release.
     *
     * @throws IOException                   when the call fails reading an adapter
     * @throws UnsupportedOperationException when the model runs no per-request adapters
     * @throws IllegalStateException         when the pump did not answer within the request budget
     */
    LoraSlotKey resolveMediaLoras(LoraSelection selection) throws IOException, InterruptedException {
        if (selection == null) {
            return null;
        }
        return modelOp("resolving the request's LoRA adapters", open -> registryOf(open).resolve(selection));
    }

    private static LoraCatalog.Metadata metadataOf(EngineModel open, String name) {
        LoraHandle h = resident(registryOf(open), name);
        return h == null ? LoraCatalog.Metadata.NONE : LoraCatalog.Metadata.of(h.adapter(), open.tokenizer());
    }

    /** Runs {@code call} on the open model on the pump thread and waits for its result; null when it is released. */
    private <T> T modelOp(String what, OpenModelCall<T> call) throws IOException, InterruptedException {
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Exception> failure = new AtomicReference<>();
        AtomicReference<T> result = new AtomicReference<>();
        owner.work().put(() -> {
            try {
                EngineModel open = owner.model();
                if (open != null) {
                    result.set(call.run(open));
                }
            } catch (IOException | RuntimeException e) {
                failure.set(e);
            } finally {
                done.countDown();
            }
        });
        owner.awaitOperation(done, what);
        Exception e = failure.get();
        if (e instanceof IOException io) {
            throw io;
        }
        if (e != null) {
            throw (RuntimeException) e;
        }
        return result.get();
    }

    /** The per-request adapter registry of {@code open}, refused by name when the model runs none. */
    private static LoraRegistry registryOf(EngineModel open) {
        LoraRegistry registry = open.loraRegistry();
        if (registry == null) {
            throw new UnsupportedOperationException("lora: " + open.architecture() + " runs no per-request LoRA"
                    + " adapters (an audio-decoder model serves its base weights alone)");
        }
        return registry;
    }

    /**
     * Makes {@code name} resident from {@code path}: nothing when it already is, a reload when it is resident from
     * another path (an in-place replacement), else a load at the default scale 1.
     *
     * @throws IllegalArgumentException when {@code name} is a startup adapter's ({@code --lora}), which stays
     */
    static void ensureLoaded(LoraRegistry registry, String name, String path) throws IOException {
        LoraHandle resident = resident(registry, name);
        if (resident != null && resident.startup()) {
            throw new IllegalArgumentException("lora: " + name + " names the startup adapter (--lora) "
                    + resident.path() + ", which stays loaded; choose another name");
        }
        if (resident != null && resident.path().equals(path)) {
            return;
        }
        if (resident != null) {
            registry.unload(name);
        }
        registry.load(name, path, 1.0f);
    }

    /** Unloads {@code name} when it is resident. */
    static void ensureUnloaded(LoraRegistry registry, String name) {
        if (resident(registry, name) != null) {
            registry.unload(name);
        }
    }

    private static LoraHandle resident(LoraRegistry registry, String name) {
        for (LoraHandle h : registry.list()) {
            if (h.name().equals(name)) {
                return h;
            }
        }
        return null;
    }
}
