package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.lora.LoraHandle;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The LoRA adapters the server serves, in load order, shared by every engine replica (each replica's
 * {@code LoraRegistry} holds the same adapters): vLLM's {@code lora_requests} by name and llama.cpp server's
 * {@code lora_adapters} by index. The startup adapters ({@code --lora}, {@code --lora-scaled}) come first, as
 * llama.cpp lists them, then the per-request ones ({@code --lora-modules}, runtime loads). An adapter's
 * {@code scale} is the llama.cpp global scale — the scale a request that selects no adapter runs it at
 * ({@code POST /lora-adapters}): a startup adapter starts at its startup scale, a per-request one at 0, so such a
 * request runs the startup adapters, as llama.cpp's does, and no per-request adapter, as vLLM's does, until a
 * global scale changes.
 *
 * <p>A request selects its adapters in one of three ways ({@link #select}): llama.cpp's {@code lora} list
 * ({@code [{"id", "scale"}]}, every unlisted adapter at 0 — a startup one included), vLLM's {@code model} naming an
 * adapter (at its default scale, alone), or neither (the global scales). Ids are positions in the load order, so an
 * unload renumbers the later adapters.</p>
 *
 * <p>Thread-safe: the list is an immutable snapshot replaced on every change, the changes synchronized (the
 * administration and a replica's open both record).</p>
 */
final class LoraCatalog {

    /**
     * One served adapter.
     *
     * @param name         its name ({@code lora_name}, the {@code model} a request names)
     * @param path         the adapter file or directory it loaded from
     * @param scale        its llama.cpp global scale
     * @param startupScale the scale a startup adapter loaded at, the one slot 0 runs it at; NaN for a
     *                     per-request adapter
     * @param metadata     its PEFT metadata as llama.cpp lists it, {@link Metadata#NONE} until a replica loaded it
     */
    record Adapter(String name, String path, float scale, float startupScale, Metadata metadata) {

        /** A per-request adapter at the global scale {@code scale}, its metadata unread. */
        Adapter(String name, String path, float scale) {
            this(name, path, scale, Float.NaN, Metadata.NONE);
        }

        /** Whether it is a startup adapter ({@code --lora}), held in slot 0 and never unloaded. */
        boolean startup() {
            return !Float.isNaN(startupScale);
        }

        /** This adapter at the global scale {@code to}. */
        Adapter withScale(float to) {
            return new Adapter(name, path, to, startupScale, metadata);
        }

        /** This adapter with {@code read} as its metadata. */
        Adapter withMetadata(Metadata read) {
            return new Adapter(name, path, scale, startupScale, read);
        }

        /** The catalog entry of the startup adapter {@code handle}, at its startup scale. */
        static Adapter startup(LoraHandle handle, GGUFTokenizer tokenizer) {
            return new Adapter(handle.name(), handle.path(), handle.scale(), handle.scale(),
                    Metadata.of(handle.adapter(), tokenizer));
        }
    }

    /**
     * The PEFT metadata llama.cpp's {@code GET /lora-adapters} reports of an adapter
     * ({@code adapter.lora.task_name}, {@code adapter.lora.prompt_prefix}, {@code adapter.alora.invocation_tokens}).
     *
     * @param taskName         the task name, or null
     * @param promptPrefix     the prompt prefix, or null
     * @param invocationTokens an activated LoRA's invocation token ids, empty for a plain adapter
     * @param invocationString the invocation tokens' pieces concatenated ({@code common_token_to_piece}), empty
     *                         for a plain adapter
     */
    record Metadata(String taskName, String promptPrefix, List<Integer> invocationTokens,
                    String invocationString) {

        /** No metadata. */
        static final Metadata NONE = new Metadata(null, null, List.of(), "");

        Metadata {
            invocationTokens = List.copyOf(invocationTokens);
        }

        /** The metadata of {@code adapter}, its invocation pieces rendered by {@code tokenizer}. */
        static Metadata of(LoraAdapter adapter, GGUFTokenizer tokenizer) {
            int[] tokens = adapter.invocationTokens();
            List<Integer> ids = new ArrayList<>(tokens.length);
            ByteArrayOutputStream pieces = new ByteArrayOutputStream();
            for (int t : tokens) {
                ids.add(t);
                pieces.writeBytes(tokenizer.pieceWithSpecial(t));
            }
            return new Metadata(adapter.taskName(), adapter.promptPrefix(), ids,
                    pieces.toString(StandardCharsets.UTF_8));
        }
    }

    private volatile List<Adapter> adapters = List.of();

    /** The adapters in load order (index = llama.cpp id). */
    List<Adapter> list() {
        return adapters;
    }

    /** The adapter named {@code name}, or null. */
    Adapter find(String name) {
        return find(adapters, name);
    }

    /** Whether an adapter is named {@code name}. */
    boolean contains(String name) {
        return find(name) != null;
    }

    /**
     * Puts the startup adapters {@code startup} before every per-request adapter, in their load order. A replica's
     * later open (another replica, a wake from a full sleep) records the same adapters again: an entry already
     * recorded keeps its global scale.
     *
     * @throws IllegalArgumentException when one is not a startup adapter, or a per-request adapter takes its name
     */
    synchronized void startup(List<Adapter> startup) {
        List<Adapter> next = new ArrayList<>(startup.size() + adapters.size());
        for (Adapter a : startup) {
            if (!a.startup()) {
                throw new IllegalArgumentException(a.name() + " is not a startup adapter");
            }
            Adapter recorded = find(adapters, a.name());
            if (recorded != null && !recorded.startup()) {
                throw new IllegalArgumentException("--lora-modules: the adapter name " + a.name() + " is the name"
                        + " of the startup adapter (--lora) " + a.path());
            }
            next.add(recorded != null && recorded.path().equals(a.path()) ? a.withScale(recorded.scale()) : a);
        }
        for (Adapter a : adapters) {
            if (!a.startup()) {
                next.add(a);
            }
        }
        adapters = List.copyOf(next);
    }

    /** Records {@code metadata} as the metadata of the adapter {@code name}, when it is still served. */
    synchronized void metadata(String name, Metadata metadata) {
        List<Adapter> next = new ArrayList<>(adapters);
        for (int i = 0; i < next.size(); i++) {
            if (next.get(i).name().equals(name)) {
                next.set(i, next.get(i).withMetadata(metadata));
                adapters = List.copyOf(next);
                return;
            }
        }
    }

    /** Appends {@code adapter}, or replaces the entry of its name in place. */
    synchronized void put(Adapter adapter) {
        List<Adapter> next = new ArrayList<>(adapters);
        for (int i = 0; i < next.size(); i++) {
            if (next.get(i).name().equals(adapter.name())) {
                next.set(i, adapter);
                adapters = List.copyOf(next);
                return;
            }
        }
        next.add(adapter);
        adapters = List.copyOf(next);
    }

    /** Removes the adapter {@code name}; returns its position, or -1 when none is named so. */
    synchronized int remove(String name) {
        List<Adapter> next = new ArrayList<>(adapters);
        for (int i = 0; i < next.size(); i++) {
            if (next.get(i).name().equals(name)) {
                next.remove(i);
                adapters = List.copyOf(next);
                return i;
            }
        }
        return -1;
    }

    /** Puts {@code adapter} back at {@code index} (a refused removal's rollback). */
    synchronized void restore(int index, Adapter adapter) {
        List<Adapter> next = new ArrayList<>(adapters);
        next.add(Math.min(index, next.size()), adapter);
        adapters = List.copyOf(next);
    }

    /**
     * Sets the global scales of llama.cpp's {@code POST /lora-adapters} body: every listed id takes its scale,
     * every other adapter 0 ({@code construct_lora_list}).
     *
     * @throws IllegalArgumentException naming an invalid entry, id or scale
     */
    synchronized void setScales(Object body) {
        if (!(body instanceof List<?> entries)) {
            throw new IllegalArgumentException("Request body must be an array");
        }
        List<Adapter> current = adapters;
        float[] scales = scales(entries, current, "body");
        List<Adapter> next = new ArrayList<>(current.size());
        for (int i = 0; i < current.size(); i++) {
            next.add(current.get(i).withScale(scales[i]));
        }
        adapters = List.copyOf(next);
    }

    /**
     * The adapters a request runs: its {@code lora} list (llama.cpp), else the adapter its {@code model} names
     * (vLLM), else the global scales; null runs slot 0 — the startup adapters at their startup scales — which
     * the global scales select until one changes.
     *
     * @throws IllegalArgumentException naming the field when the list is malformed, names an unknown id, or the
     *                                  request selects adapters both ways
     */
    LoraSelection select(JsonView body, String model) {
        List<Adapter> current = adapters;
        Object raw = body.raw("lora");
        boolean listed = raw != null && !(raw instanceof List<?> l && l.isEmpty());
        boolean named = model != null && find(current, model) != null;
        if (listed && named) {
            throw new IllegalArgumentException("lora: the request names adapter " + model + " as its model;"
                    + " select adapters by model or by the lora list, not both");
        }
        if (named) {
            return LoraSelection.of(model);
        }
        if (listed) {
            if (!(raw instanceof List<?> entries)) {
                throw new IllegalArgumentException("lora must be an array of {\"id\", \"scale\"} objects");
            }
            return selection(current, scales(entries, current, "lora"));
        }
        float[] global = new float[current.size()];
        boolean slot0 = true;
        for (int i = 0; i < global.length; i++) {
            Adapter a = current.get(i);
            global[i] = a.scale();
            slot0 &= Float.compare(global[i], a.startup() ? a.startupScale() : 0.0f) == 0;
        }
        return slot0 ? null : selection(current, global);
    }

    /**
     * The selection of every adapter of non-zero {@code scales} and of every startup adapter (at 0 too, since a
     * selection replaces the startup set), or null when it names none: no startup adapter and no scale.
     */
    private static LoraSelection selection(List<Adapter> current, float[] scales) {
        LoraSelection.Builder b = LoraSelection.builder();
        boolean any = false;
        for (int i = 0; i < scales.length; i++) {
            if (scales[i] != 0.0f || current.get(i).startup()) {
                b.add(current.get(i).name(), scales[i]);
                any = true;
            }
        }
        return any ? b.build() : null;
    }

    /**
     * The per-adapter scales of a llama.cpp {@code [{"id", "scale"}]} list: listed ids at their scale (0 when the
     * entry gives none, as llama.cpp reads it), the others at 0.
     */
    private static float[] scales(List<?> entries, List<Adapter> current, String field) {
        float[] out = new float[current.size()];
        boolean[] seen = new boolean[current.size()];
        for (Object o : entries) {
            if (!(o instanceof Map<?, ?> entry)) {
                throw new IllegalArgumentException(field + " entries must be {\"id\", \"scale\"} objects");
            }
            if (!(entry.get("id") instanceof Number idValue) || idValue.doubleValue() != idValue.longValue()) {
                throw new IllegalArgumentException(field + " entry needs an integer id, got " + entry.get("id"));
            }
            long id = idValue.longValue();
            if (id < 0 || id >= current.size()) {
                throw new IllegalArgumentException(field + ": id " + id + " names no adapter (" + (current.isEmpty()
                        ? "none is loaded" : "ids 0 to " + (current.size() - 1) + " are loaded") + ")");
            }
            if (seen[(int) id]) {
                throw new IllegalArgumentException(field + ": id " + id + " is listed twice");
            }
            seen[(int) id] = true;
            Object scale = entry.get("scale");
            if (scale != null && !(scale instanceof Number)) {
                throw new IllegalArgumentException(field + ": the scale of id " + id + " must be a number");
            }
            float s = scale == null ? 0.0f : ((Number) scale).floatValue();
            if (!Float.isFinite(s)) {
                throw new IllegalArgumentException(field + ": the scale of id " + id + " is not finite");
            }
            out[(int) id] = s;
        }
        return out;
    }

    private static Adapter find(List<Adapter> list, String name) {
        for (Adapter a : list) {
            if (a.name().equals(name)) {
                return a;
            }
        }
        return null;
    }
}
