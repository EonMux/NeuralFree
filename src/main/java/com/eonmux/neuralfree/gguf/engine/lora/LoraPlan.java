package com.eonmux.neuralfree.gguf.engine.lora;

import com.eonmux.neuralfree.gguf.engine.MediaPrompt;

import java.util.Objects;

/**
 * The adapter slots of one request's rows: the key its rows run from position {@link #from()} on, and the key of
 * the rows before it. A plain selection runs one key on every row ({@code from} 0). An activated LoRA (aLoRA,
 * llama.cpp's {@code alora_invocation_tokens}, the PEFT {@code alora} form) applies only from the last occurrence
 * of its invocation sequence in the prompt on: the rows before it run the base model
 * ({@link LoraRegistry#preInvocation}); a prompt without the sequence runs the base model on every row, as
 * llama.cpp disables the aLoRA then. Over a media prompt ({@link MediaPrompt}) the sequence is searched among the
 * text tokens and {@code from} is the KV row of its first token, media rows counted in the numbering, as vLLM
 * counts a media placeholder as one token position. A null key runs in slot 0 (the startup adapters); a startup
 * activated LoRA plans the same way for a request that selects none, so its slot-0 rows start at its invocation.
 *
 * <p>Immutable.</p>
 */
public final class LoraPlan {

    /** The plan of a request that runs slot 0 on every row. */
    public static final LoraPlan DEFAULT = new LoraPlan(null, null, 0);

    private final LoraSlotKey key;
    private final LoraSlotKey preKey;
    private final int from;

    private LoraPlan(LoraSlotKey key, LoraSlotKey preKey, int from) {
        this.key = key;
        this.preKey = preKey;
        this.from = from;
    }

    /**
     * The plan of {@code selection} over {@code registry} for a prompt of {@code promptTokens}: the resolved key
     * ({@link LoraRegistry#resolve}) on every row, or, for an activated LoRA, from its invocation start on.
     *
     * @param selection    the request's selection, or null for none (slot 0, the {@link LoraRegistry#defaultKey})
     * @param promptTokens the prompt's token ids, position {@code i} at index {@code i}; null for a prompt whose
     *                     rows do not map to token positions, on which an activated LoRA is refused
     * @throws IllegalArgumentException      as {@link LoraRegistry#resolve}
     * @throws UnsupportedOperationException for an activated LoRA over rows that hold no token ids
     */
    public static LoraPlan resolve(LoraRegistry registry, LoraSelection selection, int[] promptTokens) {
        Objects.requireNonNull(registry, "registry");
        if (promptTokens == null) {
            return uniform(registry, selection, "it finds its invocation tokens among the token ids of a prompt,"
                    + " and these rows hold none");
        }
        LoraSlotKey key = selection == null ? null : registry.resolve(selection);
        LoraHandle activated = (key == null ? registry.defaultKey() : key).activated();
        if (activated == null) {
            return key == null ? DEFAULT : new LoraPlan(key, key, 0);
        }
        return activatedPlan(registry, key, activated.adapter().invocationStart(promptTokens));
    }

    /**
     * {@link #resolve(LoraRegistry, LoraSelection, int[])} over a media-aware prompt: the invocation sequence is
     * searched among the prompt's text tokens and the match's row index is the KV row that holds its first token,
     * media rows counted in the numbering (vLLM counts a media placeholder as one token position), so the adapter
     * applies from there over the media rows behind it too. A prompt without the sequence runs the base model.
     *
     * @throws IllegalArgumentException      as {@link LoraRegistry#resolve}
     * @throws UnsupportedOperationException for an activated LoRA over rows that hold no token ids
     */
    public static LoraPlan resolve(LoraRegistry registry, LoraSelection selection, MediaPrompt prompt) {
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(prompt, "prompt");
        if (prompt.isTextOnly()) {
            return resolve(registry, selection, prompt.samplerTokens());
        }
        LoraSlotKey key = selection == null ? null : registry.resolve(selection);
        LoraHandle activated = (key == null ? registry.defaultKey() : key).activated();
        if (activated == null) {
            return key == null ? DEFAULT : new LoraPlan(key, key, 0);
        }
        int start = activated.adapter().invocationStart(prompt.samplerTokens());
        int[] rows = start < 0 ? null : prompt.samplerTokenRows();
        return activatedPlan(registry, key, start < 0 ? -1 : rows[start]);
    }

    /** The plan of a resolved key whose rows run the activated LoRA {@code activated} from row {@code start} on. */
    private static LoraPlan activatedPlan(LoraRegistry registry, LoraSlotKey key, int start) {
        LoraSlotKey pre = registry.preInvocation(key == null ? registry.defaultKey() : key);
        return start < 0 ? new LoraPlan(pre, pre, 0) : new LoraPlan(key, pre, start);
    }

    /**
     * The plan of {@code selection} over {@code registry} for a pass that runs every row in one slot (a pooling
     * forward, a prompt of media rows): the resolved key ({@link LoraRegistry#resolve}) on every row.
     *
     * @param selection the request's selection, or null for none (slot 0)
     * @param refusal   why the pass cannot run an activated LoRA, the end of the refusal's message
     * @throws IllegalArgumentException      as {@link LoraRegistry#resolve}
     * @throws UnsupportedOperationException when the rows would run an activated LoRA, selected or at startup
     */
    public static LoraPlan uniform(LoraRegistry registry, LoraSelection selection, String refusal) {
        Objects.requireNonNull(registry, "registry");
        Objects.requireNonNull(refusal, "refusal");
        LoraSlotKey key = selection == null ? null : registry.resolve(selection);
        LoraHandle activated = (key == null ? registry.defaultKey() : key).activated();
        if (activated != null) {
            throw new UnsupportedOperationException("lora: " + activated.name() + " is an activated LoRA (aLoRA),"
                    + " which applies from its invocation tokens on; " + refusal);
        }
        return key == null ? DEFAULT : new LoraPlan(key, key, 0);
    }

    /** The key of the rows at position {@link #from()} and later, or null for slot 0. */
    public LoraSlotKey key() {
        return key;
    }

    /** The key of the rows before {@link #from()}, or null for slot 0; the key itself when {@code from} is 0. */
    public LoraSlotKey preKey() {
        return preKey;
    }

    /** The first row {@link #key()} runs at: an activated LoRA's invocation start, else 0. */
    public int from() {
        return from;
    }

    /** True when every row runs in slot 0. */
    public boolean isDefault() {
        return key == null && preKey == null;
    }

    /**
     * The content identity of the rows the plan computes, the key the prefix cache matches on: null for slot 0,
     * the key's {@link LoraSlotKey#identity()}, and for an activated LoRA its identity at its invocation start
     * (rows of two invocation starts differ; a startup one's rows are {@code slot0} at its start).
     */
    public String identity() {
        if (isDefault()) {
            return null;
        }
        String rows = key == null ? "slot0" : key.identity();
        return from == 0 ? rows : rows + "@" + from;
    }

    /** The adapters' names as KV cache events report them (vLLM's {@code lora_name}), or null for none. */
    public String label() {
        return key == null || key.isBare() ? null : key.label();
    }

    @Override
    public String toString() {
        return isDefault() ? "LoraPlan{slot 0}" : "LoraPlan{" + key + (from > 0 ? " from " + from + ", before "
                + preKey : "") + "}";
    }
}
