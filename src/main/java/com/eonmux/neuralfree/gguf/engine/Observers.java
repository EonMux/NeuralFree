package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The observation channels installed in a backend and its contexts: hidden taps, execution observers, matmul
 * overrides and diagnostics hooks. Consumers of prompt-row outputs read {@code ForwardBackend.observers()} and
 * treat every channel in the inventory as one that observes the rows it is installed for, so no output row a
 * channel needs may be skipped by a reuse path.
 *
 * <p>Immutable. {@link #UNDECLARED} marks a backend that does not publish its inventory; a conservative consumer
 * treats it as observed (every row is computed). Backends with a known inventory publish it with
 * {@link #declared(List)}.</p>
 */
public final class Observers {

    /**
     * The backend does not declare the observer objects installed in it and its contexts. Treated as observed:
     * consumers assume channels may be present and never skip an output row.
     */
    public static final Observers UNDECLARED = new Observers(false, Collections.emptyList());

    private final boolean declared;
    private final List<Object> channels;

    private Observers(boolean declared, List<Object> channels) {
        this.declared = declared;
        this.channels = channels;
    }

    /**
     * A published inventory of the observer objects installed in a backend and its contexts.
     *
     * @throws NullPointerException when {@code channels} is null
     */
    public static Observers declared(List<?> channels) {
        Objects.requireNonNull(channels, "channels");
        List<Object> copy = new ArrayList<>();
        for (Object channel : channels) {
            copy.add(Objects.requireNonNull(channel, "channel"));
        }
        return new Observers(true, Collections.unmodifiableList(copy));
    }

    /** True when the backend publishes its inventory ({@code false} for {@link #UNDECLARED}). */
    public boolean declared() {
        return declared;
    }

    /** The declared observer objects, unmodifiable; empty for {@link #UNDECLARED}. */
    public List<Object> channels() {
        return channels;
    }

    @Override
    public String toString() {
        return declared ? "Observers" + channels : "Observers.UNDECLARED";
    }
}
