package com.eonmux.neuralfree.rl;

import java.util.Objects;

/**
 * Validation of the states that agents and replay buffers keep. A stored state is a copy, so a caller that reuses its
 * observation array (as Gym-style environments do) cannot change transitions it already stored.
 */
final class RlInputs {

    private RlInputs() {
    }

    /**
     * A copy of {@code state}, which must hold {@code stateSize} values; {@code what} names it in the messages
     * ("State", "Next state").
     *
     * @throws NullPointerException     when {@code state} is null
     * @throws IllegalArgumentException when {@code state} does not hold {@code stateSize} values
     */
    static float[] copyOfState(float[] state, int stateSize, String what) {
        Objects.requireNonNull(state, what + " cannot be null");
        if (state.length != stateSize) {
            throw new IllegalArgumentException(what + " size mismatch: expected " + stateSize + ", got "
                + state.length);
        }
        return state.clone();
    }
}
