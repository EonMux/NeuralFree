package com.eonmux.neuralfree.rl;

import java.util.List;

public interface Environment<StateT, ActionT> {
    StateT reset();

    StepResult<StateT> step(ActionT action);

    int actionSpaceSize();

    List<Integer> observationShape();

    default boolean isContinuousActionSpace() {
        return false;
    }

    default String name() {
        return getClass().getSimpleName();
    }

    default void close() {
    }
}
