package com.eonmux.neuralfree.rl;

public interface Policy<StateT, ActionT> {
    ActionT selectAction(StateT state, boolean explore);

    default ActionT selectAction(StateT state) {
        return selectAction(state, true);
    }

    default ActionT selectActionEval(StateT state) {
        return selectAction(state, false);
    }

    default void reset() {
    }

    default String name() {
        return getClass().getSimpleName();
    }
}
