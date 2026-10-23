package com.eonmux.neuralfree.rl;

import java.util.Arrays;

/**
 * One immutable environment transition. The action is either discrete, an index ({@link #action()}), or continuous, a
 * vector ({@link #continuousAction()}, as SAC's squashed actions); a continuous transition has {@code action() == -1}.
 * The public constructors copy the arrays they are given, so a caller that reuses its observation arrays (as Gym-style
 * environments do) cannot change a transition it already built.
 */
public final class Transition {
    /** The {@link #action()} of a transition whose action is continuous. */
    public static final int NO_DISCRETE_ACTION = -1;

    private final float[] state;
    private final int action;
    private final float[] continuousAction;
    private final float reward;
    private final float[] nextState;
    private final boolean done;
    private final float priority;

    /** A transition with the discrete action {@code action} at priority 1; the states are copied. */
    public Transition(float[] state, int action, float reward, float[] nextState, boolean done) {
        this(state, action, reward, nextState, done, 1.0f);
    }

    /** A transition with the discrete action {@code action}; the states are copied. */
    public Transition(float[] state, int action, float reward, float[] nextState, boolean done, float priority) {
        this(copyOf(state), action, null, reward, copyOf(nextState), done, priority);
    }

    /**
     * A transition with the continuous action {@code continuousAction}; the states and the action are copied.
     *
     * @throws NullPointerException     when {@code continuousAction} is null
     * @throws IllegalArgumentException when {@code continuousAction} is empty
     */
    public Transition(float[] state, float[] continuousAction, float reward, float[] nextState, boolean done,
                      float priority) {
        this(copyOf(state), NO_DISCRETE_ACTION, requireAction(continuousAction).clone(), reward, copyOf(nextState),
            done, priority);
    }

    /**
     * A discrete transition that keeps {@code state} and {@code nextState} without copying them. The caller hands
     * over arrays it already copied and must not modify them afterwards.
     */
    static Transition ofOwnedStates(float[] state, int action, float reward, float[] nextState, boolean done,
                                    float priority) {
        return new Transition(state, action, null, reward, nextState, done, priority);
    }

    /**
     * A continuous transition that keeps {@code state} and {@code nextState} without copying them; the action is
     * copied. The caller hands over state arrays it already copied and must not modify them afterwards.
     *
     * @throws NullPointerException     when {@code continuousAction} is null
     * @throws IllegalArgumentException when {@code continuousAction} is empty
     */
    static Transition ofOwnedStates(float[] state, float[] continuousAction, float reward, float[] nextState,
                                    boolean done, float priority) {
        return new Transition(state, NO_DISCRETE_ACTION, requireAction(continuousAction).clone(), reward, nextState,
            done, priority);
    }

    /** Keeps every array as given: callers pass arrays that no one else holds. */
    private Transition(float[] state, int action, float[] continuousAction, float reward, float[] nextState,
                       boolean done, float priority) {
        this.state = state;
        this.action = action;
        this.continuousAction = continuousAction;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
        this.priority = priority;
    }

    public float[] state() {
        return state != null ? state.clone() : null;
    }

    /** The state itself (not copied), for readers in this package that do not modify it. */
    float[] stateRef() {
        return state;
    }

    public int action() {
        return action;
    }

    /** A copy of the continuous action, or null for a discrete transition. */
    public float[] continuousAction() {
        return continuousAction != null ? continuousAction.clone() : null;
    }

    /**
     * The continuous action itself (not copied), or null for a discrete transition, for readers in this package that
     * do not modify it.
     */
    float[] continuousActionRef() {
        return continuousAction;
    }

    private static float[] copyOf(float[] values) {
        return values != null ? values.clone() : null;
    }

    private static float[] requireAction(float[] continuousAction) {
        if (continuousAction == null) {
            throw new NullPointerException("Continuous action cannot be null");
        }
        if (continuousAction.length == 0) {
            throw new IllegalArgumentException("Continuous action cannot be empty");
        }
        return continuousAction;
    }

    public float reward() {
        return reward;
    }

    public float[] nextState() {
        return nextState != null ? nextState.clone() : null;
    }

    /** The next state itself (not copied), for readers in this package that do not modify it. */
    float[] nextStateRef() {
        return nextState;
    }

    public boolean done() {
        return done;
    }

    public float priority() {
        return priority;
    }

    public Transition withPriority(float newPriority) {
        return new Transition(state, action, continuousAction, reward, nextState, done, newPriority);
    }

    public int stateSize() {
        return state != null ? state.length : 0;
    }

    @Override
    public String toString() {
        String actionText = continuousAction != null ? Arrays.toString(continuousAction) : String.valueOf(action);
        return "Transition{action=" + actionText + ", reward=" + reward + ", done=" + done + ", priority=" + priority
            + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Transition that = (Transition) o;
        return action == that.action &&
               Float.compare(that.reward, reward) == 0 &&
               done == that.done &&
               Float.compare(that.priority, priority) == 0 &&
               Arrays.equals(state, that.state) &&
               Arrays.equals(continuousAction, that.continuousAction) &&
               Arrays.equals(nextState, that.nextState);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(state);
        result = 31 * result + action;
        result = 31 * result + Arrays.hashCode(continuousAction);
        result = 31 * result + Float.hashCode(reward);
        result = 31 * result + Arrays.hashCode(nextState);
        result = 31 * result + Boolean.hashCode(done);
        result = 31 * result + Float.hashCode(priority);
        return result;
    }
}
