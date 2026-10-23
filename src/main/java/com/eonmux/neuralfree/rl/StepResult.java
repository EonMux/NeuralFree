package com.eonmux.neuralfree.rl;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public final class StepResult<StateT> {
    private final StateT state;
    private final float reward;
    private final boolean done;
    private final Map<String, Object> info;

    public StepResult(StateT state, float reward, boolean done) {
        this(state, reward, done, Collections.emptyMap());
    }

    public StepResult(StateT state, float reward, boolean done, Map<String, Object> info) {
        this.state = state;
        this.reward = reward;
        this.done = done;
        this.info = info != null ? Collections.unmodifiableMap(new HashMap<>(info)) : Collections.emptyMap();
    }

    public StateT state() {
        return state;
    }

    public float reward() {
        return reward;
    }

    public boolean done() {
        return done;
    }

    public Map<String, Object> info() {
        return info;
    }

    public Object info(String key) {
        return info.get(key);
    }

    public Object info(String key, Object defaultValue) {
        return info.getOrDefault(key, defaultValue);
    }

    public boolean hasInfo(String key) {
        return info.containsKey(key);
    }

    public static <StateT> StepResult<StateT> of(StateT state, float reward, boolean done) {
        return new StepResult<>(state, reward, done);
    }

    public static <StateT> StepResult<StateT> of(StateT state, float reward, boolean done, Map<String, Object> info) {
        return new StepResult<>(state, reward, done, info);
    }

    @Override
    public String toString() {
        return "StepResult{state=" + state + ", reward=" + reward + ", done=" + done + ", info=" + info + '}';
    }
}
