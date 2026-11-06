package com.eonmux.neuralfree.gguf.engine.batch;

import java.util.Locale;

/**
 * The order in which a {@link BatchEngine} serves its requests (vLLM's {@code SchedulerConfig.policy},
 * {@code --scheduling-policy}), set with {@code neuralfree.scheduling.policy}:
 * <ul>
 *   <li>{@link #FCFS}: first come, first served. Admission follows arrival, a preempted sequence re-admits ahead
 *       of the queue, and running out of pages preempts the most recently admitted running sequence. A request's
 *       {@code priority} must be 0.</li>
 *   <li>{@link #PRIORITY}: the lowest {@code priority} first, among equals the earliest arrival. Queued
 *       submissions and preempted sequences compete for admission in that order, and running out of pages
 *       preempts the running sequence served last in it: the highest {@code priority} value, among equals the
 *       latest arrival.</li>
 * </ul>
 */
public enum SchedulingPolicy {

    /** First come, first served (vLLM's default {@code fcfs}). */
    FCFS("fcfs"),
    /** Lowest {@code priority} value first, ties by arrival (vLLM's {@code priority}). */
    PRIORITY("priority");

    private final String wireName;

    SchedulingPolicy(String wireName) {
        this.wireName = wireName;
    }

    /** The policy's name as {@code neuralfree.scheduling.policy} and vLLM's {@code --scheduling-policy} spell it. */
    public String wireName() {
        return wireName;
    }

    /**
     * The policy {@code value} names, case-insensitive and trimmed.
     *
     * @param key the setting the value came from, named in the refusal
     * @throws IllegalArgumentException naming {@code key} when the value is neither {@code fcfs} nor
     *                                  {@code priority}
     */
    public static SchedulingPolicy parse(String value, String key) {
        String s = value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
        for (SchedulingPolicy p : values()) {
            if (p.wireName.equals(s)) {
                return p;
            }
        }
        throw new IllegalArgumentException(key + "=" + value + " is invalid; use fcfs or priority");
    }

    /**
     * Why this policy refuses a request of {@code priority}, or null when it serves it: {@link #FCFS} serves
     * arrival order alone, so only {@code priority} 0 (vLLM: any priority other than 0 raises an error when the
     * served model does not use priority scheduling); {@link #PRIORITY} serves every value.
     */
    public String priorityRefusal(int priority) {
        if (this == PRIORITY || priority == 0) {
            return null;
        }
        return "priority " + priority + " needs priority scheduling (neuralfree.scheduling.policy=priority,"
                + " vLLM's --scheduling-policy priority); the fcfs policy serves requests in arrival order and"
                + " takes priority 0 only";
    }

    /**
     * Whether a request of {@code priority} that arrived {@code arrival}-th is served before one of
     * {@code otherPriority} that arrived {@code otherArrival}-th: by arrival under {@link #FCFS}, by priority and
     * then arrival under {@link #PRIORITY} (vLLM's {@code Request.__lt__}).
     */
    public boolean servesBefore(int priority, long arrival, int otherPriority, long otherArrival) {
        if (this == PRIORITY && priority != otherPriority) {
            return priority < otherPriority;
        }
        return arrival < otherArrival;
    }

    /** Whether {@code a} is served before {@code b}; both carry their request's priority and arrival. */
    boolean servesBefore(Submission a, Submission b) {
        return servesBefore(a.priority(), a.handle.submitId(), b.priority(), b.handle.submitId());
    }
}
