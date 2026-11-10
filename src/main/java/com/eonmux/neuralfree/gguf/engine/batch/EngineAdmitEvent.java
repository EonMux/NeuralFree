package com.eonmux.neuralfree.gguf.engine.batch;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;
import jdk.jfr.Timespan;

/**
 * The Flight Recorder event of a request's first admission into a sequence slot: how long it queued since its
 * submit (vLLM's queue time, kept over later preemptions) and the slot it took. Every running recording records it
 * unless its settings disable {@value #NAME}.
 */
@Name(EngineAdmitEvent.NAME)
@Label("Engine Admit")
@Category({"NeuralFree", "Engine"})
@Description("A request's first admission into a batch engine slot and its time in the queue")
@StackTrace(false)
final class EngineAdmitEvent extends jdk.jfr.Event {

    /** The event's type name. */
    static final String NAME = "com.eonmux.neuralfree.EngineAdmit";

    @Label("Submit Id")
    int submitId;

    @Label("Sequence")
    int seq;

    @Label("Queue Time")
    @Timespan(Timespan.NANOSECONDS)
    long queueNanos;
}
