package com.eonmux.neuralfree.gguf.engine.batch;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;
import jdk.jfr.Timespan;

/**
 * The Flight Recorder event of one {@link BatchEngine#pump()} that did work: its duration is the step's, and its
 * fields are what the step scheduled and sampled, the counters' growth over the step. Every running recording
 * ({@code -XX:StartFlightRecording}, {@code jcmd <pid> JFR.start}) records it unless its settings disable
 * {@value #NAME}; without one the pump pays one check.
 */
@Name(EngineStepEvent.NAME)
@Label("Engine Step")
@Category({"NeuralFree", "Engine"})
@Description("One batch engine step: the rows it scheduled, the tokens it sampled and its host sampling time")
@StackTrace(false)
final class EngineStepEvent extends jdk.jfr.Event {

    /** The event's type name. */
    static final String NAME = "com.eonmux.neuralfree.EngineStep";

    @Label("Decode Rows")
    @Description("Rows of decoding sequences the step evaluated, a speculative round's verify rows included")
    long decodeRows;

    @Label("Prefill Rows")
    @Description("Prompt rows the step prefilled, recomputed rows after a preemption included")
    long prefillRows;

    @Label("Cached Prompt Tokens")
    @Description("Prompt tokens of the step's admissions reused from the prefix cache")
    long cachedPromptTokens;

    @Label("Generated Tokens")
    long generatedTokens;

    @Label("Device Sampled Rows")
    @Description("Tokens sampled over a device top-k readback instead of a whole logits row")
    long deviceSampledRows;

    @Label("Host Sample Time")
    @Description("Time the step spent in the host sampler chains")
    @Timespan(Timespan.NANOSECONDS)
    long hostSampleNanos;

    @Label("Preemptions")
    long preemptions;

    @Label("Running")
    @Description("Sequences running after the step")
    int running;

    @Label("Queued")
    @Description("Submissions waiting after the step, preempted ones included")
    int queued;

    @Label("KV Rows Used")
    @Description("Memory rows in use after the step, cached prefixes included")
    int usedCells;
}
