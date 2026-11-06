package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.binder.MeterBinder;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

import java.util.function.ToDoubleFunction;

/**
 * The engine's Prometheus metrics on {@code /metrics}: llama.cpp's server set ({@code llamacpp:*} in
 * {@code tools/server/server-task.cpp}) under the {@code neuralfree_} prefix, with vLLM's preemption, KV-usage and
 * finish-reason series. Every meter reads the snapshot the pump thread published last ({@link EngineService#stats});
 * before a model loads they read zero, and a scrape never loads one.
 *
 * <table>
 *   <caption>Meters (Prometheus names)</caption>
 *   <tr><td>{@code neuralfree_prompt_tokens_total}</td><td>prompt tokens prefilled, cached ones excluded</td></tr>
 *   <tr><td>{@code neuralfree_prompt_tokens_cached_total}</td><td>prompt tokens reused from the prefix cache</td></tr>
 *   <tr><td>{@code neuralfree_prompt_seconds_total}</td><td>time spent prefilling</td></tr>
 *   <tr><td>{@code neuralfree_tokens_predicted_total}</td><td>tokens sampled</td></tr>
 *   <tr><td>{@code neuralfree_tokens_predicted_seconds_total}</td><td>time spent decoding</td></tr>
 *   <tr><td>{@code neuralfree_decode_total}</td><td>batched decode forwards</td></tr>
 *   <tr><td>{@code neuralfree_recomputed_tokens_total}</td><td>tokens prefilled again after a preemption</td></tr>
 *   <tr><td>{@code neuralfree_preemptions_total}</td><td>running sequences preempted</td></tr>
 *   <tr><td>{@code neuralfree_request_success_total{finished_reason}}</td><td>finished requests: stop, length,
 *       abort</td></tr>
 *   <tr><td>{@code neuralfree_request_failures_total}</td><td>failed requests</td></tr>
 *   <tr><td>{@code neuralfree_requests_processing}</td><td>sequences running</td></tr>
 *   <tr><td>{@code neuralfree_requests_deferred}</td><td>requests waiting, preempted ones included</td></tr>
 *   <tr><td>{@code neuralfree_kv_cache_usage}</td><td>share of KV rows in use, cached prefixes included</td></tr>
 *   <tr><td>{@code neuralfree_tokens_max}</td><td>longest finished sequence</td></tr>
 *   <tr><td>{@code neuralfree_busy_slots_per_decode}</td><td>sequences per decode forward, on average</td></tr>
 * </table>
 */
@Singleton
public class EngineMetrics implements MeterBinder {

    private static final String PREFIX = "neuralfree.";

    @Inject
    EngineLifecycle lifecycle;

    /** Test and manual wiring: meters over an explicit lifecycle. */
    EngineMetrics(EngineLifecycle lifecycle) {
        this.lifecycle = lifecycle;
    }

    public EngineMetrics() {
    }

    @Override
    public void bindTo(MeterRegistry registry) {
        counter(registry, "prompt.tokens", "Prompt tokens prefilled, excluding cached tokens",
                s -> s.promptTokens());
        counter(registry, "prompt.tokens.cached", "Prompt tokens reused from the prefix cache",
                s -> s.promptTokensCached());
        counter(registry, "prompt.seconds", "Time spent prefilling prompts", BatchStats::promptSeconds);
        counter(registry, "tokens.predicted", "Tokens sampled", s -> s.generatedTokens());
        counter(registry, "tokens.predicted.seconds", "Time spent decoding", BatchStats::decodeSeconds);
        counter(registry, "decode", "Batched decode forwards", s -> s.decodeSteps());
        counter(registry, "recomputed.tokens", "Tokens prefilled again after a preemption",
                s -> s.recomputedTokens());
        counter(registry, "preemptions", "Running sequences preempted to free memory", s -> s.preemptions());
        finishCounters(registry);
        counter(registry, "request.failures", "Requests that failed", s -> s.failed());
        gauge(registry, "requests.processing", "Sequences prefilling or decoding", s -> s.running());
        gauge(registry, "requests.deferred", "Requests waiting for a slot or for memory", s -> s.waiting());
        gauge(registry, "kv.cache.usage", "Share of KV memory rows in use, cached prefixes included",
                BatchStats::kvUsage);
        gauge(registry, "tokens.max", "Longest finished sequence (prompt and generation)",
                s -> s.maxSequenceLength());
        gauge(registry, "busy.slots.per.decode", "Average sequences per decode forward", BatchStats::rowsPerDecode);
    }

    /** vLLM's {@code request_success_total} by {@code finished_reason}. */
    private void finishCounters(MeterRegistry registry) {
        String help = "Requests finished, by reason";
        FunctionCounter.builder(PREFIX + "request.success", this, m -> m.read(s -> s.finished(StopReason.EOS)
                        + s.finished(StopReason.STOP_STRING)))
                .description(help).tag("finished_reason", "stop").register(registry);
        FunctionCounter.builder(PREFIX + "request.success", this, m -> m.read(s -> s.finished(StopReason.MAX_TOKENS)
                        + s.finished(StopReason.CONTEXT_FULL)))
                .description(help).tag("finished_reason", "length").register(registry);
        FunctionCounter.builder(PREFIX + "request.success", this,
                        m -> m.read(s -> s.finished(StopReason.CANCELLED)))
                .description(help).tag("finished_reason", "abort").register(registry);
    }

    private void counter(MeterRegistry registry, String name, String help, ToDoubleFunction<BatchStats> value) {
        FunctionCounter.builder(PREFIX + name, this, m -> m.read(value)).description(help).register(registry);
    }

    private void gauge(MeterRegistry registry, String name, String help, ToDoubleFunction<BatchStats> value) {
        Gauge.builder(PREFIX + name, this, m -> m.read(value)).description(help).register(registry);
    }

    /** {@code value} of the latest snapshot, or zero while no service is open. */
    double read(ToDoubleFunction<BatchStats> value) {
        EngineService service = lifecycle == null ? null : lifecycle.current();
        return value.applyAsDouble(service == null ? BatchStats.EMPTY : service.stats());
    }
}
