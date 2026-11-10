package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.binder.MeterBinder;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

import java.time.Duration;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;

/**
 * The engine's Prometheus metrics on {@code /metrics}: llama.cpp's server set ({@code llamacpp:*} in
 * {@code tools/server/server-task.cpp}) under the {@code neuralfree_} prefix, with vLLM's preemption, KV-usage and
 * finish-reason series. Every meter reads the snapshot the pump thread published last ({@link EngineService#stats});
 * before a model loads they read zero, and a scrape never loads one.
 *
 * <p>Micrometer holds the object a function meter reads through a weak reference, so the meters stay live only
 * while the binder is reachable: the container keeps its singleton, and manual wiring must keep its instance.
 *
 * <table>
 *   <caption>Meters (Prometheus names)</caption>
 *   <tr><td>{@code neuralfree_prompt_tokens_total}</td><td>prompt tokens prefilled, cached ones excluded</td></tr>
 *   <tr><td>{@code neuralfree_prompt_tokens_cached_total}</td><td>prompt tokens reused from the prefix cache</td></tr>
 *   <tr><td>{@code neuralfree_prefix_cache_hit_rate}</td><td>share of prompt tokens served from the prefix cache
 *       (vLLM's {@code gpu_prefix_cache_hit_rate})</td></tr>
 *   <tr><td>{@code neuralfree_prompt_seconds_total}</td><td>time spent prefilling</td></tr>
 *   <tr><td>{@code neuralfree_tokens_predicted_total}</td><td>tokens sampled</td></tr>
 *   <tr><td>{@code neuralfree_tokens_predicted_seconds_total}</td><td>time spent decoding</td></tr>
 *   <tr><td>{@code neuralfree_host_sample_seconds_total}</td><td>time spent in the host sampler chains</td></tr>
 *   <tr><td>{@code neuralfree_decode_total}</td><td>batched decode forwards</td></tr>
 *   <tr><td>{@code neuralfree_recomputed_tokens_total}</td><td>tokens prefilled again after a preemption</td></tr>
 *   <tr><td>{@code neuralfree_preemptions_total}</td><td>running sequences preempted</td></tr>
 *   <tr><td>{@code neuralfree_encoder_cache_rows}</td><td>rows of shared encoder output the engine's encoder
 *       cache holds</td></tr>
 *   <tr><td>{@code neuralfree_encoder_cache_hits_total}</td><td>media chunks whose encoder output the encoder
 *       cache already held</td></tr>
 *   <tr><td>{@code neuralfree_encoder_cache_evictions_total}</td><td>encoder cache entries evicted for space</td></tr>
 *   <tr><td>{@code neuralfree_spec_rounds_total}</td><td>speculative verify rounds</td></tr>
 *   <tr><td>{@code neuralfree_spec_drafted_tokens_total}</td><td>draft tokens the target verified</td></tr>
 *   <tr><td>{@code neuralfree_spec_accepted_tokens_total}</td><td>draft tokens the target confirmed</td></tr>
 *   <tr><td>{@code neuralfree_lookahead_rounds_total}</td><td>lookahead (Jacobi) decoding steps</td></tr>
 *   <tr><td>{@code neuralfree_lookahead_emitted_tokens_total}</td><td>tokens lookahead steps emitted, each step's
 *       own token included</td></tr>
 *   <tr><td>{@code neuralfree_lookahead_accepted_tokens_total}</td><td>tokens lookahead steps confirmed from an
 *       observed n-gram beyond each step's own token</td></tr>
 *   <tr><td>{@code neuralfree_request_success_total{finished_reason}}</td><td>finished requests: stop, length,
 *       abort</td></tr>
 *   <tr><td>{@code neuralfree_request_failures_total}</td><td>failed requests</td></tr>
 *   <tr><td>{@code neuralfree_requests_processing}</td><td>sequences running</td></tr>
 *   <tr><td>{@code neuralfree_requests_deferred}</td><td>requests waiting, preempted ones included</td></tr>
 *   <tr><td>{@code neuralfree_kv_cache_usage}</td><td>share of KV rows in use, cached prefixes included</td></tr>
 *   <tr><td>{@code neuralfree_tokens_max}</td><td>longest finished sequence</td></tr>
 *   <tr><td>{@code neuralfree_busy_slots_per_decode}</td><td>sequences per decode forward, on average</td></tr>
 *   <tr><td>{@code neuralfree_mm_cache_queries_total}</td><td>media items looked up in the projector's media cache
 *       (vLLM's {@code mm_cache_queries})</td></tr>
 *   <tr><td>{@code neuralfree_mm_cache_hits_total}</td><td>media items the projector's media cache served without
 *       decoding them, as an encoding or as processed media to encode (vLLM's {@code mm_cache_hits})</td></tr>
 *   <tr><td>{@code neuralfree_mm_cache_usage_bytes}</td><td>bytes the projector's media cache holds, encodings and
 *       processed media</td></tr>
 *   <tr><td>{@code neuralfree_dp_replicas}</td><td>engine replicas serving the model here (the local
 *       data-parallel size)</td></tr>
 *   <tr><td>{@code neuralfree_dp_size}</td><td>the deployment's global data-parallel size</td></tr>
 *   <tr><td>{@code neuralfree_replica_requests_processing{replica,dp_rank}}</td><td>one replica's running
 *       sequences</td></tr>
 *   <tr><td>{@code neuralfree_replica_requests_deferred{replica,dp_rank}}</td><td>one replica's waiting requests,
 *       preempted ones included</td></tr>
 *   <tr><td>{@code neuralfree_replica_kv_cache_usage{replica,dp_rank}}</td><td>one replica's share of KV rows in
 *       use</td></tr>
 *   <tr><td>{@code neuralfree_replica_prompt_tokens_in_flight{replica,dp_rank}}</td><td>the prompt tokens of one
 *       replica's requests not yet finished (what {@code minimum_tokens} balances)</td></tr>
 *   <tr><td>{@code neuralfree_time_to_first_token_seconds}</td><td>histogram: arrival to first sampled token
 *       (vLLM's {@code time_to_first_token_seconds} and its buckets)</td></tr>
 *   <tr><td>{@code neuralfree_inter_token_latency_seconds}</td><td>histogram: time between consecutive sampled
 *       tokens</td></tr>
 *   <tr><td>{@code neuralfree_e2e_request_latency_seconds}</td><td>histogram: arrival to the request's end</td></tr>
 *   <tr><td>{@code neuralfree_request_queue_time_seconds}</td><td>histogram: engine submit to first
 *       admission</td></tr>
 *   <tr><td>{@code vllm:mm_cache_queries_total{model_name,engine}}</td><td>{@code neuralfree_mm_cache_queries_total}
 *       under vLLM's name and labels</td></tr>
 *   <tr><td>{@code vllm:mm_cache_hits_total{model_name,engine}}</td><td>{@code neuralfree_mm_cache_hits_total}
 *       under vLLM's name and labels</td></tr>
 * </table>
 *
 * <p>The per-replica series are tagged with the replica's pool index ({@code replica}) and its global data-parallel
 * rank ({@code dp_rank}). The replicas exist only once the service opens, which happens after the registry binds
 * this binder, so they register when it opens ({@link #bindOpened}): through the lifecycle's open hook
 * ({@link EngineLifecycle#onOpen}) on every registry this binder bound, or when the batch runner opens its engine.
 * So do vLLM's multimodal cache counters, which keep vLLM's names and labels: the served model name
 * ({@code model_name}) and the engine index ({@code engine}, 0: the server holds one media cache for all its
 * replicas).
 */
@Singleton
public class EngineMetrics implements MeterBinder {

    private static final String PREFIX = "neuralfree.";

    @Inject
    EngineLifecycle lifecycle;

    /** The open service outside the application (the batch runner's metrics), or null to read the lifecycle. */
    private final Supplier<EngineService> current;

    /** Test and manual wiring: meters over an explicit lifecycle. */
    EngineMetrics(EngineLifecycle lifecycle) {
        this.lifecycle = lifecycle;
        this.current = null;
    }

    /** Meters over whatever service {@code current} holds, null while none is open; no lifecycle involved. */
    EngineMetrics(Supplier<EngineService> current) {
        this.current = current;
    }

    public EngineMetrics() {
        this.current = null;
    }

    @Override
    public void bindTo(MeterRegistry registry) {
        counter(registry, "prompt.tokens", "Prompt tokens prefilled, excluding cached tokens",
                s -> s.promptTokens());
        counter(registry, "prompt.tokens.cached", "Prompt tokens reused from the prefix cache",
                s -> s.promptTokensCached());
        gauge(registry, "prefix.cache.hit.rate", "Share of prompt tokens served from the prefix cache",
                BatchStats::prefixCacheHitRate);
        counter(registry, "prompt.seconds", "Time spent prefilling prompts", BatchStats::promptSeconds);
        counter(registry, "tokens.predicted", "Tokens sampled", s -> s.generatedTokens());
        counter(registry, "tokens.predicted.device", "Tokens sampled over a device top-k readback",
                BatchStats::deviceSampledRows);
        counter(registry, "tokens.predicted.seconds", "Time spent decoding", BatchStats::decodeSeconds);
        counter(registry, "host.sample.seconds", "Time the engine spent in the host sampler chains",
                BatchStats::hostSampleSeconds);
        counter(registry, "decode", "Batched decode forwards", s -> s.decodeSteps());
        counter(registry, "recomputed.tokens", "Tokens prefilled again after a preemption",
                s -> s.recomputedTokens());
        counter(registry, "preemptions", "Running sequences preempted to free memory", s -> s.preemptions());
        gauge(registry, "encoder.cache.rows", "Rows of shared encoder output the engine's encoder cache holds",
                s -> s.encoderCacheRows());
        counter(registry, "encoder.cache.hits", "Media chunks whose encoder output the encoder cache already held",
                s -> s.encoderCacheHits());
        counter(registry, "encoder.cache.evictions", "Encoder cache entries evicted for space",
                s -> s.encoderCacheEvictions());
        counter(registry, "spec.rounds", "Speculative verify rounds", s -> s.specRounds());
        counter(registry, "spec.drafted.tokens", "Draft tokens the target verified", s -> s.specDrafted());
        counter(registry, "spec.accepted.tokens", "Draft tokens the target confirmed", s -> s.specAccepted());
        counter(registry, "lookahead.rounds", "Lookahead (Jacobi) decoding steps", s -> s.lookaheadRounds());
        counter(registry, "lookahead.emitted.tokens", "Tokens lookahead steps emitted",
                s -> s.lookaheadEmitted());
        counter(registry, "lookahead.accepted.tokens", "Tokens lookahead steps confirmed from an observed n-gram",
                s -> s.lookaheadAccepted());
        finishCounters(registry);
        counter(registry, "request.failures", "Requests that failed", s -> s.failed());
        gauge(registry, "requests.processing", "Sequences prefilling or decoding", s -> s.running());
        gauge(registry, "requests.deferred", "Requests waiting for a slot or for memory", s -> s.waiting());
        gauge(registry, "kv.cache.usage", "Share of KV memory rows in use, cached prefixes included",
                BatchStats::kvUsage);
        gauge(registry, "tokens.max", "Longest finished sequence (prompt and generation)",
                s -> s.maxSequenceLength());
        gauge(registry, "busy.slots.per.decode", "Average sequences per decode forward", BatchStats::rowsPerDecode);
        mediaCacheMeters(registry);
        replicaGauges(registry);
    }

    /** The projector's media cache: lookups and hits (vLLM's {@code mm_cache_queries} / {@code mm_cache_hits}). */
    private void mediaCacheMeters(MeterRegistry registry) {
        FunctionCounter.builder(PREFIX + "mm.cache.queries", this, m -> m.readMedia(EngineService::mediaCacheQueries))
                .description("Media items looked up in the projector's media cache").register(registry);
        FunctionCounter.builder(PREFIX + "mm.cache.hits", this, m -> m.readMedia(EngineService::mediaCacheHits))
                .description("Media items the projector's media cache served without decoding them")
                .register(registry);
        Gauge.builder(PREFIX + "mm.cache.usage.bytes", this, m -> m.readMedia(EngineService::mediaCacheUsageBytes))
                .description("Bytes the projector's media cache holds").register(registry);
        FunctionCounter.builder(PREFIX + "mm.encoder_cache.loads", this,
                        m -> m.readMedia(EngineService::mediaSharedLoads))
                .description("Media items loaded from the shared encoder cache (vLLM's EC transfer)").register(registry);
        FunctionCounter.builder(PREFIX + "mm.encoder_cache.saves", this,
                        m -> m.readMedia(EngineService::mediaSharedSaves))
                .description("Media items saved to the shared encoder cache (vLLM's EC transfer)").register(registry);
    }

    /** {@code value} of the open service's media cache, 0 while no service is open. */
    double readMedia(ToDoubleFunction<EngineService> value) {
        EngineService service = service();
        return service == null ? 0 : value.applyAsDouble(service);
    }

    /**
     * The data-parallel layout: how many engine replicas serve the model here, the global data-parallel size
     * (vLLM's {@code data_parallel_size}), and each replica's series once the service opens.
     */
    private void replicaGauges(MeterRegistry registry) {
        Gauge.builder(PREFIX + "dp.replicas", this, m -> {
                    EngineService service = m.service();
                    return service == null ? 0 : service.replicaCount();
                })
                .description("Engine replicas serving the model here (the local data-parallel size)")
                .register(registry);
        Gauge.builder(PREFIX + "dp.size", this, m -> {
                    EngineService service = m.service();
                    return service == null ? 0 : service.dataParallelSize();
                })
                .description("The deployment's global data-parallel size").register(registry);
        if (lifecycle != null) {
            lifecycle.onOpen(service -> bindOpened(registry, service));
        }
    }

    /**
     * Registers on {@code registry} the meters that follow the opened {@code service}'s shape: each replica's series
     * and vLLM's multimodal cache counters, tagged with its served model name.
     */
    void bindOpened(MeterRegistry registry, EngineService service) {
        // a later elastic scale or rank removal replaces the pool, so the series follow its shape
        service.onPoolChange(() -> replicaSeries(registry, service));
        mediaCacheCounters(registry, service);
        service.latencies().add(new LatencyTimers(registry));
    }

    /**
     * vLLM's request-level latency histograms as Prometheus timers with vLLM's bucket bounds. Two instances over the
     * same registry are equal, so binding a service to a registry twice feeds its series once, and a reopened
     * service's listener feeds the same series.
     */
    static final class LatencyTimers implements RequestLatencies.Listener {
        private static final double[] TTFT_BUCKETS = {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
            0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0};
        private static final double[] ITL_BUCKETS = {0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
            1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0};
        private static final double[] REQUEST_BUCKETS = {0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0,
            30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0};

        private final MeterRegistry registry;
        private final Timer firstToken;
        private final Timer interToken;
        private final Timer endToEnd;
        private final Timer queue;

        LatencyTimers(MeterRegistry registry) {
            this.registry = registry;
            firstToken = timer(registry, "time.to.first.token",
                    "Time from a request's arrival to its first sampled token", TTFT_BUCKETS);
            interToken = timer(registry, "inter.token.latency",
                    "Time between two consecutive sampled tokens of a request", ITL_BUCKETS);
            endToEnd = timer(registry, "e2e.request.latency",
                    "Time from a request's arrival to its end, for requests that sampled a token", REQUEST_BUCKETS);
            queue = timer(registry, "request.queue.time",
                    "Time a request waited in the engine queue before its first admission", REQUEST_BUCKETS);
        }

        private static Timer timer(MeterRegistry registry, String name, String help, double[] bucketsSeconds) {
            Duration[] slos = new Duration[bucketsSeconds.length];
            for (int i = 0; i < slos.length; i++) {
                slos[i] = Duration.ofNanos(Math.round(bucketsSeconds[i] * 1e9));
            }
            return Timer.builder(PREFIX + name).description(help).serviceLevelObjectives(slos).register(registry);
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof LatencyTimers && ((LatencyTimers) o).registry == registry;
        }

        @Override
        public int hashCode() {
            return System.identityHashCode(registry);
        }

        @Override
        public void firstToken(long nanos) {
            firstToken.record(nanos, TimeUnit.NANOSECONDS);
        }

        @Override
        public void interToken(long nanos) {
            interToken.record(nanos, TimeUnit.NANOSECONDS);
        }

        @Override
        public void finished(long nanos) {
            endToEnd.record(nanos, TimeUnit.NANOSECONDS);
        }

        @Override
        public void queued(long nanos) {
            queue.record(nanos, TimeUnit.NANOSECONDS);
        }
    }

    /**
     * vLLM's {@code vllm:mm_cache_queries} and {@code vllm:mm_cache_hits} counters over the projector's media cache
     * of the opened {@code service} (the series {@code neuralfree_mm_cache_*} count too), tagged with its served
     * model name and engine 0.
     */
    private void mediaCacheCounters(MeterRegistry registry, EngineService service) {
        String model = service.served().primary();
        Tags tags = Tags.of("model_name", model == null ? "" : model, "engine", "0");
        FunctionCounter.builder("vllm:mm_cache_queries", this, m -> m.readMedia(EngineService::mediaCacheQueries))
                .description("Multi-modal cache queries, in terms of number of queried items.").tags(tags)
                .register(registry);
        FunctionCounter.builder("vllm:mm_cache_hits", this, m -> m.readMedia(EngineService::mediaCacheHits))
                .description("Multi-modal cache hits, in terms of number of cached items.").tags(tags)
                .register(registry);
    }

    /** One replica-tagged series set per replica of the opened {@code service}. */
    private void replicaSeries(MeterRegistry registry, EngineService service) {
        for (int i = 0; i < service.replicaCount(); i++) {
            int replica = i;
            Tags tags = Tags.of("replica", Integer.toString(i), "dp_rank", Integer.toString(service.replicaRank(i)));
            replicaGauge(registry, "replica.requests.processing", "One replica's sequences prefilling or decoding",
                    tags, m -> m.readReplica(replica, s -> s.running()));
            replicaGauge(registry, "replica.requests.deferred", "One replica's requests waiting for a slot or memory",
                    tags, m -> m.readReplica(replica, s -> s.waiting()));
            replicaGauge(registry, "replica.kv.cache.usage", "One replica's share of KV memory rows in use",
                    tags, m -> m.readReplica(replica, BatchStats::kvUsage));
            replicaGauge(registry, "replica.prompt.tokens.in.flight",
                    "The prompt tokens of one replica's requests not yet finished", tags, m -> {
                        EngineService current = m.service();
                        return current == null || replica >= current.replicaCount() ? 0
                                : current.replicaPromptTokensInFlight(replica);
                    });
        }
    }

    private void replicaGauge(MeterRegistry registry, String name, String help, Tags tags,
                              ToDoubleFunction<EngineMetrics> value) {
        Gauge.builder(PREFIX + name, this, value).description(help).tags(tags).register(registry);
    }

    /** {@code value} of replica {@code index}'s latest snapshot, or zero while no service holds that replica. */
    double readReplica(int index, ToDoubleFunction<BatchStats> value) {
        EngineService service = service();
        return value.applyAsDouble(service == null || index >= service.replicaCount() ? BatchStats.EMPTY
                : service.replicaStats(index));
    }

    /** The open service, or null while none is. */
    private EngineService service() {
        if (current != null) {
            return current.get();
        }
        return lifecycle == null ? null : lifecycle.current();
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
        FunctionCounter.builder(PREFIX + "request.success", this,
                        m -> m.read(s -> s.finished(StopReason.REPETITION)))
                .description(help).tag("finished_reason", "repetition").register(registry);
    }

    private void counter(MeterRegistry registry, String name, String help, ToDoubleFunction<BatchStats> value) {
        FunctionCounter.builder(PREFIX + name, this, m -> m.read(value)).description(help).register(registry);
    }

    private void gauge(MeterRegistry registry, String name, String help, ToDoubleFunction<BatchStats> value) {
        Gauge.builder(PREFIX + name, this, m -> m.read(value)).description(help).register(registry);
    }

    /** {@code value} of the latest snapshot, or zero while no service is open. */
    double read(ToDoubleFunction<BatchStats> value) {
        EngineService service = service();
        return value.applyAsDouble(service == null ? BatchStats.EMPTY : service.stats());
    }
}
