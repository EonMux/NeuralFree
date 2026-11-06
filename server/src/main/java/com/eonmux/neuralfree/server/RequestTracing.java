package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanBuilder;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.propagation.TextMapGetter;
import io.vertx.core.MultiMap;
import io.vertx.core.http.HttpServerRequest;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.logging.Logger;

/**
 * The per-request OpenTelemetry trace (vLLM's {@code vllm/tracing/otel.py}, opt-in through
 * {@code --otlp-traces-endpoint}): when {@link ServerOptions#otlpTracesEndpoint()} is set, every finished
 * {@code GenerationBatch} emits one {@code llm_request} span of {@link SpanKind#SERVER} through the global
 * tracer — the SDK Quarkus exports to the configured OTLP endpoint — carrying the {@code gen_ai.*} engine
 * attributes: the token counts, the queue, time-to-first-token, e2e and model-phase latencies, and the request's
 * sampler parameters. The span starts at the request's arrival and ends when its last choice reaches its terminal
 * state, both as explicit epoch timestamps (vLLM's {@code manual_instrument_otel}, which measures the whole
 * request after the fact rather than holding a span across threads). The arriving request's W3C
 * {@code traceparent} / {@code tracestate} headers, when present, propagate the caller's trace: the span joins it
 * as their child. A request that arrives with trace headers while tracing is off warns once, as vLLM's does. With
 * tracing off, or an OpenTelemetry SDK that exports nothing, every span is the API's no-op and the cost is a
 * handful of counters.
 */
final class RequestTracing {

    private static final Logger LOGGER = Logger.getLogger(RequestTracing.class.getName());

    /** The tracer's instrumentation scope, named after the module that owns the spans. */
    private static final String INSTRUMENTATION = "com.eonmux.neuralfree.server";

    /** The span every generation request emits (vLLM's span name). */
    private static final String SPAN_NAME = "llm_request";

    private static final AttributeKey<Long> PROMPT_TOKENS = AttributeKey.longKey("gen_ai.usage.prompt_tokens");
    private static final AttributeKey<Long> COMPLETION_TOKENS =
            AttributeKey.longKey("gen_ai.usage.completion_tokens");
    private static final AttributeKey<Long> MAX_TOKENS = AttributeKey.longKey("gen_ai.request.max_tokens");
    private static final AttributeKey<Double> TOP_P = AttributeKey.doubleKey("gen_ai.request.top_p");
    private static final AttributeKey<Double> TEMPERATURE = AttributeKey.doubleKey("gen_ai.request.temperature");
    private static final AttributeKey<String> RESPONSE_MODEL = AttributeKey.stringKey("gen_ai.response.model");
    private static final AttributeKey<String> REQUEST_ID = AttributeKey.stringKey("gen_ai.request.id");
    private static final AttributeKey<Long> N = AttributeKey.longKey("gen_ai.request.n");
    private static final AttributeKey<Double> TIME_IN_QUEUE = AttributeKey.doubleKey("gen_ai.latency.time_in_queue");
    private static final AttributeKey<Double> TIME_TO_FIRST_TOKEN =
            AttributeKey.doubleKey("gen_ai.latency.time_to_first_token");
    private static final AttributeKey<Double> E2E = AttributeKey.doubleKey("gen_ai.latency.e2e");
    private static final AttributeKey<Double> MODEL_PREFILL =
            AttributeKey.doubleKey("gen_ai.latency.time_in_model_prefill");
    private static final AttributeKey<Double> MODEL_DECODE =
            AttributeKey.doubleKey("gen_ai.latency.time_in_model_decode");
    private static final AttributeKey<Double> MODEL_INFERENCE =
            AttributeKey.doubleKey("gen_ai.latency.time_in_model_inference");

    /** An unset timestamp slot, of a choice that never admitted, never sampled or never finished. */
    private static final long NO_TIME = Long.MIN_VALUE;

    /** The {@code System.nanoTime()} and epoch instants the nanos-to-epoch conversion anchors at. */
    private static final long BASE_NANOS = System.nanoTime();
    private static final long BASE_EPOCH_NANOS = System.currentTimeMillis() * 1_000_000L;

    /** The one warning per process for trace-context requests while tracing is off. */
    private static final AtomicBoolean WARNED_DISABLED = new AtomicBoolean();

    /** The trace headers, read case-insensitively out of the arriving request's headers (the W3C carrier). */
    private static final TextMapGetter<MultiMap> TRACE_HEADERS = new TextMapGetter<>() {
        @Override
        public Iterable<String> keys(MultiMap carrier) {
            return carrier.names();
        }

        @Override
        public String get(MultiMap carrier, String key) {
            return carrier == null ? null : carrier.get(key);
        }
    };

    private RequestTracing() {
    }

    /**
     * The trace of one {@code GenerationBatch} over {@code choices} (its shared {@code request} names the sampler
     * parameters the span reports): null unless {@link ServerOptions#otlpTracesEndpoint()} is set. A batch that
     * arrives with trace context while tracing is off warns once.
     */
    static Trace begin(EngineService service, String requestId, HttpServerRequest client, int choices,
                       GenerationRequest request) {
        MultiMap headers = client == null ? null : client.headers();
        if (service.options().otlpTracesEndpoint() == null) {
            if (hasTraceContext(headers)) {
                warnDisabledOnce();
            }
            return null;
        }
        return new Trace(GlobalOpenTelemetry.getTracer(INSTRUMENTATION), requestId, service.served().primary(),
                extractParent(headers), request, System.nanoTime(), choices);
    }

    /** The first request with trace headers while tracing is off logs the warning; later ones stay quiet. */
    private static void warnDisabledOnce() {
        if (WARNED_DISABLED.compareAndSet(false, true)) {
            LOGGER.warning("Received a request with trace context but tracing is disabled; start the server with "
                    + ServerOptions.OTLP_TRACES_ENDPOINT_FLAG + " to export request spans");
        }
    }

    /** Clears the once-flag ({@code RequestTracingTest} isolates its warning case with it). */
    static void resetWarnings() {
        WARNED_DISABLED.set(false);
    }

    /** Whether the request carries W3C trace context. */
    private static boolean hasTraceContext(MultiMap headers) {
        return headers != null && (headers.contains("traceparent") || headers.contains("tracestate"));
    }

    /** The caller's W3C trace context, or null when the request carries no trace headers. */
    private static Context extractParent(MultiMap headers) {
        if (!hasTraceContext(headers)) {
            return null;
        }
        return W3CTraceContextPropagator.getInstance().extract(Context.root(), headers, TRACE_HEADERS);
    }

    /** The epoch nanos of a {@code System.nanoTime()} reading, on one monotone scale anchored at class load. */
    private static long epochNanos(long nanos) {
        return BASE_EPOCH_NANOS + (nanos - BASE_NANOS);
    }

    /** The smaller of a running minimum ({@code NO_TIME} so far) and a candidate. */
    private static long smaller(long minimum, long candidate) {
        return minimum == NO_TIME || candidate < minimum ? candidate : minimum;
    }

    /** Puts the seconds {@code nanos} spans, skipping it while unset. */
    private static void putSeconds(AttributesBuilder attrs, AttributeKey<Double> key, long nanos) {
        if (nanos != NO_TIME) {
            attrs.put(key, nanos / 1e9);
        }
    }

    /**
     * One batch's span being measured: the arrival instant, each choice's first-token instant (the engine
     * threads), its prompt size (the request thread, before the submit) and its terminal instant (the finishing
     * engine thread). The engine handles themselves stay the batch's — they are registered by the replicas'
     * pump threads after the submit returns — and the batch hands them over at {@link #finish}. {@link #finish}
     * runs once the last choice reached its terminal state — or right after a start that could not submit — and
     * emits the span with the instants projected onto the epoch scale. Fixed-size arrays; each slot is written
     * by one thread and read after every writer is done.
     */
    static final class Trace {

        private final Tracer tracer;
        private final String requestId;
        private final String model;
        private final Context parent;
        private final GenerationRequest request;
        private final long arrivalNanos;
        private final int choices;
        private final AtomicBoolean finished = new AtomicBoolean();
        /** {@code NO_TIME} until the choice's first sampled token (the earliest stick of each slot). */
        private final AtomicLongArray firstTokenNanos;
        /** {@code NO_TIME} until the choice's terminal state (written exactly once, by its own finisher). */
        private final AtomicLongArray terminalNanos;
        /** The prompt sizes, written by the request thread before the submit offers the job. */
        private final long[] promptTokens;

        Trace(Tracer tracer, String requestId, String model, Context parent, GenerationRequest request,
              long arrivalNanos, int choices) {
            this.tracer = tracer;
            this.requestId = requestId;
            this.model = model;
            this.parent = parent;
            this.request = request;
            this.arrivalNanos = arrivalNanos;
            this.choices = choices;
            firstTokenNanos = new AtomicLongArray(choices);
            terminalNanos = new AtomicLongArray(choices);
            promptTokens = new long[choices];
            for (int i = 0; i < choices; i++) {
                firstTokenNanos.set(i, NO_TIME);
                terminalNanos.set(i, NO_TIME);
            }
        }

        /**
         * Records choice {@code choice}'s prompt size at submit. Called on the request thread, before the job
         * is offered to a replica, so every finisher observes it through the queue's handoff.
         */
        void registered(int choice, long promptTokens) {
            this.promptTokens[choice] = promptTokens;
        }

        /** Records choice {@code choice}'s first sampled token (the engine thread; later tokens lose the CAS). */
        void stepped(int choice) {
            long now = System.nanoTime();
            long seen = firstTokenNanos.get(choice);
            while (seen == NO_TIME) {
                if (firstTokenNanos.compareAndSet(choice, NO_TIME, now)) {
                    return;
                }
                seen = firstTokenNanos.get(choice);
            }
        }

        /** Records choice {@code choice}'s terminal instant (the finishing engine thread, exactly once). */
        void done(int choice) {
            terminalNanos.set(choice, System.nanoTime());
        }

        /**
         * Emits the span, once. {@code sequences} are the choices' engine handles and {@code refusals} their
         * submit refusals, as the batch reads them off its submissions now (a refused choice has no handle);
         * {@code forced} is the failure that ended the whole batch before its choices reported (a start
         * refusal), or null to read the failure off the choices. Runs on the last finishing engine thread, or
         * on the request thread right after a start that could not submit.
         */
        void finish(Throwable forced, BatchSequence[] sequences, Throwable[] refusals) {
            if (!finished.compareAndSet(false, true)) {
                return;
            }
            long endNanos = System.nanoTime();
            Span span = span();
            span.setAllAttributes(attributes(endNanos, sequences));
            Throwable failure = forced != null ? forced : failure(sequences, refusals);
            if (failure != null) {
                span.setStatus(StatusCode.ERROR);
                span.recordException(failure);
            }
            span.end(epochNanos(endNanos), TimeUnit.NANOSECONDS);
        }

        /** The {@code llm_request} server span, starting at the arrival, child of the caller's trace context. */
        private Span span() {
            SpanBuilder builder = tracer.spanBuilder(SPAN_NAME)
                    .setSpanKind(SpanKind.SERVER)
                    .setStartTimestamp(epochNanos(arrivalNanos), TimeUnit.NANOSECONDS);
            if (parent != null) {
                builder.setParent(parent);
            }
            return builder.startSpan();
        }

        /** The {@code gen_ai.*} attributes of the finished batch (vLLM's {@code do_tracing} set). */
        private Attributes attributes(long endNanos, BatchSequence[] sequences) {
            AttributesBuilder attrs = Attributes.builder();
            attrs.put(E2E, (endNanos - arrivalNanos) / 1e9);
            putSeconds(attrs, TIME_IN_QUEUE, minQueue(sequences));
            putSeconds(attrs, TIME_TO_FIRST_TOKEN, minFirstToken());
            putSeconds(attrs, MODEL_PREFILL, minPrefill(sequences));
            putSeconds(attrs, MODEL_DECODE, minDecode());
            putSeconds(attrs, MODEL_INFERENCE, minInference(sequences));
            attrs.put(PROMPT_TOKENS, sumPromptTokens());
            attrs.put(COMPLETION_TOKENS, sumCompletionTokens(sequences));
            if (requestId != null) {
                attrs.put(REQUEST_ID, requestId);
            }
            if (model != null) {
                attrs.put(RESPONSE_MODEL, model);
            }
            parameters(attrs);
            return attrs.build();
        }

        /**
         * The request's sampler parameters, set when the request named them (vLLM's truthiness conditions: a
         * parameter the caller left to the model's defaults stays off the span); {@code n} always reports.
         */
        private void parameters(AttributesBuilder attrs) {
            if (choices > 0) {
                attrs.put(N, (long) choices);
            }
            if (request == null) {
                return;
            }
            SamplerConfig sampler = request.sampler();
            if (sampler.explicit().contains(SamplerConfig.Setting.TOP_P)) {
                attrs.put(TOP_P, (double) sampler.topP());
            }
            if (sampler.explicit().contains(SamplerConfig.Setting.TEMPERATURE)) {
                attrs.put(TEMPERATURE, (double) sampler.temperature());
            }
            if (request.maxNewTokensExplicit()) {
                attrs.put(MAX_TOKENS, (long) request.maxNewTokens());
            }
        }

        /** The earliest queue wait: how long the first admission of any choice took (seconds). */
        private long minQueue(BatchSequence[] sequences) {
            long minimum = NO_TIME;
            for (int i = 0; i < choices; i++) {
                long admitted = admittedAt(sequences, i);
                if (admitted != BatchSequence.NEVER_ADMITTED) {
                    minimum = smaller(minimum, admitted - arrivalNanos);
                }
            }
            return minimum;
        }

        /** Choice {@code i}'s engine handle's admission instant, {@code NEVER_ADMITTED} without a handle. */
        private static long admittedAt(BatchSequence[] sequences, int choice) {
            BatchSequence sequence = sequences[choice];
            return sequence == null ? BatchSequence.NEVER_ADMITTED : sequence.admittedAtNanos();
        }

        /** The earliest time to first token over the choices (vLLM's {@code first_token_latency}). */
        private long minFirstToken() {
            long minimum = NO_TIME;
            for (int i = 0; i < choices; i++) {
                long first = firstTokenNanos.get(i);
                if (first != NO_TIME) {
                    minimum = smaller(minimum, first - arrivalNanos);
                }
            }
            return minimum;
        }

        /** The earliest prefill: first token past admission, over the choices that show both. */
        private long minPrefill(BatchSequence[] sequences) {
            long minimum = NO_TIME;
            for (int i = 0; i < choices; i++) {
                long admitted = admittedAt(sequences, i);
                long first = firstTokenNanos.get(i);
                if (admitted != BatchSequence.NEVER_ADMITTED && first != NO_TIME) {
                    minimum = smaller(minimum, first - admitted);
                }
            }
            return minimum;
        }

        /** The earliest decode: terminal past first token, over the choices that show both. */
        private long minDecode() {
            long minimum = NO_TIME;
            for (int i = 0; i < choices; i++) {
                long first = firstTokenNanos.get(i);
                long terminal = terminalNanos.get(i);
                if (first != NO_TIME && terminal != NO_TIME) {
                    minimum = smaller(minimum, terminal - first);
                }
            }
            return minimum;
        }

        /** The earliest model inference: terminal past admission, over the choices that show both. */
        private long minInference(BatchSequence[] sequences) {
            long minimum = NO_TIME;
            for (int i = 0; i < choices; i++) {
                long admitted = admittedAt(sequences, i);
                long terminal = terminalNanos.get(i);
                if (admitted != BatchSequence.NEVER_ADMITTED && terminal != NO_TIME) {
                    minimum = smaller(minimum, terminal - admitted);
                }
            }
            return minimum;
        }

        /** The prompt tokens of every submitted choice, refused ones included (they carry a prompt too). */
        private long sumPromptTokens() {
            long sum = 0;
            for (long tokens : promptTokens) {
                sum += tokens;
            }
            return sum;
        }

        /** The generated tokens over every choice that finished with a result (a failure reports none). */
        private long sumCompletionTokens(BatchSequence[] sequences) {
            long sum = 0;
            for (int i = 0; i < choices; i++) {
                BatchSequence sequence = sequences[i];
                if (sequence != null && sequence.result() != null) {
                    sum += sequence.result().timings().generatedTokens();
                }
            }
            return sum;
        }

        /** The first choice failure: a submit refusal, else an engine failure; null when every choice served. */
        private Throwable failure(BatchSequence[] sequences, Throwable[] refusals) {
            for (Throwable refused : refusals) {
                if (refused != null) {
                    return refused;
                }
            }
            for (BatchSequence sequence : sequences) {
                if (sequence != null && sequence.state() == BatchSequence.State.FAILED) {
                    return sequence.error();
                }
            }
            return null;
        }
    }
}
