package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvConnector;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvConnectors;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferEndpoint;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferParams;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferPayload;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Disaggregated prefill/decode over the server's engine pool: the producer side exports a request's prefill rows
 * through the configured {@link KvConnector} after the sequence retires, and the consumer side pulls a transferred
 * chain before submitting the decode request, importing it into the replica that will run it.
 *
 * <p>The server-level protocol is vLLM's: a request carrying {@code kv_transfer_params} with
 * {@code do_remote_decode: true} runs on the prefill instance, which forces a one-token generation, exports the
 * prompt's cached chain when the sequence finishes and answers the request's {@code kv_transfer_params} with the
 * connector's address and the transfer id. The router forwards those parameters with the original request to the
 * decode instance, which pulls the chain ({@code do_remote_prefill: true}), imports it into its prefix cache and
 * decodes from the prompt's last token — its first token is sampled locally, as the reference proxy's decode
 * instance does. The row bytes are the slot file format, so the two instances must run the same model with the same
 * page size, KV group layout and cache dtypes; a foreign chain refuses naming the field that differs.
 *
 * <p>The service is created beside the replicas when {@code neuralfree.server.kv.transfer.config} is set and closed
 * with the service. The export runs on the replica's pump thread at the sequence's terminal state; the pull runs on
 * the request's HTTP thread. See {@link KvTransferConfig} for the settings.
 */
final class KvTransferService implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(KvTransferService.class.getName());
    /** How long a request's export state stays after it finished (the response is built within seconds). */
    private static final long EXPORT_TTL_NANOS = java.util.concurrent.TimeUnit.HOURS.toNanos(1);

    /** One request's export: whether its payload reached the connector, its token count, or its failure. */
    private static final class Export {
        private final long createdAt = System.nanoTime();
        private boolean stored;
        private int tokens;
        private Throwable failure;

        synchronized void stored(int tokens) {
            this.stored = true;
            this.tokens = tokens;
        }

        synchronized void failed(Throwable failure) {
            this.failure = failure;
        }

        synchronized boolean stored() {
            return stored;
        }

        synchronized Throwable failure() {
            return failure;
        }

        synchronized int tokens() {
            return tokens;
        }
    }

    private final KvTransferConfig config;
    private final KvConnector connector;
    /** The producer-side exports of this instance, by transfer id. */
    private final ConcurrentHashMap<String, Export> exports = new ConcurrentHashMap<>();

    private KvTransferService(KvTransferConfig config, KvConnector connector) {
        this.config = Objects.requireNonNull(config, "config");
        this.connector = Objects.requireNonNull(connector, "connector");
    }

    /**
     * The service of {@code config}, or null when no configuration is given.
     *
     * @throws IOException                   when the connector cannot start (a port conflict, an unreadable storage
     *                                       directory)
     * @throws UnsupportedOperationException when the connector is one this build does not ship, named with the reason
     */
    static KvTransferService open(KvTransferConfig config) throws IOException {
        if (config == null) {
            return null;
        }
        return new KvTransferService(config, KvConnectors.create(config));
    }

    /** The configured transfer settings. */
    KvTransferConfig config() {
        return config;
    }

    /** The connector moving chains between instances. */
    KvConnector connector() {
        return connector;
    }

    /** Whether this instance exports prefill KV. */
    boolean produces() {
        return config.produces();
    }

    /** Whether this instance imports prefill KV. */
    boolean consumes() {
        return config.consumes();
    }

    /**
     * The prefill side of a request: validates that this instance produces KV, assigns the transfer id when the
     * request names none and registers the export the sequence's retirement fills.
     *
     * @param params the request's {@code kv_transfer_params}
     * @param prompt the request's prompt tokens
     * @return the assigned parameters, naming the transfer id the export is stored under
     * @throws IllegalArgumentException when this instance's role does not export KV, or the transfer id is taken
     */
    KvTransferParams producerParams(KvTransferParams params, int[] prompt) {
        if (!config.produces()) {
            throw new IllegalArgumentException("kv_transfer_params asks this instance to prefill a request, but its"
                    + " kv_role is " + config.role().wireName());
        }
        String transferId = params.transferId() == null ? UUID.randomUUID().toString() : params.transferId();
        sweepExports();
        Export started = new Export();
        if (exports.putIfAbsent(transferId, started) != null) {
            throw new IllegalArgumentException("kv_transfer_params transfer_id " + transferId + " is already used"
                    + " by another request; name a unique transfer id per request");
        }
        KvTransferEndpoint endpoint = connector.endpoint();
        return params.withExport(transferId, endpoint.addressed() ? endpoint.host() : null,
                endpoint.addressed() ? endpoint.port() : -1, prompt.length);
    }

    /**
     * The terminal hook of a prefill request: exports the retired sequence's cached rows and stores them under the
     * transfer id. Runs on the replica's pump thread, after the sequence released its slot and before its handle
     * reports the terminal state, so the slot's history cannot be reused first.
     *
     * @param assigned the parameters {@link #producerParams} answered with
     * @param replica  the replica that ran the request
     * @param prompt   the request's prompt tokens
     */
    Consumer<BatchSequence> exporter(KvTransferParams assigned, EngineReplica replica, int[] prompt) {
        String transferId = assigned.transferId();
        Export export = exports.get(transferId);
        return sequence -> {
            if (export == null) {
                return;
            }
            try {
                if (sequence == null || sequence.state() != BatchSequence.State.DONE || sequence.seq() < 0) {
                    return;
                }
                BatchEngine engine = replica.builtEngine();
                if (engine == null) {
                    return;
                }
                byte[] chain = engine.exportKv(sequence.seq());
                if (chain.length == 0) {
                    return;
                }
                KvTransferPayload payload = new KvTransferPayload(transferId, connector.engineId(), prompt, chain);
                connector.save(payload);
                export.stored(prompt.length);
            } catch (IOException | RuntimeException e) {
                LOGGER.log(Level.WARNING, "KV transfer " + transferId + " export failed", e);
                export.failed(e);
            }
        };
    }

    /**
     * The prefill request's response parameters, read when the response is built (the export hook has run by then).
     *
     * @return the prefill instance's parameters, or null when the sequence exported nothing (cancelled or failed)
     * @throws IllegalStateException when the export failed (the answer must not advertise a transfer that is not
     *                               there)
     */
    Map<String, Object> responseParams(String transferId) {
        Export export = exports.get(transferId);
        if (export == null) {
            return null;
        }
        Throwable failure = export.failure();
        if (failure != null) {
            throw new IllegalStateException("KV transfer " + transferId + " failed: " + failure.getMessage(), failure);
        }
        if (!export.stored()) {
            return null;
        }
        KvTransferEndpoint endpoint = connector.endpoint();
        KvTransferParams params = KvTransferParams.exported(config.engineId(), null,
                endpoint.addressed() ? endpoint.host() : null, endpoint.addressed() ? endpoint.port() : -1,
                transferId, export.tokens());
        return params.toMap();
    }

    /**
     * The decode side of a request: validates that this instance consumes KV, pulls the named chain and returns it
     * for import into the replica that will decode. A failed pull follows
     * {@code kv_load_failure_policy}: {@code fail} refuses the request, {@code recompute} returns null so the
     * request prefills locally.
     *
     * @return the chain to import, or null when the request should recompute
     * @throws IllegalArgumentException when this instance's role does not import KV, or the chain does not extend
     *                                  the request's prompt
     * @throws IOException              when the pull fails and the policy is {@code fail}
     */
    byte[] importFor(KvTransferParams params, int[] prompt) throws IOException {
        if (!config.consumes()) {
            throw new IllegalArgumentException("kv_transfer_params asks this instance to decode a prefill's KV, but"
                    + " its kv_role is " + config.role().wireName());
        }
        KvTransferEndpoint remote = params.remoteHost() == null || params.remotePort() <= 0
                ? null : KvTransferEndpoint.of(params.remoteHost(), params.remotePort());
        KvTransferPayload payload;
        try {
            payload = connector.load(params.transferId(), params.remoteEngineId(), remote);
        } catch (IOException e) {
            if (config.loadFailurePolicy().equals("recompute")) {
                LOGGER.log(Level.WARNING, "KV transfer " + params.transferId() + " failed; recomputing the prompt", e);
                return null;
            }
            throw e;
        }
        if (!payload.covers(prompt)) {
            throw new IllegalArgumentException("the transferred chain of " + params.transferId() + " holds "
                    + payload.promptTokens().length + " prompt tokens that do not start the request's new prompt");
        }
        return payload.chain();
    }

    /**
     * The decode request's response parameters: the transfer is complete, so nothing remains to pull.
     *
     * @param params the request's parameters
     * @param imported whether the chain was imported (false when the request recomputed)
     */
    Map<String, Object> consumerParams(KvTransferParams params, boolean imported) {
        if (!imported) {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("do_remote_prefill", false);
            out.put("do_remote_decode", false);
            if (params.transferId() != null) {
                out.put("transfer_id", params.transferId());
            }
            return out;
        }
        return KvTransferParams.consumed(params.transferId(), params.remoteNumTokens()).toMap();
    }

    /** The role, connector and counters, for the status endpoint. */
    Map<String, Object> status() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("kv_role", config.role().wireName());
        out.put("engine_id", config.engineId());
        out.put("kv_connector", config.kvConnector());
        out.put("kv_rank", config.kvRank());
        out.put("kv_parallel_size", config.kvParallelSize());
        out.put("connector", connector.stats());
        out.put("transfers", exports.size());
        return out;
    }

    /** Drops the exports whose response was built long ago, so the map does not grow without bound. */
    private void sweepExports() {
        long now = System.nanoTime();
        exports.entrySet().removeIf(entry -> now - entry.getValue().createdAt > EXPORT_TTL_NANOS);
    }

    @Override
    public void close() {
        exports.clear();
        connector.close();
    }

    @Override
    public String toString() {
        return "KvTransferService{" + config.kvConnector() + ", " + config.role().wireName() + ", engine "
                + config.engineId() + "}";
    }
}
