package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import java.io.IOException;
import java.util.Map;

/**
 * The transport that carries a prefill instance's KV chain to a decode instance (vLLM's {@code KVConnectorBase_V1}
 * reduced to the one exchange a block-chain transfer needs).
 *
 * <p>{@link #save} stores a payload under its transfer id for a consumer to pull ({@link #load}); it runs on the
 * engine's thread and must not block on the network. {@link #load} runs on the decode instance's HTTP thread and
 * pulls the payload the prefill instance saved, validating the engine id. {@link #endpoint} is what the prefill
 * instance's {@code kv_transfer_params} advertise. A connector that cannot provide a direction refuses by name in
 * {@link #save} or {@link #load}, so a misconfigured role fails loudly.</p>
 *
 * <p>Implemented by {@link TcpKvConnector} (a socket pull/push transport) and {@link FileKvConnector} (shared
 * storage). Not thread-safe by contract beyond what each implementation documents; the engine calls
 * {@link #save} from one thread at a time.</p>
 */
public interface KvConnector extends AutoCloseable {

    /** The connector's configured name ({@code TcpKvConnector}, {@code FileKvConnector}). */
    String name();

    /** This instance's engine id. */
    String engineId();

    /** This instance's role. */
    KvRole role();

    /** The address a consumer pulls from, or {@link KvTransferEndpoint#none()} for shared storage. */
    KvTransferEndpoint endpoint();

    /**
     * Stores {@code payload} for consumers to pull.
     *
     * @throws UnsupportedOperationException when this instance's role does not export KV
     * @throws IOException                   when the payload cannot be stored
     */
    void save(KvTransferPayload payload) throws IOException;

    /**
     * Pulls the payload saved under {@code transferId}.
     *
     * @param expectedEngineId the exporting engine id the request named, or null to accept any
     * @param remote           the exporting instance's address, null for shared storage
     * @throws UnsupportedOperationException when this instance's role does not import KV
     * @throws IOException                   when the transfer fails or the exporting engine differs from
     *                                       {@code expectedEngineId}
     */
    KvTransferPayload load(String transferId, String expectedEngineId, KvTransferEndpoint remote) throws IOException;

    /**
     * Drops a stored payload no consumer will pull (the consumer's completion notice); an unknown id is ignored.
     */
    default void release(String transferId) {
    }

    /** The connector's counters and settings, for the status endpoint. */
    Map<String, Object> stats();

    /** Releases the connector's listener, sockets and stored payloads. */
    @Override
    void close();
}
