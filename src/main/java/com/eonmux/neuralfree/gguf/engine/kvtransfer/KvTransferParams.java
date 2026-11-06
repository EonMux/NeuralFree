package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * One request's {@code kv_transfer_params} (vLLM's request field of the same name): which side of a disaggregated
 * transfer it runs on and, on the decode side, where the prefill instance's KV cache is.
 *
 * <p>The prefill instance receives {@code do_remote_decode: true} and answers the transfer's parameters —
 * {@code do_remote_prefill: true}, {@code remote_engine_id}, {@code remote_host}, {@code remote_port},
 * {@code remote_request_id}, {@code remote_num_tokens} and {@code transfer_id} — which the router forwards with the
 * original request to the decode instance. The decode instance receives {@code do_remote_prefill: true} and those
 * fields, pulls the chain and decodes; it answers {@code do_remote_prefill: false}. vLLM's
 * {@code remote_block_ids} names vLLM block ids, which this connector does not exchange (a chain moves as one
 * payload); it is refused by name rather than ignored. Immutable.
 */
public final class KvTransferParams {

    /** The per-request field carrying these parameters. */
    public static final String FIELD = "kv_transfer_params";

    private static final Set<String> FIELDS = fields();

    private final boolean doRemotePrefill;
    private final boolean doRemoteDecode;
    private final String remoteEngineId;
    private final String remoteRequestId;
    private final String remoteHost;
    private final int remotePort;
    private final String transferId;
    private final int remoteNumTokens;

    private KvTransferParams(boolean doRemotePrefill, boolean doRemoteDecode, String remoteEngineId,
                             String remoteRequestId, String remoteHost, int remotePort, String transferId,
                             int remoteNumTokens) {
        this.doRemotePrefill = doRemotePrefill;
        this.doRemoteDecode = doRemoteDecode;
        this.remoteEngineId = remoteEngineId;
        this.remoteRequestId = remoteRequestId;
        this.remoteHost = remoteHost;
        this.remotePort = remotePort;
        this.transferId = transferId;
        this.remoteNumTokens = remoteNumTokens;
    }

    private static Set<String> fields() {
        Set<String> out = new LinkedHashSet<>();
        out.add("do_remote_prefill");
        out.add("do_remote_decode");
        out.add("remote_engine_id");
        out.add("remote_request_id");
        out.add("remote_host");
        out.add("remote_port");
        out.add("remote_num_tokens");
        out.add("transfer_id");
        return java.util.Collections.unmodifiableSet(out);
    }

    /** The prefill side's request: {@code do_remote_decode} set, every remote field empty. */
    public static KvTransferParams prefillRequest() {
        return new KvTransferParams(false, true, null, null, null, -1, null, -1);
    }

    /** The decode side's request: {@code do_remote_prefill} set and the prefill instance's location named. */
    public static KvTransferParams decodeRequest(String remoteEngineId, String remoteRequestId, String remoteHost,
                                                 int remotePort, String transferId, int remoteNumTokens) {
        if (transferId == null || transferId.isBlank()) {
            throw new IllegalArgumentException("do_remote_prefill needs transfer_id");
        }
        return new KvTransferParams(true, false, remoteEngineId, remoteRequestId,
                remoteHost == null || remoteHost.isBlank() ? null : remoteHost.trim(), remotePort, transferId,
                remoteNumTokens);
    }

    /** The parameters the prefill instance answers with once its chain is exported. */
    public static KvTransferParams exported(String remoteEngineId, String remoteRequestId, String remoteHost,
                                            int remotePort, String transferId, int remoteNumTokens) {
        return new KvTransferParams(true, false, remoteEngineId, remoteRequestId, remoteHost, remotePort,
                transferId, remoteNumTokens);
    }

    /** The decode instance's answer: the transfer happened, nothing remains to pull. */
    public static KvTransferParams consumed(String transferId, int remoteNumTokens) {
        return new KvTransferParams(false, false, null, null, null, -1, transferId, remoteNumTokens);
    }

    /**
     * The parameters a request's {@code kv_transfer_params} object holds.
     *
     * @throws IllegalArgumentException naming an unknown or mistyped field, both directions set at once, or a
     *                                  decode request without its remote location
     */
    public static KvTransferParams parse(JsonView view) {
        for (String key : view.keys()) {
            if (!FIELDS.contains(key)) {
                String hint = key.equals("remote_block_ids")
                        ? ": this connector transfers a whole block chain under transfer_id, not vLLM block ids"
                        : "";
                throw new IllegalArgumentException(view.where(key) + " is not a supported kv_transfer_params field"
                        + hint);
            }
        }
        boolean remotePrefill = view.optBool("do_remote_prefill", false);
        boolean remoteDecode = view.optBool("do_remote_decode", false);
        if (remotePrefill && remoteDecode) {
            throw new IllegalArgumentException(at(view) + " sets both do_remote_prefill and do_remote_decode;"
                    + " a request runs on one side of the transfer");
        }
        if (!remotePrefill && !remoteDecode) {
            throw new IllegalArgumentException(at(view) + " sets neither do_remote_prefill nor"
                    + " do_remote_decode; name the side the request runs on");
        }
        String engineId = optionalText(view, "remote_engine_id");
        String requestId = optionalText(view, "remote_request_id");
        String host = optionalText(view, "remote_host");
        int port = view.optI32("remote_port", -1);
        String transferId = optionalText(view, "transfer_id");
        int numTokens = view.optI32("remote_num_tokens", -1);
        if (numTokens < -1) {
            throw new IllegalArgumentException(view.where("remote_num_tokens") + " must be 0 or more");
        }
        if (remotePrefill) {
            if (transferId == null) {
                throw new IllegalArgumentException(at(view) + " needs transfer_id for do_remote_prefill");
            }
            if (port != -1 && (port < 1 || port > 65535)) {
                throw new IllegalArgumentException(view.where("remote_port")
                        + " must be in [1, 65535] for do_remote_prefill, got " + port);
            }
        }
        return new KvTransferParams(remotePrefill, remoteDecode, engineId, requestId, host, port, transferId,
                numTokens);
    }

    /** The field's location in the request, naming it in a refusal. */
    private static String at(JsonView view) {
        return view.path() == null || view.path().isEmpty() ? view.source() : view.source() + ":" + view.path();
    }

    private static String optionalText(JsonView view, String key) {
        Object raw = view.raw(key);
        if (raw == null) {
            return null;
        }
        if (!(raw instanceof String)) {
            throw new IllegalArgumentException(view.where(key) + " must be a string");
        }
        String text = ((String) raw).trim();
        return text.isEmpty() ? null : text;
    }

    /** Whether this request runs on the prefill (producer) side. */
    public boolean doRemotePrefill() {
        return doRemotePrefill;
    }

    /** Whether this request asks the instance to export its prompt's KV (the prefill side). */
    public boolean doRemoteDecode() {
        return doRemoteDecode;
    }

    /** The prefill instance's engine id, or null. */
    public String remoteEngineId() {
        return remoteEngineId;
    }

    /** The request id the prefill instance served the same prompt under, or null. */
    public String remoteRequestId() {
        return remoteRequestId;
    }

    /** The prefill connector's host, or null. */
    public String remoteHost() {
        return remoteHost;
    }

    /** The prefill connector's port, or -1. */
    public int remotePort() {
        return remotePort;
    }

    /** The transfer id both instances name, or null on a prefill request (the instance assigns one). */
    public String transferId() {
        return transferId;
    }

    /** The tokens the prefill exported, or -1 when unknown. */
    public int remoteNumTokens() {
        return remoteNumTokens;
    }

    /**
     * A copy with the export's {@code transferId}, {@code remoteHost} and {@code remotePort}; the request's direction
     * flags stay as sent. The prefill instance's answer is {@link #exported}.
     */
    public KvTransferParams withExport(String transferId, String remoteHost, int remotePort, int tokens) {
        return new KvTransferParams(doRemotePrefill, doRemoteDecode, remoteEngineId, remoteRequestId, remoteHost,
                remotePort, transferId, tokens);
    }

    /**
     * The parameters as the response's {@code kv_transfer_params} object: every non-null field under vLLM's name,
     * so a router can forward the prefill instance's answer unchanged to the decode instance.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("do_remote_prefill", doRemotePrefill);
        out.put("do_remote_decode", doRemoteDecode);
        if (remoteEngineId != null) {
            out.put("remote_engine_id", remoteEngineId);
        }
        if (remoteRequestId != null) {
            out.put("remote_request_id", remoteRequestId);
        }
        if (remoteHost != null) {
            out.put("remote_host", remoteHost);
        }
        if (remotePort > 0) {
            out.put("remote_port", remotePort);
        }
        if (transferId != null) {
            out.put("transfer_id", transferId);
        }
        if (remoteNumTokens >= 0) {
            out.put("remote_num_tokens", remoteNumTokens);
        }
        return out;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof KvTransferParams)) {
            return false;
        }
        KvTransferParams other = (KvTransferParams) o;
        return doRemotePrefill == other.doRemotePrefill && doRemoteDecode == other.doRemoteDecode
                && remotePort == other.remotePort && remoteNumTokens == other.remoteNumTokens
                && Objects.equals(remoteEngineId, other.remoteEngineId)
                && Objects.equals(remoteRequestId, other.remoteRequestId)
                && Objects.equals(remoteHost, other.remoteHost)
                && Objects.equals(transferId, other.transferId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(doRemotePrefill, doRemoteDecode, remoteEngineId, remoteRequestId, remoteHost,
                remotePort, transferId, remoteNumTokens);
    }

    @Override
    public String toString() {
        return "KvTransferParams{" + (doRemotePrefill ? "prefill" : "decode") + ", transfer " + transferId
                + ", remote " + remoteHost + ":" + remotePort + "}";
    }
}
