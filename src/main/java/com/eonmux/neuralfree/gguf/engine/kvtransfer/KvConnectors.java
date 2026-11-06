package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import java.io.IOException;
import java.util.Locale;

/**
 * Creates the connector a {@link KvTransferConfig} names. The reference engines' connectors are refused by name
 * with the reason they cannot run here, so a deployment that copies a vLLM, Mooncake or LMCache configuration never
 * silently falls back to another transport.
 */
public final class KvConnectors {

    private KvConnectors() {
    }

    /**
     * The connector {@code config} names.
     *
     * @throws IllegalArgumentException      for an unknown connector name, naming the ones this build serves
     * @throws UnsupportedOperationException for a reference connector that needs native bindings, named with the
     *                                       reason
     * @throws IOException                   when the connector cannot start (a port conflict, an unreadable storage
     *                                       directory)
     */
    public static KvConnector create(KvTransferConfig config) throws IOException {
        String name = config.kvConnector().toLowerCase(Locale.ROOT);
        switch (name) {
            case "tcpkvconnector":
            case "tcpconnector":
            case "tcp":
                return new TcpKvConnector(config);
            case "filekvconnector":
            case "fileconnector":
            case "file":
                return new FileKvConnector(config);
            case "nixlconnector":
            case "nixl":
                throw unsupported(config, "NIXL", "its RDMA/UCX transfer engine and native bindings");
            case "mooncakeconnector":
            case "mooncakestoreconnector":
            case "mooncake":
                throw unsupported(config, "Mooncake", "its transfer-engine library and native bindings");
            case "lmcacheconnectorv1":
            case "lmcacheconnector":
            case "lmcache":
                throw unsupported(config, "LMCache", "its Python cache/store service");
            case "multiconnector":
                throw new UnsupportedOperationException("kv_connector MultiConnector chains several connectors;"
                        + " this build runs one connector per instance (TcpKvConnector or FileKvConnector)");
            case "offloadingconnector":
            case "simplecpuoffloadconnector":
                throw new UnsupportedOperationException("kv_connector " + config.kvConnector() + " offloads KV to"
                        + " another tier of the same instance, not between instances; this build's tiers are"
                        + " neuralfree.kv.cache.ram.mib and neuralfree.kv.cache.disk.mib");
            default:
                throw new IllegalArgumentException("unknown kv_connector " + config.kvConnector()
                        + "; this build serves TcpKvConnector and FileKvConnector");
        }
    }

    private static UnsupportedOperationException unsupported(KvTransferConfig config, String family, String needs) {
        return new UnsupportedOperationException("kv_connector " + config.kvConnector() + " (" + family + ") needs "
                + needs + ", which this build does not ship; configure TcpKvConnector (a socket pull/push) or"
                + " FileKvConnector (shared storage) instead");
    }
}
