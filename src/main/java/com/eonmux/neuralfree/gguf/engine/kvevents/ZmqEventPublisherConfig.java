package com.eonmux.neuralfree.gguf.engine.kvevents;

import java.util.Objects;

/**
 * The transport configuration of {@link ZmqEventPublisher}, vLLM's {@code KVEventsConfig} fields:
 * {@code endpoint} (the PUB address), {@code replay_endpoint} (the ROUTER address serving missed batches, null for
 * none), {@code buffer_steps} (how many past batches the replay serves), {@code hwm} (the PUB socket's high-water
 * mark, 0 for none), {@code max_queue_size} (the batches buffered between the engine's thread and the publisher
 * thread, 0 for no bound, as Python's {@code Queue(maxsize=0)}) and {@code topic}, plus the data-parallel rank the
 * publisher stamps and offsets its ports by. Endpoints are {@code tcp://host:port} addresses: JeroMQ emulates
 * {@code ipc://} over loopback TCP, which no libzmq subscriber can reach, and its {@code inproc://} sockets serve
 * only the publisher's own context, so both are refused by name. {@link #nodeIp()} is vLLM's {@code get_ip()}, the
 * node address an ephemeral wildcard bind reports. Immutable.
 */
public final class ZmqEventPublisherConfig {

    /** vLLM's default PUB endpoint. */
    public static final String DEFAULT_ENDPOINT = "tcp://*:5557";
    /** vLLM's default replay buffer. */
    public static final int DEFAULT_BUFFER_STEPS = 10_000;
    /** vLLM's default PUB high-water mark. */
    public static final int DEFAULT_HWM = 100_000;
    /** vLLM's default queue bound. */
    public static final int DEFAULT_MAX_QUEUE_SIZE = 100_000;
    /**
     * The node IP an ephemeral wildcard bind reports, the property form of vLLM's {@code VLLM_HOST_IP}.
     */
    public static final String PROP_HOST_IP = "neuralfree.host.ip";

    private static final String TCP = "tcp://";
    private static final int MAX_PORT = 65_535;

    private final String endpoint;
    private final String replayEndpoint;
    private final int bufferSteps;
    private final int hwm;
    private final int maxQueueSize;
    private final String topic;
    private final int dataParallelRank;

    /**
     * @throws IllegalArgumentException when an endpoint (the replay one may be null) is not a
     *                                  {@code tcp://host:port} address with a port in {@code [0, 65535]}, a count
     *                                  is negative, the topic is null, or the data-parallel rank is negative
     */
    public ZmqEventPublisherConfig(String endpoint, String replayEndpoint, int bufferSteps, int hwm, int maxQueueSize,
                                   String topic, int dataParallelRank) {
        this.endpoint = endpoint("endpoint", endpoint);
        this.replayEndpoint = replayEndpoint == null ? null : endpoint("replay_endpoint", replayEndpoint);
        this.bufferSteps = nonNegative("buffer_steps", bufferSteps);
        this.hwm = nonNegative("hwm", hwm);
        this.maxQueueSize = nonNegative("max_queue_size", maxQueueSize);
        this.topic = Objects.requireNonNull(topic, "topic");
        this.dataParallelRank = nonNegative("data-parallel rank", dataParallelRank);
    }

    /** The configuration vLLM's {@code KVEventsConfig} defaults to. */
    public static ZmqEventPublisherConfig defaults() {
        return new ZmqEventPublisherConfig(DEFAULT_ENDPOINT, null, DEFAULT_BUFFER_STEPS, DEFAULT_HWM,
                DEFAULT_MAX_QUEUE_SIZE, "", 0);
    }

    /** {@code value}, refused unless it is a {@code tcp://host:port} address; {@code field} names it. */
    private static String endpoint(String field, String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must be a tcp://host:port address, got " + value);
        }
        if (!value.startsWith(TCP)) {
            throw new IllegalArgumentException(field + " must be a tcp://host:port address (JeroMQ carries no"
                    + " libzmq-compatible ipc:// or cross-context inproc:// transport), got " + value);
        }
        port(field, value);
        return value;
    }

    /** The port of a {@code tcp://host:port} {@code endpoint}. */
    private static int port(String field, String endpoint) {
        int lastColon = endpoint.lastIndexOf(':');
        if (lastColon < TCP.length()) {
            throw new IllegalArgumentException(field + " has no port, got " + endpoint);
        }
        String digits = endpoint.substring(lastColon + 1);
        int port;
        try {
            port = Integer.parseInt(digits);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(field + " port must be a number, got " + endpoint, e);
        }
        if (port < 0 || port > MAX_PORT) {
            throw new IllegalArgumentException(field + " port must be between 0 and " + MAX_PORT + ", got "
                    + endpoint);
        }
        return port;
    }

    private static int nonNegative(String name, int value) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be >= 0, got " + value);
        }
        return value;
    }

    /** The PUB address; a wildcard host ({@code tcp://*:5557}) binds, any other host connects. */
    public String endpoint() { return endpoint; }

    /** The ROUTER replay address (always bound), or null for none. */
    public String replayEndpoint() { return replayEndpoint; }

    /** How many past batches the replay socket serves. */
    public int bufferSteps() { return bufferSteps; }

    /** The PUB socket's high-water mark; 0 sets none. */
    public int hwm() { return hwm; }

    /** The batches buffered between the engine's thread and the publisher thread; 0 bounds nothing. */
    public int maxQueueSize() { return maxQueueSize; }

    /** The topic of every published frame (vLLM's {@code topic}; subscribers match it as a prefix). */
    public String topic() { return topic; }

    /** The data-parallel rank this publisher stamps and offsets its ports by. */
    public int dataParallelRank() { return dataParallelRank; }

    /** This configuration with {@link #dataParallelRank()} set to {@code rank}. */
    public ZmqEventPublisherConfig withDataParallelRank(int rank) {
        if (rank == dataParallelRank) {
            return this;
        }
        return new ZmqEventPublisherConfig(endpoint, replayEndpoint, bufferSteps, hwm, maxQueueSize, topic, rank);
    }

    /**
     * {@code endpoint} with its port moved up by {@code dataParallelRank}, as vLLM's
     * {@code ZmqEventPublisher.offset_endpoint_port} does, so every rank binds its own address; a port of 0 (an
     * ephemeral bind) and rank 0 leave the endpoint as it is, and null stays null.
     *
     * @throws IllegalArgumentException when the offset port passes 65535
     */
    public static String offsetEndpointPort(String endpoint, int dataParallelRank) {
        if (endpoint == null || dataParallelRank == 0) {
            return endpoint;
        }
        int port = port("endpoint", endpoint);
        if (port == 0) {
            return endpoint;
        }
        long moved = (long) port + dataParallelRank;
        if (moved > MAX_PORT) {
            throw new IllegalArgumentException("endpoint " + endpoint + " offset by the data-parallel rank "
                    + dataParallelRank + " passes port " + MAX_PORT);
        }
        return endpoint.substring(0, endpoint.lastIndexOf(':')) + ":" + moved;
    }

    /**
     * The node IP an ephemeral wildcard ({@code tcp://*:0}) bind reports itself at, vLLM's {@code get_ip()}: the
     * {@code neuralfree.host.ip} property when set, else the address the primary interface holds — found the way
     * Python finds it, a UDP connect to a public address (which sends nothing) selecting the route — else, with a
     * warning, {@code 0.0.0.0}.
     */
    public static String nodeIp() {
        String configured = System.getProperty(PROP_HOST_IP);
        if (configured != null && !configured.isBlank()) {
            return configured.trim();
        }
        for (String probe : new String[] {"8.8.8.8", "2001:4860:4860::8888"}) {
            try (java.net.DatagramSocket socket = new java.net.DatagramSocket()) {
                socket.connect(new java.net.InetSocketAddress(probe, 80));
                java.net.InetAddress address = socket.getLocalAddress();
                if (!address.isAnyLocalAddress()) {
                    return address.getHostAddress();
                }
            } catch (Exception e) {
                // unreachable probe: the next one, or the fallback, answers
            }
        }
        java.util.logging.Logger.getLogger(ZmqEventPublisherConfig.class.getName()).warning(
                "Failed to get the IP address, using 0.0.0.0 by default. The value can be set by the " + PROP_HOST_IP
                        + " property.");
        return "0.0.0.0";
    }

    /** Whether {@code address} is an IPv6 literal (vLLM's {@code is_valid_ipv6_address}). */
    static boolean ipv6(String address) {
        try {
            return java.net.InetAddress.getByName(address) instanceof java.net.Inet6Address
                    && address.indexOf(':') >= 0;
        } catch (Exception e) {
            return false;
        }
    }

    /** {@code tcp://host:port}, the host bracketed for an IPv6 literal (vLLM's {@code get_tcp_uri}). */
    static String tcpUri(String ip, int port) {
        return "tcp://" + (ipv6(ip) ? "[" + ip + "]" : ip) + ":" + port;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ZmqEventPublisherConfig)) {
            return false;
        }
        ZmqEventPublisherConfig c = (ZmqEventPublisherConfig) o;
        return endpoint.equals(c.endpoint) && Objects.equals(replayEndpoint, c.replayEndpoint)
                && bufferSteps == c.bufferSteps && hwm == c.hwm && maxQueueSize == c.maxQueueSize
                && topic.equals(c.topic) && dataParallelRank == c.dataParallelRank;
    }

    @Override
    public int hashCode() {
        return Objects.hash(endpoint, replayEndpoint, bufferSteps, hwm, maxQueueSize, topic, dataParallelRank);
    }

    @Override
    public String toString() {
        return "ZmqEventPublisherConfig{endpoint=" + endpoint + ", replay_endpoint=" + replayEndpoint
                + ", buffer_steps=" + bufferSteps + ", hwm=" + hwm + ", max_queue_size=" + maxQueueSize
                + ", topic=" + topic + ", data_parallel_rank=" + dataParallelRank + "}";
    }
}
