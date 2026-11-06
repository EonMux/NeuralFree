package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The engine methods {@code POST /collective_rpc} dispatches to every replica (vLLM's development RPC route): the
 * weight-transfer lifecycle an RL trainer drives through it ({@code init_weight_transfer_engine},
 * {@code shutdown_weight_transfer_engine}, {@code start_weight_update}, {@code update_weights},
 * {@code finish_weight_update}, {@code update_weight_version}), plus the reads that verify the pool —
 * {@code get_weight_version} and {@code reset_prefix_cache}. Each call runs on every replica's pump thread, in
 * pool order, and answers one result per replica: the methods that return nothing answer null entries, the reads
 * answer each replica's own value.
 *
 * <p>The supported methods are a closed set: an unknown one refuses by name, listing the set, and so does a
 * positional or keyword argument the method does not take. Payloads are strings, as vLLM's HTTP route carries
 * them; the weight methods read the same JSON the dedicated routes read (a serialized init info, update tensor or
 * version).</p>
 */
final class CollectiveRpc {

    /** The methods the route dispatches, in the order the unsupported-method refusal lists them. */
    static final List<String> METHODS = List.of(
            "init_weight_transfer_engine",
            "shutdown_weight_transfer_engine",
            "start_weight_update",
            "update_weights",
            "finish_weight_update",
            "update_weight_version",
            "get_weight_version",
            "reset_prefix_cache");

    private final EngineService service;

    CollectiveRpc(EngineService service) {
        this.service = Objects.requireNonNull(service, "service");
    }

    /**
     * Runs {@code method} on every replica and returns the per-replica results in pool order.
     *
     * @param args       positional string arguments, at most one
     * @param kwargs     keyword string arguments, at most one
     * @param waitMillis how long each replica's pump may take to answer; 0 or less takes the request budget
     * @throws IllegalArgumentException when the method or an argument is not supported
     * @throws IllegalStateException    when the engine cannot serve, the transfer is not initialized, a replica is
     *                                  busy, or a pump did not answer in time
     * @throws UnsupportedOperationException when a replica's model placement takes no live update
     */
    List<Object> call(String method, List<String> args, Map<String, String> kwargs, long waitMillis)
            throws InterruptedException {
        switch (method) {
            case "init_weight_transfer_engine":
                return init(args, kwargs, waitMillis);
            case "shutdown_weight_transfer_engine":
                return shutdown(args, kwargs, waitMillis);
            case "start_weight_update":
                return start(args, kwargs, waitMillis);
            case "update_weights":
                return update(args, kwargs, waitMillis);
            case "finish_weight_update":
                return finish(args, kwargs, waitMillis);
            case "update_weight_version":
                return stampVersion(args, kwargs, waitMillis);
            case "get_weight_version":
                return versions(args, kwargs, waitMillis);
            case "reset_prefix_cache":
                return resetPrefixCache(args, kwargs, waitMillis);
            default:
                throw new IllegalArgumentException("the collective RPC method '" + method + "' is not supported:"
                        + " the methods are " + String.join(", ", METHODS));
        }
    }

    private List<Object> init(List<String> args, Map<String, String> kwargs, long waitMillis)
            throws InterruptedException {
        String payload = payload("init_weight_transfer_engine", args, kwargs, "init_info", "init_info", false);
        String backend = WeightTransferBodies.initPayload(payload);
        ServerJson.requireLoaded(service);
        service.weightTransfer().init(backend, waitMillis);
        return nulls();
    }

    private List<Object> shutdown(List<String> args, Map<String, String> kwargs, long waitMillis)
            throws InterruptedException {
        noArguments("shutdown_weight_transfer_engine", args, kwargs);
        service.weightTransfer().shutdown(waitMillis);
        return nulls();
    }

    private List<Object> start(List<String> args, Map<String, String> kwargs, long waitMillis)
            throws InterruptedException {
        noArguments("start_weight_update", args, kwargs);
        ServerJson.requireLoaded(service);
        service.weightTransfer().start(waitMillis);
        return nulls();
    }

    private List<Object> update(List<String> args, Map<String, String> kwargs, long waitMillis)
            throws InterruptedException {
        String payload = payload("update_weights", args, kwargs, "update_info", "update_info", true);
        List<WeightUpdateTensor> tensors = WeightTransferBodies.updatePayload(payload);
        ServerJson.requireLoaded(service);
        service.weightTransfer().apply(tensors, waitMillis);
        return nulls();
    }

    private List<Object> finish(List<String> args, Map<String, String> kwargs, long waitMillis)
            throws InterruptedException {
        String payload = payload("finish_weight_update", args, kwargs, "weight_version", "weight_version", false);
        String version = WeightTransferBodies.versionPayload(payload, "weight_version", false);
        ServerJson.requireLoaded(service);
        service.weightTransfer().finish(version, waitMillis);
        return nulls();
    }

    private List<Object> stampVersion(List<String> args, Map<String, String> kwargs, long waitMillis)
            throws InterruptedException {
        String payload = payload("update_weight_version", args, kwargs, "new_version", "new_version", true);
        String version = WeightTransferBodies.versionPayload(payload, "new_version", true);
        ServerJson.requireLoaded(service);
        service.weightTransfer().setVersion(version, waitMillis);
        return nulls();
    }

    private List<Object> versions(List<String> args, Map<String, String> kwargs, long waitMillis)
            throws InterruptedException {
        noArguments("get_weight_version", args, kwargs);
        ServerJson.requireLoaded(service);
        List<Object> out = new ArrayList<>();
        out.addAll(service.weightTransfer().versions(waitMillis));
        return out;
    }

    private List<Object> resetPrefixCache(List<String> args, Map<String, String> kwargs, long waitMillis)
            throws InterruptedException {
        noArguments("reset_prefix_cache", args, kwargs);
        ServerJson.requireLoaded(service);
        List<Object> out = new ArrayList<>();
        for (EngineReplica replica : service.replicas()) {
            out.add(replica.resetPrefixCacheOp(waitMillis));
        }
        return out;
    }

    /** One null result per replica: what a method that returns nothing answers. */
    private List<Object> nulls() {
        return new ArrayList<>(Collections.nCopies(service.replicaCount(), null));
    }

    /** Refuses any positional or keyword argument: the method takes none. */
    private static void noArguments(String method, List<String> args, Map<String, String> kwargs) {
        if (!args.isEmpty()) {
            throw ServerJson.badRequest("the collective RPC method '" + method + "' takes no positional"
                    + " arguments, got '" + args.get(0) + "'");
        }
        if (!kwargs.isEmpty()) {
            throw ServerJson.badRequest("the collective RPC method '" + method + "' takes no keyword arguments,"
                    + " got '" + kwargs.keySet().iterator().next() + "'");
        }
    }

    /**
     * The one payload a method carries: its positional argument or its keyword, not both.
     *
     * @param argName   the payload's name in a missing-argument refusal
     * @param kwargName the only keyword the method accepts
     * @param required  whether an absent or blank payload refuses
     */
    private static String payload(String method, List<String> args, Map<String, String> kwargs,
                                  String argName, String kwargName, boolean required) {
        if (args.size() > 1) {
            throw ServerJson.badRequest("the collective RPC method '" + method + "' takes at most one positional"
                    + " argument, got " + args.size());
        }
        for (String key : kwargs.keySet()) {
            if (!key.equals(kwargName)) {
                throw ServerJson.badRequest("the collective RPC method '" + method + "' carries the unsupported"
                        + " keyword argument '" + key + "': the keyword is " + kwargName);
            }
        }
        if (!args.isEmpty() && kwargs.containsKey(kwargName)) {
            throw ServerJson.badRequest("the collective RPC method '" + method + "' carries both a positional"
                    + " argument and the keyword '" + kwargName + "'");
        }
        String value = args.isEmpty() ? kwargs.get(kwargName) : args.get(0);
        if (required && (value == null || value.isBlank())) {
            throw ServerJson.badRequest("the collective RPC method '" + method + "' requires the " + argName
                    + " argument");
        }
        return value;
    }
}
