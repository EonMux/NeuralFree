package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorDevice;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.hip.HipDevice;
import com.eonmux.neuralfree.acceleration.metal.MetalDevice;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.acceleration.vulkan.VkDevice;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.rpc.RemoteComputeBackend;
import com.eonmux.neuralfree.gguf.rpc.RpcServerSpec;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * The GPU devices of a layer split: every device the configured backend mode can reach, narrowed by the
 * {@code neuralfree.gpu.devices} selectors (llama.cpp {@code --device}: an index into the enumeration, or a
 * device-name substring), and opened, one {@link ComputeBackend} per device. Without selectors the split runs over
 * every device of the mode, discrete before integrated, and drops the duplicate views one GPU offers through several
 * APIs ({@link DeviceIdentity#samePhysicalDevice}). The {@code neuralfree.rpc} servers are remote devices
 * ({@link RemoteComputeBackend}) appended after the local selection in configuration order, one view per device the
 * server serves ({@code rpc<i>:<device>}); each is a layer target of its own and none is deduplicated against a
 * local device. When {@code neuralfree.gpu.devices} is set, the selectors choose from the local devices and the
 * remote servers together (a server is named as {@code rpc<i>}, or by a substring of its endpoint), so a remote can
 * also be left out; an unnamed remote server takes part otherwise. The main GPU ({@code neuralfree.main.gpu}) holds
 * the embedding and the output head. {@link #selectOne} opens the single device one selector names, and
 * {@link #openMainGpu} the one GPU the main GPU picks for a load without a split.
 */
public final class GpuDevices {

    /** One selected device and its open backend. */
    public static final class Selected {
        private final ComputeBackend backend;
        private final String name;
        private final boolean remote;

        /** A device opened elsewhere: its backend and display name. */
        public Selected(ComputeBackend backend, String name) {
            this(backend, name, false);
        }

        /** A device opened elsewhere, with whether its backend is a connection this split owns. */
        public Selected(ComputeBackend backend, String name, boolean remote) {
            this.backend = backend;
            this.name = name;
            this.remote = remote;
        }

        public ComputeBackend backend() { return backend; }
        public String name() { return name; }

        /** Whether the backend is a remote device connection rather than a locally opened GPU. */
        public boolean remote() { return remote; }

        /**
         * Closes the connection of a remote device; a local device's backend belongs to the acceleration that opened
         * it and is left alone. Idempotent.
         */
        public void close() {
            if (remote) {
                backend.close();
            }
        }

        @Override
        public String toString() {
            return name;
        }
    }

    /**
     * The APIs, Metal first (the native Apple path beats its Vulkan view), then the rest of the
     * built-ins, then every accelerator API in declaration order.
     */
    private static final BackendType[] APIS = apis();

    private static BackendType[] apis() {
        List<BackendType> order = new ArrayList<>();
        order.add(BackendType.METAL);
        for (BackendType api : BackendType.values()) {
            if (!api.isAccelerator() && api != BackendType.METAL) {
                order.add(api);
            }
        }
        for (BackendType api : BackendType.values()) {
            if (api.isAccelerator()) {
                order.add(api);
            }
        }
        return order.toArray(new BackendType[0]);
    }

    private GpuDevices() {
    }

    /**
     * Opens the devices of {@code cfg}: the {@code neuralfree.gpu.devices} selectors when given — choosing from the
     * local devices and the {@code neuralfree.rpc} servers together — otherwise every enumerated local device,
     * deduplicated across APIs and ordered discrete first, then every remote server in configuration order. Each
     * server contributes one view per device it serves ({@code rpc-server -d list}). Returns an empty list when
     * fewer than two devices remain and no remote server is configured (or selectors picked the subset), so a
     * single-GPU host loads as one device; the caller closes the opened backends then.
     *
     * @param acceleration the enumeration the backends are created over; its {@link HWAcceleration#shutdown()}
     *                     closes what this method created
     * @throws IllegalArgumentException when a selector matches no device (the message lists the available ones),
     *                                  {@code neuralfree.main.gpu} is beyond the last device, or remote devices are
     *                                  configured without selectors but fewer than two devices are available
     */
    public static List<Selected> select(HWAcceleration acceleration, EngineConfig cfg) {
        Objects.requireNonNull(acceleration, "acceleration");
        Objects.requireNonNull(cfg, "cfg");
        List<Candidate> pool = candidates(acceleration, cfg);
        if (pool.isEmpty()) {
            throw new IllegalArgumentException(EngineConfig.PROP_SPLIT_MODE + "=layer found no GPU device; the"
                    + " " + EngineConfig.PROP_BACKEND + "=" + cfg.backend() + " mode enumerated none and no "
                    + EngineConfig.PROP_RPC + " server is configured");
        }
        List<Candidate> picked = cfg.gpuDevices().isEmpty()
                ? withRemotes(autoSelection(localCandidates(pool)), pool)
                : bySelectors(pool, cfg.gpuDevices());
        int main = cfg.mainGpu();
        if (main >= picked.size()) {
            throw new IllegalArgumentException(EngineConfig.PROP_MAIN_GPU + "=" + main + " is beyond the "
                    + picked.size() + " selected device(s): " + names(picked));
        }
        Map<RpcServerSpec, List<RemoteComputeBackend>> opened = new LinkedHashMap<>();
        List<Selected> out = new ArrayList<>(picked.size());
        try {
            for (Candidate c : picked) {
                if (!c.remote()) {
                    out.add(c.open(acceleration));
                    continue;
                }
                List<RemoteComputeBackend> views = remoteViews(c.spec(), opened);
                for (int d = 0; d < views.size(); d++) {
                    out.add(new Selected(views.get(d), "rpc" + c.serverIndex() + ":" + d + " '"
                            + c.spec().describe() + "'", true));
                }
            }
        } catch (RuntimeException e) {
            closeAll(out);
            closeViews(opened);
            throw e;
        }
        if (cfg.gpuDevices().isEmpty() && !cfg.rpcServers().isEmpty() && out.size() < 2) {
            closeAll(out);
            closeViews(opened);
            throw new IllegalArgumentException(EngineConfig.PROP_RPC + " serves " + out.size()
                    + " remote device(s), but " + EngineConfig.PROP_SPLIT_MODE + "=layer needs at least two devices"
                    + " in total");
        }
        if (out.size() < 2) {
            for (Selected s : out) {
                closeQuietly(s.backend());
            }
            return new ArrayList<>();
        }
        return out;
    }

    /**
     * The views of one remote server, connecting once per server: the first pick opens every device
     * ({@link RemoteComputeBackend#openAll}) and the later picks reuse them.
     */
    private static List<RemoteComputeBackend> remoteViews(RpcServerSpec spec,
                                                          Map<RpcServerSpec, List<RemoteComputeBackend>> opened) {
        List<RemoteComputeBackend> views = opened.get(spec);
        if (views == null) {
            views = openRemoteAll(spec);
            opened.put(spec, views);
        }
        return views;
    }

    private static List<RemoteComputeBackend> openRemoteAll(RpcServerSpec spec) {
        try {
            return RemoteComputeBackend.openAll(spec);
        } catch (IOException e) {
            throw new IllegalStateException(EngineConfig.PROP_RPC + " entry " + spec.describe()
                    + " could not be reached: " + e.getMessage(), e);
        }
    }

    /** Closes every opened backend, local and remote. */
    private static void closeAll(List<Selected> devices) {
        for (Selected device : devices) {
            try {
                device.close();
            } catch (RuntimeException ignored) {
                // the split is already failing; the connections are unusable either way
            }
        }
    }

    /** Closes the views of servers whose connection was opened before the selection failed. */
    private static void closeViews(Map<RpcServerSpec, List<RemoteComputeBackend>> opened) {
        for (List<RemoteComputeBackend> views : opened.values()) {
            for (RemoteComputeBackend view : views) {
                try {
                    view.close();
                } catch (RuntimeException ignored) {
                    // the split is already failing; the connections are unusable either way
                }
            }
        }
    }

    /**
     * Closes the remote connections of {@code devices}; local devices are left to the acceleration that opened them.
     * Used by a split that releases devices it opened before a session took them over.
     */
    public static void closeRemote(List<Selected> devices) {
        for (Selected device : devices) {
            try {
                device.close();
            } catch (RuntimeException ignored) {
                // the split is already failing; the connections are unusable either way
            }
        }
    }

    /**
     * Opens the peer GPUs of a tensor-parallel row split ({@code neuralfree.split.mode=row}): the devices
     * {@code cfg} selects minus the one {@code mainBackend} runs on. The main device is rank 0 of the split and
     * holds its own shard, so a {@code neuralfree.gpu.devices} selector that names it only skips. Without selectors
     * every device of the mode participates, discrete before integrated, with the further views of one physical GPU
     * dropped ({@link DeviceIdentity#samePhysicalDevice}), as {@link #select} picks the devices of a layer split.
     * When the identity of the main device is unknown, its view is matched by device name.
     *
     * @return the peers in rank order (rank 1, 2, ...); empty when the machine offers no second device, so the
     *         caller loads the model on one device
     * @throws IllegalArgumentException when a selector matches no device (the message lists the available ones)
     */
    public static List<Selected> selectPeers(HWAcceleration acceleration, EngineConfig cfg,
                                             ComputeBackend mainBackend) {
        Objects.requireNonNull(acceleration, "acceleration");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(mainBackend, "mainBackend");
        List<Candidate> candidates = candidates(acceleration);
        if (candidates.isEmpty()) {
            return new ArrayList<>();
        }
        List<Candidate> pool = cfg.gpuDevices().isEmpty()
                ? autoSelection(candidates)
                : bySelectors(candidates, cfg.gpuDevices());
        DeviceIdentity mainIdentity = mainBackend.capabilities().getIdentity();
        String mainName = mainBackend.capabilities().getDeviceName();
        List<Selected> out = new ArrayList<>(pool.size());
        for (Candidate c : pool) {
            if (isMainDevice(c, mainIdentity, mainName)) {
                continue;
            }
            out.add(c.open(acceleration));
        }
        return out;
    }

    /** Whether a candidate is a view of the main backend's device: by identity, else by device name. */
    private static boolean isMainDevice(Candidate c, DeviceIdentity mainIdentity, String mainName) {
        if (samePhysical(c.identity(), mainIdentity)) {
            return true;
        }
        if (c.identity() != null && mainIdentity != null) {
            return false;
        }
        return mainName != null && !mainName.isEmpty()
                && c.name().toLowerCase(Locale.ROOT).contains(mainName.toLowerCase(Locale.ROOT));
    }

    /**
     * Opens the one device {@code selector} names: an index into the enumeration or a device-name substring, as
     * the {@code neuralfree.gpu.devices} entries take them (llama.cpp {@code --mmproj-device}).
     *
     * @param acceleration the enumeration the backend is created over; its {@link HWAcceleration#shutdown()}
     *                     closes what this method created
     * @param property     the setting the selector came from, named in the error
     * @throws IllegalArgumentException when the selector matches no device (the message lists the available ones)
     */
    public static Selected selectOne(HWAcceleration acceleration, String selector, String property) {
        Objects.requireNonNull(acceleration, "acceleration");
        Objects.requireNonNull(selector, "selector");
        Objects.requireNonNull(property, "property");
        List<Candidate> candidates = candidates(acceleration);
        Candidate match = match(candidates, selector);
        if (match == null) {
            throw new IllegalArgumentException(property + " '" + selector + "' names no device; the devices of this"
                    + " machine are " + names(candidates));
        }
        return match.open(acceleration);
    }

    /**
     * Opens the local devices an rpc-server serves ({@code rpc-server -d list}): the selectors when given, in
     * selector order, otherwise every enumerated device of the acceleration, deduplicated across APIs and ordered
     * discrete first — the device list a server then reports at its handshake, one connection per client serving all
     * of them (llama.cpp rpc-server {@code -d}).
     *
     * @param acceleration the enumeration the backends are created over; its {@link HWAcceleration#shutdown()}
     *                     closes what this method created
     * @param selectors    the device selectors (indices or name substrings), empty for every device
     * @param property     the setting the selectors came from, named in the error
     * @throws IllegalArgumentException when a selector matches no device (the message lists the available ones) or
     *                                  the acceleration enumerates no device
     */
    public static List<Selected> selectServerDevices(HWAcceleration acceleration, List<String> selectors,
                                                     String property) {
        Objects.requireNonNull(acceleration, "acceleration");
        Objects.requireNonNull(selectors, "selectors");
        Objects.requireNonNull(property, "property");
        List<Candidate> pool = candidates(acceleration);
        if (pool.isEmpty()) {
            throw new IllegalArgumentException(property + " found no GPU device; the backend mode of the server"
                    + " enumerated none");
        }
        List<Candidate> picked = selectors.isEmpty() ? autoSelection(pool) : bySelectors(pool, selectors);
        List<Selected> out = new ArrayList<>(picked.size());
        for (Candidate c : picked) {
            out.add(c.open(acceleration));
        }
        return out;
    }

    /**
     * Opens the one GPU {@code neuralfree.main.gpu} picks for a one-device load ({@link EngineConfig#picksMainGpu()},
     * which includes a layer split over a single listed device index): entry {@link EngineConfig#mainGpu()} of the
     * {@code neuralfree.gpu.devices} list, or, without a list, that index into the devices a split would select
     * (every device of the mode, deduplicated across APIs, discrete first), as llama.cpp keeps
     * {@code devices[main_gpu]} under {@code --split-mode none}.
     *
     * @param acceleration the enumeration the backend is created over; its {@link HWAcceleration#shutdown()}
     *                     closes what this method created
     * @throws IllegalArgumentException when the configuration does not pick its GPU by the main GPU, when the
     *                                  entry names no device, or when the main GPU is beyond the devices
     */
    public static Selected openMainGpu(HWAcceleration acceleration, EngineConfig cfg) {
        Objects.requireNonNull(acceleration, "acceleration");
        Objects.requireNonNull(cfg, "cfg");
        if (!cfg.picksMainGpu()) {
            throw new IllegalArgumentException(EngineConfig.PROP_MAIN_GPU + " does not pick the GPU of " + cfg);
        }
        int main = cfg.mainGpu();
        if (!cfg.gpuDevices().isEmpty()) {
            if (main >= cfg.gpuDevices().size()) {
                throw new IllegalArgumentException(EngineConfig.PROP_MAIN_GPU + "=" + main + " is beyond the "
                        + cfg.gpuDevices().size() + " device(s) of " + EngineConfig.PROP_GPU_DEVICES + " "
                        + cfg.gpuDevices());
            }
            return selectOne(acceleration, cfg.gpuDevices().get(main),
                    EngineConfig.PROP_GPU_DEVICES + " entry " + main + " (" + EngineConfig.PROP_MAIN_GPU + ")");
        }
        List<Candidate> picked = autoSelection(candidates(acceleration));
        if (main >= picked.size()) {
            throw new IllegalArgumentException(EngineConfig.PROP_MAIN_GPU + "=" + main + " is beyond the "
                    + picked.size() + " GPU(s) of this machine: " + names(picked));
        }
        return picked.get(main).open(acceleration);
    }

    /** Every device the acceleration enumerated, in API order. */
    private static List<Candidate> candidates(HWAcceleration acceleration) {
        List<Candidate> out = new ArrayList<>();
        for (BackendType api : APIS) {
            int index = 0;
            switch (api) {
                case OPENCL:
                    for (CLDevice d : acceleration.getCLDevices()) {
                        out.add(candidate(api, index++, d, clName(d), () -> acceleration.createBackend(d)));
                    }
                    break;
                case CUDA:
                    for (CUDevice d : acceleration.getCUDevices()) {
                        out.add(candidate(api, index++, d, d.getDeviceName(), () -> acceleration.createBackend(d)));
                    }
                    break;
                case VULKAN:
                    for (VkDevice d : acceleration.getVkDevices()) {
                        out.add(candidate(api, index++, d, d.getDeviceName(), () -> acceleration.createBackend(d)));
                    }
                    break;
                case HIP:
                    for (HipDevice d : acceleration.getHipDevices()) {
                        out.add(candidate(api, index++, d, d.getDeviceName(), () -> acceleration.createBackend(d)));
                    }
                    break;
                case METAL:
                    for (MetalDevice d : acceleration.getMetalDevices()) {
                        out.add(candidate(api, index++, d, d.getDeviceName(), () -> acceleration.createBackend(d)));
                    }
                    break;
                default:
                    if (!api.isAccelerator()) {
                        throw new IllegalStateException("no device enumeration for " + api);
                    }
                    for (AcceleratorDevice d : acceleration.getAcceleratorDevices(api)) {
                        out.add(candidate(api, index++, d.name(), d.identity(), d.deviceClass(),
                                () -> acceleration.createBackend(d)));
                    }
                    break;
            }
        }
        return out;
    }

    /**
     * The local devices plus one remote candidate per {@code neuralfree.rpc} server, in configuration order. A remote
     * device carries no identity, so it is never merged with a local view of the same physical GPU; its connection is
     * opened lazily, once per server, when the first of its picked devices opens.
     */
    private static List<Candidate> candidates(HWAcceleration acceleration, EngineConfig cfg) {
        List<Candidate> out = candidates(acceleration);
        int index = 0;
        for (RpcServerSpec spec : cfg.rpcServers()) {
            out.add(Candidate.remote(index, spec));
            index++;
        }
        return out;
    }

    private static List<Candidate> localCandidates(List<Candidate> candidates) {
        List<Candidate> out = new ArrayList<>(candidates.size());
        for (Candidate c : candidates) {
            if (!c.remote()) {
                out.add(c);
            }
        }
        return out;
    }

    /** {@code picked}, then every remote candidate (config order); remotes are not subject to the selectors. */
    private static List<Candidate> withRemotes(List<Candidate> picked, List<Candidate> candidates) {
        List<Candidate> out = new ArrayList<>(picked);
        for (Candidate c : candidates) {
            if (c.remote() && !out.contains(c)) {
                out.add(c);
            }
        }
        return out;
    }

    /** The device name of an OpenCL device, from the info query the constructor ran. */
    private static String clName(CLDevice device) {
        Object name = device.getDeviceInfo() == null ? null : device.getDeviceInfo().get("CL_DEVICE_NAME");
        return name instanceof String ? (String) name : null;
    }

    private static Candidate candidate(BackendType api, int index, Object device, String deviceName,
                                       Supplier<ComputeBackend> opener) {
        return new Candidate(api, index, name(api, index, deviceName), identity(device), deviceClass(device),
                opener, false);
    }

    private static Candidate candidate(BackendType api, int index, String deviceName, DeviceIdentity identity,
                                       DeviceClass deviceClass, Supplier<ComputeBackend> opener) {
        return new Candidate(api, index, name(api, index, deviceName), identity, deviceClass, opener, false);
    }

    private static String name(BackendType api, int index, String deviceName) {
        String n = deviceName == null || deviceName.trim().isEmpty() ? "unknown" : deviceName.trim();
        return api.toString().toLowerCase(Locale.ROOT) + index + " '" + n + "'";
    }

    private static DeviceIdentity identity(Object device) {
        try {
            if (device instanceof CLDevice) {
                return ((CLDevice) device).getIdentity();
            }
            if (device instanceof CUDevice) {
                return ((CUDevice) device).getIdentity();
            }
            if (device instanceof VkDevice) {
                return ((VkDevice) device).getIdentity();
            }
            if (device instanceof HipDevice) {
                return ((HipDevice) device).getIdentity();
            }
        } catch (RuntimeException ignored) {
            // an unnamed device is matched by name only
        }
        return null;
    }

    private static DeviceClass deviceClass(Object device) {
        try {
            if (device instanceof CLDevice) {
                Object unified = ((CLDevice) device).getDeviceInfo() == null ? null
                        : ((CLDevice) device).getDeviceInfo().get("CL_DEVICE_HOST_UNIFIED_MEMORY");
                return unified instanceof Boolean && !((Boolean) unified) ? DeviceClass.DISCRETE
                        : DeviceClass.INTEGRATED;
            }
            if (device instanceof CUDevice) {
                return ((CUDevice) device).getDeviceClass();
            }
            if (device instanceof VkDevice) {
                return ((VkDevice) device).getInfo().deviceClass();
            }
            if (device instanceof HipDevice) {
                return ((HipDevice) device).getDeviceClass();
            }
        } catch (RuntimeException ignored) {
            // an unqueried device sorts last
        }
        return DeviceClass.UNKNOWN;
    }

    /** Every device, minus the further views of one already taken, discrete before the rest. */
    private static List<Candidate> autoSelection(List<Candidate> candidates) {
        List<Candidate> unique = new ArrayList<>(candidates.size());
        for (Candidate c : candidates) {
            boolean duplicate = false;
            for (Candidate taken : unique) {
                if (samePhysical(c.identity(), taken.identity())) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                unique.add(c);
            }
        }
        List<Candidate> out = new ArrayList<>(unique);
        out.sort((a, b) -> rank(a.deviceClass()) - rank(b.deviceClass()));
        return out;
    }

    private static int rank(DeviceClass deviceClass) {
        switch (deviceClass) {
            case DISCRETE:
                return 0;
            case INTEGRATED:
                return 1;
            default:
                return 2;
        }
    }

    /** The devices the selectors name, in selector order: an index into the enumeration or a name substring. */
    private static List<Candidate> bySelectors(List<Candidate> candidates, List<String> selectors) {
        List<Candidate> out = new ArrayList<>(selectors.size());
        for (String selector : selectors) {
            Candidate match = match(candidates, selector);
            if (match == null) {
                throw new IllegalArgumentException(EngineConfig.PROP_GPU_DEVICES + " entry '" + selector
                        + "' names no device; the devices are " + names(candidates));
            }
            if (!out.contains(match)) {
                out.add(match);
            }
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException(EngineConfig.PROP_GPU_DEVICES + " selects no device of "
                    + names(candidates));
        }
        return out;
    }

    private static Candidate match(List<Candidate> candidates, String selector) {
        String s = selector.trim();
        int index = index(s);
        if (index >= 0) {
            for (Candidate c : candidates) {
                if (c.apiIndex() == index) {
                    return c;
                }
            }
        }
        String lower = s.toLowerCase(Locale.ROOT);
        for (Candidate c : candidates) {
            if (c.name().toLowerCase(Locale.ROOT).contains(lower)) {
                return c;
            }
        }
        return null;
    }

    /** The non-negative index of an all-digit selector, or -1. */
    private static int index(String selector) {
        if (selector.isEmpty()) {
            return -1;
        }
        for (int i = 0; i < selector.length(); i++) {
            if (!Character.isDigit(selector.charAt(i))) {
                return -1;
            }
        }
        return Integer.parseInt(selector);
    }

    private static boolean samePhysical(DeviceIdentity a, DeviceIdentity b) {
        return a != null && b != null && a.samePhysicalDevice(b);
    }

    private static String names(List<Candidate> candidates) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < candidates.size(); i++) {
            sb.append(i == 0 ? "" : ", ").append(candidates.get(i).name());
        }
        return sb.append(']').toString();
    }

    private static void closeQuietly(ComputeBackend backend) {
        try {
            backend.close();
        } catch (RuntimeException ignored) {
            // the split did not engage; the shutdown of the acceleration releases the backend anyway
        }
    }

    /** One enumerated device: its API, enumeration index, name, identity, class and backend opener. */
    private static final class Candidate {
        private final BackendType api;
        private final int apiIndex;
        private final String name;
        private final DeviceIdentity identity;
        private final DeviceClass deviceClass;
        private final Supplier<ComputeBackend> opener;
        private final boolean remote;
        /** The {@code neuralfree.rpc} configuration index of a remote server, -1 on a local device. */
        private final int serverIndex;
        private final RpcServerSpec spec;

        Candidate(BackendType api, int apiIndex, String name, DeviceIdentity identity, DeviceClass deviceClass,
                  Supplier<ComputeBackend> opener, boolean remote) {
            this(api, apiIndex, name, identity, deviceClass, opener, remote, -1, null);
        }

        Candidate(BackendType api, int apiIndex, String name, DeviceIdentity identity, DeviceClass deviceClass,
                  Supplier<ComputeBackend> opener, boolean remote, int serverIndex, RpcServerSpec spec) {
            this.api = api;
            this.apiIndex = apiIndex;
            this.name = name;
            this.identity = identity;
            this.deviceClass = deviceClass;
            if (!remote) {
                Objects.requireNonNull(opener, "opener");
            }
            this.opener = opener;
            this.remote = remote;
            this.serverIndex = serverIndex;
            this.spec = spec;
        }

        /** The server-level candidate of {@code neuralfree.rpc} entry {@code index}; every device of it picks it. */
        static Candidate remote(int index, RpcServerSpec spec) {
            return new Candidate(null, -1, "rpc" + index + " '" + spec.describe() + "'", null,
                    DeviceClass.UNKNOWN, null, true, index, spec);
        }

        BackendType api() { return api; }
        int apiIndex() { return apiIndex; }
        DeviceIdentity identity() { return identity; }
        DeviceClass deviceClass() { return deviceClass; }
        String name() { return name; }
        boolean remote() { return remote; }
        int serverIndex() { return serverIndex; }
        RpcServerSpec spec() { return spec; }

        Selected open(HWAcceleration acceleration) {
            return new Selected(opener.get(), name, remote);
        }
    }
}
