package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * The AUTO backend policy of {@link HWAcceleration}: device first, then API.
 *
 * <p>The same GPU is usually visible through OpenCL, CUDA and Vulkan at once. AUTO first decides
 * which <em>physical device</em> to run on, then which API to drive it with, recognising the views
 * of one device across APIs by {@link DeviceIdentity#samePhysicalDevice(DeviceIdentity)}:</p>
 * <ol>
 *   <li><b>Target device.</b> The first device, in any API, whose name contains
 *       {@code -Dneuralfree.device}; otherwise the OpenCL ranking's winner
 *       ({@code CLDeviceRanking}, discrete first and then measured), else the Vulkan
 *       ranking's, else the CUDA ranking's.</li>
 *   <li><b>API order on the target.</b> For each API of {@code -Dneuralfree.backend.order}
 *       (default {@code vulkan,cuda,opencl}), the API's view of the target device is turned into a
 *       backend and returned when the requirement accepts it; otherwise it is closed and the
 *       reason logged.</li>
 *   <li><b>Fallback.</b> When no API ran the target, each API's own ranking winner is tried,
 *       discrete before integrated before the rest, then in API order, under the same
 *       requirement. An API and device pair is never tried twice.</li>
 * </ol>
 * <p>{@link #selectAuto} never throws for a backend that fails: every failure becomes a reason in
 * the returned {@link Selection}. The class touches no system property except in
 * {@link #deviceFilter()} and {@link #apiOrder()}, so the policy is testable without a GPU.</p>
 */
final class BackendSelector {

    private static final Logger LOGGER = LoggingUtil.getLogger(BackendSelector.class);

    /** API-independent device filter: a case-insensitive substring of the device name. */
    static final String DEVICE_PROPERTY = "neuralfree.device";
    /** Comma-separated APIs AUTO may use, in preference order, for example {@code cuda,opencl}. */
    static final String ORDER_PROPERTY = "neuralfree.backend.order";

    /** AUTO's API preference when {@link #ORDER_PROPERTY} is unset. */
    static final List<BackendType> DEFAULT_ORDER =
            Collections.unmodifiableList(Arrays.asList(BackendType.VULKAN, BackendType.CUDA, BackendType.OPENCL));

    /** Whose ranking winner becomes the target device when no filter names one: OpenCL measures. */
    private static final List<BackendType> TARGET_SOURCES =
            Collections.unmodifiableList(Arrays.asList(BackendType.OPENCL, BackendType.VULKAN, BackendType.CUDA));

    private BackendSelector() {
    }

    /** Creates the backend of one candidate device; may throw when the driver refuses. */
    @FunctionalInterface
    interface BackendFactory {
        ComputeBackend create();
    }

    /** One device as one API sees it, with the means to build a backend on it. Immutable. */
    static final class Candidate {
        final BackendType api;
        final String name;
        final DeviceIdentity identity;
        final DeviceClass deviceClass;
        final BackendFactory factory;

        Candidate(BackendType api, DeviceIdentity identity, DeviceClass deviceClass, BackendFactory factory) {
            if (api == null || identity == null || factory == null) {
                throw new IllegalArgumentException("candidate api, identity and factory must not be null");
            }
            this.api = api;
            this.identity = identity;
            this.name = identity.name().isEmpty() ? "<unnamed>" : identity.name();
            this.deviceClass = deviceClass == null ? DeviceClass.UNKNOWN : deviceClass;
            this.factory = factory;
        }

        @Override
        public String toString() {
            return label(api) + " on '" + name + "'";
        }
    }

    /** What one API offers: its devices, its own ranking's winner, and why it offers none. Immutable. */
    static final class ApiView {
        final BackendType api;
        final List<Candidate> devices;
        final Candidate winner;
        final String absentReason;

        /**
         * @param devices      every usable device of the API, in enumeration order
         * @param winner       the API ranking's choice among {@code devices}; null only when there are none
         * @param absentReason why the API offers no device; consulted only when {@code devices} is empty
         */
        ApiView(BackendType api, List<Candidate> devices, Candidate winner, String absentReason) {
            if (api == null || devices == null) {
                throw new IllegalArgumentException("api view needs an api and a device list");
            }
            if (!devices.isEmpty() && (winner == null || !devices.contains(winner))) {
                throw new IllegalArgumentException(label(api) + " view: the winner must be one of its devices");
            }
            for (Candidate c : devices) {
                if (c == null || c.api != api) {
                    throw new IllegalArgumentException(label(api) + " view holds a device of another API: " + c);
                }
            }
            this.api = api;
            this.devices = Collections.unmodifiableList(new ArrayList<>(devices));
            this.winner = devices.isEmpty() ? null : winner;
            this.absentReason = absentReason == null || absentReason.isEmpty()
                    ? label(api) + ": no devices found" : absentReason;
        }

        /** A view of an API that offers no device. */
        static ApiView absent(BackendType api, String reason) {
            return new ApiView(api, Collections.<Candidate>emptyList(), null, reason);
        }
    }

    /** Result of {@link #selectAuto}: the accepted backend, or null, and every reason a candidate was passed over. */
    static final class Selection {
        final ComputeBackend backend;
        final Candidate chosen;
        final List<String> reasons;

        Selection(ComputeBackend backend, Candidate chosen, List<String> reasons) {
            this.backend = backend;
            this.chosen = chosen;
            this.reasons = Collections.unmodifiableList(new ArrayList<>(reasons));
        }
    }

    /**
     * Runs the AUTO policy.
     *
     * @param views        one view per enumerated API; APIs absent from the list offer nothing
     * @param order        the APIs AUTO may use, in preference order
     * @param deviceFilter case-insensitive device-name substring, or null for none
     * @param requirement  condition a created backend must meet
     */
    static Selection selectAuto(List<ApiView> views, List<BackendType> order, String deviceFilter,
                                BackendRequirement requirement) {
        if (views == null || order == null || requirement == null) {
            throw new IllegalArgumentException("views, order and requirement must not be null");
        }
        Map<BackendType, ApiView> byApi = new EnumMap<>(BackendType.class);
        for (ApiView view : views) {
            if (byApi.put(view.api, view) != null) {
                throw new IllegalArgumentException("two views of " + label(view.api));
            }
        }
        List<String> reasons = new ArrayList<>();
        List<Candidate> attempted = new ArrayList<>();

        Candidate target = target(byApi, deviceFilter);
        if (target != null) {
            LOGGER.info("AUTO backend selection targets device '{}' ({})", target.name, target.deviceClass);
            for (BackendType api : order) {
                ApiView view = byApi.get(api);
                if (view == null || view.devices.isEmpty()) {
                    continue;
                }
                Candidate onTarget = viewOf(view, target);
                if (onTarget == null) {
                    LOGGER.info("AUTO: {} has no view of target device '{}'", label(api), target.name);
                    continue;
                }
                attempted.add(onTarget);
                ComputeBackend backend = tryCandidate(onTarget, requirement, reasons);
                if (backend != null) {
                    return new Selection(backend, onTarget, reasons);
                }
            }
        }

        List<Candidate> fallback = new ArrayList<>();
        for (BackendType api : order) {
            ApiView view = byApi.get(api);
            if (view != null && view.winner != null && !attempted.contains(view.winner)) {
                fallback.add(view.winner);
            }
        }
        fallback.sort(Comparator.<Candidate>comparingInt(c -> classRank(c.deviceClass))
                .thenComparingInt(c -> order.indexOf(c.api)));
        for (Candidate candidate : fallback) {
            ComputeBackend backend = tryCandidate(candidate, requirement, reasons);
            if (backend != null) {
                return new Selection(backend, candidate, reasons);
            }
        }

        for (BackendType api : order) {
            ApiView view = byApi.get(api);
            if (view == null) {
                reasons.add(label(api) + ": not enumerated");
            } else if (view.devices.isEmpty()) {
                reasons.add(view.absentReason);
            }
        }
        return new Selection(null, null, reasons);
    }

    /**
     * Applies a requirement to a backend an explicit mode created: returns it when accepted,
     * otherwise closes it and throws.
     *
     * @throws NoBackendAvailableException {@code "<API> backend unavailable: <reason>"} when rejected
     */
    static ComputeBackend requireAccepted(BackendType api, ComputeBackend backend, BackendRequirement requirement) {
        if (api == null || backend == null || requirement == null) {
            throw new IllegalArgumentException("api, backend and requirement must not be null");
        }
        String reason = checkRequirement(requirement, backend);
        if (reason == null) {
            return backend;
        }
        closeRejected(backend, label(api));
        throw new NoBackendAvailableException(label(api) + " backend unavailable: " + reason);
    }

    /**
     * The elements whose name contains {@code filter}, case-insensitively; all of them, with a
     * warning, when the filter is empty or matches none.
     */
    static <T> List<T> filterByName(List<T> items, Function<T, String> nameOf, String filter, String what) {
        if (items == null || items.isEmpty() || filter == null || filter.trim().isEmpty()) {
            return items == null ? Collections.<T>emptyList() : items;
        }
        String needle = filter.trim().toLowerCase(Locale.ROOT);
        List<T> matching = new ArrayList<>();
        for (T item : items) {
            String name = nameOf.apply(item);
            if (name != null && name.toLowerCase(Locale.ROOT).contains(needle)) {
                matching.add(item);
            }
        }
        if (matching.isEmpty()) {
            LOGGER.warn("-D{}={} matched none of {} {} device(s); ranking normally", DEVICE_PROPERTY, filter,
                    items.size(), what);
            return items;
        }
        return Collections.unmodifiableList(matching);
    }

    /** The value of {@code -Dneuralfree.device}, trimmed, or null when unset or blank. */
    static String deviceFilter() {
        String value = System.getProperty(DEVICE_PROPERTY);
        return value == null || value.trim().isEmpty() ? null : value.trim();
    }

    /**
     * The API order from {@code -Dneuralfree.backend.order}, or {@link #DEFAULT_ORDER} when unset.
     *
     * @throws IllegalArgumentException when the property names an unknown or repeated API
     */
    static List<BackendType> apiOrder() {
        return parseOrder(System.getProperty(ORDER_PROPERTY));
    }

    /**
     * Parses a comma-separated API list ({@code vulkan}, {@code cuda}, {@code opencl}, any case).
     * Null or blank gives {@link #DEFAULT_ORDER}.
     *
     * @throws IllegalArgumentException when a token is unknown or repeated, or the list is empty
     */
    static List<BackendType> parseOrder(String spec) {
        if (spec == null || spec.trim().isEmpty()) {
            return DEFAULT_ORDER;
        }
        Set<BackendType> seen = EnumSet.noneOf(BackendType.class);
        List<BackendType> order = new ArrayList<>();
        for (String token : spec.split(",")) {
            String t = token.trim();
            if (t.isEmpty()) {
                continue;
            }
            BackendType api = parseApi(t);
            if (!seen.add(api)) {
                throw new IllegalArgumentException("-D" + ORDER_PROPERTY + "=" + spec + " lists " + t + " twice");
            }
            order.add(api);
        }
        if (order.isEmpty()) {
            throw new IllegalArgumentException("-D" + ORDER_PROPERTY + "=" + spec + " names no API");
        }
        return Collections.unmodifiableList(order);
    }

    /** The name users read: OpenCL, CUDA or Vulkan. */
    static String label(BackendType api) {
        switch (api) {
            case OPENCL:
                return "OpenCL";
            case CUDA:
                return "CUDA";
            case VULKAN:
                return "Vulkan";
            default:
                return api.name();
        }
    }

    /**
     * Fallback preference of a device class, lower first: discrete, integrated, virtual, then a
     * device whose class is unknown (most likely a GPU), and a CPU device last.
     */
    static int classRank(DeviceClass deviceClass) {
        switch (deviceClass) {
            case DISCRETE:
                return 0;
            case INTEGRATED:
                return 1;
            case VIRTUAL:
                return 2;
            case UNKNOWN:
                return 3;
            case CPU:
            default:
                return 4;
        }
    }

    private static BackendType parseApi(String token) {
        switch (token.toLowerCase(Locale.ROOT)) {
            case "vulkan":
                return BackendType.VULKAN;
            case "cuda":
                return BackendType.CUDA;
            case "opencl":
                return BackendType.OPENCL;
            default:
                throw new IllegalArgumentException("-D" + ORDER_PROPERTY + " names unknown API '" + token
                        + "'; expected vulkan, cuda or opencl");
        }
    }

    /** The device AUTO aims for, or null when no API offers any device. */
    private static Candidate target(Map<BackendType, ApiView> byApi, String deviceFilter) {
        if (deviceFilter != null && !deviceFilter.trim().isEmpty()) {
            String needle = deviceFilter.trim().toLowerCase(Locale.ROOT);
            for (BackendType api : TARGET_SOURCES) {
                ApiView view = byApi.get(api);
                if (view == null) {
                    continue;
                }
                for (Candidate c : view.devices) {
                    if (c.name.toLowerCase(Locale.ROOT).contains(needle)) {
                        LOGGER.info("AUTO target device '{}': matched -D{}={}", c.name, DEVICE_PROPERTY, deviceFilter);
                        return c;
                    }
                }
            }
            LOGGER.warn("-D{}={} matched no device of any API; selecting normally", DEVICE_PROPERTY, deviceFilter);
        }
        for (BackendType api : TARGET_SOURCES) {
            ApiView view = byApi.get(api);
            if (view != null && view.winner != null) {
                return view.winner;
            }
        }
        return null;
    }

    /** The API's view of the target device: its ranking winner when that is the device, else the first match. */
    private static Candidate viewOf(ApiView view, Candidate target) {
        if (view.winner == target || view.winner.identity.samePhysicalDevice(target.identity)) {
            return view.winner;
        }
        for (Candidate c : view.devices) {
            if (c == target || c.identity.samePhysicalDevice(target.identity)) {
                return c;
            }
        }
        return null;
    }

    /** Creates and checks one candidate; null, with the reason recorded and logged, when it is not usable. */
    private static ComputeBackend tryCandidate(Candidate candidate, BackendRequirement requirement,
                                               List<String> reasons) {
        ComputeBackend backend;
        try {
            backend = candidate.factory.create();
        } catch (RuntimeException | LinkageError e) {
            skip(candidate, "backend creation failed: " + describe(e), reasons);
            return null;
        }
        if (backend == null) {
            skip(candidate, "backend creation returned nothing", reasons);
            return null;
        }
        String reason = checkRequirement(requirement, backend);
        if (reason != null) {
            closeRejected(backend, candidate.toString());
            skip(candidate, reason, reasons);
            return null;
        }
        LOGGER.info("Auto-detection selected {} backend on '{}'", label(candidate.api), candidate.name);
        return backend;
    }

    private static String checkRequirement(BackendRequirement requirement, ComputeBackend backend) {
        try {
            return requirement.check(backend);
        } catch (RuntimeException e) {
            return "requirement check failed: " + describe(e);
        }
    }

    private static void skip(Candidate candidate, String reason, List<String> reasons) {
        LOGGER.info("AUTO skipped {}: {}", candidate, reason);
        reasons.add(candidate + ": " + reason);
    }

    private static void closeRejected(ComputeBackend backend, String what) {
        try {
            backend.close();
        } catch (RuntimeException e) {
            LOGGER.warn("Closing the rejected {} backend failed: {}", what, describe(e));
        }
    }

    private static String describe(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isEmpty() ? t.getClass().getSimpleName() : message;
    }
}
