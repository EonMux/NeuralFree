package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorBackendProvider;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorDevice;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorDeviceRanking;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorProviders;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The accelerator devices one {@link HWAcceleration} enumerated, with the policy that ranks them
 * and opens their backends: the devices of each enumerated accelerator API, the reason an API
 * offers none, and the {@link BackendSelector.ApiView} AUTO sees.
 *
 * <p>A provider that throws, that returns no list, or that enumerates a device of another API is
 * recorded as that API's reason instead of failing the instance; an API without an installed
 * provider records how to install one. The set is not thread-safe; {@link HWAcceleration} builds it
 * during construction and clears it on shutdown.</p>
 */
final class AcceleratorBackendSet {

    private static final Logger LOGGER = LoggingUtil.getLogger(AcceleratorBackendSet.class);

    private final AcceleratorProviders providers;
    private final Map<BackendType, List<AcceleratorDevice>> devices = new EnumMap<>(BackendType.class);
    private final Map<BackendType, String> failures = new EnumMap<>(BackendType.class);

    AcceleratorBackendSet(AcceleratorProviders providers) {
        this.providers = Objects.requireNonNull(providers, "providers must not be null");
    }

    /**
     * Enumerates one accelerator API through its installed provider.
     *
     * @throws IllegalArgumentException when {@code api} is not an accelerator API
     */
    void enumerate(BackendType api) {
        if (api == null || !api.isAccelerator()) {
            throw new IllegalArgumentException("API must be an accelerator type, got " + api);
        }
        String failure = providers.failureReason(api);
        if (failure != null) {
            failures.put(api, failure);
            return;
        }
        AcceleratorBackendProvider provider = providers.providerFor(api);
        if (provider == null) {
            failures.put(api, AcceleratorProviders.missingProviderReason(api));
            return;
        }
        String providerName = provider.getClass().getName();
        try {
            List<AcceleratorDevice> reported = provider.devices();
            if (reported == null) {
                throw new IllegalStateException("the provider returned no device list");
            }
            List<AcceleratorDevice> usable = new ArrayList<>(reported.size());
            for (AcceleratorDevice device : reported) {
                if (device == null) {
                    continue;
                }
                if (device.backendType() != api) {
                    throw new IllegalStateException("the provider enumerated a " + device.backendType().label()
                            + " device '" + device.name() + "' but serves " + api.label());
                }
                usable.add(device);
            }
            devices.put(api, Collections.unmodifiableList(usable));
            LOGGER.info("Found {} {} capable device(s)", usable.size(), api.label());
        } catch (RuntimeException | LinkageError e) {
            String reason = "provider " + providerName + " failed to enumerate devices: " + describe(e);
            failures.put(api, reason);
            LOGGER.warn("{} initialization failed: {}", api.label(), reason);
        }
    }

    /** The devices an API enumerated, in provider order; empty when it reported none. */
    List<AcceleratorDevice> devices(BackendType api) {
        List<AcceleratorDevice> found = devices.get(api);
        return found == null ? Collections.<AcceleratorDevice>emptyList() : found;
    }

    /** The API's own ranking winner, or null when it has no device. */
    AcceleratorDevice fastest(BackendType api) {
        return AcceleratorDeviceRanking.fastest(devices(api), api);
    }

    /**
     * Why an API offers no device: its provider failure when recorded, "no devices found" when it
     * was enumerated, otherwise how to enumerate it.
     */
    String reason(BackendType api, boolean enumerated) {
        if (!enumerated) {
            return api.label() + ": not enumerated by this instance (use HWAcceleration.forMode or include it in"
                    + " the API set)";
        }
        String failure = failures.get(api);
        return failure != null ? api.label() + ": " + failure : api.label() + ": no devices found";
    }

    /** What an accelerator API offers AUTO: its devices as candidates and the ranking's winner. */
    BackendSelector.ApiView view(BackendType api, boolean enumerated) {
        List<AcceleratorDevice> reported = devices(api);
        if (reported.isEmpty()) {
            return BackendSelector.ApiView.absent(api, reason(api, enumerated));
        }
        AcceleratorDevice best = fastest(api);
        List<BackendSelector.Candidate> candidates = new ArrayList<>(reported.size());
        BackendSelector.Candidate winner = null;
        for (AcceleratorDevice device : reported) {
            BackendSelector.Candidate candidate = new BackendSelector.Candidate(api, device.identity(),
                    device.deviceClass(), device::openBackend);
            candidates.add(candidate);
            if (device == best) {
                winner = candidate;
            }
        }
        return new BackendSelector.ApiView(api, candidates, winner == null ? candidates.get(0) : winner, null);
    }

    /**
     * Opens the backend of the API's fastest device.
     *
     * @throws NoBackendAvailableException when the API has no device, no provider, or the provider
     *                                     cannot open its backend
     */
    ComputeBackend openFastestOrThrow(BackendType api) {
        List<AcceleratorDevice> reported = devices(api);
        if (reported.isEmpty()) {
            throw new NoBackendAvailableException(api.label() + " backend requested but no " + api.label()
                    + " devices available. Install the vendor runtime and a " + api.label()
                    + " backend provider jar (" + AcceleratorBackendProvider.class.getName()
                    + " in META-INF/services). Cause: " + reason(api, true));
        }
        try {
            AcceleratorDevice device = AcceleratorDeviceRanking.fastest(reported, api);
            if (device != null) {
                return device.openBackend();
            }
        } catch (RuntimeException | LinkageError e) {
            LOGGER.warn("Failed to create {} backend: {}", api.label(), describe(e));
            throw new NoBackendAvailableException(api.label() + " backend unavailable: backend creation failed: "
                    + describe(e), e);
        }
        throw new NoBackendAvailableException(api.label() + " backend unavailable: no device was selected");
    }

    /** Releases the enumerated devices; a later createBackend finds none. */
    void clear() {
        devices.clear();
        failures.clear();
    }

    private static String describe(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isEmpty() ? t.getClass().getSimpleName() : message;
    }
}
