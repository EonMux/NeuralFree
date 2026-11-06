package com.eonmux.neuralfree.acceleration.backend.spi;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Picks the fastest device of one accelerator API: a forced name match first, then compute units
 * times clock with integrated devices ranked below every discrete one.
 *
 * <p>The list is single-vendor, so the attributes the provider reports are comparable across its
 * entries and no timed probe is needed, as for CUDA and HIP. A device whose compute-unit count or
 * clock is unknown ranks below every device that reports both, but stays in the running; ties keep
 * enumeration order.</p>
 *
 * <ol>
 *   <li><b>An explicit choice wins.</b> {@code -Dneuralfree.<i>api</i>.device=<i>substring</i>}
 *       (for example {@code neuralfree.sycl.device}), or, when unset, {@code -Dneuralfree.device},
 *       selects the first device whose name contains that text, case-insensitively. A filter that
 *       matches nothing is reported and the ranking proceeds normally.</li>
 *   <li><b>Compute units times clock</b>, negated for integrated devices.</li>
 * </ol>
 */
public final class AcceleratorDeviceRanking {

    private static final Logger LOGGER = LoggingUtil.getLogger(AcceleratorDeviceRanking.class);

    /** API-independent device filter, consulted when the API's own property is unset. */
    public static final String GLOBAL_DEVICE_PROPERTY = "neuralfree.device";

    private AcceleratorDeviceRanking() {
    }

    /**
     * The device to run on, or null when the list is empty.
     *
     * @throws IllegalArgumentException when {@code api} is null or not an accelerator API
     */
    public static AcceleratorDevice fastest(List<AcceleratorDevice> devices, BackendType api) {
        if (api == null || !api.isAccelerator()) {
            throw new IllegalArgumentException("API must be an accelerator type, got " + api);
        }
        if (devices == null || devices.isEmpty()) {
            return null;
        }
        List<Profile> profiles = new ArrayList<>(devices.size());
        List<AcceleratorDevice> candidates = new ArrayList<>(devices.size());
        for (AcceleratorDevice device : devices) {
            if (device != null) {
                profiles.add(Profile.of(device));
                candidates.add(device);
            }
        }
        int winner = rank(profiles, api);
        if (winner < 0) {
            return null;
        }
        AcceleratorDevice best = candidates.get(winner);
        if (candidates.size() > 1) {
            LOGGER.info("Selected {} device '{}' of {} available", api.label(), best.name(), candidates.size());
        }
        return best;
    }

    /** The device-name property that forces one device of an API. */
    static String deviceProperty(BackendType api) {
        return "neuralfree." + api.name().toLowerCase(Locale.ROOT) + ".device";
    }

    /** Index of the winning profile, or -1 when the list is empty. */
    static int rank(List<Profile> profiles, BackendType api) {
        if (profiles == null || profiles.isEmpty()) {
            return -1;
        }
        int forced = forcedChoice(profiles, api);
        if (forced >= 0) {
            return forced;
        }
        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < profiles.size(); i++) {
            double score = score(profiles.get(i));
            if (best < 0 || score > bestScore) {
                best = i;
                bestScore = score;
            }
        }
        return best;
    }

    /** Compute units times clock, negated when integrated; negative infinity when unmeasured. */
    static double score(Profile p) {
        if (p.computeUnits <= 0 || p.clockMhz <= 0) {
            return Double.NEGATIVE_INFINITY;
        }
        double score = (double) p.computeUnits * p.clockMhz;
        // A rank, not a ratio: enough to put every discrete device above every integrated one.
        return p.deviceClass == DeviceClass.INTEGRATED ? -score : score;
    }

    private static int forcedChoice(List<Profile> profiles, BackendType api) {
        String property = deviceProperty(api);
        String wanted = System.getProperty(property);
        if (wanted == null || wanted.trim().isEmpty()) {
            property = GLOBAL_DEVICE_PROPERTY;
            wanted = System.getProperty(GLOBAL_DEVICE_PROPERTY);
        }
        if (wanted == null || wanted.trim().isEmpty()) {
            return -1;
        }
        String needle = wanted.trim().toLowerCase(Locale.ROOT);
        for (int i = 0; i < profiles.size(); i++) {
            if (profiles.get(i).name.toLowerCase(Locale.ROOT).contains(needle)) {
                LOGGER.info("Selected {} device '{}': matched -D{}={}", api.label(), profiles.get(i).name, property,
                        wanted);
                return i;
            }
        }
        LOGGER.warn("-D{}={} matched none of {} {} device(s); ranking normally", property, wanted, profiles.size(),
                api.label());
        return -1;
    }

    /** The attributes the ranking compares: a device's name, throughput and class. */
    static final class Profile {
        final String name;
        final int computeUnits;
        final int clockMhz;
        final DeviceClass deviceClass;

        private Profile(String name, int computeUnits, int clockMhz, DeviceClass deviceClass) {
            this.name = name;
            this.computeUnits = computeUnits;
            this.clockMhz = clockMhz;
            this.deviceClass = deviceClass;
        }

        static Profile of(AcceleratorDevice device) {
            Objects.requireNonNull(device, "device");
            DeviceCapabilities caps = device.capabilities();
            return new Profile(device.name(), caps.getComputeUnits(), caps.getMaxClockFrequencyMhz(),
                    device.deviceClass());
        }
    }
}
