package com.eonmux.neuralfree.acceleration.metal;

import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Picks the fastest Metal device: an explicit name match first, then the recommended working-set
 * size, with integrated GPUs ranked below every discrete one.
 *
 * <ol>
 *   <li><b>An explicit choice wins.</b> {@code -Dneuralfree.metal.device=<i>substring</i>} (or,
 *       when unset, {@code -Dneuralfree.device}) selects the first device whose name contains that
 *       text, case-insensitively. A filter that matches nothing is reported and the ranking
 *       proceeds normally.</li>
 *   <li><b>Working-set size</b>, negated for integrated devices. On Apple silicon every GPU is
 *       integrated, so the size decides; on a Mac with a discrete AMD GPU the discrete device wins
 *       even when its working set is smaller.</li>
 * </ol>
 */
public final class MetalDeviceRanking {

    private static final Logger LOGGER = LoggingUtil.getLogger(MetalDeviceRanking.class);

    /** Forces a Metal device by name substring, case-insensitive. */
    public static final String DEVICE_PROPERTY = "neuralfree.metal.device";
    /** API-independent device filter, consulted when {@link #DEVICE_PROPERTY} is unset. */
    public static final String GLOBAL_DEVICE_PROPERTY = "neuralfree.device";

    private MetalDeviceRanking() {
    }

    /** The device to run on, or null when the list is empty. Never throws for a device that cannot be queried. */
    public static MetalDevice fastest(List<MetalDevice> devices) {
        if (devices == null || devices.isEmpty()) {
            return null;
        }
        List<MetalDevice> candidates = new ArrayList<>(devices.size());
        for (MetalDevice device : devices) {
            if (device != null) {
                candidates.add(device);
            }
        }
        if (candidates.isEmpty()) {
            return null;
        }
        MetalDevice best = forcedChoice(candidates);
        if (best == null) {
            best = candidates.get(0);
            for (MetalDevice device : candidates) {
                if (score(device) > score(best)) {
                    best = device;
                }
            }
        }
        if (candidates.size() > 1) {
            LOGGER.info("Selected Metal device '{}' of {} available", best.getDeviceName(), candidates.size());
        }
        return best;
    }

    private static MetalDevice forcedChoice(List<MetalDevice> devices) {
        String property = DEVICE_PROPERTY;
        String wanted = System.getProperty(DEVICE_PROPERTY);
        if (wanted == null || wanted.trim().isEmpty()) {
            property = GLOBAL_DEVICE_PROPERTY;
            wanted = System.getProperty(GLOBAL_DEVICE_PROPERTY);
        }
        if (wanted == null || wanted.trim().isEmpty()) {
            return null;
        }
        String needle = wanted.trim().toLowerCase(Locale.ROOT);
        for (MetalDevice device : devices) {
            if (device.getDeviceName().toLowerCase(Locale.ROOT).contains(needle)) {
                LOGGER.info("Selected Metal device '{}': matched -D{}={}", device.getDeviceName(), property, wanted);
                return device;
            }
        }
        LOGGER.warn("-D{}={} matched none of {} Metal device(s); ranking normally", property, wanted, devices.size());
        return null;
    }

    private static double score(MetalDevice device) {
        double size;
        try {
            size = Math.max(1L, device.getRecommendedMaxWorkingSet());
        } catch (RuntimeException e) {
            LOGGER.debug("Metal working-set query failed for '{}': {}", device.getDeviceName(), e.toString());
            return Double.NEGATIVE_INFINITY;
        }
        // Discrete devices outrank every integrated one; among integrated GPUs the larger working
        // set wins, so the negated reciprocal keeps the order while staying below zero.
        return device.getDeviceClass() == DeviceClass.DISCRETE ? size : -1.0 / size;
    }
}
