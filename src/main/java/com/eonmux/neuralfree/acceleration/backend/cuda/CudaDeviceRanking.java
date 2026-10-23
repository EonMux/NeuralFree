package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;

/**
 * Picks the fastest CUDA device: a forced name match first, then multiprocessors
 * times clock with integrated devices ranked below every discrete one.
 *
 * <p>Unlike the OpenCL list this one is single-vendor, so multiprocessor count and
 * clock really are comparable across its entries and no timed probe is needed.
 * Integrated parts (Tegra, the iGPU half of a laptop) lose to any discrete device
 * present, matching {@link com.eonmux.neuralfree.acceleration.ocl.CLDeviceRanking}
 * for OpenCL.</p>
 *
 * <ol>
 *   <li><b>An explicit choice wins.</b> {@code -Dneuralfree.cuda.device=<i>substring</i>}
 *       (or, when unset, {@code -Dneuralfree.device}) selects the first device whose
 *       name contains that text, case-insensitively. A filter that matches nothing is
 *       reported and the ranking proceeds normally.</li>
 *   <li><b>Multiprocessors times clock</b>, negated for integrated devices. A device
 *       whose multiprocessor count or clock could not be read ranks below every device
 *       that answered, but stays in the running. Ties keep enumeration order.</li>
 * </ol>
 */
public final class CudaDeviceRanking {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaDeviceRanking.class);

    /** Forces a CUDA device by name substring, case-insensitive. */
    public static final String DEVICE_PROPERTY = "neuralfree.cuda.device";
    /** API-independent device filter, consulted when {@link #DEVICE_PROPERTY} is unset. */
    public static final String GLOBAL_DEVICE_PROPERTY = "neuralfree.device";

    /** Profile value of a numeric attribute whose query failed. */
    static final int UNKNOWN = -1;

    private CudaDeviceRanking() {
    }

    /** The device to run on, or null when the list is empty. Never throws for a device that cannot be queried. */
    public static CUDevice fastest(List<CUDevice> devices) {
        if (devices == null || devices.isEmpty()) {
            return null;
        }
        List<Profile> profiles = new ArrayList<>(devices.size());
        List<CUDevice> candidates = new ArrayList<>(devices.size());
        for (CUDevice device : devices) {
            if (device != null) {
                profiles.add(profileOf(device));
                candidates.add(device);
            }
        }
        int winner = rank(profiles);
        if (winner < 0) {
            return null;
        }
        CUDevice best = candidates.get(winner);
        if (candidates.size() > 1) {
            LOGGER.info("Selected CUDA device '{}' of {} available", best.getDeviceName(), candidates.size());
        }
        return best;
    }

    /** Index of the winning profile, or -1 when the list is empty. */
    static int rank(List<Profile> profiles) {
        if (profiles == null || profiles.isEmpty()) {
            return -1;
        }
        int forced = forcedChoice(profiles);
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

    /** Multiprocessors times clock, negated when integrated; negative infinity when unmeasured. */
    static double score(Profile p) {
        if (p.multiprocessors == UNKNOWN || p.clockKhz == UNKNOWN) {
            return Double.NEGATIVE_INFINITY;
        }
        double score = (double) Math.max(1, p.multiprocessors) * Math.max(1, p.clockKhz);
        // A rank, not a ratio: enough to put every discrete device above every integrated one.
        return Boolean.TRUE.equals(p.integrated) ? -score : score;
    }

    private static int forcedChoice(List<Profile> profiles) {
        String property = DEVICE_PROPERTY;
        String wanted = System.getProperty(DEVICE_PROPERTY);
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
                LOGGER.info("Selected CUDA device '{}': matched -D{}={}", profiles.get(i).name, property, wanted);
                return i;
            }
        }
        LOGGER.warn("-D{}={} matched none of {} CUDA device(s); ranking normally", property, wanted, profiles.size());
        return -1;
    }

    private static Profile profileOf(CUDevice device) {
        return new Profile(device.getDeviceName(),
                integratedOf(device),
                attributeOf(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "multiprocessor count"),
                attributeOf(device, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, "clock rate"));
    }

    private static Boolean integratedOf(CUDevice device) {
        try {
            return device.getAttribute(CU_DEVICE_ATTRIBUTE_INTEGRATED) != 0;
        } catch (RuntimeException e) {
            LOGGER.debug("CUDA integrated query failed for '{}': {}", device.getDeviceName(), e.toString());
            return null;
        }
    }

    private static int attributeOf(CUDevice device, int attribute, String what) {
        try {
            return Math.max(0, device.getAttribute(attribute));
        } catch (RuntimeException e) {
            LOGGER.debug("CUDA {} query failed for '{}': {}", what, device.getDeviceName(), e.toString());
            return UNKNOWN;
        }
    }

    /** What the ranking knows about one candidate, separate from {@link CUDevice} so the policy is testable. */
    static final class Profile {
        final String name;
        /** Null when the attribute query failed: unknown, not integrated. */
        final Boolean integrated;
        /** {@link #UNKNOWN} when the query failed. */
        final int multiprocessors;
        /** {@link #UNKNOWN} when the query failed. */
        final int clockKhz;

        Profile(String name, Boolean integrated, int multiprocessors, int clockKhz) {
            this.name = name == null ? "<unnamed>" : name;
            this.integrated = integrated;
            this.multiprocessors = multiprocessors;
            this.clockKhz = clockKhz;
        }
    }
}
