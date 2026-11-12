package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.CudaException;
import jcuda.driver.CUctx_flags;
import jcuda.driver.CUdevice;
import jcuda.driver.CUresult;
import org.apache.logging.log4j.Logger;

import java.util.Locale;

import static jcuda.driver.JCudaDriver.cuDevicePrimaryCtxGetState;
import static jcuda.driver.JCudaDriver.cuDevicePrimaryCtxSetFlags;

/**
 * How a host thread waits for the device in a CUDA primary context ({@code CU_CTX_SCHED_*}), chosen with
 * {@value #PROPERTY}: {@code auto} (the driver's default: spin while the process has fewer active contexts than
 * logical processors, yield otherwise), {@code spin}, {@code yield} or {@code block} ({@code BLOCKING_SYNC}, which
 * frees the waiting core at the cost of wake-up latency). The flag belongs to the device's primary context, so it is
 * set before the context first becomes active; once a context of the device is active, a different request is
 * reported and the active scheduling stays.
 */
public enum CudaScheduling {
    AUTO("auto", CUctx_flags.CU_CTX_SCHED_AUTO),
    SPIN("spin", CUctx_flags.CU_CTX_SCHED_SPIN),
    YIELD("yield", CUctx_flags.CU_CTX_SCHED_YIELD),
    BLOCK("block", CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC);

    /** System property naming the scheduling; unset leaves the primary context's flags as they are. */
    public static final String PROPERTY = "neuralfree.cuda.sched";

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaScheduling.class);

    private final String flagName;
    private final int flag;

    CudaScheduling(String flagName, int flag) {
        this.flagName = flagName;
        this.flag = flag;
    }

    /** The property value naming this scheduling. */
    public String flagName() {
        return flagName;
    }

    /** The {@code CU_CTX_SCHED_*} flag. */
    public int flag() {
        return flag;
    }

    /**
     * The scheduling {@value #PROPERTY} names, or null when it is unset or blank.
     *
     * @throws IllegalArgumentException when the value names no scheduling
     */
    public static CudaScheduling configured() {
        return parse(System.getProperty(PROPERTY));
    }

    /**
     * The scheduling {@code value} names (case-insensitive), or null for null or blank.
     *
     * @throws IllegalArgumentException when the value names no scheduling
     */
    public static CudaScheduling parse(String value) {
        if (value == null || value.trim().isEmpty()) {
            return null;
        }
        String v = value.trim().toLowerCase(Locale.ROOT);
        for (CudaScheduling s : values()) {
            if (s.flagName.equals(v)) {
                return s;
            }
        }
        throw new IllegalArgumentException(PROPERTY + "=" + value + " names no CUDA scheduling; use auto, spin,"
                + " yield or block");
    }

    /** The scheduling a context's flags hold. */
    static CudaScheduling ofFlags(int flags) {
        int sched = flags & CUctx_flags.CU_CTX_SCHED_MASK;
        for (CudaScheduling s : values()) {
            if (s.flag == sched) {
                return s;
            }
        }
        return AUTO;
    }

    /**
     * Sets this scheduling on {@code device}'s primary context before it becomes active, keeping its other flags.
     * An already active context with another scheduling keeps it, with a warning.
     *
     * @throws RuntimeException the translated driver failure of the state query or the flag update
     */
    void applyTo(CUdevice device) {
        int[] flags = new int[1];
        int[] active = new int[1];
        CudaErrors.check("cuDevicePrimaryCtxGetState", () -> cuDevicePrimaryCtxGetState(device, flags, active));
        CudaScheduling current = ofFlags(flags[0]);
        if (current == this) {
            return;
        }
        if (active[0] != 0) {
            warnActive(device, current);
            return;
        }
        int updated = (flags[0] & ~CUctx_flags.CU_CTX_SCHED_MASK) | flag;
        int result;
        try {
            result = cuDevicePrimaryCtxSetFlags(device, updated);
        } catch (CudaException e) {
            result = CudaErrors.codeOf(e);
        }
        if (result == CUresult.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) {
            warnActive(device, current);
            return;
        }
        CudaErrors.call(result, "cuDevicePrimaryCtxSetFlags(" + flagName + ")");
        LOGGER.info("CUDA {}: host threads wait with {} scheduling ({}={})", device, name().toLowerCase(Locale.ROOT),
                PROPERTY, flagName);
    }

    private void warnActive(CUdevice device, CudaScheduling current) {
        LOGGER.warn("CUDA {}: {}={} not applied, the device's primary context is already active with {}"
                + " scheduling; set it before the first CUDA backend of the process opens", device, PROPERTY,
                flagName, current.flagName);
    }
}
