package com.eonmux.neuralfree;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.BackendRequirement;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.tensor.gpu.TrainKernels;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Entry point of the training framework: {@link #main} creates the GPU backend named by {@value #BACKEND_PROPERTY},
 * reports whether the train kernels run on it and checks a buffer round trip; {@link #createTrainingBackend} creates
 * that backend for applications.
 */
public class NeuralFree {

    /**
     * System property naming the API the training framework runs on: {@code auto}, {@code opencl}, {@code cuda} or
     * {@code vulkan} (default {@code auto}). Pass it to the JVM as {@code -Dneuralfree.backend=<api>}; the
     * {@code runDemos} Gradle task forwards {@code -Pneuralfree.backend=<api>} as this property.
     */
    public static final String BACKEND_PROPERTY = "neuralfree.backend";

    private static final Logger LOGGER = LoggingUtil.getLogger(NeuralFree.class);

    /**
     * Backends AUTO accepts for the training framework: the train kernel library is registered and available
     * ({@link TrainKernels#requirement()}), and its context opens there, which also checks the work-group size and
     * prewarms every kernel ({@link TrainKernels#availability}). A backend AUTO rejects is closed, which releases the
     * context it opened.
     */
    private static final BackendRequirement TRAIN_REQUIREMENT =
            TrainKernels.requirement().and(NeuralFree::trainContextFailure);

    private HWAcceleration hardwareAcceleration = null;
    private ComputeBackend computeBackend = null;

    /** Runs the backend check described above; the arguments are ignored. */
    public static void main(String[] args) {
        new NeuralFree();
    }

    public NeuralFree() {
        LoggingUtil.getLogger(this).info("NeuralFree - Copyright (c) 2017-2026 EONMUX LTD and Amir Farhad Eslampanah."
                + " Licensed under the NeuralFree License (see LICENSE).");
        LoggingUtil.getLogger(this).info("Organizations with annual revenue above USD 20M, or projects generating more"
                + " than USD 20M in revenue, need a commercial licence from EONMUX LTD (see COMMERCIAL-LICENSE.md).");

        ACCELERATION_MODE mode;
        try {
            mode = backendFromProperty();
        } catch (IllegalArgumentException e) {
            LoggingUtil.getLogger(this).error(e.getMessage());
            return;
        }

        //Enumerate the APIs the mode can use: all three for AUTO, otherwise only the named one
        hardwareAcceleration = HWAcceleration.forMode(mode);

        //Release the devices if the JVM exits before the work below finishes
        Thread releaseHook = new Thread(hardwareAcceleration::shutdown, "NeuralFree-gpu-release");
        Runtime.getRuntime().addShutdownHook(releaseHook);
        try {
            try {
                computeBackend = createTrainingBackend(hardwareAcceleration, mode);
            } catch (NoBackendAvailableException e) {
                LoggingUtil.getLogger(this).error("No GPU backend available: {}", e.getMessage());
                return;
            }

            DeviceCapabilities caps = computeBackend.capabilities();
            LoggingUtil.getLogger(this).info("ComputeBackend created - API: {}, Device: {}, VRAM: {} MB, Compute Units: {}",
                    computeBackend.backendType(), caps.getDeviceName(), caps.getTotalVramBytes() / (1024 * 1024),
                    caps.getComputeUnits());
            LoggingUtil.getLogger(this).info("Train kernels: {}", describeTrainKernels(computeBackend));

            testComputeBackendBuffers(computeBackend);

            LoggingUtil.getLogger(this).info("ComputeBackend integration test completed successfully");
        } finally {
            releaseDevices(releaseHook);
        }
    }

    /**
     * Releases the devices on the calling thread, unless the JVM is already shutting down: the shutdown hook then
     * releases them, and releasing here as well would race with it.
     */
    private void releaseDevices(Thread releaseHook) {
        boolean hookRemoved;
        try {
            hookRemoved = Runtime.getRuntime().removeShutdownHook(releaseHook);
        } catch (IllegalStateException e) {
            LoggingUtil.getLogger(this).info("JVM shutdown in progress; the shutdown hook releases the GPU devices");
            return;
        }
        if (hookRemoved) {
            //Closes the backend created above, then releases the devices; GPU memory is not managed by the JVM
            hardwareAcceleration.shutdown();
            computeBackend = null;
        }
    }

    /**
     * The acceleration mode named by {@code value}: {@code auto}, {@code opencl}, {@code cuda} or {@code vulkan},
     * case-insensitive and trimmed.
     *
     * @throws IllegalArgumentException when {@code value} is null or names no GPU API
     */
    public static ACCELERATION_MODE parseBackend(String value) {
        if (value == null) {
            throw new IllegalArgumentException("backend must not be null; use auto, opencl, cuda or vulkan");
        }
        String name = value.trim().toUpperCase(Locale.ROOT);
        for (ACCELERATION_MODE mode : ACCELERATION_MODE.values()) {
            if (mode.name().equals(name)) {
                return mode;
            }
        }
        if ("CPU".equals(name)) {
            throw new IllegalArgumentException("backend '" + value + "' is invalid here: the training framework runs "
                    + "its tensors on a GPU backend and takes its CPU path per op when the train kernels are "
                    + "unavailable; use auto, opencl, cuda or vulkan");
        }
        throw new IllegalArgumentException("backend '" + value + "' is invalid; use auto, opencl, cuda or vulkan");
    }

    /**
     * The mode {@link #BACKEND_PROPERTY} names, or AUTO when it is unset or empty.
     *
     * @throws IllegalArgumentException when the property names no GPU API
     */
    public static ACCELERATION_MODE backendFromProperty() {
        String value = System.getProperty(BACKEND_PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return ACCELERATION_MODE.AUTO;
        }
        try {
            return parseBackend(value);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(BACKEND_PROPERTY + ": " + e.getMessage(), e);
        }
    }

    /**
     * Creates the backend the training framework runs on.
     *
     * <p>AUTO selects, device first, a backend on which the train kernel library is available. When a device was
     * found but no API serves the library on it, it logs a WARN giving, for each enumerated API, the reason the
     * library is unavailable on each backend AUTO tried or why that API offered none, and selects the backend AUTO
     * picks without that requirement; tensor ops on it take their CPU path. An explicit API creates that API's
     * backend; when the library is unavailable there, the same WARN is logged and the backend is still returned.</p>
     *
     * @param hw   enumerates the devices; {@code hw.shutdown()} closes the returned backend
     * @param mode the API, or AUTO; null means AUTO
     * @return the backend; never null
     * @throws NoBackendAvailableException when no backend of the requested API (or of any API, for AUTO) can be
     *                                     created at all
     */
    public static ComputeBackend createTrainingBackend(HWAcceleration hw, ACCELERATION_MODE mode) {
        Objects.requireNonNull(hw, "hardware acceleration must not be null");
        ACCELERATION_MODE requested = mode == null ? ACCELERATION_MODE.AUTO : mode;
        if (requested == ACCELERATION_MODE.AUTO) {
            return createAutoTrainingBackend(hw);
        }
        ComputeBackend backend = hw.createBackend(requested, BackendRequirement.ANY);
        String failure = TRAIN_REQUIREMENT.check(backend);
        if (failure != null) {
            warnTrainUnavailable(Collections.singletonList(describe(backend) + ": " + failure));
        }
        return backend;
    }

    /**
     * AUTO under {@link #TRAIN_REQUIREMENT}, recording per API why each backend it tried was rejected; without any
     * rejection no device was usable at all, and the selection's exception propagates unchanged.
     */
    private static ComputeBackend createAutoTrainingBackend(HWAcceleration hw) {
        Map<BackendType, List<String>> rejections = new EnumMap<>(BackendType.class);
        BackendRequirement recording = backend -> {
            String reason;
            try {
                reason = TRAIN_REQUIREMENT.check(backend);
            } catch (RuntimeException e) {
                recordRejection(rejections, backend, "opening the train library failed: " + e.getMessage());
                throw e;
            }
            if (reason != null) {
                recordRejection(rejections, backend, reason);
            }
            return reason;
        };
        try {
            return hw.createBackend(ACCELERATION_MODE.AUTO, recording);
        } catch (NoBackendAvailableException e) {
            if (rejections.isEmpty()) {
                throw e;
            }
            warnTrainUnavailable(reasonsPerApi(hw, rejections));
            return hw.createBackend(ACCELERATION_MODE.AUTO, BackendRequirement.ANY);
        }
    }

    private static void recordRejection(Map<BackendType, List<String>> rejections, ComputeBackend backend,
                                        String reason) {
        rejections.computeIfAbsent(backend.backendType(), api -> new ArrayList<>())
                .add(describe(backend) + ": " + reason);
    }

    /**
     * One entry per API {@code hw} enumerated: the train-library reason of each of its backends AUTO rejected, else
     * why AUTO had no backend of that API to check.
     */
    private static List<String> reasonsPerApi(HWAcceleration hw, Map<BackendType, List<String>> rejections) {
        List<String> reasons = new ArrayList<>();
        for (BackendType api : BackendType.values()) {
            if (!hw.getEnumeratedApis().contains(api)) {
                continue;
            }
            List<String> rejected = rejections.get(api);
            if (rejected != null) {
                reasons.addAll(rejected);
            } else if (deviceCount(hw, api) == 0) {
                reasons.add(api + ": no usable device was enumerated (see the initialization log)");
            } else {
                reasons.add(api + ": no backend could be created on its devices (see the AUTO selection log)");
            }
        }
        return reasons;
    }

    private static int deviceCount(HWAcceleration hw, BackendType api) {
        switch (api) {
            case OPENCL:
                return hw.getCLDevices().size();
            case CUDA:
                return hw.getCUDevices().size();
            case VULKAN:
                return hw.getVkDevices().size();
            default:
                throw new IllegalArgumentException("Unknown backend type: " + api);
        }
    }

    private static void warnTrainUnavailable(List<String> reasons) {
        LOGGER.warn("The train kernel library is unavailable, so tensor ops take their CPU path. Availability per "
                + "API: {}", String.join("; ", reasons));
    }

    private static String describe(ComputeBackend backend) {
        return backend.backendType() + " on '" + backend.capabilities().getDeviceName() + "'";
    }

    /**
     * "available", or "unavailable (tensor ops take their CPU path): reason" with the cached reason of
     * {@link TrainKernels#availability}, which opens the library on the first call for a backend.
     */
    public static String describeTrainKernels(ComputeBackend backend) {
        Availability availability = TrainKernels.availability(Objects.requireNonNull(backend, "backend"));
        return availability.available() ? "available"
                : "unavailable (tensor ops take their CPU path): " + availability.reason();
    }

    /** Null when the train context opens on {@code backend}, otherwise the cached reason it does not. */
    private static String trainContextFailure(ComputeBackend backend) {
        Availability availability = TrainKernels.availability(backend);
        return availability.available() ? null : availability.reason();
    }

    /**
     * Test ComputeBackend buffer operations to verify integration
     */
    private void testComputeBackendBuffers(ComputeBackend backend) {
        LoggingUtil.getLogger(this).info("Testing ComputeBackend buffer operations...");

        ComputeQueue transferQueue = backend.transferQueue();

        int numFloats = 16;
        float[] srcData = new float[numFloats];
        float[] dstData = new float[numFloats];
        for (int i = 0; i < numFloats; i++) {
            srcData[i] = i * 1.5f;
        }

        try (GpuBuffer buffer = backend.allocateBuffer(numFloats * Float.BYTES, false)) {
            LoggingUtil.getLogger(this).info("Allocated buffer: {} bytes", buffer.sizeInBytes());

            try (GpuEvent writeEvent = transferQueue.enqueueWriteBuffer(buffer, 0, numFloats * Float.BYTES, srcData)) {
                LoggingUtil.getLogger(this).info("Enqueued write buffer operation");
                awaitTransfer(writeEvent, "Write");
            }
            LoggingUtil.getLogger(this).info("Write operation completed");

            try (GpuEvent readEvent = transferQueue.enqueueReadBuffer(buffer, 0, numFloats * Float.BYTES, dstData)) {
                LoggingUtil.getLogger(this).info("Enqueued read buffer operation");
                awaitTransfer(readEvent, "Read");
            }
            LoggingUtil.getLogger(this).info("Read operation completed");

            transferQueue.finish();
        }
        LoggingUtil.getLogger(this).info("Buffer released");

        boolean dataMatches = true;
        for (int i = 0; i < numFloats; i++) {
            if (Math.abs(srcData[i] - dstData[i]) > 0.0001f) {
                dataMatches = false;
                LoggingUtil.getLogger(this).error("Data mismatch at index {}: expected {}, got {}", i, srcData[i], dstData[i]);
            }
        }

        if (dataMatches) {
            LoggingUtil.getLogger(this).info("Buffer round-trip test PASSED - data verified correctly");
        } else {
            LoggingUtil.getLogger(this).error("Buffer round-trip test FAILED");
        }
    }

    private static void awaitTransfer(GpuEvent event, String what) {
        try {
            event.awaitCompletion();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(what + " operation interrupted", e);
        }
    }
}
