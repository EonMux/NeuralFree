package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * Where {@link MmprojModel#openVisionEncoder(MmprojOptions)} runs the vision encoder, the counterpart of llama.cpp
 * {@code --mmproj-offload} / {@code --no-mmproj-offload} (offload on by default) and {@code --mmproj-device}
 * ({@code -mmdev}; {@code none} keeps the encoder on the CPU), and of vLLM's {@code --mm-encoder-tp-mode}:
 * <ul>
 *   <li>offload off: the CPU encoder;</li>
 *   <li>offload on with a device selector (an index into the enumeration or a device-name substring, as
 *       {@code neuralfree.gpu.devices} takes them): the GPU encoder on that device, which must exist and run the
 *       LLM kernels;</li>
 *   <li>offload on without a selector: the GPU encoder on the device the backend mode selects (AUTO follows
 *       {@code neuralfree.device}, as llama.cpp's mmproj device follows {@code --device}), or the CPU encoder when
 *       the host has no usable GPU.</li>
 * </ul>
 * A model the device cannot run ({@link GpuVisionEncoder#unsupportedReason}) runs on the CPU encoder in every
 * case, with the reason logged.
 *
 * <p>{@link #tpMode()} spreads the encoder over a rank group ({@link #devices()}, vLLM's tensor-parallel ranks): a
 * {@link MmEncoderTpMode#DATA} group of two or more devices binds one whole-weight encoder per device and encodes a
 * batch of images across them ({@link DataParallelVisionEncoder}); the default {@link MmEncoderTpMode#WEIGHTS} runs
 * the encoder whole on one device, so a group of more than one device is refused by name rather than half used.
 * Without a device group the encoder is the ordinary single-device one in either mode, as vLLM with
 * {@code tensor_parallel_size=1} is. Immutable.
 */
public final class MmprojOptions {

    /** {@code true} / {@code false}: offload the encoder to a GPU (default {@code true}). */
    public static final String PROP_OFFLOAD = "neuralfree.mmproj.offload";
    /** Device selector of the encoder, or {@value #NO_DEVICE} for the CPU. */
    public static final String PROP_DEVICE = "neuralfree.mmproj.device";
    /** The {@link #PROP_DEVICE} value that keeps the encoder on the CPU. */
    public static final String NO_DEVICE = "none";
    /** The encoder's tensor/data-parallel mode (vLLM's {@code --mm-encoder-tp-mode}): {@code weights} or {@code data}. */
    public static final String PROP_TP_MODE = "neuralfree.mmproj.tp.mode";
    /** The comma-separated device selectors of the encoder's rank group, one rank per entry. */
    public static final String PROP_DEVICES = "neuralfree.mmproj.devices";

    private static final MmprojOptions DEFAULTS = new MmprojOptions(true, null, MmEncoderTpMode.WEIGHTS,
            Collections.<String>emptyList(), EngineConfig.BackendMode.AUTO);

    private final boolean offload;
    private final String device;
    private final MmEncoderTpMode tpMode;
    private final List<String> devices;
    private final EngineConfig.BackendMode backend;

    private MmprojOptions(boolean offload, String device, MmEncoderTpMode tpMode, List<String> devices,
                          EngineConfig.BackendMode backend) {
        List<String> ranks = List.copyOf(devices);
        if (device != null && !ranks.isEmpty()) {
            throw new IllegalArgumentException(PROP_DEVICE + " names the encoder's one device and " + PROP_DEVICES
                    + " its rank group; give one or the other");
        }
        if (tpMode == MmEncoderTpMode.WEIGHTS && ranks.size() > 1) {
            throw new IllegalArgumentException(PROP_DEVICES + " names " + ranks.size() + " devices but "
                    + PROP_TP_MODE + "=" + tpMode + " splits no encoder weights across ranks; the encoder runs"
                    + " whole on one device (give one device, or set " + PROP_TP_MODE + "="
                    + MmEncoderTpMode.DATA + " to spread the images over the ranks)");
        }
        if (!offload && ranks.size() > 1) {
            throw new IllegalArgumentException(PROP_DEVICES + " names " + ranks.size() + " devices but the encoder"
                    + " runs on the CPU (" + PROP_OFFLOAD + "=false, " + PROP_DEVICE + "=" + NO_DEVICE
                    + ", or a CPU backend), which is one rank");
        }
        this.offload = offload;
        this.device = device;
        this.tpMode = Objects.requireNonNull(tpMode, "tpMode");
        this.devices = ranks;
        this.backend = Objects.requireNonNull(backend, "backend");
    }

    /** Offload on, no device selector, weights mode, AUTO backend mode. */
    public static MmprojOptions defaults() {
        return DEFAULTS;
    }

    /** The CPU encoder ({@code --no-mmproj-offload}). */
    public static MmprojOptions cpu() {
        return new MmprojOptions(false, null, MmEncoderTpMode.WEIGHTS, Collections.<String>emptyList(),
                EngineConfig.BackendMode.AUTO);
    }

    /**
     * The GPU encoder on the device {@code selector} names ({@code --mmproj-device}); {@value #NO_DEVICE} gives
     * {@link #cpu()}.
     *
     * @throws IllegalArgumentException for a blank selector or one naming several devices
     */
    public static MmprojOptions onDevice(String selector) {
        String s = checkSelector(selector);
        return NO_DEVICE.equals(s.toLowerCase(Locale.ROOT)) ? cpu()
                : new MmprojOptions(true, s, MmEncoderTpMode.WEIGHTS, Collections.<String>emptyList(),
                        EngineConfig.BackendMode.AUTO);
    }

    /** These options with offload on or off; turning offload off drops the device selector and the rank group. */
    public MmprojOptions withOffload(boolean on) {
        return new MmprojOptions(on, on ? device : null, tpMode,
                on ? devices : Collections.<String>emptyList(), backend);
    }

    /**
     * These options with the API the GPU encoder may use; {@link EngineConfig.BackendMode#CPU} turns offload off
     * and drops the device selector and the rank group.
     */
    public MmprojOptions withBackend(EngineConfig.BackendMode mode) {
        Objects.requireNonNull(mode, "mode");
        return mode.usesGpu() ? new MmprojOptions(offload, device, tpMode, devices, mode)
                : new MmprojOptions(false, null, tpMode, Collections.<String>emptyList(), mode);
    }

    /**
     * These options with the encoder parallelism {@code mode}; turning weights mode back on with a rank group of
     * two or more devices is refused, as {@link #PROP_TP_MODE} is.
     *
     * @throws IllegalArgumentException when the options already name a rank group and {@code mode} is weights
     */
    public MmprojOptions withTpMode(MmEncoderTpMode mode) {
        return new MmprojOptions(offload, device, mode, devices, backend);
    }

    /**
     * These options with the rank group {@code selectors}; empty gives the single selected device again.
     *
     * @throws IllegalArgumentException when an entry is blank, the group repeats a selector, or the group holds more
     *                                  than one device under weights mode
     */
    public MmprojOptions withDevices(List<String> selectors) {
        Objects.requireNonNull(selectors, "selectors");
        return new MmprojOptions(offload, device, tpMode, checkSelectors(selectors), backend);
    }

    /**
     * The options of the system properties: {@value #PROP_OFFLOAD}, {@value #PROP_DEVICE}, {@value #PROP_TP_MODE},
     * {@value #PROP_DEVICES} and the backend mode of {@code neuralfree.backend}. A CPU backend mode keeps the
     * encoder on the CPU unless {@value #PROP_DEVICE} names a device (llama.cpp: {@code --device none} also turns
     * the mmproj offload off).
     *
     * @throws IllegalArgumentException naming the property when a value is malformed
     */
    public static MmprojOptions fromSystemProperties() {
        return fromProperties(System.getProperties());
    }

    /**
     * {@link #fromSystemProperties()} over {@code p}.
     *
     * @throws IllegalArgumentException naming the property when a value is malformed
     */
    public static MmprojOptions fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        boolean offload = booleanProperty(p, PROP_OFFLOAD, true);
        String backendValue = trimmed(p.getProperty(EngineConfig.PROP_BACKEND));
        EngineConfig.BackendMode mode = backendValue == null ? EngineConfig.BackendMode.AUTO
                : EngineConfig.BackendMode.parse(backendValue);
        MmEncoderTpMode tpMode = tpMode(p.getProperty(PROP_TP_MODE));
        List<String> devices = devices(p.getProperty(PROP_DEVICES));
        String deviceValue = p.getProperty(PROP_DEVICE);
        if (deviceValue != null) {
            String s;
            try {
                s = checkSelector(deviceValue);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(PROP_DEVICE + "=" + deviceValue + ": " + e.getMessage(), e);
            }
            if (!devices.isEmpty()) {
                throw new IllegalArgumentException(PROP_DEVICE + "=" + deviceValue + " and " + PROP_DEVICES + "="
                        + p.getProperty(PROP_DEVICES) + " both name the encoder's device(s); give one or the other");
            }
            if (NO_DEVICE.equals(s.toLowerCase(Locale.ROOT)) || !offload) {
                return new MmprojOptions(false, null, tpMode, Collections.<String>emptyList(),
                        mode.usesGpu() ? mode : EngineConfig.BackendMode.AUTO);
            }
            return new MmprojOptions(true, s, tpMode, Collections.<String>emptyList(),
                    mode.usesGpu() ? mode : EngineConfig.BackendMode.AUTO);
        }
        if (!mode.usesGpu()) {
            refuseDevicesOnCpu(devices);
            return new MmprojOptions(false, null, tpMode, Collections.<String>emptyList(), mode);
        }
        if (!offload) {
            refuseDevicesOnCpu(devices);
            return new MmprojOptions(false, null, tpMode, Collections.<String>emptyList(), mode);
        }
        return new MmprojOptions(true, null, tpMode, devices, mode);
    }

    /** Refuses a rank group the CPU encoder cannot spread over, naming the property. */
    private static void refuseDevicesOnCpu(List<String> devices) {
        if (!devices.isEmpty()) {
            throw new IllegalArgumentException(PROP_DEVICES + " names " + devices.size() + " device(s) but the"
                    + " encoder runs on the CPU (" + PROP_OFFLOAD + "=false or a CPU backend), which is one rank;"
                    + " unset " + PROP_DEVICES + " or enable the GPU offload");
        }
    }

    /** The mode of {@value #PROP_TP_MODE}; absent or blank gives {@link MmEncoderTpMode#WEIGHTS}. */
    private static MmEncoderTpMode tpMode(String value) {
        String v = trimmed(value);
        if (v == null) {
            return MmEncoderTpMode.WEIGHTS;
        }
        try {
            return MmEncoderTpMode.parse(v);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_TP_MODE + "=" + v + ": " + e.getMessage(), e);
        }
    }

    /** The rank group of {@value #PROP_DEVICES}; absent or blank gives no group. */
    private static List<String> devices(String value) {
        String v = trimmed(value);
        if (v == null) {
            return Collections.emptyList();
        }
        List<String> entries = new ArrayList<>();
        for (String entry : v.split(",", -1)) {
            entries.add(entry);
        }
        try {
            return checkSelectors(entries);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_DEVICES + "=" + v + ": " + e.getMessage(), e);
        }
    }

    /** The selectors of one rank group, each trimmed and non-blank and none repeated. */
    private static List<String> checkSelectors(List<String> selectors) {
        List<String> out = new ArrayList<>(selectors.size());
        for (int i = 0; i < selectors.size(); i++) {
            String s = trimmed(selectors.get(i));
            if (s == null) {
                throw new IllegalArgumentException("entry " + i + " of the encoder's device group is blank");
            }
            if (s.indexOf(',') >= 0) {
                throw new IllegalArgumentException("entry " + i + " of the encoder's device group '" + s
                        + "' names several devices; give one selector per rank");
            }
            if (out.contains(s)) {
                throw new IllegalArgumentException("the encoder's device group repeats '" + s + "'");
            }
            out.add(s);
        }
        return out;
    }

    private static String checkSelector(String selector) {
        String s = trimmed(selector);
        if (s == null) {
            throw new IllegalArgumentException("the mmproj device selector is blank");
        }
        if (s.indexOf(',') >= 0) {
            throw new IllegalArgumentException("only one device may be specified for the mmproj encoder, got '"
                    + s + "'");
        }
        return s;
    }

    private static boolean booleanProperty(Properties p, String key, boolean def) {
        String v = trimmed(p.getProperty(key));
        if (v == null) {
            return def;
        }
        if ("true".equalsIgnoreCase(v)) {
            return true;
        }
        if ("false".equalsIgnoreCase(v)) {
            return false;
        }
        throw new IllegalArgumentException(key + "=" + v + " is not a boolean; use true or false");
    }

    private static String trimmed(String v) {
        if (v == null) {
            return null;
        }
        String t = v.trim();
        return t.isEmpty() ? null : t;
    }

    /** True when the encoder may run on a GPU. */
    public boolean offload() {
        return offload;
    }

    /** The device selector, or null to let the backend mode choose. */
    public String device() {
        return device;
    }

    /** How the encoder spreads over its rank group. */
    public MmEncoderTpMode tpMode() {
        return tpMode;
    }

    /**
     * The device selectors of the encoder's rank group, in rank order; empty when the encoder runs on the one
     * selected device (or the backend's choice).
     */
    public List<String> devices() {
        return devices;
    }

    /** The API the GPU encoder may use. */
    public EngineConfig.BackendMode backend() {
        return backend;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof MmprojOptions)) {
            return false;
        }
        MmprojOptions other = (MmprojOptions) o;
        return offload == other.offload && Objects.equals(device, other.device) && tpMode == other.tpMode
                && devices.equals(other.devices) && backend == other.backend;
    }

    @Override
    public int hashCode() {
        return Objects.hash(offload, device, tpMode, devices, backend);
    }

    @Override
    public String toString() {
        return "MmprojOptions{offload=" + offload + (device != null ? ", device='" + device + "'" : "")
                + ", tp=" + tpMode + (devices.isEmpty() ? "" : ", devices=" + devices) + ", backend=" + backend
                + "}";
    }
}
