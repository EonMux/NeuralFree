package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * An opened mmproj file: the clip GGUF that carries a vision or audio encoder and its projector beside a text
 * model (the counterpart of llama.cpp {@code mtmd_init_from_file} over tools/mtmd/clip.cpp). Opening reads the
 * metadata and the tensor table only &mdash; the projector registry resolves the {@code clip.projector_type} of
 * every encoder the file declares, the {@code clip.*} hparams parse, and the family's tensor requirement table
 * validates &mdash; so a file that cannot run fails at open with the key or tensor named. A file that pairs a
 * runnable encoder with one whose family the registry rejects (the Gemma 3n file: the gemma3na audio tower beside
 * the gemma3nv MobileNetV5 vision tower) opens for the runnable one; the other is still reported by {@link #caps},
 * but {@link #supportsVision} / {@link #supportsAudio} leave it out and opening it refuses with the registry's
 * reason ({@link #unrunnableReason}). A file whose every encoder is rejected fails at open. The host weights load
 * once, when the first CPU encoder binds ({@link #openCpuVisionEncoder}, {@link #openAudioEncoder}, or a vision
 * encoder that stays on the CPU); a file with both a vision and an audio encoder serves both from the same weights.
 * {@link #openVisionEncoder} binds the CPU encoder over the host weights, or the GPU encoder over weights uploaded
 * to its device, per the device choice of {@link MmprojOptions} (llama.cpp {@code --mmproj-offload}, default on,
 * and {@code --mmproj-device}); a {@link MmEncoderTpMode#DATA} rank group binds one whole encoder per device and
 * spreads encode batches over them ({@link DataParallelVisionEncoder}). Both run every runnable family; a model the
 * chosen device cannot run ({@link GpuVisionEncoder#unsupportedReason}) runs on the CPU encoder, with the reason
 * logged.
 *
 * <p>{@link #caps} and {@link #supportsVision} / {@link #supportsAudio} / {@link #supportsGenAudio} are the
 * capability probe of {@code mtmd_get_cap_from_file}; {@link #memoryUsageBytes} reports the mmproj weight bytes
 * ({@code mtmd_get_memory_usage} without the context limits); {@link #caps} describes the file, the
 * {@code supports*} probes what NeuralFree runs of it. Immutable over the file; close the model and every
 * encoder it opened.
 */
public final class MmprojModel implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(MmprojModel.class);

    /** Capabilities of an mmproj file (llama.cpp {@code mtmd_caps}). */
    public static final class Caps {
        private final boolean inpVision;
        private final boolean inpAudio;
        private final boolean genAudio;

        private Caps(boolean inpVision, boolean inpAudio, boolean genAudio) {
            this.inpVision = inpVision;
            this.inpAudio = inpAudio;
            this.genAudio = genAudio;
        }

        /** The file feeds images ({@code clip.has_vision_encoder}). */
        public boolean inpVision() {
            return inpVision;
        }

        /** The file feeds audio ({@code clip.has_audio_encoder}). */
        public boolean inpAudio() {
            return inpAudio;
        }

        /** The file generates audio ({@code clip.has_gen_audio_encoder}). */
        public boolean genAudio() {
            return genAudio;
        }

        @Override
        public String toString() {
            return "Caps{vision=" + inpVision + ", audio=" + inpAudio + ", genAudio=" + genAudio + "}";
        }
    }

    private final ModelSource source;
    private final GgufKv kv;
    private final TensorCatalog tensors;
    private final Map<ProjectorModality, ClipHparams> hparams;
    private final Map<ProjectorModality, String> unrunnable;
    private CpuWeights weights;
    private InputTransformLoader.Result transforms;

    private MmprojModel(ModelSource source, GgufKv kv, TensorCatalog tensors,
                        Map<ProjectorModality, ClipHparams> hparams, Map<ProjectorModality, String> unrunnable) {
        this.source = source;
        this.kv = kv;
        this.tensors = tensors;
        this.hparams = hparams;
        this.unrunnable = unrunnable;
    }

    /** {@link #open(String, ModelSourceOptions)} with the process default options. */
    public static MmprojModel open(String path) throws IOException {
        return open(path, com.eonmux.neuralfree.gguf.ModelSourceOptions.fromSystemProperties());
    }

    /**
     * Opens the mmproj GGUF at {@code path}.
     *
     * @throws FileNotFoundException    when the path is not a readable GGUF file
     * @throws IOException              when the file cannot be parsed
     * @throws IllegalArgumentException when the file is no clip projector file
     *                                  ({@code general.architecture} is not {@code clip} or absent)
     * @throws UnsupportedProjectorException when no encoder of the file has a runnable registry row (the first
     *                                       refusal), or a runnable row is registered for another modality
     */
    public static MmprojModel open(String path, com.eonmux.neuralfree.gguf.ModelSourceOptions options)
            throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(options, "options");
        ModelSource source = ModelSources.parseGguf(path);
        try {
            GgufKv kv = GgufKv.of(source);
            String arch = kv.archName();
            if (!"clip".equals(arch)) {
                throw new IllegalArgumentException("'" + path + "' has general.architecture '" + arch
                        + "': an mmproj file is a clip projector GGUF; load text models through"
                        + " ModelSources.open");
            }
            if (ClipDualEncoder.isTwoTower(kv)) {
                throw new IllegalArgumentException("'" + path + "' is a CLIP two-tower embedding model without a"
                        + " projector (no clip.projector_type): serve it as the model, it opens through"
                        + " ClipDualEncoder.open, not as an mmproj beside a text model");
            }
            TensorCatalog tc = TensorCatalog.of(source);
            Map<ProjectorModality, ClipHparams> hparams = new EnumMap<>(ProjectorModality.class);
            for (ProjectorModality m : ProjectorModality.values()) {
                if (m.media() && hasEncoder(kv, m)) {
                    hparams.put(m, ClipHparams.from(kv, m));
                }
            }
            Map<ProjectorModality, String> unrunnable = validateEncoders(path, hparams, tc);
            return new MmprojModel(source, kv, tc, Collections.unmodifiableMap(hparams),
                    Collections.unmodifiableMap(unrunnable));
        } catch (RuntimeException | Error e) {
            try {
                source.close();
            } catch (IOException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /**
     * Resolves every encoder's registry row and validates the runnable ones' tensor tables; returns the refusal
     * of each encoder whose family the registry rejects.
     *
     * @throws UnsupportedProjectorException when every encoder is rejected, or a row's modality differs
     */
    private static Map<ProjectorModality, String> validateEncoders(String path,
                                                                   Map<ProjectorModality, ClipHparams> hparams,
                                                                   TensorCatalog tc) {
        Map<ProjectorModality, String> unrunnable = new EnumMap<>(ProjectorModality.class);
        RuntimeException first = null;
        for (ClipHparams hp : hparams.values()) {
            ProjectorEntry entry;
            try {
                entry = ProjectorRegistry.get(hp.projectorType().typeName());
            } catch (UnsupportedProjectorException e) {
                unrunnable.put(hp.modality(), e.getMessage());
                first = first == null ? e : first;
                continue;
            }
            if (entry.modality() != hp.modality()) {
                throw new UnsupportedProjectorException(hp.projectorType().typeName(), "projector type '"
                        + hp.projectorType() + "' is a " + entry.modality() + " projector; '" + path
                        + "' declares it for its " + hp.modality() + " encoder");
            }
            try {
                entry.tensors().validate(tc, hp.nLayer());
            } catch (MissingTensorException e) {
                // a hybrid file may declare an encoder whose tensors it does not carry (a Gemma 3n audio file
                // names its vision tower without vision tensors): that modality does not run and the file serves
                // its other encoders; a file whose every encoder is unrunnable still fails at open below
                unrunnable.put(hp.modality(),
                        "projector type '" + hp.projectorType().typeName() + "' " + e.getMessage());
                first = first == null ? e : first;
            }
        }
        if (first != null && unrunnable.size() == hparams.size()) {
            throw first;
        }
        for (Map.Entry<ProjectorModality, String> e : unrunnable.entrySet()) {
            LOGGER.warn("mmproj '{}': the {} encoder does not run ({}); the file serves its other encoders", path,
                    e.getKey().name().toLowerCase(Locale.ROOT), e.getValue());
        }
        return unrunnable;
    }

    private static boolean hasEncoder(GgufKv kv, ProjectorModality modality) {
        String key;
        switch (modality) {
            case VISION:
                key = "clip.has_vision_encoder";
                break;
            case AUDIO:
                key = "clip.has_audio_encoder";
                break;
            case TEXT:
                key = "clip.has_text_encoder";
                break;
            default:
                key = "clip.has_gen_audio_encoder";
        }
        return kv.bool(key, false);
    }

    /** The capabilities of the file: every encoder it declares, whether or not NeuralFree runs it. */
    public Caps caps() {
        return new Caps(hparams.containsKey(ProjectorModality.VISION),
                hparams.containsKey(ProjectorModality.AUDIO),
                hparams.containsKey(ProjectorModality.GEN_AUDIO));
    }

    /** True when the file feeds images through an encoder NeuralFree runs. */
    public boolean supportsVision() {
        return runs(ProjectorModality.VISION);
    }

    /** True when the file feeds audio through an encoder NeuralFree runs. */
    public boolean supportsAudio() {
        return runs(ProjectorModality.AUDIO);
    }

    /** True when the file generates audio through an encoder NeuralFree runs. */
    public boolean supportsGenAudio() {
        return runs(ProjectorModality.GEN_AUDIO);
    }

    private boolean runs(ProjectorModality modality) {
        return hparams.containsKey(modality) && !unrunnable.containsKey(modality);
    }

    /**
     * Why the file's encoder of {@code modality} does not run (the projector registry's refusal of its family), or
     * null when the encoder runs or the file has none.
     */
    public String unrunnableReason(ProjectorModality modality) {
        return unrunnable.get(Objects.requireNonNull(modality, "modality"));
    }

    /** The hparams of the encoder of {@code modality}, refusing an encoder whose family does not run. */
    private ClipHparams runnableHparams(ProjectorModality modality) {
        ClipHparams hp = hparams(modality);
        String reason = unrunnable.get(modality);
        if (reason != null) {
            throw new UnsupportedProjectorException(hp.projectorType().typeName(), reason);
        }
        return hp;
    }

    /** The hparams of the vision encoder.
     *
     * @throws IllegalArgumentException when the file has no vision encoder
     */
    public ClipHparams visionHparams() {
        return hparams(ProjectorModality.VISION);
    }

    /** The hparams of the audio encoder.
     *
     * @throws IllegalArgumentException when the file has no audio encoder
     */
    public ClipHparams audioHparams() {
        return hparams(ProjectorModality.AUDIO);
    }

    private ClipHparams hparams(ProjectorModality modality) {
        ClipHparams hp = hparams.get(modality);
        if (hp == null) {
            throw new IllegalArgumentException("the file has no " + modality.name().toLowerCase(java.util.Locale.ROOT)
                    + " encoder");
        }
        return hp;
    }

    /** Sum of the stored sizes of the file's tensors, the mmproj weight bytes. */
    public long memoryUsageBytes() {
        long total = 0;
        for (com.eonmux.neuralfree.gguf.GGUFTensorInfo t : source.getTensors()) {
            total += t.sizeInBytes();
        }
        return total;
    }

    /**
     * {@link #openVisionEncoder(MmprojOptions)} with {@link MmprojOptions#fromSystemProperties()}: the GPU encoder
     * unless {@value MmprojOptions#PROP_OFFLOAD}{@code =false}, {@value MmprojOptions#PROP_DEVICE}{@code =none} or
     * a CPU backend mode keeps it on the host.
     *
     * @throws IllegalArgumentException when the file has no vision encoder or a property is malformed
     * @throws UnsupportedProjectorException when the vision encoder's family does not run
     * @throws IOException              when a tensor cannot be read
     */
    public VisionEncoder openVisionEncoder() throws IOException {
        return openVisionEncoder(MmprojOptions.fromSystemProperties());
    }

    /**
     * Binds a vision encoder on the device {@code options} choose: the CPU encoder when offload is off; the GPU
     * encoder on the selected device, or on the backend mode's best device when no selector is given. Without a
     * selector, a host with no usable GPU under the AUTO mode, and on any device a model the device cannot run
     * ({@link GpuVisionEncoder#unsupportedReason}), give the CPU encoder. A GPU encoder owns its backend and device
     * weights; every call binds a new executor, close it after use.
     *
     * @throws IllegalArgumentException      when the file has no vision encoder, or the selector names no device or
     *                                       a device that cannot run the LLM kernels
     * @throws NoBackendAvailableException   when an explicit backend mode (not AUTO) has no usable device
     * @throws UnsupportedProjectorException when the vision encoder's family does not run
     * @throws IOException                   when a tensor cannot be read
     */
    public VisionEncoder openVisionEncoder(MmprojOptions options) throws IOException {
        Objects.requireNonNull(options, "options");
        ClipVisionModel model = visionModel();
        if (!options.offload()) {
            return openCpu(model);
        }
        HWAcceleration acceleration = HWAcceleration.forMode(options.backend().toAccelerationMode());
        List<GpuDevices.Selected> targets;
        try {
            targets = gpuTargets(acceleration, options);
        } catch (RuntimeException | Error e) {
            acceleration.shutdown();
            throw e;
        }
        if (targets.isEmpty()) {
            acceleration.shutdown();
            LOGGER.info("mmproj {}: no GPU backend can run the LLM kernels; the vision encoder runs on the CPU",
                    model.hparams().projectorType().typeName());
            return openCpu(model);
        }
        if (options.tpMode() == MmEncoderTpMode.DATA && targets.size() > 1) {
            return openDataParallel(model, targets, acceleration);
        }
        GpuDevices.Selected target = targets.get(0);
        return openOnBackend(model, target.backend(), target.name(), acceleration::shutdown);
    }

    /**
     * The GPU encoder on every device of {@code options}' rank group as one {@link DataParallelVisionEncoder}
     * ({@link MmEncoderTpMode#DATA}); a model any rank's device cannot run falls back to the one whole encoder on
     * the first rank's device, with the reason logged, as vLLM falls back from data to weights parallelism when the
     * encoder does not support it. The data-parallel encoder owns {@code acceleration}, whose backends it releases
     * after every rank.
     */
    private VisionEncoder openDataParallel(ClipVisionModel model, List<GpuDevices.Selected> targets,
                                           HWAcceleration acceleration) throws IOException {
        InputTransformLoader.Result t;
        WeightStore store;
        try {
            t = transforms();
            store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        } catch (IOException | RuntimeException | Error e) {
            acceleration.shutdown();
            throw e;
        }
        List<VisionEncoder> ranks = new ArrayList<>(targets.size());
        boolean handedOver = false;
        try {
            for (GpuDevices.Selected target : targets) {
                LlmKernels kernels = LlmKernels.open(target.backend());
                String reason = GpuVisionEncoder.unsupportedReason(model, kernels);
                if (reason != null) {
                    for (VisionEncoder rank : ranks) {
                        closeOwner(rank);
                    }
                    handedOver = true;
                    LOGGER.warn("mmproj on {}: {}; the vision encoder keeps one whole encoder on {}",
                            target.name(), reason, targets.get(0).name());
                    return openOnBackend(model, targets.get(0).backend(), targets.get(0).name(),
                            acceleration::shutdown);
                }
                ranks.add(GpuVisionEncoder.open(model, kernels, target.name(), source.getTensors(), store, t, null));
            }
            DataParallelVisionEncoder encoder = DataParallelVisionEncoder.of(ranks, acceleration::shutdown);
            handedOver = true;
            LOGGER.info("mmproj {}: data-parallel vision encoder over {} ranks ({})",
                    model.hparams().projectorType().typeName(), ranks.size(), encoder.device());
            return encoder;
        } finally {
            closeStore(store);
            if (!handedOver) {
                for (VisionEncoder rank : ranks) {
                    closeOwner(rank);
                }
                acceleration.shutdown();
            }
        }
    }

    /**
     * Binds the GPU vision encoder on {@code backend}, which stays owned by the caller (the text model's backend,
     * for example); a model the device cannot run ({@link GpuVisionEncoder#unsupportedReason}) gets the CPU
     * encoder.
     *
     * @throws IllegalArgumentException when the file has no vision encoder
     * @throws UnsupportedProjectorException when the vision encoder's family does not run
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when the backend cannot run the LLM
     *                                                                        kernels
     * @throws IOException              when a tensor cannot be read
     */
    public VisionEncoder openVisionEncoder(ComputeBackend backend) throws IOException {
        Objects.requireNonNull(backend, "backend");
        return openOnBackend(visionModel(), backend, deviceName(backend), null);
    }

    /**
     * Binds the CPU vision encoder over the file's weights: the geometry resolves per tensor and shape, then the
     * weights load. Every call binds a new executor; close it (or this model) after use.
     *
     * @throws IllegalArgumentException when the file has no vision encoder
     * @throws UnsupportedProjectorException when the vision encoder's family does not run
     * @throws IOException              when a tensor cannot be read
     */
    public CpuVisionEncoder openCpuVisionEncoder() throws IOException {
        return openCpu(visionModel());
    }

    /**
     * Binds the CPU audio encoder over the file's weights: the geometry resolves per tensor and shape, then the
     * weights load (once per file; a vision encoder of the same file shares them). Every call binds a new
     * executor; close it (or this model) after use.
     *
     * @throws IllegalArgumentException      when the file has no audio encoder, or a tensor shape differs from the
     *                                       graph's
     * @throws UnsupportedProjectorException when the audio encoder's family does not run
     * @throws IOException                   when a tensor cannot be read
     */
    public CpuAudioEncoder openAudioEncoder() throws IOException {
        ClipHparams hp = runnableHparams(ProjectorModality.AUDIO);
        ClipAudioModel model = ClipAudioModel.from(hp, tensors);
        CpuWeights w = weights();
        return CpuAudioEncoder.bind(model, w.matVec(), w::floats, w::get);
    }

    private ClipVisionModel visionModel() {
        return ClipVisionModel.from(runnableHparams(ProjectorModality.VISION), tensors);
    }

    private CpuVisionEncoder openCpu(ClipVisionModel model) throws IOException {
        CpuWeights w = weights();
        return CpuVisionEncoder.bind(model, w.matVec(), w::floats, w::get);
    }

    /** The host weights of the file, opened on first use and shared by every CPU encoder bound afterwards. */
    private CpuWeights weights() throws IOException {
        if (weights == null) {
            weights = CpuWeights.open(source, tensors, transforms());
        }
        return weights;
    }

    private InputTransformLoader.Result transforms() throws IOException {
        if (transforms == null) {
            transforms = InputTransformLoader.load(source, kv, tensors, ActivationQuant.WEIGHT_ONLY);
        }
        return transforms;
    }

    /**
     * The devices the encoder runs on: the rank group {@link MmprojOptions#devices()} names, in rank order, or the
     * one {@link #gpuTarget} device ({@code options.device()} or the backend mode's choice); empty when AUTO finds
     * no GPU that runs the LLM kernels.
     */
    static List<GpuDevices.Selected> gpuTargets(HWAcceleration acceleration, MmprojOptions options) {
        if (options.devices().isEmpty()) {
            GpuDevices.Selected one = gpuTarget(acceleration, options);
            return one == null ? Collections.<GpuDevices.Selected>emptyList() : Collections.singletonList(one);
        }
        List<GpuDevices.Selected> out = new ArrayList<>(options.devices().size());
        for (String selector : options.devices()) {
            GpuDevices.Selected s = GpuDevices.selectOne(acceleration, selector, MmprojOptions.PROP_DEVICES);
            Availability a = LlmKernels.availability(s.backend());
            if (!a.available()) {
                throw new IllegalArgumentException(MmprojOptions.PROP_DEVICES + " '" + selector + "' selects "
                        + s.name() + ", which cannot run the LLM kernels: " + a.reason());
            }
            out.add(s);
        }
        return out;
    }

    /**
     * The device of {@code options}: the selected one, or the backend mode's choice; null when AUTO finds no GPU
     * that runs the LLM kernels.
     */
    static GpuDevices.Selected gpuTarget(HWAcceleration acceleration, MmprojOptions options) {
        if (options.device() != null) {
            GpuDevices.Selected s = GpuDevices.selectOne(acceleration, options.device(), MmprojOptions.PROP_DEVICE);
            Availability a = LlmKernels.availability(s.backend());
            if (!a.available()) {
                throw new IllegalArgumentException(MmprojOptions.PROP_DEVICE + " '" + options.device() + "' selects "
                        + s.name() + ", which cannot run the LLM kernels: " + a.reason());
            }
            return s;
        }
        try {
            ComputeBackend b = acceleration.createBackend(options.backend().toAccelerationMode(),
                    LlmKernels.requirement());
            return new GpuDevices.Selected(b, deviceName(b));
        } catch (NoBackendAvailableException e) {
            if (options.backend() != EngineConfig.BackendMode.AUTO) {
                throw e;
            }
            LOGGER.debug("mmproj offload: {}", e.getMessage());
            return null;
        }
    }

    /**
     * The GPU encoder on {@code backend}, or the CPU encoder when the device cannot run the model; {@code owner}
     * releases the backend (null when the caller keeps it) and passes to the GPU encoder.
     */
    private VisionEncoder openOnBackend(ClipVisionModel model, ComputeBackend backend, String device,
                                        AutoCloseable owner) throws IOException {
        LlmKernels kernels;
        InputTransformLoader.Result t;
        WeightStore store;
        try {
            kernels = LlmKernels.open(backend);
            String reason = GpuVisionEncoder.unsupportedReason(model, kernels);
            if (reason != null) {
                LOGGER.warn("mmproj on {}: {}; the vision encoder runs on the CPU", device, reason);
                closeOwner(owner);
                return openCpu(model);
            }
            t = transforms();
            store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        } catch (IOException | RuntimeException | Error e) {
            closeOwner(owner);
            throw e;
        }
        try {
            GpuVisionEncoder enc = GpuVisionEncoder.open(model, kernels, device, source.getTensors(), store, t,
                    owner);
            LOGGER.info("mmproj {}: vision encoder on {}", model.hparams().projectorType().typeName(), device);
            return enc;
        } finally {
            closeStore(store);
        }
    }

    /** Releases the upload mapping; the uploaded weights no longer read it. */
    private static void closeStore(WeightStore store) {
        try {
            store.close();
        } catch (Exception e) {
            LOGGER.warn("releasing the mmproj upload mapping failed: {}", e.getMessage());
        }
    }

    private static void closeOwner(AutoCloseable owner) {
        if (owner == null) {
            return;
        }
        try {
            owner.close();
        } catch (Exception e) {
            LOGGER.warn("releasing the mmproj GPU backend failed: {}", e.getMessage());
        }
    }

    static String deviceName(ComputeBackend backend) {
        return backend.backendType().toString().toLowerCase(Locale.ROOT) + " '"
                + backend.capabilities().getDeviceName() + "'";
    }

    /** The names of the file's tensors, in file order. */
    public List<String> tensorNames() {
        List<String> out = new ArrayList<>();
        for (com.eonmux.neuralfree.gguf.GGUFTensorInfo t : source.getTensors()) {
            out.add(t.name());
        }
        return out;
    }

    /** The metadata view of the file. */
    public GgufKv metadata() {
        return kv;
    }

    @Override
    public void close() throws IOException {
        if (weights != null) {
            weights.close();
            weights = null;
        }
        source.close();
    }
}
