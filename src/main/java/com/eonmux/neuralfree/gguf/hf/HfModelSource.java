package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.store.HfTensorRegions;
import com.eonmux.neuralfree.gguf.hf.store.HfWeightStore;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicy;
import com.eonmux.neuralfree.gguf.hf.store.ProducedSegments;
import com.eonmux.neuralfree.gguf.hf.store.SidecarDirectory;
import com.eonmux.neuralfree.gguf.hf.store.VirtualLayout;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A Hugging Face safetensors checkpoint seen as a GGUF model: the metadata and tensor infos the converter would write
 * (built by {@link HfSourceAssembler}), with tensor data read from the checkpoint files at virtual offsets and derived
 * tensors produced on demand ({@link VirtualLayout}). Metadata and tensors are immutable after {@link #open}; the
 * getters follow {@link com.eonmux.neuralfree.gguf.GGUFFileParser}: an unmodifiable metadata map, {@code getString}
 * null unless the value is a string, and the GGUF value and array element types of every key. Each weight store and
 * region set gets the materialization policy of its reader ({@link MaterializePolicy}); derived tensors kept on disk
 * live in one private {@link SidecarDirectory} under {@link ModelSourceOptions#cacheDir()}. Thread-safe.
 */
public final class HfModelSource implements ModelSource {

    private static final Logger LOG = LogManager.getLogger(HfModelSource.class);

    private final String filePath;
    private final ModelSourceOptions options;
    private final HfSourceAssembler.Assembly assembly;
    private final Map<String, Object> metadata;
    private final Map<String, GGUFValueType> valueTypes;
    private final Map<String, GGUFValueType> elementTypes;
    private final List<GGUFTensorInfo> tensors;
    private final SidecarDirectory sidecars;
    private final List<AutoCloseable> readers = new ArrayList<>();
    private boolean closed;

    private HfModelSource(String filePath, ModelSourceOptions options, HfSourceAssembler.Assembly assembly) {
        this.filePath = filePath;
        this.options = options;
        this.assembly = assembly;
        HfMetadata md = assembly.metadata();
        this.metadata = Collections.unmodifiableMap(new LinkedHashMap<>(md.values()));
        this.valueTypes = Collections.unmodifiableMap(new LinkedHashMap<>(md.types()));
        this.elementTypes = Collections.unmodifiableMap(new LinkedHashMap<>(md.elementTypes()));
        this.tensors = assembly.layout().tensors();
        this.sidecars = new SidecarDirectory(options.cacheDir());
    }

    /** {@link #open(Path, ModelSourceOptions, Path, ActivationQuant)} for {@link ActivationQuant#WEIGHT_ONLY}. */
    public static HfModelSource open(Path path, ModelSourceOptions o, Path target) throws IOException {
        return open(path, o, target, ActivationQuant.WEIGHT_ONLY);
    }

    /**
     * Opens the checkpoint at {@code path} (a model directory or one {@code .safetensors} file) as a language model.
     *
     * @param target the model a draft checkpoint serves, or null
     * @param mode   how the engine treats the checkpoint's activation quantization; CHECKPOINT refuses activation
     *               schemes NeuralFree does not emulate
     * @throws IOException when the checkpoint cannot be read or is invalid
     */
    public static HfModelSource open(Path path, ModelSourceOptions o, Path target, ActivationQuant mode)
            throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(o, "o");
        Objects.requireNonNull(mode, "mode");
        HfSourceAssembler.Assembly a = HfSourceAssembler.assemble(path, o, target, mode, HfArchEntry.Kind.TEXT);
        return opened(path, o, a);
    }

    /**
     * Opens the vision / audio projector view of the checkpoint directory {@code dir}.
     *
     * @throws com.eonmux.neuralfree.gguf.hparams.GgufFormatException when no projector conversion is registered for
     *                                                                the checkpoint's architecture
     * @throws IOException                                            when the checkpoint cannot be read or is invalid
     */
    public static HfModelSource openProjector(Path dir, ModelSourceOptions o) throws IOException {
        Objects.requireNonNull(dir, "dir");
        Objects.requireNonNull(o, "o");
        HfSourceAssembler.Assembly a = HfSourceAssembler.assemble(dir, o, null, ActivationQuant.WEIGHT_ONLY,
                HfArchEntry.Kind.MMPROJ);
        return opened(dir, o, a);
    }

    private static HfModelSource opened(Path path, ModelSourceOptions o, HfSourceAssembler.Assembly a) {
        HfModelSource s = new HfModelSource(path.toString(), o, a);
        VirtualLayout l = a.layout();
        LOG.info("Opened Hugging Face checkpoint {} as '{}': {} tensors ({} derived), {} metadata keys",
                path, a.ctx().arch().ggufName(), l.tensors().size(), l.produced().size(), s.metadata.size());
        return s;
    }

    @Override
    public ModelFormat format() {
        return ModelFormat.SAFETENSORS;
    }

    @Override
    public String getFilePath() {
        return filePath;
    }

    @Override
    public Map<String, Object> getMetadata() {
        return metadata;
    }

    @Override
    public GGUFValueType getValueType(String key) {
        return valueTypes.get(key);
    }

    @Override
    public GGUFValueType getArrayElementType(String key) {
        return elementTypes.get(key);
    }

    @Override
    public String getString(String key) {
        Object v = metadata.get(key);
        return v instanceof String ? (String) v : null;
    }

    @Override
    public List<GGUFTensorInfo> getTensors() {
        return tensors;
    }

    @Override
    public long getTensorCount() {
        return tensors.size();
    }

    /** The load context the view was built with. */
    public HfLoadContext context() {
        return assembly.ctx();
    }

    /** The tensor plan behind {@link #getTensors()}. */
    public TensorPlan plan() {
        return assembly.plan();
    }

    /** The virtual tensor-data layout. */
    public VirtualLayout layout() {
        return assembly.layout();
    }

    /** The private directory of the derived tensors kept on disk. */
    public SidecarDirectory sidecars() {
        return sidecars;
    }

    /**
     * A new weight store over the checkpoint files and the derived tensors, which it materializes as the
     * {@link MaterializePolicy} of {@code use} decides. The source closes it when it closes.
     *
     * @throws IOException when the source is closed or a checkpoint file cannot be mapped
     */
    @Override
    public WeightStore createMappedStore(StoreUse use) throws IOException {
        Objects.requireNonNull(use, "use");
        synchronized (readers) {
            ensureOpen();
            HfWeightStore store = new HfWeightStore(layout(), view(MaterializePolicy.readerOf(use)));
            readers.add(store);
            return store;
        }
    }

    /**
     * New tensor regions for the CPU weights: stored tensors mapped or read as {@code mode} selects, derived tensors
     * as the CPU row of {@link MaterializePolicy} decides. The source closes them when it closes.
     *
     * @throws IllegalArgumentException when the materialization setting is NONE, which the CPU weights cannot use
     * @throws IOException              when the source is closed or a checkpoint file cannot be opened
     */
    @Override
    public TensorRegions openRegions(TensorReadMode mode) throws IOException {
        Objects.requireNonNull(mode, "mode");
        synchronized (readers) {
            ensureOpen();
            HfTensorRegions regions = new HfTensorRegions(layout(), view(MaterializePolicy.Reader.CPU_REGIONS), mode,
                    filePath);
            readers.add(regions);
            return regions;
        }
    }

    private ProducedSegments.View view(MaterializePolicy.Reader reader) {
        ProducedSegments produced = layout().produced();
        return produced.view(MaterializePolicy.of(reader, options, produced.tensors()), sidecars);
    }

    private void ensureOpen() throws IOException {
        if (closed) {
            throw new IOException("Hugging Face model source " + filePath + " is closed");
        }
    }

    /**
     * Closes, in reverse open order, every weight store and region set, the sidecar directory, the load's resources
     * and the checkpoint. The first failure is thrown with later ones suppressed; later calls do nothing.
     */
    @Override
    public void close() throws IOException {
        List<AutoCloseable> toClose;
        synchronized (readers) {
            if (closed) {
                return;
            }
            closed = true;
            toClose = new ArrayList<>(readers);
            readers.clear();
        }
        Collections.reverse(toClose);
        toClose.add(sidecars);
        toClose.add(assembly.resources());
        toClose.add(assembly.checkpoint());
        IOException first = null;
        for (AutoCloseable c : toClose) {
            try {
                c.close();
            } catch (Exception e) {
                IOException io = e instanceof IOException ? (IOException) e
                        : new IOException("closing " + c + " of " + filePath + " failed", e);
                if (first == null) {
                    first = io;
                } else {
                    first.addSuppressed(io);
                }
            }
        }
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "HfModelSource{" + filePath + ", " + assembly.ctx().arch().ggufName() + ", " + tensors.size()
                + " tensors}";
    }
}
