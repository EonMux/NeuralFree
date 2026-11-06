package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorFilters;
import com.eonmux.neuralfree.gguf.hf.config.QuantArgs;
import com.eonmux.neuralfree.gguf.hf.quant.Fp4Scheme;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlanBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * The package-visible parts of the {@link HfSourceAssembler} steps: the {@code --no-mtp} drop behind
 * {@link HfSourceAssembler#filter}, the recording sink behind {@link HfSourceAssembler#plan}, the {@code prec_a4}
 * flags, and the checkpoint wrapper that hides the sentence-transformers dense side files.
 */
final class HfAssemblerParts {

    private HfAssemblerParts() {
    }

    /**
     * The {@code --no-mtp} drop and the {@code --mtp} selection: layers past the base block count and the
     * {@code mtp}/{@code nextn} tensors.
     */
    static final class MtpFilter {
        private MtpFilter() {
        }

        static HfTensorIndex drop(HfTensorIndex index, int baseBlocks) {
            Set<String> remove = new LinkedHashSet<>();
            for (String name : index.names()) {
                if (isMtp(name, baseBlocks)) {
                    remove.add(name);
                }
            }
            return remove.isEmpty() ? index : index.withReplaced(Collections.emptyMap(), remove);
        }

        /**
         * The {@code --mtp} view of {@code index}: the MTP tensors and {@code rootKeep}, every other tensor dropped.
         */
        static HfTensorIndex only(HfTensorIndex index, Set<String> rootKeep, int baseBlocks) {
            Set<String> remove = new LinkedHashSet<>();
            for (String name : index.names()) {
                if (!keepForMtpOnly(name, rootKeep, baseBlocks)) {
                    remove.add(name);
                }
            }
            return remove.isEmpty() ? index : index.withReplaced(Collections.emptyMap(), remove);
        }

        /** Whether a {@code --mtp} export keeps {@code name}: a shared root tensor, or an MTP tensor. */
        static boolean keepForMtpOnly(String name, Set<String> rootKeep, int baseBlocks) {
            return rootKeep.contains(name) || isMtp(name, baseBlocks);
        }

        static boolean isMtp(String name, int baseBlocks) {
            if (name.startsWith("mtp.") || name.contains(".mtp.")) {
                return true;
            }
            for (String part : name.split("\\.")) {
                if (part.equals("mtp") || part.startsWith("mtp_") || part.contains("nextn")) {
                    return true;
                }
            }
            return HfLoadContext.bidOf(name) >= baseBlocks;
        }
    }

    /**
     * A sink that forwards to the plan builder, through the pass's online requantization when there is one, and
     * records the GGUF tensors each checkpoint tensor adds. {@link #flush} ends the pass.
     */
    static final class RecordingSink implements TensorSink {
        private final TensorPlanBuilder b;
        private final HfOnlineRequant requant;
        private final Map<String, List<String>> added = new LinkedHashMap<>();
        private String current;

        RecordingSink(TensorPlanBuilder b, HfOnlineRequant requant) {
            this.b = b;
            this.requant = requant;
        }

        void begin(String hfName) {
            current = hfName;
        }

        Map<String, List<String>> added() {
            return added;
        }

        @Override
        public void add(String ggufName, RowSource src) {
            if (requant == null) {
                b.add(ggufName, src);
            } else {
                requant.add(b, current, ggufName, src);
            }
            if (current != null) {
                added.computeIfAbsent(current, k -> new ArrayList<>()).add(ggufName);
            }
        }

        /** Adds the tensors the requantization still holds ({@link HfOnlineRequant#flush}). */
        void flush() {
            if (requant != null) {
                requant.flush(b);
            }
        }

        @Override
        public void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            b.addKv(key, type, elementType, value);
        }

        @Override
        public void drop(String hfName, String reason) {
            b.drop(hfName, reason);
        }

        @Override
        public boolean isConsumed(String hfName) {
            return b.isConsumed(hfName);
        }

        @Override
        public void consume(String hfName) {
            b.consume(hfName);
        }
    }

    /** The {@code prec_a4} flags of {@code _tag_prec_a4} (LC/conversion/base.py:665-675). */
    static final class PrecA4 {
        private PrecA4() {
        }

        /**
         * {@code false} for every NVFP4 {@code .weight} tensor a hook planned from a checkpoint tensor whose module is
         * W4A16_NVFP4.
         */
        static Map<String, Boolean> flags(HfLoadContext ctx, Map<String, QuantArgs> moduleArgs,
                                          Map<String, List<String>> added, TensorPlan plan) {
            Set<String> weightOnly = new HashSet<>();
            for (Map.Entry<String, QuantArgs> e : moduleArgs.entrySet()) {
                if (Fp4Scheme.weightOnlyNvfp4(e.getValue())) {
                    for (String hf : new String[]{e.getKey() + ".weight", e.getKey()}) {
                        String filtered = HfTensorFilters.filterName(ctx, hf);
                        if (filtered != null) {
                            weightOnly.add(filtered);
                        }
                    }
                }
            }
            if (weightOnly.isEmpty()) {
                return Collections.emptyMap();
            }
            Map<String, Boolean> out = new TreeMap<>();
            for (Map.Entry<String, List<String>> e : added.entrySet()) {
                if (!weightOnly.contains(e.getKey())) {
                    continue;
                }
                for (String gguf : e.getValue()) {
                    if (gguf.endsWith(".weight") && plan.has(gguf) && plan.type(gguf) == GGMLType.NVFP4) {
                        out.put(gguf, false);
                    }
                }
            }
            return out;
        }
    }

    /** A checkpoint whose sentence-transformers {@code N_Dense} side files read as absent. */
    static final class WithoutDenseModules implements CheckpointContainer {
        /** A path inside a {@code <n>_Dense} module directory, after any {@code ./} segments. */
        private static final Pattern DENSE_PATH = Pattern.compile("(?:\\.?/)*\\d+_Dense(?:/.*)?");
        private final CheckpointContainer in;

        WithoutDenseModules(CheckpointContainer in) {
            this.in = in;
        }

        @Override
        public String description() { return in.description(); }

        @Override
        public List<StTensor> tensors() { return in.tensors(); }

        @Override
        public StTensor tensor(String name) { return in.tensor(name); }

        @Override
        public RowSource source(String name) { return in.source(name); }

        @Override
        public List<DataSegment> segments() { return in.segments(); }

        @Override
        public long directVirtualOffset(String name) { return in.directVirtualOffset(name); }

        @Override
        public SafetensorsFile openSideFile(String relPath) throws IOException {
            if (DENSE_PATH.matcher(relPath.replace('\\', '/')).matches()) {
                throw new FileNotFoundException(relPath + " is not read: sentence-transformers dense modules are"
                        + " disabled (" + ModelSourceOptions.PROP_ST_DENSE + "=false)");
            }
            return in.openSideFile(relPath);
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

        @Override
        public String toString() {
            return in + " without dense modules";
        }
    }
}
