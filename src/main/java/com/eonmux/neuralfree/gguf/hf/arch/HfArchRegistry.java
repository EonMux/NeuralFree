package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.names.HfArchNames;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Registry of the Hugging Face architectures NeuralFree converts at load time: the merge of the twelve
 * {@code HfArchTable*} tables and the rejected entries this class owns (the names llama.cpp's converter lists as text
 * models without registering a class for them). Built once on first use; a Hugging Face name registered twice for
 * one {@link HfArchEntry.Kind} fails the build with {@link IllegalStateException}.
 *
 * <p>The name of a checkpoint follows the converter's {@code get_model_architecture}
 * (LC/conversion/base.py:3130-3159): for {@link HfArchEntry.Kind#TEXT} it is {@link HfConfig#architecture()}; for
 * {@link HfArchEntry.Kind#MMPROJ} it is the root {@code architectures[0]} (or the {@code ssm_cfg} layer), replaced by
 * {@code vision_config.architectures[0]} when that is present.</p>
 */
public final class HfArchRegistry {

    private static final ThreadLocal<Map<HfArchEntry.Kind, Map<String, HfArchEntry>>> OVERLAY = new ThreadLocal<>();

    private HfArchRegistry() {
    }

    private static final class Holder {
        static final Map<HfArchEntry.Kind, Map<String, HfArchEntry>> ENTRIES = merge(Arrays.asList(
                rejectedEntries(),
                HfArchTableLlama.entries(),
                HfArchTableDenseClassic.entries(),
                HfArchTableDense.entries(),
                HfArchTableMoeClassic.entries(),
                HfArchTableMoeSwa.entries(),
                HfArchTableMoeMla.entries(),
                HfArchTableHybridSsm.entries(),
                HfArchTableHybridLinear.entries(),
                HfArchTableSpecial.entries(),
                HfArchTableSpecialGen.entries(),
                HfArchTableMmproj.entries(),
                HfArchTableDrafts.entries()));
    }

    /**
     * The entry of the checkpoint configured by {@code cfg}, opened as {@code kind}.
     *
     * @throws GgufFormatException with the entry's reason for a rejected entry; for a text name llama.cpp converts but
     *                             no table registers; for a projector-only name opened as a text model; for a name
     *                             llama.cpp's converter does not know; for a projector name without an entry
     */
    public static HfArchEntry resolve(HfConfig cfg, HfArchEntry.Kind kind) {
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(kind, "kind");
        String name = kind == HfArchEntry.Kind.TEXT ? cfg.architecture() : projectorName(cfg);
        HfArchEntry e = lookup(kind, name);
        if (e != null) {
            if (e.isRejected()) {
                throw new GgufFormatException(e.rejection());
            }
            return e;
        }
        if (kind == HfArchEntry.Kind.MMPROJ) {
            Optional<HfArchNames.Row> projector = HfArchNames.row(name, HfArchEntry.Kind.MMPROJ);
            if (projector.isPresent()) {
                throw new GgufFormatException("Hugging Face projector architecture '" + name
                        + "' (llama.cpp converter class " + projector.get().converterClass() + ", "
                        + projector.get().source() + ") has no projector conversion in NeuralFree");
            }
            throw new GgufFormatException("Hugging Face architecture '" + name + "' has no projector conversion:"
                    + " llama.cpp's converter does not know it as a vision/audio projector");
        }
        Optional<HfArchNames.Row> row = HfArchNames.row(name, HfArchEntry.Kind.TEXT);
        if (row.isPresent()) {
            throw new GgufFormatException("Hugging Face architecture '" + name + "' (llama.cpp converter class "
                    + row.get().converterClass() + ", " + row.get().source() + ") has no conversion in NeuralFree");
        }
        if (lookup(HfArchEntry.Kind.MMPROJ, name) != null
                || HfArchNames.row(name, HfArchEntry.Kind.MMPROJ).isPresent()) {
            throw new GgufFormatException("'" + name + "' is a vision/audio projector checkpoint; open it with"
                    + " ModelSources.openProjector");
        }
        throw new GgufFormatException("Hugging Face architecture '" + name + "' is not supported: llama.cpp's"
                + " converter does not know it");
    }

    /** True when a table registers {@code hfName} for {@code kind}. */
    public static boolean has(String hfName, HfArchEntry.Kind kind) {
        return lookup(kind, hfName) != null;
    }

    /**
     * The names {@code archs.tsv} lists for {@code kind} that no table registers, in table order (unmodifiable). Empty
     * for {@link HfArchEntry.Kind#TEXT} once every text family is registered.
     */
    public static List<String> unmapped(HfArchEntry.Kind kind) {
        Map<String, HfArchEntry> registered = Holder.ENTRIES.get(Objects.requireNonNull(kind, "kind"));
        List<String> out = new ArrayList<>();
        for (HfArchNames.Row r : HfArchNames.rows(kind)) {
            if (!registered.containsKey(r.hfName())) {
                out.add(r.hfName());
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * Test support: makes {@code extra} visible to {@link #resolve} and {@link #has} on the calling thread until the
     * returned handle is closed. An overlay entry takes precedence over a table entry of the same name and kind.
     *
     * @throws IllegalArgumentException when {@code extra} registers a name twice for one kind
     * @throws IllegalStateException    when the thread already has an overlay
     */
    public static Overlay withExtraForTest(List<HfArchEntry> extra) {
        Objects.requireNonNull(extra, "extra");
        if (OVERLAY.get() != null) {
            throw new IllegalStateException("an HfArchRegistry test overlay is already installed on this thread");
        }
        Map<HfArchEntry.Kind, Map<String, HfArchEntry>> m;
        try {
            m = merge(Collections.singletonList(extra));
        } catch (IllegalStateException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
        OVERLAY.set(m);
        return new Overlay();
    }

    /** Removes the calling thread's test overlay when closed. */
    public static final class Overlay implements AutoCloseable {
        private Overlay() {
        }

        @Override
        public void close() {
            OVERLAY.remove();
        }
    }

    private static HfArchEntry lookup(HfArchEntry.Kind kind, String name) {
        Map<HfArchEntry.Kind, Map<String, HfArchEntry>> overlay = OVERLAY.get();
        if (overlay != null) {
            HfArchEntry e = overlay.get(kind).get(name);
            if (e != null) {
                return e;
            }
        }
        return Holder.ENTRIES.get(kind).get(name);
    }

    /**
     * Merges tables into one map per kind.
     *
     * @throws IllegalStateException when a name appears twice for one kind
     */
    static Map<HfArchEntry.Kind, Map<String, HfArchEntry>> merge(List<List<HfArchEntry>> tables) {
        Map<HfArchEntry.Kind, Map<String, HfArchEntry>> m = new EnumMap<>(HfArchEntry.Kind.class);
        for (HfArchEntry.Kind k : HfArchEntry.Kind.values()) {
            m.put(k, new LinkedHashMap<>());
        }
        for (List<HfArchEntry> table : tables) {
            for (HfArchEntry e : table) {
                for (String name : e.hfNames()) {
                    HfArchEntry prev = m.get(e.kind()).put(name, e);
                    if (prev != null) {
                        throw new IllegalStateException("Hugging Face architecture '" + name + "' is registered"
                                + " twice as " + e.kind() + " in the HF architecture tables");
                    }
                }
            }
        }
        for (Map.Entry<HfArchEntry.Kind, Map<String, HfArchEntry>> e : m.entrySet()) {
            e.setValue(Collections.unmodifiableMap(e.getValue()));
        }
        return Collections.unmodifiableMap(m);
    }

    /**
     * The text names llama.cpp's converter lists without a registered class ({@code archs.tsv} TEXT rows whose class
     * is {@code -}); converting them fails in the converter too.
     */
    static List<HfArchEntry> rejectedEntries() {
        List<HfArchEntry> out = new ArrayList<>();
        for (HfArchNames.Row r : HfArchNames.rows(HfArchEntry.Kind.TEXT)) {
            if (!r.hasClass()) {
                boolean projector = HfArchNames.row(r.hfName(), HfArchEntry.Kind.MMPROJ).isPresent();
                out.add(HfArchEntry.rejected(Collections.singletonList(r.hfName()), "'" + r.hfName()
                        + "' has no text model conversion in llama.cpp's converter"
                        + (projector ? "; only its projector converts, open it with ModelSources.openProjector"
                        : "")));
            }
        }
        return out;
    }

    /** The converter's projector name of {@code cfg} (base.py:3130-3159 with {@code ModelType.MMPROJ}). */
    private static String projectorName(HfConfig cfg) {
        JsonView root = cfg.raw();
        String name = null;
        Object arches = root.raw("architectures");
        if (arches instanceof List && !((List<?>) arches).isEmpty()) {
            name = nameOf(cfg, ((List<?>) arches).get(0), "architectures[0]");
        } else if (root.has("ssm_cfg")) {
            Object ssm = root.raw("ssm_cfg");
            Object layer = ssm instanceof Map ? ((Map<?, ?>) ssm).get("layer") : null;
            name = (layer == null ? "Mamba" : nameOf(cfg, layer, "ssm_cfg.layer")) + "ForCausalLM";
        }
        Object vision = root.raw("vision_config");
        if (vision instanceof Map && ((Map<?, ?>) vision).get("architectures") != null) {
            Object va = ((Map<?, ?>) vision).get("architectures");
            if (!(va instanceof List) || ((List<?>) va).isEmpty()) {
                throw new GgufFormatException(cfg.source() + ": 'vision_config.architectures' must be a non-empty"
                        + " array");
            }
            name = nameOf(cfg, ((List<?>) va).get(0), "vision_config.architectures[0]");
        }
        if (name == null) {
            throw new GgufFormatException(cfg.source() + ": failed to detect the model architecture");
        }
        return name;
    }

    private static String nameOf(HfConfig cfg, Object v, String where) {
        if (!(v instanceof String) || ((String) v).isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": '" + where + "' must be a non-empty string");
        }
        return (String) v;
    }
}
