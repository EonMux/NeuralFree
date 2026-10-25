package com.eonmux.neuralfree.gguf.hf.names;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Hugging Face tensor name to GGUF tensor name for one architecture and block count: the port of gguf-py's
 * {@code TensorNameMap} (LC/gguf-py/gguf/tensor_mapping.py:3081-3112) over the generated tables of
 * {@link HfNameTables}.
 *
 * <p>Construction follows {@code TensorNameMap.__init__} exactly: for every model-level tensor of
 * {@code mappings_cfg} that the architecture lists ({@code MODEL_TENSORS[arch]}) the GGUF name itself and each
 * pattern are keys, in table order; then for each block {@code 0..nBlocks-1}, for every block tensor of
 * {@code block_mappings_cfg} the architecture lists, the formatted GGUF name and each formatted pattern are keys.
 * A later assignment overwrites an earlier one, so table order and the architecture's tensor list decide the
 * patterns two tensors share. An architecture's {@code arch_block_mappings_cfg} entries replace the pattern list of
 * their tensor in place and append new tensors in override order; unlike tensor_mapping.py:3091, which updates the
 * shared class dict, the merge is per map and leaves the tables untouched.</p>
 *
 * <p>Lookups follow {@code get_type_and_name}: an exact key first, then for each try-suffix the name without it,
 * with the suffix appended to the result. Maps are immutable, thread-safe and cached per (architecture, block
 * count).</p>
 */
public final class HfTensorNameMap {

    /** Suffixes {@code ModelBase.map_tensor_name} retries (base.py:622). */
    private static final String[] DEFAULT_SUFFIXES = {".weight", ".bias"};

    private static final Map<Key, HfTensorNameMap> CACHE = new ConcurrentHashMap<>();

    /** A mapped name: the converter's {@code MODEL_TENSOR} and the GGUF tensor name. Immutable. */
    public static final class Mapped {
        private final String tensor;
        private final String ggufName;

        Mapped(String tensor, String ggufName) {
            this.tensor = tensor;
            this.ggufName = ggufName;
        }

        /** The {@code MODEL_TENSOR} name ({@code ATTN_Q}). */
        public String tensor() { return tensor; }

        /** The GGUF tensor name ({@code blk.3.attn_q.weight}). */
        public String ggufName() { return ggufName; }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Mapped)) {
                return false;
            }
            Mapped m = (Mapped) o;
            return tensor.equals(m.tensor) && ggufName.equals(m.ggufName);
        }

        @Override
        public int hashCode() {
            return Objects.hash(tensor, ggufName);
        }

        @Override
        public String toString() {
            return ggufName + " (" + tensor + ")";
        }
    }

    private static final class Key {
        final Arch arch;
        final int nBlocks;

        Key(Arch arch, int nBlocks) {
            this.arch = arch;
            this.nBlocks = nBlocks;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Key && ((Key) o).arch == arch && ((Key) o).nBlocks == nBlocks;
        }

        @Override
        public int hashCode() {
            return arch.hashCode() * 31 + nBlocks;
        }
    }

    private final Arch arch;
    private final int nBlocks;
    private final Set<String> tensors;
    private final Map<String, Mapped> mapping;

    private HfTensorNameMap(Arch arch, int nBlocks) {
        this.arch = arch;
        this.nBlocks = nBlocks;
        this.tensors = HfNameTables.archTensors(arch.ggufName());
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("llama.cpp's converter lists no tensors for architecture '"
                    + arch.ggufName() + "'");
        }
        this.mapping = build(arch, nBlocks, tensors);
    }

    /**
     * The map of {@code arch} with {@code nBlocks} blocks.
     *
     * @throws IllegalArgumentException for {@link Arch#UNKNOWN}, an architecture the converter has no tensor list
     *                                  for, or a block count outside {@code [0, HfConfig.MAX_LAYERS]}
     */
    public static HfTensorNameMap forArch(Arch arch, int nBlocks) {
        Objects.requireNonNull(arch, "arch");
        if (arch == Arch.UNKNOWN) {
            throw new IllegalArgumentException("no tensor name map for an unknown architecture");
        }
        if (nBlocks < 0 || nBlocks > HfConfig.MAX_LAYERS) {
            throw new IllegalArgumentException("block count " + nBlocks + " is outside [0, " + HfConfig.MAX_LAYERS
                    + "]");
        }
        return CACHE.computeIfAbsent(new Key(arch, nBlocks), k -> new HfTensorNameMap(k.arch, k.nBlocks));
    }

    private static Map<String, Mapped> build(Arch arch, int nBlocks, Set<String> tensors) {
        Map<String, String> names = HfNameTables.tensorNames();
        Map<String, Mapped> m = new HashMap<>();
        for (Map.Entry<String, List<String>> e : HfNameTables.globalPatterns().entrySet()) {
            String tensor = e.getKey();
            if (!tensors.contains(tensor)) {
                continue;
            }
            Mapped target = new Mapped(tensor, names.get(tensor));
            m.put(target.ggufName(), target);
            for (String key : e.getValue()) {
                m.put(key, target);
            }
        }
        Map<String, List<String>> block = new LinkedHashMap<>(HfNameTables.blockPatterns());
        block.putAll(HfNameTables.blockOverrides(arch.ggufName()));
        List<Map.Entry<String, List<String>>> listed = new ArrayList<>();
        for (Map.Entry<String, List<String>> e : block.entrySet()) {
            if (tensors.contains(e.getKey())) {
                listed.add(e);
            }
        }
        for (int bid = 0; bid < nBlocks; bid++) {
            String b = Integer.toString(bid);
            for (Map.Entry<String, List<String>> e : listed) {
                String tensor = e.getKey();
                String template = names.get(tensor);
                String name = template.replace("{bid}", b);
                if (name.indexOf('{') >= 0) {
                    throw new IllegalStateException("tensor '" + tensor + "' of architecture '" + arch.ggufName()
                            + "' has a GGUF name template with a placeholder besides {bid}: " + template);
                }
                Mapped target = new Mapped(tensor, name);
                m.put(name, target);
                for (String key : e.getValue()) {
                    m.put(key.replace("{bid}", b), target);
                }
            }
        }
        return m;
    }

    /** The architecture. */
    public Arch arch() {
        return arch;
    }

    /** The number of blocks the map covers. */
    public int blockCount() {
        return nBlocks;
    }

    /** True when the converter maps {@code tensor} ({@code MODEL_TENSOR} name) for this architecture. */
    public boolean hasTensor(String tensor) {
        return tensors.contains(tensor);
    }

    /**
     * The mapping of {@code hfName}: the exact key, else for each of {@code trySuffixes} that the name ends with,
     * the name without it, with the suffix appended to the GGUF name ({@code get_type_and_name}).
     */
    public Optional<Mapped> map(String hfName, String... trySuffixes) {
        Objects.requireNonNull(hfName, "hfName");
        Mapped exact = mapping.get(hfName);
        if (exact != null) {
            return Optional.of(exact);
        }
        for (String suffix : trySuffixes) {
            if (!suffix.isEmpty() && hfName.endsWith(suffix)) {
                Mapped base = mapping.get(hfName.substring(0, hfName.length() - suffix.length()));
                if (base != null) {
                    return Optional.of(new Mapped(base.tensor(), base.ggufName() + suffix));
                }
            }
        }
        return Optional.empty();
    }

    /**
     * The GGUF name of {@code hfName}, retrying {@code .weight} and {@code .bias} ({@code map_tensor_name},
     * base.py:622-626).
     *
     * @throws GgufFormatException {@code "Can not map tensor '<name>'"} when no key matches
     */
    public String mapOrThrow(String hfName) {
        Optional<Mapped> m = map(hfName, DEFAULT_SUFFIXES);
        if (!m.isPresent()) {
            throw new GgufFormatException("Can not map tensor '" + hfName + "'");
        }
        return m.get().ggufName();
    }

    /**
     * The GGUF base name of {@code tensor} in block {@code bid}, without a suffix ({@code format_tensor_name},
     * base.py:600-607, for hooks that build names directly).
     *
     * @param bid the block, or -1 for a model-level tensor
     * @throws IllegalArgumentException when the architecture does not list the tensor, or the block does not fit it
     */
    public String ggufName(String tensor, int bid) {
        if (!tensors.contains(tensor)) {
            throw new IllegalArgumentException("Missing " + tensor + " for MODEL_TENSORS of " + arch.ggufName());
        }
        return HfNameTables.formatTensorName(tensor, bid);
    }

    /** {@link #ggufName(String, int)} followed by {@code suffix} ({@code .weight}, {@code .bias}, or empty). */
    public String ggufName(String tensor, int bid, String suffix) {
        return ggufName(tensor, bid) + Objects.requireNonNull(suffix, "suffix");
    }

    /**
     * True when {@code ggufName} is {@code tensor}'s name in block {@code bid} followed by {@code suffix}
     * ({@code match_model_tensor_name}, base.py:609-620): false when the architecture does not list the tensor, when
     * a block tensor gets no block ({@code bid < 0}) or a model-level tensor gets one.
     */
    public boolean matches(String ggufName, String tensor, int bid, String suffix) {
        if (!tensors.contains(tensor)) {
            return false;
        }
        String template = HfNameTables.tensorNames().get(tensor);
        boolean hasBid = template.contains("{bid}");
        if (hasBid != (bid >= 0)) {
            return false;
        }
        return Objects.equals(ggufName, HfNameTables.formatTensorName(tensor, bid) + suffix);
    }

    @Override
    public String toString() {
        return "HfTensorNameMap{" + arch.ggufName() + ", " + nBlocks + " blocks, " + mapping.size() + " keys}";
    }
}
