package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Registry of every architecture: the merge of {@link ArchTableDense}, {@link ArchTableMoe},
 * {@link ArchTableHybrid}, {@link ArchTableSpecial} and the rejected entries for {@link Arch#CLIP},
 * {@link Arch#GPTJ} and {@link Arch#UNKNOWN}, which this class owns. Built once on first use; a duplicate
 * row fails the build, a missing row fails the lookup.
 */
public final class ArchRegistry {

    private ArchRegistry() {
    }

    private static final class Holder {
        static final Map<Arch, ArchEntry> ENTRIES = merge(Arrays.asList(
                rejectedEntries(),
                ArchTableDense.entries(),
                ArchTableMoe.entries(),
                ArchTableHybrid.entries(),
                ArchTableSpecial.entries()));
    }

    /**
     * The entry of an architecture.
     *
     * @throws UnsupportedArchException when no table registers it
     */
    public static ArchEntry get(Arch arch) {
        ArchEntry e = Holder.ENTRIES.get(arch);
        if (e == null) {
            throw new UnsupportedArchException(arch, "architecture '" + arch.ggufName()
                    + "' has no entry in the architecture tables");
        }
        return e;
    }

    /** The entry for a {@code general.architecture} value; unknown names resolve to {@link Arch#UNKNOWN}. */
    public static ArchEntry get(String ggufName) {
        return get(Arch.fromGgufName(ggufName));
    }

    /** Shorthand for {@code get(arch).traits()}. */
    public static ArchTraits traits(Arch arch) {
        return get(arch).traits();
    }

    /** True when a table registers the architecture. */
    public static boolean has(Arch arch) {
        return Holder.ENTRIES.containsKey(arch);
    }

    /** Architectures no table registers. Empty once every table is complete. */
    public static Set<Arch> missing() {
        Set<Arch> out = EnumSet.allOf(Arch.class);
        out.removeAll(Holder.ENTRIES.keySet());
        return Collections.unmodifiableSet(out);
    }

    /**
     * Merges tables into one map.
     *
     * @throws IllegalStateException when an architecture appears twice
     */
    static Map<Arch, ArchEntry> merge(List<List<ArchEntry>> tables) {
        Map<Arch, ArchEntry> m = new EnumMap<>(Arch.class);
        for (List<ArchEntry> table : tables) {
            for (ArchEntry e : table) {
                ArchEntry prev = m.put(e.arch(), e);
                if (prev != null) {
                    throw new IllegalStateException("architecture '" + e.arch().ggufName()
                            + "' is registered twice in the architecture tables");
                }
            }
        }
        return Collections.unmodifiableMap(m);
    }

    /** Architectures llama.cpp cannot run as a text model. */
    static List<ArchEntry> rejectedEntries() {
        List<ArchEntry> out = new ArrayList<>(3);
        out.add(rejected(Arch.CLIP, "architecture 'clip' is a vision/audio projector without a text model"
                + " (llama.cpp loads no hparams for clip); load the language model GGUF instead"));
        out.add(rejected(Arch.GPTJ, "architecture 'gptj' has no model implementation in llama.cpp and is not"
                + " supported"));
        out.add(rejected(Arch.UNKNOWN, "unknown architecture: general.architecture does not name a llama.cpp"
                + " architecture"));
        return out;
    }

    private static ArchEntry rejected(Arch arch, String reason) {
        ArchTraits t = ArchTraits.builder(arch).unsupported(reason).build();
        return ArchEntry.of(t, ArchHparamsHook.NONE);
    }
}
