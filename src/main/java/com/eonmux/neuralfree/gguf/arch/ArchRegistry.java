package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.plugin.ArchProvider;
import com.eonmux.neuralfree.gguf.plugin.Plugins;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Registry of every architecture: the merge of {@link ArchTableDense}, {@link ArchTableMoe},
 * {@link ArchTableHybrid}, {@link ArchTableSpecial} and the rejected entries for {@link Arch#CLIP},
 * {@link Arch#UNKNOWN}, which this class owns, plus the rows operator-installed plugin
 * jars contribute through {@link Plugins#load(ArchProvider.class)}. Built once on first use; a duplicate
 * row fails the build, a missing row fails the lookup, and a plugin row whose architecture collides with a
 * built-in row or with the row of another plugin is rejected there with both sources named.
 */
public final class ArchRegistry {

    private ArchRegistry() {
    }

    private static final class Holder {
        static final Map<Arch, ArchEntry> ENTRIES = withProviderRows();

        private static Map<Arch, ArchEntry> withProviderRows() {
            Map<Arch, ArchEntry> builtIn = merge(Arrays.asList(
                    rejectedEntries(),
                    ArchTableDense.entries(),
                    ArchTableMoe.entries(),
                    ArchTableHybrid.entries(),
                    ArchTableSpecial.entries(),
                    ArchTableDraft.entries()));
            return mergeProviders(builtIn, Plugins.load(ArchProvider.class));
        }
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

    /**
     * Adds the rows of plugin {@link ArchProvider}s on top of the built-in map, in provider-name order
     * ({@link Plugins#load} sorts them). A row for an architecture the built-in map or an earlier provider
     * registers is rejected with both sources named: plugins never replace built-ins.
     *
     * <p>Public for the plugin tests, which rebuild what {@link Holder} builds without the
     * {@code META-INF/services} resources of the production class path.
     *
     * @throws IllegalStateException with both sources named
     */
    public static Map<Arch, ArchEntry> mergeProviders(Map<Arch, ArchEntry> builtIn, List<ArchProvider> providers) {
        Objects.requireNonNull(builtIn, "builtIn");
        Objects.requireNonNull(providers, "providers");
        Map<Arch, ArchEntry> m = new EnumMap<>(Arch.class);
        m.putAll(builtIn);
        Map<Arch, String> source = new EnumMap<>(Arch.class);
        for (Arch a : builtIn.keySet()) {
            source.put(a, "the built-in architecture tables");
        }
        for (ArchProvider p : providers) {
            addProviderRows(m, source, p);
        }
        return Collections.unmodifiableMap(m);
    }

    private static void addProviderRows(Map<Arch, ArchEntry> m, Map<Arch, String> source, ArchProvider p) {
        String name = p.getClass().getName();
        List<ArchEntry> rows = p.archEntries();
        if (rows == null) {
            throw new IllegalStateException("plugin provider " + name + " returned null rows");
        }
        for (ArchEntry e : rows) {
            if (e == null) {
                throw new IllegalStateException("plugin provider " + name + " returned a null row");
            }
            Arch a = e.arch();
            String prev = source.get(a);
            if (prev != null) {
                throw new IllegalStateException("architecture '" + a.ggufName() + "' is registered by " + prev
                        + " and by plugin provider " + name + "; plugins never replace built-in rows"
                        + " or the row of another provider");
            }
            source.put(a, "plugin provider " + name);
            m.put(a, e);
        }
    }

    /** Architectures llama.cpp cannot run as a text model. */
    static List<ArchEntry> rejectedEntries() {
        List<ArchEntry> out = new ArrayList<>(2);
        out.add(rejected(Arch.CLIP, "architecture 'clip' is a vision/audio projector without a text model;"
                + " open its GGUF beside a text model with"
                + " com.eonmux.neuralfree.gguf.mm.MmprojModel.open (openVisionEncoder for the image towers,"
                + " openAudioEncoder for the audio towers) and the language model GGUF through"
                + " ModelSources.open; a CLIP two-tower embedding file (no projector) opens through"
                + " com.eonmux.neuralfree.gguf.mm.ClipDualEncoder.open"));
        out.add(rejected(Arch.UNKNOWN, "unknown architecture: general.architecture does not name a llama.cpp"
                + " architecture"));
        return out;
    }

    private static ArchEntry rejected(Arch arch, String reason) {
        ArchTraits t = ArchTraits.builder(arch).unsupported(reason).build();
        return ArchEntry.of(t, ArchHparamsHook.NONE);
    }
}
