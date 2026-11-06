package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * The expert-ranking session of one loaded model, Avalon's {@code moe-cache-ranking.bin} cycle (common/common.cpp:1254
 * and ggml/src/ggml-cuda/moe-cache.cu: the preload, the routed-row profile and the profile save). Opened at load:
 * <ul>
 *   <li><b>Preload.</b> {@value #RANKING_FILE_NAME} next to the model file (inside the model directory for a
 *       converted checkpoint) is read at open and kept hottest first, each entry with its share of the ranking's
 *       heat; the MoE expert cache fills its slots from it before the first graph. On by default, like
 *       {@code GGML_CUDA_MOE_CACHE_PRELOAD}: {@link #PROP_PRELOAD neuralfree.moe.cache.ranking} names another
 *       ranking file, and {@code off} turns the preload off.</li>
 *   <li><b>Profile save.</b> With {@link #PROP_SAVE neuralfree.moe.cache.ranking.save} set to a file, the routed
 *       rows noted by the forward blocks (through {@link #heat()}) are counted, merged with that file's current
 *       heats of the same model shape and written back as a version 2 ranking when the model closes — the
 *       {@code GGML_CUDA_MOE_CACHE_PROFILE_SAVE} cycle. Without the property nothing records and nothing writes.</li>
 * </ul>
 * A configured preload file that does not exist is refused at open, naming the property; a ranking whose contents
 * cannot be read is named in the log and skipped, and the save is best effort: neither fails the model. Mutable and
 * owned by the model's thread, like the forward context. One session per loaded model.
 */
public final class MoeCacheRanking implements AutoCloseable {

    /** Name of the ranking file next to the model (Avalon common/common.cpp:1254-1263). */
    public static final String RANKING_FILE_NAME = "moe-cache-ranking.bin";

    /**
     * Ranking file to preload from: unset or blank (the file next to the model when it exists), {@code off}, or a
     * named file.
     */
    public static final String PROP_PRELOAD = "neuralfree.moe.cache.ranking";
    /** File the routed-row profile is written to at close; unset, blank or {@code off} keeps it off. */
    public static final String PROP_SAVE = "neuralfree.moe.cache.ranking.save";

    private static final Logger LOGGER = LogManager.getLogger(MoeCacheRanking.class);
    private static final String OFF = "off";

    /** How the ranking file to preload from is chosen. */
    public enum PreloadMode {
        /** The {@value RANKING_FILE_NAME} next to the model, when it exists (the default). */
        AUTO,
        /** No preload. */
        OFF,
        /** The file named by {@link Options#preloadPath()}. */
        PATH
    }

    /** Where the session preloads from and where it saves its profile. Immutable. */
    public static final class Options {
        private static final Options AUTO = new Options(PreloadMode.AUTO, null, null);

        private final PreloadMode preloadMode;
        private final Path preloadPath;
        private final Path savePath;

        /**
         * @throws IllegalArgumentException when {@code preloadMode} is {@link PreloadMode#PATH} without a path or
         *                                 another mode with one
         */
        public Options(PreloadMode preloadMode, Path preloadPath, Path savePath) {
            this.preloadMode = Objects.requireNonNull(preloadMode, "preloadMode");
            if (preloadMode == PreloadMode.PATH) {
                this.preloadPath = Objects.requireNonNull(preloadPath, "preloadPath");
            } else {
                if (preloadPath != null) {
                    throw new IllegalArgumentException("preload mode " + preloadMode + " takes no path, got "
                            + preloadPath);
                }
                this.preloadPath = null;
            }
            this.savePath = savePath;
        }

        /** Defaults: the ranking next to the model, no profile save. */
        public static Options auto() {
            return AUTO;
        }

        /**
         * The options of {@code props}: {@link #PROP_PRELOAD} {@code off} (any case) disables the preload, another
         * non-blank value names the ranking file, a blank one keeps the default; {@link #PROP_SAVE} names the
         * profile file unless it is blank or {@code off}.
         *
         * @throws IllegalArgumentException naming the property when a value is not a path
         */
        public static Options fromProperties(Properties props) {
            Objects.requireNonNull(props, "props");
            String save = trimmed(props.getProperty(PROP_SAVE));
            Path savePath = save == null || save.equalsIgnoreCase(OFF) ? null : path(PROP_SAVE, save);
            Options out = new Options(PreloadMode.AUTO, null, savePath);
            String preload = trimmed(props.getProperty(PROP_PRELOAD));
            return preload == null ? out : out.withPreload(preload);
        }

        /**
         * These options with another preload setting, read as a {@link #PROP_PRELOAD} value: {@code off} (any case)
         * disables the preload, anything else names the ranking file. The save path is kept.
         *
         * @throws IllegalArgumentException naming the property when the value is blank or not a path
         */
        public Options withPreload(String value) {
            String preload = trimmed(value);
            if (preload == null) {
                throw new IllegalArgumentException(PROP_PRELOAD + " is empty; use a ranking file or off");
            }
            if (preload.equalsIgnoreCase(OFF)) {
                return new Options(PreloadMode.OFF, null, savePath);
            }
            return new Options(PreloadMode.PATH, path(PROP_PRELOAD, preload), savePath);
        }

        private static String trimmed(String value) {
            if (value == null || value.trim().isEmpty()) {
                return null;
            }
            return value.trim();
        }

        private static Path path(String property, String value) {
            try {
                return Paths.get(value);
            } catch (InvalidPathException e) {
                throw new IllegalArgumentException(property + "='" + value + "' is not a path: " + e.getMessage(), e);
            }
        }

        public PreloadMode preloadMode() {
            return preloadMode;
        }

        /** The ranking file to preload from; non-null only in {@link PreloadMode#PATH}. */
        public Path preloadPath() {
            return preloadPath;
        }

        /** The file the routed-row profile is written to at close; null when the save is off. */
        public Path savePath() {
            return savePath;
        }

        /** True for the defaults: the ranking next to the model and no profile save. */
        public boolean isDefault() {
            return preloadMode == PreloadMode.AUTO && savePath == null;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Options)) {
                return false;
            }
            Options other = (Options) o;
            return preloadMode == other.preloadMode && Objects.equals(preloadPath, other.preloadPath)
                    && Objects.equals(savePath, other.savePath);
        }

        @Override
        public int hashCode() {
            return Objects.hash(preloadMode, preloadPath, savePath);
        }

        @Override
        public String toString() {
            return "ranking=" + (preloadPath != null ? preloadPath.toString()
                    : preloadMode.name().toLowerCase(Locale.ROOT))
                    + (savePath != null ? ", rankingSave=" + savePath : "");
        }
    }

    private final int nExperts;
    private final MoeExpertHeat heat;
    private final Path preloadedFrom;
    private final List<RankedExpert> ranking;
    private final int rankingLayers;
    private final int rankingExperts;
    private final double[] shares;
    private final Path savePath;
    private boolean closed;

    private MoeCacheRanking(int nExperts, MoeExpertHeat heat, Path preloadedFrom, MoeRankingFile.Parsed parsed,
                            Path savePath) {
        this.nExperts = nExperts;
        this.heat = heat;
        this.preloadedFrom = parsed != null ? preloadedFrom : null;
        this.ranking = parsed != null ? parsed.entries() : Collections.<RankedExpert>emptyList();
        this.rankingLayers = parsed != null ? parsed.nLayers() : 0;
        this.rankingExperts = parsed != null ? parsed.nExperts() : 0;
        this.shares = sharesOf(ranking);
        this.savePath = savePath;
    }

    /**
     * Opens the session of a model loaded from {@code modelFile} (a GGUF file or a converted checkpoint's
     * directory; null when the model has no file, which leaves the default preload without a ranking), reading its
     * ranking per {@code options}.
     *
     * @param nExperts the largest expert count of the model's routed MoE layers ({@link #maxExperts}); the profile
     *                 save merges an old ranking only of the same shape and writes it into the header
     * @throws IOException              when the configured preload file ({@link #PROP_PRELOAD}) is not a regular file
     * @throws IllegalArgumentException when the expert count is negative
     */
    public static MoeCacheRanking open(Path modelFile, int nExperts, Options options) throws IOException {
        Objects.requireNonNull(options, "options");
        if (nExperts < 0) {
            throw new IllegalArgumentException("expert count must be >= 0, got " + nExperts);
        }
        Path preload = preloadFile(modelFile, options);
        MoeRankingFile.Parsed parsed = null;
        if (preload != null) {
            try {
                parsed = MoeRankingFile.read(preload);
                LOGGER.info("MoE expert ranking preloaded from {}: {} pair(s)", preload, parsed.entries().size());
            } catch (IOException e) {
                LOGGER.warn("MoE expert ranking {} is ignored: {}", preload, e.getMessage());
            }
        }
        MoeExpertHeat heat = options.savePath() != null ? new MoeExpertHeat() : null;
        return new MoeCacheRanking(nExperts, heat, preload, parsed, options.savePath());
    }

    /** The file the options preload from, or null for none. */
    private static Path preloadFile(Path modelFile, Options options) throws IOException {
        switch (options.preloadMode()) {
            case PATH:
                if (!Files.isRegularFile(options.preloadPath())) {
                    throw new IOException(PROP_PRELOAD + "=" + options.preloadPath() + " is not a regular file");
                }
                return options.preloadPath();
            case AUTO:
                return modelFile != null ? beside(modelFile) : null;
            default:
                return null;
        }
    }

    /** The ranking file next to a model file, or inside the model directory of a converted checkpoint. */
    private static Path beside(Path modelFile) {
        Path absolute = modelFile.toAbsolutePath();
        Path container = Files.isDirectory(absolute) ? absolute : absolute.getParent();
        if (container == null) {
            return null;
        }
        Path candidate = container.resolve(RANKING_FILE_NAME);
        return Files.isRegularFile(candidate) ? candidate : null;
    }

    /** Each entry's heat over the ranking's total heat, the preload's demand shares (0 without heats). */
    private static double[] sharesOf(List<RankedExpert> ranking) {
        double total = 0;
        for (RankedExpert e : ranking) {
            total += e.heat();
        }
        double[] shares = new double[ranking.size()];
        if (total > 0) {
            for (int i = 0; i < ranking.size(); i++) {
                shares[i] = ranking.get(i).heat() / total;
            }
        }
        return shares;
    }

    /**
     * The largest expert count of the model's routed MoE layers, 0 when the model has none.
     *
     * @throws NullPointerException when the spec is null
     */
    public static int maxExperts(ModelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        int max = 0;
        for (LayerSpec layer : spec.layers()) {
            if (layer.moe() != null && MoeBlocks.isMoeKind(layer.ffnKind())) {
                max = Math.max(max, layer.moe().nExpert());
            }
        }
        return max;
    }

    /**
     * The routed-row recorder the forward blocks count into; null when no profile save is configured, so the
     * forward blocks skip the recording entirely.
     */
    public MoeExpertHeat heat() {
        return heat;
    }

    /** True when a ranking with at least one entry was preloaded at open. */
    public boolean preloaded() {
        return !ranking.isEmpty();
    }

    /** The ranking file read at open, or null when none was read. */
    public Path preloadedFrom() {
        return preloadedFrom;
    }

    /** The preloaded ranking, hottest first as the file holds it; empty without a preload. */
    public List<RankedExpert> ranking() {
        return ranking;
    }

    /** The layer count the preloaded ranking's header names; 0 without a preload. */
    public int rankingLayers() {
        return rankingLayers;
    }

    /** The expert count per layer the preloaded ranking's header names; 0 without a preload. */
    public int rankingExperts() {
        return rankingExperts;
    }

    /**
     * The preloaded entries of layers and experts inside {@code [0, nLayers) x [0, nExperts)}, in the file's order
     * (hottest first); the rest are dropped.
     */
    public List<RankedExpert> within(int nLayers, int nExperts) {
        List<RankedExpert> out = new ArrayList<>(ranking.size());
        for (RankedExpert e : ranking) {
            if (e.layer() < nLayers && e.expert() < nExperts) {
                out.add(e);
            }
        }
        return out;
    }

    /**
     * The preloaded entry's share of the ranking's total heat (Avalon's {@code ranking_src} demand share).
     *
     * @throws IllegalArgumentException when the index is outside the ranking
     */
    public double share(int index) {
        if (index < 0 || index >= shares.length) {
            throw new IllegalArgumentException("ranking share index must be in [0, " + shares.length + "), got "
                    + index);
        }
        return shares[index];
    }

    /**
     * Writes the configured profile file when any routed rows were counted: the session's heats merged with the
     * file's current heats (of the same model shape), as a version 2 ranking hottest first. Failures are named in
     * the log; close is best effort, like the reference's session end. Idempotent.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (heat == null || heat.isEmpty()) {
            return;
        }
        MoeExpertHeat total = new MoeExpertHeat();
        total.merge(heat.snapshot());
        mergeSaved(total);
        List<RankedExpert> out = total.snapshot();
        try {
            MoeRankingFile.write(savePath, out, nExperts);
            LOGGER.info("MoE expert profile: {} (layer, expert) pair(s) written to {}", out.size(), savePath);
        } catch (IOException | RuntimeException e) {
            LOGGER.warn("MoE expert profile {} was not written: {}", savePath, e.getMessage());
        }
    }

    /** Adds the saved file's current heats of the same model shape to the totals. */
    private void mergeSaved(MoeExpertHeat total) {
        if (!Files.exists(savePath)) {
            return;
        }
        try {
            MoeRankingFile.Parsed saved = MoeRankingFile.read(savePath);
            if (saved.nExperts() != nExperts) {
                LOGGER.info("MoE expert profile {}: saved ranking of {} expert(s) is not merged, the model has {}",
                        savePath, saved.nExperts(), nExperts);
                return;
            }
            total.merge(saved.entries());
        } catch (IOException e) {
            LOGGER.info("MoE expert profile {}: saved ranking not merged: {}", savePath, e.getMessage());
        }
    }

    @Override
    public String toString() {
        return "MoeCacheRanking{preload=" + (preloadedFrom != null ? preloadedFrom + " (" + ranking.size() + " pair(s))"
                : "none") + ", save=" + (savePath != null ? savePath : "off") + "}";
    }
}
