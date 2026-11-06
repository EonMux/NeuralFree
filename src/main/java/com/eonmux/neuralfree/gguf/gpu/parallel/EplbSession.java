package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.engine.EplbConfig;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.ops.moe.MoeExpertHeat;
import com.eonmux.neuralfree.gguf.ops.moe.MoeRankingFile;
import com.eonmux.neuralfree.gguf.ops.moe.RankedExpert;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The expert-parallel load balancer session of one loaded model (vLLM's {@code EplbState} without its runtime
 * rebalance): the routed-row profile the expert placement reads at load, and the recorder of the rows counted while
 * the model runs, written back when it closes.
 *
 * <p>The profile is a {@code moe-cache-ranking.bin} ranking ({@link MoeRankingFile}): routed rows per (weight layer,
 * expert). Its entries feed {@link EplbPlacement}, which packs the experts over the ranks; the layers it does not
 * count keep their contiguous blocks. With a save file the routed MoE blocks of rank 0 count their selections into a
 * {@link MoeExpertHeat}; {@link #close()} merges the file's current heats of the same shape and writes the totals, so
 * a deployment alternates recording runs and balanced runs on one file. Without either path nothing is read and
 * nothing is recorded. The session owns nothing else; {@link #close()} is idempotent and best effort.</p>
 */
public final class EplbSession implements AutoCloseable {

    private static final Logger LOGGER = LogManager.getLogger(EplbSession.class);

    private final ModelSpec spec;
    private final Path profilePath;
    private final List<RankedExpert> profile;
    private final Path savePath;
    private final MoeExpertHeat heat;
    private final EplbPlacement.Options options;
    private final int nExperts;
    private boolean closed;

    private EplbSession(ModelSpec spec, Path profilePath, List<RankedExpert> profile, Path savePath,
                        MoeExpertHeat heat, EplbPlacement.Options options) {
        this.spec = spec;
        this.profilePath = profilePath;
        this.profile = profile;
        this.savePath = savePath;
        this.heat = heat;
        this.options = options;
        this.nExperts = MoeCacheRanking.maxExperts(spec);
    }

    /**
     * Opens the session of {@code spec} from the configured paths.
     *
     * @param profile the expert-load profile read now, or null for none
     * @param save    the file the counted load is written to at close, or null for none
     * @throws IOException              when the profile is not a regular file or cannot be read
     * @throws IllegalArgumentException when both paths are null
     */
    public static EplbSession open(Path profile, Path save, ModelSpec spec) throws IOException {
        return open(profile, save, spec, EplbPlacement.Options.defaults());
    }

    /**
     * Opens the session of {@code spec} from a configuration: the profile read now, the recorder when a save file is
     * configured, and the placement options (policy, redundant experts, groups, nodes). Runtime rebalancing alone
     * needs no paths: nothing is read and nothing is recorded, and the placement starts from the contiguous blocks
     * with any redundant replicas spread from zero load.
     *
     * @return the session, or null when the balancer is off
     * @throws IOException              when the profile is configured but is not a regular file or cannot be read
     * @throws IllegalArgumentException when the balancer is on without a profile, a save path or runtime rebalancing
     */
    public static EplbSession open(EplbConfig cfg, ModelSpec spec) throws IOException {
        Objects.requireNonNull(cfg, "cfg");
        if (!cfg.enabled()) {
            return null;
        }
        if (cfg.profile() == null && cfg.save() == null && !cfg.rebalance()) {
            throw new IllegalArgumentException(EplbConfig.PROP_EPLB + "=on needs " + EplbConfig.PROP_EPLB_PROFILE
                    + ", " + EplbConfig.PROP_EPLB_SAVE + " or " + EplbConfig.PROP_EPLB_REBALANCE + "=on: without one"
                    + " nothing is read, recorded or rearranged");
        }
        EplbPlacement.Options options = EplbPlacement.Options.of(cfg.policy(), cfg.numRedundant(), cfg.numGroups(),
                cfg.numNodes());
        if (cfg.profile() == null && cfg.save() == null) {
            return new EplbSession(spec, null, Collections.emptyList(), null, null, options);
        }
        return open(cfg.profile(), cfg.save(), spec, options);
    }

    /**
     * Opens the session of {@code spec} from the configured paths.
     *
     * @param profile the expert-load profile read now, or null for none
     * @param save    the file the counted load is written to at close, or null for none
     * @throws IOException              when the profile is not a regular file or cannot be read
     * @throws IllegalArgumentException when both paths are null
     */
    private static EplbSession open(Path profile, Path save, ModelSpec spec,
                                    EplbPlacement.Options options) throws IOException {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(options, "options");
        if (profile == null && save == null) {
            throw new IllegalArgumentException("an EPLB session needs a profile, a save path or both");
        }
        List<RankedExpert> entries = Collections.emptyList();
        if (profile != null) {
            if (!Files.isRegularFile(profile)) {
                throw new IOException("the expert-load profile " + profile + " is not a regular file");
            }
            entries = MoeRankingFile.read(profile).entries();
        }
        MoeExpertHeat recorder = save != null ? new MoeExpertHeat() : null;
        if (profile == null) {
            LOGGER.info("expert load balancer: recording the expert load into {}", save);
        } else {
            LOGGER.info("expert load balancer: {} profile entr{} from {}{}", entries.size(),
                    entries.size() == 1 ? "y" : "ies", profile, save != null ? ", recording into " + save : "");
        }
        return new EplbSession(spec, profile, entries, save, recorder, options);
    }

    /** The profile read at open, or null when none was configured. */
    public Path profilePath() {
        return profilePath;
    }

    /** The profile's (layer, expert) routed rows, empty without a profile. */
    public List<RankedExpert> profile() {
        return profile;
    }

    /** The file the counted load is written to, or null when the session records nothing. */
    public Path savePath() {
        return savePath;
    }

    /** The recorder the routed MoE blocks count into, or null when the session records nothing. */
    public MoeExpertHeat heat() {
        return heat;
    }

    /** True when the session has a profile to balance from. */
    public boolean balances() {
        return !profile.isEmpty();
    }

    /**
     * The load-balanced placement of the spec's experts over {@code ranks}, or null when the split keeps its
     * contiguous expert blocks: without a profile the placement exists only when redundant experts are configured,
     * in which case every layer is placed with zero load so the physical banks hold the replicas from the start
     * (all redundant slots land on the lowest expert id, as the reference's ties do with no counted load).
     *
     * @throws IllegalArgumentException from {@link EplbPlacement#of} for a profile or a physical count that cannot
     *                                  be packed
     */
    public EplbPlacement placement(int ranks) {
        if (!profile.isEmpty()) {
            return EplbPlacement.of(spec, ranks, profile, options);
        }
        return options.numRedundant() > 0 ? EplbPlacement.uniform(spec, ranks, options) : null;
    }

    /**
     * Writes the counted load into the save path when any routed rows were counted: the session's heats merged with
     * the file's current heats of the same expert count, as a version 2 ranking. Failures are named in the log; close
     * is best effort. Idempotent.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (heat == null || savePath == null || heat.isEmpty() || nExperts <= 0) {
            return;
        }
        MoeExpertHeat total = new MoeExpertHeat();
        total.merge(heat.snapshot());
        mergeSaved(total);
        List<RankedExpert> entries = total.snapshot();
        if (entries.isEmpty()) {
            return;
        }
        try {
            MoeRankingFile.write(savePath, entries, nExperts);
            LOGGER.info("expert load profile: {} (layer, expert) pair(s) written to {}", entries.size(), savePath);
        } catch (IOException | RuntimeException e) {
            LOGGER.warn("expert load profile {} was not written: {}", savePath, e.getMessage());
        }
    }

    /** Adds the saved file's current heats of the same expert count to the totals. */
    private void mergeSaved(MoeExpertHeat total) {
        if (savePath == null || !Files.isRegularFile(savePath)) {
            return;
        }
        try {
            MoeRankingFile.Parsed saved = MoeRankingFile.read(savePath);
            if (saved.nExperts() != nExperts) {
                LOGGER.info("expert load profile {}: saved ranking of {} expert(s) is not merged, the model has {}",
                        savePath, saved.nExperts(), nExperts);
                return;
            }
            total.merge(saved.entries());
        } catch (IOException e) {
            LOGGER.info("expert load profile {}: saved ranking not merged: {}", savePath, e.getMessage());
        }
    }

    @Override
    public String toString() {
        return "EplbSession{profile=" + (profilePath != null ? profilePath + " (" + profile.size() + " entries)"
                : "none") + ", save=" + (savePath != null ? savePath : "off") + "}";
    }
}
