package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Locale;
import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * Chooses the context length of a model from its memory budget:
 * <ul>
 *   <li>with {@code neuralfree.ctx} set, that length, validated: it must be positive and its memory plan must fit
 *       the budget; a length above {@code n_ctx_train} is allowed with a warning, as llama.cpp allows it;</li>
 *   <li>otherwise {@code n_ctx_train} when its plan fits, else the largest multiple of {@link #CONTEXT_STEP}
 *       below {@code n_ctx_train} whose plan fits (llama.cpp pads contexts to 256, llama-context.cpp:387). There
 *       is no fixed cap.</li>
 * </ul>
 * The size of a context is {@link #requiredBytes}: the plan's {@link MemoryPlan#totalBytes()} (including the
 * recurrent-state snapshot shadows when {@link MemoryConfig#stateSnapshots()} and the DSV4 compressor rollback planes
 * of {@link MemoryConfig#rollbackSnapshots()}, which llama.cpp allocates with the cache) plus, with
 * {@link MemoryConfig#nextnMemory()}, the caches of the model's own NextN / MTP draft context with the draft-context
 * settings ({@link MemoryPlanner#planNextn(ModelSpec, MemoryPlan, int)}), plus the memory of a
 * {@link Companion} model that shares the device (a separate speculative drafter), both allocated after the fit, as
 * llama.cpp fits the main and the draft context together (common/common.cpp:1252-1280, common/fit.cpp:203-258). No
 * part decreases as the context grows (rows, rings and compressed caches are non-decreasing in {@code nCtx}), so the
 * search is a binary search. When nothing fits, a {@link ContextBudgetException} reports the smallest plan and the
 * budget so the caller can free memory (for example by streaming weights) and retry. {@link #reservedBytes} gives the
 * bytes the fit held beside the chosen plan, which the caller keeps out of later budgets until the drafter allocates
 * them. Stateless and thread-safe.
 */
public final class ContextFitter {

    private static final Logger LOGGER = LogManager.getLogger(ContextFitter.class);

    /** Granularity of automatically chosen contexts. */
    public static final int CONTEXT_STEP = 256;

    /** Budget meaning "no limit". */
    public static final long UNLIMITED = Long.MAX_VALUE;

    private static final double MIB = 1024.0 * 1024.0;

    private ContextFitter() {
    }

    /** {@link #fit(ModelSpec, MemoryConfig, long, Companion)} without a companion model. */
    public static int fit(ModelSpec spec, MemoryConfig cfg, long budgetBytes) {
        return fit(spec, cfg, budgetBytes, Companion.NONE, WHOLE_PLAN);
    }

    /** {@link #fit(ModelSpec, MemoryConfig, long, Companion, PlanBytes)} with the default sizer. */
    public static int fit(ModelSpec spec, MemoryConfig cfg, long budgetBytes, Companion companion) {
        return fit(spec, cfg, budgetBytes, companion, WHOLE_PLAN);
    }

    /**
     * The context length to plan with, leaving room for {@code companion} at every candidate context, where a
     * candidate's size is what {@code sizer} reads from its plan instead of the plan's whole byte total. A hybrid
     * load sizes only the layers it places on its device this way.
     *
     * @param budgetBytes bytes the sized plan, the NextN draft memory and the companion may occupy together;
     *                    {@link #UNLIMITED} for no limit
     * @throws IllegalArgumentException when the budget is negative or the model has no positive training context
     *                                  while the context is automatic
     * @throws ContextBudgetException   when the user's context, or every automatic context, exceeds the budget
     */
    public static int fit(ModelSpec spec, MemoryConfig cfg, long budgetBytes, Companion companion, PlanBytes sizer) {
        return fit(spec, cfg, budgetBytes, companion, sizer, null);
    }

    /**
     * The context length whose plan over the layers {@code layerFilter} accepts (their {@code il}) fits
     * {@code budgetBytes}: the per-device fit of a layer split, where each device holds only its own layers' caches
     * ({@link MemoryPlanner#planForContext(ModelSpec, MemoryConfig, int, IntPredicate)}), with the NextN / MTP
     * draft memory when {@code cfg} reserves it. No logging per candidate; the caller logs the chosen context.
     *
     * @throws IllegalArgumentException when the budget is negative or the model has no positive training context
     *                                  while the context is automatic
     * @throws ContextBudgetException   when the user's context, or every automatic context, exceeds the budget
     */
    public static int fitLayers(ModelSpec spec, MemoryConfig cfg, long budgetBytes, IntPredicate layerFilter) {
        return fitLayers(spec, cfg, budgetBytes, layerFilter, Companion.NONE);
    }

    /**
     * {@link #fitLayers(ModelSpec, MemoryConfig, long, IntPredicate)} leaving room for {@code companion} at every
     * candidate context: the device of a layer split that also holds a drafter (the main GPU of a split with a
     * draft model).
     */
    public static int fitLayers(ModelSpec spec, MemoryConfig cfg, long budgetBytes, IntPredicate layerFilter,
                                Companion companion) {
        Objects.requireNonNull(layerFilter, "layerFilter");
        return fit(spec, cfg, budgetBytes, companion, WHOLE_PLAN, layerFilter);
    }

    private static int fit(ModelSpec spec, MemoryConfig cfg, long budgetBytes, Companion companion, PlanBytes sizer,
                           IntPredicate layerFilter) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(companion, "companion");
        Objects.requireNonNull(sizer, "sizer");
        if (budgetBytes < 0) {
            throw new IllegalArgumentException("memory budget must be >= 0 bytes, got " + budgetBytes);
        }
        String name = spec.arch().ggufName();
        int train = spec.hparams().nCtxTrain();
        if (!cfg.autoContext()) {
            return checkUserContext(spec, cfg, budgetBytes, train, companion, sizer, layerFilter);
        }
        if (train <= 0) {
            throw new IllegalArgumentException(name + " has no positive training context (n_ctx_train = " + train
                    + "); set " + MemoryConfig.PROP_CTX + " explicitly");
        }
        Sized full = size(spec, cfg, train, companion, sizer, layerFilter);
        if (full.bytes() <= budgetBytes) {
            return train;
        }
        int maxSteps = (train - 1) / CONTEXT_STEP;
        if (maxSteps == 0) {
            throw new ContextBudgetException(name, train, full, budgetBytes);
        }
        Sized smallest = size(spec, cfg, CONTEXT_STEP, companion, sizer, layerFilter);
        if (smallest.bytes() > budgetBytes) {
            throw new ContextBudgetException(name, CONTEXT_STEP, smallest, budgetBytes);
        }
        int lo = 1;
        int hi = maxSteps;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (size(spec, cfg, mid * CONTEXT_STEP, companion, sizer, layerFilter).bytes() <= budgetBytes) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        int nCtx = lo * CONTEXT_STEP;
        LOGGER.info("{}: context {} of the training context {} fits the memory budget of {} MiB "
                        + "(the training context needs {} MiB{})", name, nCtx, train, mib(budgetBytes),
                mib(full.bytes()), companion == Companion.NONE ? "" : ", " + companion.name() + " included");
        return nCtx;
    }

    /** {@link #requiredBytes(ModelSpec, MemoryConfig, int, Companion)} without a companion model. */
    public static long requiredBytes(ModelSpec spec, MemoryConfig cfg, int nCtx) {
        return requiredBytes(spec, cfg, nCtx, Companion.NONE, WHOLE_PLAN);
    }

    /** {@link #requiredBytes(ModelSpec, MemoryConfig, int, Companion, PlanBytes)} with the default sizer. */
    public static long requiredBytes(ModelSpec spec, MemoryConfig cfg, int nCtx, Companion companion) {
        return requiredBytes(spec, cfg, nCtx, companion, WHOLE_PLAN);
    }

    /**
     * Bytes a context of {@code nCtx} needs as {@code sizer} reads it from its plan, plus, with
     * {@link MemoryConfig#nextnMemory()}, the NextN / MTP draft memory, plus the memory of {@code companion} at that
     * context.
     *
     * @throws IllegalArgumentException for a non-positive context or a spec that cannot be planned
     */
    public static long requiredBytes(ModelSpec spec, MemoryConfig cfg, int nCtx, Companion companion, PlanBytes sizer) {
        Objects.requireNonNull(companion, "companion");
        return size(spec, cfg, nCtx, companion, sizer, null).bytes();
    }

    /**
     * Bytes a context of {@code nCtx} needs over the layers {@code layerFilter} accepts (their {@code il}): the
     * per-device size of a layer split ({@link #fitLayers}), with the NextN / MTP draft memory when {@code cfg}
     * reserves it. No companion model.
     *
     * @throws IllegalArgumentException for a non-positive context or a spec that cannot be planned
     */
    public static long requiredLayerBytes(ModelSpec spec, MemoryConfig cfg, int nCtx, IntPredicate layerFilter) {
        return requiredLayerBytes(spec, cfg, nCtx, layerFilter, Companion.NONE);
    }

    /** {@link #requiredLayerBytes(ModelSpec, MemoryConfig, int, IntPredicate)} plus {@code companion}'s memory. */
    public static long requiredLayerBytes(ModelSpec spec, MemoryConfig cfg, int nCtx, IntPredicate layerFilter,
                                          Companion companion) {
        Objects.requireNonNull(layerFilter, "layerFilter");
        Objects.requireNonNull(companion, "companion");
        return size(spec, cfg, nCtx, companion, WHOLE_PLAN, layerFilter).bytes();
    }

    /**
     * Bytes the fit held beside {@code plan} for memory allocated after it: the NextN / MTP draft memory when
     * {@code cfg} reserves it ({@link MemoryConfig#nextnMemory()}) plus the memory of {@code companion}. Budgets
     * measured after the plan's own allocations (for example the streaming slots) leave these bytes out.
     *
     * @param plan the plan of the chosen context, planned with {@code cfg}
     */
    public static long reservedBytes(ModelSpec spec, MemoryConfig cfg, MemoryPlan plan, Companion companion) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(cfg, "cfg");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(companion, "companion");
        return new Sized(spec, cfg, plan, companion, WHOLE_PLAN).reservedBytes();
    }

    private static Sized size(ModelSpec spec, MemoryConfig cfg, int nCtx, Companion companion, PlanBytes sizer,
                              IntPredicate layerFilter) {
        MemoryPlan plan = layerFilter != null ? MemoryPlanner.planQuietly(spec, cfg, nCtx, layerFilter)
                : MemoryPlanner.planQuietly(spec, cfg, nCtx);
        return new Sized(spec, cfg, plan, companion, sizer);
    }

    /** Sizes a candidate plan for the fit; the whole model's byte total by default. */
    @FunctionalInterface
    public interface PlanBytes {
        long of(MemoryPlan plan);
    }

    /** The default {@link PlanBytes}: the plan's whole byte total. */
    public static final PlanBytes WHOLE_PLAN = MemoryPlan::totalBytes;

    /** A plan with the bytes of the NextN draft memory and of the companion reserved beside it. */
    private static final class Sized {
        final MemoryPlan plan;
        final PlanBytes sizer;
        final long nextnBytes;
        final long companionBytes;
        final String companionName;

        Sized(ModelSpec spec, MemoryConfig cfg, MemoryPlan plan, Companion companion, PlanBytes sizer) {
            this.plan = plan;
            this.sizer = sizer;
            this.nextnBytes = cfg.nextnMemory() ? MemoryPlanner.nextnBytes(spec, plan) : 0L;
            this.companionBytes = companion.bytes(plan);
            this.companionName = companion.name();
        }

        long reservedBytes() {
            return Math.addExact(nextnBytes, companionBytes);
        }

        long bytes() {
            return Math.addExact(sizer.of(plan), reservedBytes());
        }
    }

    private static int checkUserContext(ModelSpec spec, MemoryConfig cfg, long budgetBytes, int train,
                                        Companion companion, PlanBytes sizer, IntPredicate layerFilter) {
        String name = spec.arch().ggufName();
        int nCtx = cfg.nCtx();
        if (nCtx <= 0) {
            throw new IllegalArgumentException(MemoryConfig.PROP_CTX + " must be positive, got " + nCtx);
        }
        Sized sized = size(spec, cfg, nCtx, companion, sizer, layerFilter);
        if (sized.bytes() > budgetBytes) {
            throw new ContextBudgetException(name, nCtx, sized, budgetBytes);
        }
        if (train > 0 && nCtx > train) {
            LOGGER.warn("{}: {}={} exceeds the training context {}; positions beyond it were never seen in training",
                    name, MemoryConfig.PROP_CTX, nCtx, train);
        }
        return nCtx;
    }

    private static String mib(long bytes) {
        return bytes == UNLIMITED ? "unlimited" : String.format(Locale.ROOT, "%.1f", bytes / MIB);
    }

    /**
     * The memory of a second model that shares the device and whose context follows this model's context, so the two
     * are fitted together (llama.cpp {@code common_fit_extra_model}, common/fit.h:14-22, measured again at every
     * candidate context, common/fit.cpp:203-258): a speculative drafter. At a candidate plan it takes its fixed bytes
     * (the weights it places on the device), its step bytes for every token of its step ubatch (its forward scratch,
     * the draft ubatch set with {@link #withStepUbatch} when one is set, else the plan's ubatch) and, for a drafter
     * with caches of its own, the bytes of those caches at the candidate's context and ubatch (the draft ubatch of
     * the cache settings when one is set). Immutable.
     */
    public static final class Companion {

        /** Bytes of a companion's own caches at a context and ubatch; non-decreasing in the context. */
        @FunctionalInterface
        public interface CacheBytes {
            /**
             * @throws IllegalArgumentException when the caches cannot be sized for this context
             */
            long at(int nCtx, int ubatch);
        }

        /** No companion model. */
        public static final Companion NONE = new Companion("no companion", 0L, 0L, null, null, 0);

        private final String name;
        private final long fixedBytes;
        private final long bytesPerStepToken;
        private final CacheBytes caches;
        private final String cacheOwner;
        /** The ubatch the step bytes are counted over; 0 follows the target plan's ubatch. */
        private final int stepUbatch;

        private Companion(String name, long fixedBytes, long bytesPerStepToken, CacheBytes caches,
                          String cacheOwner, int stepUbatch) {
            this.name = name;
            this.fixedBytes = fixedBytes;
            this.bytesPerStepToken = bytesPerStepToken;
            this.caches = caches;
            this.cacheOwner = cacheOwner;
            this.stepUbatch = stepUbatch;
        }

        /**
         * A companion without caches of its own (it holds none, or reads the target's caches).
         *
         * @param name              how the companion is named in messages, for example "draft model weights"
         * @param fixedBytes        bytes independent of the context
         * @param bytesPerStepToken bytes per token of the target plan's ubatch
         * @throws IllegalArgumentException for a blank name or negative sizes
         */
        public static Companion of(String name, long fixedBytes, long bytesPerStepToken) {
            check(name, fixedBytes, bytesPerStepToken);
            return new Companion(name.trim(), fixedBytes, bytesPerStepToken, null, null, 0);
        }

        /**
         * A companion with caches of its own: {@code drafter} planned with {@code memory} at the target plan's
         * context and ubatch, with the draft ubatch of {@code memory} when one is set
         * ({@link MemoryConfig#draftUbatchOr}).
         *
         * @throws IllegalArgumentException for a blank name, negative sizes, or a drafter that cannot be planned
         *                                  with {@code memory}
         */
        public static Companion withCaches(String name, long fixedBytes, long bytesPerStepToken, ModelSpec drafter,
                                           MemoryConfig memory) {
            Objects.requireNonNull(drafter, "drafter");
            Objects.requireNonNull(memory, "memory");
            CacheBytes planned = (nCtx, ubatch) -> MemoryPlanner.planQuietly(drafter,
                    memory.toBuilder().nCtx(nCtx).ubatch(memory.draftUbatchOr(ubatch)).build(), nCtx).totalBytes();
            return sized(name, fixedBytes, bytesPerStepToken, planned, drafter.arch().ggufName(),
                    Math.min(memory.ubatch(), CONTEXT_STEP));
        }

        /**
         * A companion with caches of its own that {@code caches} sizes at the target plan's context and ubatch, for
         * a drafter whose caches no {@link MemoryPlan} describes.
         *
         * @param owner what holds the caches, for messages
         * @throws IllegalArgumentException for a blank name or owner, negative sizes, or caches that cannot be sized
         */
        public static Companion withCacheBytes(String name, long fixedBytes, long bytesPerStepToken, String owner,
                                               CacheBytes caches) {
            Objects.requireNonNull(caches, "caches");
            if (owner == null || owner.trim().isEmpty()) {
                throw new IllegalArgumentException("companion cache owner is blank");
            }
            return sized(name, fixedBytes, bytesPerStepToken, caches, owner.trim(), 1);
        }

        /**
         * A companion counting its step bytes over {@code ubatch} tokens instead of the target plan's ubatch: the
         * step size a draft context runs its forward passes with (llama.cpp {@code -ubd}, whose 0 keeps the target's
         * {@code -ub}). 0 keeps this companion's rule.
         */
        public Companion withStepUbatch(int ubatch) {
            if (ubatch < 0) {
                throw new IllegalArgumentException(name + ": step ubatch must be >= 0, got " + ubatch);
            }
            if (ubatch == 0 || ubatch == stepUbatch) {
                return this;
            }
            return new Companion(name, fixedBytes, bytesPerStepToken, caches, cacheOwner, ubatch);
        }

        private static Companion sized(String name, long fixedBytes, long bytesPerStepToken, CacheBytes caches,
                                       String owner, int probeUbatch) {
            check(name, fixedBytes, bytesPerStepToken);
            Companion c = new Companion(name.trim(), fixedBytes, bytesPerStepToken, caches, owner, 0);
            c.cacheBytes(CONTEXT_STEP, probeUbatch);
            return c;
        }

        /**
         * A companion carrying both sides' sizes: {@code this} plus {@code other} — the fixed bytes and the
         * per-step bytes summed, caches combined (each sized for the context, the sums kept), the narrower step
         * ubatch rule kept ({@code 0} follows the target plan). {@link #NONE} returns the other side unchanged,
         * so an absent companion adds nothing.
         */
        public Companion plus(Companion other) {
            Objects.requireNonNull(other, "other");
            if (this == NONE) {
                return other;
            }
            if (other == NONE) {
                return this;
            }
            String joined = name + " + " + other.name;
            CacheBytes both = caches == null ? other.caches
                    : other.caches == null ? caches
                    : (nCtx, ubatch) -> Math.addExact(caches.at(nCtx, ubatch), other.caches.at(nCtx, ubatch));
            String owner = cacheOwner == null ? other.cacheOwner
                    : other.cacheOwner == null ? cacheOwner : cacheOwner + " and " + other.cacheOwner;
            int stepUbatch = this.stepUbatch != 0 ? this.stepUbatch : other.stepUbatch;
            Companion c = new Companion(joined, Math.addExact(fixedBytes, other.fixedBytes),
                    Math.addExact(bytesPerStepToken, other.bytesPerStepToken), both, owner, stepUbatch);
            c.cacheBytes(CONTEXT_STEP, 1);
            return c;
        }

        private static void check(String name, long fixedBytes, long bytesPerStepToken) {
            if (name == null || name.trim().isEmpty()) {
                throw new IllegalArgumentException("companion name is blank");
            }
            if (fixedBytes < 0 || bytesPerStepToken < 0) {
                throw new IllegalArgumentException(name + ": companion sizes must be >= 0, got " + fixedBytes
                        + " fixed bytes and " + bytesPerStepToken + " bytes per step token");
            }
        }

        /** The name used in messages. */
        public String name() {
            return name;
        }

        /** Bytes independent of the context. */
        public long fixedBytes() {
            return fixedBytes;
        }

        /** Whether the companion has caches of its own. */
        public boolean hasCaches() {
            return caches != null;
        }

        /** Bytes the companion takes beside a target plan. */
        public long bytes(MemoryPlan targetPlan) {
            Objects.requireNonNull(targetPlan, "targetPlan");
            int stepTokens = stepUbatch > 0 ? stepUbatch : targetPlan.ubatch();
            long b = Math.addExact(fixedBytes, Math.multiplyExact(bytesPerStepToken, (long) stepTokens));
            return caches == null ? b : Math.addExact(b, cacheBytes(targetPlan.nCtx(), targetPlan.ubatch()));
        }

        private long cacheBytes(int nCtx, int ubatch) {
            long b = caches.at(nCtx, ubatch);
            if (b < 0) {
                throw new IllegalArgumentException(name + ": caches of " + cacheOwner + " sized " + b + " bytes at a"
                        + " context of " + nCtx);
            }
            return b;
        }

        @Override
        public String toString() {
            return name + "{fixed=" + mib(fixedBytes) + " MiB, perStepToken=" + bytesPerStepToken + " B"
                    + (caches != null ? ", caches of " + cacheOwner : "") + '}';
        }
    }

    /** No context the caller allows fits the memory budget. Carries the plan that was tried. */
    public static final class ContextBudgetException extends IllegalStateException {

        private static final long serialVersionUID = 1L;

        private final int nCtx;
        private final long requiredBytes;
        private final long budgetBytes;

        private ContextBudgetException(String arch, int nCtx, Sized sized, long budgetBytes) {
            super(arch + ": a context of " + nCtx + " tokens needs " + mib(sized.bytes()) + " MiB of memory ("
                    + breakdown(sized) + ") but the budget is " + mib(budgetBytes) + " MiB; lower "
                    + MemoryConfig.PROP_CTX + ", use a smaller KV type (" + MemoryConfig.PROP_KV_TYPE
                    + "=q8_0, q5_0 or q4_0) or free device memory");
            this.nCtx = nCtx;
            this.requiredBytes = sized.bytes();
            this.budgetBytes = budgetBytes;
        }

        private static String breakdown(Sized sized) {
            MemoryPlan plan = sized.plan;
            StringBuilder sb = new StringBuilder("KV ").append(mib(plan.kvBytes()))
                    .append(", recurrent state ").append(mib(plan.stateBytes()));
            if (plan.snapshotBytes() > 0) {
                sb.append(", state snapshots ").append(mib(plan.snapshotBytes()));
            }
            sb.append(", side caches ").append(mib(plan.sideCacheBytes()))
                    .append(", positions ").append(mib(plan.positionBytes()));
            if (sized.nextnBytes > 0) {
                sb.append(", NextN draft memory ").append(mib(sized.nextnBytes));
            }
            if (sized.companionBytes > 0) {
                sb.append(", ").append(sized.companionName).append(' ').append(mib(sized.companionBytes));
            }
            return sb.toString();
        }

        /** Context that was tried (the smallest allowed one in automatic mode). */
        public int nCtx() { return nCtx; }
        public long requiredBytes() { return requiredBytes; }
        public long budgetBytes() { return budgetBytes; }
    }
}
