package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;

import java.util.Objects;

/**
 * Static facts about an architecture that do not depend on the GGUF file. Values that llama.cpp reads from
 * metadata (SWA windows and patterns, scales, eps) are resolved by the architecture's hparams hook into
 * {@link com.eonmux.neuralfree.gguf.hparams.ModelHparams}, not stored here.
 *
 * <p>Immutable; built with {@link #builder(Arch)}. Required: rope type, memory kind and support class, plus
 * the rejection reason when the support class is {@link SupportClass#NA}.
 */
public final class ArchTraits {

    private final Arch arch;
    private final RopeType ropeType;
    private final boolean mropeWhenSections;
    private final MropeTextPositions mropeTextPositions;
    private final MemoryKind memoryKind;
    private final boolean causalDefault;
    private final boolean forcedNonCausal;
    private final NormKind normKind;
    private final ActOp ffnAct;
    private final FfnGating ffnGating;
    private final GatingFunc defaultGating;
    private final PostNormRole postNormRole;
    private final ResidualKind residualKind;
    private final SupportClass support;
    private final String unsupportedReason;
    private final Arch graphFamily;

    private ArchTraits(Builder b) {
        this.arch = b.arch;
        this.ropeType = b.ropeType;
        this.mropeWhenSections = b.mropeWhenSections;
        this.mropeTextPositions = b.mropeTextPositions;
        this.memoryKind = b.memoryKind;
        this.causalDefault = b.causalDefault;
        this.forcedNonCausal = b.forcedNonCausal;
        this.normKind = b.normKind;
        this.ffnAct = b.ffnAct;
        this.ffnGating = b.ffnGating;
        this.defaultGating = b.defaultGating;
        this.postNormRole = b.postNormRole;
        this.residualKind = b.residualKind;
        this.support = b.support;
        this.unsupportedReason = b.unsupportedReason;
        this.graphFamily = b.graphFamily != null ? b.graphFamily : b.arch;
    }

    public static Builder builder(Arch arch) {
        return new Builder(arch);
    }

    public Arch arch() {
        return arch;
    }

    /** RoPE type from llama.cpp {@code llama_model_rope_type} when no metadata condition applies. */
    public RopeType ropeType() {
        return ropeType;
    }

    /** True when the RoPE type becomes {@link RopeType#MROPE} if {@code rope.dimension_sections} is set. */
    public boolean mropeWhenSections() {
        return mropeWhenSections;
    }

    /** Text-token position pattern for multi-section RoPE. */
    public MropeTextPositions mropeTextPositions() {
        return mropeTextPositions;
    }

    public MemoryKind memoryKind() {
        return memoryKind;
    }

    /** Causality used when {@code attention.causal} is absent. */
    public boolean causalDefault() {
        return causalDefault;
    }

    /** True when attention is non-causal regardless of metadata (diffusion and some embedding models). */
    public boolean forcedNonCausal() {
        return forcedNonCausal;
    }

    /** Dominant block norm. */
    public NormKind normKind() {
        return normKind;
    }

    /** FFN activation applied to the gate (or to the up projection when ungated). */
    public ActOp ffnAct() {
        return ffnAct;
    }

    public FfnGating ffnGating() {
        return ffnGating;
    }

    /** MoE gating used when {@code expert_gating_func} is absent or zero. */
    public GatingFunc defaultGating() {
        return defaultGating;
    }

    public PostNormRole postNormRole() {
        return postNormRole;
    }

    public ResidualKind residualKind() {
        return residualKind;
    }

    public SupportClass support() {
        return support;
    }

    /** Why the architecture cannot run; non-null exactly when {@link #support()} is {@link SupportClass#NA}. */
    public String unsupportedReason() {
        return unsupportedReason;
    }

    /** Architecture whose llama.cpp graph this one reuses ({@link #arch()} when it has its own). */
    public Arch graphFamily() {
        return graphFamily;
    }

    @Override
    public String toString() {
        return "ArchTraits{" + arch.ggufName() + ", rope=" + ropeType + ", memory=" + memoryKind
                + ", support=" + support + "}";
    }

    /** Mutable builder; {@link #build()} validates required fields. */
    public static final class Builder {
        private final Arch arch;
        private RopeType ropeType;
        private boolean mropeWhenSections;
        private MropeTextPositions mropeTextPositions = MropeTextPositions.PPP0;
        private MemoryKind memoryKind;
        private boolean causalDefault = true;
        private boolean forcedNonCausal;
        private NormKind normKind = NormKind.RMS;
        private ActOp ffnAct = ActOp.SILU;
        private FfnGating ffnGating = FfnGating.PAR;
        private GatingFunc defaultGating = GatingFunc.SOFTMAX;
        private PostNormRole postNormRole = PostNormRole.NONE;
        private ResidualKind residualKind = ResidualKind.STANDARD;
        private SupportClass support;
        private String unsupportedReason;
        private Arch graphFamily;

        private Builder(Arch arch) {
            this.arch = Objects.requireNonNull(arch, "arch");
        }

        public Builder ropeType(RopeType v) { this.ropeType = Objects.requireNonNull(v); return this; }
        public Builder mropeWhenSections(boolean v) { this.mropeWhenSections = v; return this; }
        public Builder mropeTextPositions(MropeTextPositions v) { this.mropeTextPositions = Objects.requireNonNull(v); return this; }
        public Builder memoryKind(MemoryKind v) { this.memoryKind = Objects.requireNonNull(v); return this; }
        public Builder causalDefault(boolean v) { this.causalDefault = v; return this; }
        public Builder forcedNonCausal(boolean v) { this.forcedNonCausal = v; return this; }
        public Builder normKind(NormKind v) { this.normKind = Objects.requireNonNull(v); return this; }
        public Builder ffnAct(ActOp v) { this.ffnAct = Objects.requireNonNull(v); return this; }
        public Builder ffnGating(FfnGating v) { this.ffnGating = Objects.requireNonNull(v); return this; }
        public Builder defaultGating(GatingFunc v) { this.defaultGating = Objects.requireNonNull(v); return this; }
        public Builder postNormRole(PostNormRole v) { this.postNormRole = Objects.requireNonNull(v); return this; }
        public Builder residualKind(ResidualKind v) { this.residualKind = Objects.requireNonNull(v); return this; }
        public Builder support(SupportClass v) { this.support = Objects.requireNonNull(v); return this; }
        public Builder graphFamily(Arch v) { this.graphFamily = Objects.requireNonNull(v); return this; }

        /** Marks the architecture {@link SupportClass#NA} with the message shown when it is loaded. */
        public Builder unsupported(String reason) {
            if (reason == null || reason.trim().isEmpty()) {
                throw new IllegalArgumentException("unsupported reason must be non-empty for " + arch);
            }
            this.support = SupportClass.NA;
            this.unsupportedReason = reason;
            return this;
        }

        public ArchTraits build() {
            if (support == null) {
                throw new IllegalStateException("support class not set for " + arch);
            }
            if (support == SupportClass.NA) {
                if (unsupportedReason == null) {
                    throw new IllegalStateException("NA architecture " + arch + " needs a reason; use unsupported()");
                }
                if (ropeType == null) {
                    ropeType = RopeType.NONE;
                }
                if (memoryKind == null) {
                    memoryKind = MemoryKind.NONE;
                }
            } else if (unsupportedReason != null) {
                throw new IllegalStateException("supported architecture " + arch + " must not carry a rejection reason");
            }
            if (ropeType == null) {
                throw new IllegalStateException("rope type not set for " + arch);
            }
            if (memoryKind == null) {
                throw new IllegalStateException("memory kind not set for " + arch);
            }
            return new ArchTraits(this);
        }
    }
}
