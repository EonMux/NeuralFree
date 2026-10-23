package com.eonmux.neuralfree.gguf.spec;

/**
 * Multi-head latent attention (deepseek2 family, kimi-linear MLA layers, DSA hosts). Q comes from
 * {@code wq} or the {@code q_lora} path ({@code wq_a -> q_a_norm -> wq_b}); K/V come from the compressed latent
 * {@code kv_a_mqa} split into {@code kvLoraRank} latent dims and {@code headKRope} shared RoPE dims.
 *
 * <p>With {@link #latentCache()} the KV cache stores one row of width {@code kvLoraRank + headKRope} per token
 * and V aliases the first {@code kvLoraRank} values of that row (llama.cpp absorbed MLA, d_k = kv_lora + n_rot,
 * d_v = kv_lora). Otherwise per-head K ({@code headKNope + headKRope}) and V ({@code headV}) are stored.
 * Immutable.
 */
public final class MlaSpec {

    private final int nHead;
    private final int qLoraRank;
    private final int kvLoraRank;
    private final int headKNope;
    private final int headKRope;
    private final int headV;
    private final boolean absorbed;
    private final boolean latentCache;
    private final boolean ropeApplied;
    private final float kqScale;

    private MlaSpec(Builder b) {
        this.nHead = b.nHead;
        this.qLoraRank = b.qLoraRank;
        this.kvLoraRank = b.kvLoraRank;
        this.headKNope = b.headKNope;
        this.headKRope = b.headKRope;
        this.headV = b.headV;
        this.absorbed = b.absorbed;
        this.latentCache = b.latentCache;
        this.ropeApplied = b.ropeApplied;
        this.kqScale = b.kqScale;
    }

    public static Builder builder() {
        return new Builder();
    }

    public int nHead() { return nHead; }
    /** Q low-rank dim; 0 when Q uses a single {@code wq}. */
    public int qLoraRank() { return qLoraRank; }
    public int kvLoraRank() { return kvLoraRank; }
    public int headKNope() { return headKNope; }
    /** Shared RoPE dims of K ({@code n_rot}). */
    public int headKRope() { return headKRope; }
    public int headV() { return headV; }
    /** {@code attn_k_b} / {@code attn_v_b} are present, so Q can be absorbed into the latent space. */
    public boolean absorbed() { return absorbed; }
    /** The cache stores latent rows (MemoryConfig {@code neuralfree.mla.cache} resolved). */
    public boolean latentCache() { return latentCache; }
    /** False when the MLA layer applies no RoPE (kimi-linear). */
    public boolean ropeApplied() { return ropeApplied; }
    /** {@code kq_scale}, including the YaRN mscale correction. */
    public float kqScale() { return kqScale; }

    /** Full per-head K dim {@code headKNope + headKRope}. */
    public int headK() { return headKNope + headKRope; }
    /** Width of one latent cache row: {@code kvLoraRank + headKRope}. */
    public int latentRowWidth() { return kvLoraRank + headKRope; }

    @Override
    public String toString() {
        return "MlaSpec{nHead=" + nHead + ", qLora=" + qLoraRank + ", kvLora=" + kvLoraRank + ", nope=" + headKNope
                + ", rope=" + headKRope + ", headV=" + headV + ", absorbed=" + absorbed + ", latent=" + latentCache
                + ", ropeApplied=" + ropeApplied + ", kqScale=" + kqScale + '}';
    }

    public static final class Builder {
        private int nHead;
        private int qLoraRank;
        private int kvLoraRank;
        private int headKNope;
        private int headKRope;
        private int headV;
        private boolean absorbed;
        private boolean latentCache = true;
        private boolean ropeApplied = true;
        private float kqScale = Float.NaN;

        private Builder() {
        }

        public Builder nHead(int v) { this.nHead = v; return this; }
        public Builder qLoraRank(int v) { this.qLoraRank = v; return this; }
        public Builder kvLoraRank(int v) { this.kvLoraRank = v; return this; }
        public Builder headKNope(int v) { this.headKNope = v; return this; }
        public Builder headKRope(int v) { this.headKRope = v; return this; }
        public Builder headV(int v) { this.headV = v; return this; }
        public Builder absorbed(boolean v) { this.absorbed = v; return this; }
        public Builder latentCache(boolean v) { this.latentCache = v; return this; }
        public Builder ropeApplied(boolean v) { this.ropeApplied = v; return this; }
        public Builder kqScale(float v) { this.kqScale = v; return this; }

        /**
         * @throws IllegalArgumentException for non-positive dims, a latent cache without absorbed weights, or a
         *                                  missing kqScale
         */
        public MlaSpec build() {
            if (nHead <= 0 || kvLoraRank <= 0 || headKNope <= 0 || headV <= 0 || headKRope < 0 || qLoraRank < 0) {
                throw new IllegalArgumentException("MLA dims must be positive: nHead=" + nHead + " kvLora=" + kvLoraRank
                        + " nope=" + headKNope + " rope=" + headKRope + " headV=" + headV + " qLora=" + qLoraRank);
            }
            if (ropeApplied && (headKRope <= 0 || headKRope % 2 != 0)) {
                throw new IllegalArgumentException("MLA with RoPE needs an even headKRope > 0, got " + headKRope);
            }
            if (latentCache && !absorbed) {
                throw new IllegalArgumentException("MLA latent cache needs absorbed attn_k_b/attn_v_b weights");
            }
            if (!(kqScale > 0f) || Float.isInfinite(kqScale)) {
                throw new IllegalArgumentException("MLA kqScale must be a positive finite value, got " + kqScale);
            }
            return new MlaSpec(this);
        }
    }
}
