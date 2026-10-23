package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.GluOp;

import java.util.Objects;

/**
 * A dense feed-forward network: the layer FFN, a MoE shared expert, arctic's parallel residual MLP, or the RWKV
 * channel mix. Follows llama.cpp {@code build_ffn} (src/llama-graph.cpp:1940-2110):
 * <ul>
 *   <li>ungated: {@code down(act(up(x)))};</li>
 *   <li>{@link FfnGating#SEQ} with a gate: {@code down(act(gate(up(x))))};</li>
 *   <li>{@link FfnGating#PAR}: {@code down(glu(gate(x), up(x)))};</li>
 *   <li>{@link FfnGating#FUSED_UP}: {@code ffn_up} holds {@code [gate | up]} rows ({@code 2 * nFf}),
 *       {@code down(glu(first half, second half))}.</li>
 * </ul>
 * With {@link #activationSparsityStdMul()} {@code > 0} (gemma3n) the gate rows pass through the Gaussian top-k
 * filter before the GLU. With {@link #biases()} the projections add their optional {@code <base>.bias} tensors when
 * the file carries them; without it the graph passes {@code NULL} for every bias and the tensors are not read.
 * Immutable.
 */
public final class FfnSpec {

    /** Which FFN of the layer this is; selects the tensor names. */
    public enum Role {
        /** {@code ffn_gate} / {@code ffn_up} / {@code ffn_down}. */
        DENSE(TensorNames.FFN_GATE, TensorNames.FFN_UP, TensorNames.FFN_DOWN),
        /** {@code ffn_gate_shexp} / {@code ffn_up_shexp} / {@code ffn_down_shexp}. */
        SHARED_EXPERT(TensorNames.FFN_GATE_SHEXP, TensorNames.FFN_UP_SHEXP, TensorNames.FFN_DOWN_SHEXP),
        /** arctic residual MLP summed with the MoE branch; same tensors as {@link #DENSE}. */
        PARALLEL_DENSE(TensorNames.FFN_GATE, TensorNames.FFN_UP, TensorNames.FFN_DOWN),
        /**
         * RWKV channel mix: {@code sigmoid(receptance(xr)) * value(relu(key(xk))^2)} (RWKV6; RWKV7 has no
         * receptance); gate is the receptance, up the key and down the value. The receptance gate is not a GLU,
         * so the spec is built with {@code act(false, RELU_SQR)}.
         */
        RWKV_CHANNEL_MIX(TensorNames.CHANNEL_MIX_RECEPTANCE, TensorNames.CHANNEL_MIX_KEY, TensorNames.CHANNEL_MIX_VALUE);

        private final String gate;
        private final String up;
        private final String down;

        Role(String gate, String up, String down) {
            this.gate = gate;
            this.up = up;
            this.down = down;
        }

        /** Gate tensor base name. */
        public String gate() { return gate; }
        /** Up tensor base name. */
        public String up() { return up; }
        /** Down tensor base name. */
        public String down() { return down; }
    }

    private final Role role;
    private final int nFf;
    private final FfnGating gating;
    private final boolean gated;
    private final GluOp glu;
    private final ActOp act;
    private final ActParams params;
    private final float activationSparsityStdMul;
    private final boolean biases;

    private FfnSpec(Builder b) {
        this.role = b.role;
        this.nFf = b.nFf;
        this.gating = b.gating;
        this.gated = b.gated;
        this.glu = b.glu;
        this.act = b.act;
        this.params = b.params;
        this.activationSparsityStdMul = b.activationSparsityStdMul;
        this.biases = b.biases;
    }

    public static Builder builder(Role role) {
        return new Builder(Objects.requireNonNull(role, "role"));
    }

    public Role role() { return role; }
    /** Hidden width ({@code n_ff} of the layer, or of the shared expert). */
    public int nFf() { return nFf; }
    public FfnGating gating() { return gating; }
    /** A gate projection (or the gate half of a fused up) exists. */
    public boolean gated() { return gated; }
    /** GLU op for PAR / FUSED_UP gating; null otherwise. */
    public GluOp glu() { return glu; }
    /** Activation for ungated and SEQ FFNs; null when {@link #glu()} is set. */
    public ActOp act() { return act; }
    public ActParams params() { return params; }
    /** gemma3n Gaussian top-k multiplier on the gate rows; 0 disables. */
    public float activationSparsityStdMul() { return activationSparsityStdMul; }
    /** The projections add {@code <base>.bias} when present; false when the graph ignores those tensors. */
    public boolean biases() { return biases; }

    /** True when the activation is a GLU of gate and up. */
    public boolean usesGlu() { return glu != null; }

    /** Gate weight suffix ({@code ffn_gate.weight} for DENSE); meaningful only when {@link #gated()}. */
    public String gateWeight() { return TensorNames.weight(role.gate()); }
    public String upWeight() { return TensorNames.weight(role.up()); }
    public String downWeight() { return TensorNames.weight(role.down()); }

    @Override
    public String toString() {
        return "FfnSpec{" + role + ", nFf=" + nFf + ", " + gating + (gated ? " gated" : " ungated")
                + ", op=" + (glu != null ? glu : act) + ", sparsity=" + activationSparsityStdMul
                + (biases ? "" : ", biases unused") + '}';
    }

    public static final class Builder {
        private final Role role;
        private int nFf;
        private FfnGating gating = FfnGating.PAR;
        private boolean gated = true;
        private GluOp glu = GluOp.SWIGLU;
        private ActOp act;
        private ActParams params = ActParams.NONE;
        private float activationSparsityStdMul;
        private boolean biases = true;

        private Builder(Role role) {
            this.role = role;
        }

        public Builder nFf(int v) { this.nFf = v; return this; }

        /** PAR or FUSED_UP gating with a GLU op. */
        public Builder glu(FfnGating gating, GluOp op) {
            this.gating = Objects.requireNonNull(gating, "gating");
            this.glu = Objects.requireNonNull(op, "glu");
            this.act = null;
            this.gated = true;
            return this;
        }

        /** Ungated ({@code gated == false}) or SEQ-gated FFN with a plain activation. */
        public Builder act(boolean gated, ActOp op) {
            this.gating = FfnGating.SEQ;
            this.gated = gated;
            this.act = Objects.requireNonNull(op, "act");
            this.glu = null;
            return this;
        }

        public Builder params(ActParams v) { this.params = Objects.requireNonNull(v, "params"); return this; }
        public Builder activationSparsityStdMul(float v) { this.activationSparsityStdMul = v; return this; }
        /** Whether the projections add their optional biases (default true). */
        public Builder biases(boolean v) { this.biases = v; return this; }

        /** @throws IllegalArgumentException for a non-positive width or an op that does not fit the gating */
        public FfnSpec build() {
            if (nFf <= 0) {
                throw new IllegalArgumentException(role + " FFN width must be positive, got " + nFf);
            }
            if (gating == FfnGating.SEQ ? (act == null || glu != null) : (glu == null || act != null || !gated)) {
                throw new IllegalArgumentException(role + " FFN gating " + gating + " needs "
                        + (gating == FfnGating.SEQ ? "an activation" : "a GLU op") + ", got glu=" + glu + " act=" + act);
            }
            if (!(activationSparsityStdMul >= 0f)) {
                throw new IllegalArgumentException(role + " FFN activation sparsity multiplier must be >= 0, got "
                        + activationSparsityStdMul);
            }
            if (activationSparsityStdMul > 0f && glu == null) {
                throw new IllegalArgumentException(role + " FFN activation sparsity applies to gate rows of a GLU FFN");
            }
            if (role == Role.RWKV_CHANNEL_MIX && (gating != FfnGating.SEQ || act != ActOp.RELU_SQR)) {
                throw new IllegalArgumentException("RWKV channel mix uses act(false, RELU_SQR) on the key; got "
                        + gating + "/" + act);
            }
            return new FfnSpec(this);
        }
    }
}
