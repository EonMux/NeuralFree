package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.names.HfArchNames;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * How a vLLM {@code FusedMoE} parameter ({@code experts.w13_weight}, {@code experts.w2_weight} and their biases)
 * splits back into the per-expert tensors of the Hugging Face checkpoint the converter reads, chosen by the
 * checkpoint's converter class ({@link HfArchNames}).
 *
 * <p>vLLM packs the gate and up projections of every expert into {@code w13} (first half gate, second half up) and
 * the down projection into {@code w2}. The converter classes name those projections either {@code gate_proj}/
 * {@code up_proj}/{@code down_proj} or Mixtral's {@code w1}/{@code w3}/{@code w2}; a few keep them fused. The family
 * is a property of the converter class, so no split is guessed from a tensor shape alone: a class this table does
 * not cover is refused by name.</p>
 */
public final class VllmExpertLayouts {

    /** How the Hugging Face checkpoint spells the expert projections. */
    public enum Kind {
        /** One tensor per expert and projection, under {@code <prefix>.<expert>.<part><tail>}. */
        INDEXED,
        /** The fused gate/up stack and the down stack under {@code <prefix><suffix>}, as vLLM stores them. */
        FUSED,
        /** A family whose expert layout NeuralFree does not convert from a vLLM state dict. */
        UNSUPPORTED
    }

    /** One converter class's expert layout. Immutable. */
    public static final class Layout {
        private final Kind kind;
        private final String gate;
        private final String up;
        private final String down;
        private final String gateUpSuffix;
        private final String downSuffix;
        private final boolean transpose;
        private final String refusal;

        private Layout(Kind kind, String gate, String up, String down, String gateUpSuffix,
                       String downSuffix, boolean transpose, String refusal) {
            this.kind = kind;
            this.gate = gate;
            this.up = up;
            this.down = down;
            this.gateUpSuffix = gateUpSuffix;
            this.downSuffix = downSuffix;
            this.transpose = transpose;
            this.refusal = refusal;
        }

        static Layout indexed(String gate, String up, String down) {
            return new Layout(Kind.INDEXED, gate, up, down, null, null, false, null);
        }

        static Layout fused(String gateUpSuffix, String downSuffix, boolean transpose) {
            return new Layout(Kind.FUSED, null, null, null, gateUpSuffix, downSuffix, transpose, null);
        }

        static Layout unsupported(String refusal) {
            return new Layout(Kind.UNSUPPORTED, null, null, null, null, null, false, refusal);
        }

        /** How the checkpoint spells the expert projections. */
        public Kind kind() {
            return kind;
        }

        /** The gate projection's name in {@link Kind#INDEXED} layouts. */
        public String gate() {
            return gate;
        }

        /** The up projection's name in {@link Kind#INDEXED} layouts. */
        public String up() {
            return up;
        }

        /** The down projection's name in {@link Kind#INDEXED} layouts. */
        public String down() {
            return down;
        }

        /** The suffix of the fused gate/up stack in {@link Kind#FUSED} layouts. */
        public String gateUpSuffix() {
            return gateUpSuffix;
        }

        /** The suffix of the down stack in {@link Kind#FUSED} layouts. */
        public String downSuffix() {
            return downSuffix;
        }

        /**
         * Whether a fused stack is stored transposed (vLLM's {@code [experts, projection, hidden]} against the
         * checkpoint's {@code [experts, hidden, projection]}).
         */
        public boolean transpose() {
            return transpose;
        }

        /** Why the family is refused, or null. */
        public String refusal() {
            return refusal;
        }

        @Override
        public String toString() {
            switch (kind) {
                case INDEXED:
                    return "indexed(" + gate + ", " + up + ", " + down + ")";
                case FUSED:
                    return "fused(" + gateUpSuffix + ", " + downSuffix + (transpose ? ", transposed" : "") + ")";
                default:
                    return "unsupported: " + refusal;
            }
        }
    }

    private static final Map<String, Layout> BY_CONVERTER_CLASS = layouts();
    private static final Map<String, Layout> BY_ARCHITECTURE = architectures();

    private VllmExpertLayouts() {
    }

    /**
     * The families NeuralFree registers directly (they are vLLM-only, absent from the converter's {@code archs.tsv})
     * whose expert projections are DeepSeek's per-expert {@code gate_proj}/{@code up_proj}/{@code down_proj}.
     */
    private static Map<String, Layout> architectures() {
        Map<String, Layout> m = new HashMap<>();
        Layout gateUpDown = Layout.indexed("gate_proj", "up_proj", "down_proj");
        for (String arch : new String[]{
                "PanguProMoEV2ForCausalLM", "PanguUltraMoEForCausalLM",
                "LongcatFlashForCausalLM", "LongCatFlashMTPModel", "LongcatFlashNgramForCausalLM",
                "Param2MoEForCausalLM", "SarvamMLAForCausalLM", "MistralLarge3ForCausalLM",
                "DeepseekV3BidirectionalModel"}) {
            m.put(arch, gateUpDown);
        }
        return Collections.unmodifiableMap(m);
    }

    private static Map<String, Layout> layouts() {
        Map<String, Layout> m = new HashMap<>();
        Layout gateUpDown = Layout.indexed("gate_proj", "up_proj", "down_proj");
        // DeepSeek's family, the Qwen MoE family, the Bailing/Sarvam family and Jamba name the projections
        for (String cls : new String[]{
                "Qwen2MoeModel", "Qwen3MoeModel", "Qwen3NextModel",
                "DeepseekModel", "DeepseekV2Model", "DeepseekV32Model", "DeepseekV41Model", "DeepseekV4Model",
                "BailingMoeModel", "BailingMoeV2Model", "BailingMoeV3Model", "SarvamMoEModel",
                "Dots1Model", "Ernie4_5MoeModel", "JambaModel", "MimoV2Model"}) {
            m.put(cls, gateUpDown);
        }
        Layout w1w2w3 = Layout.indexed("w1", "w3", "w2");
        // Mixtral's spelling and the classes that follow it
        for (String cls : new String[]{
                "LlamaModel", "PhiMoeModel", "ArcticModel",
                "MiniMaxText01Model", "MiniMaxM2Model", "MiniMaxM3Model",
                "KimiLinearModel", "KimiK3Model"}) {
            m.put(cls, w1w2w3);
        }
        // Llama4 keeps the fused stacks, transposed against vLLM's layout
        m.put("Llama4Model", Layout.fused(".gate_up_proj", ".down_proj", true));
        // families whose fused stack is interleaved or packed rather than a contiguous split
        m.put("GptOssModel", Layout.unsupported("its fused expert stacks are interleaved or packed"));
        m.put("DbrxModel", Layout.unsupported("its experts are unindexed three-dimensional stacks"));
        m.put("GrokModel", Layout.unsupported("its experts use the linear/linear_1/linear_v naming"));
        m.put("Gemma4Model", Layout.unsupported("its expert layout is not a per-expert split"));
        return Collections.unmodifiableMap(m);
    }

    /**
     * The expert layout of the checkpoint configured by {@code config}, or empty when its converter class is not
     * covered.
     */
    public static Optional<Layout> of(com.eonmux.neuralfree.gguf.hf.config.HfConfig config) {
        Layout byArchitecture = BY_ARCHITECTURE.get(config.architecture());
        if (byArchitecture != null) {
            return Optional.of(byArchitecture);
        }
        Optional<HfArchNames.Row> row = HfArchNames.row(config.architecture(), HfArchEntry.Kind.TEXT);
        if (!row.isPresent() || !row.get().hasClass()) {
            return Optional.empty();
        }
        return Optional.ofNullable(BY_CONVERTER_CLASS.get(row.get().converterClass()));
    }

    /** Every converter class this table covers (unmodifiable), for messages. */
    public static List<String> coveredClasses() {
        return Collections.unmodifiableList(new java.util.ArrayList<>(BY_CONVERTER_CLASS.keySet()));
    }
}
