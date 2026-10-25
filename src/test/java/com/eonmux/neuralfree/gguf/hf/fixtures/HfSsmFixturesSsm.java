package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the plain converted converter classes: {@code MambaModel} and {@code Mamba2Model}
 * (mamba.py:18-200). Two layers, hidden size 8, inner size 16 (expand 2), state size 16 and the exact tensor
 * names and configuration keys of the converter and the generated pattern tables, so the registered entries
 * build and claim a complete model of 16 padded vocabulary rows.
 */
public final class HfSsmFixturesSsm {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int INNER = 16;
    private static final int STATE = 16;
    private static final int CONV = 4;
    private static final int DT_RANK = 1;

    private HfSsmFixturesSsm() {
    }

    /**
     * {@code MambaForCausalLM}: the builtin GPT-NeoX vocabulary (no tokenizer files) at the pad-8 size of 16 rows,
     * {@code d_conv} 4, {@code dt_rank} 1 and mixer tensors of the plain Mamba-1 (mamba.py:46-100).
     */
    public static SyntheticHf.Builder mamba() {
        SyntheticHf.Builder b = builder("{'architectures': ['MambaForCausalLM'], 'hidden_size': " + DIM
                + ", 'num_hidden_layers': " + LAYERS + ", 'vocab_size': 10}");
        b.tokenizerJson(null);
        b.tokenizerConfigJson(null);
        b.bf16("backbone.embeddings.weight", 16, DIM);
        b.bf16("backbone.final_layer_norm.weight", DIM);
        b.bf16("embed_out.weight", DIM, 16);
        for (int l = 0; l < LAYERS; l++) {
            String p = "backbone.layers." + l + ".";
            b.bf16(p + "norm.weight", DIM);
            b.bf16(p + "mixer.in_proj.weight", 2 * INNER, DIM);
            b.bf16(p + "mixer.conv1d.weight", INNER, 1, CONV);
            b.bf16(p + "mixer.conv1d.bias", INNER);
            b.bf16(p + "mixer.x_proj.weight", DT_RANK + 2 * STATE, INNER);
            b.bf16(p + "mixer.dt_proj.weight", INNER, DT_RANK);
            b.bf16(p + "mixer.dt_proj.bias", INNER);
            b.bf16(p + "mixer.A_log", INNER, STATE);
            b.bf16(p + "mixer.D", INNER);
            b.bf16(p + "mixer.out_proj.weight", DIM, INNER);
        }
        return b;
    }

    /**
     * {@code Mamba2ForCausalLM}: the pad-16 Mamba-2 with grouped norms ({@code n_groups} 4), {@code dt_rank}
     * {@code d_inner / mamba_d_head} = 4 heads, the A/D trailing-1 reshapes and the {@code dt_bias} rename
     * (mamba.py:143-200); the GPT-2 tokenizer covers the padded vocabulary.
     */
    public static SyntheticHf.Builder mamba2() {
        SyntheticHf.Builder b = builder("{'architectures': ['Mamba2ForCausalLM'], 'hidden_size': " + DIM
                + ", 'num_hidden_layers': " + LAYERS + ", 'vocab_size': 10, 'mamba_d_head': 4, 'n_groups': 4"
                + ", 'state_size': " + STATE + "}");
        b.bf16("backbone.embeddings.weight", 16, DIM);
        b.bf16("backbone.final_layer_norm.weight", DIM);
        b.bf16("lm_head.weight", DIM, 16);
        for (int l = 0; l < LAYERS; l++) {
            String p = "backbone.layers." + l + ".";
            b.bf16(p + "norm.weight", DIM);
            b.bf16(p + "mixer.in_proj.weight", 2 * INNER + 2 * 4 * STATE + 4, DIM);
            b.bf16(p + "mixer.conv1d.weight", INNER, 1, CONV);
            b.bf16(p + "mixer.conv1d.bias", INNER);
            b.bf16(p + "mixer.dt_bias", INNER);
            b.bf16(p + "mixer.A_log", INNER / 4, STATE);
            b.bf16(p + "mixer.D", INNER);
            b.bf16(p + "mixer.norm.weight", 4 * (INNER / 4));
            b.bf16(p + "mixer.out_proj.weight", DIM, INNER);
        }
        return b;
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }

    /**
     * One end-to-end fixture: its leading HF architecture name and the checkpoint builder.
     */
    public static final class Named {
        /** The HF architecture of the fixture. */
        public final String hfArch;
        /** The builder of the checkpoint. */
        public final Supplier<SyntheticHf.Builder> fixture;

        Named(String hfArch, Supplier<SyntheticHf.Builder> fixture) {
            this.hfArch = hfArch;
            this.fixture = fixture;
        }

        @Override
        public String toString() {
            return hfArch;
        }
    }

    private static Named named(String hfArch, Supplier<SyntheticHf.Builder> fixture) {
        return new Named(hfArch, fixture);
    }

    /** Every fixture of this file, in table order. */
    public static List<Named> all() {
        return Arrays.asList(
                named("MambaForCausalLM", HfSsmFixturesSsm::mamba),
                named("Mamba2ForCausalLM", HfSsmFixturesSsm::mamba2));
    }
}
