package com.eonmux.neuralfree.gguf.mm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Predicate;

/**
 * The block layout of a Gemma 3n MobileNetV5 encoder, discovered from the tensor names the way clip.cpp
 * {@code load_tensors} discovers it: up to four stages {@code v.blk.<stage>.<index>.}, each a run of blocks from
 * index 0 that ends at the first index with no block tensor. A block holding {@code conv_exp} is an edge residual,
 * one holding {@code attn.query.proj} a multi-query attention block, one holding {@code dw_start} or
 * {@code pw_exp} a universal inverted residual. The first block of every stage downsamples; the features after
 * the last blocks of stages 2 and 3 (after the last block when there are fewer than four stages) feed the
 * multi-scale fusion adapter. Immutable.
 */
final class MobileNetV5Layout {

    /** Stages the loader looks for. */
    static final int STAGES = 4;

    static final String STEM_CONV = "v.conv_stem.conv.weight";
    static final String STEM_BIAS = "v.conv_stem.conv.bias";
    static final String STEM_NORM = "v.conv_stem.bn.weight";
    static final String MSFA_EXPAND = "v.msfa.ffn.pw_exp.conv.weight";
    static final String MSFA_EXPAND_NORM = "v.msfa.ffn.pw_exp.bn.weight";
    static final String MSFA_PROJECT = "v.msfa.ffn.pw_proj.conv.weight";
    static final String MSFA_PROJECT_NORM = "v.msfa.ffn.pw_proj.bn.weight";
    static final String MSFA_NORM = "v.msfa.norm.weight";
    static final String INPUT_PROJECTION = "mm.input_projection.weight";
    static final String SOFT_EMB_NORM = "mm.soft_emb_norm.weight";

    static final String CONV_EXP = "conv_exp.weight";
    static final String BN1 = "bn1.weight";
    static final String CONV_PWL = "conv_pwl.weight";
    static final String BN2 = "bn2.weight";
    static final String DW_START = "dw_start.conv.weight";
    static final String DW_START_NORM = "dw_start.bn.weight";
    static final String PW_EXP = "pw_exp.conv.weight";
    static final String PW_EXP_NORM = "pw_exp.bn.weight";
    static final String DW_MID = "dw_mid.conv.weight";
    static final String DW_MID_NORM = "dw_mid.bn.weight";
    static final String PW_PROJ = "pw_proj.conv.weight";
    static final String PW_PROJ_NORM = "pw_proj.bn.weight";
    static final String LAYER_SCALE = "layer_scale.gamma";
    static final String ATTN_NORM = "norm.weight";
    static final String QUERY = "attn.query.proj.weight";
    static final String KEY = "attn.key.proj.weight";
    static final String VALUE = "attn.value.proj.weight";
    static final String OUTPUT = "attn.output.proj.weight";
    static final String KEY_DOWN = "attn.key.down_conv.weight";
    static final String KEY_NORM = "attn.key.norm.weight";
    static final String VALUE_DOWN = "attn.value.down_conv.weight";
    static final String VALUE_NORM = "attn.value.norm.weight";

    /** The three block graphs of models/mobilenetv5.cpp. */
    enum Kind {
        /** {@code build_edge_residual}: 3x3 expansion conv, GELU, pointwise projection. */
        EDGE,
        /** {@code build_inverted_residual}: optional depthwise start, expansion, depthwise mid and projection. */
        INVERTED,
        /** {@code build_mobilenet_attn}: multi-query attention with optionally downsampled keys and values. */
        ATTENTION
    }

    /** One block: its stage, index within the stage and graph. */
    static final class Block {
        final int stage;
        final int index;
        final Kind kind;
        private final String prefix;

        Block(int stage, int index, Kind kind) {
            this.stage = stage;
            this.index = index;
            this.kind = kind;
            this.prefix = "v.blk." + stage + "." + index + ".";
        }

        /** The full name of the block tensor {@code suffix}. */
        String name(String suffix) {
            return prefix + suffix;
        }

        @Override
        public String toString() {
            return prefix + kind;
        }
    }

    private final List<Block> blocks;
    private final int[] stageEnds;

    private MobileNetV5Layout(List<Block> blocks, int[] stageEnds) {
        this.blocks = blocks;
        this.stageEnds = stageEnds;
    }

    /**
     * The layout of the names {@code has} admits.
     *
     * @throws IllegalArgumentException for a block that holds the tensors of two block graphs (named)
     */
    static MobileNetV5Layout discover(Predicate<String> has) {
        List<Block> blocks = new ArrayList<>();
        List<Integer> ends = new ArrayList<>();
        for (int stage = 0; stage < STAGES; stage++) {
            int found = 0;
            for (int index = 0; ; index++) {
                Kind kind = kindOf(has, new Block(stage, index, Kind.EDGE));
                if (kind == null) {
                    break;
                }
                blocks.add(new Block(stage, index, kind));
                found++;
            }
            if (found > 0) {
                ends.add(blocks.size() - 1);
            }
        }
        int[] stageEnds = new int[ends.size()];
        for (int i = 0; i < stageEnds.length; i++) {
            stageEnds[i] = ends.get(i);
        }
        return new MobileNetV5Layout(Collections.unmodifiableList(blocks), stageEnds);
    }

    private static Kind kindOf(Predicate<String> has, Block probe) {
        boolean edge = has.test(probe.name(CONV_EXP));
        boolean inverted = has.test(probe.name(DW_START)) || has.test(probe.name(PW_EXP));
        boolean attention = has.test(probe.name(QUERY));
        if ((edge ? 1 : 0) + (inverted ? 1 : 0) + (attention ? 1 : 0) > 1) {
            throw new IllegalArgumentException("MobileNetV5 block " + probe.name("") + " holds the tensors of more"
                    + " than one block graph (conv_exp: edge residual, dw_start / pw_exp: inverted residual,"
                    + " attn.query.proj: attention); a block runs one graph");
        }
        return edge ? Kind.EDGE : attention ? Kind.ATTENTION : inverted ? Kind.INVERTED : null;
    }

    /** The blocks in run order (unmodifiable). */
    List<Block> blocks() {
        return blocks;
    }

    /** The index of every non-empty stage's last block (a copy). */
    int[] stageEnds() {
        return stageEnds.clone();
    }

    /** True when block {@code i} opens a stage, which downsamples by 2. */
    boolean stageStart(int i) {
        if (i == 0) {
            return true;
        }
        for (int end : stageEnds) {
            if (i == end + 1) {
                return true;
            }
        }
        return false;
    }

    /** True when the features after block {@code i} feed the fusion adapter. */
    boolean fusionPoint(int i) {
        if (stageEnds.length >= STAGES) {
            return i == stageEnds[2] || i == stageEnds[3];
        }
        return i == blocks.size() - 1;
    }
}
