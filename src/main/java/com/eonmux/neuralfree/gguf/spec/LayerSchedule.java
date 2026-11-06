package com.eonmux.neuralfree.gguf.spec;

/**
 * Execution order of the trunk: {@link #slots()} virtual layers ("slots"), each with its own KV / state slot,
 * mapped onto physical weight layers ({@code blk.N} tensors). Immutable.
 *
 * <ul>
 *   <li>{@link Kind#LINEAR}: slot {@code s} uses weights {@code s}.</li>
 *   <li>{@link Kind#LOOPED} (nanbeige, models/nanbeige.cpp:66-71, 164-171): {@code nLoops} passes over
 *       {@code nPhysical} layers; slot {@code s} uses weights {@code s % nPhysical}. After slot {@code s} the
 *       stream is replaced by {@code rms(x) * output_norm} when {@link #normAfterSlot(int)}: {@code nLoops > 1},
 *       {@code (s + 1) % nPhysical == 0}, {@code s + 1 < slots()} and the GGUF does not set
 *       {@code skip_loop_final_norm}.</li>
 *   <li>{@link Kind#HRM} (hrm_text, models/hrm-text.cpp): {@code H} high cycles of {@code L} low-stack passes and
 *       one high-stack pass, {@code lps} layers per pass, {@code slots = lps * H * (L + 1)}. Pass {@code p} covers
 *       slots {@code [p * lps, (p + 1) * lps)}; it is a high pass when {@code p % (L + 1) == L}. Low passes use
 *       weights {@code [0, lps)}, high passes {@code [lps, 2 * lps)}. Streams start as {@code zH = embeddings},
 *       {@code zL = hrm.z_l_init}; a pass reads {@code zH + zL} at its first slot, ends with a weightless RMS norm
 *       after its last slot and writes {@code zL} (low) or {@code zH} (high). The trunk output is {@code zH}.</li>
 *   <li>{@link Kind#REPEAT} (Dory, IQuestLoopCoder): the trunk is {@code nInput} plain layers, then the block
 *       {@code [nInput, nInput + nRecurrent)} run {@code nLoops} times, then {@code nOutput} plain layers. Every
 *       pass of the repeated block is a slot with its own KV cache, so iteration {@code i} of block layer {@code j}
 *       is slot {@code nInput + i * nRecurrent + j}. {@link #loopCombine()} optionally normalizes the stream at the
 *       pass boundaries: a pass starts from {@code justNorm(state + input)} and ends by writing
 *       {@code state = justNorm(x)} (Dory); {@link LoopCombine#NONE} carries the stream unchanged (IQuestLoopCoder).
 *       </li>
 * </ul>
 */
public final class LayerSchedule {

    /** Schedule shape. */
    public enum Kind { LINEAR, LOOPED, HRM, REPEAT }

    /** Residual stream a hierarchical pass writes. */
    public enum Stream { LOW, HIGH }

    /** Stream combination at the boundaries of a {@link Kind#REPEAT} pass. */
    public enum LoopCombine {
        /** Carry the stream unchanged into the next pass. */
        NONE,
        /** A pass starts from {@code justNorm(state + input)}; its result becomes {@code state = justNorm(x)}. */
        JUSTNORM
    }

    private final Kind kind;
    private final int slots;
    private final int nPhysical;
    private final int nLoops;
    private final boolean loopNorm;
    private final int layersPerStack;
    private final int hCycles;
    private final int lCycles;
    private final int nInput;
    private final int nRecurrent;
    private final int nOutput;
    private final LoopCombine loopCombine;
    private final float loopEps;

    private LayerSchedule(Kind kind, int slots, int nPhysical, int nLoops, boolean loopNorm, int layersPerStack,
                          int hCycles, int lCycles, int nInput, int nRecurrent, int nOutput, LoopCombine loopCombine,
                          float loopEps) {
        this.kind = kind;
        this.slots = slots;
        this.nPhysical = nPhysical;
        this.nLoops = nLoops;
        this.loopNorm = loopNorm;
        this.layersPerStack = layersPerStack;
        this.hCycles = hCycles;
        this.lCycles = lCycles;
        this.nInput = nInput;
        this.nRecurrent = nRecurrent;
        this.nOutput = nOutput;
        this.loopCombine = loopCombine;
        this.loopEps = loopEps;
    }

    /** One slot per physical layer. */
    public static LayerSchedule linear(int nLayer) {
        if (nLayer <= 0) {
            throw new IllegalArgumentException("layer count must be positive, got " + nLayer);
        }
        return new LayerSchedule(Kind.LINEAR, nLayer, nLayer, 1, false, 0, 0, 0, 0, 0, 0, LoopCombine.NONE, 0f);
    }

    /**
     * No trunk slots: a model whose every block is a NextN block, so llama.cpp's {@code n_layer()} is 0
     * (gemma4-assistant writes {@code nextn_predict_layers = block_count}, conversion/gemma.py:974, and builds its
     * blocks {@code [0, n_layer_nextn)}, models/gemma4-assistant.cpp:47, 133).
     */
    public static LayerSchedule empty() {
        return new LayerSchedule(Kind.LINEAR, 0, 0, 1, false, 0, 0, 0, 0, 0, 0, LoopCombine.NONE, 0f);
    }

    /**
     * nanbeige loops.
     *
     * @param nPhysical          physical layers
     * @param nLoops             passes ({@code num_loops}), at least 1
     * @param skipLoopFinalNorm  {@code skip_loop_final_norm}
     */
    public static LayerSchedule looped(int nPhysical, int nLoops, boolean skipLoopFinalNorm) {
        if (nPhysical <= 0 || nLoops < 1) {
            throw new IllegalArgumentException("looped schedule needs nPhysical > 0 and nLoops >= 1, got "
                    + nPhysical + "/" + nLoops);
        }
        long slots = (long) nPhysical * nLoops;
        if (slots > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("looped schedule has too many slots: " + slots);
        }
        return new LayerSchedule(Kind.LOOPED, (int) slots, nPhysical, nLoops, nLoops > 1 && !skipLoopFinalNorm, 0, 0, 0,
                0, 0, 0, LoopCombine.NONE, 0f);
    }

    /**
     * hrm_text stacks.
     *
     * @param layersPerStack {@code hrm.layers_per_stack}
     * @param hCycles        {@code hrm.h_cycles}
     * @param lCycles        {@code hrm.l_cycles}
     */
    public static LayerSchedule hrm(int layersPerStack, int hCycles, int lCycles) {
        if (layersPerStack <= 0 || hCycles <= 0 || lCycles <= 0) {
            throw new IllegalArgumentException("HRM schedule needs positive layers per stack and cycles, got "
                    + layersPerStack + "/" + hCycles + "/" + lCycles);
        }
        long slots = (long) layersPerStack * hCycles * (lCycles + 1L);
        if (slots > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("HRM schedule has too many slots: " + slots);
        }
        return new LayerSchedule(Kind.HRM, (int) slots, 2 * layersPerStack, 0, false, layersPerStack, hCycles, lCycles,
                0, 0, 0, LoopCombine.NONE, 0f);
    }

    /**
     * A trunk that runs {@code nInput} plain layers, then the block of {@code nRecurrent} layers {@code nLoops}
     * times, then {@code nOutput} plain layers. Every pass of the repeated block is a slot with its own KV cache.
     *
     * @param nPhysical  physical weight layers, equal to {@code nInput + nRecurrent + nOutput}
     * @param nInput     layers before the repeated block
     * @param nRecurrent layers in the repeated block
     * @param nLoops     passes over the repeated block, at least 1
     * @param nOutput    layers after the repeated block
     * @param combine    stream combination at the pass boundaries
     * @param eps        justnorm epsilon of {@link LoopCombine#JUSTNORM}; ignored for {@link LoopCombine#NONE}
     */
    public static LayerSchedule repeat(int nPhysical, int nInput, int nRecurrent, int nLoops, int nOutput,
                                       LoopCombine combine, float eps) {
        if (nInput < 0 || nRecurrent <= 0 || nOutput < 0 || nLoops < 1) {
            throw new IllegalArgumentException("repeat schedule needs nInput >= 0, nRecurrent > 0, nOutput >= 0 and"
                    + " nLoops >= 1, got " + nInput + "/" + nRecurrent + "/" + nOutput + "/" + nLoops);
        }
        if ((long) nInput + nRecurrent + nOutput != nPhysical) {
            throw new IllegalArgumentException("repeat schedule ranges " + nInput + " + " + nRecurrent + " + "
                    + nOutput + " do not cover the " + nPhysical + " physical layers");
        }
        if (combine == LoopCombine.JUSTNORM && (!(eps > 0f) || Float.isInfinite(eps))) {
            throw new IllegalArgumentException("justnorm loop combination needs a positive finite eps, got " + eps);
        }
        long slots = (long) nInput + (long) nRecurrent * nLoops + nOutput;
        if (slots > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("repeat schedule has too many slots: " + slots);
        }
        return new LayerSchedule(Kind.REPEAT, (int) slots, nPhysical, nLoops, false, 0, 0, 0, nInput, nRecurrent,
                nOutput, combine, combine == LoopCombine.JUSTNORM ? eps : 0f);
    }

    public Kind kind() { return kind; }
    /** Number of virtual layers (KV / state slots). */
    public int slots() { return slots; }
    /** Number of physical weight layers. */
    public int nPhysical() { return nPhysical; }
    /** Loop passes (LOOPED / REPEAT); 1 for LINEAR, 0 for HRM. */
    public int nLoops() { return nLoops; }
    public int layersPerStack() { return layersPerStack; }
    public int hCycles() { return hCycles; }
    public int lCycles() { return lCycles; }
    /** REPEAT: plain layers before the repeated block. */
    public int nInputLayers() { return nInput; }
    /** REPEAT: layers of the repeated block. */
    public int nRecurrentLayers() { return nRecurrent; }
    /** REPEAT: plain layers after the repeated block. */
    public int nOutputLayers() { return nOutput; }
    /** REPEAT: stream combination at the pass boundaries. */
    public LoopCombine loopCombine() { return loopCombine; }
    /** REPEAT: justnorm epsilon of {@link LoopCombine#JUSTNORM}; 0 otherwise. */
    public float loopEps() { return loopEps; }
    /** First slot after the input layers ({@code nInput}). */
    public int recurrentStart() { return nInput; }
    /** First slot after the repeated block ({@code nInput + nRecurrent * nLoops}). */
    public int recurrentEnd() { return nInput + nRecurrent * nLoops; }

    /**
     * Pass index of {@code slot}: {@code slot / nPhysical} for LOOPED, {@code (slot - nInput) / nRecurrent} inside
     * the repeated block of REPEAT, -1 for the plain layers of a REPEAT schedule and every slot of the other kinds.
     */
    public int loopIndex(int slot) {
        check(slot);
        switch (kind) {
            case LOOPED:
                return slot / nPhysical;
            case REPEAT:
                return slot < nInput || slot >= recurrentEnd() ? -1 : (slot - nInput) / nRecurrent;
            default:
                return -1;
        }
    }

    /** REPEAT: this slot opens a pass of the repeated block (a pass reads the carried state here). */
    public boolean loopStart(int slot) {
        check(slot);
        return kind == Kind.REPEAT && slot >= nInput && slot < recurrentEnd() && (slot - nInput) % nRecurrent == 0;
    }

    /** REPEAT: this slot closes a pass of the repeated block (its result becomes the carried state). */
    public boolean loopEnd(int slot) {
        check(slot);
        return kind == Kind.REPEAT && slot >= nInput && slot < recurrentEnd()
                && (slot - nInput + 1) % nRecurrent == 0;
    }

    /** Physical layer whose tensors slot {@code slot} uses. */
    public int weightLayer(int slot) {
        check(slot);
        switch (kind) {
            case LOOPED:
                return slot % nPhysical;
            case REPEAT:
                if (slot < nInput) {
                    return slot;
                }
                if (slot < recurrentEnd()) {
                    return nInput + (slot - nInput) % nRecurrent;
                }
                return nInput + nRecurrent + (slot - recurrentEnd());
            case HRM:
                return stream(slot) == Stream.HIGH ? layersPerStack + slot % layersPerStack : slot % layersPerStack;
            default:
                return slot;
        }
    }

    /** KV / recurrent-state slot of a virtual layer: every slot owns its own memory. */
    public int memorySlot(int slot) {
        check(slot);
        return slot;
    }

    /** True when {@code rms(x) * output_norm} replaces the stream after this slot (nanbeige loop boundary). */
    public boolean normAfterSlot(int slot) {
        check(slot);
        return kind == Kind.LOOPED && loopNorm && (slot + 1) % nPhysical == 0 && slot + 1 < slots;
    }

    /** HRM: this slot starts a pass and reads {@code zH + zL}. */
    public boolean passStart(int slot) {
        check(slot);
        return kind == Kind.HRM && slot % layersPerStack == 0;
    }

    /** HRM: this slot ends a pass; a weightless RMS norm follows and the result is written to {@link #stream(int)}. */
    public boolean passEnd(int slot) {
        check(slot);
        return kind == Kind.HRM && (slot + 1) % layersPerStack == 0;
    }

    /**
     * HRM: the stream the pass containing {@code slot} writes.
     *
     * @throws IllegalStateException for non-HRM schedules
     */
    public Stream stream(int slot) {
        check(slot);
        if (kind != Kind.HRM) {
            throw new IllegalStateException(kind + " schedules have a single residual stream");
        }
        int pass = slot / layersPerStack;
        return pass % (lCycles + 1) == lCycles ? Stream.HIGH : Stream.LOW;
    }

    private void check(int slot) {
        if (slot < 0 || slot >= slots) {
            throw new IndexOutOfBoundsException("slot " + slot + " outside [0, " + slots + ")");
        }
    }

    @Override
    public String toString() {
        switch (kind) {
            case LOOPED:
                return "LayerSchedule{LOOPED " + nPhysical + " x " + nLoops + (loopNorm ? ", loop norm" : "") + '}';
            case HRM:
                return "LayerSchedule{HRM lps=" + layersPerStack + ", H=" + hCycles + ", L=" + lCycles + ", slots=" + slots + '}';
            case REPEAT:
                return "LayerSchedule{REPEAT " + nInput + " + " + nRecurrent + " x " + nLoops + " + " + nOutput
                        + (loopCombine == LoopCombine.JUSTNORM ? ", justnorm " + loopEps : "") + ", slots=" + slots + '}';
            default:
                return "LayerSchedule{LINEAR " + slots + '}';
        }
    }
}
