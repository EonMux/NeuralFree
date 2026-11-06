package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.gpu.forward.GpuRecurrentTreeRound;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;

/**
 * A recurrent GPU mixer whose single-pass tree step the tree verify serves: the mixer runs the verify step's rows
 * one at a time against per-row stack state ({@link #treeStep}), its stateful dispatches re-pointed at the round's
 * stack buffers, never touching the committed state, and commits by replaying rows ({@code GpuMixer#decode} from
 * the round's captured input rows). Covered today: the gated DeltaNet, KDA and the Mamba-family mixers (Mamba-1,
 * Mamba-2, PLaMo-2); a mixer of another recurrent kind refuses by name through {@link #treeStepRefusal()}.
 */
public interface GpuRecurrentTreeMixer {

    /**
     * Why this mixer has no single-pass tree step, or null when {@link #treeStep} runs. A mixer that serves several
     * recurrent kinds answers per kind, so the backend's probe refuses the uncovered ones by name.
     */
    String treeStepRefusal();

    /** The row stride of {@link #treeStep}'s input and output rows: the layer's mixer-input width. */
    int treeRowWidth();

    /**
     * The masked verify step of one tree round: the rows of the round in ascending order — row 0 (depth 0, the
     * pending token) first, then every proposal node — each run as one token against the stack slot the round
     * seeds from the node's parent's slot ({@code in} holds the step's input rows at {@link #treeRowWidth()},
     * {@code out} receives one branch row per row). The committed state stays untouched.
     *
     * @throws UnsupportedOperationException per {@link #treeStepRefusal()}
     */
    void treeStep(GpuRecurrentTreeRound round, ScratchSlot in, ScratchSlot out);
}
