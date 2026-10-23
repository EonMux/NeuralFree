package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader.Batch;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader.Pick;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Rows;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Source;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The Gemma 4 assistant drafter (llama.cpp src/models/gemma4-assistant.cpp), driven like llama.cpp
 * common/speculative.cpp {@code common_speculative_impl_draft_mtp} with shared memory (1432-1876,
 * {@code is_mem_shared}).
 *
 * <p>Graph (gemma4-assistant.cpp:94-204): {@code x = target.token_embd[tok] * sqrt(n_embd_backbone)};
 * {@code cur = nextn.pre_projection . [x | h]}; the assistant blocks (Q-only attention over the target's KV caches
 * with the gemma4 sandwich norms, GELU FFN and {@code layer_output_scale}), run by the layer executor; then
 * {@code output_norm}; logits from the tied {@code token_embd}; {@code h_next = nextn.post_projection . normed}. Each
 * attention layer reads target layer {@code n_layer - 2} when it is a sliding-window layer and {@code n_layer - 1}
 * otherwise (llama-model.cpp:3250-3262, {@link MemoryPlanner#planDrafter}); the drafter writes no cache.
 *
 * <p>Protocol: the target's calls only refresh the verify rows (its post-output-norm hidden state, gemma4.cpp:412-422:
 * tap {@code n_layer} normed on the host) and the pending row; {@code accept(n)} makes verify row
 * {@code min(n, rows - 1)} pending. A draft at {@code P = n_past} runs every step at position {@code P}
 * (speculative.cpp:1824-1829): first {@code (lastToken, pending)}, then each drafted token with the previous step's
 * {@code h_next}. The target's caches hold positions below {@code P} then, so the queries see exactly the verified
 * context.
 */
public final class Gemma4AssistantDraftModel implements DraftModel {

    private final MtpHead head;
    private final DraftModelLoader.Options options;
    private final ModelSpec draft;
    private final ModelSpec target;
    private final int nEmbd;
    private final int nBackbone;
    private final int vocab;
    private final float eps;
    private final float embdScale;
    private final String targetEmbd;
    private final String preProj;
    private final String postProj;
    private final String outputNorm;
    private final String headWeight;
    private final String targetOutputNorm;
    private final List<LayerSpec> blocks;

    private ForwardBackend backend;
    private SequenceState targetSequence;
    private DraftModelLoader.TargetFeed feed;
    private MtpHead.Memory memory;
    private MtpHead.Stack stack;
    private float[] targetNorm;
    private Rows tokEmbd;
    private Rows hIn;
    private Rows xh;
    private Rows x;
    private Rows xn;
    private Rows hNext;
    private float[] logits;

    private DraftModelLoader.HiddenRows hidden;

    /**
     * @throws IllegalArgumentException when the drafter is not a gemma4-assistant model or does not fit the target
     *                                  (backbone width, vocabulary, tensors)
     */
    Gemma4AssistantDraftModel(MtpHead head, ModelSpec draft, DraftModelLoader.Options options) {
        this.head = Objects.requireNonNull(head, "head");
        this.draft = Objects.requireNonNull(draft, "draft");
        this.options = Objects.requireNonNull(options, "options");
        this.target = head.target();
        if (draft.arch() != Arch.GEMMA4_ASSISTANT) {
            throw new IllegalArgumentException(draft.arch().ggufName() + " is not a gemma4-assistant drafter");
        }
        String name = draft.arch().ggufName();
        this.nEmbd = draft.hparams().nEmbd();
        this.nBackbone = draft.hparams().nEmbdInp();
        if (target.hparams().nEmbd() != nBackbone) {
            throw new IllegalArgumentException(name + " reads a backbone of " + nBackbone + " floats; the target "
                    + target.arch().ggufName() + " has n_embd " + target.hparams().nEmbd());
        }
        List<LayerSpec> all = new ArrayList<>(draft.layers());
        all.addAll(draft.nextnLayers());
        if (all.isEmpty()) {
            throw new IllegalArgumentException(name + " has no assistant blocks");
        }
        this.blocks = Collections.unmodifiableList(all);
        this.eps = draft.hparams().normRmsEps();
        this.embdScale = (float) Math.sqrt(nBackbone);
        this.targetEmbd = requiredName(target.embedding().tokenEmbd(), Source.TARGET, "the target token_embd");
        this.preProj = required(TensorNames.weight(TensorNames.NEXTN_PROJ_PRE), Source.DRAFT);
        this.postProj = required(TensorNames.weight(TensorNames.NEXTN_PROJ_POST), Source.DRAFT);
        this.outputNorm = requiredName(draft.head().outputNorm(), Source.DRAFT, "output_norm");
        this.headWeight = requiredName(draft.head().output(), Source.DRAFT, "the tied token_embd head");
        this.targetOutputNorm = requiredName(target.head().outputNorm(), Source.TARGET, "the target output_norm");
        checkShape(preProj, 2L * nBackbone, nEmbd);
        checkShape(postProj, nEmbd, nBackbone);
        long[] headShape = head.shape(Source.DRAFT, headWeight);
        long targetVocab = target.head().vocabRows();
        if (headShape[1] != targetVocab) {
            throw new IllegalArgumentException(name + " predicts " + headShape[1] + " tokens; the target vocabulary"
                    + " has " + targetVocab);
        }
        this.vocab = Math.toIntExact(headShape[1]);
    }

    private String required(String tensor, Source s) {
        if (!head.has(s, tensor)) {
            throw new IllegalArgumentException(draft.arch().ggufName() + " needs " + tensor);
        }
        return tensor;
    }

    private String requiredName(TensorRef ref, Source s, String what) {
        if (ref == null || !head.has(s, ref.name())) {
            throw new IllegalArgumentException(draft.arch().ggufName() + " needs " + what);
        }
        return ref.name();
    }

    private void checkShape(String tensor, long cols, long rows) {
        long[] ne = head.shape(Source.DRAFT, tensor);
        if (ne.length < 2 || ne[0] != cols || ne[1] != rows) {
            throw new IllegalArgumentException(tensor + " has shape " + Arrays.toString(ne) + ", expected [" + cols
                    + ", " + rows + "]");
        }
    }

    @Override
    public void bindTarget(EngineModel model) {
        Objects.requireNonNull(model, "target");
        if (backend != null) {
            throw new IllegalStateException("the gemma4-assistant drafter is already bound");
        }
        ForwardBackend fb = model.forwardBackend();
        ModelMemory tm = fb.memory();
        if (tm == null) {
            throw new IllegalArgumentException(target.arch().ggufName() + ": the target runs without memory");
        }
        MemoryPlan plan = MemoryPlanner.planDrafter(draft, target, tm.plan());
        this.memory = head.targetCaches(draft, plan);
        this.stack = head.stack(draft, Source.DRAFT, memory, readingTarget(blocks, plan));
        this.targetNorm = head.vector(Source.TARGET, targetOutputNorm);
        if (targetNorm.length != nBackbone) {
            throw new IllegalArgumentException(targetOutputNorm + " has " + targetNorm.length + " values, n_embd is "
                    + nBackbone);
        }
        this.tokEmbd = head.rows("g4a.embd", nBackbone);
        this.hIn = head.rows("g4a.h", nBackbone);
        this.xh = head.rows("g4a.xh", 2 * nBackbone);
        this.x = head.rows("g4a.x", nEmbd);
        this.xn = head.rows("g4a.xn", nEmbd);
        this.hNext = head.rows("g4a.h_next", nBackbone);
        this.logits = new float[vocab];
        this.hidden = new DraftModelLoader.HiddenRows(nBackbone);
        this.targetSequence = tm.sequence();
        // a verify call spans at most one memory ubatch, so keeping that many rows per call keeps every verify row
        int keep = Math.max(head.ubatch(), tm.plan().ubatch());
        this.feed = new DraftModelLoader.TargetFeed(new int[]{target.nLayer()}, nBackbone, tm, keep);
        fb.setHiddenTap(feed);
        this.backend = fb;
    }

    /**
     * The blocks with their attention marked as reading another layer's cache (the target layer the plan names), so
     * the mixers project Q only (gemma4-assistant.cpp:146-157 has no K/V projection).
     *
     * @throws IllegalArgumentException when a block's index equals the target layer it reads, which the attention
     *                                  spec cannot tell from owning a cache
     */
    static List<LayerSpec> readingTarget(List<LayerSpec> layers, MemoryPlan plan) {
        List<LayerSpec> out = new ArrayList<>(layers.size());
        for (LayerSpec l : layers) {
            LayerMemoryPlan lp = plan.layer(l.il());
            AttentionSpec a = l.attention();
            if (a == null || lp.sharedFrom() == null || !lp.sharedFrom().target()) {
                out.add(l);
                continue;
            }
            int owner = lp.sharedFrom().layer();
            if (owner == l.il()) {
                throw new IllegalArgumentException("assistant block " + l.il() + " reads target layer " + owner
                        + " of the same index");
            }
            out.add(copy(l, a.toBuilder().kvOwnerLayer(owner).build()));
        }
        return Collections.unmodifiableList(out);
    }

    private static LayerSpec copy(LayerSpec l, AttentionSpec attention) {
        LayerSpec.Builder b = LayerSpec.builder(l.il(), l.mixerKind(), l.ffnKind())
                .weightLayer(l.weightLayer())
                .stack(l.stack())
                .norm(l.norm())
                .residual(l.residual())
                .attention(attention)
                .crossAttention(l.crossAttention())
                .parallelAttention(l.parallelAttention())
                .mla(l.mla())
                .dsa(l.dsa())
                .msa(l.msa())
                .dsv4(l.dsv4())
                .recurrent(l.recurrent())
                .ffn(l.ffn())
                .moe(l.moe())
                .rope(l.rope());
        for (String e : l.extras()) {
            b.extra(e);
        }
        return b.build();
    }

    @Override
    public int[] draft(int lastToken, int n) {
        requireBound();
        if (n < 0) {
            throw new IllegalArgumentException("draft length " + n + " is negative");
        }
        sync();
        int pos = targetSequence.nPast();
        if (n == 0 || pos >= targetSequence.nCtx()) {
            return new int[0];
        }
        float[] h = hidden.pending().clone();
        int[] out = new int[n];
        int made = 0;
        int[] one = new int[]{lastToken};
        for (int i = 0; i < n; i++) {
            head.embed(Source.TARGET, targetEmbd, one, 1, tokEmbd);
            head.scale(tokEmbd, embdScale, 1);
            head.write(hIn, h, 1);
            head.concat(tokEmbd, hIn, xh, 1);
            head.project(Source.DRAFT, preProj, xh, x, 1);
            stack.run(pos, 1, one, x);
            head.norm(NormKind.RMS, Source.DRAFT, outputNorm, eps, x, xn, 1);
            head.logits(Source.DRAFT, headWeight, xn, 0, logits);
            Pick p = DraftModelLoader.pick(logits, vocab);
            if (p.p < options.pMin()) {
                break;
            }
            out[made++] = p.id;
            one[0] = p.id;
            if (i + 1 < n) {
                head.project(Source.DRAFT, postProj, xn, hNext, 1);
                head.readRow(hNext, 0, h);
            }
        }
        return Arrays.copyOf(out, made);
    }

    @Override
    public void accept(int nAccepted) {
        requireBound();
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count " + nAccepted + " is negative");
        }
        sync();
        hidden.accept(nAccepted);
    }

    @Override
    public void reset() {
        if (backend == null) {
            return;
        }
        feed.clear();
        hidden.clear();
    }

    @Override
    public void close() {
        try {
            if (backend != null) {
                backend.setHiddenTap(HiddenTap.NONE);
            }
        } finally {
            backend = null;
            head.close();
        }
    }

    private void requireBound() {
        if (backend == null) {
            throw new IllegalStateException("the gemma4-assistant drafter is not bound to a target");
        }
    }

    /** Takes the hidden rows of every target call since the last sync; the last call's rows become the verify rows. */
    private void sync() {
        for (Batch b : feed.drain()) {
            int m = b.size();
            float[] h = b.rows[0];
            CpuOps.norm(NormKind.RMS, target.hparams().normRmsEps(), h, 0, h, 0, m, nBackbone, targetNorm, null);
            hidden.record(b.pos0, h, m);
        }
    }

    @Override
    public String toString() {
        return "Gemma4AssistantDraftModel{" + blocks.size() + " blocks, n_embd " + nEmbd + ", backbone " + nBackbone
                + '}';
    }
}
