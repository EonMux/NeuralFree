package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.DraftHeadOptions;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;

/**
 * A draft source whose tensors are the ones a {@code dflash} drafter derives at bind time under
 * {@link DraftHeadOptions}: the head the drafter would use — its own {@code output.weight}, else the target's,
 * else the tied token embeddings — restated as the derived candidate head of {@code neuralfree.draft.head.vocab},
 * with the synthesized {@code d2t} that maps its rows back to the full vocabulary, the DSpark {@code markov_w2}
 * gathered at those rows, and, at {@code neuralfree.draft.prec}, the drafter's own projection matrices stored at a
 * lower precision. Everything downstream — the spec ({@code nVocabDraft} is the {@code d2t} length), the host
 * weights, the device upload, the memory fit — reads the derived source as it reads a file.
 *
 * <p>The derived head keeps the draft tokenizer's special ids (unknown, BOS, EOS, end-of-turn and -message,
 * separator, padding, class, mask and the FIM tokens) and every extra end-of-generation id a converted model
 * registered under {@code neuralfree.tokenizer.extra_eog_ids} — the think and tool-stop ids among them — then
 * the ids of the frequency ranking in rank order and an ascending fill of the rest, so the {@code d2t} stays
 * ascending and the device readout keeps its fast path. Without a head precision the kept rows are copied verbatim in the head's own storage, which for a
 * GGUF matrix is exact; with one ({@code neuralfree.draft.head.prec}, vLLM's {@code draft_lm_head_quantization})
 * the rows are decoded and stored as Q8_0, FP8 E4M3 with per-row scales, or NVFP4 with E4M3 scales per 16 values
 * and the per-tensor global scale in the {@code <base>.scale} sibling. A head the target lends folds its
 * output scale into the kept rows and refuses a PRISM input Hadamard, which a borrowed row cannot reproduce; a
 * head from the draft keeps its own {@code <base>.scale}, which rides along in the draft file — except under
 * NVFP4, whose rows fold the file's scale in and whose sibling carries only the new global scale.
 *
 * <p>The tensor table is planned from metadata alone; the derived bytes are built when a store or region first
 * reads them, so a memory fit can measure the derived source without encoding anything. The wrapper holds no
 * resources: the underlying source stays with its owner, the stores and regions it hands out close what they
 * opened, and the payloads are byte arrays. Immutable after construction; the payload build is synchronized.
 */
public final class DerivedDraftSource implements ModelSource {

    /**
     * The drafter's own projection matrices a {@code neuralfree.draft.prec} stores at a lower precision: the
     * encoder, the per-block attention and FFN matrices, the DFlash2 convolutions and the selector's hidden
     * projection. The token embeddings, the full-vocabulary selector tables and the head itself are not matrices
     * the stack multiplies with, and an already quantized tensor is not re-quantized.
     */
    private static final Set<String> DRAFTER_MATRICES = new HashSet<>(Arrays.asList(
            TensorNames.weight(TensorNames.FC), TensorNames.weight(TensorNames.DFLASH_SELECTOR_HIDDEN),
            TensorNames.weight(TensorNames.ATTN_Q), TensorNames.weight(TensorNames.ATTN_K),
            TensorNames.weight(TensorNames.ATTN_V), TensorNames.weight(TensorNames.ATTN_OUT),
            TensorNames.weight(TensorNames.FFN_GATE), TensorNames.weight(TensorNames.FFN_UP),
            TensorNames.weight(TensorNames.FFN_DOWN), TensorNames.weight(TensorNames.DFLASH_ATTN_CONV_PROJ),
            TensorNames.weight(TensorNames.DFLASH_FFN_CONV_PROJ)));

    /** GGUF tensor-data alignment the derived offsets follow. */
    private static final long ALIGNMENT = 32;

    /** The property names the refusals name. */
    private static final String PROP_VOCAB = "neuralfree.draft.head.vocab";
    private static final String PROP_HEAD_PREC = "neuralfree.draft.head.prec";

    private final ModelSource draft;
    private final List<GGUFTensorInfo> tensors;
    private final List<Planned> plan;
    private final TreeMap<Long, GGUFTensorInfo> derivedByOffset = new TreeMap<>();
    private volatile Map<Long, byte[]> payloads;

    private DerivedDraftSource(ModelSource draft, List<GGUFTensorInfo> tensors, List<Planned> plan) {
        this.draft = draft;
        this.tensors = tensors;
        this.plan = plan;
        for (Planned p : plan) {
            derivedByOffset.put(p.offset, p.info);
        }
    }

    /**
     * The derived draft source of {@code draft} under {@code options}, or {@code draft} itself when the options
     * derive nothing.
     *
     * @param draft   the draft file's source; the caller keeps its ownership
     * @param target  the target model's host weights, which lend their head when the draft carries none, or null
     *                for a memory fit of a draft that carries its own head; the wrapper reads no tensor bytes
     *                through them at plan time
     * @param options the derived draft head settings
     * @throws IOException              when the draft's metadata cannot be read or the ranking file cannot be read
     * @throws IllegalArgumentException naming the {@code neuralfree.draft.head.*} or {@code neuralfree.draft.prec}
     *                                  property when a setting cannot serve this draft, including a null target
     *                                  for a draft whose head is borrowed
     */
    public static ModelSource wrap(ModelSource draft, CpuWeights target, DraftHeadOptions options)
            throws IOException {
        Objects.requireNonNull(draft, "draft");
        Objects.requireNonNull(options, "options");
        if (!options.enabled()) {
            return draft;
        }
        GgufKv kv = GgufKv.of(draft);
        ModelHparams hp = HparamsLoader.load(kv);
        TensorCatalog catalog = TensorCatalog.of(draft);
        return new Derivation(draft, kv, hp, catalog, target, options).plan();
    }

    // ------------------------------------------------------------------ planning

    /** One derived tensor's bytes, built over the draft's mapped regions when a reader first asks for them. */
    private interface Payload {
        byte[] build(TensorRegions regions) throws IOException;
    }

    /** One planned derived tensor: its table entry and the bytes to build for it. */
    private static final class Planned {
        private final long offset;
        private final GGUFTensorInfo info;
        private final Payload build;

        Planned(long offset, GGUFTensorInfo info, Payload build) {
            this.offset = offset;
            this.info = info;
            this.build = build;
        }
    }

    /**
     * The NVFP4 global scale a restated tensor's build computes for the {@code <base>.scale} sibling build that
     * follows it in the plan: the payloads build once, in plan order, so the sibling reads the value its tensor
     * left.
     */
    private static final class GlobalScale {
        private float value = Float.NaN;

        /** The scale the tensor's encoding build computed. */
        void set(float v) {
            value = v;
        }

        /**
         * The scale.
         *
         * @throws IllegalStateException when the sibling builds before its tensor
         */
        float requireValue() {
            if (Float.isNaN(value)) {
                throw new IllegalStateException("an NVFP4 scale sibling is built before its tensor is encoded");
            }
            return value;
        }
    }

    /** One derivation pass over a draft source: plans the tensor table, then the payloads on demand. */
    private static final class Derivation {
        private final ModelSource draft;
        private final GgufKv kv;
        private final ModelHparams hp;
        private final TensorCatalog catalog;
        private final CpuWeights target;
        private final DraftHeadOptions options;
        private final boolean reduce;
        private final int nVocab;
        private final int nEmbd;

        Derivation(ModelSource draft, GgufKv kv, ModelHparams hp, TensorCatalog catalog, CpuWeights target,
                   DraftHeadOptions options) {
            this.draft = draft;
            this.kv = kv;
            this.hp = hp;
            this.catalog = catalog;
            this.target = target;
            this.options = options;
            this.reduce = options.vocabSize() > 0;
            this.nVocab = hp.nVocab();
            this.nEmbd = hp.nEmbd();
        }

        /** Plans the derived tensor table, or returns the draft itself when nothing is derived. */
        ModelSource plan() throws IOException {
            if (reduce) {
                refuseDsv4();
                refuseFileD2t();
            }
            int[] d2t = reduce ? subset() : null;
            Head head = d2t != null || options.headPrec() != null ? resolve() : null;
            List<NamedPlan> named = new ArrayList<>();
            if (head != null) {
                headPlans(head, d2t, named);
            }
            if (d2t != null) {
                named.add(d2tPlan(d2t));
                NamedPlan markov = markovPlan(d2t);
                if (markov != null) {
                    named.add(markov);
                }
            }
            matricesPlans(named);
            if (named.isEmpty()) {
                return draft;
            }
            return assemble(named);
        }

        private void refuseDsv4() {
            if (hp.dsv4HcMult() > 0) {
                throw new IllegalArgumentException(PROP_VOCAB
                        + " derives a reduced draft head, but a DeepSeek-V4 DSpark draft has no reduced vocabulary"
                        + " in llama.cpp (dflash.cpp:1108-1280)");
            }
        }

        private void refuseFileD2t() {
            if (infoOrNull(TensorNames.D2T) != null) {
                throw new IllegalArgumentException(PROP_VOCAB
                        + " derives a reduced draft head, but the draft already ships a " + TensorNames.D2T
                        + " mapping of its own");
            }
        }

        /** The kept target ids of the derived head. */
        private int[] subset() throws IOException {
            List<Integer> mandatory = new ArrayList<>();
            addMandatory(mandatory, GgufKeys.TOKENIZER_UNK_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_BOS_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_EOS_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_EOT_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_EOM_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_SEP_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_PAD_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_CLS_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_MASK_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_FIM_PRE_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_FIM_SUF_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_FIM_MID_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_FIM_PAD_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_FIM_REP_ID);
            addMandatory(mandatory, GgufKeys.TOKENIZER_FIM_SEP_ID);
            addIds(mandatory, kv.i32Array(GgufKeys.NF_TOKENIZER_EXTRA_EOG_IDS));
            for (int id : options.keepIds()) {
                mandatory.add(id);
            }
            int[] ranked = options.frequencyFile() == null ? null
                    : DraftHeadQuantizer.readRankedIds(Path.of(options.frequencyFile()), nVocab);
            int[] ids = new int[mandatory.size()];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = mandatory.get(i);
            }
            return DraftHeadQuantizer.subset(nVocab, options.vocabSize(), ranked, ids, PROP_VOCAB);
        }

        private void addMandatory(List<Integer> out, String key) {
            int id = kv.i32(key, -1);
            if (id >= 0 && id < nVocab) {
                out.add(id);
            }
        }

        /** Adds the ids of an optional metadata array that lie inside the vocabulary. */
        private void addIds(List<Integer> out, int[] ids) {
            if (ids == null) {
                return;
            }
            for (int id : ids) {
                if (id >= 0 && id < nVocab) {
                    out.add(id);
                }
            }
        }

        /** The head the drafter resolves: the draft's {@code output.weight}, the target's, or the tied embeddings. */
        private Head resolve() {
            GGUFTensorInfo draftHead = draftHeadInfo(TensorNames.weight(TensorNames.OUTPUT));
            if (draftHead != null) {
                return Head.ofDraft(draftHead);
            }
            if (target == null) {
                throw new IllegalArgumentException(PROP_VOCAB
                        + " derives the head the drafter uses, but the draft has no "
                        + TensorNames.weight(TensorNames.OUTPUT) + " and the target's head is not reachable here");
            }
            CpuTensorRef targetHead = target.getOrNull(TensorNames.weight(TensorNames.OUTPUT));
            String name = TensorNames.weight(TensorNames.OUTPUT);
            if (targetHead == null) {
                name = TensorNames.weight(TensorNames.TOKEN_EMBD);
                targetHead = target.getOrNull(name);
            }
            if (targetHead == null) {
                throw new IllegalArgumentException(PROP_VOCAB
                        + " derives the head the drafter uses, but neither the draft nor the target holds "
                        + TensorNames.weight(TensorNames.OUTPUT) + " or "
                        + TensorNames.weight(TensorNames.TOKEN_EMBD));
            }
            return Head.ofTarget(name, targetHead, target.of(targetHead));
        }

        /** The draft's output head, or null when the drafter borrows the target's. */
        private GGUFTensorInfo draftHeadInfo(String name) {
            GGUFTensorInfo info = infoOrNull(name);
            if (info == null) {
                return null;
            }
            if (info.rank() != 2 || info.dim(0) != nEmbd || info.dim(1) < nVocab) {
                throw new IllegalArgumentException(PROP_VOCAB + " derives the head "
                        + info.name() + " [" + info.dim(0) + ", " + info.dim(1) + "], which does not cover the "
                        + nVocab + " tokens of width " + nEmbd);
            }
            return info;
        }

        /**
         * The derived head and, at NVFP4, its {@code <base>.scale} sibling: verbatim kept rows, or decoded rows
         * stored at the configured precision. NVFP4 folds the borrowed head's output scale — or a draft-own
         * head's own {@code <base>.scale} — into the rows, and the sibling carries the new per-tensor global
         * scale, which the runtime readout multiplies the logits by (llama-graph.cpp:1643-1645).
         */
        private void headPlans(Head head, int[] d2t, List<NamedPlan> named) {
            int rows = d2t != null ? d2t.length : head.rows;
            GGMLType type = options.headPrec() == null ? head.type
                    : DraftHeadQuantizer.rowType(options.headPrec(), head.cols);
            DraftHeadOptions.Prec prec = options.headPrec();
            Payload build;
            // NVFP4 restates the rows without the file's scale and recomputes it as the sibling's value;
            // Q8_0 and FP8 keep the rows in the file's scale space, whose sibling rides along unchanged
            GlobalScale global = prec == DraftHeadOptions.Prec.NVFP4 ? new GlobalScale() : null;
            String scaleSibling = global != null ? scaleName(head.name) : null;
            if (prec == null) {
                build = regions -> DraftHeadQuantizer.copyRows(headRef(regions, head), d2t, head.cols);
            } else {
                CpuMatVec.Transforms transforms = head.transforms;
                if (transforms.input() != null) {
                    throw new IllegalArgumentException(PROP_HEAD_PREC + "="
                            + prec.name().toLowerCase(Locale.ROOT) + " derives the head "
                            + head.name + ", which carries a PRISM input Hadamard the derived rows cannot"
                            + " reproduce; keep the head's own precision");
                }
                float scale = transforms.scale(0);
                int cols = head.cols;
                String ownScale = head.info != null ? scaleSibling : null;
                build = regions -> {
                    CpuTensorRef ref = headRef(regions, head);
                    float fold = scale * (ownScale == null ? 1.0f : fileScale(regions, ownScale));
                    DraftHeadQuantizer.Encoded encoded = DraftHeadQuantizer.encode(prec, rows, cols,
                            (r, dst) -> {
                                ref.decodeRows(d2t != null ? d2t[r] : r, 1, dst, 0);
                                if (fold != 1.0f) {
                                    for (int c = 0; c < cols; c++) {
                                        dst[c] *= fold;
                                    }
                                }
                            });
                    if (global != null) {
                        global.set(encoded.outputScale()[0]);
                    }
                    return encoded.bytes();
                };
            }
            named.add(new NamedPlan(head.name, new long[] {head.cols, rows}, type, build));
            if (global != null) {
                named.add(new NamedPlan(scaleSibling, new long[] {1}, GGMLType.F32,
                        regions -> scalarBytes(global.requireValue())));
            }
        }

        /** A read-only view over the head's rows, from the draft's regions or the borrowed target tensor. */
        private static CpuTensorRef headRef(TensorRegions regions, Head head) throws IOException {
            return head.info == null ? head.targetHead : draftRef(regions, head.info);
        }

        /** The synthesized {@code d2t}: the target id of each derived row, an I64 vector. */
        private NamedPlan d2tPlan(int[] d2t) {
            ByteBuffer b = ByteBuffer.allocate(8 * d2t.length).order(ByteOrder.LITTLE_ENDIAN);
            for (int id : d2t) {
                b.putLong(id);
            }
            byte[] bytes = b.array();
            return new NamedPlan(TensorNames.D2T, new long[] {d2t.length}, GGMLType.I64, regions -> bytes);
        }

        /** The DSpark {@code markov_w2} gathered at the derived rows, so row {@code r} reads {@code d2t[r]}. */
        private NamedPlan markovPlan(int[] d2t) {
            GGUFTensorInfo w2 = infoOrNull(TensorNames.weight(TensorNames.DSPARK_MARKOV_W2));
            if (w2 == null) {
                return null;
            }
            if (w2.rank() != 2 || w2.dim(0) > Integer.MAX_VALUE - 8 || w2.dim(1) != nVocab) {
                throw new IllegalArgumentException(PROP_VOCAB + "=" + options.vocabSize()
                        + " derives a reduced draft head, but the draft's " + w2.name() + " has shape "
                        + Arrays.toString(w2.dims()) + ", not a [rank, " + nVocab + "] table over the head rows");
            }
            int cols = (int) w2.dim(0);
            return new NamedPlan(w2.name(), new long[] {cols, d2t.length}, w2.type(),
                    regions -> DraftHeadQuantizer.copyRows(draftRef(regions, w2), d2t, cols));
        }

        /**
         * The drafter's own projection matrices at the configured precision, in file order. An NVFP4 matrix is
         * followed by its {@code <base>.scale} sibling, which carries the matrix's global scale; the rows fold the
         * file's own sibling in, as the head's do.
         */
        private void matricesPlans(List<NamedPlan> named) {
            if (options.matricesPrec() == null) {
                return;
            }
            DraftHeadOptions.Prec prec = options.matricesPrec();
            for (GGUFTensorInfo info : draft.getTensors()) {
                if (!isFloatMatrix(info) || !DRAFTER_MATRICES.contains(info.name())
                        || info.rank() != 2 || info.dim(0) > Integer.MAX_VALUE - 8
                        || info.dim(1) > Integer.MAX_VALUE - 8) {
                    continue;
                }
                int cols = (int) info.dim(0);
                int rows = (int) info.dim(1);
                GGMLType type = DraftHeadQuantizer.rowType(prec, cols);
                String ownScale = prec == DraftHeadOptions.Prec.NVFP4 ? scaleName(info.name()) : null;
                GlobalScale global = prec == DraftHeadOptions.Prec.NVFP4 ? new GlobalScale() : null;
                named.add(new NamedPlan(info.name(), info.dims(), type, regions -> {
                    CpuTensorRef ref = draftRef(regions, info);
                    float fold = ownScale == null ? 1.0f : fileScale(regions, ownScale);
                    DraftHeadQuantizer.Encoded encoded = DraftHeadQuantizer.encode(prec, rows, cols,
                            (r, dst) -> {
                                ref.decodeRows(r, 1, dst, 0);
                                if (fold != 1.0f) {
                                    for (int c = 0; c < cols; c++) {
                                        dst[c] *= fold;
                                    }
                                }
                            });
                    if (global != null) {
                        global.set(encoded.outputScale()[0]);
                    }
                    return encoded.bytes();
                }));
                if (global != null) {
                    named.add(new NamedPlan(ownScale, new long[] {1}, GGMLType.F32,
                            regions -> scalarBytes(global.requireValue())));
                }
            }
        }

        private static boolean isFloatMatrix(GGUFTensorInfo info) {
            return info.type() == GGMLType.F32 || info.type() == GGMLType.F16 || info.type() == GGMLType.BF16;
        }

        /** The {@code <base>.scale} name of a {@code <base>.weight} tensor. */
        private static String scaleName(String weight) {
            String suffix = "." + TensorNames.WEIGHT;
            if (!weight.endsWith(suffix)) {
                throw new IllegalArgumentException(weight + " is no <base>.weight tensor");
            }
            return TensorNames.scale(weight.substring(0, weight.length() - suffix.length()));
        }

        /**
         * The value of the file's {@code scaleName} sibling: the scale an NVFP4 restatement folds into its rows, so
         * the derived sibling carries only the new global scale.
         *
         * @throws IllegalArgumentException when the sibling exists but is not one float value
         */
        private float fileScale(TensorRegions regions, String scaleName) throws IOException {
            GGUFTensorInfo info = infoOrNull(scaleName);
            if (info == null) {
                return 1.0f;
            }
            if (info.rank() != 1 || info.dim(0) != 1 || !isFloatMatrix(info)) {
                throw new IllegalArgumentException("the draft's " + scaleName + " is " + info.type() + " of "
                        + (info.rank() == 1 ? info.dim(0) + " values" : "rank " + info.rank())
                        + ", not one float value; an NVFP4 derived tensor cannot fold it into its rows");
            }
            CpuTensorRef ref = draftRef(regions, info);
            float[] v = new float[1];
            ref.decodeRow(0, v, 0);
            return v[0];
        }

        /** The little-endian F32 bytes of a one-element scale tensor. */
        private static byte[] scalarBytes(float v) {
            return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(v).array();
        }

        /** The info of a draft tensor by its catalog name, or null. */
        private GGUFTensorInfo infoOrNull(String name) {
            com.eonmux.neuralfree.gguf.spec.TensorRef ref = catalog.get(name);
            return ref == null ? null : catalog.info(ref);
        }

        /** A read-only view over one draft tensor. */
        private static CpuTensorRef draftRef(TensorRegions regions, GGUFTensorInfo info) throws IOException {
            long bytes = info.sizeInBytes();
            if (regions.contiguousRunBytes(info) != bytes) {
                throw new IllegalArgumentException("tensor " + info.name() + " of the draft is stored in pieces,"
                        + " which a derived draft tensor cannot copy");
            }
            return CpuTensorRef.of(info.type(), regions.region(info, 0L, bytes), info.dim(1), info.dim(0));
        }

        /**
         * The derived source: every derived tensor at an aligned offset after the file's data, replacements in the
         * file's list positions, a replacement whose name the file does not carry (the borrowed head restated as
         * {@code output.weight}) and the synthesized ones appended, with the payload builds in the same order.
         */
        private ModelSource assemble(List<NamedPlan> named) {
            long next = ALIGNMENT;
            for (GGUFTensorInfo info : draft.getTensors()) {
                next = Math.max(next, align(info.offset() + info.sizeInBytes()));
            }
            List<Planned> planned = new ArrayList<>();
            List<Planned> replaced = new ArrayList<>();
            List<Planned> added = new ArrayList<>();
            for (NamedPlan p : named) {
                GGUFTensorInfo info = new GGUFTensorInfo(p.name, p.dims, p.type, next);
                next = align(next + info.sizeInBytes());
                Planned entry = new Planned(info.offset(), info, p.build);
                planned.add(entry);
                (p.replaces ? replaced : added).add(entry);
            }
            List<GGUFTensorInfo> tensors = new ArrayList<>();
            for (GGUFTensorInfo info : draft.getTensors()) {
                Planned swap = take(replaced, info.name());
                tensors.add(swap != null ? swap.info : info);
            }
            // a plan whose name the file does not carry (a borrowed head restated as output.weight) appends
            tensors.addAll(plannedOf(replaced));
            tensors.addAll(plannedOf(added));
            return new DerivedDraftSource(draft, Collections.unmodifiableList(tensors),
                    Collections.unmodifiableList(planned));
        }

        /** The infos of the planned tensors, in their planned order. */
        private static List<GGUFTensorInfo> plannedOf(List<Planned> entries) {
            List<GGUFTensorInfo> out = new ArrayList<>();
            for (Planned p : entries) {
                out.add(p.info);
            }
            return out;
        }

        /** The derived tensor that replaces {@code name}, removing it from {@code replaced}. */
        private static Planned take(List<Planned> replaced, String name) {
            for (int i = 0; i < replaced.size(); i++) {
                if (replaced.get(i).info.name().equals(name)) {
                    return replaced.remove(i);
                }
            }
            return null;
        }
    }

    /** One planned derived tensor before it takes its offset: name, dims, type and the bytes to build. */
    private static final class NamedPlan {
        private final String name;
        private final long[] dims;
        private final GGMLType type;
        private final Payload build;
        private final boolean replaces;

        NamedPlan(String name, long[] dims, GGMLType type, Payload build) {
            this.name = name;
            this.dims = dims;
            this.type = type;
            this.build = build;
            this.replaces = !name.equals(TensorNames.D2T);
        }
    }

    /** The head the drafter uses, over the draft's own tensor or a borrowed target tensor. */
    private static final class Head {
        private final String name;
        private final int cols;
        private final int rows;
        private final GGMLType type;
        private final GGUFTensorInfo info;
        private final CpuTensorRef targetHead;
        private final CpuMatVec.Transforms transforms;

        private Head(String name, int cols, int rows, GGMLType type, GGUFTensorInfo info, CpuTensorRef targetHead,
                     CpuMatVec.Transforms transforms) {
            this.name = name;
            this.cols = cols;
            this.rows = rows;
            this.type = type;
            this.info = info;
            this.targetHead = targetHead;
            this.transforms = transforms;
        }

        static Head ofDraft(GGUFTensorInfo info) {
            return new Head(info.name(), (int) info.dim(0), (int) info.dim(1), info.type(), info, null,
                    CpuMatVec.Transforms.NONE);
        }

        static Head ofTarget(String name, CpuTensorRef ref, CpuMatVec.Transforms transforms) {
            return new Head(name, (int) ref.cols(), (int) ref.rows(), ref.type(), null, ref, transforms);
        }
    }

    // ------------------------------------------------------------------ payloads

    /** The derived tensors' bytes, built once on the first read. */
    private Map<Long, byte[]> ensurePayloads() throws IOException {
        Map<Long, byte[]> built = payloads;
        if (built != null) {
            return built;
        }
        synchronized (this) {
            if (payloads == null) {
                payloads = buildPayloads();
            }
            return payloads;
        }
    }

    private Map<Long, byte[]> buildPayloads() throws IOException {
        Map<Long, byte[]> out = new LinkedHashMap<>();
        try (TensorRegions regions = draft.openRegions(TensorReadMode.MMAP)) {
            for (Planned p : plan) {
                out.put(p.offset, p.build.build(regions));
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ ModelSource

    @Override
    public ModelFormat format() {
        return draft.format();
    }

    @Override
    public String getFilePath() {
        return draft.getFilePath();
    }

    @Override
    public Map<String, Object> getMetadata() {
        return draft.getMetadata();
    }

    @Override
    public GGUFValueType getValueType(String key) {
        return draft.getValueType(key);
    }

    @Override
    public GGUFValueType getArrayElementType(String key) {
        return draft.getArrayElementType(key);
    }

    @Override
    public String getString(String key) {
        return draft.getString(key);
    }

    @Override
    public List<GGUFTensorInfo> getTensors() {
        return tensors;
    }

    @Override
    public long getTensorCount() {
        return tensors.size();
    }

    @Override
    public WeightStore createMappedStore(StoreUse use) throws IOException {
        return new DerivedStore(draft.createMappedStore(use));
    }

    @Override
    public TensorRegions openRegions(TensorReadMode mode) throws IOException {
        return new DerivedRegions(draft.openRegions(mode));
    }

    /**
     * Closes nothing: the derived payloads are byte arrays and the underlying source stays with its owner, whose
     * close already covers it. The stores and regions this source hands out close what they opened.
     */
    @Override
    public void close() {
    }

    private static long align(long offset) {
        long rem = offset % ALIGNMENT;
        return rem == 0 ? offset : offset + ALIGNMENT - rem;
    }

    /** True when the byte range lies inside a derived tensor. */
    private boolean isDerived(long offset, int size) {
        Map.Entry<Long, GGUFTensorInfo> e = derivedByOffset.floorEntry(offset);
        return e != null && offset >= e.getKey() && offset + size <= e.getKey() + e.getValue().sizeInBytes();
    }

    /** The slice of a derived tensor's payload the byte range covers. */
    private byte[] payload(long offset, int size) throws IOException {
        Map<Long, byte[]> built = ensurePayloads();
        long start = derivedByOffset.floorEntry(offset).getKey();
        return Arrays.copyOfRange(built.get(start), (int) (offset - start), (int) (offset - start) + size);
    }

    /** The store of a derived source: derived tensors from their payloads, the file's from the mapped store. */
    private final class DerivedStore implements WeightStore {
        private final WeightStore store;

        DerivedStore(WeightStore store) {
            this.store = store;
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
            readTensor(tensorOffset, size, dst, 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
            if (isDerived(tensorOffset, size)) {
                System.arraycopy(payload(tensorOffset, size), 0, dst, dstOffset, size);
                return;
            }
            store.readTensor(tensorOffset, size, dst, dstOffset);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
            if (isDerived(tensorOffset, size)) {
                dst.put(payload(tensorOffset, size));
                return;
            }
            store.readTensorDirect(tensorOffset, size, dst);
        }

        @Override
        public void close() throws IOException {
            store.close();
        }
    }

    /** The regions of a derived source: derived tensors from their payloads, the file's from the mapped regions. */
    private final class DerivedRegions implements TensorRegions {
        private final TensorRegions regions;

        DerivedRegions(TensorRegions regions) {
            this.regions = regions;
        }

        @Override
        public ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) throws IOException {
            if (isDerived(info.offset(), (int) info.sizeInBytes())) {
                ByteBuffer b = ByteBuffer.wrap(ensurePayloads().get(info.offset()));
                b.position((int) byteOff);
                b.limit((int) (byteOff + length));
                return b.slice().order(ByteOrder.LITTLE_ENDIAN);
            }
            return regions.region(info, byteOff, length);
        }

        @Override
        public long contiguousRunBytes(GGUFTensorInfo info) {
            return isDerived(info.offset(), (int) info.sizeInBytes()) ? info.sizeInBytes()
                    : regions.contiguousRunBytes(info);
        }

        @Override
        public String description() {
            return regions.description() + " with the derived draft head";
        }

        @Override
        public void close() throws IOException {
            regions.close();
        }
    }
}
