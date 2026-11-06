package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.batch.TreeBudgetPolicy;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.SuffixDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.SyntheticVerifier;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.function.IntConsumer;
import java.util.function.Supplier;

/**
 * The drafter and verifier settings of speculative decoding beside the draft file, the draft lengths and the draft
 * types ({@link EngineConfig#draftTypes()}): the sizes of each n-gram drafter type, the lookup cache files of
 * {@code ngram-cache}, the suffix decoding limits and how proposals are verified. Immutable; built with
 * {@link #builder()} or read with {@link #fromProperties(Properties)}.
 *
 * <p>Properties (absent keys keep the defaults, the {@link NgramDrafter} and {@link SuffixDrafter} defaults):
 * <ul>
 *   <li>{@code neuralfree.draft.ngram.simple.size.n} / {@code .size.m} (llama.cpp {@code --spec-ngram-simple-size-n}
 *       / {@code -size-m}, vLLM {@code prompt_lookup_max}): the lookup n-gram and the draft m-gram of
 *       {@code ngram-simple}, llama.cpp's defaults 12 and 48. {@code .size.n.min} (vLLM {@code prompt_lookup_min})
 *       below {@code .size.n} searches vLLM's size range instead of llama.cpp's single size; unset, it equals
 *       {@code .size.n}.</li>
 *   <li>{@code neuralfree.draft.ngram.map.k.size.n} / {@code .size.m} / {@code .min.hits} and the same keys under
 *       {@code neuralfree.draft.ngram.map.k4v} (llama.cpp {@code --spec-ngram-map-k[4v]-size-n} / {@code -size-m} /
 *       {@code -min-hits}): the key n-gram, the value m-gram and the key hits before a proposal, default 12, 48,
 *       1.</li>
 *   <li>{@code neuralfree.draft.ngram.mod.n.match} / {@code .n.min} / {@code .n.max} (llama.cpp
 *       {@code --spec-ngram-mod-n-match} / {@code -n-min} / {@code -n-max}): the hashed n-gram size, the shortest
 *       draft kept and the longest drafted of {@code ngram-mod}, default 24, 48, 64.</li>
 *   <li>{@code neuralfree.draft.lookup.cache.static} / {@code .dynamic} (llama.cpp {@code --lookup-cache-static} /
 *       {@code --lookup-cache-dynamic}): the {@code ngram-cache} files in the {@code common_ngram_cache_save} layout;
 *       the static one is read only, the dynamic one is read when it exists and written when the session closes,
 *       with the n-grams of every generation added.</li>
 *   <li>{@code neuralfree.draft.suffix.max.tree.depth}, {@code .max.spec.factor}, {@code .max.spec.offset},
 *       {@code .min.token.prob}, {@code .max.cached.requests} (vLLM {@code suffix_decoding_*}): the {@code suffix}
 *       drafter's limits, default 64, 1.0, 0.0, 0.1, and the past requests its request tree keeps, default 10000
 *       (0 turns the request tree off).</li>
 *   <li>{@code neuralfree.draft.verify} (vLLM {@code rejection_sample_method}, {@link VerifyMode}): {@code match}
 *       (default, llama.cpp's target-sample match), {@code rejection} (llama.cpp
 *       {@code --spec-draft-sampling probabilistic}), {@code block}, {@code fly} or {@code synthetic}.</li>
 *   <li>{@code neuralfree.draft.fly.window}, {@code .fly.entropy.threshold}, {@code .fly.entropy.top.k} (vLLM
 *       {@code fly_window_size}, {@code fly_entropy_threshold}, {@code fly_entropy_top_k}): the {@code fly} mode's
 *       window (0 = {@code min(6, k - 1)}), entropy gate and entropy top-k, default 0, 0.3, 3.</li>
 *   <li>{@code neuralfree.draft.synth.len} or {@code neuralfree.draft.synth.rates} (llama.cpp
 *       {@code --spec-synth-len} / {@code --spec-synth-rates}, vLLM {@code synthetic_acceptance_length} /
 *       {@code _rates}): the benchmarking acceptance of {@code synthetic} verification, which either selects when
 *       {@code neuralfree.draft.verify} is unset; it runs only with {@code neuralfree.spec.synthetic=true}.</li>
 *   <li>{@code neuralfree.draft.verify.device} (llama.cpp {@code -bs} / {@code --backend-sampling}, on by default as
 *       Avalon runs it on NVIDIA): {@code match} verification of an argmax chain selects every verified row's token
 *       on the target's device and reads back only the ids instead of {@code k + 1} logits rows, default
 *       {@code true}.</li>
 *   <li>{@code neuralfree.draft.dflash.device} (llama.cpp {@code --spec-draft-backend-sampling}, default on there
 *       too): a {@code dflash} drafter of a GPU target drafts on the target's device — its encoder, decoder stack,
 *       KV cache and a readout that reads back only what it drafts (greedy ids, the DSpark chain, the DFlash2
 *       selector's ids; the top candidates under a {@code p_min}) — default {@code true}. Drafts the device stack
 *       does not run (the DeepSeek-V4 DSpark backbone, a quantized draft cache) and layer-split targets draft on
 *       the host, as with {@code false}.</li>
 *   <li>{@code neuralfree.draft.sample.method} (vLLM {@code draft_sample_method}): how a draft model picks its
 *       tokens, {@code greedy} (default: the argmax, a one-hot proposal) or {@code probabilistic} (a draw from the
 *       draft distribution at the request's temperature, whose probabilities rejection sampling then reads). The
 *       batch engine's draft model runs it; the single-sequence drafters draft greedily and refuse
 *       {@code probabilistic} by name.</li>
 *   <li>{@code neuralfree.draft.heterogeneous.vocab} (vLLM {@code use_heterogeneous_vocab}): a separate draft model
 *       whose vocabulary differs from the target's drafts across the token-level intersection (TLI) instead of
 *       requiring shared token ids. Greedy drafting only, as vLLM restricts it. Default {@code false}.</li>
 *   <li>{@code neuralfree.draft.p.split} (llama.cpp {@code --spec-draft-p-split}): the probability above which a
 *       secondary draft candidate splits a branch into a new one when the draft model speculates over a paged
 *       memory with several sequences ({@link ModelDrafter}), default 0.1 as in llama.cpp's speculative
 *       example.</li>
 *   <li>{@code neuralfree.draft.tree}, {@code neuralfree.draft.tree.temp} and {@code neuralfree.draft.tree.nodes}
 *       (at most {@link #MAX_TREE_NODES}): a {@code dflash} drafter proposes a draft tree (DFlash2 selector
 *       lattice, DSpark Markov-conditioned candidates, or plain DFlash top-k) that the target verifies in one
 *       forward, the temperature its builder scales the path-conditional scores at, and the fixed verify width in
 *       tree nodes, default {@code false}, 1.0 and 16 (SpecPilot's {@code TREE_MODE=ddtree}, arXiv 2611.02501).</li>
 *   <li>{@code neuralfree.draft.tlv} and {@code neuralfree.draft.tlv.bits} (SpecPilot's two-level verification,
 *       arXiv 2611.02501): the target loads a quantized copy of its own projection matrices (round-to-nearest at
 *       the stored precision it undercuts; 6 bits as {@code Q6_K}, 8 as {@code Q8_0}, default 6), the draft tree
 *       is verified once on that copy with plain argmax and nothing committed, the copy's accepted chain and bonus
 *       token anchor a second, narrower draft, and the exact target verifies chain and new tree in one forward and
 *       commits. Only the exact target decides a token, so the output is the target-only greedy output;
 *       {@code bits} counts the copy into the context fit and the device placement, and a target whose weights the
 *       copy does not undercut (already 8-bit or lower) is refused by name. Needs the draft tree, greedy decoding
 *       and a GGUF target. No llama.cpp or vLLM command-line flag names the mechanism (SpecPilot drives it through
 *       environment variables), so both settings are properties-only.</li>
 *       tree nodes, default {@code false}, 1.0 and 16 (SpecPilot's {@code TREE_MODE=ddtree}, arXiv 2611.02501).
 *       A separate draft model ({@link ModelDrafter}) proposes its tree from its own draft sequences' top
 *       candidates under the same keys, and the target verifies it in the same one forward.</li>
 *   <li>{@code neuralfree.draft.tree.recurrent}: whether the tree verify covers a target with recurrent layers
 *       (Gated DeltaNet, KDA, Mamba-2 — hybrid or recurrent), default {@code true}: the round captures each
 *       recurrent layer's mixer inputs during the masked verify and replays the accepted path from the unchanged
 *       committed state, so the committed state equals the state after exactly the accepted tokens. Off, a tree
 *       round refuses recurrent targets by name again; a target whose recurrent mixers the round does not cover
 *       (Mamba-1, PLaMo-2) is refused either way. The round needs the drafter's single-pass tree and runs without
 *       the two-level quantized view, and a GPU target needs paged KV ({@code neuralfree.kv.paged=true}).</li>
 *   <li>{@code neuralfree.draft.tree.rows}, {@code neuralfree.draft.tree.budget}, {@code .tree.thresh.gamma},
       {@code .tree.thresh.lambda}, {@code .tree.mean}, {@code .tree.fixed}, {@code .tree.min} and
       {@code .tree.branches}: the batch engine's verify budget over one step's tree proposals (SpecPilot's
       {@code SPECPILOT_DB_RV_TOKENS} and {@code SPECPILOT_DB_RP}, arXiv 2611.02501). {@code rows} bounds the tree
       rows the step lays out over all its sequences — a constant, or the tier table {@code bs:T;...} /
       {@code bs:xF;...}; {@code budget} divides it over the proposals by their path scores, {@code full} (default)
       verifying every drafted node, {@code topt} keeping the strongest nodes over all requests,
       {@code thresh} keeping a request's nodes while {@code gamma * score >= log lambda} and {@code fixed} giving
       every request the same count; {@code min} floors every request's kept nodes; {@code mean} sets the
       {@code topt} total without a {@code rows} tier as {@code round(mean(bs) * bs)}. Every cut is a per-request
       node prefix, so it moves no token — it only shortens the round. No llama.cpp or vLLM flag names the budget
       (SpecPilot drives it through environment variables), so these settings are properties-only.
       {@code branches} is the branch budget a batch draft model splits a request's tree over, default 4.</li>
   <li>{@code neuralfree.draft.head.vocab}, {@code neuralfree.draft.head.freq}, {@code neuralfree.draft.head.keep},
 *       {@code neuralfree.draft.head.prec} (vLLM {@code draft_lm_head_quantization} of a {@code --speculative-config})
 *       and {@code neuralfree.draft.prec}: the derived candidate head of a {@code dflash} drafter
 *       ({@link DraftHeadOptions}), built once at bind time and used only by the drafter, whose committed tokens
 *       the target's own head verifies unchanged. The vocabulary keeps that many rows of the head the drafter
 *       would use — the draft's own {@code output.weight}, else the target's, else the tied token embeddings —
 *       ranked by the frequency file ({@code head.freq}: JSON {@code {"ranked_ids": [...]}} or lines of
 *       {@code <id> <count>}), always keeping the draft's special ids and the keep list, else by ascending token
 *       id (the BPE merge-rank prior); a synthesized {@code d2t} maps the kept rows back to full-vocabulary ids.
 *       {@code head.prec} stores the kept rows as {@code fp8} (E4M3 with per-row scales) or {@code q8_0},
 *       {@code prec} the drafter's own projection matrices the same way; without a precision the kept rows are
 *       copied in the head's own storage, which is exact. No llama.cpp or vLLM command-line flag names a reduced
 *       draft vocabulary, so the vocabulary size is properties-only.</li>
 *   <li>{@code neuralfree.weights.w12} and {@code neuralfree.weights.w12.head}: the target's bfloat16-representable
 *       projection weights stream to the device as lossless {@link com.eonmux.neuralfree.gguf.GGMLType#W12}
 *       rows instead of their stored type, and the same for the output head. Both default {@code false}; the
 *       head flag needs the first and is refused by name otherwise. The old {@code neuralfree.draft.w12[.head]}
 *       spellings are accepted as aliases. No llama.cpp or vLLM spelling exists
 *       (SpecPilot drives the format through environment variables).</li>
 *   <li>{@code neuralfree.weights.w12.f16} ({@link W12F16Mode}): how an F16 tensor's weights meet the
 *       bfloat16 grid the W12 rows store, {@code refuse} (default) or {@code round}. {@code refuse} refuses the
 *       load's off-grid weights by name, naming the tensor, the first offending element and this property, so
 *       the W12 run keeps the lossless contract. {@code round} rounds every weight onto the grid at upload
 *       (round-to-nearest-even): lossy, each off-grid weight loses its 3 low mantissa bits and the output can
 *       differ from the F16 run; the load reports how many weights moved and their largest relative change.
 *       On-grid F16 tensors and BF16 tensors are unaffected either way.</li>
 * </ul>
 */
public final class SpeculativeOptions {

    public static final String PROP_NGRAM_SIMPLE = "neuralfree.draft.ngram.simple";
    public static final String PROP_NGRAM_MAP_K = "neuralfree.draft.ngram.map.k";
    public static final String PROP_NGRAM_MAP_K4V = "neuralfree.draft.ngram.map.k4v";
    public static final String PROP_NGRAM_MOD = "neuralfree.draft.ngram.mod";
    public static final String PROP_LOOKUP_CACHE_STATIC = "neuralfree.draft.lookup.cache.static";
    public static final String PROP_LOOKUP_CACHE_DYNAMIC = "neuralfree.draft.lookup.cache.dynamic";
    public static final String PROP_SUFFIX_MAX_TREE_DEPTH = "neuralfree.draft.suffix.max.tree.depth";
    public static final String PROP_SUFFIX_MAX_SPEC_FACTOR = "neuralfree.draft.suffix.max.spec.factor";
    public static final String PROP_SUFFIX_MAX_SPEC_OFFSET = "neuralfree.draft.suffix.max.spec.offset";
    public static final String PROP_SUFFIX_MIN_TOKEN_PROB = "neuralfree.draft.suffix.min.token.prob";
    public static final String PROP_SUFFIX_MAX_CACHED_REQUESTS = "neuralfree.draft.suffix.max.cached.requests";
    public static final String PROP_VERIFY = "neuralfree.draft.verify";
    public static final String PROP_FLY_WINDOW = "neuralfree.draft.fly.window";
    public static final String PROP_FLY_ENTROPY_THRESHOLD = "neuralfree.draft.fly.entropy.threshold";
    public static final String PROP_FLY_ENTROPY_TOP_K = "neuralfree.draft.fly.entropy.top.k";
    public static final String PROP_SYNTH_LEN = "neuralfree.draft.synth.len";
    public static final String PROP_SYNTH_RATES = "neuralfree.draft.synth.rates";
    public static final String PROP_VERIFY_DEVICE = "neuralfree.draft.verify.device";
    public static final String PROP_DFLASH_DEVICE = "neuralfree.draft.dflash.device";
    public static final String PROP_DRAFT_SAMPLE_METHOD = "neuralfree.draft.sample.method";
    public static final String PROP_HETEROGENEOUS_VOCAB = "neuralfree.draft.heterogeneous.vocab";
    public static final String PROP_P_SPLIT = "neuralfree.draft.p.split";
    public static final String PROP_TREE = "neuralfree.draft.tree";
    public static final String PROP_TREE_TEMP = "neuralfree.draft.tree.temp";
    public static final String PROP_TREE_NODES = "neuralfree.draft.tree.nodes";
    /** Whether the tree verify covers targets with recurrent layers (Gated DeltaNet, KDA, Mamba-2). */
    public static final String PROP_TREE_RECURRENT = "neuralfree.draft.tree.recurrent";
    /**
     * The tree rows one batch verify step lays out over all its sequences: a constant total, or the tier table
     * {@code bs:T;...} / {@code bs:xF;...} (SpecPilot's {@code SPECPILOT_DB_RV_TOKENS}).
     */
    public static final String PROP_TREE_ROWS = "neuralfree.draft.tree.rows";
    /** How a step's tree-row total divides over its proposals ({@code TreeBudgetPolicy}). */
    public static final String PROP_TREE_BUDGET = "neuralfree.draft.tree.budget";
    /** The {@code thresh} policy's calibration exponent (SpecPilot's {@code SPECPILOT_DB_RP_GAMMA}). */
    public static final String PROP_TREE_THRESH_GAMMA = "neuralfree.draft.tree.thresh.gamma";
    /** The {@code thresh} policy's price in acceptance-probability units (SpecPilot's {@code SPECPILOT_DB_RP_LAMBDA}). */
    public static final String PROP_TREE_THRESH_LAMBDA = "neuralfree.draft.tree.thresh.lambda";
    /**
     * The {@code topt} policy's mean verified nodes per request without a rows tier (SpecPilot's
     * {@code SPECPILOT_DB_RP_NODES}): a constant, or {@code bs:mean;...} tiers.
     */
    public static final String PROP_TREE_MEAN = "neuralfree.draft.tree.mean";
    /** The {@code fixed} policy's node count per request (SpecPilot's {@code SPECPILOT_DB_RP_FIXED}). */
    public static final String PROP_TREE_FIXED = "neuralfree.draft.tree.fixed";
    /** The fewest nodes a request keeps (SpecPilot's {@code SPECPILOT_DB_RP_MIN}). */
    public static final String PROP_TREE_MIN = "neuralfree.draft.tree.min";
    /**
     * The branch budget a batch draft model splits a request's tree over ({@code TreeBranches}): at most this many
     * leaves, the draft sequences it holds per target sequence.
     */
    public static final String PROP_TREE_BRANCHES = "neuralfree.draft.tree.branches";
    /** Two-level verification: a quantized view of the target proposes, the exact target verifies. */
    public static final String PROP_TLV = "neuralfree.draft.tlv";
    /** The Q copy's bit width: 6 ({@code Q6_K} rows) or 8 ({@code Q8_0} rows). */
    public static final String PROP_TLV_BITS = "neuralfree.draft.tlv.bits";

    /** Rows the derived candidate head of a {@code dflash} drafter keeps. */
    public static final String PROP_DRAFT_HEAD_VOCAB = "neuralfree.draft.head.vocab";
    /** The token-frequency ranking file of the derived candidate head. */
    public static final String PROP_DRAFT_HEAD_FREQ = "neuralfree.draft.head.freq";
    /** Token ids the derived candidate head always keeps. */
    public static final String PROP_DRAFT_HEAD_KEEP = "neuralfree.draft.head.keep";
    /** Storage of the derived candidate head (vLLM {@code draft_lm_head_quantization}). */
    public static final String PROP_DRAFT_HEAD_PREC = "neuralfree.draft.head.prec";
    /** Storage of the {@code dflash} drafter's own projection matrices. */
    public static final String PROP_DRAFT_PREC = "neuralfree.draft.prec";
    /** Streams the target's bfloat16-representable projection weights as lossless W12 rows. */
    public static final String PROP_W12 = "neuralfree.weights.w12";
    /** Also streams the output head ({@code output.weight} / {@code lm_head.weight}) when {@link #PROP_W12}. */
    public static final String PROP_W12_HEAD = "neuralfree.weights.w12.head";
    /** Accepted alias of {@link #PROP_W12} (the property's name before it left the draft namespace). */
    public static final String PROP_W12_ALIAS = "neuralfree.draft.w12";
    /** Accepted alias of {@link #PROP_W12_HEAD}. */
    public static final String PROP_W12_HEAD_ALIAS = "neuralfree.draft.w12.head";
    /** How an F16 tensor's weights meet the bfloat16 grid the W12 rows store ({@link W12F16Mode}). */
    public static final String PROP_W12_F16 = "neuralfree.weights.w12.f16";

    /** Default of {@link #flyEntropyThreshold()} (vLLM {@code fly_entropy_threshold}). */
    public static final double DEFAULT_FLY_ENTROPY_THRESHOLD = 0.3;
    /** Default of {@link #flyEntropyTopK()} (vLLM {@code fly_entropy_top_k}). */
    public static final int DEFAULT_FLY_ENTROPY_TOP_K = 3;
    /** Default of {@link #pSplit()}: llama.cpp's {@code --spec-draft-p-split} default. */
    public static final double DEFAULT_P_SPLIT = 0.1;
    /** Default of {@link #treeTemp()}: the drafter's own score scale, as SpecPilot's {@code TREE_TEMP=1}. */
    public static final double DEFAULT_TREE_TEMP = 1.0;
    /** Default of {@link #treeNodes()}: SpecPilot's 64-node budget shrunk to the RTX 5080 laptop measurements. */
    public static final int DEFAULT_TREE_NODES = 16;
    /** Largest tree width the per-row ancestor bitmask covers (one {@code long} per row, itself included). */
    public static final int MAX_TREE_NODES = 63;
    /** Default of {@link #treeBudgetPolicy()}: verify every drafted node, cut only by the step's row caps. */
    public static final TreeBudgetPolicy DEFAULT_TREE_BUDGET = TreeBudgetPolicy.FULL;
    /** Default of {@link #treeThreshGamma()}: acceptance probability {@code ~ exp(gamma * score)} at score scale 1. */
    public static final double DEFAULT_TREE_THRESH_GAMMA = 1.0;
    /** Default of {@link #treeThreshLambda()}: a price of 0 keeps every node the total holds. */
    public static final double DEFAULT_TREE_THRESH_LAMBDA = 0.0;
    /** Default of {@link #treeFixed()}: the reference's fixed per-request node count. */
    public static final int DEFAULT_TREE_FIXED = 16;
    /** Default of {@link #treeMinNodes()}: every request keeps at least its root. */
    public static final int DEFAULT_TREE_MIN_NODES = 1;
    /** Default of {@link #treeBranches()}: the branch budget of a batch draft model's tree proposals. */
    public static final int DEFAULT_TREE_BRANCHES = 4;
    /** Default of {@link #treeRecurrent()}: the covered recurrent families verify in the tree round. */
    public static final boolean DEFAULT_TREE_RECURRENT = true;
    /** Default of {@link #tlvBits()}: SpecPilot's 6-bit attention and FFN Q copy ({@code SPECPILOT_D3_QBITS}). */
    public static final int DEFAULT_TLV_BITS = 6;

    private static final SpeculativeOptions DEFAULTS = builder().build();

    private final Map<NgramDrafter.Kind, NgramDrafter.Options> ngram;
    private final String lookupCacheStatic;
    private final String lookupCacheDynamic;
    private final SuffixDrafter.Options suffix;
    private final VerifyMode verifyMode;
    private final int flyWindow;
    private final double flyEntropyThreshold;
    private final int flyEntropyTopK;
    private final Double synthLength;
    private final double[] synthRates;
    private final boolean verifyOnDevice;
    private final boolean dflashOnDevice;
    private final boolean probabilisticDraft;
    private final boolean heterogeneousVocab;
    private final double pSplit;
    private final boolean treeVerify;
    private final double treeTemp;
    private final int treeNodes;
    private final boolean treeRecurrent;
    private final String treeRows;
    private final TreeBudgetPolicy treeBudgetPolicy;
    private final double treeThreshGamma;
    private final double treeThreshLambda;
    private final String treeMean;
    private final int treeFixed;
    private final int treeMinNodes;
    private final int treeBranches;
    private final boolean twoLevelVerify;
    private final int tlvBits;
    private final DraftHeadOptions draftHead;
    private final boolean w12;
    private final boolean w12Head;
    private final W12F16Mode w12F16;

    private SpeculativeOptions(Builder b, VerifyMode verifyMode) {
        this.ngram = new EnumMap<>(b.ngram);
        this.lookupCacheStatic = b.lookupCacheStatic;
        this.lookupCacheDynamic = b.lookupCacheDynamic;
        this.suffix = b.suffix;
        this.verifyMode = verifyMode;
        this.flyWindow = b.flyWindow;
        this.flyEntropyThreshold = b.flyEntropyThreshold;
        this.flyEntropyTopK = b.flyEntropyTopK;
        this.synthLength = b.synthLength;
        this.synthRates = b.synthRates == null ? null : b.synthRates.clone();
        this.verifyOnDevice = b.verifyOnDevice;
        this.dflashOnDevice = b.dflashOnDevice;
        this.probabilisticDraft = b.probabilisticDraft;
        this.heterogeneousVocab = b.heterogeneousVocab;
        this.pSplit = b.pSplit;
        this.treeVerify = b.treeVerify;
        this.treeTemp = b.treeTemp;
        this.treeNodes = b.treeNodes;
        this.treeRecurrent = b.treeRecurrent;
        this.treeRows = b.treeRows;
        this.treeBudgetPolicy = b.treeBudgetPolicy;
        this.treeThreshGamma = b.treeThreshGamma;
        this.treeThreshLambda = b.treeThreshLambda;
        this.treeMean = b.treeMean;
        this.treeFixed = b.treeFixed;
        this.treeMinNodes = b.treeMinNodes;
        this.treeBranches = b.treeBranches;
        this.twoLevelVerify = b.twoLevelVerify;
        this.tlvBits = b.tlvBits;
        this.draftHead = b.draftHead;
        this.w12 = b.w12;
        this.w12Head = b.w12Head;
        this.w12F16 = b.w12F16;
    }

    /** The defaults: the drafters' default sizes, no lookup cache files, {@link VerifyMode#MATCH}. */
    public static SpeculativeOptions defaults() {
        return DEFAULTS;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * The options of a property set; absent keys keep the defaults.
     *
     * @throws IllegalArgumentException naming the property when a value cannot be parsed or is out of range
     */
    public static SpeculativeOptions fromProperties(Properties p) {
        Objects.requireNonNull(p, "properties");
        Builder b = builder()
                .ngram(mapProperties(p, NgramDrafter.Kind.SIMPLE, PROP_NGRAM_SIMPLE))
                .ngram(mapProperties(p, NgramDrafter.Kind.MAP_K, PROP_NGRAM_MAP_K))
                .ngram(mapProperties(p, NgramDrafter.Kind.MAP_K4V, PROP_NGRAM_MAP_K4V))
                .ngram(modProperties(p))
                .lookupCacheStatic(EngineProperties.stringProperty(p, PROP_LOOKUP_CACHE_STATIC))
                .lookupCacheDynamic(EngineProperties.stringProperty(p, PROP_LOOKUP_CACHE_DYNAMIC))
                .suffix(suffixProperties(p));
        String verify = EngineProperties.stringProperty(p, PROP_VERIFY);
        if (verify != null) {
            b.verifyMode(named(PROP_VERIFY, verify, () -> VerifyMode.parse(verify)));
        }
        b.flyWindow(EngineProperties.intProperty(p, PROP_FLY_WINDOW, 0));
        String threshold = EngineProperties.stringProperty(p, PROP_FLY_ENTROPY_THRESHOLD);
        if (threshold != null) {
            b.flyEntropyThreshold(parseDouble(PROP_FLY_ENTROPY_THRESHOLD, threshold));
        }
        b.flyEntropyTopK(EngineProperties.intProperty(p, PROP_FLY_ENTROPY_TOP_K, DEFAULT_FLY_ENTROPY_TOP_K));
        String len = EngineProperties.stringProperty(p, PROP_SYNTH_LEN);
        if (len != null) {
            b.synthLength(parseDouble(PROP_SYNTH_LEN, len));
        }
        String rates = EngineProperties.stringProperty(p, PROP_SYNTH_RATES);
        if (rates != null) {
            b.synthRates(parseRates(PROP_SYNTH_RATES, rates));
        }
        b.verifyOnDevice(EngineProperties.booleanProperty(p, PROP_VERIFY_DEVICE, true));
        b.dflashOnDevice(EngineProperties.booleanProperty(p, PROP_DFLASH_DEVICE, true));
        String method = EngineProperties.stringProperty(p, PROP_DRAFT_SAMPLE_METHOD);
        if (method != null) {
            b.probabilisticDraft(parseDraftSampleMethod(PROP_DRAFT_SAMPLE_METHOD, method));
        }
        b.heterogeneousVocab(EngineProperties.booleanProperty(p, PROP_HETEROGENEOUS_VOCAB, false));
        String split = EngineProperties.stringProperty(p, PROP_P_SPLIT);
        if (split != null) {
            b.pSplit(parseDouble(PROP_P_SPLIT, split));
        }
        b.treeVerify(EngineProperties.booleanProperty(p, PROP_TREE, false));
        String treeTemp = EngineProperties.stringProperty(p, PROP_TREE_TEMP);
        if (treeTemp != null) {
            b.treeTemp(parseDouble(PROP_TREE_TEMP, treeTemp));
        }
        b.treeNodes(EngineProperties.intProperty(p, PROP_TREE_NODES, DEFAULT_TREE_NODES));
        b.treeRecurrent(EngineProperties.booleanProperty(p, PROP_TREE_RECURRENT, DEFAULT_TREE_RECURRENT));
        String rows = EngineProperties.stringProperty(p, PROP_TREE_ROWS);
        if (rows != null) {
            b.treeRows(rows);
        }
        String budget = EngineProperties.stringProperty(p, PROP_TREE_BUDGET);
        if (budget != null) {
            b.treeBudgetPolicy(TreeBudgetPolicy.parse(PROP_TREE_BUDGET, budget));
        }
        String gamma = EngineProperties.stringProperty(p, PROP_TREE_THRESH_GAMMA);
        if (gamma != null) {
            b.treeThreshGamma(parseDouble(PROP_TREE_THRESH_GAMMA, gamma));
        }
        String lambda = EngineProperties.stringProperty(p, PROP_TREE_THRESH_LAMBDA);
        if (lambda != null) {
            b.treeThreshLambda(parseDouble(PROP_TREE_THRESH_LAMBDA, lambda));
        }
        String mean = EngineProperties.stringProperty(p, PROP_TREE_MEAN);
        if (mean != null) {
            b.treeMean(mean);
        }
        b.treeFixed(EngineProperties.intProperty(p, PROP_TREE_FIXED, DEFAULT_TREE_FIXED));
        b.treeMinNodes(EngineProperties.intProperty(p, PROP_TREE_MIN, DEFAULT_TREE_MIN_NODES));
        b.treeBranches(EngineProperties.intProperty(p, PROP_TREE_BRANCHES, DEFAULT_TREE_BRANCHES));
        b.twoLevelVerify(EngineProperties.booleanProperty(p, PROP_TLV, false));
        b.tlvBits(EngineProperties.intProperty(p, PROP_TLV_BITS, DEFAULT_TLV_BITS));
        b.draftHead(draftHeadProperties(p));
        b.w12(EngineProperties.booleanPropertyAlias(p, PROP_W12, PROP_W12_ALIAS, false));
        b.w12Head(EngineProperties.booleanPropertyAlias(p, PROP_W12_HEAD, PROP_W12_HEAD_ALIAS, false));
        String f16 = EngineProperties.stringProperty(p, PROP_W12_F16);
        if (f16 != null) {
            b.w12F16(W12F16Mode.parse(PROP_W12_F16, f16));
        }
        return b.build();
    }

    /**
     * The derived candidate head of a {@code dflash} drafter, or null when none of its keys is set to a derived
     * setting.
     */
    private static DraftHeadOptions draftHeadProperties(Properties p) {
        DraftHeadOptions.Builder b = DraftHeadOptions.builder();
        intKey(p, PROP_DRAFT_HEAD_VOCAB, b::vocabSize);
        String freq = EngineProperties.stringProperty(p, PROP_DRAFT_HEAD_FREQ);
        if (freq != null) {
            b.frequencyFile(freq);
        }
        String keep = EngineProperties.stringProperty(p, PROP_DRAFT_HEAD_KEEP);
        if (keep != null) {
            int[] ids = parseIds(PROP_DRAFT_HEAD_KEEP, keep);
            named(PROP_DRAFT_HEAD_KEEP, keep, () -> {
                b.keepIds(ids);
                return null;
            });
        }
        String prec = EngineProperties.stringProperty(p, PROP_DRAFT_HEAD_PREC);
        if (prec != null) {
            b.headPrec(DraftHeadOptions.Prec.of(PROP_DRAFT_HEAD_PREC, prec));
        }
        String matrices = EngineProperties.stringProperty(p, PROP_DRAFT_PREC);
        if (matrices != null) {
            b.matricesPrec(DraftHeadOptions.Prec.of(PROP_DRAFT_PREC, matrices));
        }
        DraftHeadOptions o = b.build();
        return o.enabled() ? o : null;
    }

    /**
     * Comma-separated token ids ({@code neuralfree.draft.head.keep}), each a non-negative integer.
     *
     * @throws IllegalArgumentException naming {@code key} for an empty list entry or an entry that is not an
     *                                  integer
     */
    public static int[] parseIds(String key, String v) {
        String[] parts = v.split(",");
        int[] out = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String part = parts[i].trim();
            if (part.isEmpty()) {
                throw new IllegalArgumentException(key + "=" + v + " entry " + i + " is empty");
            }
            try {
                out[i] = Integer.parseInt(part);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(key + "=" + v + " entry " + i + " is not an integer: " + part, e);
            }
        }
        return out;
    }

    /**
     * The {@code size.n}, {@code size.m} and, for {@code ngram-simple}, {@code size.n.min} or, for the map kinds,
     * {@code min.hits} keys under {@code prefix}.
     */
    private static NgramDrafter.Options mapProperties(Properties p, NgramDrafter.Kind kind, String prefix) {
        NgramDrafter.Builder b = NgramDrafter.Options.builder(kind);
        intKey(p, prefix + ".size.n", b::sizeN);
        intKey(p, prefix + ".size.m", b::sizeM);
        if (kind == NgramDrafter.Kind.SIMPLE) {
            intKey(p, prefix + ".size.n.min", b::sizeNMin);
        } else {
            intKey(p, prefix + ".min.hits", b::minHits);
        }
        return named(prefix, "", b::build);
    }

    /**
     * {@code ngram-mod}: {@code n.match} the hashed n-gram, {@code n.max} the draft cap, {@code n.min} the floor a
     * shorter chain is dropped below (a chain of the full {@code n.max} is kept).
     */
    private static NgramDrafter.Options modProperties(Properties p) {
        NgramDrafter.Builder b = NgramDrafter.Options.builder(NgramDrafter.Kind.MOD);
        intKey(p, PROP_NGRAM_MOD + ".n.match", b::sizeN);
        intKey(p, PROP_NGRAM_MOD + ".n.max", b::sizeM);
        intKey(p, PROP_NGRAM_MOD + ".n.min", b::draftMin);
        return named(PROP_NGRAM_MOD, "", b::build);
    }

    private static SuffixDrafter.Options suffixProperties(Properties p) {
        SuffixDrafter.Builder b = SuffixDrafter.Options.builder();
        intKey(p, PROP_SUFFIX_MAX_TREE_DEPTH, b::maxTreeDepth);
        doubleKey(p, PROP_SUFFIX_MAX_SPEC_FACTOR, b::maxSpecFactor);
        doubleKey(p, PROP_SUFFIX_MAX_SPEC_OFFSET, b::maxSpecOffset);
        doubleKey(p, PROP_SUFFIX_MIN_TOKEN_PROB, b::minTokenProb);
        intKey(p, PROP_SUFFIX_MAX_CACHED_REQUESTS, b::maxCachedRequests);
        return named("neuralfree.draft.suffix", "", b::build);
    }

    private static void intKey(Properties p, String key, IntConsumer setter) {
        String v = EngineProperties.stringProperty(p, key);
        if (v != null) {
            int n = EngineProperties.intProperty(p, key, 0);
            named(key, v, () -> {
                setter.accept(n);
                return null;
            });
        }
    }

    private interface DoubleSetter {
        Object set(double v);
    }

    private static void doubleKey(Properties p, String key, DoubleSetter setter) {
        String v = EngineProperties.stringProperty(p, key);
        if (v != null) {
            double d = parseDouble(key, v);
            named(key, v, () -> setter.set(d));
        }
    }

    /** {@code key=value: message} for a refusal of {@code parse}; an empty {@code value} names only the key. */
    private static <T> T named(String key, String value, Supplier<T> parse) {
        try {
            return parse.get();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(key + (value.isEmpty() ? "" : "=" + value) + ": " + e.getMessage(), e);
        }
    }

    private static double parseDouble(String key, String v) {
        try {
            return Double.parseDouble(v.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + v + " is not a number", e);
        }
    }

    /**
     * A {@code draft_sample_method} value: {@code greedy} (false) or {@code probabilistic} (true), case-insensitive.
     *
     * @throws IllegalArgumentException naming {@code key} for any other value
     */
    public static boolean parseDraftSampleMethod(String key, String v) {
        String m = v.trim().toLowerCase(Locale.ROOT);
        if (m.equals("greedy")) {
            return false;
        }
        if (m.equals("probabilistic")) {
            return true;
        }
        throw new IllegalArgumentException(key + "=" + v + ": expected greedy or probabilistic");
    }

    /**
     * Comma-separated per-position rates ({@code P0,P1,...}), each a number.
     *
     * @throws IllegalArgumentException naming {@code key} for an empty list or an entry that is not a number
     */
    public static double[] parseRates(String key, String v) {
        List<Double> out = new ArrayList<>();
        for (String part : v.split(",")) {
            out.add(parseDouble(key, part));
        }
        double[] rates = new double[out.size()];
        for (int i = 0; i < rates.length; i++) {
            rates[i] = out.get(i);
        }
        return rates;
    }

    /**
     * The sizes of the n-gram drafter of {@code kind}; their caches and corpus are unset (the lookup cache files are
     * {@link #lookupCacheStatic()} and {@link #lookupCacheDynamic()}).
     */
    public NgramDrafter.Options ngramOptions(NgramDrafter.Kind kind) {
        return ngram.get(Objects.requireNonNull(kind, "kind"));
    }

    /** The static {@code ngram-cache} file ({@code --lookup-cache-static}), or null. */
    public String lookupCacheStatic() { return lookupCacheStatic; }

    /** The dynamic {@code ngram-cache} file ({@code --lookup-cache-dynamic}), or null. */
    public String lookupCacheDynamic() { return lookupCacheDynamic; }

    /** Whether a lookup cache file is configured. */
    public boolean hasLookupCacheFiles() {
        return lookupCacheStatic != null || lookupCacheDynamic != null;
    }

    /** The {@code suffix} drafter's limits, without a corpus. */
    public SuffixDrafter.Options suffixOptions() { return suffix; }

    /** How proposals are verified ({@link VerifyMode#MATCH} by default). */
    public VerifyMode verifyMode() { return verifyMode; }

    /** FLy's window, or 0 for {@code min(6, k - 1)}. */
    public int flyWindow() { return flyWindow; }

    /** FLy's entropy gate. */
    public double flyEntropyThreshold() { return flyEntropyThreshold; }

    /** The processed probabilities FLy's entropy covers. */
    public int flyEntropyTopK() { return flyEntropyTopK; }

    /** The synthetic mean acceptance length, or null. */
    public Double synthLength() { return synthLength; }

    /** The synthetic per-position acceptance rates, or null. */
    public double[] synthRates() { return synthRates == null ? null : synthRates.clone(); }

    /** Whether {@code match} verification of an argmax chain selects on the target's device (default true). */
    public boolean verifyOnDevice() { return verifyOnDevice; }

    /** Whether a {@code dflash} drafter of a GPU target drafts on the target's device (default true). */
    public boolean dflashOnDevice() { return dflashOnDevice; }

    /** Whether a draft model drafts by a draw from its distribution (vLLM {@code draft_sample_method=probabilistic}). */
    public boolean probabilisticDraft() { return probabilisticDraft; }

    /**
     * Whether a separate draft model drafts across two vocabularies by their token-level intersection (vLLM
     * {@code use_heterogeneous_vocab}).
     */
    public boolean heterogeneousVocab() { return heterogeneousVocab; }

    /**
     * The draft branch split probability (llama.cpp {@code --spec-draft-p-split}, default 0.1): a draft-model
     * speculation over a paged memory whose {@code neuralfree.parallel} holds more than one sequence splits a
     * branch when a secondary candidate's probability exceeds it.
     */
    public double pSplit() { return pSplit; }

    /**
     * Whether a {@code dflash} drafter proposes a draft tree the target verifies in one forward (SpecPilot's
     * {@code TREE_MODE=ddtree}, arXiv 2611.02501), instead of its single selector walk, default {@code false}. A
     * separate draft model keeps its {@code neuralfree.draft.p.split} branching either way.
     */
    public boolean treeVerify() { return treeVerify; }

    /** The temperature the tree builder scales its path scores at ({@link #DEFAULT_TREE_TEMP}). */
    public double treeTemp() { return treeTemp; }

    /**
     * The fixed verify width in tree nodes: every single-pass verify runs one forward of {@code treeNodes + 1}
     * rows (the pending token plus the nodes, shorter trees padded), whatever the proposal drafts, so the step's
     * shape never changes ({@link #DEFAULT_TREE_NODES}).
     */
    public int treeNodes() { return treeNodes; }

    /**
     * Whether the tree verify covers a target with recurrent layers (Gated DeltaNet, KDA, Mamba-2 hybrid or
     * recurrent; default {@link #DEFAULT_TREE_RECURRENT}): the round captures each recurrent layer's mixer inputs
     * during the masked verify and replays the accepted path from the unchanged committed state, so the committed
     * state equals the state after exactly the accepted tokens. Off, the tree verify refuses recurrent targets by
     * name again. Targets whose recurrent mixers the round does not cover (Mamba-1, PLaMo-2) are refused either
     * way.
     */
    public boolean treeRecurrent() { return treeRecurrent; }

    /**
     * The tree rows one batch verify step lays out over all its sequences ({@code neuralfree.draft.tree.rows}), or
     * null: a constant total, or the tier table {@code bs:T;...} / {@code bs:xF;...}, first tier that covers the
     * batch size winning (SpecPilot's {@code SPECPILOT_DB_RV_TOKENS}). Unset, a step lays out every drafted node.
     */
    public String treeRows() { return treeRows; }

    /** How a step's tree-row total divides over its proposals ({@link #DEFAULT_TREE_BUDGET}). */
    public TreeBudgetPolicy treeBudgetPolicy() { return treeBudgetPolicy; }

    /** The {@code thresh} policy's calibration exponent ({@link #DEFAULT_TREE_THRESH_GAMMA}). */
    public double treeThreshGamma() { return treeThreshGamma; }

    /** The {@code thresh} policy's price in acceptance-probability units ({@link #DEFAULT_TREE_THRESH_LAMBDA}). */
    public double treeThreshLambda() { return treeThreshLambda; }

    /**
     * The {@code topt} policy's mean verified nodes per request without a rows tier ({@code neuralfree.draft.tree.mean}),
     * or null: a constant, or {@code bs:mean;...} tiers (SpecPilot's {@code SPECPILOT_DB_RP_NODES}).
     */
    public String treeMean() { return treeMean; }

    /** The {@code fixed} policy's node count per request ({@link #DEFAULT_TREE_FIXED}). */
    public int treeFixed() { return treeFixed; }

    /** The fewest nodes a request keeps ({@link #DEFAULT_TREE_MIN_NODES}). */
    public int treeMinNodes() { return treeMinNodes; }

    /** The branch budget a batch draft model splits a request's tree over ({@link #DEFAULT_TREE_BRANCHES}). */
    public int treeBranches() { return treeBranches; }

    /**
     * Whether two-level verification builds a quantized view of the target ({@code neuralfree.draft.tlv},
     * SpecPilot's TLV, arXiv 2611.02501): the view verifies the draft tree once greedily, its accepted chain and
     * bonus anchor a second draft, and the exact target verifies the composed tree in one pass and commits — only
     * the exact target ever decides a token. Greedy decoding only; needs the draft tree
     * ({@link #treeVerify()}) and a target whose weights the Q copy undercuts ({@link #tlvBits()}).
     */
    public boolean twoLevelVerify() { return twoLevelVerify; }

    /**
     * The Q copy's bits per weight ({@link #DEFAULT_TLV_BITS}): 6 stores a matrix's rows as {@code Q6_K}
     * (6.5625 bits, whole 256-value blocks), 8 as {@code Q8_0} (8.5 bits, whole 32-value blocks); a matrix whose
     * rows or stored type the copy does not undercut stays exact. Any other value is refused by name.
     */
    public int tlvBits() { return tlvBits; }

    /**
     * The derived candidate head of a {@code dflash} drafter, or null when none is configured; see
     * {@link DraftHeadOptions}.
     */
    public DraftHeadOptions draftHead() { return draftHead; }

    /**
     * Whether the target's bfloat16-representable projection weights stream to the device as lossless
     * {@link com.eonmux.neuralfree.gguf.GGMLType#W12} rows (default false; the flag of the SpecPilot W12
     * target weights). Off, or on a CPU target, the weights keep their stored type.
     */
    public boolean w12() { return w12; }

    /** Whether {@link #w12()} also streams the output head (default false). */
    public boolean w12Head() { return w12Head; }

    /**
     * How an F16 tensor's weights meet the bfloat16 grid the W12 rows store ({@link W12F16Mode#REFUSE} by
     * default, which refuses off-grid weights by name; {@link W12F16Mode#ROUND} rounds them lossily).
     */
    public W12F16Mode w12F16() { return w12F16; }

    /**
     * Sets the verification of {@code decoder}: the mode, FLy's settings, the synthetic acceptance and the device
     * selection of {@code match} verification. Call after
     * the decoder's draft max is set, which a synthetic length resolves against.
     *
     * @throws IllegalArgumentException when the synthetic length exceeds the draft max
     */
    public SpeculativeDecoder.Builder applyTo(SpeculativeDecoder.Builder decoder) {
        decoder.verifyMode(verifyMode).flyWindow(flyWindow).flyEntropyThreshold(flyEntropyThreshold)
                .flyEntropyTopK(flyEntropyTopK).verifyOnDevice(verifyOnDevice);
        if (synthRates != null) {
            decoder.syntheticAcceptanceRates(synthRates);
        } else if (synthLength != null) {
            decoder.syntheticAcceptanceLength(synthLength);
        }
        return decoder;
    }

    /**
     * These options restated as a {@link Builder}. A verification mode equal to the one {@link Builder#build()}
     * derives (synthetic with a synthetic acceptance, match without) stays unset, so a later synthetic setting
     * still selects synthetic verification.
     */
    public Builder toBuilder() {
        Builder b = builder().lookupCacheStatic(lookupCacheStatic).lookupCacheDynamic(lookupCacheDynamic)
                .suffix(suffix).flyWindow(flyWindow).flyEntropyThreshold(flyEntropyThreshold)
                .flyEntropyTopK(flyEntropyTopK).verifyOnDevice(verifyOnDevice).dflashOnDevice(dflashOnDevice)
                .probabilisticDraft(probabilisticDraft).heterogeneousVocab(heterogeneousVocab).pSplit(pSplit)
                .treeVerify(treeVerify).treeTemp(treeTemp).treeNodes(treeNodes)
                .treeRecurrent(treeRecurrent)
                .treeRows(treeRows).treeBudgetPolicy(treeBudgetPolicy).treeThreshGamma(treeThreshGamma)
                .treeThreshLambda(treeThreshLambda).treeMean(treeMean).treeFixed(treeFixed)
                .treeMinNodes(treeMinNodes).treeBranches(treeBranches)
                .twoLevelVerify(twoLevelVerify).tlvBits(tlvBits)
                .draftHead(draftHead)
                .w12(w12).w12Head(w12Head)
                .w12F16(w12F16);
        for (NgramDrafter.Options o : ngram.values()) {
            b.ngram(o);
        }
        boolean synth = synthLength != null || synthRates != null;
        b.verifyMode = verifyMode == (synth ? VerifyMode.SYNTHETIC : VerifyMode.MATCH) ? null : verifyMode;
        b.synthLength = synthLength;
        b.synthRates = synthRates == null ? null : synthRates.clone();
        return b;
    }

    /** True when every setting is the default. */
    public boolean isDefault() {
        return changes().isEmpty();
    }

    /** The settings that differ from {@link #defaults()}, each as {@code name=value}. */
    private List<String> changes() {
        List<String> out = new ArrayList<>();
        for (NgramDrafter.Options o : ngram.values()) {
            NgramDrafter.Options d = NgramDrafter.Options.builder(o.kind()).build();
            if (o.sizeN() != d.sizeN() || o.sizeNMin() != o.sizeN() || o.sizeM() != d.sizeM()
                    || o.minHits() != d.minHits() || o.draftMin() != d.draftMin()) {
                out.add(o.kind().name().toLowerCase(Locale.ROOT) + "=[n=" + (o.lookupRange() ? o.sizeNMin() + ".."
                        : "") + o.sizeN() + ", m=" + o.sizeM() + ", minHits=" + o.minHits() + ", min="
                        + o.draftMin() + "]");
            }
        }
        SuffixDrafter.Options s = SuffixDrafter.Options.builder().build();
        if (suffix.maxTreeDepth() != s.maxTreeDepth() || suffix.maxSpecFactor() != s.maxSpecFactor()
                || suffix.maxSpecOffset() != s.maxSpecOffset() || suffix.minTokenProb() != s.minTokenProb()
                || suffix.maxCachedRequests() != s.maxCachedRequests()) {
            out.add("suffix=[depth=" + suffix.maxTreeDepth() + ", factor=" + suffix.maxSpecFactor() + ", offset="
                    + suffix.maxSpecOffset() + ", minProb=" + suffix.minTokenProb() + ", cachedRequests="
                    + suffix.maxCachedRequests() + "]");
        }
        if (lookupCacheStatic != null) {
            out.add("lookupCacheStatic=" + lookupCacheStatic);
        }
        if (lookupCacheDynamic != null) {
            out.add("lookupCacheDynamic=" + lookupCacheDynamic);
        }
        if (verifyMode != VerifyMode.MATCH) {
            out.add("verify=" + verifyMode);
        }
        if (flyWindow != 0 || flyEntropyThreshold != DEFAULT_FLY_ENTROPY_THRESHOLD
                || flyEntropyTopK != DEFAULT_FLY_ENTROPY_TOP_K) {
            out.add("fly=[window=" + flyWindow + ", threshold=" + flyEntropyThreshold + ", topK=" + flyEntropyTopK
                    + "]");
        }
        if (synthLength != null) {
            out.add("synthLength=" + synthLength);
        }
        if (synthRates != null) {
            out.add("synthRates=" + Arrays.toString(synthRates));
        }
        if (!verifyOnDevice) {
            out.add("verifyOnDevice=false");
        }
        if (!dflashOnDevice) {
            out.add("dflashOnDevice=false");
        }
        if (probabilisticDraft) {
            out.add("draftSampleMethod=probabilistic");
        }
        if (heterogeneousVocab) {
            out.add("heterogeneousVocab=true");
        }
        if (pSplit != DEFAULT_P_SPLIT) {
            out.add("pSplit=" + pSplit);
        }
        if (treeVerify) {
            out.add("tree=true");
        }
        if (treeTemp != DEFAULT_TREE_TEMP) {
            out.add("treeTemp=" + treeTemp);
        }
        if (treeNodes != DEFAULT_TREE_NODES) {
            out.add("treeNodes=" + treeNodes);
        }
        if (!treeRecurrent) {
            out.add("treeRecurrent=false");
        }
        if (treeRows != null) {
            out.add("treeRows=" + treeRows);
        }
        if (treeBudgetPolicy != DEFAULT_TREE_BUDGET) {
            out.add("treeBudget=" + treeBudgetPolicy.name().toLowerCase(Locale.ROOT));
        }
        if (treeThreshGamma != DEFAULT_TREE_THRESH_GAMMA || treeThreshLambda != DEFAULT_TREE_THRESH_LAMBDA) {
            out.add("treeThresh=[gamma=" + treeThreshGamma + ", lambda=" + treeThreshLambda + "]");
        }
        if (treeMean != null) {
            out.add("treeMean=" + treeMean);
        }
        if (treeFixed != DEFAULT_TREE_FIXED) {
            out.add("treeFixed=" + treeFixed);
        }
        if (treeMinNodes != DEFAULT_TREE_MIN_NODES) {
            out.add("treeMin=" + treeMinNodes);
        }
        if (treeBranches != DEFAULT_TREE_BRANCHES) {
            out.add("treeBranches=" + treeBranches);
        }
        if (twoLevelVerify) {
            out.add("tlv=true");
        }
        if (tlvBits != DEFAULT_TLV_BITS) {
            out.add("tlvBits=" + tlvBits);
        }
        if (draftHead != null && draftHead.enabled()) {
            out.add("draftHead" + draftHead.describe());
        }
        if (w12) {
            out.add("w12=true");
        }
        if (w12Head) {
            out.add("w12Head=true");
        }
        if (w12F16 != W12F16Mode.DEFAULT) {
            out.add("w12F16=" + w12F16.propertyNameValue());
        }
        return out;
    }

    @Override
    public String toString() {
        return "SpeculativeOptions" + changes();
    }

    public static final class Builder {
        private final Map<NgramDrafter.Kind, NgramDrafter.Options> ngram = new EnumMap<>(NgramDrafter.Kind.class);
        private String lookupCacheStatic;
        private String lookupCacheDynamic;
        private SuffixDrafter.Options suffix = SuffixDrafter.Options.builder().build();
        private VerifyMode verifyMode;
        private int flyWindow;
        private double flyEntropyThreshold = DEFAULT_FLY_ENTROPY_THRESHOLD;
        private int flyEntropyTopK = DEFAULT_FLY_ENTROPY_TOP_K;
        private Double synthLength;
        private double[] synthRates;
        private boolean verifyOnDevice = true;
        private boolean dflashOnDevice = true;
        private boolean probabilisticDraft;
        private boolean heterogeneousVocab;
        private double pSplit = DEFAULT_P_SPLIT;
        private boolean treeVerify;
        private double treeTemp = DEFAULT_TREE_TEMP;
        private int treeNodes = DEFAULT_TREE_NODES;
        private boolean treeRecurrent = DEFAULT_TREE_RECURRENT;
        private String treeRows;
        private TreeBudgetPolicy treeBudgetPolicy = DEFAULT_TREE_BUDGET;
        private double treeThreshGamma = DEFAULT_TREE_THRESH_GAMMA;
        private double treeThreshLambda = DEFAULT_TREE_THRESH_LAMBDA;
        private String treeMean;
        private int treeFixed = DEFAULT_TREE_FIXED;
        private int treeMinNodes = DEFAULT_TREE_MIN_NODES;
        private int treeBranches = DEFAULT_TREE_BRANCHES;
        private boolean twoLevelVerify;
        private int tlvBits = DEFAULT_TLV_BITS;
        private DraftHeadOptions draftHead;
        private boolean w12;
        private boolean w12Head;
        private W12F16Mode w12F16 = W12F16Mode.DEFAULT;

        private Builder() {
            for (NgramDrafter.Kind kind : NgramDrafter.Kind.values()) {
                ngram.put(kind, NgramDrafter.Options.builder(kind).build());
            }
        }

        /**
         * The sizes of the n-gram drafter of {@code options.kind()}, replacing the earlier ones of that kind.
         *
         * @throws IllegalArgumentException when the options carry a cache or a corpus (runtime state, not
         *                                  configuration)
         */
        public Builder ngram(NgramDrafter.Options options) {
            Objects.requireNonNull(options, "options");
            if (options.dynamicCache() != null || options.staticCache() != null || options.corpus() != null) {
                throw new IllegalArgumentException("n-gram options of the configuration carry no caches or corpus;"
                        + " set the cache files with " + PROP_LOOKUP_CACHE_STATIC + " / "
                        + PROP_LOOKUP_CACHE_DYNAMIC);
            }
            ngram.put(options.kind(), options);
            return this;
        }

        /** The static {@code ngram-cache} file; null or blank removes it. */
        public Builder lookupCacheStatic(String path) {
            this.lookupCacheStatic = blankToNull(path);
            return this;
        }

        /** The dynamic {@code ngram-cache} file; null or blank removes it. */
        public Builder lookupCacheDynamic(String path) {
            this.lookupCacheDynamic = blankToNull(path);
            return this;
        }

        /**
         * The {@code suffix} drafter's limits.
         *
         * @throws IllegalArgumentException when the options carry a corpus
         */
        public Builder suffix(SuffixDrafter.Options options) {
            Objects.requireNonNull(options, "options");
            if (options.corpus() != null) {
                throw new IllegalArgumentException("suffix options of the configuration carry no corpus");
            }
            this.suffix = options;
            return this;
        }

        /** The verification mode; null restores the default ({@link VerifyMode#MATCH}, or synthetic with rates). */
        public Builder verifyMode(VerifyMode v) {
            this.verifyMode = v;
            return this;
        }

        public Builder flyWindow(int v) {
            if (v < 0) {
                throw new IllegalArgumentException(PROP_FLY_WINDOW + " must not be negative, got " + v);
            }
            this.flyWindow = v;
            return this;
        }

        public Builder flyEntropyThreshold(double v) {
            if (!(v >= 0.0) || !Double.isFinite(v)) {
                throw new IllegalArgumentException(PROP_FLY_ENTROPY_THRESHOLD + " must be finite and >= 0, got " + v);
            }
            this.flyEntropyThreshold = v;
            return this;
        }

        public Builder flyEntropyTopK(int v) {
            if (v <= 0) {
                throw new IllegalArgumentException(PROP_FLY_ENTROPY_TOP_K + " must be positive, got " + v);
            }
            this.flyEntropyTopK = v;
            return this;
        }

        /** The synthetic mean acceptance length, including the target token; null removes it. */
        public Builder synthLength(Double v) {
            if (v != null && !(v >= 1.0)) {
                throw new IllegalArgumentException(PROP_SYNTH_LEN + " must be at least 1, got " + v);
            }
            this.synthLength = v;
            return this;
        }

        /**
         * The synthetic per-position acceptance rates; null removes them.
         *
         * @throws IllegalArgumentException naming the property for rates {@link SyntheticVerifier} refuses
         */
        public Builder synthRates(double[] v) {
            if (v != null) {
                named(PROP_SYNTH_RATES, "", () -> new SyntheticVerifier(v));
            }
            this.synthRates = v == null ? null : v.clone();
            return this;
        }

        /** Whether a draft model drafts by a draw from its distribution ({@code draft_sample_method}). */
        public Builder probabilisticDraft(boolean v) {
            this.probabilisticDraft = v;
            return this;
        }

        /**
         * Whether a separate draft model drafts across two vocabularies by their token-level intersection
         * ({@code neuralfree.draft.heterogeneous.vocab}, vLLM {@code use_heterogeneous_vocab}).
         */
        public Builder heterogeneousVocab(boolean v) {
            this.heterogeneousVocab = v;
            return this;
        }

        /**
         * The draft branch split probability (llama.cpp {@code --spec-draft-p-split}, default 0.1).
         *
         * @throws IllegalArgumentException for a probability outside {@code [0, 1]}
         */
        public Builder pSplit(double v) {
            if (!(v >= 0.0 && v <= 1.0)) {
                throw new IllegalArgumentException(PROP_P_SPLIT + " must be in [0, 1], got " + v);
            }
            this.pSplit = v;
            return this;
        }

        /** Streams the target's bfloat16-representable projection weights as lossless W12 blocks. */
        public Builder w12(boolean v) {
            this.w12 = v;
            return this;
        }

        /** Also streams the output head; needs {@link #w12(boolean)}. */
        public Builder w12Head(boolean v) {
            this.w12Head = v;
            return this;
        }

        /**
         * How an F16 tensor's weights meet the bfloat16 grid the W12 rows store.
         *
         * @throws NullPointerException when {@code v} is null
         */
        public Builder w12F16(W12F16Mode v) {
            this.w12F16 = Objects.requireNonNull(v, "w12F16");
            return this;
        }

        /**
         * Whether a {@code dflash} drafter proposes a draft tree verified in one target forward
         * ({@code neuralfree.draft.tree}; default false).
         */
        public Builder treeVerify(boolean v) {
            this.treeVerify = v;
            return this;
        }

        /**
         * The tree builder's score temperature.
         *
         * @throws IllegalArgumentException for a non-positive temperature
         */
        public Builder treeTemp(double v) {
            if (!(v > 0.0) || !Double.isFinite(v)) {
                throw new IllegalArgumentException(PROP_TREE_TEMP + " must be finite and positive, got " + v);
            }
            this.treeTemp = v;
            return this;
        }

        /**
         * The fixed verify width in tree nodes ({@code neuralfree.draft.tree.nodes}; default
         * {@link #DEFAULT_TREE_NODES}).
         *
         * @throws IllegalArgumentException outside 1..{@link #MAX_TREE_NODES}, the per-row ancestor bitmask's reach
         */
        public Builder treeNodes(int v) {
            if (v < 1 || v > MAX_TREE_NODES) {
                throw new IllegalArgumentException(PROP_TREE_NODES + " must lie between 1 and " + MAX_TREE_NODES
                        + " (one long names a verify batch's rows), got " + v);
            }
            this.treeNodes = v;
            return this;
        }

        /**
         * Whether the tree verify covers targets with recurrent layers ({@code neuralfree.draft.tree.recurrent};
         * default true).
         */
        public Builder treeRecurrent(boolean v) {
            this.treeRecurrent = v;
            return this;
        }

        /**
         * The tree rows one batch verify step lays out over all its sequences ({@code neuralfree.draft.tree.rows});
         * null keeps every drafted node.
         *
         * @throws IllegalArgumentException for a tier whose total is below one row or whose scaled factor is
         *                                  negative; the tiers parse when the batch budget builds
         */
        public Builder treeRows(String v) {
            this.treeRows = blankToNull(v);
            return this;
        }

        /**
         * How a step's tree-row total divides over its proposals ({@code neuralfree.draft.tree.budget}).
         *
         * @throws NullPointerException for a null policy
         */
        public Builder treeBudgetPolicy(TreeBudgetPolicy v) {
            this.treeBudgetPolicy = Objects.requireNonNull(v, "treeBudgetPolicy");
            return this;
        }

        /**
         * The {@code thresh} policy's calibration exponent.
         *
         * @throws IllegalArgumentException for a negative or non-finite exponent
         */
        public Builder treeThreshGamma(double v) {
            if (!(v >= 0.0) || !Double.isFinite(v)) {
                throw new IllegalArgumentException(PROP_TREE_THRESH_GAMMA + " must be finite and >= 0, got " + v);
            }
            this.treeThreshGamma = v;
            return this;
        }

        /**
         * The {@code thresh} policy's price in acceptance-probability units; at most 0 keeps every node.
         *
         * @throws IllegalArgumentException for a negative price
         */
        public Builder treeThreshLambda(double v) {
            if (!(v >= 0.0) || !Double.isFinite(v)) {
                throw new IllegalArgumentException(PROP_TREE_THRESH_LAMBDA + " must be finite and >= 0, got " + v);
            }
            this.treeThreshLambda = v;
            return this;
        }

        /**
         * The {@code topt} policy's mean verified nodes per request without a rows tier; null keeps the full-width
         * total. The value parses when the batch budget builds.
         */
        public Builder treeMean(String v) {
            this.treeMean = blankToNull(v);
            return this;
        }

        /**
         * The {@code fixed} policy's node count per request.
         *
         * @throws IllegalArgumentException below one
         */
        public Builder treeFixed(int v) {
            if (v < 1) {
                throw new IllegalArgumentException(PROP_TREE_FIXED + " must be at least 1, got " + v);
            }
            this.treeFixed = v;
            return this;
        }

        /**
         * The fewest nodes a request keeps.
         *
         * @throws IllegalArgumentException below one
         */
        public Builder treeMinNodes(int v) {
            if (v < 1) {
                throw new IllegalArgumentException(PROP_TREE_MIN + " must be at least 1, got " + v);
            }
            this.treeMinNodes = v;
            return this;
        }

        /**
         * The branch budget a batch draft model splits a request's tree over
         * ({@code neuralfree.draft.tree.branches}).
         *
         * @throws IllegalArgumentException below one
         */
        public Builder treeBranches(int v) {
            if (v < 1) {
                throw new IllegalArgumentException(PROP_TREE_BRANCHES + " must be at least 1, got " + v);
            }
            this.treeBranches = v;
            return this;
        }

        /**
         * Whether two-level verification builds a quantized target view ({@code neuralfree.draft.tlv}; default
         * false).
         */
        public Builder twoLevelVerify(boolean v) {
            this.twoLevelVerify = v;
            return this;
        }

        /**
         * The Q copy's bits per weight ({@code neuralfree.draft.tlv.bits}; default {@link #DEFAULT_TLV_BITS}).
         *
         * @throws IllegalArgumentException naming the property for any value but 6 ({@code Q6_K}) or 8 ({@code Q8_0})
         */
        public Builder tlvBits(int v) {
            if (v != 6 && v != 8) {
                throw new IllegalArgumentException(PROP_TLV_BITS + " must be 6 (Q6_K rows) or 8 (Q8_0 rows), got " + v);
            }
            this.tlvBits = v;
            return this;
        }

        /**
         * Whether {@code match} verification of an argmax chain selects every verified row's token on the target's
         * device ({@link ForwardBackend#forwardRowsGreedy}; default true).
         */
        public Builder verifyOnDevice(boolean v) {
            this.verifyOnDevice = v;
            return this;
        }

        /**
         * Whether a {@code dflash} drafter of a GPU target drafts on the target's device
         * ({@code DflashDevice}; default true).
         */
        public Builder dflashOnDevice(boolean v) {
            this.dflashOnDevice = v;
            return this;
        }

        /**
         * The derived candidate head of a {@code dflash} drafter; null keeps the drafter's own head and matrices.
         */
        public Builder draftHead(DraftHeadOptions v) {
            this.draftHead = v;
            return this;
        }

        /**
         * Builds the options. Synthetic acceptance selects {@link VerifyMode#SYNTHETIC} when no mode is set.
         *
         * @throws IllegalArgumentException when both synthetic settings are given, when they are given beside
         *                                  another mode, or when {@code synthetic} has neither
         */
        public SpeculativeOptions build() {
            boolean synth = synthLength != null || synthRates != null;
            if (synthLength != null && synthRates != null) {
                throw new IllegalArgumentException(PROP_SYNTH_LEN + " and " + PROP_SYNTH_RATES
                        + " exclude each other; set one");
            }
            VerifyMode mode = verifyMode != null ? verifyMode : synth ? VerifyMode.SYNTHETIC : VerifyMode.MATCH;
            if (synth && mode != VerifyMode.SYNTHETIC) {
                throw new IllegalArgumentException((synthLength != null ? PROP_SYNTH_LEN : PROP_SYNTH_RATES)
                        + " applies to synthetic verification only, but " + PROP_VERIFY + "=" + mode);
            }
            if (!synth && mode == VerifyMode.SYNTHETIC) {
                throw new IllegalArgumentException(PROP_VERIFY + "=synthetic needs " + PROP_SYNTH_LEN + " or "
                        + PROP_SYNTH_RATES);
            }
            if (w12Head && !w12) {
                throw new IllegalArgumentException(PROP_W12_HEAD + " needs " + PROP_W12 + "=true, which streams"
                        + " the target's weights it applies to");
            }
            return new SpeculativeOptions(this, mode);
        }

        private static String blankToNull(String v) {
            return v == null || v.trim().isEmpty() ? null : v.trim();
        }
    }
}
