package com.eonmux.neuralfree.gguf.hparams;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.PoolerAct;
import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SwaType;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Mutable builder of {@link ModelHparams}, filled by {@link HparamsLoader} and the architecture hooks.
 *
 * <p>Accessors mirror {@link ModelHparams}: {@code x()} reads and {@code x(v)} writes a value; per-layer
 * values use {@code x(il)}, {@code x(il, v)}, {@code xArray()} and {@code xArray(values)}. Array arguments and
 * results are copies. Per-layer arrays have length {@link #nLayerAll()} and are (re)sized by
 * {@link #layerCounts(int, int)}, which keeps existing values. Defaults match llama.cpp
 * {@code llama_hparams} (src/llama-hparams.h) and {@code load_hparams} (src/llama-model.cpp:1361-1453).
 */
public final class ModelHparamsBuilder implements Cloneable {

    private final Arch arch;
    private final ArchTraits traits;
    private final boolean recurrentDefault;
    private int nLayerAll;
    private int nextnLayers;
    private boolean[] moeOverride;
    private int[] kvOwnerOverride;

    // ---- identity ----
    String name = null;
    int fileType = -1;

    // ---- sizes ----
    int nVocab = 0;
    int nCtxTrain = 0;
    int nEmbd = 0;
    int nEmbdInpImpl = 0;
    int nEmbdInpEncImpl = 0;
    int nEmbdOutImpl = 0;
    int nEmbdPerLayer = 0;
    int nLayerDecision = 0;
    int routerLayer = -1;
    int posnetEmbd = 0;
    int posnetLayers = 0;
    int convnextEmbd = 0;
    int convnextLayers = 0;
    int nDeepstackLayers = 0;
    int nTokensDelay = 0;

    // ---- head dimensions ----
    int headKFull = 0;
    int headVFull = 0;
    int headKSwa = 0;
    int headVSwa = 0;
    int nRotFull = 0;
    int nRotSwa = 0;
    int headKMlaImpl = 0;
    int headVMlaImpl = 0;
    int headKMlaSwa = 0;
    int headVMlaSwa = 0;
    int nLoraQ = 0;
    int nLoraKv = 0;
    int nLoraKvSwa = 0;

    // ---- norms ----
    float normEps = 0f;
    float normRmsEps = 0f;
    float normGroupEps = 0f;
    float postNormEps = 0f;
    int nNormGroups = 0;
    boolean swinNorm = false;
    boolean normBeforeResidual = false;
    boolean normBeforeFc = false;
    boolean useParallelResidual = false;

    // ---- attention ----
    boolean causal = true;
    int nonCausalType = ModelHparams.NON_CAUSAL_ALL;
    float attentionScale = 0f;
    float attnOutScale = 0f;
    float attnValueScale = 0f;
    boolean attnSoftcapEnabled = false;
    float attnSoftcap = 50.0f;
    float routerSoftcap = 30.0f;
    float finalSoftcap = 30.0f;
    float clampKqv = 0f;
    float maxAlibiBias = 0f;
    boolean useAlibi = false;
    boolean alibiSqrt = false;
    int nRelAttnBuckets = 0;
    int attnTempLength = 0;
    int attnTempFloorScale = 0;
    float attnTempScale = 0f;
    float attnTempOffset = 0f;
    boolean attnTempPow = false;
    int noRopeLayerStep = 4;
    boolean useKqNorm = false;
    int nLayerKvFromStart = -1;
    int sharedKvLayers = 0;
    int nValueExpert = 0;
    int nValueExpertUsed = 0;
    float xsaEps = 0f;

    // ---- limite: dense cross-layer connections and the sigmoid logit cap ----
    int muddTapCount = 0;
    int muddFf = 0;
    int[] muddTapIdx = new int[0];
    float[] finalLogitSigmoidCapping = new float[3];

    // ---- sliding window ----
    SwaType swaType = SwaType.NONE;
    int nSwa = 0;
    /** Per-layer windows (0 = full attention) when the layers use different sizes; empty for the scalar window. */
    int[] swaWindowArr = new int[0];
    /** Learned static sink tokens per layer ({@code attention.sink_token_count}); 0 without them. */
    int sinkTokenCount = 0;

    // ---- rope ----
    RopeType ropeType = RopeType.NONE;
    float ropeFreqBase = 10000.0f;
    float ropeFreqBaseSwa = 10000.0f;
    float ropeFreqBaseSpatial = 0f;
    float ropeFreqScale = 1.0f;
    float ropeFreqScaleSwa = 1.0f;
    RopeScalingType ropeScalingType = RopeScalingType.LINEAR;
    float ropeScalingAlpha = 0f;
    float ropeScalingFactor = 0f;
    float ropeScalingMixedB = Float.NaN;
    boolean ropeFope = false;
    boolean ropeFopeSepHead = false;
    int ropeFopeNumInvFreq = 0;
    float ropeFopeInitFactor = 0f;
    float ropeAttnFactor = 1.0f;
    boolean moePerGroupTopK = false;
    int nCtxOrigYarn = 0;
    boolean ropeFinetuned = false;
    float yarnLogMul = 0f;
    float yarnExtFactor = -1.0f;
    float yarnAttnFactor = 1.0f;
    float yarnBetaFast = 32.0f;
    float yarnBetaSlow = 1.0f;
    int[] ropeSections = new int[4];
    int ropeSectionOrder = ModelHparams.SECTION_ORDER_TYXZ;
    MropeTextPositions mropeTextPositions = MropeTextPositions.PPP0;

    // ---- dual-chunk attention ----
    /** {@code attention.dual_chunk.chunk_size}; 0 when the model has no dual-chunk attention. */
    int dualChunkChunkSize = 0;
    int dualChunkLocalSize = 0;
    int dualChunkOriginalMaxPositionEmbeddings = 0;
    boolean dualChunkSparseEnabled = false;
    int dualChunkSparseThreshold = 32768;
    int dualChunkSparseLastQ = 64;

    // ---- recurrent and linear attention ----
    int ssmDConv = 0;
    int ssmDInner = 0;
    int ssmDState = 0;
    int ssmDtRank = 0;
    int ssmNGroup = 0;
    boolean ssmDtBcRms = false;
    float ssmNormEps = 0f;
    float ssmBetaScale = 1f;
    int ssmGroupNormSize = 1;
    boolean ssmLinearSilu = false;
    boolean ssmQkNorm = false;
    boolean ssmQScale = false;
    int ssmRopeDims = 0;
    GdnHeadMap gdnHeadMap = null;
    int headLinearAttn = 0;
    int kdaHeadDim = 0;
    boolean kdaSafeGate = false;
    float kdaGateLowerBound = Float.NEGATIVE_INFINITY;
    KdaGateMode kdaGateMode = null;
    int wkvHeadSize = 0;
    int tokenShiftCount = 2;
    int rescaleEveryN = 0;
    int timeMixExtraDim = 0;
    int timeDecayExtraDim = 0;
    int nLoraDecay = 0;
    int nLoraIclr = 0;
    int nLoraValueResMix = 0;
    int nLoraGate = 0;
    int shortconvLCache = 0;
    int fullAttentionInterval = 0;
    int attnResBlockSize = 0;

    // ---- zamba2 shared transformer blocks ----
    int zamba2MemBlocks = 0;
    int zamba2AdapterRank = 0;
    boolean zamba2AttentionAdapter = false;
    boolean zamba2MemoryRope = false;

    // ---- Inkling relative attention and short convolutions ----
    int inklingRelExtent = 0;
    int inklingRelDim = 0;
    int inklingSconvKernel = 0;
    int inklingLogScalingFloor = 0;
    float inklingLogScalingAlpha = 0.1f;

    // ---- mixture of experts ----
    int nExpert = 0;
    int nExpertZero = 0;
    int nExpertShared = 0;
    int nExpertGroups = 0;
    int nGroupUsed = 0;
    int nGroupExperts = 0;
    int nFfShexp = 0;
    int nFfChexp = 0;
    int nLayerDenseLead = 0;
    int nMoeLayerStep = 0;
    int moeEveryN = 0;
    int moeLatentSize = 0;
    int nExpertLatent = 0;
    float expertGroupScale = 0.05f;
    float expertWeightsScale = 0f;
    boolean expertWeightsNorm = false;
    GatingFunc gating = null;

    // ---- sparse attention, compressed attention and hyper-connections ----
    int indexerNHead = 0;
    int indexerHeadSize = 0;
    int indexerTopK = 0;
    int indexerKpool = 0;
    boolean indexerKpoolSelectTail = true;
    int indexerKpoolRow = 3;
    boolean indexerKpoolByOrder = false;
    int indexerBlockSize = 0;
    int indexerLocalBlocks = 0;
    /** Step5: rotary width of each 256-wide indexer head and the indexer provider group count. */
    int indexerRopeDim = 0;
    int indexerGroupCount = 0;
    /** Step5: residual accumulator kept in F32 and RMS norm weights stored as {@code 1 + weight}. */
    boolean fp32Residual;
    boolean zeroCenteredNorms;
    int dsv4OGroupCount = 0;
    int dsv4OLoraRank = 0;
    int dsv4HcMult = 0;
    int dsv4HcSinkhornIters = 0;
    int dsv4HashLayerCount = 0;
    float dsv4CompressRopeBase = 0f;
    float dsv4HcEps = 0f;
    int dsv41CandidateSourceLayer = -1;
    int dsv41CandidateTopKBlocks = 0;
    int dsv41CandidateBlockSize = 0;
    int engramPadTokenId = -1;
    int[] engramTokenMap = new int[0];
    int hcLowRank = 0;
    float hcMagnitude = 0f;

    // ---- per-layer embeddings and altup ----
    int pleNgramSize = 0;
    int pleHeadsPerNgram = 0;
    int pleConvKernel = 0;
    int pleNHeads = 0;
    int pleHeadDim = 0;
    int pleEosTokenId = 0;
    int pleImageTokenId = 0;
    int ngramVocabSize = 0;
    int ngramNeighborCount = 0;
    int ngramSplitCount = 0;
    int ngramEosTokenId = -1;
    long[] pleLayerMultipliers = new long[0];
    long[] pleHeadOffsets = new long[0];
    long[] pleHeadVocabSizes = new long[0];
    int nAltup = 4;
    int iAltupAct = 0;
    int laurelRank = 64;
    int nEmbdAltup = 256;

    // ---- scaling ----
    float logitScale = 0f;
    boolean logitScaleDivides = false;
    float residualScale = 0f;
    float embeddingScale = 0f;
    float outputHiddenScale = 0f;
    float situBeta = 1.0f;
    float situLinearBeta = 0f;

    // ---- control flow ----
    int numLoops = 1;
    boolean skipLoopFinalNorm = false;
    int hrmLayersPerStack = 0;
    int hrmHCycles = 0;
    int hrmLCycles = 0;
    boolean hrmPrefixLm = false;
    /** Repeat-block schedule: 0 disables; otherwise the block [nInput, nInput + nRecurrent) runs nLoops times. */
    int repeatLoops = 0;
    int repeatInputLayers = 0;
    int repeatRecurrentLayers = 0;
    int repeatOutputLayers = 0;
    boolean repeatJustNorm = false;
    float repeatJustNormEps = 1e-6f;
    /** Dory's {@code ngpt_alpha_init / initializer_range}, the slerp-residual alpha scale. */
    float slerpAlphaScale = 0f;
    /** IQuestLoopCoder: loop passes after the first gate a global (loop 0) and a local (own) attention. */
    boolean repeatDualAttention = false;
    int repeatLocalWindow = 0;

    // ---- heads, encoders and pooling ----
    PoolingKind pooling = PoolingKind.NONE;
    PoolingKind poolingCls = null;
    int nClsOut = 1;
    String[] classifierLabels = new String[0];
    ClassifierScope classifierScope;
    String[] decisionLabels = new String[0];
    String decisionType = null;
    int decisionMaxHeadTokens = 0;
    Map<String, Float> decisionTemperatures = Collections.emptyMap();
    Map<Integer, float[]> decisionScoreBias = Collections.emptyMap();
    int decisionRoutingLayers = 0;
    int decisionJointLayers = 0;
    int decisionHeadCount = 0;
    ActOp classifierActivation = ActOp.TANH;
    PoolerAct poolerActivation = null;
    boolean classifierTokenDirect = false;
    String classifierSparse = null;
    int sparseClsToken = 101;
    int sparseSepToken = 102;
    int classifierBosToken = -1;
    int classifierEosToken = -1;
    int classifierStepTagId = -1;
    boolean poolingNormalize = false;
    String hiddenActivation = null;
    int decStartTokenId = -1;
    int decNLayer = 0;
    int dense2FeatIn = 0;
    int dense2FeatOut = 0;
    int dense3FeatIn = 0;
    int dense3FeatOut = 0;
    boolean diffusionShiftLogits = true;

    // ---- drafters and adapters ----
    int[] targetLayers = new int[0];
    int targetHiddenSize = 0;
    int dflashBlockSize = 0;
    int dflashConvKernelSize = 0;
    int dflashConvGroupSize = 0;
    int dflashSelectorRank = 0;
    int dflashSelectorTopK = 0;
    int lilicorrHiddenSize = 0;
    int lilicorrCandidateTopK = 0;
    int lilicorrNumHeads = 0;
    int lilicorrLayers = 0;
    float lilicorrVectorEps = 0f;
    float lilicorrLogitScale = 0f;
    int lilicorrFactorDim = 0;
    int lilicorrMlpHiddenSize = 0;
    int adapterCount = 0;
    int adapterLoraRank = 0;
    float adapterRouterGain = 0f;
    int[] adapterTokenIdsActivate = new int[0];
    int[] adapterTokenIdsSubstitute = new int[0];

    // ---- per-layer arrays (length nLayerAll) ----
    int[] nHeadArr;
    int[] nHeadKvArr;
    int[] nFfArr;
    int[] nFfExpArr;
    int[] nExpertUsedArr;
    boolean[] swaArr;
    boolean[] mlpArr;
    boolean[] ropeArr;
    boolean[] recurrentArr;
    boolean[] hybridArr;
    int[] sharedBlockArr;
    boolean[] indexerFullArr;
    boolean[] pleArr;
    int[] compressRatioArr;
    int[] dsv41KvSourceArr;
    int[] dsv41TopkSourceArr;
    int[] deepstackMappingArr;
    float[] xieluAlphaNArr;
    float[] xieluAlphaPArr;
    float[] xieluBetaArr;
    float[] xieluEpsArr;
    float[] swigluClampExpArr;
    float[] swigluClampShexpArr;

    // ---- identity accessors ----
    public ModelHparamsBuilder name(String v) { this.name = v; return this; }
    public String name() { return name; }
    public ModelHparamsBuilder fileType(int v) { this.fileType = v; return this; }
    public int fileType() { return fileType; }

    // ---- sizes accessors ----
    public ModelHparamsBuilder nVocab(int v) { this.nVocab = v; return this; }
    public int nVocab() { return nVocab; }
    public ModelHparamsBuilder nCtxTrain(int v) { this.nCtxTrain = v; return this; }
    public int nCtxTrain() { return nCtxTrain; }
    public ModelHparamsBuilder nEmbd(int v) { this.nEmbd = v; return this; }
    public int nEmbd() { return nEmbd; }
    public ModelHparamsBuilder nEmbdInpImpl(int v) { this.nEmbdInpImpl = v; return this; }
    public int nEmbdInpImpl() { return nEmbdInpImpl; }
    public ModelHparamsBuilder nEmbdInpEncImpl(int v) { this.nEmbdInpEncImpl = v; return this; }
    public int nEmbdInpEncImpl() { return nEmbdInpEncImpl; }
    public ModelHparamsBuilder nEmbdOutImpl(int v) { this.nEmbdOutImpl = v; return this; }
    public int nEmbdOutImpl() { return nEmbdOutImpl; }
    public ModelHparamsBuilder nEmbdPerLayer(int v) { this.nEmbdPerLayer = v; return this; }
    public int nEmbdPerLayer() { return nEmbdPerLayer; }
    public ModelHparamsBuilder nLayerDecision(int v) { this.nLayerDecision = v; return this; }
    public int nLayerDecision() { return nLayerDecision; }
    public ModelHparamsBuilder routerLayer(int v) { this.routerLayer = v; return this; }
    public int routerLayer() { return routerLayer; }
    public ModelHparamsBuilder posnetEmbd(int v) { this.posnetEmbd = v; return this; }
    public int posnetEmbd() { return posnetEmbd; }
    public ModelHparamsBuilder posnetLayers(int v) { this.posnetLayers = v; return this; }
    public int posnetLayers() { return posnetLayers; }
    public ModelHparamsBuilder convnextEmbd(int v) { this.convnextEmbd = v; return this; }
    public int convnextEmbd() { return convnextEmbd; }
    public ModelHparamsBuilder convnextLayers(int v) { this.convnextLayers = v; return this; }
    public int convnextLayers() { return convnextLayers; }
    public ModelHparamsBuilder nDeepstackLayers(int v) { this.nDeepstackLayers = v; return this; }
    public int nDeepstackLayers() { return nDeepstackLayers; }
    public ModelHparamsBuilder nTokensDelay(int v) { this.nTokensDelay = v; return this; }
    public int nTokensDelay() { return nTokensDelay; }

    // ---- head dimensions accessors ----
    public ModelHparamsBuilder headKFull(int v) { this.headKFull = v; return this; }
    public int headKFull() { return headKFull; }
    public ModelHparamsBuilder headVFull(int v) { this.headVFull = v; return this; }
    public int headVFull() { return headVFull; }
    public ModelHparamsBuilder headKSwa(int v) { this.headKSwa = v; return this; }
    public int headKSwa() { return headKSwa; }
    public ModelHparamsBuilder headVSwa(int v) { this.headVSwa = v; return this; }
    public int headVSwa() { return headVSwa; }
    public ModelHparamsBuilder nRotFull(int v) { this.nRotFull = v; return this; }
    public int nRotFull() { return nRotFull; }
    public ModelHparamsBuilder nRotSwa(int v) { this.nRotSwa = v; return this; }
    public int nRotSwa() { return nRotSwa; }
    public ModelHparamsBuilder headKMlaImpl(int v) { this.headKMlaImpl = v; return this; }
    public int headKMlaImpl() { return headKMlaImpl; }
    public ModelHparamsBuilder headVMlaImpl(int v) { this.headVMlaImpl = v; return this; }
    public int headVMlaImpl() { return headVMlaImpl; }
    public ModelHparamsBuilder headKMlaSwa(int v) { this.headKMlaSwa = v; return this; }
    public int headKMlaSwa() { return headKMlaSwa; }
    public ModelHparamsBuilder headVMlaSwa(int v) { this.headVMlaSwa = v; return this; }
    public int headVMlaSwa() { return headVMlaSwa; }
    public ModelHparamsBuilder nLoraQ(int v) { this.nLoraQ = v; return this; }
    public int nLoraQ() { return nLoraQ; }
    public ModelHparamsBuilder nLoraKv(int v) { this.nLoraKv = v; return this; }
    public int nLoraKv() { return nLoraKv; }
    public ModelHparamsBuilder nLoraKvSwa(int v) { this.nLoraKvSwa = v; return this; }
    public int nLoraKvSwa() { return nLoraKvSwa; }

    // ---- norms accessors ----
    public ModelHparamsBuilder normEps(float v) { this.normEps = v; return this; }
    public float normEps() { return normEps; }
    public ModelHparamsBuilder normRmsEps(float v) { this.normRmsEps = v; return this; }
    public float normRmsEps() { return normRmsEps; }
    public ModelHparamsBuilder normGroupEps(float v) { this.normGroupEps = v; return this; }
    public float normGroupEps() { return normGroupEps; }
    public ModelHparamsBuilder postNormEps(float v) { this.postNormEps = v; return this; }
    public float postNormEps() { return postNormEps; }
    public ModelHparamsBuilder nNormGroups(int v) { this.nNormGroups = v; return this; }
    public int nNormGroups() { return nNormGroups; }
    public ModelHparamsBuilder swinNorm(boolean v) { this.swinNorm = v; return this; }
    public boolean swinNorm() { return swinNorm; }
    public ModelHparamsBuilder normBeforeResidual(boolean v) { this.normBeforeResidual = v; return this; }
    public boolean normBeforeResidual() { return normBeforeResidual; }
    public ModelHparamsBuilder normBeforeFc(boolean v) { this.normBeforeFc = v; return this; }
    public boolean normBeforeFc() { return normBeforeFc; }
    public ModelHparamsBuilder useParallelResidual(boolean v) { this.useParallelResidual = v; return this; }
    public boolean useParallelResidual() { return useParallelResidual; }

    // ---- attention accessors ----
    public ModelHparamsBuilder causal(boolean v) { this.causal = v; return this; }
    public boolean causal() { return causal; }
    public ModelHparamsBuilder nonCausalType(int v) { this.nonCausalType = v; return this; }
    public int nonCausalType() { return nonCausalType; }
    public ModelHparamsBuilder attentionScale(float v) { this.attentionScale = v; return this; }
    public float attentionScale() { return attentionScale; }
    public ModelHparamsBuilder xsaEps(float v) { this.xsaEps = v; return this; }
    public float xsaEps() { return xsaEps; }
    public ModelHparamsBuilder muddTapCount(int v) { this.muddTapCount = v; return this; }
    public int muddTapCount() { return muddTapCount; }
    public ModelHparamsBuilder muddFf(int v) { this.muddFf = v; return this; }
    public int muddFf() { return muddFf; }
    public ModelHparamsBuilder muddTapIdxArray(int[] v) {
        // a flat [n_layer][n_mudd_taps] table, not a per-layer array
        int expected = this.nLayerAll * this.muddTapCount;
        if (v == null || v.length != expected) {
            throw new IllegalArgumentException("muddTapIdx needs " + expected + " entries (block count x tap count),"
                    + " got " + (v == null ? 0 : v.length));
        }
        this.muddTapIdx = v.clone();
        return this;
    }
    public int[] muddTapIdxArray() { return muddTapIdx.clone(); }
    public ModelHparamsBuilder finalLogitSigmoidCapping(float[] v) {
        if (v == null || v.length != 3) {
            throw new IllegalArgumentException("finalLogitSigmoidCapping needs scale, shift and temperature, got "
                    + (v == null ? "null" : v.length + " values"));
        }
        this.finalLogitSigmoidCapping = v.clone();
        return this;
    }
    public float[] finalLogitSigmoidCapping() { return finalLogitSigmoidCapping.clone(); }
    public ModelHparamsBuilder attnOutScale(float v) { this.attnOutScale = v; return this; }
    public float attnOutScale() { return attnOutScale; }
    public ModelHparamsBuilder attnValueScale(float v) { this.attnValueScale = v; return this; }
    public float attnValueScale() { return attnValueScale; }
    public ModelHparamsBuilder attnSoftcapEnabled(boolean v) { this.attnSoftcapEnabled = v; return this; }
    public boolean attnSoftcapEnabled() { return attnSoftcapEnabled; }
    public ModelHparamsBuilder attnSoftcap(float v) { this.attnSoftcap = v; return this; }
    public float attnSoftcap() { return attnSoftcap; }
    public ModelHparamsBuilder routerSoftcap(float v) { this.routerSoftcap = v; return this; }
    public float routerSoftcap() { return routerSoftcap; }
    public ModelHparamsBuilder finalSoftcap(float v) { this.finalSoftcap = v; return this; }
    public float finalSoftcap() { return finalSoftcap; }
    public ModelHparamsBuilder clampKqv(float v) { this.clampKqv = v; return this; }
    public float clampKqv() { return clampKqv; }
    public ModelHparamsBuilder maxAlibiBias(float v) { this.maxAlibiBias = v; return this; }
    public float maxAlibiBias() { return maxAlibiBias; }
    public ModelHparamsBuilder useAlibi(boolean v) { this.useAlibi = v; return this; }
    public boolean useAlibi() { return useAlibi; }
    public ModelHparamsBuilder alibiSqrt(boolean v) { this.alibiSqrt = v; return this; }
    public boolean alibiSqrt() { return alibiSqrt; }
    public ModelHparamsBuilder nRelAttnBuckets(int v) { this.nRelAttnBuckets = v; return this; }
    public int nRelAttnBuckets() { return nRelAttnBuckets; }
    public ModelHparamsBuilder attnTempLength(int v) { this.attnTempLength = v; return this; }
    public int attnTempLength() { return attnTempLength; }
    public ModelHparamsBuilder attnTempFloorScale(int v) { this.attnTempFloorScale = v; return this; }
    public int attnTempFloorScale() { return attnTempFloorScale; }
    public ModelHparamsBuilder attnTempScale(float v) { this.attnTempScale = v; return this; }
    public float attnTempScale() { return attnTempScale; }
    public ModelHparamsBuilder attnTempOffset(float v) { this.attnTempOffset = v; return this; }
    public float attnTempOffset() { return attnTempOffset; }
    public ModelHparamsBuilder attnTempPow(boolean v) { this.attnTempPow = v; return this; }
    public boolean attnTempPow() { return attnTempPow; }
    public ModelHparamsBuilder noRopeLayerStep(int v) { this.noRopeLayerStep = v; return this; }
    public int noRopeLayerStep() { return noRopeLayerStep; }
    public ModelHparamsBuilder useKqNorm(boolean v) { this.useKqNorm = v; return this; }
    public boolean useKqNorm() { return useKqNorm; }
    public ModelHparamsBuilder nLayerKvFromStart(int v) { this.nLayerKvFromStart = v; return this; }
    public int nLayerKvFromStart() { return nLayerKvFromStart; }
    public ModelHparamsBuilder sharedKvLayers(int v) { this.sharedKvLayers = v; return this; }
    public int sharedKvLayers() { return sharedKvLayers; }
    public ModelHparamsBuilder nValueExpert(int v) { this.nValueExpert = v; return this; }
    public int nValueExpert() { return nValueExpert; }
    public ModelHparamsBuilder nValueExpertUsed(int v) { this.nValueExpertUsed = v; return this; }
    public int nValueExpertUsed() { return nValueExpertUsed; }

    // ---- sliding window accessors ----
    public ModelHparamsBuilder swaType(SwaType v) { this.swaType = v; return this; }
    public SwaType swaType() { return swaType; }
    public ModelHparamsBuilder nSwa(int v) { this.nSwa = v; return this; }
    public int nSwa() { return nSwa; }

    /**
     * Per-layer sliding windows, one entry per block ({@code 0} = full attention). When set, it overrides the
     * scalar {@link #nSwa()} for each layer; {@link #nSwa()} should then hold the largest one.
     *
     * @throws IllegalArgumentException when the array does not hold one entry per block
     */
    public ModelHparamsBuilder nSwaArray(int[] v) { this.swaWindowArr = layerArray("nSwa", v).clone(); return this; }

    /** Learned static sink tokens per attention layer; 0 without them. */
    public ModelHparamsBuilder sinkTokenCount(int v) { this.sinkTokenCount = v; return this; }
    public int sinkTokenCount() { return sinkTokenCount; }

    // ---- rope accessors ----
    public ModelHparamsBuilder ropeType(RopeType v) { this.ropeType = v; return this; }
    public RopeType ropeType() { return ropeType; }
    public ModelHparamsBuilder ropeFreqBase(float v) { this.ropeFreqBase = v; return this; }
    public float ropeFreqBase() { return ropeFreqBase; }
    public ModelHparamsBuilder ropeFreqBaseSwa(float v) { this.ropeFreqBaseSwa = v; return this; }
    public float ropeFreqBaseSwa() { return ropeFreqBaseSwa; }
    public ModelHparamsBuilder ropeFreqBaseSpatial(float v) { this.ropeFreqBaseSpatial = v; return this; }
    public float ropeFreqBaseSpatial() { return ropeFreqBaseSpatial; }
    public ModelHparamsBuilder ropeFreqScale(float v) { this.ropeFreqScale = v; return this; }
    public float ropeFreqScale() { return ropeFreqScale; }
    public ModelHparamsBuilder ropeFreqScaleSwa(float v) { this.ropeFreqScaleSwa = v; return this; }
    public float ropeFreqScaleSwa() { return ropeFreqScaleSwa; }
    public ModelHparamsBuilder ropeScalingType(RopeScalingType v) { this.ropeScalingType = v; return this; }
    public RopeScalingType ropeScalingType() { return ropeScalingType; }
    public ModelHparamsBuilder ropeScalingAlpha(float v) { this.ropeScalingAlpha = v; return this; }
    public float ropeScalingAlpha() { return ropeScalingAlpha; }
    public ModelHparamsBuilder ropeScalingFactor(float v) { this.ropeScalingFactor = v; return this; }
    public float ropeScalingFactor() { return ropeScalingFactor; }
    public ModelHparamsBuilder ropeScalingMixedB(float v) { this.ropeScalingMixedB = v; return this; }
    public float ropeScalingMixedB() { return ropeScalingMixedB; }
    public ModelHparamsBuilder ropeFope(boolean v) { this.ropeFope = v; return this; }
    public boolean ropeFope() { return ropeFope; }
    public ModelHparamsBuilder ropeFopeSepHead(boolean v) { this.ropeFopeSepHead = v; return this; }
    public boolean ropeFopeSepHead() { return ropeFopeSepHead; }
    public ModelHparamsBuilder ropeFopeNumInvFreq(int v) { this.ropeFopeNumInvFreq = v; return this; }
    public int ropeFopeNumInvFreq() { return ropeFopeNumInvFreq; }
    public ModelHparamsBuilder ropeFopeInitFactor(float v) { this.ropeFopeInitFactor = v; return this; }
    public float ropeFopeInitFactor() { return ropeFopeInitFactor; }
    public ModelHparamsBuilder ropeAttnFactor(float v) { this.ropeAttnFactor = v; return this; }
    public float ropeAttnFactor() { return ropeAttnFactor; }
    public ModelHparamsBuilder moePerGroupTopK(boolean v) { this.moePerGroupTopK = v; return this; }
    public boolean moePerGroupTopK() { return moePerGroupTopK; }
    public ModelHparamsBuilder nCtxOrigYarn(int v) { this.nCtxOrigYarn = v; return this; }
    public int nCtxOrigYarn() { return nCtxOrigYarn; }

    // ---- dual-chunk attention accessors ----
    public ModelHparamsBuilder dualChunkChunkSize(int v) { this.dualChunkChunkSize = v; return this; }
    public int dualChunkChunkSize() { return dualChunkChunkSize; }
    public ModelHparamsBuilder dualChunkLocalSize(int v) { this.dualChunkLocalSize = v; return this; }
    public int dualChunkLocalSize() { return dualChunkLocalSize; }
    public ModelHparamsBuilder dualChunkOriginalMaxPositionEmbeddings(int v) {
        this.dualChunkOriginalMaxPositionEmbeddings = v;
        return this;
    }
    public int dualChunkOriginalMaxPositionEmbeddings() { return dualChunkOriginalMaxPositionEmbeddings; }
    public ModelHparamsBuilder dualChunkSparseEnabled(boolean v) { this.dualChunkSparseEnabled = v; return this; }
    public boolean dualChunkSparseEnabled() { return dualChunkSparseEnabled; }
    public ModelHparamsBuilder dualChunkSparseThreshold(int v) { this.dualChunkSparseThreshold = v; return this; }
    public int dualChunkSparseThreshold() { return dualChunkSparseThreshold; }
    public ModelHparamsBuilder dualChunkSparseLastQ(int v) { this.dualChunkSparseLastQ = v; return this; }
    public int dualChunkSparseLastQ() { return dualChunkSparseLastQ; }
    public ModelHparamsBuilder ropeFinetuned(boolean v) { this.ropeFinetuned = v; return this; }
    public boolean ropeFinetuned() { return ropeFinetuned; }
    public ModelHparamsBuilder yarnLogMul(float v) { this.yarnLogMul = v; return this; }
    public float yarnLogMul() { return yarnLogMul; }
    public ModelHparamsBuilder yarnExtFactor(float v) { this.yarnExtFactor = v; return this; }
    public float yarnExtFactor() { return yarnExtFactor; }
    public ModelHparamsBuilder yarnAttnFactor(float v) { this.yarnAttnFactor = v; return this; }
    public float yarnAttnFactor() { return yarnAttnFactor; }
    public ModelHparamsBuilder yarnBetaFast(float v) { this.yarnBetaFast = v; return this; }
    public float yarnBetaFast() { return yarnBetaFast; }
    public ModelHparamsBuilder yarnBetaSlow(float v) { this.yarnBetaSlow = v; return this; }
    public float yarnBetaSlow() { return yarnBetaSlow; }
    public ModelHparamsBuilder ropeSections(int[] v) {
        if (Objects.requireNonNull(v, "ropeSections").length != ModelHparams.MROPE_SECTIONS) {
            throw new IllegalArgumentException("ropeSections needs " + ModelHparams.MROPE_SECTIONS + " entries, got " + v.length);
        }
        this.ropeSections = v.clone();
        return this;
    }
    public int[] ropeSections() { return ropeSections.clone(); }
    public ModelHparamsBuilder ropeSectionOrder(int v) { this.ropeSectionOrder = v; return this; }
    public int ropeSectionOrder() { return ropeSectionOrder; }
    public ModelHparamsBuilder mropeTextPositions(MropeTextPositions v) { this.mropeTextPositions = v; return this; }
    public MropeTextPositions mropeTextPositions() { return mropeTextPositions; }

    // ---- recurrent and linear attention accessors ----
    public ModelHparamsBuilder ssmDConv(int v) { this.ssmDConv = v; return this; }
    public int ssmDConv() { return ssmDConv; }
    public ModelHparamsBuilder ssmDInner(int v) { this.ssmDInner = v; return this; }
    public int ssmDInner() { return ssmDInner; }
    public ModelHparamsBuilder ssmDState(int v) { this.ssmDState = v; return this; }
    public int ssmDState() { return ssmDState; }
    public ModelHparamsBuilder ssmDtRank(int v) { this.ssmDtRank = v; return this; }
    public int ssmDtRank() { return ssmDtRank; }
    public ModelHparamsBuilder ssmNGroup(int v) { this.ssmNGroup = v; return this; }
    public int ssmNGroup() { return ssmNGroup; }
    public ModelHparamsBuilder ssmDtBcRms(boolean v) { this.ssmDtBcRms = v; return this; }
    public boolean ssmDtBcRms() { return ssmDtBcRms; }
    public ModelHparamsBuilder ssmNormEps(float v) { this.ssmNormEps = v; return this; }
    public float ssmNormEps() { return ssmNormEps; }
    public ModelHparamsBuilder ssmBetaScale(float v) { this.ssmBetaScale = v; return this; }
    public float ssmBetaScale() { return ssmBetaScale; }
    /** Group count of the lightning-attention output norm ({@code ssm.group_norm_size}). */
    public ModelHparamsBuilder ssmGroupNormSize(int v) { this.ssmGroupNormSize = v; return this; }
    public int ssmGroupNormSize() { return ssmGroupNormSize; }
    /** SiLU on the fused lightning-attention Q/K/V projection ({@code ssm.linear_silu}). */
    public ModelHparamsBuilder ssmLinearSilu(boolean v) { this.ssmLinearSilu = v; return this; }
    public boolean ssmLinearSilu() { return ssmLinearSilu; }
    /** Per-head RMS norm of the lightning-attention Q and K ({@code ssm.qk_norm}). */
    public ModelHparamsBuilder ssmQkNorm(boolean v) { this.ssmQkNorm = v; return this; }
    public boolean ssmQkNorm() { return ssmQkNorm; }
    /** {@code head_size^-0.5} scale on the lightning-attention Q ({@code ssm.q_scale}). */
    public ModelHparamsBuilder ssmQScale(boolean v) { this.ssmQScale = v; return this; }
    public boolean ssmQScale() { return ssmQScale; }
    /** Rotary dims of the lightning-attention heads ({@code ssm.rope_dimension_count}). */
    public ModelHparamsBuilder ssmRopeDims(int v) { this.ssmRopeDims = v; return this; }
    public int ssmRopeDims() { return ssmRopeDims; }
    public ModelHparamsBuilder gdnHeadMap(GdnHeadMap v) { this.gdnHeadMap = v; return this; }
    public GdnHeadMap gdnHeadMap() { return gdnHeadMap; }
    public ModelHparamsBuilder headLinearAttn(int v) { this.headLinearAttn = v; return this; }
    public int headLinearAttn() { return headLinearAttn; }
    public ModelHparamsBuilder kdaHeadDim(int v) { this.kdaHeadDim = v; return this; }
    public int kdaHeadDim() { return kdaHeadDim; }
    public ModelHparamsBuilder kdaSafeGate(boolean v) { this.kdaSafeGate = v; return this; }
    public boolean kdaSafeGate() { return kdaSafeGate; }
    public ModelHparamsBuilder kdaGateLowerBound(float v) { this.kdaGateLowerBound = v; return this; }
    public float kdaGateLowerBound() { return kdaGateLowerBound; }
    public ModelHparamsBuilder kdaGateMode(KdaGateMode v) { this.kdaGateMode = v; return this; }
    public KdaGateMode kdaGateMode() { return kdaGateMode; }
    public ModelHparamsBuilder wkvHeadSize(int v) { this.wkvHeadSize = v; return this; }
    public int wkvHeadSize() { return wkvHeadSize; }
    public ModelHparamsBuilder tokenShiftCount(int v) { this.tokenShiftCount = v; return this; }
    public int tokenShiftCount() { return tokenShiftCount; }
    public ModelHparamsBuilder rescaleEveryN(int v) { this.rescaleEveryN = v; return this; }
    public int rescaleEveryN() { return rescaleEveryN; }
    public ModelHparamsBuilder timeMixExtraDim(int v) { this.timeMixExtraDim = v; return this; }
    public int timeMixExtraDim() { return timeMixExtraDim; }
    public ModelHparamsBuilder timeDecayExtraDim(int v) { this.timeDecayExtraDim = v; return this; }
    public int timeDecayExtraDim() { return timeDecayExtraDim; }
    public ModelHparamsBuilder nLoraDecay(int v) { this.nLoraDecay = v; return this; }
    public int nLoraDecay() { return nLoraDecay; }
    public ModelHparamsBuilder nLoraIclr(int v) { this.nLoraIclr = v; return this; }
    public int nLoraIclr() { return nLoraIclr; }
    public ModelHparamsBuilder nLoraValueResMix(int v) { this.nLoraValueResMix = v; return this; }
    public int nLoraValueResMix() { return nLoraValueResMix; }
    public ModelHparamsBuilder nLoraGate(int v) { this.nLoraGate = v; return this; }
    public int nLoraGate() { return nLoraGate; }
    public ModelHparamsBuilder shortconvLCache(int v) { this.shortconvLCache = v; return this; }
    public int shortconvLCache() { return shortconvLCache; }
    public ModelHparamsBuilder fullAttentionInterval(int v) { this.fullAttentionInterval = v; return this; }
    public int fullAttentionInterval() { return fullAttentionInterval; }
    public ModelHparamsBuilder attnResBlockSize(int v) { this.attnResBlockSize = v; return this; }
    public int attnResBlockSize() { return attnResBlockSize; }
    public ModelHparamsBuilder inklingRelExtent(int v) { this.inklingRelExtent = v; return this; }
    public int inklingRelExtent() { return inklingRelExtent; }
    public ModelHparamsBuilder inklingRelDim(int v) { this.inklingRelDim = v; return this; }
    public int inklingRelDim() { return inklingRelDim; }
    public ModelHparamsBuilder inklingSconvKernel(int v) { this.inklingSconvKernel = v; return this; }
    public int inklingSconvKernel() { return inklingSconvKernel; }
    public ModelHparamsBuilder inklingLogScalingFloor(int v) { this.inklingLogScalingFloor = v; return this; }
    public int inklingLogScalingFloor() { return inklingLogScalingFloor; }
    public ModelHparamsBuilder inklingLogScalingAlpha(float v) { this.inklingLogScalingAlpha = v; return this; }
    public float inklingLogScalingAlpha() { return inklingLogScalingAlpha; }

    // ---- zamba2 shared transformer block accessors ----
    public ModelHparamsBuilder zamba2MemBlocks(int v) { this.zamba2MemBlocks = v; return this; }
    public int zamba2MemBlocks() { return zamba2MemBlocks; }
    public ModelHparamsBuilder zamba2AdapterRank(int v) { this.zamba2AdapterRank = v; return this; }
    public int zamba2AdapterRank() { return zamba2AdapterRank; }
    public ModelHparamsBuilder zamba2AttentionAdapter(boolean v) { this.zamba2AttentionAdapter = v; return this; }
    public boolean zamba2AttentionAdapter() { return zamba2AttentionAdapter; }
    public ModelHparamsBuilder zamba2MemoryRope(boolean v) { this.zamba2MemoryRope = v; return this; }
    public boolean zamba2MemoryRope() { return zamba2MemoryRope; }

    // ---- mixture of experts accessors ----
    public ModelHparamsBuilder nExpert(int v) { this.nExpert = v; return this; }
    public int nExpert() { return nExpert; }
    public ModelHparamsBuilder nExpertZero(int v) { this.nExpertZero = v; return this; }
    public int nExpertZero() { return nExpertZero; }
    public ModelHparamsBuilder nExpertShared(int v) { this.nExpertShared = v; return this; }
    public int nExpertShared() { return nExpertShared; }
    public ModelHparamsBuilder nExpertGroups(int v) { this.nExpertGroups = v; return this; }
    public int nExpertGroups() { return nExpertGroups; }
    public ModelHparamsBuilder nGroupUsed(int v) { this.nGroupUsed = v; return this; }
    public int nGroupUsed() { return nGroupUsed; }
    public ModelHparamsBuilder nGroupExperts(int v) { this.nGroupExperts = v; return this; }
    public int nGroupExperts() { return nGroupExperts; }
    public ModelHparamsBuilder nFfShexp(int v) { this.nFfShexp = v; return this; }
    public int nFfShexp() { return nFfShexp; }
    public ModelHparamsBuilder nFfChexp(int v) { this.nFfChexp = v; return this; }
    public int nFfChexp() { return nFfChexp; }
    public ModelHparamsBuilder nLayerDenseLead(int v) { this.nLayerDenseLead = v; return this; }
    public int nLayerDenseLead() { return nLayerDenseLead; }
    public ModelHparamsBuilder nMoeLayerStep(int v) { this.nMoeLayerStep = v; return this; }
    public int nMoeLayerStep() { return nMoeLayerStep; }
    public ModelHparamsBuilder moeEveryN(int v) { this.moeEveryN = v; return this; }
    public int moeEveryN() { return moeEveryN; }
    public ModelHparamsBuilder moeLatentSize(int v) { this.moeLatentSize = v; return this; }
    public int moeLatentSize() { return moeLatentSize; }
    public ModelHparamsBuilder nExpertLatent(int v) { this.nExpertLatent = v; return this; }
    public int nExpertLatent() { return nExpertLatent; }
    public ModelHparamsBuilder expertGroupScale(float v) { this.expertGroupScale = v; return this; }
    public float expertGroupScale() { return expertGroupScale; }
    public ModelHparamsBuilder expertWeightsScale(float v) { this.expertWeightsScale = v; return this; }
    public float expertWeightsScale() { return expertWeightsScale; }
    public ModelHparamsBuilder expertWeightsNorm(boolean v) { this.expertWeightsNorm = v; return this; }
    public boolean expertWeightsNorm() { return expertWeightsNorm; }
    public ModelHparamsBuilder gating(GatingFunc v) { this.gating = v; return this; }
    public GatingFunc gating() { return gating; }

    // ---- sparse attention, compressed attention and hyper-connections accessors ----
    public ModelHparamsBuilder indexerNHead(int v) { this.indexerNHead = v; return this; }
    public int indexerNHead() { return indexerNHead; }
    public ModelHparamsBuilder indexerHeadSize(int v) { this.indexerHeadSize = v; return this; }
    public int indexerHeadSize() { return indexerHeadSize; }
    public ModelHparamsBuilder indexerTopK(int v) { this.indexerTopK = v; return this; }
    public int indexerTopK() { return indexerTopK; }
    public ModelHparamsBuilder indexerKpool(int v) { this.indexerKpool = v; return this; }
    public int indexerKpool() { return indexerKpool; }
    public ModelHparamsBuilder indexerKpoolSelectTail(boolean v) { this.indexerKpoolSelectTail = v; return this; }
    public boolean indexerKpoolSelectTail() { return indexerKpoolSelectTail; }
    public ModelHparamsBuilder indexerKpoolRow(int v) { this.indexerKpoolRow = v; return this; }
    public int indexerKpoolRow() { return indexerKpoolRow; }
    public ModelHparamsBuilder indexerKpoolByOrder(boolean v) { this.indexerKpoolByOrder = v; return this; }
    public boolean indexerKpoolByOrder() { return indexerKpoolByOrder; }
    public ModelHparamsBuilder indexerBlockSize(int v) { this.indexerBlockSize = v; return this; }
    public int indexerBlockSize() { return indexerBlockSize; }
    public ModelHparamsBuilder indexerLocalBlocks(int v) { this.indexerLocalBlocks = v; return this; }
    public int indexerLocalBlocks() { return indexerLocalBlocks; }
    public ModelHparamsBuilder indexerRopeDim(int v) { this.indexerRopeDim = v; return this; }
    public int indexerRopeDim() { return indexerRopeDim; }
    public ModelHparamsBuilder indexerGroupCount(int v) { this.indexerGroupCount = v; return this; }
    public int indexerGroupCount() { return indexerGroupCount; }
    public ModelHparamsBuilder fp32Residual(boolean v) { this.fp32Residual = v; return this; }
    public boolean fp32Residual() { return fp32Residual; }
    public ModelHparamsBuilder zeroCenteredNorms(boolean v) { this.zeroCenteredNorms = v; return this; }
    public boolean zeroCenteredNorms() { return zeroCenteredNorms; }
    public ModelHparamsBuilder dsv4OGroupCount(int v) { this.dsv4OGroupCount = v; return this; }
    public int dsv4OGroupCount() { return dsv4OGroupCount; }
    public ModelHparamsBuilder dsv4OLoraRank(int v) { this.dsv4OLoraRank = v; return this; }
    public int dsv4OLoraRank() { return dsv4OLoraRank; }
    public ModelHparamsBuilder dsv4HcMult(int v) { this.dsv4HcMult = v; return this; }
    public int dsv4HcMult() { return dsv4HcMult; }
    public ModelHparamsBuilder dsv4HcSinkhornIters(int v) { this.dsv4HcSinkhornIters = v; return this; }
    public int dsv4HcSinkhornIters() { return dsv4HcSinkhornIters; }
    public ModelHparamsBuilder dsv4HashLayerCount(int v) { this.dsv4HashLayerCount = v; return this; }
    public int dsv4HashLayerCount() { return dsv4HashLayerCount; }
    public ModelHparamsBuilder dsv4CompressRopeBase(float v) { this.dsv4CompressRopeBase = v; return this; }
    public float dsv4CompressRopeBase() { return dsv4CompressRopeBase; }
    public ModelHparamsBuilder dsv4HcEps(float v) { this.dsv4HcEps = v; return this; }
    public float dsv4HcEps() { return dsv4HcEps; }
    public ModelHparamsBuilder dsv41CandidateSourceLayer(int v) { this.dsv41CandidateSourceLayer = v; return this; }
    public int dsv41CandidateSourceLayer() { return dsv41CandidateSourceLayer; }
    public ModelHparamsBuilder dsv41CandidateTopKBlocks(int v) { this.dsv41CandidateTopKBlocks = v; return this; }
    public int dsv41CandidateTopKBlocks() { return dsv41CandidateTopKBlocks; }
    public ModelHparamsBuilder dsv41CandidateBlockSize(int v) { this.dsv41CandidateBlockSize = v; return this; }
    public int dsv41CandidateBlockSize() { return dsv41CandidateBlockSize; }
    public ModelHparamsBuilder engramPadTokenId(int v) { this.engramPadTokenId = v; return this; }
    public int engramPadTokenId() { return engramPadTokenId; }
    public ModelHparamsBuilder engramTokenMap(int[] v) { this.engramTokenMap = Objects.requireNonNull(v, "engramTokenMap").clone(); return this; }
    public int[] engramTokenMap() { return engramTokenMap.clone(); }
    public ModelHparamsBuilder hcLowRank(int v) { this.hcLowRank = v; return this; }
    public int hcLowRank() { return hcLowRank; }
    public ModelHparamsBuilder hcMagnitude(float v) { this.hcMagnitude = v; return this; }
    public float hcMagnitude() { return hcMagnitude; }

    // ---- per-layer embeddings and altup accessors ----
    public ModelHparamsBuilder pleNgramSize(int v) { this.pleNgramSize = v; return this; }
    public int pleNgramSize() { return pleNgramSize; }
    public ModelHparamsBuilder pleHeadsPerNgram(int v) { this.pleHeadsPerNgram = v; return this; }
    public int pleHeadsPerNgram() { return pleHeadsPerNgram; }
    public ModelHparamsBuilder pleConvKernel(int v) { this.pleConvKernel = v; return this; }
    public int pleConvKernel() { return pleConvKernel; }
    public ModelHparamsBuilder pleNHeads(int v) { this.pleNHeads = v; return this; }
    public int pleNHeads() { return pleNHeads; }
    public ModelHparamsBuilder pleHeadDim(int v) { this.pleHeadDim = v; return this; }
    public int pleHeadDim() { return pleHeadDim; }
    public ModelHparamsBuilder pleEosTokenId(int v) { this.pleEosTokenId = v; return this; }
    public int pleEosTokenId() { return pleEosTokenId; }
    public ModelHparamsBuilder pleImageTokenId(int v) { this.pleImageTokenId = v; return this; }
    public int pleImageTokenId() { return pleImageTokenId; }
    public ModelHparamsBuilder ngramVocabSize(int v) { this.ngramVocabSize = v; return this; }
    public int ngramVocabSize() { return ngramVocabSize; }
    public ModelHparamsBuilder ngramNeighborCount(int v) { this.ngramNeighborCount = v; return this; }
    public int ngramNeighborCount() { return ngramNeighborCount; }
    public ModelHparamsBuilder ngramSplitCount(int v) { this.ngramSplitCount = v; return this; }
    public int ngramSplitCount() { return ngramSplitCount; }
    public ModelHparamsBuilder ngramEosTokenId(int v) { this.ngramEosTokenId = v; return this; }
    public int ngramEosTokenId() { return ngramEosTokenId; }
    public ModelHparamsBuilder pleLayerMultipliers(long[] v) { this.pleLayerMultipliers = Objects.requireNonNull(v, "pleLayerMultipliers").clone(); return this; }
    public long[] pleLayerMultipliers() { return pleLayerMultipliers.clone(); }
    public ModelHparamsBuilder pleHeadOffsets(long[] v) { this.pleHeadOffsets = Objects.requireNonNull(v, "pleHeadOffsets").clone(); return this; }
    public long[] pleHeadOffsets() { return pleHeadOffsets.clone(); }
    public ModelHparamsBuilder pleHeadVocabSizes(long[] v) { this.pleHeadVocabSizes = Objects.requireNonNull(v, "pleHeadVocabSizes").clone(); return this; }
    public long[] pleHeadVocabSizes() { return pleHeadVocabSizes.clone(); }
    public ModelHparamsBuilder nAltup(int v) { this.nAltup = v; return this; }
    public int nAltup() { return nAltup; }
    public ModelHparamsBuilder iAltupAct(int v) { this.iAltupAct = v; return this; }
    public int iAltupAct() { return iAltupAct; }
    public ModelHparamsBuilder laurelRank(int v) { this.laurelRank = v; return this; }
    public int laurelRank() { return laurelRank; }
    public ModelHparamsBuilder nEmbdAltup(int v) { this.nEmbdAltup = v; return this; }
    public int nEmbdAltup() { return nEmbdAltup; }

    // ---- scaling accessors ----
    public ModelHparamsBuilder logitScale(float v) { this.logitScale = v; return this; }
    public float logitScale() { return logitScale; }
    public ModelHparamsBuilder logitScaleDivides(boolean v) { this.logitScaleDivides = v; return this; }
    public boolean logitScaleDivides() { return logitScaleDivides; }
    public ModelHparamsBuilder residualScale(float v) { this.residualScale = v; return this; }
    public float residualScale() { return residualScale; }
    public ModelHparamsBuilder embeddingScale(float v) { this.embeddingScale = v; return this; }
    public float embeddingScale() { return embeddingScale; }
    public ModelHparamsBuilder outputHiddenScale(float v) { this.outputHiddenScale = v; return this; }
    public float outputHiddenScale() { return outputHiddenScale; }
    public ModelHparamsBuilder situBeta(float v) { this.situBeta = v; return this; }
    public float situBeta() { return situBeta; }
    public ModelHparamsBuilder situLinearBeta(float v) { this.situLinearBeta = v; return this; }
    public float situLinearBeta() { return situLinearBeta; }

    // ---- control flow accessors ----
    public ModelHparamsBuilder numLoops(int v) { this.numLoops = v; return this; }
    public int numLoops() { return numLoops; }
    public ModelHparamsBuilder skipLoopFinalNorm(boolean v) { this.skipLoopFinalNorm = v; return this; }
    public boolean skipLoopFinalNorm() { return skipLoopFinalNorm; }
    public ModelHparamsBuilder hrmLayersPerStack(int v) { this.hrmLayersPerStack = v; return this; }
    public int hrmLayersPerStack() { return hrmLayersPerStack; }
    public ModelHparamsBuilder hrmHCycles(int v) { this.hrmHCycles = v; return this; }
    public int hrmHCycles() { return hrmHCycles; }
    public ModelHparamsBuilder hrmLCycles(int v) { this.hrmLCycles = v; return this; }
    public int hrmLCycles() { return hrmLCycles; }
    public ModelHparamsBuilder hrmPrefixLm(boolean v) { this.hrmPrefixLm = v; return this; }
    public boolean hrmPrefixLm() { return hrmPrefixLm; }

    // ---- repeat-block schedule ----

    public ModelHparamsBuilder repeatLoops(int v) { this.repeatLoops = v; return this; }
    public int repeatLoops() { return repeatLoops; }
    public ModelHparamsBuilder repeatInputLayers(int v) { this.repeatInputLayers = v; return this; }
    public int repeatInputLayers() { return repeatInputLayers; }
    public ModelHparamsBuilder repeatRecurrentLayers(int v) { this.repeatRecurrentLayers = v; return this; }
    public int repeatRecurrentLayers() { return repeatRecurrentLayers; }
    public ModelHparamsBuilder repeatOutputLayers(int v) { this.repeatOutputLayers = v; return this; }
    public int repeatOutputLayers() { return repeatOutputLayers; }
    public ModelHparamsBuilder repeatJustNorm(boolean v) { this.repeatJustNorm = v; return this; }
    public boolean repeatJustNorm() { return repeatJustNorm; }
    public ModelHparamsBuilder repeatJustNormEps(float v) { this.repeatJustNormEps = v; return this; }
    public float repeatJustNormEps() { return repeatJustNormEps; }
    public ModelHparamsBuilder slerpAlphaScale(float v) { this.slerpAlphaScale = v; return this; }
    public float slerpAlphaScale() { return slerpAlphaScale; }
    public ModelHparamsBuilder repeatDualAttention(boolean v) { this.repeatDualAttention = v; return this; }
    public boolean repeatDualAttention() { return repeatDualAttention; }
    public ModelHparamsBuilder repeatLocalWindow(int v) { this.repeatLocalWindow = v; return this; }
    public int repeatLocalWindow() { return repeatLocalWindow; }

    // ---- heads, encoders and pooling accessors ----
    public ModelHparamsBuilder pooling(PoolingKind v) { this.pooling = v; return this; }
    public PoolingKind pooling() { return pooling; }
    public ModelHparamsBuilder poolingCls(PoolingKind v) { this.poolingCls = v; return this; }
    public PoolingKind poolingCls() { return poolingCls; }
    public ModelHparamsBuilder nClsOut(int v) { this.nClsOut = v; return this; }
    public int nClsOut() { return nClsOut; }
    public ModelHparamsBuilder classifierLabels(String[] v) { this.classifierLabels = Objects.requireNonNull(v, "classifierLabels").clone(); return this; }
    public String[] classifierLabels() { return classifierLabels.clone(); }
    public ModelHparamsBuilder decisionLabels(String[] v) { this.decisionLabels = Objects.requireNonNull(v, "decisionLabels").clone(); return this; }
    public String[] decisionLabels() { return decisionLabels.clone(); }
    public ModelHparamsBuilder decisionType(String v) { this.decisionType = v; return this; }
    public String decisionType() { return decisionType; }
    public ModelHparamsBuilder decisionMaxHeadTokens(int v) { this.decisionMaxHeadTokens = v; return this; }
    public int decisionMaxHeadTokens() { return decisionMaxHeadTokens; }
    public ModelHparamsBuilder decisionTemperatures(Map<String, Float> v) {
        this.decisionTemperatures = v == null ? Collections.emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<>(v));
        return this;
    }
    public Map<String, Float> decisionTemperatures() { return decisionTemperatures; }
    public ModelHparamsBuilder decisionScoreBias(Map<Integer, float[]> v) {
        if (v == null) {
            this.decisionScoreBias = Collections.emptyMap();
            return this;
        }
        Map<Integer, float[]> copy = new LinkedHashMap<>();
        for (Map.Entry<Integer, float[]> e : v.entrySet()) {
            copy.put(e.getKey(), e.getValue().clone());
        }
        this.decisionScoreBias = Collections.unmodifiableMap(copy);
        return this;
    }
    public Map<Integer, float[]> decisionScoreBias() { return decisionScoreBias; }
    public ModelHparamsBuilder decisionRoutingLayers(int v) { this.decisionRoutingLayers = v; return this; }
    public int decisionRoutingLayers() { return decisionRoutingLayers; }
    public ModelHparamsBuilder decisionJointLayers(int v) { this.decisionJointLayers = v; return this; }
    public int decisionJointLayers() { return decisionJointLayers; }
    public ModelHparamsBuilder decisionHeadCount(int v) { this.decisionHeadCount = v; return this; }
    public int decisionHeadCount() { return decisionHeadCount; }
    public ModelHparamsBuilder classifierScope(ClassifierScope v) { this.classifierScope = v; return this; }
    public ClassifierScope classifierScope() { return classifierScope; }
    public ModelHparamsBuilder classifierActivation(ActOp v) { this.classifierActivation = v; return this; }
    public ActOp classifierActivation() { return classifierActivation; }
    public ModelHparamsBuilder poolerActivation(PoolerAct v) { this.poolerActivation = v; return this; }
    public PoolerAct poolerActivation() { return poolerActivation; }
    public ModelHparamsBuilder classifierTokenDirect(boolean v) { this.classifierTokenDirect = v; return this; }
    public boolean classifierTokenDirect() { return classifierTokenDirect; }
    public ModelHparamsBuilder classifierSparse(String v) { this.classifierSparse = v; return this; }
    public String classifierSparse() { return classifierSparse; }
    public ModelHparamsBuilder sparseClsToken(int v) { this.sparseClsToken = v; return this; }
    public int sparseClsToken() { return sparseClsToken; }
    public ModelHparamsBuilder sparseSepToken(int v) { this.sparseSepToken = v; return this; }
    public int sparseSepToken() { return sparseSepToken; }
    public ModelHparamsBuilder classifierBosToken(int v) { this.classifierBosToken = v; return this; }
    public int classifierBosToken() { return classifierBosToken; }
    public ModelHparamsBuilder classifierEosToken(int v) { this.classifierEosToken = v; return this; }
    public int classifierEosToken() { return classifierEosToken; }
    public ModelHparamsBuilder classifierStepTagId(int v) { this.classifierStepTagId = v; return this; }
    public int classifierStepTagId() { return classifierStepTagId; }
    public ModelHparamsBuilder poolingNormalize(boolean v) { this.poolingNormalize = v; return this; }
    public boolean poolingNormalize() { return poolingNormalize; }
    public ModelHparamsBuilder hiddenActivation(String v) { this.hiddenActivation = v; return this; }
    public String hiddenActivation() { return hiddenActivation; }
    public ModelHparamsBuilder decStartTokenId(int v) { this.decStartTokenId = v; return this; }
    public int decStartTokenId() { return decStartTokenId; }
    public ModelHparamsBuilder decNLayer(int v) { this.decNLayer = v; return this; }
    public int decNLayer() { return decNLayer; }
    public ModelHparamsBuilder dense2FeatIn(int v) { this.dense2FeatIn = v; return this; }
    public int dense2FeatIn() { return dense2FeatIn; }
    public ModelHparamsBuilder dense2FeatOut(int v) { this.dense2FeatOut = v; return this; }
    public int dense2FeatOut() { return dense2FeatOut; }
    public ModelHparamsBuilder dense3FeatIn(int v) { this.dense3FeatIn = v; return this; }
    public int dense3FeatIn() { return dense3FeatIn; }
    public ModelHparamsBuilder dense3FeatOut(int v) { this.dense3FeatOut = v; return this; }
    public int dense3FeatOut() { return dense3FeatOut; }
    public ModelHparamsBuilder diffusionShiftLogits(boolean v) { this.diffusionShiftLogits = v; return this; }
    public boolean diffusionShiftLogits() { return diffusionShiftLogits; }

    // ---- drafters and adapters accessors ----
    public ModelHparamsBuilder targetLayers(int[] v) { this.targetLayers = Objects.requireNonNull(v, "targetLayers").clone(); return this; }
    public int[] targetLayers() { return targetLayers.clone(); }
    public ModelHparamsBuilder targetHiddenSize(int v) { this.targetHiddenSize = v; return this; }
    public int targetHiddenSize() { return targetHiddenSize; }
    public ModelHparamsBuilder dflashBlockSize(int v) { this.dflashBlockSize = v; return this; }
    public int dflashBlockSize() { return dflashBlockSize; }
    public ModelHparamsBuilder dflashConvKernelSize(int v) { this.dflashConvKernelSize = v; return this; }
    public int dflashConvKernelSize() { return dflashConvKernelSize; }
    public ModelHparamsBuilder dflashConvGroupSize(int v) { this.dflashConvGroupSize = v; return this; }
    public int dflashConvGroupSize() { return dflashConvGroupSize; }
    public ModelHparamsBuilder dflashSelectorRank(int v) { this.dflashSelectorRank = v; return this; }
    public int dflashSelectorRank() { return dflashSelectorRank; }
    public ModelHparamsBuilder dflashSelectorTopK(int v) { this.dflashSelectorTopK = v; return this; }
    public int dflashSelectorTopK() { return dflashSelectorTopK; }
    public ModelHparamsBuilder lilicorrHiddenSize(int v) { this.lilicorrHiddenSize = v; return this; }
    public int lilicorrHiddenSize() { return lilicorrHiddenSize; }
    public ModelHparamsBuilder lilicorrCandidateTopK(int v) { this.lilicorrCandidateTopK = v; return this; }
    public int lilicorrCandidateTopK() { return lilicorrCandidateTopK; }
    public ModelHparamsBuilder lilicorrNumHeads(int v) { this.lilicorrNumHeads = v; return this; }
    public int lilicorrNumHeads() { return lilicorrNumHeads; }
    public ModelHparamsBuilder lilicorrLayers(int v) { this.lilicorrLayers = v; return this; }
    public int lilicorrLayers() { return lilicorrLayers; }
    public ModelHparamsBuilder lilicorrVectorEps(float v) { this.lilicorrVectorEps = v; return this; }
    public float lilicorrVectorEps() { return lilicorrVectorEps; }
    public ModelHparamsBuilder lilicorrLogitScale(float v) { this.lilicorrLogitScale = v; return this; }
    public float lilicorrLogitScale() { return lilicorrLogitScale; }
    public ModelHparamsBuilder lilicorrFactorDim(int v) { this.lilicorrFactorDim = v; return this; }
    public int lilicorrFactorDim() { return lilicorrFactorDim; }
    public ModelHparamsBuilder lilicorrMlpHiddenSize(int v) { this.lilicorrMlpHiddenSize = v; return this; }
    public int lilicorrMlpHiddenSize() { return lilicorrMlpHiddenSize; }
    public ModelHparamsBuilder adapterCount(int v) { this.adapterCount = v; return this; }
    public int adapterCount() { return adapterCount; }
    public ModelHparamsBuilder adapterLoraRank(int v) { this.adapterLoraRank = v; return this; }
    public int adapterLoraRank() { return adapterLoraRank; }
    public ModelHparamsBuilder adapterRouterGain(float v) { this.adapterRouterGain = v; return this; }
    public float adapterRouterGain() { return adapterRouterGain; }
    public ModelHparamsBuilder adapterTokenIdsActivate(int[] v) { this.adapterTokenIdsActivate = Objects.requireNonNull(v, "adapterTokenIdsActivate").clone(); return this; }
    public int[] adapterTokenIdsActivate() { return adapterTokenIdsActivate.clone(); }
    public ModelHparamsBuilder adapterTokenIdsSubstitute(int[] v) { this.adapterTokenIdsSubstitute = Objects.requireNonNull(v, "adapterTokenIdsSubstitute").clone(); return this; }
    public int[] adapterTokenIdsSubstitute() { return adapterTokenIdsSubstitute.clone(); }

    // ---- per-layer accessors ----
    public int nHead(int il) { return nHeadArr[layer(il)]; }
    public ModelHparamsBuilder nHead(int il, int v) { nHeadArr[layer(il)] = v; return this; }
    public int[] nHeadArray() { return nHeadArr.clone(); }
    public ModelHparamsBuilder nHeadArray(int[] v) { nHeadArr = layerArray("nHead", v).clone(); return this; }
    public int nHeadKv(int il) { return nHeadKvArr[layer(il)]; }
    public ModelHparamsBuilder nHeadKv(int il, int v) { nHeadKvArr[layer(il)] = v; return this; }
    public int[] nHeadKvArray() { return nHeadKvArr.clone(); }
    public ModelHparamsBuilder nHeadKvArray(int[] v) { nHeadKvArr = layerArray("nHeadKv", v).clone(); return this; }
    public int nFf(int il) { return nFfArr[layer(il)]; }
    public ModelHparamsBuilder nFf(int il, int v) { nFfArr[layer(il)] = v; return this; }
    public int[] nFfArray() { return nFfArr.clone(); }
    public ModelHparamsBuilder nFfArray(int[] v) { nFfArr = layerArray("nFf", v).clone(); return this; }
    public int nFfExp(int il) { return nFfExpArr[layer(il)]; }
    public ModelHparamsBuilder nFfExp(int il, int v) { nFfExpArr[layer(il)] = v; return this; }
    public int[] nFfExpArray() { return nFfExpArr.clone(); }
    public ModelHparamsBuilder nFfExpArray(int[] v) { nFfExpArr = layerArray("nFfExp", v).clone(); return this; }
    public int nExpertUsed(int il) { return nExpertUsedArr[layer(il)]; }
    public ModelHparamsBuilder nExpertUsed(int il, int v) { nExpertUsedArr[layer(il)] = v; return this; }
    public int[] nExpertUsedArray() { return nExpertUsedArr.clone(); }
    public ModelHparamsBuilder nExpertUsedArray(int[] v) { nExpertUsedArr = layerArray("nExpertUsed", v).clone(); return this; }
    public boolean isSwa(int il) { return swaArr[layer(il)]; }
    public ModelHparamsBuilder isSwa(int il, boolean v) { swaArr[layer(il)] = v; return this; }
    public boolean[] isSwaArray() { return swaArr.clone(); }
    public ModelHparamsBuilder isSwaArray(boolean[] v) { swaArr = layerArray("isSwa", v).clone(); return this; }
    /** True when layer {@code il} is an MLP-only block (Dory's {@code layer_types} entry "mlp"). */
    public boolean isMlp(int il) { return mlpArr[layer(il)]; }
    public ModelHparamsBuilder isMlp(int il, boolean v) { mlpArr[layer(il)] = v; return this; }
    public boolean[] isMlpArray() { return mlpArr.clone(); }
    public ModelHparamsBuilder isMlpArray(boolean[] v) { mlpArr = layerArray("isMlp", v).clone(); return this; }
    public boolean ropeEnabled(int il) { return ropeArr[layer(il)]; }
    public ModelHparamsBuilder ropeEnabled(int il, boolean v) { ropeArr[layer(il)] = v; return this; }
    public boolean[] ropeEnabledArray() { return ropeArr.clone(); }
    public ModelHparamsBuilder ropeEnabledArray(boolean[] v) { ropeArr = layerArray("ropeEnabled", v).clone(); return this; }
    public boolean isRecurrent(int il) { return recurrentArr[layer(il)]; }
    public ModelHparamsBuilder isRecurrent(int il, boolean v) { recurrentArr[layer(il)] = v; return this; }
    public boolean[] isRecurrentArray() { return recurrentArr.clone(); }
    public ModelHparamsBuilder isRecurrentArray(boolean[] v) { recurrentArr = layerArray("isRecurrent", v).clone(); return this; }
    public boolean isHybridLayer(int il) { return hybridArr[layer(il)]; }
    public ModelHparamsBuilder isHybridLayer(int il, boolean v) { hybridArr[layer(il)] = v; return this; }
    public boolean[] isHybridLayerArray() { return hybridArr.clone(); }
    public ModelHparamsBuilder isHybridLayerArray(boolean[] v) { hybridArr = layerArray("isHybridLayer", v).clone(); return this; }
    public int zamba2SharedBlock(int il) { return sharedBlockArr[layer(il)]; }
    public ModelHparamsBuilder zamba2SharedBlock(int il, int v) { sharedBlockArr[layer(il)] = v; return this; }
    public int[] zamba2SharedBlockArray() { return sharedBlockArr.clone(); }
    public ModelHparamsBuilder zamba2SharedBlockArray(int[] v) { sharedBlockArr = layerArray("zamba2SharedBlock", v).clone(); return this; }
    public boolean isIndexerFull(int il) { return indexerFullArr[layer(il)]; }
    public ModelHparamsBuilder isIndexerFull(int il, boolean v) { indexerFullArr[layer(il)] = v; return this; }
    public boolean[] isIndexerFullArray() { return indexerFullArr.clone(); }
    public ModelHparamsBuilder isIndexerFullArray(boolean[] v) { indexerFullArr = layerArray("isIndexerFull", v).clone(); return this; }
    public boolean isPleLayer(int il) { return pleArr[layer(il)]; }
    public ModelHparamsBuilder isPleLayer(int il, boolean v) { pleArr[layer(il)] = v; return this; }
    public boolean[] isPleLayerArray() { return pleArr.clone(); }
    public ModelHparamsBuilder isPleLayerArray(boolean[] v) { pleArr = layerArray("isPleLayer", v).clone(); return this; }
    public int compressRatio(int il) { return compressRatioArr[layer(il)]; }
    public ModelHparamsBuilder compressRatio(int il, int v) { compressRatioArr[layer(il)] = v; return this; }
    public int[] compressRatioArray() { return compressRatioArr.clone(); }
    public ModelHparamsBuilder compressRatioArray(int[] v) { compressRatioArr = layerArray("compressRatio", v).clone(); return this; }
    public int dsv41KvSource(int il) { return dsv41KvSourceArr[layer(il)]; }
    public ModelHparamsBuilder dsv41KvSource(int il, int v) { dsv41KvSourceArr[layer(il)] = v; return this; }
    public int[] dsv41KvSourceArray() { return dsv41KvSourceArr.clone(); }
    public ModelHparamsBuilder dsv41KvSourceArray(int[] v) { dsv41KvSourceArr = layerArray("dsv41KvSource", v).clone(); return this; }
    public int dsv41TopkSource(int il) { return dsv41TopkSourceArr[layer(il)]; }
    public ModelHparamsBuilder dsv41TopkSource(int il, int v) { dsv41TopkSourceArr[layer(il)] = v; return this; }
    public int[] dsv41TopkSourceArray() { return dsv41TopkSourceArr.clone(); }
    public ModelHparamsBuilder dsv41TopkSourceArray(int[] v) { dsv41TopkSourceArr = layerArray("dsv41TopkSource", v).clone(); return this; }
    public int deepstackMapping(int il) { return deepstackMappingArr[layer(il)]; }
    public ModelHparamsBuilder deepstackMapping(int il, int v) { deepstackMappingArr[layer(il)] = v; return this; }
    public int[] deepstackMappingArray() { return deepstackMappingArr.clone(); }
    public ModelHparamsBuilder deepstackMappingArray(int[] v) { deepstackMappingArr = layerArray("deepstackMapping", v).clone(); return this; }
    public float xieluAlphaN(int il) { return xieluAlphaNArr[layer(il)]; }
    public ModelHparamsBuilder xieluAlphaN(int il, float v) { xieluAlphaNArr[layer(il)] = v; return this; }
    public float[] xieluAlphaNArray() { return xieluAlphaNArr.clone(); }
    public ModelHparamsBuilder xieluAlphaNArray(float[] v) { xieluAlphaNArr = layerArray("xieluAlphaN", v).clone(); return this; }
    public float xieluAlphaP(int il) { return xieluAlphaPArr[layer(il)]; }
    public ModelHparamsBuilder xieluAlphaP(int il, float v) { xieluAlphaPArr[layer(il)] = v; return this; }
    public float[] xieluAlphaPArray() { return xieluAlphaPArr.clone(); }
    public ModelHparamsBuilder xieluAlphaPArray(float[] v) { xieluAlphaPArr = layerArray("xieluAlphaP", v).clone(); return this; }
    public float xieluBeta(int il) { return xieluBetaArr[layer(il)]; }
    public ModelHparamsBuilder xieluBeta(int il, float v) { xieluBetaArr[layer(il)] = v; return this; }
    public float[] xieluBetaArray() { return xieluBetaArr.clone(); }
    public ModelHparamsBuilder xieluBetaArray(float[] v) { xieluBetaArr = layerArray("xieluBeta", v).clone(); return this; }
    public float xieluEps(int il) { return xieluEpsArr[layer(il)]; }
    public ModelHparamsBuilder xieluEps(int il, float v) { xieluEpsArr[layer(il)] = v; return this; }
    public float[] xieluEpsArray() { return xieluEpsArr.clone(); }
    public ModelHparamsBuilder xieluEpsArray(float[] v) { xieluEpsArr = layerArray("xieluEps", v).clone(); return this; }
    public float swigluClampExp(int il) { return swigluClampExpArr[layer(il)]; }
    public ModelHparamsBuilder swigluClampExp(int il, float v) { swigluClampExpArr[layer(il)] = v; return this; }
    public float[] swigluClampExpArray() { return swigluClampExpArr.clone(); }
    public ModelHparamsBuilder swigluClampExpArray(float[] v) { swigluClampExpArr = layerArray("swigluClampExp", v).clone(); return this; }
    public float swigluClampShexp(int il) { return swigluClampShexpArr[layer(il)]; }
    public ModelHparamsBuilder swigluClampShexp(int il, float v) { swigluClampShexpArr[layer(il)] = v; return this; }
    public float[] swigluClampShexpArray() { return swigluClampShexpArr.clone(); }
    public ModelHparamsBuilder swigluClampShexpArray(float[] v) { swigluClampShexpArr = layerArray("swigluClampShexp", v).clone(); return this; }

    private void allocLayerArrays(int n, boolean keep) {
        nHeadArr = resize(keep ? nHeadArr : null, n, 0);
        nHeadKvArr = resize(keep ? nHeadKvArr : null, n, 0);
        nFfArr = resize(keep ? nFfArr : null, n, 0);
        nFfExpArr = resize(keep ? nFfExpArr : null, n, 0);
        nExpertUsedArr = resize(keep ? nExpertUsedArr : null, n, 0);
        swaArr = resize(keep ? swaArr : null, n, false);
        /* Without the per-layer windows the scalar nSwa applies; a zero array would silently disable every window. */
        swaWindowArr = keep && swaWindowArr.length > 0 ? resize(swaWindowArr, n, 0) : new int[0];
        mlpArr = resize(keep ? mlpArr : null, n, false);
        ropeArr = resize(keep ? ropeArr : null, n, true);
        recurrentArr = resize(keep ? recurrentArr : null, n, recurrentDefault);
        hybridArr = resize(keep ? hybridArr : null, n, false);
        sharedBlockArr = resize(keep ? sharedBlockArr : null, n, -1);
        indexerFullArr = resize(keep ? indexerFullArr : null, n, false);
        pleArr = resize(keep ? pleArr : null, n, false);
        compressRatioArr = resize(keep ? compressRatioArr : null, n, 0);
        dsv41KvSourceArr = resize(keep ? dsv41KvSourceArr : null, n, -1);
        dsv41TopkSourceArr = resize(keep ? dsv41TopkSourceArr : null, n, -1);
        deepstackMappingArr = resize(keep ? deepstackMappingArr : null, n, -1);
        xieluAlphaNArr = resize(keep ? xieluAlphaNArr : null, n, 0f);
        xieluAlphaPArr = resize(keep ? xieluAlphaPArr : null, n, 0f);
        xieluBetaArr = resize(keep ? xieluBetaArr : null, n, 0f);
        xieluEpsArr = resize(keep ? xieluEpsArr : null, n, 0f);
        swigluClampExpArr = resize(keep ? swigluClampExpArr : null, n, 0f);
        swigluClampShexpArr = resize(keep ? swigluClampShexpArr : null, n, 0f);
    }

    /** Builder for an architecture; per-layer arrays are empty until {@link #layerCounts(int, int)}. */
    public ModelHparamsBuilder(ArchTraits traits) {
        this.traits = Objects.requireNonNull(traits, "traits");
        this.arch = traits.arch();
        this.recurrentDefault = arch.isRecurrent();
        this.ropeType = traits.ropeType();
        this.mropeTextPositions = traits.mropeTextPositions();
        this.causal = traits.causalDefault();
        allocLayerArrays(0, false);
    }

    public Arch arch() {
        return arch;
    }

    public ArchTraits traits() {
        return traits;
    }

    /** All blocks including NextN/MTP blocks ({@code n_layer_all}). */
    public int nLayerAll() {
        return nLayerAll;
    }

    /** NextN/MTP blocks at the end of the stack ({@code n_layer_nextn}). */
    public int nextnLayers() {
        return nextnLayers;
    }

    /** Trunk blocks, {@code nLayerAll - nextnLayers} (llama-hparams.cpp:347-349). */
    public int nLayer() {
        return nLayerAll - nextnLayers;
    }

    /**
     * Sets the block counts and resizes every per-layer array, keeping the values of surviving layers and
     * giving new layers the llama.cpp defaults.
     *
     * @throws GgufFormatException when {@code nLayerAll} is outside {@code [1, MAX_LAYERS]} or
     *                             {@code nextnLayers} outside {@code [0, nLayerAll]}
     * @throws IllegalStateException when MoE or KV-owner overrides were already set
     */
    public ModelHparamsBuilder layerCounts(int nLayerAll, int nextnLayers) {
        if (nLayerAll < 1 || nLayerAll > ModelHparams.MAX_LAYERS) {
            throw new GgufFormatException(arch.ggufName() + ": block count " + nLayerAll + " outside [1, "
                    + ModelHparams.MAX_LAYERS + "]");
        }
        if (nextnLayers < 0 || nextnLayers > nLayerAll) {
            throw new GgufFormatException(arch.ggufName() + ": nextn_predict_layers " + nextnLayers
                    + " outside [0, " + nLayerAll + "]");
        }
        if (moeOverride != null || kvOwnerOverride != null) {
            throw new IllegalStateException("layer counts must be set before MoE or KV-owner layer overrides");
        }
        allocLayerArrays(nLayerAll, true);
        this.nLayerAll = nLayerAll;
        this.nextnLayers = nextnLayers;
        return this;
    }

    /** True when any layer is a sliding-window layer ({@code is_swa_any}). */
    public boolean isSwaAny() {
        for (boolean b : swaArr) {
            if (b) {
                return true;
            }
        }
        return false;
    }

    /** True when the multi-section RoPE sections are set ({@code use_mrope}, llama-hparams.cpp:351-353). */
    public boolean useMrope() {
        return ropeSections[0] > 0 && ropeSections[1] > 0;
    }

    /**
     * Marks the MoE layers explicitly, replacing the derived rule of {@link ModelHparams#isMoeLayer(int)}.
     */
    public ModelHparamsBuilder moeLayers(boolean[] moe) {
        this.moeOverride = layerArray("moeLayers", moe).clone();
        return this;
    }

    /**
     * Sets the layer whose KV cache each layer reads ({@code il} itself when it owns one), replacing the
     * derived rule of {@link ModelHparams#kvOwnerLayer(int)}.
     */
    public ModelHparamsBuilder kvOwnerLayers(int[] owners) {
        int[] copy = layerArray("kvOwnerLayers", owners).clone();
        for (int il = 0; il < copy.length; il++) {
            if (copy[il] < 0 || copy[il] > il) {
                throw new IllegalArgumentException("KV owner of layer " + il + " must be in [0, " + il + "], got "
                        + copy[il]);
            }
        }
        this.kvOwnerOverride = copy;
        return this;
    }

    /**
     * Builds the immutable hparams. MoE layers default to {@code nExpert > 0}, {@code il >= nLayerDenseLead},
     * {@code (il + 1) % nMoeLayerStep == 0} when a step is set (llama4.cpp:62, ernie4-5-moe.cpp:64) and, for
     * the bert-graph architectures only, {@code il % moeEveryN == 1} when set (bert.cpp:171); other
     * architectures reading {@code moe_every_n_layers} (k2-horizon) ignore it in their graph
     * (k2-horizon.cpp:67). KV owners follow the gemma3n/gemma4 reuse rule
     * (gemma3n.cpp:10-13, gemma4.cpp:14-18): layers at or after {@code nLayerKvFromStart} read layer
     * {@code nLayerKvFromStart - (isSwa(il) ? 2 : 1)}.
     *
     * @throws IllegalStateException when the layer counts were never set
     * @throws GgufFormatException when KV reuse is configured with {@code nLayerKvFromStart < 2}
     */
    public ModelHparams build() {
        if (nLayerAll == 0) {
            throw new IllegalStateException("layer counts not set for " + arch.ggufName());
        }
        boolean[] moe = moeOverride != null ? moeOverride.clone() : derivedMoeLayers();
        int[] owners = kvOwnerOverride != null ? kvOwnerOverride.clone() : derivedKvOwners();
        return new ModelHparams(snapshot(), moe, owners);
    }

    private boolean[] derivedMoeLayers() {
        boolean[] moe = new boolean[nLayerAll];
        if (nExpert == 0) {
            return moe;
        }
        boolean everyN = moeEveryN > 0 && usesBertGraph();
        for (int il = 0; il < nLayerAll; il++) {
            moe[il] = il >= nLayerDenseLead
                    && (nMoeLayerStep <= 0 || (il + 1) % nMoeLayerStep == 0)
                    && (!everyN || il % moeEveryN == 1);
        }
        return moe;
    }

    /**
     * True when the llama.cpp bert graph builds this architecture ({@link ArchTraits#graphFamily()}), the only graph
     * applying {@code il % moe_every_n_layers == 1}.
     */
    private boolean usesBertGraph() {
        return traits.graphFamily() == Arch.BERT;
    }

    private int[] derivedKvOwners() {
        int[] owners = new int[nLayerAll];
        for (int il = 0; il < nLayerAll; il++) {
            owners[il] = il;
            if (nLayerKvFromStart >= 0 && il >= nLayerKvFromStart) {
                if (nLayerKvFromStart < 2) {
                    throw new GgufFormatException(arch.ggufName() + ": KV reuse needs at least two KV layers, got "
                            + nLayerKvFromStart);
                }
                owners[il] = nLayerKvFromStart - (swaArr[il] ? 2 : 1);
            }
        }
        return owners;
    }

    private ModelHparamsBuilder snapshot() {
        ModelHparamsBuilder c;
        try {
            c = (ModelHparamsBuilder) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError(e);
        }
        c.ropeSections = ropeSections.clone();
        c.pleLayerMultipliers = pleLayerMultipliers.clone();
        c.pleHeadOffsets = pleHeadOffsets.clone();
        c.pleHeadVocabSizes = pleHeadVocabSizes.clone();
        c.engramTokenMap = engramTokenMap.clone();
        c.classifierLabels = classifierLabels.clone();
        c.decisionLabels = decisionLabels.clone();
        c.targetLayers = targetLayers.clone();
        c.adapterTokenIdsActivate = adapterTokenIdsActivate.clone();
        c.adapterTokenIdsSubstitute = adapterTokenIdsSubstitute.clone();
        c.nHeadArr = nHeadArr.clone();
        c.nHeadKvArr = nHeadKvArr.clone();
        c.nFfArr = nFfArr.clone();
        c.nFfExpArr = nFfExpArr.clone();
        c.nExpertUsedArr = nExpertUsedArr.clone();
        c.swaArr = swaArr.clone();
        c.swaWindowArr = swaWindowArr.clone();
        c.mlpArr = mlpArr.clone();
        c.ropeArr = ropeArr.clone();
        c.recurrentArr = recurrentArr.clone();
        c.hybridArr = hybridArr.clone();
        c.sharedBlockArr = sharedBlockArr.clone();
        c.indexerFullArr = indexerFullArr.clone();
        c.pleArr = pleArr.clone();
        c.compressRatioArr = compressRatioArr.clone();
        c.dsv41KvSourceArr = dsv41KvSourceArr.clone();
        c.dsv41TopkSourceArr = dsv41TopkSourceArr.clone();
        c.deepstackMappingArr = deepstackMappingArr.clone();
        c.xieluAlphaNArr = xieluAlphaNArr.clone();
        c.xieluAlphaPArr = xieluAlphaPArr.clone();
        c.xieluBetaArr = xieluBetaArr.clone();
        c.xieluEpsArr = xieluEpsArr.clone();
        c.swigluClampExpArr = swigluClampExpArr.clone();
        c.swigluClampShexpArr = swigluClampShexpArr.clone();
        c.moeOverride = null;
        c.kvOwnerOverride = null;
        return c;
    }

    private int layer(int il) {
        if (il < 0 || il >= nLayerAll) {
            throw new IndexOutOfBoundsException("layer " + il + " outside [0, " + nLayerAll + ")");
        }
        return il;
    }

    private int[] layerArray(String name, int[] v) {
        checkLength(name, Objects.requireNonNull(v, name).length);
        return v;
    }

    private boolean[] layerArray(String name, boolean[] v) {
        checkLength(name, Objects.requireNonNull(v, name).length);
        return v;
    }

    private float[] layerArray(String name, float[] v) {
        checkLength(name, Objects.requireNonNull(v, name).length);
        return v;
    }

    private void checkLength(String name, int length) {
        if (length != nLayerAll) {
            throw new IllegalArgumentException(name + " has " + length + " entries, expected " + nLayerAll);
        }
    }

    private static int[] resize(int[] old, int n, int dflt) {
        int[] out = new int[n];
        Arrays.fill(out, dflt);
        if (old != null) {
            System.arraycopy(old, 0, out, 0, Math.min(old.length, n));
        }
        return out;
    }

    private static boolean[] resize(boolean[] old, int n, boolean dflt) {
        boolean[] out = new boolean[n];
        Arrays.fill(out, dflt);
        if (old != null) {
            System.arraycopy(old, 0, out, 0, Math.min(old.length, n));
        }
        return out;
    }

    private static float[] resize(float[] old, int n, float dflt) {
        float[] out = new float[n];
        Arrays.fill(out, dflt);
        if (old != null) {
            System.arraycopy(old, 0, out, 0, Math.min(old.length, n));
        }
        return out;
    }
}
