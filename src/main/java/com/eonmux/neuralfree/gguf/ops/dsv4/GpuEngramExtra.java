package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * The DeepSeek-V4.1 Engram n-gram embedding of one layer on the GPU ({@code LayerSpec.Extras.ENGRAM},
 * models/deepseek41.cpp:88-131, 240-292, 456-520): a per-layer n-gram hash gathers {@code ple_head_dim}-wide rows of
 * the shared {@code per_layer_token_embd} table, projects them through the layer's {@code engram_kv} into one key per
 * hyper-connection stream and one value, and adds {@code value * gate} to every stream, where the gate scores the
 * layer input against its key.
 *
 * <p>The hash is integer bookkeeping over the step's token record ({@link GpuScratch#readStepTokens}), computed on
 * the host exactly as {@link CpuEngramExtra} does; the tensor work runs on the device: {@code GpuOps.embed} gathers
 * the rows, the projections and norms are the shared kernels, and {@value Dsv4Kernels#ENGRAM_ADD} applies the gate
 * and the add. An embedding row (a negative token id) is hashed as the padding token and receives no contribution.
 * The extra runs before the hyper-connection extra, on the {@code hc * n_embd}-wide residual stream. Not
 * thread-safe.</p>
 */
public final class GpuEngramExtra implements GpuLayerExtra {

    private static final String TEMP_EMB = "dsv4.engram.emb";
    private static final String TEMP_KV = "dsv4.engram.kv";
    private static final String TEMP_KEY_ROWS = "dsv4.engram.keys";
    private static final String TEMP_NORM_X = "dsv4.engram.x";
    private static final String TEMP_NORM_KEY = "dsv4.engram.normkey";
    private static final String TEMP_ACTIVE = "dsv4.engram.active";

    private GpuOps ops;
    private GpuScratch scratch;
    private GpuMemory memory;
    private GpuWeight table;
    private GpuWeight engramKv;
    private GpuProjectionDeltas.Site kvDeltas;
    private GpuWeight gateW;
    private int il;
    private int ubatch;
    private int nEmbd;
    private int hc;
    private int nGram;
    private int perGram;
    private int nCols;
    private int headDim;
    private int padToken;
    private int[] tokenMap;
    private long[] multipliers;
    private long[] offsets;
    private long[] vocab;
    private float eps;
    private PreparedDispatch engramAdd;
    private ScratchSlot emb;
    private ScratchSlot kvOut;
    private ScratchSlot keyRows;
    private ScratchSlot normX;
    private ScratchSlot normKey;
    private ScratchSlot activeGpu;
    private GpuBuffer placeholders;
    private int[] stepIds;
    private int[] window;
    private long[] toks;
    private int[] ids;

    /** An unbound Engram extra; {@link #bind} resolves its layer's hash module and tensors. */
    public GpuEngramExtra() {
    }

    /** The hash and the gate are per row, exact over any step layout. */
    @Override
    public String batchRefusal() {
        return null;
    }

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (ops != null) {
            throw new IllegalStateException("Engram extra is already bound");
        }
        ModelSpec spec = ctx.spec();
        ModelHparams hp = spec.hparams();
        String arch = spec.arch().ggufName();
        String where = arch + " layer " + layer.stack() + "." + layer.il();
        GpuMemory mem = ctx.memory();
        if (mem == null) {
            throw new IllegalStateException(where + ": the Engram hash needs the model memory's token record");
        }
        int e = module(where, hp, layer.weightLayer());
        this.nEmbd = hp.nEmbd();
        this.hc = layer.residual().hcMult();
        if (hc < 1 || layer.residual().streamWidth() != (long) hc * nEmbd) {
            throw new IllegalArgumentException(where + ": the Engram embedding needs a residual of hc*n_embd, got "
                    + layer.residual());
        }
        this.nGram = hp.pleNgramSize();
        this.perGram = hp.pleHeadsPerNgram();
        this.nCols = hp.pleNHeads();
        this.headDim = hp.pleHeadDim();
        if (nGram < 2 || perGram < 1 || nCols != (nGram - 1) * perGram || headDim < 1
                || (long) nCols * headDim != nEmbd) {
            throw new IllegalArgumentException(where + ": Engram hparams out of range (ngram " + nGram
                    + ", heads/ngram " + perGram + ", heads " + nCols + ", head dim " + headDim + ", n_embd "
                    + nEmbd + ")");
        }
        this.multipliers = slice(hp.pleLayerMultipliers(), e * nGram, nGram, where, "ple_layer_multipliers");
        this.offsets = slice(hp.pleHeadOffsets(), e * nCols, nCols, where, "ple_head_offsets");
        this.vocab = slice(hp.pleHeadVocabSizes(), e * nCols, nCols, where, "ple_head_vocab_sizes");
        this.padToken = hp.engramPadTokenId();
        this.tokenMap = hp.engramTokenMap();
        if (padToken < 0 || padToken >= tokenMap.length) {
            throw new IllegalArgumentException(where + ": engram.pad_token_id " + padToken + " is outside the "
                    + tokenMap.length + " entries of engram.token_map");
        }
        this.padToken = tokenMap[padToken];
        for (int c = 0; c < nCols; c++) {
            if (vocab[c] < 2 || offsets[c] < 0 || vocab[c] > Integer.MAX_VALUE
                    || offsets[c] > Integer.MAX_VALUE || vocab[c] > Integer.MAX_VALUE - offsets[c]) {
                throw new IllegalArgumentException(where + ": Engram head " + c + " has bucket range ["
                        + offsets[c] + ", " + (offsets[c] + vocab[c]) + ")");
            }
        }
        this.eps = hp.normRmsEps();
        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        String tableName = TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD);
        String kvName = TensorNames.weight(TensorNames.ENGRAM_KV);
        String gateName = TensorNames.weight(TensorNames.ENGRAM_GATE);
        GpuWeight tbl = weights.get(tableName);
        if (tbl.cols() != headDim) {
            throw new IllegalArgumentException(where + ": " + tableName + " rows hold " + tbl.cols()
                    + " floats, expected ple_head_dim " + headDim);
        }
        if (!tbl.isSingleShard()) {
            throw new UnsupportedOperationException(where + ": " + tableName + " is split into " + tbl.shardCount()
                    + " row shards; the device gather needs one buffer");
        }
        for (int c = 0; c < nCols; c++) {
            if (offsets[c] + vocab[c] > tbl.rows()) {
                throw new IllegalArgumentException(where + ": Engram head " + c + " rows [" + offsets[c] + ", "
                        + (offsets[c] + vocab[c]) + ") fall outside the " + tbl.rows() + " rows of " + tableName);
            }
        }
        this.engramKv = weights.layer(layer, kvName);
        long kvWidth = (long) nEmbd * (hc + 1);
        if (engramKv.cols() != nEmbd || engramKv.rows() != kvWidth) {
            throw new IllegalArgumentException(where + ": " + kvName + " has shape [" + engramKv.cols() + ", "
                    + engramKv.rows() + "], expected [" + nEmbd + ", " + kvWidth + "]");
        }
        if (!GpuOps.supportsWeightType(engramKv.type())) {
            throw new UnsupportedOperationException(where + ": no GPU matvec kernels for " + kvName + " of type "
                    + engramKv.type());
        }
        GpuWeight gate = weights.layer(layer, gateName);
        if (gate.type() != GGMLType.F32 || !gate.isSingleShard() || gate.cols() * gate.rows() != (long) nEmbd * hc) {
            throw new IllegalArgumentException(where + ": " + gateName + " must hold " + ((long) nEmbd * hc)
                    + " F32 values, got " + gate);
        }
        this.gateW = gate;
        this.kvDeltas = ctx.projectionDeltas().serve(layer, kvName, engramKv);
        int u = ctx.scratch().ubatch();
        this.ubatch = u;
        GpuScratch s = ctx.scratch();
        this.emb = s.temp(TEMP_EMB, Math.multiplyExact((long) u, nEmbd));
        this.kvOut = s.temp(TEMP_KV, Math.multiplyExact((long) u, kvWidth));
        this.keyRows = s.temp(TEMP_KEY_ROWS, Math.multiplyExact((long) u, (long) hc * nEmbd));
        this.normX = s.temp(TEMP_NORM_X, Math.multiplyExact((long) u, (long) hc * nEmbd));
        this.normKey = s.temp(TEMP_NORM_KEY, Math.multiplyExact((long) u, (long) hc * nEmbd));
        this.activeGpu = s.temp(TEMP_ACTIVE, u);
        this.placeholders = ctx.placeholders().readOnlyFloats();
        this.window = new int[u + nGram - 1];
        this.stepIds = new int[u];
        this.toks = new long[nGram];
        this.ids = new int[Math.multiplyExact(u, nCols)];
        int local = opsLocal(ctx);
        int engramLocal = Math.min(Dsv4Kernels.LOCAL, local);
        Dsv4Kernels.checkEngram(u, nEmbd, hc, engramLocal);
        this.engramAdd = ctx.ops().prepare(Dsv4Kernels.ENGRAM_ADD, Dsv4Kernels.engramGlobal(u, hc, engramLocal),
                Dsv4Kernels.local(engramLocal),
                new Object[]{u, nEmbd, hc, 1.0f / (float) Math.sqrt(nEmbd), 0, activeGpu.buffer(), normX.buffer(),
                        normKey.buffer(), gateW.shard(0), kvOut.buffer(), ctx.placeholders().scratchFloats()});
        this.table = tbl;
        this.il = layer.il();
        this.scratch = s;
        this.memory = mem;
        this.ops = ctx.ops();
    }

    private static int opsLocal(GpuForwardContext ctx) {
        return ctx.ops().local();
    }

    /** Module index of Engram layer {@code wl}: the number of Engram layers before it. */
    private static int module(String where, ModelHparams hp, int wl) {
        if (!hp.isPleLayer(wl)) {
            throw new IllegalArgumentException(where + " lists '" + LayerSpec.Extras.ENGRAM
                    + "' but is not an Engram layer of the model");
        }
        int e = 0;
        for (int j = 0; j < wl; j++) {
            if (hp.isPleLayer(j)) {
                e++;
            }
        }
        return e;
    }

    private static long[] slice(long[] v, int from, int n, String where, String what) {
        if (from < 0 || (long) from + n > v.length) {
            throw new IllegalArgumentException(where + ": " + what + " holds " + v.length + " values, needs ["
                    + from + ", " + (from + n) + ") for this Engram module");
        }
        long[] out = new long[n];
        System.arraycopy(v, from, out, 0, n);
        return out;
    }

    @Override
    public void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
        requireBound();
        checkStep(nTokens, x);
        readStepTokens(nTokens);
        runSegment(pos0, nTokens, 0, 0, memory.sequence(), true, x);
    }

    /**
     * One batched step: every segment hashes its own rows against its own sequence's token record, so the segments
     * run one at a time (the paged memory's per-sequence record, positions per entry).
     */
    @Override
    public void beforeBlock(StepLayout s, ScratchSlot x) {
        requireBound();
        checkStep(s.nTokens(), x);
        readStepTokens(s.nTokens());
        PagedSequences paged = memory.pagedSequences();
        for (int seg = 0; seg < s.nSeg(); seg++) {
            SequenceState record = paged != null ? paged.sequence(s.seq(seg)) : memory.sequence();
            runSegment(s.pos0(seg), s.len(seg), (long) s.tokStart(seg) * hc * nEmbd, s.tokStart(seg), record,
                    paged == null, x);
        }
    }

    /** Copies the step's token ids to {@link #stepIds} (the hash may run over any subset of them). */
    private void readStepTokens(int n) {
        if (stepIds.length < n) {
            throw new IllegalStateException("Engram extra holds " + stepIds.length + " step ids, needs " + n);
        }
        scratch.readStepTokens(stepIds, 0, n);
    }

    private void requireBound() {
        if (ops == null) {
            throw new IllegalStateException("Engram extra is not bound");
        }
    }

    private void checkStep(int n, ScratchSlot x) {
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("Engram step of " + n + " rows is outside [1, " + ubatch + "]");
        }
        long need = (long) n * hc * nEmbd;
        if (x.floats() < need) {
            throw new IllegalArgumentException("residual stream holds " + x.floats() + " floats, the Engram"
                    + " layer needs " + need);
        }
    }

    /**
     * One segment of {@code n} rows at {@code x[xOff]} whose token ids are {@code stepIds[tokOff ..]}: the hash reads
     * the predecessors from {@code record} and the embedding is added to every hyper-connection stream of the row.
     */
    private void runSegment(int pos0, int n, long xOff, int tokOff, SequenceState record, boolean contiguous,
                            ScratchSlot x) {
        if (n == 0) {
            return;
        }
        fillWindow(record, contiguous, pos0, n, tokOff);
        CpuEngramExtra.hashRows(window, nGram - 1, n, nGram, perGram, nCols, multipliers, offsets, vocab, padToken,
                tokenMap, toks, ids);
        ByteBuffer activeBytes = ByteBuffer.allocate(n * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int t = 0; t < n; t++) {
            activeBytes.putInt(stepIds[tokOff + t] < 0 ? 0 : 1);
        }
        write(activeGpu, activeBytes.array());
        embedRows(n * nCols);
        ops.matmul(engramKv, emb.buffer(), 0, kvOut.buffer(), 0, n);
        kvDeltas.applyRows(n, emb.buffer(), 0, kvOut.buffer(), 0);
        int rows = n * hc;
        rowNorm(x.buffer(), (int) xOff, normX.buffer(), rows);
        ops.copy2d(kvOut.buffer(), 0, (hc + 1) * nEmbd, keyRows.buffer(), 0, hc * nEmbd, n, hc * nEmbd);
        rowNorm(keyRows.buffer(), 0, normKey.buffer(), rows);
        engramAdd.setInt(Dsv4Kernels.ARG_ENGRAM_N_TOKENS, n);
        engramAdd.setInt(Dsv4Kernels.ARG_ENGRAM_X_OFF, (int) xOff);
        engramAdd.setBuffer(Dsv4Kernels.ARG_ENGRAM_ACTIVE, activeGpu.buffer());
        engramAdd.setBuffer(Dsv4Kernels.ARG_ENGRAM_NORM_X, normX.buffer());
        engramAdd.setBuffer(Dsv4Kernels.ARG_ENGRAM_NORM_KEY, normKey.buffer());
        engramAdd.setBuffer(Dsv4Kernels.ARG_ENGRAM_GATE_W, gateW.shard(0));
        engramAdd.setBuffer(Dsv4Kernels.ARG_ENGRAM_KV_OUT, kvOut.buffer());
        engramAdd.setBuffer(Dsv4Kernels.ARG_ENGRAM_X, x.buffer());
        engramAdd.setInt(Dsv4Kernels.ARG_ENGRAM_HC, hc);
        engramAdd.setGlobal(Dsv4Kernels.engramGlobal(n, hc, Math.min(Dsv4Kernels.LOCAL, ops.local())));
        engramAdd.dispatch();
    }

    /**
     * Gathers the {@code rows} hashed embedding rows into {@link #emb}: one host-id {@code GpuOps.embed} per
     * {@code ubatch} rows, since a lookup call covers at most one step of ids while a token contributes one id per
     * Engram head.
     */
    private void embedRows(int rows) {
        for (int from = 0; from < rows; from += ubatch) {
            int count = Math.min(ubatch, rows - from);
            int[] chunk = new int[count];
            System.arraycopy(ids, from, chunk, 0, count);
            ops.embed(table, chunk, emb.buffer(), from * headDim);
        }
    }

    /**
     * {@code LLM_RowNorm} (RMS, no weight) over {@code rows} hyper-connection rows starting at {@code inOff}: the
     * per-stream rows of a step are wider than the step itself, so the call goes through the raw dispatch.
     */
    private void rowNorm(GpuBuffer in, int inOff, GpuBuffer out, int rows) {
        long local = ops.local();
        ops.enqueue(GpuOps.ROW_NORM, new long[]{(long) rows * local}, new long[]{local},
                NormKernels.rowNormArgs(NormKind.RMS, eps, rows, nEmbd, nEmbd, in, inOff, out, 0, null, null, 1,
                        placeholders));
    }

    private void write(ScratchSlot slot, byte[] bytes) {
        com.eonmux.neuralfree.acceleration.backend.GpuEvent e = ops.queue().enqueueWriteBuffer(slot.buffer(), 0,
                bytes.length, bytes);
        if (e != null) {
            e.close();
        }
    }

    /** The hash window of a segment: predecessors first, then the step's token ids (see {@code CpuEngramExtra}). */
    private void fillWindow(SequenceState record, boolean contiguous, int pos0, int n, int tokOff) {
        int back = nGram - 1;
        if (pos0 > 0 && (contiguous ? record.nPast() < pos0 : record.nPast() == 0)) {
            throw new IllegalStateException("Engram at position " + pos0 + " needs the tokens before it in the"
                    + " sequence record, which holds " + record.nPast());
        }
        for (int w = 0; w < back; w++) {
            int pos = pos0 - back + w;
            window[w] = pos < 0 ? -1 : record.tokenAtOrBefore(pos);
        }
        System.arraycopy(stepIds, tokOff, window, back, n);
    }

    @Override
    public void close() {
        RuntimeException failure = Closeables.close(engramAdd, null);
        engramAdd = null;
        ops = null;
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuEngramExtra{" + (ops == null ? "unbound" : "layer " + il + ", ngram " + nGram + "x" + perGram
                + ", hc " + hc) + '}';
    }
}
