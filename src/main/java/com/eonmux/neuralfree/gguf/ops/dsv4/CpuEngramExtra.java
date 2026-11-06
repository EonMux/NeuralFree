package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * The DeepSeek-V4.1 Engram n-gram embedding of one layer ({@code LayerSpec.Extras.ENGRAM}, models/deepseek41.cpp:
 * 88-131, 240-292, 456-520): a per-layer n-gram hash gathers {@code ple_head_dim}-wide rows of the shared
 * {@code per_layer_token_embd} table, projects them through the layer's {@code engram_kv} into one key per
 * hyper-connection stream and one value, and adds {@code value * gate} to every stream, where the gate scores the
 * layer input against its key.
 *
 * <p>The hash reads the step's tokens from {@link CpuScratch#TOKEN_IDS} and their predecessors from the memory's
 * {@link SequenceState}, like {@code CpuNgramPleExtra}; an embedding row (a null token id) is hashed as the padding
 * token and receives no contribution. The hash arithmetic is unsigned 64-bit, as in {@code engram.py}. The extra
 * runs before the hyper-connection extra, on the {@code hc * n_embd}-wide residual stream. Not thread-safe.</p>
 */
public final class CpuEngramExtra implements CpuLayerExtra {

    private static final String TEMP_EMB = "dsv4.engram.emb";
    private static final String TEMP_KV = "dsv4.engram.kv";
    private static final String TEMP_KEY_ROWS = "dsv4.engram.keys";
    private static final String TEMP_NORM_X = "dsv4.engram.x";
    private static final String TEMP_NORM_KEY = "dsv4.engram.normkey";
    private static final String TEMP_GATE = "dsv4.engram.gate";

    private static final int ROW_GRAIN = 16;

    private CpuMatVec matVec;
    private CpuWeights weights;
    private CpuScratch scratch;
    private CpuMemory memory;
    private int il;
    private int ubatch;
    private int nEmbd;
    private int hc;
    private int headDim;
    private int nGram;
    private int perGram;
    private int nCols;
    private int padToken;
    private int[] tokenMap;
    private long[] multipliers;
    private long[] offsets;
    private long[] vocab;
    private float eps;
    private CpuTensorRef table;
    private CpuTensorRef engramKv;
    private CpuProjectionDeltas.Site kvDeltas;
    private float[] gateW;
    private long[] toks;
    private int[] window;
    private int[] ids;
    private float[] emb;
    private float[] kvOut;
    private float[] keyRows;
    private float[] normX;
    private float[] normKey;
    private float[] gates;

    /** An unbound Engram extra; {@link #bind} resolves its layer's hash module and tensors. */
    public CpuEngramExtra() {
    }

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (matVec != null) {
            throw new IllegalStateException("Engram extra is already bound");
        }
        ModelSpec spec = ctx.spec();
        ModelHparams hp = spec.hparams();
        String arch = spec.arch().ggufName();
        String where = arch + " layer " + layer.stack() + "." + layer.il();
        if (!ctx.hasMemory()) {
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
        CpuWeights w = ctx.weights();
        String tableName = TensorNames.weight(TensorNames.PER_LAYER_TOKEN_EMBD);
        String kvName = TensorNames.weight(TensorNames.ENGRAM_KV);
        String gateName = TensorNames.weight(TensorNames.ENGRAM_GATE);
        CpuTensorRef tbl = w.get(tableName);
        if (tbl.cols() != headDim) {
            throw new IllegalArgumentException(where + ": " + tableName + " rows hold " + tbl.cols()
                    + " floats, expected ple_head_dim " + headDim);
        }
        for (int c = 0; c < nCols; c++) {
            if (offsets[c] + vocab[c] > tbl.rows()) {
                throw new IllegalArgumentException(where + ": Engram head " + c + " rows [" + offsets[c] + ", "
                        + (offsets[c] + vocab[c]) + ") fall outside the " + tbl.rows() + " rows of " + tableName);
            }
        }
        this.engramKv = w.layer(layer, kvName);
        long kvWidth = (long) nEmbd * (hc + 1);
        if (engramKv.cols() != nEmbd || engramKv.rows() != kvWidth) {
            throw new IllegalArgumentException(where + ": " + kvName + " has shape [" + engramKv.cols()
                    + ", " + engramKv.rows() + "], expected [" + nEmbd + ", " + kvWidth + "]");
        }
        CpuTensorRef engramGate = w.layer(layer, gateName);
        if (engramGate.cols() != nEmbd || engramGate.rows() != hc) {
            throw new IllegalArgumentException(where + ": " + gateName + " has shape ["
                    + engramGate.cols() + ", " + engramGate.rows() + "], expected [" + nEmbd + ", " + hc + "]");
        }
        this.gateW = w.layerFloatsOrNull(layer, gateName);
        if (gateW == null) {
            throw new IllegalArgumentException(where + ": missing " + gateName);
        }
        if (gateW.length != (long) nEmbd * hc) {
            throw new IllegalArgumentException(where + ": " + gateName + " holds " + gateW.length
                    + " values, expected " + ((long) nEmbd * hc));
        }
        this.kvDeltas = ctx.projectionDeltas().serve(layer, kvName, engramKv, ctx.matVec());
        int u = ctx.scratch().ubatch();
        this.ubatch = u;
        this.window = new int[u + nGram - 1];
        this.ids = new int[Math.multiplyExact(u, nCols)];
        this.toks = new long[nGram];
        CpuScratch s = ctx.scratch();
        this.emb = s.temp(TEMP_EMB, Math.multiplyExact(u, nEmbd));
        this.kvOut = s.temp(TEMP_KV, Math.multiplyExact(u, Math.multiplyExact(nEmbd, hc + 1)));
        this.keyRows = s.temp(TEMP_KEY_ROWS, Math.multiplyExact(u, Math.multiplyExact(hc, nEmbd)));
        this.normX = s.temp(TEMP_NORM_X, Math.multiplyExact(u, Math.multiplyExact(hc, nEmbd)));
        this.normKey = s.temp(TEMP_NORM_KEY, Math.multiplyExact(u, Math.multiplyExact(hc, nEmbd)));
        this.gates = s.temp(TEMP_GATE, Math.multiplyExact(u, hc));
        this.table = tbl;
        this.il = layer.il();
        this.weights = w;
        this.scratch = s;
        this.memory = ctx.memory();
        this.matVec = ctx.matVec();
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
    public void beforeBlock(int pos0, int n, float[] x) {
        checkBound();
        checkStep(n, x);
        int[] step = scratch.ints(CpuScratch.TOKEN_IDS);
        runSegment(pos0, n, 0, step, 0, memory.sequence(), true, x);
    }

    /**
     * One batched step: every segment hashes its own rows against its own sequence's token record, so the segments
     * run one at a time (the paged memory's per-sequence record, positions per entry).
     */
    @Override
    public void beforeBlock(StepLayout s, float[] x) {
        checkBound();
        checkStep(s.nTokens(), x);
        int[] step = scratch.ints(CpuScratch.TOKEN_IDS);
        boolean paged = memory.pagedSequences() != null;
        for (int seg = 0; seg < s.nSeg(); seg++) {
            SequenceState record = paged ? memory.pagedSequences().sequence(s.seq(seg)) : memory.sequence();
            runSegment(s.pos0(seg), s.len(seg), (long) s.tokStart(seg) * hc * nEmbd, step, s.tokStart(seg), record,
                    !paged, x);
        }
    }

    private void checkBound() {
        if (matVec == null) {
            throw new IllegalStateException("Engram extra is not bound");
        }
    }

    private void checkStep(int n, float[] x) {
        if (n < 0 || n > ubatch) {
            throw new IllegalArgumentException("Engram step of " + n + " rows is outside [1, " + ubatch + "]");
        }
        long need = (long) n * hc * nEmbd;
        if (x.length < need) {
            throw new IllegalArgumentException("residual stream holds " + x.length + " floats, the Engram layer"
                    + " needs " + need);
        }
    }

    /**
     * One segment of {@code n} rows at {@code x[xOff]} whose token ids are {@code step[tokOff ..]}: the hash reads
     * the predecessors from {@code record} and the embedding is added to every hyper-connection stream of the row.
     */
    private void runSegment(int pos0, int n, long xOff, int[] step, int tokOff, SequenceState record,
                            boolean contiguous, float[] x) {
        if (n == 0) {
            return;
        }
        fillWindow(record, contiguous, pos0, n, step, tokOff);
        hash(n);
        for (int i = 0; i < n * nCols; i++) {
            weights.embedRow(table, ids[i], emb, i * headDim);
        }
        matVec.matmul(engramKv, emb, 0, kvOut, 0, n);
        kvDeltas.applyRows(n, emb, 0, kvOut, 0);
        int rows = n * hc;
        NormMath.rowNorm(NormKind.RMS, eps, rows, nEmbd, nEmbd, 1, x, (int) xOff, null, null, normX, 0);
        for (int r = 0; r < rows; r++) {
            int t = r / hc;
            int h = r - t * hc;
            System.arraycopy(kvOut, t * (hc + 1) * nEmbd + h * nEmbd, keyRows, r * nEmbd, nEmbd);
        }
        NormMath.rowNorm(NormKind.RMS, eps, rows, nEmbd, nEmbd, 1, keyRows, 0, null, null, normKey, 0);
        final float scale = 1.0f / (float) Math.sqrt((float) nEmbd);
        CpuParallel.parallelFor(rows, ROW_GRAIN, (from, to) -> {
            for (int r = from; r < to; r++) {
                int t = r / hc;
                int h = r - t * hc;
                int base = r * nEmbd;
                int gateBase = h * nEmbd;
                float dot = 0f;
                for (int i = 0; i < nEmbd; i++) {
                    dot += normX[base + i] * normKey[base + i] * gateW[gateBase + i];
                }
                gates[r] = step[tokOff + t] < 0 ? 0f : gate(dot * scale);
            }
        });
        CpuParallel.parallelFor(rows, ROW_GRAIN, (from, to) -> {
            for (int r = from; r < to; r++) {
                int t = r / hc;
                float g = gates[r];
                int valueOff = t * (hc + 1) * nEmbd + hc * nEmbd;
                int rowOff = (int) xOff + r * nEmbd;
                for (int i = 0; i < nEmbd; i++) {
                    x[rowOff + i] += kvOut[valueOff + i] * g;
                }
            }
        });
    }

    /** The hash window of a segment: predecessors first, then the step's token ids (see {@link #hash(int)}). */
    private void fillWindow(SequenceState record, boolean contiguous, int pos0, int n, int[] step, int tokOff) {
        int back = nGram - 1;
        if (pos0 > 0 && (contiguous ? record.nPast() < pos0 : record.nPast() == 0)) {
            throw new IllegalStateException("Engram at position " + pos0 + " needs the tokens before it in the"
                    + " sequence record, which holds " + record.nPast());
        }
        for (int w = 0; w < back; w++) {
            int pos = pos0 - back + w;
            window[w] = pos < 0 ? -1 : record.tokenAtOrBefore(pos);
        }
        System.arraycopy(step, tokOff, window, back, n);
    }

    /**
     * The n-gram row ids of {@code n} tokens (engram.py {@code NgramHashState.forward}): shift 0 is the token
     * itself, shift {@code s} the token {@code s} positions back; the first missing token (sequence start, an
     * embedding row or an id outside the map) blocks that shift and every longer one, and blocked slots hash as the
     * padding token. The unsigned 64-bit rolling hash {@code XOR_s toks[s] * mult[s]} selects, for each head of the
     * n-gram {@code o}, row {@code rolling mod vocab + offset}.
     */
    private void hash(int n) {
        hashRows(window, nGram - 1, n, nGram, perGram, nCols, multipliers, offsets, vocab, padToken, tokenMap, toks,
                ids);
    }

    /**
     * {@link #hash(int)} over explicit hash inputs, the unit-parity form: {@code window[back .. back + n)} holds the
     * step's token ids and {@code window[0 .. back)} the {@code back} predecessors (see {@link #fillWindow}).
     *
     * @param toks scratch of at least {@code nGram} entries
     * @param ids  receives {@code n * nCols} row ids
     * @throws IllegalArgumentException when an array is too short or a hash input is inconsistent
     */
    static void hashRows(int[] window, int back, int n, int nGram, int perGram, int nCols, long[] multipliers,
                         long[] offsets, long[] vocab, int padToken, int[] tokenMap, long[] toks, int[] ids) {
        Objects.requireNonNull(window, "window");
        Objects.requireNonNull(multipliers, "multipliers");
        Objects.requireNonNull(offsets, "offsets");
        Objects.requireNonNull(vocab, "vocab");
        Objects.requireNonNull(tokenMap, "tokenMap");
        Objects.requireNonNull(toks, "toks");
        Objects.requireNonNull(ids, "ids");
        if (nGram < 2 || perGram < 1 || nCols != (nGram - 1) * perGram || back != nGram - 1) {
            throw new IllegalArgumentException("Engram hash needs n_gram >= 2, n_cols = (n_gram - 1) * heads, got "
                    + nGram + "/" + perGram + "/" + nCols + " with back " + back);
        }
        if (multipliers.length < nGram || offsets.length < nCols || vocab.length < nCols
                || toks.length < nGram || ids.length < (long) n * nCols || window.length < back + n) {
            throw new IllegalArgumentException("Engram hash arrays are too short for n_gram " + nGram + ", n_cols "
                    + nCols + " and n " + n);
        }
        long pad = padToken;
        for (int t = 0; t < n; t++) {
            long c = mapped(tokenMap, window[back + t]);
            boolean blocked = c < 0;
            toks[0] = blocked ? pad : c;
            for (int s = 1; s < nGram; s++) {
                c = mapped(tokenMap, window[back + t - s]);
                blocked |= c < 0;
                toks[s] = blocked ? pad : c;
            }
            long rolling = toks[0] * multipliers[0];
            for (int o = 1; o < nGram; o++) {
                rolling ^= toks[o] * multipliers[o];
                for (int h = 0; h < perGram; h++) {
                    int col = (o - 1) * perGram + h;
                    ids[t * nCols + col] = (int) (Long.remainderUnsigned(rolling, vocab[col]) + offsets[col]);
                }
            }
        }
    }

    /** Token id through {@code engram.token_map}, or -1 for an embedding row or an id outside the map. */
    static long mapped(int[] tokenMap, int token) {
        return token < 0 || token >= tokenMap.length ? -1L : tokenMap[token];
    }

    /** {@code sigmoid(copysign(sqrt(max(|s|, 1e-6)), s))}; zero takes the positive sign, as {@code copysign} does. */
    static float gate(float s) {
        float mag = (float) Math.sqrt(Math.max(Math.abs(s), 1e-6f));
        return ActMath.sigmoid(s < 0f ? -mag : mag);
    }

    /** The Engram hash and addition are per-segment over its own token record, so batching is supported. */
    @Override
    public String batchRefusal() {
        return null;
    }

    @Override
    public String toString() {
        return "CpuEngramExtra{" + (matVec == null ? "unbound" : "layer " + il + ", ngram " + nGram + "x" + perGram
                + ", hc " + hc) + '}';
    }
}
