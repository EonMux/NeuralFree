package com.eonmux.neuralfree.gguf.quant.write;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * The imatrix statistics report ({@code show_statistics} of llama.cpp {@code tools/imatrix/imatrix.cpp}):
 * per weight tensor the mean, standard deviation, skewness and kurtosis of the per-column averages — the plain
 * activations ({@code sum / count}) when the entry carries them, else the expected squared activations
 * ({@code sum2 / count}, the reference's legacy path) — the normalized entropy of the energy distribution, the
 * energy sum {@code ∑ E[A²]}, and against the same tensor of the previous block (within the trunk, or within the
 * NextN/MTP layers) the gain, L2 distance, cosine similarity, Pearson correlation and covariance of the per-column
 * averages — aggregated once more per block layer.
 *
 * <p>{@link #compute} takes the entries of one or more combined reads and returns the sorted {@link Report};
 * {@code format} renders it as the two fixed-width tables the reference prints, with the L2 distance columns when
 * no tensor is legacy and the NextN layers labelled {@code mtp<i>}. {@link #stored} gives the twelve values of
 * every tensor's {@code .stats} row that {@code save_imatrix} writes.</p>
 */
public final class ImatrixStatistics {

    private ImatrixStatistics() {
    }

    /** The statistics of one weight tensor, in the sorted order of the report. */
    public static final class Tensor {
        private final String name;
        private final String shortName;
        private final int layer;
        private final double sum;
        private final float mean;
        private final long elements;
        private final float stdDeviation;
        private final float skewness;
        private final float kurtosis;
        private final float hNorm;
        private final float gain;
        private final float l2Dist;
        private final float cossim;
        private final float pearson;
        private final float covariance;
        private final boolean legacy;

        Tensor(String name, String shortName, int layer, double sum, float mean, long elements, float stdDeviation,
               float skewness, float kurtosis, float hNorm, float gain, float l2Dist, float cossim, float pearson,
               float covariance, boolean legacy) {
            this.name = name;
            this.shortName = shortName;
            this.layer = layer;
            this.sum = sum;
            this.mean = mean;
            this.elements = elements;
            this.stdDeviation = stdDeviation;
            this.skewness = skewness;
            this.kurtosis = kurtosis;
            this.hNorm = hNorm;
            this.gain = gain;
            this.l2Dist = l2Dist;
            this.cossim = cossim;
            this.pearson = pearson;
            this.covariance = covariance;
            this.legacy = legacy;
        }

        /** True when the entry has no plain activation sums: the moments run over the squared-sum averages. */
        public boolean legacy() {
            return legacy;
        }

        /** The stored GGUF tensor name. */
        public String name() {
            return name;
        }

        /** The name between the block index and {@code weight}, or the whole name without a block. */
        public String shortName() {
            return shortName;
        }

        /** The block index, {@code -1} without a numeric one. */
        public int layer() {
            return layer;
        }

        /** The energy sum {@code ∑ E[A²]} over the collected columns. */
        public double sum() {
            return sum;
        }

        /** The mean of the per-column averages (plain activations, or squared ones for a legacy entry). */
        public float mean() {
            return mean;
        }

        /** The columns the statistics were computed over. */
        public long elements() {
            return elements;
        }

        public float stdDeviation() {
            return stdDeviation;
        }

        public float skewness() {
            return skewness;
        }

        public float kurtosis() {
            return kurtosis;
        }

        /** The normalized entropy of the energy distribution, {@code 100 * H / log2(elements)}. */
        public float hNorm() {
            return hNorm;
        }

        /** The RMS gain against the same tensor of the previous block; NaN without one. */
        public float gain() {
            return gain;
        }

        public float l2Dist() {
            return l2Dist;
        }

        public float cossim() {
            return cossim;
        }

        public float pearson() {
            return pearson;
        }

        public float covariance() {
            return covariance;
        }
    }

    /** The per-layer aggregation of the tensor statistics. */
    public static final class Layer {
        private final int layer;
        private final double sum;
        private final float gain;
        private final float l2Dist;
        private final float cossim;
        private final float pearson;
        private final float covariance;

        Layer(int layer, double sum, float gain, float l2Dist, float cossim, float pearson, float covariance) {
            this.layer = layer;
            this.sum = sum;
            this.gain = gain;
            this.l2Dist = l2Dist;
            this.cossim = cossim;
            this.pearson = pearson;
            this.covariance = covariance;
        }

        /** The block index; {@code -1} for the non-block tensors (the output head). */
        public int layer() {
            return layer;
        }

        /** The summed {@code ∑ E[A²]} of the layer's tensors. */
        public double sum() {
            return sum;
        }

        public float gain() {
            return gain;
        }

        public float l2Dist() {
            return l2Dist;
        }

        public float cossim() {
            return cossim;
        }

        public float pearson() {
            return pearson;
        }

        public float covariance() {
            return covariance;
        }
    }

    /** The computed report: the tensor rows in print order, the layer rows, and the layer-table gap flag. */
    public static final class Report {
        private final List<Tensor> tensors;
        private final List<Layer> layers;
        private final boolean layerGap;
        private final int nextnStart;
        private final boolean legacy;

        Report(List<Tensor> tensors, List<Layer> layers, boolean layerGap, int nextnStart) {
            this.tensors = List.copyOf(tensors);
            this.layers = List.copyOf(layers);
            this.layerGap = layerGap;
            this.nextnStart = nextnStart;
            boolean anyLegacy = false;
            for (Tensor t : tensors) {
                anyLegacy |= t.legacy();
            }
            this.legacy = anyLegacy;
        }

        /** The first NextN/MTP block, {@link Integer#MAX_VALUE} when the matrix has none. */
        public int nextnStart() {
            return nextnStart;
        }

        /** True when a tensor lacks plain activation sums, which prints the legacy columns. */
        public boolean legacy() {
            return legacy;
        }

        /** The table label of block {@code blk}: its index, {@code mtp<i>} for a NextN layer, {@code -} without. */
        public String label(int blk) {
            if (blk < 0 || blk == Integer.MAX_VALUE) {
                return "-";
            }
            return blk >= nextnStart ? "mtp" + (blk - nextnStart) : String.valueOf(blk);
        }

        /** The tensor rows, sorted by block index then tensor name, the output head last. */
        public List<Tensor> tensors() {
            return tensors;
        }

        /** The layer rows, empty when a layer sequence gap suppresses the layer table. */
        public List<Layer> layers() {
            return layers;
        }

        /** True when the collected block layers are not contiguous, which suppresses the layer table. */
        public boolean layerGap() {
            return layerGap;
        }

        /** The two fixed-width tables as the reference prints them (the legacy or the activation columns). */
        public String format() {
            StringBuilder out = new StringBuilder();
            out.append(String.format(Locale.ROOT, "%n%6s | %-40s | %10s%10s%12s%12s%9s | %17s%8s | ",
                    "Layer", "Tensor", "Mean", "StdDev", "Skew", "Kurt", "H Norm", "∑ E[A²]", "Gain"));
            out.append(legacy ? String.format(Locale.ROOT, "%10s%10s%n", "PCC", "Cov")
                    : String.format(Locale.ROOT, "%12s%10s%10s%n", "L2 Dist", "PCC", "Cov"));
            out.append(dashes(legacy ? 153 : 165)).append('\n');
            for (Tensor t : tensors) {
                out.append(String.format(Locale.ROOT, "%6s | %-40s | %10.4f%10.4f%12.4f%12.4f%8.2f%% | %14.4f%8.2f | ",
                        label(t.layer()), ellipsize(t.name(), 40), t.mean(), t.stdDeviation(), t.skewness(),
                        t.kurtosis(), t.hNorm(), t.sum(), t.gain()));
                out.append(legacy ? String.format(Locale.ROOT, "%10.4f%10.4f%n", t.pearson(), t.covariance())
                        : String.format(Locale.ROOT, "%12.4f%10.4f%10.4f%n", t.l2Dist(), t.pearson(),
                        t.covariance()));
            }
            if (layerGap) {
                out.append("\nlayer sequence gap detected; layer statistics will not be shown\n");
                return out.toString();
            }
            out.append(String.format(Locale.ROOT, "%n%6s | %17s%8s | ", "Layer", "∑ E[A²]", "Gain"));
            out.append(legacy ? String.format(Locale.ROOT, "%9s%9s%12s%n", "CosSim", "PCC", "Cov")
                    : String.format(Locale.ROOT, "%12s%9s%9s%12s%n", "L2 Dist", "CosSim", "PCC", "Cov"));
            out.append(dashes(legacy ? 64 : 76)).append('\n');
            for (Layer l : layers) {
                out.append(String.format(Locale.ROOT, "%6s | %14.4f%8.2f | ", label(l.layer()), l.sum(), l.gain()));
                out.append(legacy ? String.format(Locale.ROOT, "%9.4f%9.4f%12.4f%n", l.cossim(), l.pearson(),
                        l.covariance()) : String.format(Locale.ROOT, "%12.4f%9.4f%9.4f%12.4f%n", l.l2Dist(),
                        l.cossim(), l.pearson(), l.covariance()));
            }
            out.append('\n');
            return out.toString();
        }

        private static String dashes(int n) {
            StringBuilder out = new StringBuilder(n);
            for (int i = 0; i < n; i++) {
                out.append('-');
            }
            return out.toString();
        }

        private static String ellipsize(String s, int width) {
            return s.length() <= width ? s : ".." + s.substring(s.length() - (width - 2));
        }
    }

    // ------------------------------------------------------------------ computation

    /** {@link #compute(List, int)} of a matrix without a recorded NextN layer count. */
    public static Report compute(List<ImatrixFile.Entry> entries) {
        return compute(entries, 0);
    }

    /**
     * Computes the report over the entries of one read or a combined set: per tensor the vector statistics, then
     * the cross-layer comparison and the per-layer aggregation. Tensors whose matrices all count zero rows hold
     * no row, as the reference's statistics skip them.
     *
     * @param nLayerNextn the NextN/MTP layers among the top blocks ({@code imatrix.n_layer_nextn}); 0 finds them by
     *                    their {@code nextn.} tensors
     */
    public static Report compute(List<ImatrixFile.Entry> entries, int nLayerNextn) {
        List<Vector> vectors = vectors(entries, nLayerNextn);
        sort(vectors);
        int nextnStart = nextnLayerStart(vectors, nLayerNextn);
        return buildReport(vectors, nextnStart);
    }

    /**
     * The {@code .stats} row of every entry, by tensor name: the {@link ImatrixFile#STATS_SCHEMA} values
     * {@code sum_sq, mean, elements, std_deviation, skewness, kurtosis, gain, h_norm, l2_dist, cossim, pearson,
     * covariance} of llama.cpp's {@code save_imatrix}, zeros for an entry without statistics (no counted rows).
     */
    public static Map<String, float[]> stored(List<ImatrixFile.Entry> entries, int nLayerNextn) {
        Map<String, Tensor> rows = new TreeMap<>();
        for (Vector v : vectors(entries, nLayerNextn)) {
            rows.put(v.tensor, v.row());
        }
        Map<String, float[]> out = new TreeMap<>();
        for (ImatrixFile.Entry e : entries) {
            Tensor t = rows.get(e.tensor());
            out.put(e.tensor(), t == null ? new float[ImatrixFile.STATS_SCHEMA.size()] : new float[]{
                (float) t.sum(), t.mean(), (float) t.elements(), t.stdDeviation(), t.skewness(), t.kurtosis(),
                t.gain(), t.hNorm(), t.l2Dist(), t.cossim(), t.pearson(), t.covariance()});
        }
        return out;
    }

    /** The vector statistics of every entry with counted rows, compared across layers. */
    private static List<Vector> vectors(List<ImatrixFile.Entry> entries, int nLayerNextn) {
        if (nLayerNextn < 0) {
            throw new IllegalArgumentException("the NextN layer count " + nLayerNextn + " is negative");
        }
        List<Vector> vectors = new ArrayList<>();
        for (ImatrixFile.Entry e : entries) {
            Vector v = vectorStats(e);
            if (v != null) {
                vectors.add(v);
            }
        }
        crossLayer(vectors, nextnLayerStart(vectors, nLayerNextn));
        return vectors;
    }

    /** Sort key of the reference's comparer: block index, output tensors last, then the short name. */
    private static int sortKey(Vector v, int fallback) {
        if (v.layer >= 0) {
            return v.layer;
        }
        return v.tensor.contains("output") ? Integer.MAX_VALUE : fallback;
    }

    private static void sort(List<Vector> vectors) {
        vectors.sort((a, b) -> {
            int ka = sortKey(a, Integer.MAX_VALUE - 1);
            int kb = sortKey(b, Integer.MAX_VALUE - 1);
            if (ka != kb) {
                return Integer.compare(ka, kb);
            }
            return a.shortName.compareTo(b.shortName);
        });
    }

    /**
     * The first NextN block (llama.cpp {@code nextn_layer_start}): the top {@code nLayerNextn} blocks when the
     * count is known, else the first block holding a {@code nextn.} tensor, else the maximum value.
     */
    private static int nextnLayerStart(List<Vector> vectors, int nLayerNextn) {
        int first = Integer.MAX_VALUE;
        int maxBlk = -1;
        for (Vector v : vectors) {
            if (v.layer < 0) {
                continue;
            }
            maxBlk = Math.max(maxBlk, v.layer);
            if (v.shortName.startsWith("nextn.")) {
                first = Math.min(first, v.layer);
            }
        }
        if (nLayerNextn > 0 && maxBlk >= 0) {
            return Math.max(0, maxBlk + 1 - nLayerNextn);
        }
        return first;
    }

    /** The per-column statistics of one entry's squared-activation sums; null when no matrix counted rows. */
    private static Vector vectorStats(ImatrixFile.Entry e) {
        Moments run = firstPass(e);
        if (run.valid == 0) {
            return null;
        }
        double[] dev = secondPass(e, run);
        double variance = run.valid > 1 ? dev[0] / run.valid : 0.0;
        float std = (float) Math.sqrt(Math.max(variance, 0.0));
        float skew = 0.0f;
        float kurt = 0.0f;
        if (std > 1e-10f) {
            double m2 = dev[0] / run.valid;
            skew = (float) (dev[1] / run.valid / (m2 * Math.sqrt(m2)));
            kurt = (float) (dev[2] / run.valid / (m2 * m2) - 3.0);
        }
        return new Vector(e.tensor(), shortName(e.tensor()), layerOf(e.tensor()), run.sum, (float) run.mean,
                run.valid, std, skew, kurt, (float) dev[3], e);
    }

    /**
     * The running energy sum, the mean of the column averages (plain activations, or the energies of a legacy
     * entry), the positive energy sum and the valid-column count (Welford), over the columns whose average and
     * energy are both finite.
     */
    private static Moments firstPass(ImatrixFile.Entry e) {
        Moments out = new Moments();
        Columns c = new Columns(e);
        for (int m = 0; m < e.nMat(); m++) {
            if (c.counts[m] <= 0) {
                continue;
            }
            double invC = 1.0 / c.counts[m];
            for (int j = m * e.cols(); j < (m + 1) * e.cols(); j++) {
                double val = c.sumSq[j] * invC;
                double v = c.sums == null ? val : c.sums[j] * invC;
                if (!Double.isFinite(v) || !Double.isFinite(val)) {
                    continue;
                }
                out.sum += val;
                out.valid++;
                out.mean += (v - out.mean) / out.valid;
                if (val > 0.0) {
                    out.sumEnergy += val;
                }
            }
        }
        return out;
    }

    /** The central moments of the column averages around {@code run.mean} and the entropy of the energies. */
    private static double[] secondPass(ImatrixFile.Entry e, Moments run) {
        double[] dev = new double[4]; // {sqDiff, cuDiff, qdDiff, entropy}
        double invEnergy = run.sumEnergy > 0.0 ? 1.0 / run.sumEnergy : 0.0;
        double log2Inv = 1.0 / Math.log(2.0);
        Columns c = new Columns(e);
        for (int m = 0; m < e.nMat(); m++) {
            if (c.counts[m] <= 0) {
                continue;
            }
            double invC = 1.0 / c.counts[m];
            for (int j = m * e.cols(); j < (m + 1) * e.cols(); j++) {
                double val = c.sumSq[j] * invC;
                double v = c.sums == null ? val : c.sums[j] * invC;
                if (!Double.isFinite(v) || !Double.isFinite(val)) {
                    continue;
                }
                double diff = v - run.mean;
                dev[0] += diff * diff;
                dev[1] += diff * diff * diff;
                dev[2] += diff * diff * diff * diff;
                if (invEnergy > 0.0) {
                    double p = Math.max(0.0, val) * invEnergy;
                    if (p > 1e-10) {
                        dev[3] -= p * Math.log(p) * log2Inv;
                    }
                }
            }
        }
        return dev;
    }

    /** One entry's arrays, copied once per pass. */
    private static final class Columns {
        final float[] sumSq;
        final float[] sums;
        final long[] counts;

        Columns(ImatrixFile.Entry e) {
            this.sumSq = e.sumSq();
            this.sums = e.sums();
            this.counts = e.counts();
        }
    }

    /** The first-pass accumulators of one entry's column means. */
    private static final class Moments {
        double sum;
        double mean;
        double sumEnergy;
        long valid;
    }

    /** The cross-layer comparison of llama.cpp {@code compute_tensor_statistics}, against the previous block. */
    private static void crossLayer(List<Vector> vectors, int nextnStart) {
        Map<String, Integer> index = new TreeMap<>();
        for (int i = 0; i < vectors.size(); i++) {
            index.put(vectors.get(i).tensor, i);
        }
        for (Vector v : vectors) {
            if (v.layer <= 0 || v.layer == nextnStart) {
                continue;
            }
            int first = v.layer > nextnStart ? nextnStart : 0;
            String tag = "blk." + v.layer;
            int at = v.tensor.indexOf(tag);
            if (at < 0) {
                continue;
            }
            Vector prev = null;
            for (int p = v.layer - 1; p >= first && prev == null; p--) {
                String candidate = v.tensor.substring(0, at) + "blk." + p + v.tensor.substring(at + tag.length());
                Integer at2 = index.get(candidate);
                prev = at2 == null ? null : vectors.get(at2);
            }
            if (prev == null) {
                continue;
            }
            boolean activations = v.entry.hasSums() && prev.entry.hasSums();
            float[] curr = averages(v, activations);
            float[] before = averages(prev, activations);
            if (curr == null || before == null || curr.length != before.length) {
                continue;
            }
            if (!compare(v, prev, curr, before)) {
                continue;
            }
            // the reference records the previous energies with the gain, only over a previous side with energy
            if (prev.sum > 1e-10) {
                v.prevEnergy = new double[]{v.sum, prev.sum, v.elements, prev.elements};
            }
        }
    }

    /**
     * The flat per-column averages over every matrix with counted rows, of the plain sums when
     * {@code activations} else of the squared sums, NaN elsewhere; null when no matrix counted rows.
     */
    private static float[] averages(Vector v, boolean activations) {
        ImatrixFile.Entry e = v.entry;
        float[] out = new float[e.nMat() * e.cols()];
        Arrays.fill(out, Float.NaN);
        float[] sumSq = activations ? e.sums() : e.sumSq();
        long[] counts = e.counts();
        boolean hasValid = false;
        for (int m = 0; m < e.nMat(); m++) {
            if (counts[m] <= 0) {
                continue;
            }
            hasValid = true;
            float invC = (float) (1.0 / counts[m]);
            for (int j = 0; j < e.cols(); j++) {
                out[m * e.cols() + j] = sumSq[m * e.cols() + j] * invC;
            }
        }
        return hasValid ? out : null;
    }

    /**
     * Fills the comparison fields of {@code v} from the two per-column average vectors; false, leaving them unset,
     * when no aligned pair is finite.
     */
    private static boolean compare(Vector v, Vector prev, float[] curr, float[] before) {
        double[] moments = alignedSums(curr, before);
        if (moments[0] == 0) {
            return false;
        }
        v.nFeatures = (long) moments[0];
        v.dotProd = moments[1];
        v.norm1Sq = moments[2];
        v.norm2Sq = moments[3];
        v.l2Sq = moments[4];
        double meanC = moments[0] > 0 ? moments[5] / moments[0] : 0.0;
        double meanP = moments[0] > 0 ? moments[6] / moments[0] : 0.0;
        double[] centered = centeredSums(curr, before, meanC, meanP);
        v.covSum = centered[0];
        v.varCSum = centered[1];
        v.varPSum = centered[2];
        v.l2Dist = (float) Math.sqrt(v.l2Sq);
        if (moments[0] > 1) {
            v.covariance = (float) (centered[0] / moments[0]);
        }
        v.cossim = clampCos(v.norm1Sq, v.norm2Sq, v.dotProd);
        v.pearson = clampCos(v.varCSum, v.varPSum, v.covSum);
        if (prev.sum > 1e-10) {
            v.gain = (float) (Math.sqrt(v.sum / v.elements) / Math.sqrt(prev.sum / prev.elements));
        } else {
            v.gain = (float) (v.sum <= 1e-10 ? 1.0 : Double.NaN);
        }
        return true;
    }

    /** The aligned-pair sums {valid, dot, norm1², norm2², l2², sumC, sumP} of the two vectors. */
    private static double[] alignedSums(float[] curr, float[] before) {
        double[] out = new double[7];
        for (int i = 0; i < curr.length; i++) {
            double c = curr[i];
            double p = before[i];
            if (!Double.isFinite(c) || !Double.isFinite(p)) {
                continue;
            }
            out[0] += 1;
            out[1] += c * p;
            out[2] += c * c;
            out[3] += p * p;
            double diff = c - p;
            out[4] += diff * diff;
            out[5] += c;
            out[6] += p;
        }
        return out;
    }

    /** The centered sums {cov, varC, varP} of the two vectors around their own means. */
    private static double[] centeredSums(float[] curr, float[] before, double meanC, double meanP) {
        double[] out = new double[3];
        for (int i = 0; i < curr.length; i++) {
            double c = curr[i];
            double p = before[i];
            if (!Double.isFinite(c) || !Double.isFinite(p)) {
                continue;
            }
            double dc = c - meanC;
            double dp = p - meanP;
            out[0] += dc * dp;
            out[1] += dc * dc;
            out[2] += dp * dp;
        }
        return out;
    }

    /** Cosine/Pearson value of the two moment sums over {@code dot}, clamped to [-1, 1] (the reference rules). */
    private static float clampCos(double norm1Sq, double norm2Sq, double dot) {
        if (norm1Sq > 0.0 && norm2Sq > 0.0) {
            return clamp((float) (dot / (Math.sqrt(norm1Sq) * Math.sqrt(norm2Sq))));
        }
        return norm1Sq == 0.0 && norm2Sq == 0.0 ? Float.NaN : 0.0f;
    }

    private static float clamp(float v) {
        return Math.max(-1.0f, Math.min(1.0f, v));
    }

    /** The per-layer aggregation of llama.cpp {@code compute_layer_statistics}, plus the gap check. */
    private static Report buildReport(List<Vector> vectors, int nextnStart) {
        TreeMap<Integer, Agg> aggs = new TreeMap<>();
        List<Tensor> tensors = new ArrayList<>(vectors.size());
        for (Vector v : vectors) {
            tensors.add(v.row());
            if (v.layer <= 0) {
                continue;
            }
            if (v.norm1Sq == 0.0 && v.norm2Sq == 0.0 && v.l2Sq == 0.0) {
                continue;
            }
            Agg a = aggs.computeIfAbsent(v.layer, k -> new Agg());
            a.sum += v.sum;
            a.dotProd += v.dotProd;
            a.norm1Sq += v.norm1Sq;
            a.norm2Sq += v.norm2Sq;
            a.l2Sq += v.l2Sq;
            a.covSum += v.covSum;
            a.varC += v.varCSum;
            a.varP += v.varPSum;
            a.nFeatures += Math.max(0, v.nFeatures);
            addEnergies(a, v);
        }
        boolean gap = layerGap(aggs.keySet(), nextnStart);
        List<Layer> layers = new ArrayList<>();
        if (!gap) {
            for (Map.Entry<Integer, Agg> e : aggs.entrySet()) {
                layers.add(layerRow(e.getKey(), e.getValue()));
            }
        }
        return new Report(tensors, layers, gap, nextnStart);
    }

    /** Adds one tensor's prev/curr energy pair to the layer aggregate (the reference's elements_prev gate). */
    private static void addEnergies(Agg a, Vector v) {
        double[] pair = v.prevEnergy;
        if (pair == null) {
            return;
        }
        a.energyCurr += pair[0];
        a.energyPrev += pair[1];
        a.elementsCurr += (long) pair[2];
        a.elementsPrev += (long) pair[3];
    }

    /** True when the collected trunk layers below {@code nextnStart} are not a contiguous range. */
    private static boolean layerGap(Iterable<Integer> layers, int nextnStart) {
        int min = Integer.MAX_VALUE;
        int max = -1;
        int n = 0;
        for (Integer l : layers) {
            if (l >= 0 && l < nextnStart) {
                n++;
                min = Math.min(min, l);
                max = Math.max(max, l);
            }
        }
        return n > 0 && n != max - min + 1;
    }

    private static Layer layerRow(int layer, Agg a) {
        float cossim = clampCos(a.norm1Sq, a.norm2Sq, a.dotProd);
        float gain = Float.NaN;
        if (a.elementsCurr > 0 && a.elementsPrev > 0 && a.energyPrev > 0.0) {
            gain = (float) (Math.sqrt(a.energyCurr / a.elementsCurr) / Math.sqrt(a.energyPrev / a.elementsPrev));
        }
        float l2 = (float) Math.sqrt(a.l2Sq);
        float covariance = a.nFeatures > 0 ? (float) (a.covSum / a.nFeatures) : Float.NaN;
        float pearson = clampCos(a.varC, a.varP, a.covSum);
        return new Layer(layer, a.sum, gain, l2, cossim, pearson, covariance);
    }

    /** One layer's moment sums (llama.cpp {@code layer_aggregation}). */
    private static final class Agg {
        double sum;
        double dotProd;
        double norm1Sq;
        double norm2Sq;
        double l2Sq;
        double covSum;
        double varC;
        double varP;
        double energyCurr;
        double energyPrev;
        long elementsCurr;
        long elementsPrev;
        long nFeatures;
    }

    /** The mutable computation record behind one report row. */
    private static final class Vector {
        final String tensor;
        final String shortName;
        final int layer;
        final double sum;
        final float mean;
        final long elements;
        final float stdDeviation;
        final float skewness;
        final float kurtosis;
        final float entropy;
        final ImatrixFile.Entry entry;
        float gain = Float.NaN;
        float l2Dist = Float.NaN;
        float cossim = Float.NaN;
        float pearson = Float.NaN;
        float covariance = Float.NaN;
        double dotProd;
        double norm1Sq;
        double norm2Sq;
        double l2Sq;
        double covSum;
        double varCSum;
        double varPSum;
        long nFeatures;
        /** {currEnergy, prevEnergy, currElements, prevElements} of the pair, set with the energies. */
        double[] prevEnergy;

        Vector(String tensor, String shortName, int layer, double sum, float mean, long elements, float stdDeviation,
               float skewness, float kurtosis, float entropy, ImatrixFile.Entry entry) {
            this.tensor = tensor;
            this.shortName = shortName;
            this.layer = layer;
            this.sum = sum;
            this.mean = mean;
            this.elements = elements;
            this.stdDeviation = stdDeviation;
            this.skewness = skewness;
            this.kurtosis = kurtosis;
            this.entropy = entropy;
            this.entry = entry;
        }

        /** The immutable report row of this record. */
        Tensor row() {
            float hNorm = elements > 1 ? (float) (100.0 * entropy * Math.log(2.0) / Math.log(elements)) : Float.NaN;
            return new Tensor(tensor, shortName, layer, sum, mean, elements, stdDeviation, skewness, kurtosis,
                    hNorm, gain, l2Dist, cossim, pearson, covariance, !entry.hasSums());
        }
    }

    /** The block index of a stored tensor name, {@code -1} without a numeric one (the reference's parser). */
    private static int layerOf(String tensor) {
        String[] parts = tensor.split("\\.");
        for (int i = 0; i < parts.length - 1; i++) {
            if (parts[i].equals("blk")) {
                try {
                    return Integer.parseInt(parts[i + 1]);
                } catch (NumberFormatException e) {
                    return -1;
                }
            }
        }
        return -1;
    }

    /** The name between the block index and {@code weight}, or the whole name without one (the reference). */
    private static String shortName(String tensor) {
        String[] parts = tensor.split("\\.");
        int weightAt = -1;
        for (int i = 1; i < parts.length; i++) {
            if (parts[i].equals("weight")) {
                weightAt = i;
                break;
            }
        }
        if (weightAt < 0) {
            return tensor;
        }
        StringBuilder out = new StringBuilder();
        for (int j = 0; j < parts.length; j++) {
            if (parts[j].equals("blk")) {
                j += parts.length > 4 ? 1 : 2;
                continue;
            }
            if (j == weightAt) {
                break;
            }
            if (out.length() > 0) {
                out.append('.');
            }
            out.append(parts[j]);
        }
        return out.length() > 0 ? out.toString() : tensor;
    }
}
