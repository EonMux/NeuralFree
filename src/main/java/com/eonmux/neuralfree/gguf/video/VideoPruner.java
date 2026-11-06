package com.eonmux.neuralfree.gguf.video;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Video token pruning, the port of vLLM's selectors (vllm/multimodal/video_prune): {@link Method#EVS} keeps the
 * tokens that differ most from the frame before them, always keeping every token of the first frame (evs.py
 * {@code compute_retention_mask}); {@link Method#VIDCOM2} scores every token by multi-scale Gaussian similarity to
 * the video and per-frame centers, gives distinctive frames a larger budget and keeps each frame's least similar
 * tokens (vidcom2.py). Both keep {@code (1 - q)} of a video's rows; EVS keeps at least one frame's rows and VidCom2
 * at least one row per frame. Static.
 */
public final class VideoPruner {

    /** The pruning algorithms of vLLM's {@code video_pruning_method}. */
    public enum Method {
        /** Elastic Video Sampling: cosine dissimilarity to the previous frame, first frame always kept. */
        EVS,
        /** VidCom2 (Liu et al., EMNLP 2025): multi-scale Gaussian similarity to the video and frame centers. */
        VIDCOM2;

        /**
         * The method of vLLM's name, case-insensitively and with surrounding space ignored.
         *
         * @throws IllegalArgumentException for a name other than {@code evs} or {@code vidcom2}
         */
        public static Method of(String name) {
            String key = name == null ? "" : name.trim().toLowerCase(Locale.ROOT);
            for (Method m : values()) {
                if (m.key().equals(key)) {
                    return m;
                }
            }
            throw new IllegalArgumentException("video pruning method '" + name + "' is not supported; the methods"
                    + " are " + Method.EVS.key() + " and " + Method.VIDCOM2.key());
        }

        /** The lower-case name vLLM's configuration uses. */
        public String key() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /** Multi-scale Gaussian bandwidths of the VidCom2 reference. */
    private static final double[] ALPHAS = {0.125, 0.25, 0.5, 1.0, 2.0};
    /** The share of channels VidCom2 scores in, the lowest-variance half. */
    private static final double LOW_VAR_CHANNEL_RATIO = 0.5;
    /** VidCom2's softmax temperature over the per-frame scores. */
    private static final double SOFTMAX_TEMPERATURE = 0.01;
    /** The epsilon of a torch cosine similarity. */
    private static final double COSINE_EPS = 1e-8;

    private VideoPruner() {
    }

    /**
     * The rows a rate {@code q} keeps of a video of {@code tokensPerFrame} rows over {@code numFrames} frames under
     * EVS: {@code int(total * (1 - q))}, at least one frame's rows (evs.py
     * {@code compute_retained_tokens_count}).
     */
    public static int retainedCount(int tokensPerFrame, int numFrames, double q) {
        long total = requireRetainedInputs(tokensPerFrame, numFrames, q);
        int kept = (int) (total * (1 - q));
        return Math.max(tokensPerFrame, Math.min(kept, (int) total));
    }

    /**
     * The rows a rate {@code q} keeps under VidCom2: {@code int(total * (1 - q))}, at least one row per frame,
     * whose per-token floor the method's budgets cannot go below (vidcom2.py
     * {@code compute_retained_tokens_count}).
     */
    public static int vidcom2RetainedCount(int tokensPerFrame, int numFrames, double q) {
        long total = requireRetainedInputs(tokensPerFrame, numFrames, q);
        int kept = (int) (total * (1 - q));
        return Math.max(numFrames, Math.min(kept, (int) total));
    }

    /** The total rows of a video the pruners accept, with the rate checked. */
    private static long requireRetainedInputs(int tokensPerFrame, int numFrames, double q) {
        if (tokensPerFrame <= 0 || numFrames <= 0) {
            throw new IllegalArgumentException("a video prunes over positive frames and tokens per frame, got "
                    + numFrames + " frames of " + tokensPerFrame + " tokens");
        }
        if (!(q >= 0) || q >= 1) {
            throw new IllegalArgumentException("the pruning rate must be in [0, 1), got " + q);
        }
        return (long) tokensPerFrame * numFrames;
    }

    /**
     * The EVS retention mask of a video's embedding rows: {@code rows} holds every frame's {@code rowsPerFrame}
     * rows in order, {@code frames} the frame count, {@code q} the rate. Row {@code i} is kept when
     * {@code mask[i]}; the count is {@link #retainedCount}. The score of a token is its cosine dissimilarity to
     * the token at the same grid cell of the previous frame; the first frame's tokens score above everything and
     * are always kept.
     */
    public static boolean[] evsMask(float[][] rows, int frames, int rowsPerFrame, double q) {
        int width = requireRows(rows, frames, rowsPerFrame);
        int keep = retainedCount(rowsPerFrame, frames, q);
        int total = frames * rowsPerFrame;
        double[] dissimilarity = new double[total];
        for (int f = 1; f < frames; f++) {
            for (int t = 0; t < rowsPerFrame; t++) {
                dissimilarity[f * rowsPerFrame + t] = 1 - cosine(rows[(f - 1) * rowsPerFrame + t],
                        rows[f * rowsPerFrame + t], width);
            }
        }
        // evs.py prepends 255 * ones: the first frame's tokens outrank every dissimilarity in [0, 1]
        for (int t = 0; t < rowsPerFrame; t++) {
            dissimilarity[t] = 255;
        }
        Integer[] order = new Integer[total];
        for (int i = 0; i < total; i++) {
            order[i] = i;
        }
        // torch.argsort(..., descending=True, stable=True): ties keep the earlier row
        Arrays.sort(order, Comparator.<Integer>comparingDouble(i -> dissimilarity[i]).reversed()
                .thenComparingInt(i -> i));
        BitSet kept = new BitSet(total);
        for (int i = 0; i < keep; i++) {
            kept.set(order[i]);
        }
        boolean[] out = new boolean[total];
        for (int i = 0; i < total; i++) {
            out[i] = kept.get(i);
        }
        return out;
    }

    /**
     * The VidCom2 retention mask of a video's embedding rows, the reference implementation's five steps: score in
     * the lowest-variance half of the channels, multi-scale Gaussian similarity to the video and per-frame
     * centers, a per-frame budget weighted by how distinctive the frame is, each frame's smallest-similarity
     * tokens kept, and the rounding drift reconciled to the exact retained count.
     */
    public static boolean[] vidcom2Mask(float[][] rows, int frames, int rowsPerFrame, double q) {
        int width = requireRows(rows, frames, rowsPerFrame);
        int total = frames * rowsPerFrame;
        int keep = Math.min(vidcom2RetainedCount(rowsPerFrame, frames, q), total);
        int channels = Math.max(1, (int) (width * LOW_VAR_CHANNEL_RATIO));
        int[] selected = lowestVarianceChannels(rows, total, width, channels);
        double[][] norm = normalized(rows, total, selected, channels);
        double[] videoCenter = center(norm, 0, total);
        double[][] frameCenters = new double[frames][];
        for (int f = 0; f < frames; f++) {
            frameCenters[f] = center(norm, f * rowsPerFrame, rowsPerFrame);
        }
        double[][] videoScore = new double[frames][rowsPerFrame];
        double[][] similarity = new double[frames][rowsPerFrame];
        for (int f = 0; f < frames; f++) {
            for (int t = 0; t < rowsPerFrame; t++) {
                double[] x = norm[f * rowsPerFrame + t];
                videoScore[f][t] = gaussian(x, videoCenter);
                similarity[f][t] = videoScore[f][t] + gaussian(x, frameCenters[f]);
            }
        }
        // per-frame budgets: distinctive frames (low similarity to the video center) get a larger share
        double[] frameScores = new double[frames];
        for (int f = 0; f < frames; f++) {
            double sum = 0;
            for (int t = 0; t < rowsPerFrame; t++) {
                sum += videoScore[f][t];
            }
            frameScores[f] = -sum / rowsPerFrame;
        }
        double[] probs = softmax(scale(frameScores, 1 / SOFTMAX_TEMPERATURE));
        double probsMean = mean(probs);
        int[] budgets = new int[frames];
        for (int f = 0; f < frames; f++) {
            double scale = Math.min(1.0, (1 - q) * (1 + probs[f] - probsMean));
            budgets[f] = Math.max(1, Math.min(rowsPerFrame, (int) Math.rint(scale * rowsPerFrame)));
        }
        boolean[] out = new boolean[total];
        int current = 0;
        for (int f = 0; f < frames; f++) {
            // the frame's smallest-similarity tokens, ties by the earlier row
            final int frame = f;
            Integer[] order = new Integer[rowsPerFrame];
            for (int t = 0; t < rowsPerFrame; t++) {
                order[t] = t;
            }
            Arrays.sort(order, Comparator.<Integer>comparingDouble(t -> similarity[frame][t])
                    .thenComparingInt(t -> t));
            for (int i = 0; i < budgets[f]; i++) {
                out[f * rowsPerFrame + order[i]] = true;
                current++;
            }
        }
        reconcile(out, similarity, current, keep);
        return out;
    }

    /**
     * Drifts the per-frame budgets to the exact retained count (vidcom2.py step 5): the retained rows with the
     * highest similarity are dropped, or the dropped rows with the lowest similarity are added.
     */
    static void reconcile(boolean[] out, double[][] similarity, int current, int keep) {
        if (current > keep) {
            Integer[] retained = scoreOrder(out, similarity, true, false);
            for (int i = 0; i < current - keep; i++) {
                out[retained[i]] = false;
            }
        } else if (current < keep) {
            Integer[] dropped = scoreOrder(out, similarity, false, true);
            for (int i = 0; i < Math.min(keep - current, dropped.length); i++) {
                out[dropped[i]] = true;
            }
        }
    }

    /**
     * The retained (or dropped) row indices over all frames, by similarity: descending for the highest first,
     * ascending for the lowest first; ties keep the earlier row.
     */
    private static Integer[] scoreOrder(boolean[] out, double[][] similarity, boolean retained, boolean ascending) {
        int frames = similarity.length;
        int rowsPerFrame = similarity[0].length;
        List<Integer> at = new ArrayList<>();
        for (int f = 0; f < frames; f++) {
            for (int t = 0; t < rowsPerFrame; t++) {
                if (out[f * rowsPerFrame + t] == retained) {
                    at.add(f * rowsPerFrame + t);
                }
            }
        }
        Integer[] idx = at.toArray(new Integer[0]);
        Comparator<Integer> byScore = Comparator.comparingDouble(
                i -> similarity[i / rowsPerFrame][i % rowsPerFrame]);
        if (!ascending) {
            byScore = byScore.reversed();
        }
        Arrays.sort(idx, byScore.thenComparingInt(i -> i));
        return idx;
    }

    private static int requireRows(float[][] rows, int frames, int rowsPerFrame) {
        Objects.requireNonNull(rows, "rows");
        if (frames <= 0 || rowsPerFrame <= 0) {
            throw new IllegalArgumentException("a video prunes over positive frames and rows per frame, got "
                    + frames + " frames of " + rowsPerFrame + " rows");
        }
        if (rows.length != (long) frames * rowsPerFrame) {
            throw new IllegalArgumentException(rows.length + " embedding rows are not " + frames + " frames of "
                    + rowsPerFrame + " rows");
        }
        int width = rows[0].length;
        for (float[] row : rows) {
            if (row.length != width) {
                throw new IllegalArgumentException("embedding rows share one width: a row of " + row.length
                        + " differs from " + width);
            }
        }
        return width;
    }

    /** The {@code k} lowest-variance channel indices over all rows (vidcom2.py step 1, ties by channel index). */
    private static int[] lowestVarianceChannels(float[][] rows, int total, int width, int k) {
        double[] variance = new double[width];
        double[] means = new double[width];
        for (int r = 0; r < total; r++) {
            for (int c = 0; c < width; c++) {
                means[c] += rows[r][c];
            }
        }
        for (int c = 0; c < width; c++) {
            means[c] /= total;
        }
        for (int r = 0; r < total; r++) {
            for (int c = 0; c < width; c++) {
                double d = rows[r][c] - means[c];
                variance[c] += d * d;
            }
        }
        for (int c = 0; c < width; c++) {
            variance[c] /= total;
        }
        Integer[] order = new Integer[width];
        for (int c = 0; c < width; c++) {
            order[c] = c;
        }
        // torch.topk(largest=False): ascending variance, stable ties
        Arrays.sort(order, Comparator.<Integer>comparingDouble(c -> variance[c]).thenComparingInt(c -> c));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = order[i];
        }
        return out;
    }

    /** The rows over the {@code selected} channels, each row L2-normalized (vidcom2.py step 2). */
    private static double[][] normalized(float[][] rows, int total, int[] selected, int channels) {
        double[][] out = new double[total][channels];
        for (int r = 0; r < total; r++) {
            double norm = 0;
            for (int i = 0; i < channels; i++) {
                double v = rows[r][selected[i]];
                out[r][i] = v;
                norm += v * v;
            }
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (int i = 0; i < channels; i++) {
                    out[r][i] /= norm;
                }
            }
        }
        return out;
    }

    /** The mean row of {@code rows[at .. at + count)}. */
    private static double[] center(double[][] rows, int at, int count) {
        double[] out = new double[rows[at].length];
        for (int r = at; r < at + count; r++) {
            for (int c = 0; c < out.length; c++) {
                out[c] += rows[r][c];
            }
        }
        for (int c = 0; c < out.length; c++) {
            out[c] /= count;
        }
        return out;
    }

    /** The sum of Gaussian kernels over every alpha: {@code exp(-dist^2 / (2 a))} summed (vidcom2.py). */
    private static double gaussian(double[] x, double[] center) {
        double distSq = 0;
        for (int c = 0; c < x.length; c++) {
            double d = x[c] - center[c];
            distSq += d * d;
        }
        double out = 0;
        for (double alpha : ALPHAS) {
            out += Math.exp(-distSq / (2 * alpha));
        }
        return out;
    }

    private static double[] scale(double[] values, double factor) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] * factor;
        }
        return out;
    }

    /** Softmax over shifted values (torch's softmax: the largest shifts to 0). */
    private static double[] softmax(double[] values) {
        double max = values[0];
        for (double v : values) {
            max = Math.max(max, v);
        }
        double[] out = new double[values.length];
        double sum = 0;
        for (int i = 0; i < values.length; i++) {
            out[i] = Math.exp(values[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }
        return out;
    }

    private static double mean(double[] values) {
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    /** The cosine similarity of two rows, torch's {@code cosine_similarity} with its epsilon clamp. */
    private static double cosine(float[] a, float[] b, int width) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int c = 0; c < width; c++) {
            dot += (double) a[c] * b[c];
            na += (double) a[c] * a[c];
            nb += (double) b[c] * b[c];
        }
        return dot / Math.max(Math.sqrt(na) * Math.sqrt(nb), COSINE_EPS);
    }
}
