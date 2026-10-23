package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Additive deltas on the weight products of one model on the CPU, the twin of {@code GpuProjectionDeltas}: the LoRA
 * terms of llama.cpp {@code build_lora_mm} ({@code y = W x [* w_s] + scale * B (A x)}, src/llama-graph.cpp:1629-1667),
 * {@code build_lora_mm_id} (per selected expert, :1706-1749) and {@code build_lora_get_rows} (token-embedding rows,
 * :2694-2709), and the graniteswitch per-token switched LoRA (src/models/granite-switch.cpp:211-220, 357-359).
 *
 * <p>A product is addressed by the canonical name of its base tensor: {@link LayerSpec#tensorName} of the layer's
 * weight suffix ({@code blk.3.attn_q.weight}) for layer tensors, the model-level name ({@code output.weight},
 * {@code token_embd.weight}) otherwise. A block that applies deltas takes the live site of each of its products at
 * bind time and calls its {@code apply} right after the base product (after a {@code w_s} the product folds in),
 * before any bias. Providers register deltas after the serving blocks are bound; a delta on a product no bound block
 * serves, or on a site of another kind, is refused, so a provider never has its delta silently dropped. A block keeps
 * its live site, so deltas registered or removed later take effect from the next step. Layers that share physical
 * weights (loop slots) share the site of each tensor.</p>
 *
 * <p>Shared by every block of one model through the forward context. Not thread-safe: one forward pass at a
 * time.</p>
 */
public final class CpuProjectionDeltas {

    /** What a site's deltas take as input. */
    public enum Kind {
        /** A dense product {@code y = W x} over token rows. */
        PROJECTION,
        /** A per-expert product {@code y_p = W[e_p] x_p} over the selected (token, expert) pairs. */
        EXPERTS,
        /** A table lookup {@code y_t = E[id_t]}. */
        EMBEDDING
    }

    /** One additive delta on a dense product's output rows. */
    @FunctionalInterface
    public interface Delta {
        /**
         * {@code y[t] += delta(x[t])} for {@code t < n}: row {@code t} of {@code x} at {@code xOff + t*xStride}, of
         * {@code y} at {@code yOff + t*yStride}.
         */
        void add(int n, float[] x, int xOff, int xStride, float[] y, int yOff, int yStride);
    }

    /** One additive delta on a per-expert product's pair rows. */
    @FunctionalInterface
    public interface ExpertDelta {
        /**
         * {@code y[p] += delta_{ids[p]}(x_p)} for the {@code pairs} pairs {@code p = t*k + j}: input row of pair
         * {@code p} at {@code xOff + (tokenRows ? p / k : p) * inDim}, output row at {@code yOff + p * outDim}.
         */
        void add(int pairs, int k, int[] ids, float[] x, int xOff, boolean tokenRows, float[] y, int yOff);
    }

    /** One additive delta on looked-up table rows. */
    @FunctionalInterface
    public interface EmbeddingDelta {
        /** {@code y[t] += delta(ids[idsOff + t])} for {@code t < n}, row {@code t} of {@code y} at {@code yOff + t*yStride}. */
        void add(int n, int[] ids, int idsOff, float[] y, int yOff, int yStride);
    }

    /** The served state of one product: its kind, shape, refusal and registered deltas. */
    public abstract static class Served<D> {
        private final String name;
        private final Kind kind;
        final List<D> deltas = new ArrayList<>();
        private boolean served;
        private String refusal;
        private long inDim = -1L;
        private long outDim = -1L;
        private long experts = -1L;

        Served(String name, Kind kind) {
            this.name = name;
            this.kind = kind;
        }

        /** Canonical tensor name of the product. */
        public String name() { return name; }
        public Kind kind() { return kind; }
        /** Product input width (the vocabulary size of a table), or -1 when the serving block did not give it. */
        public long inDim() { return inDim; }
        /** Product output width (the row width of a table), or -1 when the serving block did not give it. */
        public long outDim() { return outDim; }
        /** Expert count of an {@link Kind#EXPERTS} site, 1 for the other kinds, -1 when not given. */
        public long experts() { return experts; }
        /** Why the serving block cannot take deltas, or null. */
        public String refusal() { return refusal; }

        /** True when no delta is registered. */
        public boolean isEmpty() {
            return deltas.isEmpty();
        }

        void markServed(String why, long in, long out, long nExperts) {
            if (in >= 0 && inDim >= 0 && (in != inDim || out != outDim || nExperts != experts)) {
                throw new IllegalStateException(name + " is served as {" + inDim + " -> " + outDim + ", " + experts
                        + " experts} and again as {" + in + " -> " + out + ", " + nExperts + " experts}");
            }
            served = true;
            if (refusal == null) {
                // A refusal sticks: any serving block that cannot apply deltas would otherwise drop them silently.
                refusal = why;
            }
            if (in >= 0) {
                inDim = in;
                outDim = out;
                experts = nExperts;
            }
        }

        @Override
        public String toString() {
            return "CpuProjectionDeltas." + kind + "{" + name + ", " + deltas.size() + " deltas"
                    + (served ? "" : ", unserved") + '}';
        }
    }

    /** The registered deltas of one dense product. */
    public static final class Site extends Served<Delta> {
        private Site(String name) {
            super(name, Kind.PROJECTION);
        }

        /** Adds every registered delta to {@code y}; see {@link Delta#add}. */
        public void apply(int n, float[] x, int xOff, int xStride, float[] y, int yOff, int yStride) {
            for (int i = 0; i < deltas.size(); i++) {
                deltas.get(i).add(n, x, xOff, xStride, y, yOff, yStride);
            }
        }

        /**
         * {@link #apply} over dense rows: the strides are the input and output widths the site was served with.
         *
         * @throws IllegalStateException when the serving block gave no shape
         */
        public void applyRows(int n, float[] x, int xOff, float[] y, int yOff) {
            if (deltas.isEmpty()) {
                return;
            }
            if (inDim() < 0) {
                throw new IllegalStateException(name() + " was served without its shape; pass the row strides");
            }
            apply(n, x, xOff, (int) inDim(), y, yOff, (int) outDim());
        }
    }

    /** The registered deltas of one per-expert product. */
    public static final class ExpertSite extends Served<ExpertDelta> {
        private ExpertSite(String name) {
            super(name, Kind.EXPERTS);
        }

        /** Adds every registered delta to the pair rows of {@code y}; see {@link ExpertDelta#add}. */
        public void apply(int pairs, int k, int[] ids, float[] x, int xOff, boolean tokenRows, float[] y, int yOff) {
            for (int i = 0; i < deltas.size(); i++) {
                deltas.get(i).add(pairs, k, ids, x, xOff, tokenRows, y, yOff);
            }
        }
    }

    /** The registered deltas of one table lookup. */
    public static final class EmbeddingSite extends Served<EmbeddingDelta> {
        private EmbeddingSite(String name) {
            super(name, Kind.EMBEDDING);
        }

        /** Adds every registered delta to the looked-up rows; see {@link EmbeddingDelta#add}. */
        public void apply(int n, int[] ids, int idsOff, float[] y, int yOff, int yStride) {
            for (int i = 0; i < deltas.size(); i++) {
                deltas.get(i).add(n, ids, idsOff, y, yOff, yStride);
            }
        }
    }

    private final Map<String, Served<?>> sites = new HashMap<>();

    // ================================================================== serving

    /**
     * The live site of {@code layer}'s dense product {@code weight}, marked as served by the calling block, without
     * a shape.
     *
     * @param refusal null when the block applies deltas to this product, otherwise why it cannot (for example a
     *                product whose input the matvec rotates), which makes registration fail with it
     */
    public Site serve(LayerSpec layer, String weight, String refusal) {
        Site s = site(key(layer, weight), Site.class, Kind.PROJECTION);
        s.markServed(refusal, -1L, -1L, -1L);
        return s;
    }

    /**
     * The live site of {@code layer}'s dense product {@code weight} by {@code w} through {@code mv}, with its shape; a
     * weight whose input {@code mv} rotates (PRISM Hadamard) refuses deltas, because a delta would need the rotated
     * rows.
     */
    public Site serve(LayerSpec layer, String weight, CpuTensorRef w, CpuMatVec mv) {
        Objects.requireNonNull(w, "w");
        Site s = site(key(layer, weight), Site.class, Kind.PROJECTION);
        s.markServed(prismRefusal(key(layer, weight), w, mv), w.cols(), w.rows(), 1L);
        return s;
    }

    /** The live site of a model-level dense product ({@code output.weight}, ...) of {@code inDim -> outDim}. */
    public Site serveModel(String name, long inDim, long outDim, String refusal) {
        Site s = site(Objects.requireNonNull(name, "name"), Site.class, Kind.PROJECTION);
        s.markServed(refusal, inDim, outDim, 1L);
        return s;
    }

    /**
     * The live site of the model-level dense product {@code name} by {@code w} through {@code mv}; refusals as
     * {@link #serve(LayerSpec, String, CpuTensorRef, CpuMatVec)}.
     */
    public Site serveModel(String name, CpuTensorRef w, CpuMatVec mv) {
        Objects.requireNonNull(w, "w");
        return serveModel(name, w.cols(), w.rows(), prismRefusal(name, w, mv));
    }

    /**
     * The live site of {@code layer}'s per-expert product {@code weight} by an expert tensor of {@code nExpert}
     * experts of {@code {inDim -> outDim}} through {@code mv}.
     */
    public ExpertSite serveExperts(LayerSpec layer, String weight, CpuTensorRef w, long nExpert, CpuMatVec mv) {
        Objects.requireNonNull(w, "w");
        if (nExpert < 1 || w.rows() % nExpert != 0) {
            throw new IllegalArgumentException(w + " does not hold " + nExpert + " experts");
        }
        ExpertSite s = site(key(layer, weight), ExpertSite.class, Kind.EXPERTS);
        s.markServed(prismRefusal(key(layer, weight), w, mv), w.cols(), w.rows() / nExpert, nExpert);
        return s;
    }

    /** The live site of the lookup of table {@code name} with {@code nVocab} rows of {@code width} values. */
    public EmbeddingSite serveEmbedding(String name, long nVocab, long width, String refusal) {
        EmbeddingSite s = site(Objects.requireNonNull(name, "name"), EmbeddingSite.class, Kind.EMBEDDING);
        s.markServed(refusal, nVocab, width, 1L);
        return s;
    }

    private static String prismRefusal(String name, CpuTensorRef w, CpuMatVec mv) {
        return mv != null && mv.transformsOf(w).input() != null ? name + " rotates its input (PRISM Hadamard) inside"
                + " the matvec; a LoRA delta would need the rotated rows" : null;
    }

    // ================================================================== registration

    /** The served state of product {@code name}, or null when no bound block serves it. */
    public Served<?> served(String name) {
        Served<?> s = sites.get(Objects.requireNonNull(name, "name"));
        return s != null && s.served ? s : null;
    }

    /**
     * Adds {@code delta} to {@code layer}'s dense product {@code weight}.
     *
     * @throws IllegalStateException         when no bound block serves that product, or it is not a dense product
     * @throws UnsupportedOperationException when the serving block cannot apply a delta to it
     */
    public void register(LayerSpec layer, String weight, Delta delta) {
        register(key(layer, weight), delta);
    }

    /** Adds {@code delta} to the dense product of tensor {@code name}; failures as {@link #register(LayerSpec, String, Delta)}. */
    public void register(String name, Delta delta) {
        Objects.requireNonNull(delta, "delta");
        registered(name, Site.class, Kind.PROJECTION).deltas.add(delta);
    }

    /** Adds {@code delta} to the per-expert product of tensor {@code name}; failures as {@link #register(String, Delta)}. */
    public void registerExperts(String name, ExpertDelta delta) {
        Objects.requireNonNull(delta, "delta");
        registered(name, ExpertSite.class, Kind.EXPERTS).deltas.add(delta);
    }

    /** Adds {@code delta} to the lookup of table {@code name}; failures as {@link #register(String, Delta)}. */
    public void registerEmbedding(String name, EmbeddingDelta delta) {
        Objects.requireNonNull(delta, "delta");
        registered(name, EmbeddingSite.class, Kind.EMBEDDING).deltas.add(delta);
    }

    /** Removes {@code delta} (by identity) from {@code layer}'s product {@code weight}; absent ones are ignored. */
    public void unregister(LayerSpec layer, String weight, Object delta) {
        unregister(key(layer, weight), delta);
    }

    /** Removes {@code delta} (by identity) from the product of tensor {@code name}; absent ones are ignored. */
    public void unregister(String name, Object delta) {
        Served<?> s = sites.get(Objects.requireNonNull(name, "name"));
        if (s == null) {
            return;
        }
        for (int i = s.deltas.size() - 1; i >= 0; i--) {
            if (s.deltas.get(i) == delta) {
                s.deltas.remove(i);
            }
        }
    }

    private <S extends Served<?>> S registered(String name, Class<S> type, Kind kind) {
        Served<?> s = sites.get(Objects.requireNonNull(name, "name"));
        if (s == null || !s.served) {
            throw new IllegalStateException("no bound block applies deltas to " + name + "; the model's forward pass"
                    + " does not run this product");
        }
        if (s.kind != kind) {
            throw new IllegalStateException(name + " is a " + s.kind + " product, not a " + kind + " one");
        }
        if (s.refusal != null) {
            throw new UnsupportedOperationException("delta on " + name + ": " + s.refusal);
        }
        return type.cast(s);
    }

    private <S extends Served<?>> S site(String name, Class<S> type, Kind kind) {
        Served<?> s = sites.get(name);
        if (s == null) {
            Served<?> fresh;
            switch (kind) {
                case PROJECTION:
                    fresh = new Site(name);
                    break;
                case EXPERTS:
                    fresh = new ExpertSite(name);
                    break;
                default:
                    fresh = new EmbeddingSite(name);
                    break;
            }
            sites.put(name, fresh);
            s = fresh;
        }
        if (s.kind != kind) {
            throw new IllegalStateException(name + " is served as a " + s.kind + " product and again as " + kind);
        }
        return type.cast(s);
    }

    /** The canonical name of {@code layer}'s tensor {@code weight}: {@code <stack>.<weightLayer>.<weight>}. */
    public static String key(LayerSpec layer, String weight) {
        Objects.requireNonNull(layer, "layer");
        Objects.requireNonNull(weight, "weight");
        return layer.tensorName(weight);
    }

    @Override
    public String toString() {
        return "CpuProjectionDeltas{" + sites.size() + " sites}";
    }
}
