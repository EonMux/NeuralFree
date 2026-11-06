package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.engine.lora.AdapterRows;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;

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
 * <p>Every delta belongs to an adapter slot (vLLM's multi-LoRA batching, punica_wrapper) and applies only to the rows
 * the pass's {@link AdapterRows} table puts in its slot, run by run over the consecutive rows of that slot: slot 0
 * holds the startup adapters (llama.cpp {@code --lora}), a slot {@code s >= 1} the per-request adapters
 * {@code LoraRegistry} opened there, which replace the startup set for its rows. A registration binds the factor its
 * slot's rows run the delta at (1.0 for a plain registration, a slot-key's factor of the resident factors for a
 * rescaled startup set). The forward backend sets the table ({@link #rows()}) before the rows' products run;
 * {@link #openSlot} and {@link #closeSlot} track the slots a registry opened, including one that holds no delta.</p>
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
         * {@code y[t] += scale * delta(x[t])} for {@code t < n}: row {@code t} of {@code x} at {@code xOff + t*xStride},
         * of {@code y} at {@code yOff + t*yStride}; {@code scale} is the factor the delta's slot runs the delta at
         * (1.0 for a plain registration).
         */
        void add(float scale, int n, float[] x, int xOff, int xStride, float[] y, int yOff, int yStride);
    }

    /** One additive delta on a per-expert product's pair rows. */
    @FunctionalInterface
    public interface ExpertDelta {
        /**
         * {@code y[p] += scale * delta_{ids[p]}(x_p)} for the {@code pairs} pairs {@code p = t*k + j}: input row of
         * pair {@code p} at {@code xOff + (tokenRows ? p / k : p) * inDim}, output row at {@code yOff + p * outDim}.
         */
        void add(float scale, int pairs, int k, int[] ids, float[] x, int xOff, boolean tokenRows, float[] y,
                 int yOff);
    }

    /** One additive delta on looked-up table rows. */
    @FunctionalInterface
    public interface EmbeddingDelta {
        /** {@code y[t] += scale * delta(ids[idsOff + t])} for {@code t < n}, row {@code t} of {@code y} at {@code yOff + t*yStride}. */
        void add(float scale, int n, int[] ids, int idsOff, float[] y, int yOff, int yStride);
    }

    /** One delta registered in a slot, bound to the factor its slot's rows run it at. */
    public static final class Bound<D> {
        final D delta;
        final float scale;

        Bound(D delta, float scale) {
            this.delta = delta;
            this.scale = scale;
        }
    }

    /** The served state of one product: its kind, shape, refusal and registered deltas. */
    public abstract static class Served<D> {
        private final String name;
        private final Kind kind;
        /** The adapter-slot table of the pass, shared by every site of the model. */
        final AdapterRows rows;
        /** The slot-0 deltas, applied to the rows of slot 0. */
        final List<Bound<D>> deltas = new ArrayList<>();
        /** The deltas of slots 1 and up, applied to the rows of their slot. */
        final Map<Integer, List<Bound<D>>> slotDeltas = new TreeMap<>();
        private boolean served;
        private String refusal;
        private long inDim = -1L;
        private long outDim = -1L;
        private long experts = -1L;

        Served(String name, Kind kind, AdapterRows rows) {
            this.name = name;
            this.kind = kind;
            this.rows = rows;
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

        /** True when no delta is registered in any slot. */
        public boolean isEmpty() {
            return deltas.isEmpty() && slotDeltas.isEmpty();
        }

        /** The deltas of slot {@code slot}, or null when it holds none. */
        List<Bound<D>> slotted(int slot) {
            return slot == 0 ? deltas : slotDeltas.get(slot);
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
                    + (slotDeltas.isEmpty() ? "" : ", slots " + slotDeltas.keySet())
                    + (served ? "" : ", unserved") + '}';
        }
    }

    /** The registered deltas of one dense product. */
    public static final class Site extends Served<Delta> {
        private Site(String name, AdapterRows rows) {
            super(name, Kind.PROJECTION, rows);
        }

        /**
         * Adds the registered deltas to {@code y}; see {@link Delta#add}. Each slot's deltas cover the rows the pass
         * runs in that slot.
         *
         * @throws IllegalStateException when the pass's rows run in different slots and {@code n} is not their count
         */
        public void apply(int n, float[] x, int xOff, int xStride, float[] y, int yOff, int yStride) {
            if (isEmpty()) {
                return;
            }
            if (rows.uniform() >= 0) {
                List<Bound<Delta>> all = slotted(rows.uniform());
                for (int i = 0; all != null && i < all.size(); i++) {
                    Bound<Delta> b = all.get(i);
                    b.delta.add(b.scale, n, x, xOff, xStride, y, yOff, yStride);
                }
                return;
            }
            rows.checkSpan(name(), n);
            for (int start = 0; start < n; ) {
                int end = rows.runEnd(start, n);
                List<Bound<Delta>> run = slotted(rows.slot(start));
                for (int i = 0; run != null && i < run.size(); i++) {
                    Bound<Delta> b = run.get(i);
                    b.delta.add(b.scale, end - start, x, xOff + start * xStride, xStride, y,
                            yOff + start * yStride, yStride);
                }
                start = end;
            }
        }

        /**
         * {@link #apply} over dense rows: the strides are the input and output widths the site was served with.
         *
         * @throws IllegalStateException when the serving block gave no shape
         */
        public void applyRows(int n, float[] x, int xOff, float[] y, int yOff) {
            if (isEmpty()) {
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
        private ExpertSite(String name, AdapterRows rows) {
            super(name, Kind.EXPERTS, rows);
        }

        /**
         * Adds the registered deltas to the pair rows of {@code y}; see {@link ExpertDelta#add}. Token {@code t}
         * of the pass owns pairs {@code [t*k, (t+1)*k)}; a slot's deltas cover the pairs of its tokens.
         *
         * @throws IllegalStateException when the pass's rows run in different slots and the pairs do not cover them
         */
        public void apply(int pairs, int k, int[] ids, float[] x, int xOff, boolean tokenRows, float[] y, int yOff) {
            if (isEmpty()) {
                return;
            }
            if (rows.uniform() >= 0) {
                List<Bound<ExpertDelta>> run = slotted(rows.uniform());
                for (int i = 0; run != null && i < run.size(); i++) {
                    Bound<ExpertDelta> b = run.get(i);
                    b.delta.add(b.scale, pairs, k, ids, x, xOff, tokenRows, y, yOff);
                }
                return;
            }
            if (k < 1 || pairs % k != 0) {
                throw new IllegalStateException(name() + ": " + pairs + " pairs are not whole tokens of " + k
                        + " experts");
            }
            int tokens = pairs / k;
            rows.checkSpan(name(), tokens);
            int in = (int) inDim();
            int out = (int) outDim();
            for (int start = 0; start < tokens; ) {
                int end = rows.runEnd(start, tokens);
                List<Bound<ExpertDelta>> run = slotted(rows.slot(start));
                if (run != null && !run.isEmpty()) {
                    int[] runIds = Arrays.copyOfRange(ids, start * k, end * k);
                    int runX = xOff + (tokenRows ? start : start * k) * in;
                    for (int i = 0; i < run.size(); i++) {
                        Bound<ExpertDelta> b = run.get(i);
                        b.delta.add(b.scale, (end - start) * k, k, runIds, x, runX, tokenRows, y,
                                yOff + start * k * out);
                    }
                }
                start = end;
            }
        }
    }

    /** The registered deltas of one table lookup. */
    public static final class EmbeddingSite extends Served<EmbeddingDelta> {
        private EmbeddingSite(String name, AdapterRows rows) {
            super(name, Kind.EMBEDDING, rows);
        }

        /**
         * Adds the registered deltas to the looked-up rows; see {@link EmbeddingDelta#add}. Each slot's deltas
         * cover the rows the pass runs in that slot.
         *
         * @throws IllegalStateException when the pass's rows run in different slots and {@code n} is not their count
         */
        public void apply(int n, int[] ids, int idsOff, float[] y, int yOff, int yStride) {
            if (isEmpty()) {
                return;
            }
            if (rows.uniform() >= 0) {
                List<Bound<EmbeddingDelta>> all = slotted(rows.uniform());
                for (int i = 0; all != null && i < all.size(); i++) {
                    Bound<EmbeddingDelta> b = all.get(i);
                    b.delta.add(b.scale, n, ids, idsOff, y, yOff, yStride);
                }
                return;
            }
            rows.checkSpan(name(), n);
            for (int start = 0; start < n; ) {
                int end = rows.runEnd(start, n);
                List<Bound<EmbeddingDelta>> run = slotted(rows.slot(start));
                for (int i = 0; run != null && i < run.size(); i++) {
                    Bound<EmbeddingDelta> b = run.get(i);
                    b.delta.add(b.scale, end - start, ids, idsOff + start, y, yOff + start * yStride, yStride);
                }
                start = end;
            }
        }

        /**
         * Adds the deltas of row {@code row} of the pass to its one looked-up row: those of the row's slot
         * ({@code y += scale * delta(ids[idsOff])} at {@code yOff}).
         */
        public void applyRow(int row, int[] ids, int idsOff, float[] y, int yOff) {
            List<Bound<EmbeddingDelta>> run = slotted(rows.slot(row));
            for (int i = 0; run != null && i < run.size(); i++) {
                Bound<EmbeddingDelta> b = run.get(i);
                b.delta.add(b.scale, 1, ids, idsOff, y, yOff, 0);
            }
        }
    }

    private final Map<String, Served<?>> sites = new HashMap<>();
    /** The adapter slot of every row of the current pass. */
    private final AdapterRows rows = new AdapterRows();
    /** The slots >= 1 a registry opened on this model ({@link #openSlot}), the slots {@link #holdsSlot} reports. */
    private final Set<Integer> openSlots = new HashSet<>();

    /** The adapter-slot table of the current pass; the forward backend sets it before the rows' products run. */
    public AdapterRows rows() {
        return rows;
    }

    /** True when slot {@code slot} >= 1 is open on this model: rows may run in it, with or without deltas. */
    public boolean holdsSlot(int slot) {
        return openSlots.contains(slot);
    }

    /**
     * Marks slot {@code slot} >= 1 open: rows may run in it from now on; it applies the deltas registered in it,
     * none for a slot that runs the bare base model.
     *
     * @throws IllegalArgumentException for a slot below 1
     */
    public void openSlot(int slot) {
        if (slot < 1) {
            throw new IllegalArgumentException("per-request adapter slots start at 1, got " + slot);
        }
        openSlots.add(slot);
    }

    /** Marks slot {@code slot} closed; its deltas must be unregistered before the slot opens again. */
    public void closeSlot(int slot) {
        openSlots.remove(slot);
    }

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
        CpuMatVec.Transforms t = mv != null ? mv.transformsOf(w) : null;
        return t != null && !t.transform().isIdentity() ? name + " transforms its input (PRISM Hadamard or input"
                + " transform) inside the matvec; a LoRA delta would need the transformed rows" : null;
    }

    // ================================================================== registration

    /** The served state of product {@code name}, or null when no bound block serves it. */
    public Served<?> served(String name) {
        Served<?> s = sites.get(Objects.requireNonNull(name, "name"));
        return s != null && s.served ? s : null;
    }

    /**
     * The live dense site of product {@code name}, or null when no bound block serves it as a projection (it is
     * served as another kind, or not at all).
     */
    public Site site(String name) {
        Served<?> s = served(name);
        return s instanceof Site ? (Site) s : null;
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
        register(name, 0, delta);
    }

    /**
     * Adds {@code delta} to the dense product of tensor {@code name} in adapter slot {@code slot}: slot 0 applies
     * to every row, a slot {@code >= 1} to the rows the pass runs in it; failures as
     * {@link #register(LayerSpec, String, Delta)}.
     *
     * @throws IllegalArgumentException for a negative slot
     */
    public void register(String name, int slot, Delta delta) {
        register(name, slot, 1.0f, delta);
    }

    /**
     * Adds {@code delta} to the dense product of tensor {@code name} in slot {@code slot} at the factor
     * {@code scale} (a rescaled slot's factor of the resident factors); failures as
     * {@link #register(String, int, Delta)}.
     *
     * @throws IllegalArgumentException for a negative slot or a non-finite scale
     */
    public void register(String name, int slot, float scale, Delta delta) {
        Objects.requireNonNull(delta, "delta");
        add(registered(name, Site.class, Kind.PROJECTION), slot, scale, delta);
    }

    /** Adds {@code delta} to the per-expert product of tensor {@code name}; failures as {@link #register(String, Delta)}. */
    public void registerExperts(String name, ExpertDelta delta) {
        registerExperts(name, 0, delta);
    }

    /** Adds {@code delta} to the per-expert product of tensor {@code name} in slot {@code slot}; see {@link #register(String, int, Delta)}. */
    public void registerExperts(String name, int slot, ExpertDelta delta) {
        registerExperts(name, slot, 1.0f, delta);
    }

    /** Adds {@code delta} to the per-expert product of tensor {@code name} in slot {@code slot} at {@code scale}. */
    public void registerExperts(String name, int slot, float scale, ExpertDelta delta) {
        Objects.requireNonNull(delta, "delta");
        add(registered(name, ExpertSite.class, Kind.EXPERTS), slot, scale, delta);
    }

    /** Adds {@code delta} to the lookup of table {@code name}; failures as {@link #register(String, Delta)}. */
    public void registerEmbedding(String name, EmbeddingDelta delta) {
        registerEmbedding(name, 0, delta);
    }

    /** Adds {@code delta} to the lookup of table {@code name} in slot {@code slot}; see {@link #register(String, int, Delta)}. */
    public void registerEmbedding(String name, int slot, EmbeddingDelta delta) {
        registerEmbedding(name, slot, 1.0f, delta);
    }

    /** Adds {@code delta} to the lookup of table {@code name} in slot {@code slot} at {@code scale}. */
    public void registerEmbedding(String name, int slot, float scale, EmbeddingDelta delta) {
        Objects.requireNonNull(delta, "delta");
        add(registered(name, EmbeddingSite.class, Kind.EMBEDDING), slot, scale, delta);
    }

    private <D> void add(Served<D> site, int slot, float scale, D delta) {
        if (slot < 0) {
            throw new IllegalArgumentException("adapter slots are non-negative, got " + slot + " for " + site.name());
        }
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("a delta of " + site.name() + " needs a finite scale, got " + scale);
        }
        Bound<D> bound = new Bound<>(delta, scale);
        if (slot == 0) {
            site.deltas.add(bound);
            return;
        }
        site.slotDeltas.computeIfAbsent(slot, s -> new ArrayList<>()).add(bound);
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
            if (s.deltas.get(i).delta == delta) {
                s.deltas.remove(i);
            }
        }
        for (Iterator<? extends Map.Entry<Integer, ? extends List<?>>> it = s.slotDeltas.entrySet().iterator();
                it.hasNext(); ) {
            Map.Entry<Integer, ? extends List<?>> e = it.next();
            List<?> list = e.getValue();
            for (int i = list.size() - 1; i >= 0; i--) {
                if (((Bound<?>) list.get(i)).delta == delta) {
                    list.remove(i);
                }
            }
            if (list.isEmpty()) {
                it.remove();
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
                    fresh = new Site(name, rows);
                    break;
                case EXPERTS:
                    fresh = new ExpertSite(name, rows);
                    break;
                default:
                    fresh = new EmbeddingSite(name, rows);
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
