package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.ModuleList;
import com.eonmux.neuralfree.nn.activations.GELU;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Sparse mixture of feed-forward experts ({@code dModel -> dFF -> dModel} with GELU) behind a linear router that
 * sends each token to its {@code topK} most probable experts. {@code noiseStd} is recorded for callers and does not
 * enter the routing.
 *
 * <p><b>Autograd.</b> The forward pass records: the router, the softmax, the gathered gates and their normalization,
 * the row selections, the experts and the gate-weighted accumulation, so a loss on the output reaches the router and
 * every expert that received tokens (Switch Transformer routing). The top-k choice is a side computation on a
 * detached handle of the probabilities and the chosen indices are read back; no gradient flows
 * through the choice itself. With load balancing, a forward in training mode computes the load-balance term from the
 * router probabilities (recorded in grad mode) and adds it to the module's pending auxiliary loss
 * ({@link Module#setAuxiliaryLoss(Tensor)}), so a training loop that adds {@link #takeAuxiliaryLoss()} to its loss
 * trains the router towards balanced routing.
 * The terms accumulate: a module applied several times before one backward pass (weight sharing) contributes the term
 * of every application. In eval mode the term is computed under {@link ComputationGraph#noGrad()} for
 * {@link #getLoadBalanceLoss()} only and nothing is added. {@link #getLoadBalanceLoss()} and
 * {@link #getLastRouterProbs()} read detached handles of the latest forward and stay valid after the backward pass.</p>
 */
public final class MixtureOfExperts extends Module {
    private final int dModel;
    private final int dFF;
    private final int numExperts;
    private final int topK;
    private final float noiseStd;
    private final boolean loadBalancing;
    
    private Linear router;
    private ModuleList experts;
    
    private Tensor lastLoadBalanceLoss;
    private Tensor lastRouterProbs;
    private int[] lastExpertIndices;
    
    private boolean initialized = false;
    
    public MixtureOfExperts(int dModel, int dFF, int numExperts, int topK) {
        this(dModel, dFF, numExperts, topK, 0.0f, true);
    }
    
    public MixtureOfExperts(int dModel, int dFF, int numExperts, int topK, float noiseStd, boolean loadBalancing) {
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel must be positive: " + dModel);
        }
        if (dFF <= 0) {
            throw new IllegalArgumentException("dFF must be positive: " + dFF);
        }
        if (numExperts <= 0) {
            throw new IllegalArgumentException("numExperts must be positive: " + numExperts);
        }
        if (topK <= 0 || topK > numExperts) {
            throw new IllegalArgumentException("topK must be in [1, numExperts]: " + topK);
        }
        if (noiseStd < 0.0f) {
            throw new IllegalArgumentException("noiseStd must be non-negative: " + noiseStd);
        }
        
        this.dModel = dModel;
        this.dFF = dFF;
        this.numExperts = numExperts;
        this.topK = topK;
        this.noiseStd = noiseStd;
        this.loadBalancing = loadBalancing;
    }
    
    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }
    
    /**
     * Creates the Xavier-initialized router and the experts on {@code backend} from {@code random}. Runs under
     * {@link ComputationGraph#noGrad()} and records nothing.
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            createModules(backend, random);
        }
        initialized = true;
    }
    
    private void createModules(ComputeBackend backend, Random random) {
        this.router = new Linear(dModel, numExperts);
        router.initializeXavier(backend, random);
        registerModule("router", router);
        
        this.experts = new ModuleList();
        for (int i = 0; i < numExperts; i++) {
            ExpertFFN expert = new ExpertFFN(dModel, dFF);
            expert.initialize(backend, random);
            experts.add(expert);
        }
        registerModule("experts", experts);
    }
    
    /**
     * Routes each token (each {@code dModel} row of {@code input}) to its {@code topK} experts and returns the
     * gate-weighted sum of their outputs, with {@code input}'s shape.
     *
     * <p>The router logits and their softmax are computed on the input's backend. The top-k choice is {@code topK}
     * rounds of {@link TensorOps#argmax} over a {@link Tensor#detach()} handle of the router probabilities, each round
     * setting the chosen entries to {@code -inf} with {@link TensorOps#scatter}; ties go to the lowest expert index, as
     * the stable descending sort of the original host code chose them. The chosen probabilities are gathered
     * ({@link TensorOps#gather}) from the recorded probabilities and normalized by their sum when it is positive, so
     * the gates carry the router's gradient. The expert indices are then read back, because they
     * decide which expert modules run: each expert runs once on the rows routed to it
     * ({@link TensorOps#indexSelectRows}), the outputs are scaled by their gate weights and added into zeros with
     * {@link TensorOps#scatterAdd}, in slot order per token, so every output is {@code 0 + w_0 * y_0 + w_1 * y_1 + ...}
     * as the host code accumulated it.</p>
     *
     * <p>With load balancing the term {@code numExperts * sum_e meanProb_e * share_e} is computed on the device from
     * the probabilities, with the slot shares {@code share_e} a host-counted constant (Switch Transformer). In training
     * mode it is computed in the caller's grad mode (recorded in grad mode) and added to the module's pending
     * auxiliary loss ({@link Module#setAuxiliaryLoss(Tensor)}); in eval mode it is computed under
     * {@link ComputationGraph#noGrad()} and not added. {@link #getLoadBalanceLoss()} reads a detached handle of it.</p>
     *
     * <p>An input without tokens gives zeros of its shape and resets the routing state: no router probabilities, no
     * expert indices and a load-balance loss of 0. It adds no auxiliary term; terms of earlier forwards stay
     * pending.</p>
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape inputShape = input.shape();
        if (inputShape.rank() < 2) {
            throw new IllegalArgumentException("Input must have at least 2 dimensions, got rank " + inputShape.rank());
        }
        
        int lastDim = inputShape.dim(inputShape.rank() - 1);
        if (lastDim != dModel) {
            throw new IllegalArgumentException("Input last dimension must be " + dModel + ", got " + lastDim);
        }
        if (inputShape.numElements() / dModel * (long) topK > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Input " + inputShape + " routes more than 2^31-1 token slots");
        }
        int tokens = (int) (inputShape.numElements() / dModel);
        if (tokens == 0) {
            clearRouting();
            return Tensor.zeros(input.backend(), inputShape);
        }
        
        try (Tensor x = input.reshape(Shape.of(tokens, dModel));
             Tensor logits = router.forward(x);
             Tensor probs = TensorOps.softmax(logits, 1)) {
            replaceRouterProbs(probs.detach());
            Routing routing = route(probs, tokens);
            lastExpertIndices = routing.indices.clone();
            try (Tensor gates = routing.gates;
                 Tensor combined = combine(x, routing, tokens)) {
                if (loadBalancing) {
                    publishLoadBalanceLoss(probs, routing.indices, tokens);
                }
                return combined.reshape(inputShape);
            }
        }
    }
    
    /**
     * The top-k expert indices (read back, token-major {@code t * topK + s}) and normalized gates {@code [N, topK]}.
     * The selection runs on a detached handle of {@code probs} and records nothing; the gates are gathered from the
     * recorded {@code probs}.
     */
    private Routing route(Tensor probs, int tokens) {
        Tensor[] picks = new Tensor[topK];
        Tensor[] chosen = new Tensor[topK];
        Tensor selection = probs.detach();
        Tensor remaining = selection;
        try {
            for (int s = 0; s < topK; s++) {
                picks[s] = TensorOps.argmax(remaining, 1);
                try (Tensor pickCol = picks[s].reshape(Shape.of(tokens, 1))) {
                    chosen[s] = TensorOps.gather(probs, 1, pickCol);
                    if (s + 1 < topK) {
                        try (Tensor negInf = Tensor.scalar(probs.backend(), Float.NEGATIVE_INFINITY);
                             Tensor negInfCol = negInf.expand(Shape.of(tokens, 1))) {
                            Tensor next = TensorOps.scatter(remaining, 1, pickCol, negInfCol);
                            if (remaining != selection) {
                                remaining.close();
                            }
                            remaining = next;
                        }
                    }
                }
            }
            int[] indices = new int[tokens * topK];
            for (int s = 0; s < topK; s++) {
                int[] column = picks[s].toIntArray();
                for (int t = 0; t < tokens; t++) {
                    indices[t * topK + s] = column[t];
                }
            }
            return new Routing(indices, normalizedGates(chosen, tokens));
        } finally {
            if (remaining != selection) {
                remaining.close();
            }
            selection.close();
            closeAll(picks);
            closeAll(chosen);
        }
    }
    
    /** The chosen probabilities {@code [N, topK]} divided by their row sum where that sum is positive. */
    private Tensor normalizedGates(Tensor[] chosen, int tokens) {
        Shape gateShape = Shape.of(tokens, topK);
        try (Tensor values = TensorOps.cat(Arrays.asList(chosen), 1);
             Tensor total = TensorOps.sum(values, 1);
             Tensor totalCol = total.reshape(Shape.of(tokens, 1));
             Tensor zero = Tensor.scalar(values.backend(), 0.0f);
             Tensor positive = TensorOps.gt(totalCol, zero);
             Tensor positiveAll = positive.expand(gateShape);
             Tensor normalized = TensorOps.div(values, totalCol)) {
            return TensorOps.where(positiveAll, normalized, values);
        }
    }
    
    /**
     * Runs each expert once on its routed rows of {@code x} and adds the gate-weighted outputs into zeros
     * {@code [N, dModel]}, in slot-major row order so each token accumulates its slots in order.
     */
    private Tensor combine(Tensor x, Routing routing, int tokens) {
        int slots = tokens * topK;
        int[] slotRow = new int[slots];
        List<Tensor> outputs = new ArrayList<>();
        Tensor all = null;
        try {
            int row = 0;
            for (int e = 0; e < numExperts; e++) {
                int[] rows = rowsOf(routing.indices, e, tokens, slotRow, row);
                if (rows.length == 0) {
                    continue;
                }
                try (Tensor rowIndex = Tensor.fromArray(x.backend(), rows);
                     Tensor routed = TensorOps.indexSelectRows(x, rowIndex, false)) {
                    outputs.add(experts.get(e).forward(routed));
                }
                row += rows.length;
            }
            all = outputs.size() == 1 ? outputs.get(0) : TensorOps.cat(outputs, 0);
            int[] slotToken = new int[slots];
            for (int p = 0; p < slots; p++) {
                slotToken[p] = p % tokens;
            }
            try (Tensor order = Tensor.fromArray(x.backend(), slotRow);
                 Tensor ordered = TensorOps.indexSelectRows(all, order, false);
                 Tensor gatesBySlot = routing.gates.transpose(1, 0);
                 Tensor gateCol = gatesBySlot.reshape(Shape.of(slots, 1));
                 Tensor weighted = TensorOps.mul(ordered, gateCol);
                 Tensor target = Tensor.fromArray(x.backend(), slotToken);
                 Tensor targetCol = target.reshape(Shape.of(slots, 1));
                 Tensor targetAll = targetCol.expand(Shape.of(slots, dModel));
                 Tensor zeros = Tensor.zeros(x.backend(), Shape.of(tokens, dModel))) {
                return TensorOps.scatterAdd(zeros, 0, targetAll, weighted);
            }
        } finally {
            if (all != null && outputs.size() != 1) {
                all.close();
            }
            closeAll(outputs.toArray(new Tensor[0]));
        }
    }
    
    /**
     * The tokens routed to {@code expert}, in ascending token-major slot order, and their rows in the expert-major
     * concatenation of expert outputs: {@code slotRow[s * tokens + t] = firstRow + j} for the j-th routed slot.
     */
    private int[] rowsOf(int[] indices, int expert, int tokens, int[] slotRow, int firstRow) {
        int count = 0;
        for (int index : indices) {
            if (index == expert) {
                count++;
            }
        }
        int[] rows = new int[count];
        int j = 0;
        for (int p = 0; p < indices.length; p++) {
            if (indices[p] == expert) {
                int t = p / topK;
                int s = p % topK;
                rows[j] = t;
                slotRow[s * tokens + t] = firstRow + j;
                j++;
            }
        }
        return rows;
    }
    
    /**
     * {@code numExperts * sum_e meanProb_e * share_e} as a device scalar, recorded from {@code probs}:
     * {@code meanProb_e} is the router probability of expert e summed over tokens and divided by the token count,
     * {@code share_e} the fraction of the {@code tokens * topK} slots routed to it (counted on the host from the
     * read-back indices, a constant).
     */
    private Tensor loadBalanceLoss(Tensor probs, int[] indices, int tokens) {
        float[] share = new float[numExperts];
        for (int index : indices) {
            share[index] += 1.0f;
        }
        for (int e = 0; e < numExperts; e++) {
            share[e] /= (tokens * topK);
        }
        try (Tensor probSum = TensorOps.sum(probs, 0);
             Tensor count = Tensor.scalar(probs.backend(), (float) tokens);
             Tensor meanProb = TensorOps.div(probSum, count);
             Tensor scaled = TensorOps.mul(meanProb, (float) numExperts);
             Tensor shareTensor = Tensor.fromArray(probs.backend(), share);
             Tensor terms = TensorOps.mul(scaled, shareTensor)) {
            return TensorOps.sum(terms);
        }
    }
    
    /**
     * The load-balance term of this forward. In training mode it is computed in the caller's grad mode and added to
     * the module's pending auxiliary loss ({@link Module#setAuxiliaryLoss(Tensor)}, which takes it over), and a
     * detached handle of it is kept for {@link #getLoadBalanceLoss()}; in eval mode it is computed under
     * {@link ComputationGraph#noGrad()}, records nothing and is kept for {@link #getLoadBalanceLoss()} only.
     */
    private void publishLoadBalanceLoss(Tensor probs, int[] indices, int tokens) {
        if (!isTraining()) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                replaceLoadBalanceLoss(loadBalanceLoss(probs, indices, tokens));
            }
            return;
        }
        Tensor term = loadBalanceLoss(probs, indices, tokens);
        Tensor handle;
        try {
            handle = term.detach();
        } catch (RuntimeException e) {
            try {
                ComputationGraph.current().release(term);
            } catch (RuntimeException releaseFailure) {
                e.addSuppressed(releaseFailure);
            } finally {
                term.close();
            }
            throw e;
        }
        replaceLoadBalanceLoss(handle);
        setAuxiliaryLoss(term);
    }

    /**
     * The routing state of a forward without tokens: no router probabilities, an empty expert index list and a
     * load-balance loss of 0. The pending auxiliary terms of earlier forwards are left as they are.
     */
    private void clearRouting() {
        replaceRouterProbs(null);
        replaceLoadBalanceLoss(null);
        lastExpertIndices = new int[0];
    }

    private void replaceRouterProbs(Tensor probs) {
        Tensor previous = lastRouterProbs;
        lastRouterProbs = probs;
        if (previous != null) {
            previous.close();
        }
    }
    
    private void replaceLoadBalanceLoss(Tensor loss) {
        Tensor previous = lastLoadBalanceLoss;
        lastLoadBalanceLoss = loss;
        if (previous != null) {
            previous.close();
        }
    }
    
    private static void closeAll(Tensor[] tensors) {
        for (Tensor t : tensors) {
            if (t != null) {
                t.close();
            }
        }
    }
    
    /**
     * The load-balance loss of the last forward with load balancing (a host read of a detached handle of the device
     * scalar, valid after the backward pass), else 0.
     */
    public float getLoadBalanceLoss() {
        Tensor loss = lastLoadBalanceLoss;
        return loss != null ? loss.toFloatArray()[0] : 0.0f;
    }
    
    /**
     * The router probabilities {@code [N * numExperts]} of the last forward (read from a detached handle on the
     * device, valid after the backward pass), or null.
     */
    public float[] getLastRouterProbs() {
        Tensor probs = lastRouterProbs;
        return probs != null ? probs.toFloatArray() : null;
    }
    
    public int[] getLastExpertIndices() {
        return lastExpertIndices != null ? lastExpertIndices.clone() : null;
    }
    
    @Override
    public void close() {
        replaceRouterProbs(null);
        replaceLoadBalanceLoss(null);
        super.close();
    }
    
    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("MixtureOfExperts not initialized. Call initialize(backend) first.");
        }
    }
    
    public int dModel() {
        return dModel;
    }
    
    public int dFF() {
        return dFF;
    }
    
    public int numExperts() {
        return numExperts;
    }
    
    public int topK() {
        return topK;
    }
    
    public float noiseStd() {
        return noiseStd;
    }
    
    public boolean isLoadBalancing() {
        return loadBalancing;
    }
    
    public Linear router() {
        ensureInitialized();
        return router;
    }
    
    public ModuleList experts() {
        ensureInitialized();
        return experts;
    }
    
    @Override
    public String toString() {
        return "MixtureOfExperts{" +
            "dModel=" + dModel +
            ", dFF=" + dFF +
            ", numExperts=" + numExperts +
            ", topK=" + topK +
            ", noiseStd=" + noiseStd +
            ", loadBalancing=" + loadBalancing +
            ", initialized=" + initialized +
            '}';
    }
    
    private static final class ExpertFFN extends Module {
        private final int dModel;
        private final int dFF;
        
        private Linear fc1;
        private Linear fc2;
        private Module activation;
        
        private boolean initialized = false;
        
        ExpertFFN(int dModel, int dFF) {
            this.dModel = dModel;
            this.dFF = dFF;
        }
        
        void initialize(ComputeBackend backend, Random random) {
            this.fc1 = new Linear(dModel, dFF);
            fc1.initializeXavier(backend, random);
            registerModule("fc1", fc1);
            
            this.fc2 = new Linear(dFF, dModel);
            fc2.initializeXavier(backend, random);
            registerModule("fc2", fc2);
            
            this.activation = new GELU();
            
            initialized = true;
        }
        
        @Override
        public Tensor forward(Tensor input) {
            if (!initialized) {
                throw new IllegalStateException("ExpertFFN not initialized");
            }
            
            try (Tensor h = fc1.forward(input);
                 Tensor activated = activation.forward(h)) {
                return fc2.forward(activated);
            }
        }
    }
    
    /** One routing decision: expert indices (token-major, read back) and the device gates {@code [N, topK]}. */
    private static final class Routing {
        final int[] indices;
        final Tensor gates;
        
        Routing(int[] indices, Tensor gates) {
            this.indices = indices;
            this.gates = gates;
        }
    }
}
