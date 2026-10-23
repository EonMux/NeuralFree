package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

/**
 * Reinforcement learning from human feedback: a reward model fitted to preference pairs with the logistic
 * (Bradley-Terry) loss, and PPO on the policy with rewards penalized by the divergence from a frozen reference model.
 *
 * <p>The models map a {@code [1, L]} tensor of float tokens to logits. The logits processing runs on the device.
 * For a step whose action has {@code T} tokens, the model's output is read as {@code T} rows of
 * {@code V = elements / T} logits (it must split evenly), one row per action token: a sequence's log-probability is
 * the sum over its tokens of {@code log(max(p, 1e-10))} with {@code p} the softmax of the token's row at the token
 * (tokens outside {@code [0, V)} are skipped), and the entropy is the mean over the rows of
 * {@code -sum(p log p)} over {@code p > 1e-10}. The preference loss is {@code -logSigmoid(r_preferred - r_rejected)}
 * in the overflow-free form {@code relu(-d) + softplus(-|d|)}, with {@code r} the reward model's first output
 * element.</p>
 *
 * <p><b>KL penalty.</b> As in InstructGPT and TRL, the penalty enters through the rewards: before the PPO epochs each
 * stored step's reward is reduced by {@code klCoeff * (logp_old - logp_ref)}, with {@code logp_old} the
 * log-probability stored with the step (the policy that generated it) and {@code logp_ref} the reference model's
 * log-probability of its action tokens. The penalty is a constant, so its gradient reaches the policy only through the
 * advantages; a penalty differentiated through {@code log pi(a)} on sampled actions would have zero expected gradient
 * and would not regularize. The advantages are a generalized-advantage scan over the penalized rewards and the stored
 * values, Java values, so the reference log-probabilities are read back once per update, one float per step, and the
 * scan stays on the host. The values come from the caller's critic, so a buffer that ends inside an episode needs the
 * caller's value of the last next state ({@link #trainPPOStep(float)}). {@link #getReward} returns its scalar and
 * {@link #generate} reads back each step's probability vector, from which the next token is drawn with the seeded
 * {@link Random}.</p>
 *
 * <p><b>Training.</b> Every optimizer step has one loss tensor: {@link #trainRewardModel} steps
 * the reward optimizer once per batch on the batch's mean preference loss, and {@link #trainPPOStep} steps the policy
 * optimizer once per epoch on that epoch's mean per-step loss, each recorded, back-propagated and stepped inside its
 * own {@link ComputationGraph.StepScope}, so later epochs see the updated policy. Each training call reads back
 * one scalar per optimizer step, its logged loss. The auxiliary loss terms a trained model's forward pass leaves
 * pending ({@link Module#takeAuxiliaryLoss()}, such as a mixture of experts' load-balancing term) are added to that
 * forward pass's loss, as the {@code Trainer} does; terms pending before a training call, and those of forward passes
 * that are not trained, are dropped. The reference model runs under {@link ComputationGraph#noGrad()}, in eval mode
 * (its mode is restored after each pass); when it shares no parameter with the policy and reward models it is also put
 * in eval mode and frozen with {@link Module#setRequiresGrad(boolean)} at construction, and when it is the policy
 * itself (the builder's default) its log-probabilities are the policy's before the update. A trainer built without an
 * optimizer evaluates that loss under {@code noGrad} and updates nothing. {@link #getReward} and
 * {@link #generate} run under {@code noGrad}, in eval mode, and restore the model's mode.</p>
 *
 * <p>Tolerances against the CPU reference ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}): softmax,
 * log, softplus and the entropy terms are in the transcendental class, the sums and means in the reduction class.</p>
 */
public final class RLHFTrainer implements AutoCloseable {
    private static final float MIN_PROB = 1e-10f;
    
    private final ComputeBackend backend;
    private final Module policyModel;
    private final Module rewardModel;
    private final Module referenceModel;
    
    private final Optimizer policyOptimizer;
    private final Optimizer rewardOptimizer;
    
    private final float klCoeff;
    private final float clipEpsilon;
    private final float entropyCoeff;
    private final float gamma;
    private final float lambda;
    private final int ppoEpochs;
    private final int batchSize;
    
    private final List<PreferencePair> preferenceData;
    private final List<PPOStep> ppoBuffer;
    private final Random random;
    
    private volatile boolean closed;
    private float lastRewardLoss;
    private float lastPolicyLoss;
    private float lastKLPenalty;
    
    /**
     * A trainer over the caller's models, which stay the caller's (closing the trainer does not close them). The
     * optimizers are the caller's too, built over the policy's and the reward model's parameters; either may be null
     * for a loss that is evaluated without updating. A reference model that shares no parameter with the other two
     * models is put in eval mode and frozen; a shared one (the policy itself by default) keeps its mode and flags (see
     * the class description).
     *
     * @throws NullPointerException     when the backend or a model is null
     * @throws IllegalArgumentException when {@code batchSize} is not positive
     */
    public RLHFTrainer(ComputeBackend backend, Module policyModel, Module rewardModel,
                       Module referenceModel,
                       Optimizer policyOptimizer, Optimizer rewardOptimizer,
                       float klCoeff, float clipEpsilon, float entropyCoeff,
                       float gamma, float lambda, int ppoEpochs, int batchSize, long seed) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(policyModel, "Policy model cannot be null");
        Objects.requireNonNull(rewardModel, "Reward model cannot be null");
        Objects.requireNonNull(referenceModel, "Reference model cannot be null");
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive: " + batchSize);
        }
        
        this.backend = backend;
        this.policyModel = policyModel;
        this.rewardModel = rewardModel;
        this.referenceModel = referenceModel;
        
        this.policyOptimizer = policyOptimizer;
        this.rewardOptimizer = rewardOptimizer;
        
        this.klCoeff = klCoeff;
        this.clipEpsilon = clipEpsilon;
        this.entropyCoeff = entropyCoeff;
        this.gamma = gamma;
        this.lambda = lambda;
        this.ppoEpochs = ppoEpochs;
        this.batchSize = batchSize;
        
        this.preferenceData = new ArrayList<>();
        this.ppoBuffer = new ArrayList<>();
        this.random = new Random(seed);
        
        if (sharesNoParameter(referenceModel, policyModel, rewardModel)) {
            this.referenceModel.eval();
            this.referenceModel.setRequiresGrad(false);
        }
        this.lastRewardLoss = 0.0f;
        this.lastPolicyLoss = 0.0f;
        this.lastKLPenalty = 0.0f;
    }
    
    /** Whether no parameter of {@code reference} is also a parameter of {@code first} or {@code second}. */
    private static boolean sharesNoParameter(Module reference, Module first, Module second) {
        Set<Parameter> trained = Collections.newSetFromMap(new IdentityHashMap<>());
        trained.addAll(first.parameters());
        trained.addAll(second.parameters());
        for (Parameter p : reference.parameters()) {
            if (trained.contains(p)) {
                return false;
            }
        }
        return true;
    }
    
    public static RLHFTrainerBuilder builder() {
        return new RLHFTrainerBuilder();
    }
    
    public void addPreferencePair(float[] promptTokens, float[] preferredTokens, float[] rejectedTokens) {
        checkNotClosed();
        Objects.requireNonNull(promptTokens, "Prompt tokens cannot be null");
        Objects.requireNonNull(preferredTokens, "Preferred tokens cannot be null");
        Objects.requireNonNull(rejectedTokens, "Rejected tokens cannot be null");
        
        preferenceData.add(new PreferencePair(promptTokens, preferredTokens, rejectedTokens));
    }
    
    public void addPreferencePairs(List<PreferencePair> pairs) {
        checkNotClosed();
        Objects.requireNonNull(pairs, "Pairs cannot be null");
        preferenceData.addAll(pairs);
    }
    
    /**
     * {@code epochs} shuffled passes over the preference pairs in batches of {@code batchSize}, each batch one reward
     * optimizer step on its mean preference loss; the reward model is put in training mode.
     *
     * @return the mean of the batch losses (each read before its step), or 0 without preference data
     * @throws IllegalStateException when a loss does not require grad (grad mode is disabled)
     */
    public float trainRewardModel(int epochs) {
        checkNotClosed();
        
        if (preferenceData.isEmpty()) {
            return 0.0f;
        }
        
        rewardModel.train();
        rewardModel.clearAuxiliaryLoss();
        float totalLoss = 0.0f;
        int steps = 0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            List<PreferencePair> shuffled = new ArrayList<>(preferenceData);
            Collections.shuffle(shuffled, random);
            
            for (int i = 0; i < shuffled.size(); i += batchSize) {
                int end = Math.min(i + batchSize, shuffled.size());
                List<PreferencePair> batch = shuffled.subList(i, end);
                
                float batchLoss = trainRewardBatch(batch);
                totalLoss += batchLoss;
                steps++;
            }
        }
        
        lastRewardLoss = steps > 0 ? totalLoss / steps : 0.0f;
        return lastRewardLoss;
    }
    
    /** One reward optimizer step on the batch's mean preference loss; returns that mean, read before the step. */
    private float trainRewardBatch(List<PreferencePair> batch) {
        if (rewardOptimizer == null) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor loss = meanPreferenceLoss(batch)) {
                return loss.toFloatArray()[0];
            }
        }
        try (ComputationGraph.StepScope step = ComputationGraph.current().step();
             Tensor loss = meanPreferenceLoss(batch)) {
            return RlTensorMath.optimizerStep(rewardOptimizer, loss, "reward-model");
        }
    }
    
    /** The mean over {@code batch} of each pair's preference loss plus its forward passes' auxiliary terms. */
    private Tensor meanPreferenceLoss(List<PreferencePair> batch) {
        List<Tensor> losses = new ArrayList<>(batch.size());
        try {
            for (PreferencePair pair : batch) {
                losses.add(RlTensorMath.withAuxiliaryLoss(rewardModel, preferenceLoss(pair)));
            }
            try (Tensor all = TensorOps.cat(losses, 0)) {
                return TensorOps.mean(all);
            }
        } finally {
            RlTensorMath.closeAll(losses.toArray(new AutoCloseable[0]));
        }
    }
    
    /**
     * {@code -logSigmoid(r(preferred) - r(rejected))} as a {@code [1]} tensor, by {@code relu(-d) + softplus(-|d|)}.
     */
    private Tensor preferenceLoss(PreferencePair pair) {
        try (Tensor preferredTensor = RlTensorMath.row(backend, pair.preferredTokens);
             Tensor rejectedTensor = RlTensorMath.row(backend, pair.rejectedTokens);
             Tensor preferredOut = rewardModel.forward(preferredTensor);
             Tensor rejectedOut = rewardModel.forward(rejectedTensor);
             Tensor preferredReward = RlTensorMath.firstElement(preferredOut, "reward model");
             Tensor rejectedReward = RlTensorMath.firstElement(rejectedOut, "reward model");
             Tensor margin = TensorOps.sub(preferredReward, rejectedReward);
             Tensor negMargin = TensorOps.neg(margin);
             Tensor linearPart = TensorOps.relu(negMargin);
             Tensor absMargin = TensorOps.abs(margin);
             Tensor negAbs = TensorOps.neg(absMargin);
             Tensor smoothPart = TensorOps.softplus(negAbs)) {
            return TensorOps.add(linearPart, smoothPart);
        }
    }
    
    /**
     * The reward model's first output element for {@code tokens}, read back as one scalar. Runs under
     * {@link ComputationGraph#noGrad()} in eval mode; the model's mode is restored and its auxiliary terms dropped.
     */
    public float getReward(float[] tokens) {
        checkNotClosed();
        Objects.requireNonNull(tokens, "Tokens cannot be null");
        
        boolean wasTraining = rewardModel.isTraining();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            rewardModel.eval();
            try (Tensor tokenTensor = RlTensorMath.row(backend, tokens);
                 Tensor rewardTensor = rewardModel.forward(tokenTensor);
                 Tensor reward = RlTensorMath.firstElement(rewardTensor, "reward model")) {
                return reward.toFloatArray()[0];
            } finally {
                rewardModel.clearAuxiliaryLoss();
            }
        } finally {
            rewardModel.train(wasTraining);
        }
    }
    
    /**
     * Stores one PPO step: the state, the action's tokens, the reward, whether the step ends its episode, and the
     * log-probability of the action tokens under the policy that generated them and the caller's value of the state.
     * The state and the tokens are copied.
     *
     * @throws IllegalArgumentException when {@code actionTokens} is empty
     */
    public void storePPOTransition(float[] state, float[] actionTokens, float reward, boolean done, float logProb,
                                   float value) {
        checkNotClosed();
        Objects.requireNonNull(state, "State cannot be null");
        Objects.requireNonNull(actionTokens, "Action tokens cannot be null");
        if (actionTokens.length == 0) {
            throw new IllegalArgumentException("Action tokens cannot be empty");
        }
        ppoBuffer.add(new PPOStep(state.clone(), actionTokens.clone(), reward, done, logProb, value));
    }
    
    /**
     * One PPO update over the stored steps, for a buffer whose last step ends its episode. Each step's reward is
     * reduced by its KL penalty {@code klCoeff * (logp_old - logp_ref)} (see the class description) and the advantages
     * are estimated from the penalized rewards; then {@code ppoEpochs} passes each make one policy optimizer step on
     * that epoch's mean per-step loss {@code -min(ratio * adv, clip(ratio) * adv) - entropyCoeff * H} (plus the
     * policy's auxiliary terms), with the policy model in training mode. The buffer is then cleared, and
     * {@link #lastKLPenalty()} becomes the mean penalty over the steps.
     *
     * @return the mean per-step loss averaged over the epochs, each epoch's value read before its step; 0 while the
     *         buffer holds fewer than {@code batchSize} steps or when {@code ppoEpochs} is not positive
     * @throws IllegalArgumentException when a model's output does not split into one row per action token
     * @throws IllegalStateException    when the last stored step does not end its episode (the buffer is kept; see
     *                                  {@link #trainPPOStep(float)}), or a loss does not require grad (grad mode is
     *                                  disabled)
     */
    public float trainPPOStep() {
        checkNotClosed();
        
        if (ppoBuffer.size() < batchSize) {
            return 0.0f;
        }
        if (!ppoBuffer.get(ppoBuffer.size() - 1).done) {
            throw new IllegalStateException("the PPO buffer ends inside an episode: pass the value of the last "
                    + "step's next state to trainPPOStep(float)");
        }
        return ppoTrain(0.0f);
    }
    
    /**
     * {@link #trainPPOStep()} for a buffer that may end inside an episode: {@code nextValue}, the caller's value of the
     * last stored step's next state, bootstraps the advantage estimate when that step does not end its episode (it is
     * unused otherwise).
     *
     * @throws IllegalArgumentException when {@code nextValue} is not finite, or as {@link #trainPPOStep()}
     * @throws IllegalStateException    when a loss does not require grad (grad mode is disabled)
     */
    public float trainPPOStep(float nextValue) {
        checkNotClosed();
        if (!Float.isFinite(nextValue)) {
            throw new IllegalArgumentException("The next state's value must be finite: " + nextValue);
        }
        
        if (ppoBuffer.size() < batchSize) {
            return 0.0f;
        }
        return ppoTrain(nextValue);
    }
    
    /** The update of {@link #trainPPOStep(float)} over a buffer of at least {@code batchSize} steps. */
    private float ppoTrain(float nextValue) {
        if (ppoEpochs <= 0) {
            lastPolicyLoss = 0.0f;
            lastKLPenalty = 0.0f;
        } else {
            float[] penalties = klPenalties();
            float[] advantages = advantages(penalties, nextValue);
            lastPolicyLoss = ppoUpdate(advantages);
            lastKLPenalty = mean(penalties);
        }
        
        ppoBuffer.clear();
        
        return lastPolicyLoss;
    }
    
    /** The epochs of {@link #trainPPOStep()}; returns the mean loss over the epochs. */
    private float ppoUpdate(float[] advantages) {
        policyModel.train();
        policyModel.clearAuxiliaryLoss();
        if (policyOptimizer == null) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor loss = epochLoss(advantages)) {
                return loss.toFloatArray()[0];
            }
        }
        float lossSum = 0.0f;
        for (int epoch = 0; epoch < ppoEpochs; epoch++) {
            try (ComputationGraph.StepScope step = ComputationGraph.current().step();
                 Tensor loss = epochLoss(advantages)) {
                lossSum += RlTensorMath.optimizerStep(policyOptimizer, loss, "RLHF policy");
            }
        }
        return lossSum / ppoEpochs;
    }
    
    /** One epoch's mean per-step loss over the buffer, with {@code advantages[i]} the advantage of step {@code i}. */
    private Tensor epochLoss(float[] advantages) {
        List<Tensor> losses = new ArrayList<>(ppoBuffer.size());
        try {
            for (int i = 0; i < ppoBuffer.size(); i++) {
                losses.add(ppoStepLoss(ppoBuffer.get(i), advantages[i]));
            }
            try (Tensor all = TensorOps.cat(losses, 0)) {
                return TensorOps.mean(all);
            }
        } finally {
            RlTensorMath.closeAll(losses.toArray(new AutoCloseable[0]));
        }
    }
    
    /**
     * One step's clipped-surrogate loss with the entropy bonus and the policy's auxiliary terms, as a {@code [1]}
     * tensor. The policy runs and its auxiliary terms are taken before the token positions are uploaded, so the step
     * waits for the device once.
     */
    private Tensor ppoStepLoss(PPOStep step, float advantage) {
        try (Tensor stateTensor = RlTensorMath.row(backend, step.state);
             Tensor policyLogits = policyModel.forward(stateTensor);
             Tensor policyAux = policyModel.takeAuxiliaryLoss();
             Tensor stepLoss = clippedLoss(step, advantage, policyLogits)) {
            return addAuxiliary(stepLoss, policyAux);
        }
    }
    
    /** {@code -min(ratio * adv, clip(ratio) * adv) - entropyCoeff * H} of one step, as a {@code [1]} tensor. */
    private Tensor clippedLoss(PPOStep step, float advantage, Tensor policyLogits) {
        int tokens = step.actionTokens.length;
        int[] positions = tokenPositions(policyLogits.numElements(), step.actionTokens, "policy");
        try (Tensor index = positionIndex(positions);
             Tensor policyProbs = rowSoftmax(policyLogits, tokens);
             Tensor newLogProb = sequenceLogProb(policyProbs, index);
             Tensor logRatio = TensorOps.add(newLogProb, -step.logProb);
             Tensor ratio = TensorOps.exp(logRatio);
             Tensor clipped = TensorOps.clamp(ratio, 1.0f - clipEpsilon, 1.0f + clipEpsilon);
             Tensor surrogate = TensorOps.mul(ratio, advantage);
             Tensor clippedSurrogate = TensorOps.mul(clipped, advantage);
             Tensor pessimistic = RlTensorMath.pairMin(surrogate, clippedSurrogate);
             Tensor policyTerm = TensorOps.neg(pessimistic);
             Tensor rowEntropies = RlTensorMath.rowEntropy(policyProbs, MIN_PROB);
             Tensor meanEntropy = TensorOps.mean(rowEntropies);
             Tensor entropy = meanEntropy.reshape(Shape.vector(1));
             Tensor entropyBonus = TensorOps.mul(entropy, entropyCoeff)) {
            return TensorOps.sub(policyTerm, entropyBonus);
        }
    }
    
    /**
     * Each stored step's KL penalty {@code klCoeff * (logp_old - logp_ref)}: its stored log-probability minus the
     * reference model's log-probability of its action tokens. The reference runs under
     * {@link ComputationGraph#noGrad()} and the log-probabilities are read back together, one float per step, for
     * the host advantage scan.
     */
    private float[] klPenalties() {
        int n = ppoBuffer.size();
        List<Tensor> logProbs = new ArrayList<>(n);
        float[] refLogProbs;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (PPOStep step : ppoBuffer) {
                logProbs.add(referenceLogProb(step));
            }
            try (Tensor all = TensorOps.cat(logProbs, 0)) {
                refLogProbs = all.toFloatArray();
            }
        } finally {
            RlTensorMath.closeAll(logProbs.toArray(new AutoCloseable[0]));
        }
        float[] penalties = new float[n];
        for (int i = 0; i < n; i++) {
            penalties[i] = klCoeff * (ppoBuffer.get(i).logProb - refLogProbs[i]);
        }
        return penalties;
    }
    
    /**
     * The reference model's log-probability of {@code step}'s action tokens, as a {@code [1]} tensor; the model runs
     * before the token positions are uploaded.
     */
    private Tensor referenceLogProb(PPOStep step) {
        try (Tensor stateTensor = RlTensorMath.row(backend, step.state);
             Tensor refLogits = referenceLogits(stateTensor)) {
            int[] positions = tokenPositions(refLogits.numElements(), step.actionTokens, "reference");
            try (Tensor index = positionIndex(positions);
                 Tensor refProbs = rowSoftmax(refLogits, step.actionTokens.length)) {
                return sequenceLogProb(refProbs, index);
            }
        }
    }
    
    private static float mean(float[] values) {
        double sum = 0.0;
        for (float v : values) {
            sum += v;
        }
        return (float) (sum / values.length);
    }
    
    /** The reference model's logits for {@code state}, under {@link ComputationGraph#noGrad()} in eval mode. */
    private Tensor referenceLogits(Tensor state) {
        boolean wasTraining = referenceModel.isTraining();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            referenceModel.eval();
            try {
                return referenceModel.forward(state);
            } finally {
                referenceModel.clearAuxiliaryLoss();
            }
        } finally {
            referenceModel.train(wasTraining);
        }
    }
    
    /** {@code loss + aux} as a new {@code [1]} tensor ({@code aux} null: a recorded copy of {@code loss}). */
    private static Tensor addAuxiliary(Tensor loss, Tensor aux) {
        if (aux == null) {
            return TensorOps.add(loss, 0.0f);
        }
        try (Tensor auxRow = aux.reshape(Shape.vector(1))) {
            return TensorOps.add(loss, auxRow);
        }
    }
    
    /** The softmax of {@code logits} viewed as {@code [tokens, elements / tokens]}, along its rows. */
    private static Tensor rowSoftmax(Tensor logits, int tokens) {
        int vocab = (int) (logits.numElements() / tokens);
        try (Tensor rows = logits.reshape(Shape.matrix(tokens, vocab))) {
            return TensorOps.softmax(rows, 1);
        }
    }
    
    /**
     * The flat positions {@code t * V + token_t} of the tokens inside {@code [0, V)}, in token order, with
     * {@code V = elements / tokens}.
     *
     * @throws IllegalArgumentException when {@code elements} is not a positive multiple of the token count
     */
    private static int[] tokenPositions(long elements, float[] targetTokens, String model) {
        int seqLen = targetTokens.length;
        if (elements <= 0 || elements % seqLen != 0 || elements / seqLen > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("the " + model + " model's output of " + elements
                    + " elements does not split into one row of logits per action token (" + seqLen + " tokens)");
        }
        int vocabSize = (int) (elements / seqLen);
        int[] positions = new int[seqLen];
        int count = 0;
        for (int t = 0; t < seqLen; t++) {
            int tokenIdx = (int) targetTokens[t];
            if (tokenIdx >= 0 && tokenIdx < vocabSize) {
                positions[count++] = t * vocabSize + tokenIdx;
            }
        }
        return Arrays.copyOf(positions, count);
    }
    
    /** {@code positions} as a {@code [1, count]} INT32 gather index, or null when there are none. */
    private Tensor positionIndex(int[] positions) {
        if (positions.length == 0) {
            return null;
        }
        try (Tensor flatIndex = Tensor.fromArray(backend, positions)) {
            return flatIndex.reshape(Shape.matrix(1, positions.length));
        }
    }
    
    /**
     * {@code sum log(max(probs[position], 1e-10))} over the {@link #positionIndex} positions of the row-major
     * {@code [tokens, V]} probabilities, as a {@code [1]} tensor; 0 when {@code index} is null (no token in range).
     */
    private Tensor sequenceLogProb(Tensor probs, Tensor index) {
        if (index == null) {
            return Tensor.zeros(backend, Shape.vector(1));
        }
        try (Tensor flat = probs.reshape(Shape.matrix(1, (int) probs.numElements()));
             Tensor picked = TensorOps.gather(flat, 1, index);
             Tensor logs = RlTensorMath.clampedLog(picked, MIN_PROB)) {
            return TensorOps.sum(logs, 1);
        }
    }
    
    /**
     * Generalized advantage estimation by a backward scan over the stored steps (host data), on each step's reward
     * minus {@code penalties[i]}; {@code nextValue} is the value after the last step when that step does not end its
     * episode.
     */
    private float[] advantages(float[] penalties, float nextValue) {
        int size = ppoBuffer.size();
        float[] advantages = new float[size];
        float lastValue = nextValue;
        float lastAdvantage = 0.0f;
        
        for (int i = size - 1; i >= 0; i--) {
            PPOStep step = ppoBuffer.get(i);
            
            float reward = step.reward - penalties[i];
            float delta = reward + (step.done ? 0.0f : gamma * lastValue) - step.value;
            float advantage = delta + gamma * lambda * (step.done ? 0.0f : lastAdvantage);
            
            advantages[i] = advantage;
            
            lastValue = step.value;
            lastAdvantage = advantage;
        }
        return advantages;
    }
    
    /**
     * Extends {@code promptTokens} by {@code maxLength} sampled tokens. Each step runs {@code model} on the tokens so
     * far, takes the last row of its output (its last axis is the vocabulary), divides it by {@code temperature},
     * applies the softmax on the device and reads the probabilities back to draw the next token (the first index whose
     * running sum reaches a uniform draw, 0 when rounding leaves none). Runs under {@link ComputationGraph#noGrad()} in
     * eval mode; the model's mode is restored and its auxiliary terms dropped.
     *
     * @throws IllegalArgumentException when {@code temperature} is not positive and finite
     */
    public float[] generate(Module model, float[] promptTokens, int maxLength, float temperature) {
        checkNotClosed();
        Objects.requireNonNull(model, "Model cannot be null");
        Objects.requireNonNull(promptTokens, "Prompt tokens cannot be null");
        if (!(temperature > 0.0f) || Float.isInfinite(temperature)) {
            throw new IllegalArgumentException("Temperature must be positive and finite: " + temperature);
        }
        
        float[] tokens = Arrays.copyOf(promptTokens, promptTokens.length + Math.max(0, maxLength));
        int length = promptTokens.length;
        
        boolean wasTraining = model.isTraining();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            model.eval();
            for (int i = 0; i < maxLength; i++) {
                float[] probs = nextTokenProbabilities(model, Arrays.copyOf(tokens, length), temperature);
                tokens[length++] = RlTensorMath.sampleIndex(probs, random.nextFloat(), 0);
            }
        } finally {
            model.train(wasTraining);
        }
        return tokens;
    }
    
    /** The softmax of the last output row of {@code model(tokens)} divided by {@code temperature}, read back. */
    private float[] nextTokenProbabilities(Module model, float[] tokens, float temperature) {
        try (Tensor inputTensor = RlTensorMath.row(backend, tokens);
             Tensor logits = forwardDroppingAuxiliary(model, inputTensor);
             Tensor lastRow = lastRow(logits);
             Tensor divisor = RlTensorMath.constant(backend, Shape.scalar(), temperature);
             Tensor scaled = TensorOps.div(lastRow, divisor);
             Tensor probs = TensorOps.softmax(scaled, 1)) {
            return probs.toFloatArray();
        }
    }
    
    private static Tensor forwardDroppingAuxiliary(Module model, Tensor input) {
        try {
            return model.forward(input);
        } finally {
            model.clearAuxiliaryLoss();
        }
    }
    
    /** The last row along the last axis of {@code logits}, as a {@code [1, V]} view (a scalar output is one row). */
    private static Tensor lastRow(Tensor logits) {
        int rank = logits.shape().rank();
        int vocab = rank == 0 ? 1 : logits.shape().dim(rank - 1);
        if (vocab == 0) {
            throw new IllegalStateException("the model returned an empty output");
        }
        int rows = (int) (logits.numElements() / vocab);
        try (Tensor matrix = logits.reshape(Shape.matrix(rows, vocab))) {
            return matrix.slice(0, rows - 1L, rows);
        }
    }
    
    public Module policyModel() {
        return policyModel;
    }
    
    public Module rewardModel() {
        return rewardModel;
    }
    
    public Module referenceModel() {
        return referenceModel;
    }
    
    public float lastRewardLoss() {
        return lastRewardLoss;
    }
    
    public float lastPolicyLoss() {
        return lastPolicyLoss;
    }
    
    public float lastKLPenalty() {
        return lastKLPenalty;
    }
    
    public int preferenceDataSize() {
        return preferenceData.size();
    }
    
    public int ppoBufferSize() {
        return ppoBuffer.size();
    }
    
    public ComputeBackend backend() {
        return backend;
    }
    
    public void clearPreferenceData() {
        preferenceData.clear();
    }
    
    public void clearPPOBuffer() {
        ppoBuffer.clear();
    }
    
    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("RLHFTrainer has been closed");
        }
    }
    
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            preferenceData.clear();
            ppoBuffer.clear();
        }
    }
    
    @Override
    public String toString() {
        return "RLHFTrainer{preferenceDataSize=" + preferenceData.size() + 
               ", ppoBufferSize=" + ppoBuffer.size() +
               ", klCoeff=" + klCoeff + 
               ", lastRewardLoss=" + lastRewardLoss + 
               ", lastPolicyLoss=" + lastPolicyLoss + "}";
    }
    
    public static final class PreferencePair {
        private final float[] promptTokens;
        private final float[] preferredTokens;
        private final float[] rejectedTokens;
        
        public PreferencePair(float[] promptTokens, float[] preferredTokens, float[] rejectedTokens) {
            this.promptTokens = promptTokens.clone();
            this.preferredTokens = preferredTokens.clone();
            this.rejectedTokens = rejectedTokens.clone();
        }
        
        public float[] promptTokens() {
            return promptTokens.clone();
        }
        
        public float[] preferredTokens() {
            return preferredTokens.clone();
        }
        
        public float[] rejectedTokens() {
            return rejectedTokens.clone();
        }
    }
    
    private static final class PPOStep {
        final float[] state;
        final float[] actionTokens;
        final float reward;
        final boolean done;
        final float logProb;
        final float value;
        
        PPOStep(float[] state, float[] actionTokens, float reward, boolean done, float logProb, float value) {
            this.state = state;
            this.actionTokens = actionTokens;
            this.reward = reward;
            this.done = done;
            this.logProb = logProb;
            this.value = value;
        }
    }
    
    public static final class RLHFTrainerBuilder {
        private ComputeBackend backend;
        private Module policyModel;
        private Module rewardModel;
        private Module referenceModel;
        private Optimizer policyOptimizer;
        private Optimizer rewardOptimizer;
        private float klCoeff = 0.02f;
        private float clipEpsilon = 0.2f;
        private float entropyCoeff = 0.01f;
        private float gamma = 0.99f;
        private float lambda = 0.95f;
        private int ppoEpochs = 4;
        private int batchSize = 64;
        private long seed = System.currentTimeMillis();
        
        public RLHFTrainerBuilder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }
        
        public RLHFTrainerBuilder policyModel(Module policyModel) {
            this.policyModel = policyModel;
            return this;
        }
        
        public RLHFTrainerBuilder rewardModel(Module rewardModel) {
            this.rewardModel = rewardModel;
            return this;
        }
        
        public RLHFTrainerBuilder referenceModel(Module referenceModel) {
            this.referenceModel = referenceModel;
            return this;
        }
        
        public RLHFTrainerBuilder policyOptimizer(Optimizer policyOptimizer) {
            this.policyOptimizer = policyOptimizer;
            return this;
        }
        
        public RLHFTrainerBuilder rewardOptimizer(Optimizer rewardOptimizer) {
            this.rewardOptimizer = rewardOptimizer;
            return this;
        }
        
        public RLHFTrainerBuilder klCoeff(float klCoeff) {
            this.klCoeff = klCoeff;
            return this;
        }
        
        public RLHFTrainerBuilder clipEpsilon(float clipEpsilon) {
            this.clipEpsilon = clipEpsilon;
            return this;
        }
        
        public RLHFTrainerBuilder entropyCoeff(float entropyCoeff) {
            this.entropyCoeff = entropyCoeff;
            return this;
        }
        
        public RLHFTrainerBuilder gamma(float gamma) {
            this.gamma = gamma;
            return this;
        }
        
        public RLHFTrainerBuilder lambda(float lambda) {
            this.lambda = lambda;
            return this;
        }
        
        public RLHFTrainerBuilder ppoEpochs(int ppoEpochs) {
            this.ppoEpochs = ppoEpochs;
            return this;
        }
        
        public RLHFTrainerBuilder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
        
        public RLHFTrainerBuilder seed(long seed) {
            this.seed = seed;
            return this;
        }
        
        public RLHFTrainer build() {
            if (referenceModel == null) {
                referenceModel = policyModel;
            }
            return new RLHFTrainer(backend, policyModel, rewardModel, referenceModel,
                                   policyOptimizer, rewardOptimizer,
                                   klCoeff, clipEpsilon, entropyCoeff,
                                   gamma, lambda, ppoEpochs, batchSize, seed);
        }
    }
}
