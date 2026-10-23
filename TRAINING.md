# NeuralFree Training Guide

This guide covers the API of the training framework: tensors and autograd, modules and parameters, optimizers, losses, data loading, the trainers, checkpoints, distributed and reinforcement learning, meta-learning and quantization. Every op runs on the `train` kernel library of its tensors' backend (OpenCL, CUDA or Vulkan) and falls back to its CPU reference where the library is unavailable. [USAGE.md](USAGE.md) covers the build, backend selection and [the training framework on a GPU backend](USAGE.md#the-training-framework-on-a-gpu-backend): locking, residency, ordering and the numerical tolerance classes.

The demos in `src/main/java/com/eonmux/neuralfree/demo/` are complete programs: `MLPDemo`, `CNNDemo`, `TransformerDemo`, `GNNDemo`, `AutogradDemo`, `InferenceDemo` and `GpuMatmulDemo`. `./gradlew runDemos` runs them all on one backend.

## Contents

1. [Getting a backend](#getting-a-backend)
2. [Tensors](#tensors)
3. [Autograd](#autograd)
4. [Parameters](#parameters)
5. [Modules and layers](#modules-and-layers)
6. [Optimizers and schedulers](#optimizers-and-schedulers)
7. [Loss functions](#loss-functions)
8. [Data pipeline](#data-pipeline)
9. [Training](#training)
10. [Mixed precision](#mixed-precision)
11. [Checkpoints and serialization](#checkpoints-and-serialization)
12. [Distributed training](#distributed-training)
13. [Reinforcement learning](#reinforcement-learning)
14. [Meta-learning and distillation](#meta-learning-and-distillation)
15. [Inference, quantization and export](#inference-quantization-and-export)
16. [Classical ML and other packages](#classical-ml-and-other-packages)

## Getting a backend

`NeuralFree.createTrainingBackend` prefers a backend on which the train kernels are available ([details](USAGE.md#compute-backends)):

```java
import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.NeuralFree;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

HWAcceleration hw = HWAcceleration.forMode(Config.ACCELERATION_MODE.AUTO);
try {
    ComputeBackend backend = NeuralFree.createTrainingBackend(hw, Config.ACCELERATION_MODE.AUTO);
    System.out.println("Train kernels: " + NeuralFree.describeTrainKernels(backend));
    // ... build and train models on backend ...
} finally {
    hw.shutdown();   // closes the backend, then releases the devices
}
```

The snippets below use `backend` for this backend.

## Tensors

`Tensor` factories take a backend and a `Shape`, built with `Shape.of(...)`, `Shape.matrix(rows, cols)` or `Shape.vector(n)`. The default dtype is `FLOAT32`, and `ones`, `rand`, `randn` and `eye` support only `FLOAT32`. `Tensor.fromArray(backend, int[])` makes an `INT32` vector, for example class indices, and `Tensor.fromArray(backend, byte[])` an `INT8` vector; `Tensor.zeros(backend, shape, dtype)` allocates any dtype (`FLOAT16`, `BFLOAT16`, `FLOAT64`, `INT8`, `INT16`, `INT32`, `INT64`, `BOOL`). `to(DType)` converts `FLOAT32` to and from `FLOAT64`, `INT32`, `BOOL`, `FLOAT16` and `BFLOAT16`, and `INT64` to `INT32`; any other pair throws `UnsupportedOperationException`.

```java
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

Tensor zeros  = Tensor.zeros(backend, Shape.of(3, 4));
Tensor ones   = Tensor.ones(backend, Shape.of(3, 4));
Tensor normal = Tensor.randn(backend, Shape.of(3, 4));      // N(0, 1)
Tensor unif   = Tensor.rand(backend, Shape.of(3, 4), 42L);  // U[0, 1), seeded

Tensor t = Tensor.fromArray(backend, new float[]{1, 2, 3, 4, 5, 6}).reshape(2, 3);
Tensor m = Tensor.fromArray(backend, new float[][]{{1, 2, 3}, {4, 5, 6}});

Shape shape = t.shape();             // dims() -> {2, 3}, numElements() -> 6
float[] values = t.toFloatArray();   // a readback: waits for the work issued before it

Tensor sum  = TensorOps.add(zeros, normal);
Tensor prod = TensorOps.mul(normal, 2.0f);                // scalar
Tensor mm   = TensorOps.matmul(normal, ones.transpose()); // (3x4)(4x3)
Tensor sm   = TensorOps.softmax(normal, -1);
Tensor row  = TensorOps.add(normal, Tensor.randn(backend, Shape.of(1, 4)));  // broadcasting
```

Binary ops broadcast and require both tensors on one backend. The views `reshape`, `transpose`, `permute`, `slice`, `squeeze`, `unsqueeze` and `expand` share the storage (`reshape` of a non-dense tensor copies it first) and are differentiable. `contiguous()` copies a non-contiguous tensor and is differentiable. `to(DType)` converts into a new tensor; the casts among `FLOAT32`, `FLOAT16` and `BFLOAT16` are differentiable, and the result of any other cast never requires grad. `detach()` is a new handle over the same storage that never records.

### Tensor ownership

Device buffers have no GC-based release, so ownership is explicit:
- **Tensors.** Every tensor a factory or an op returns belongs to the caller, who closes it (`Tensor` is `AutoCloseable`, and closing twice does nothing); storage is reference-counted, so a view keeps its storage alive. `contiguous()`, `to(DType)` and `withRequiresGrad(boolean)` return the tensor itself when there is nothing to change (already contiguous, the same dtype, the same flag), so close such a result only when it is a different object.
- **Graph outputs.** When the nodes of a recorded computation are released (by the backward pass through them, by `ComputationGraph.current().release(root)` or by a closing `StepScope`), their outputs are closed, except the root of the pass; take a `detach()` handle first to keep an intermediate result.
- **Modules, optimizers and loaders.** A module closes its parameters, buffers and children; an optimizer closes its state buffers; a `DataLoader` and a `TensorDataset` release what they created. Modules, loaders and datasets are `AutoCloseable` and fit try-with-resources; an `Optimizer` is not, so close it in a `finally` block.

## Autograd

Autograd is reverse-mode and recorded per thread. An op records a node exactly when grad mode is on and one of its tensor inputs requires grad (`Autograd.needsGrad`); its result then requires grad. Parameters are leaves that require grad, and `tensor.withRequiresGrad(true)` makes a leaf of any tensor without a gradient function.

```java
Tensor loss = lossFn.compute(model.forward(features), targets);
optimizer.zeroGrad();
loss.backward();       // fills grad() of every trainable parameter the loss depends on
optimizer.step();
```

### Backward passes

| Call | Effect |
|---|---|
| `loss.backward()` | Seeds with ones and releases the traversed graph |
| `loss.backward(gradOutput)` | Seeds with `gradOutput`, which stays the caller's |
| `ComputationGraph.current().backward(loss, true)` | Keeps the graph and its saved tensors for another backward pass (`retainGraph`) |
| `ComputationGraph.current().backward(loss, gradOutput, retainGraph, createGraph)` | With `createGraph`, records the gradient computation so it can be differentiated again (implies `retainGraph`); a null `gradOutput` seeds with ones |
| `Autograd.grad(output, inputs, gradOutput, retainGraph, createGraph)` | `torch.autograd.grad`: returns the gradients with respect to `inputs` as new tensors the caller owns (null where none arrives) and writes no gradient slot |

Leaves accumulate their gradients in `grad()` across backward passes until `zeroGrad()`. A second backward through a released graph throws `IllegalStateException` with "the graph that produced this tensor was released or recorded on another thread; pass retainGraph=true to backpropagate twice, or detach() it". A create-graph pass, and `Autograd.grad` with `createGraph`, throw `UnsupportedOperationException` before computing anything when a node of the pass has no differentiable backward.

```java
import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import java.util.List;

Tensor x = Tensor.randn(backend, Shape.of(4)).withRequiresGrad(true);
Tensor y = TensorOps.sum(TensorOps.mul(x, x));                              // sum(x^2)
List<Tensor> g = Autograd.grad(y, List.of(x), null, true, true);            // dy/dx = 2x, recorded
Tensor gg = TensorOps.sum(g.get(0));
ComputationGraph.current().backward(gg);                                    // x.grad() = d(sum 2x)/dx = 2
```

### Grad mode and step scopes

- **`ComputationGraph.noGrad()`** turns grad mode off until the returned scope closes; **`ComputationGraph.enableGrad()`** turns it on inside a no-grad region. Both restore the previous mode.
- **Grad mode, the graph and create-graph are per thread.** A worker thread does not inherit its caller's mode or graph. A gradient can only flow through nodes recorded on the current thread.
- **`ComputationGraph.current().step()`** opens a `StepScope`. Closing it removes every node recorded during the step that no backward pass released, and closes their outputs, except tensors kept with `spare(t)`. The trainers, the RL algorithms and MAML run each step inside one.

```java
try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
    Tensor out = model.forward(input);     // nothing is recorded
}

ComputationGraph graph = ComputationGraph.current();
try (ComputationGraph.StepScope step = graph.step();
     Tensor loss = lossFn.compute(model.forward(x), y)) {
    loss.backward();                       // releases the nodes it traversed and closes their outputs
    float value = loss.toFloatArray()[0];
}   // closes the loss (the root of the pass), then removes any node of the step no backward pass reached
```

### In-place ops

`TensorOps.addInPlace(dst, src, alpha)`, `TensorOps.mulInPlace(dst, scalar)` and `TensorOps.zeroInPlace(dst)` write `dst` without recording. They throw `IllegalStateException` when `dst` requires grad while grad mode is on, so updates of parameters run under `noGrad()`, as the optimizers do.

### Custom operations

An operation records itself with `Autograd.record`, computing its raw result with recording off:

```java
import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.GradientFunction;
import java.util.Collections;
import java.util.List;

static Tensor triple(Tensor a) {
    if (!Autograd.needsGrad(a)) {
        return TensorOps.mul(a, 3f);
    }
    Tensor raw = Autograd.untracked(() -> TensorOps.mul(a, 3f));
    return Autograd.record(raw, List.of(a), out -> new GradientFunction() {
        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return List.of(TensorOps.mul(gradOutput, 3f));   // one entry per input
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;   // the backward is made of recorded ops
        }
    });
}
```

`GradientFunction` documents the rules a backward follows: it must not close `gradOutput`, returns one entry per input (null where none is needed), computes only with the public op facades so the CPU fallback and create-graph work, and saves tensors through detached handles.

### Gradient checkpointing

`CheckpointFunction` trades compute for memory: a segment runs without recording in the forward pass and is recomputed with recording when its gradient is needed.

```java
import com.eonmux.neuralfree.autograd.CheckpointFunction;

// x is an earlier layer's output that requires grad
Tensor h = CheckpointFunction.checkpoint(block::forward, x);
Tensor out = CheckpointFunction.checkpointSequential(List.of(block1::forward, block2::forward), x);
```

`checkpointSegments(segments, every, input)` checkpoints groups of segments, and `estimateMemorySavings(numLayers, activationBytes, every)` estimates the gain. The checkpointed result is recorded only when grad mode is on and the segment's input requires grad, as with PyTorch's reentrant checkpoint: a segment applied to an input that does not require grad (raw data entering the first layer) returns an unrecorded result, and its parameters receive no gradient. A segment must be a deterministic function of its input and of leaf tensors; every random draw in it (dropout masks, reparameterization noise) goes through `CheckpointFunction.replayable`, so the recompute replays it. A checkpointed node supports neither create-graph nor `Autograd.grad`; both throw `UnsupportedOperationException`.

### Gradient checks

```java
import com.eonmux.neuralfree.autograd.GradientCheckUtil;

GradientCheckUtil.GradientCheckResult r = GradientCheckUtil.gradCheck(
        in -> TensorOps.sum(TensorOps.relu(in.get(0))),
        List.of(input), 1e-3f, 1e-2f);
boolean ok = r.passed();   // maxDifference(), relativeError(), message()
```

`gradGradCheck` checks second-order gradients. `gradCheckParameters(loss, module, params, eps, tol, maxElementsPerParameter)` checks a module's parameters against a loss supplier and restores its buffers and gradient slots.

## Parameters

A `Parameter` owns a tensor that is always an autograd leaf, plus its gradient slot. `new Parameter(tensor)` and `new Parameter(tensor, name)` take the tensor over; use `parameter.tensor()` afterwards.
- **Trainable flag.** `requiresGrad()` and `setRequiresGrad(boolean)` make the parameter trainable or frozen; freezing closes its gradient. Change it between training steps, never between a forward pass and its backward pass. A frozen parameter stays frozen through moves, loads and copies.
- **Gradient slot.** `grad()`, `setGrad(t)` (takes `t` over and closes the gradient it replaces) and `zeroGrad()` (closes it).
- **Substitution.** `parameter.substitute(value)` opens a scope in which `tensor()` returns `value` instead of the owned tensor, so a forward pass computes with other weights (second-order meta-learning, spectral normalization, functional evaluation) while gradients flow to whatever `value` was computed from. The gradient slot keeps addressing the owned tensor. Scopes nest: an inner scope shadows an outer one until it closes. They belong to the thread that opened them. Moving or converting a substituted parameter is refused, and optimizers step after the scope closes. `Module.substitute(Map<Parameter, Tensor>)` substitutes several parameters in one scope.

```java
try (Parameter.Substitution scope = model.substitute(Map.of(weight, adaptedWeight))) {
    Tensor out = model.forward(x);   // reads adaptedWeight in place of weight
}
```

## Modules and layers

Every network extends `Module` and implements `forward(Tensor)`. Layer constructors take dimensions only, and `initialize(backend)` or `initialize(backend, random)` allocates a layer's parameters on the device.

```java
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import java.util.Random;

public class MLP extends Module {
    private final Linear fc1;
    private final Linear fc2;
    private final Parameter scale;

    public MLP(ComputeBackend backend, Random random, int in, int hidden, int out) {
        fc1 = registerModule("fc1", new Linear(in, hidden));
        fc1.initialize(backend, random);
        fc2 = registerModule("fc2", new Linear(hidden, out));
        fc2.initialize(backend, random);
        scale = registerParameter("scale", new Parameter(Tensor.ones(backend, Shape.of(out)), "scale"));
    }

    @Override
    public Tensor forward(Tensor x) {
        // forwardStep closes each intermediate result once the next step has consumed it
        Tensor h = forwardStep(fc1.forward(x), x, TensorOps::relu);
        Tensor y = forwardStep(h, x, fc2::forward);
        return forwardStep(y, x, t -> TensorOps.mul(t, scale.tensor()));
    }
}
```

### The module tree

- **Registration.** `registerModule(name, child)`, `registerParameter(name, parameter)` and `registerBuffer(name, tensor[, persistent[, keepDtype]])` are protected and called by subclasses. A parameter and a buffer may not share a name. Children and parameters are keyed by dotted paths (`fc1.weight`) in `namedParameters()`, `namedBuffers()` and `stateDict()`.
- **`replaceModule(name, child)`** (protected) makes `child` the module registered under `name`, keeping the position of a replaced name, and returns the replaced child unclosed; a null child unregisters the name. A replacement that would make a module its own descendant is refused with `IllegalArgumentException`.
- **`parameters()`** lists each parameter once by identity, so tied weights are stepped once.
- **Containers.** `Sequential(m1, m2, ...)` or `Sequential(List<Module>)` chains modules, with `add` and `get(index)`. `ModuleList(Module...)` or `ModuleList(Collection)` is a list whose `forward` chains its modules; `add(index, module)` inserts and re-indexes, while `remove`, `set` and `clear` are unsupported. `ModuleDict(Map)` is a map: `put(key, module)` replaces a child in place and returns the old one unclosed, keys may not be empty or contain `.`, `remove` and `clear` are unsupported, and `forward(key, input)` runs one child (plain `forward(input)` throws).
- **Modes and moves.** `train()`, `train(boolean)`, `eval()`, `zeroGrad()`, `setRequiresGrad(boolean)` (freezes or unfreezes the tree), `to(backend)` (moves in place through the host, dropping gradients), `to(DType)` (floating-point dtypes only; buffers registered with `keepDtype` keep theirs), `parameterCount()` and `close()`.

### State dicts

`stateDict()` returns `Map<String, Object>` of row-major `float[]` copies, as PyTorch's `state_dict`: for each module, depth first, its parameters, then its persistent buffers. Buffers registered with `persistent = false` are caches (positional tables, transform matrices) and stay out. `loadStateDict(map)` ignores missing, unexpected and non-`float[]` keys. `loadStateDict(map, true)` requires exactly the keys of `stateDict()` and reports missing, unexpected and wrong-type keys together. Either way every entry is checked before anything is written, so a refused load changes nothing. Parameters keep their objects and trainable flags.

### Auxiliary losses

A module whose forward pass computes an extra loss term (a mixture of experts' load-balance term, a VAE's KL term) hands it to the protected `setAuxiliaryLoss(term)`. Terms accumulate: a module applied several times before one backward pass contributes each application's term. `model.takeAuxiliaryLoss()` returns the sum of the pending terms of the whole tree, owned by the caller, or null when none is pending, and clears them; add it to the loss before the backward pass. `clearAuxiliaryLoss()` drops the pending terms. Take the terms on the thread that ran the forward passes. The trainers, `PipelineParallel`, MAML, `KnowledgeDistiller` and `FederatedTrainer` add them for you.

### BatchNorm

`BatchNorm1d` and `BatchNorm2d` normalize over the channels on dim 1. Constructors are `(numFeatures)`, `(numFeatures, eps, momentum, affine)` and `(numFeatures, eps, Float momentum, affine)`, with defaults `eps = 1e-5` and `momentum = 0.1`. A null momentum gives the cumulative moving average of every batch, `f = 1 / num_batches_tracked`.
- **State.** The parameters `weight` and `bias` (with `affine`), and the persistent buffers `running_mean`, `running_var` and `num_batches_tracked`, so `stateDict()` and checkpoints carry them.
- **Running update.** `running_mean = (1 - f) * running_mean + f * mean` and `running_var = (1 - f) * running_var + f * var * n / (n - 1)`: the running variance takes the unbiased batch variance, while the batch is normalized with the biased one. A training batch with one value per channel is rejected.
- **Batch counter.** `num_batches_tracked` is a FLOAT32 scalar that `to(DType)` leaves FLOAT32; it counts exactly up to `2^24` batches.

### Layer catalogue

| Package | Classes (constructor arguments) |
|---|---|
| `nn.layers` | `Linear(in, out[, bias])`, `Conv1d(inCh, outCh, kernel[, stride, padding])`, `Conv2d(inCh, outCh, int[] kernel[, int[] stride, int[] padding])`, `ConvTranspose2d(inCh, outCh, int[] kernel[, int[] stride, int[] padding])`, `MaxPool2d(int[] kernel[, ...])`, `AvgPool2d(int[] kernel[, ...])`, `AdaptiveAvgPool2d(outH, outW)`, `Dropout([p])`, `Embedding(num, dim[, paddingIdx])`, `Flatten([startDim[, endDim]])` |
| `nn.norm` | `BatchNorm1d`, `BatchNorm2d` (above), `LayerNorm(int or int[][, eps, affine])`, `GroupNorm(groups, channels[, eps, affine])`, `RMSNorm(dim[, eps, hasWeight])` |
| `nn.activations` | `ReLU`, `LeakyReLU([slope])`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax([dim])`, `LogSoftmax([dim])`, `ELU([alpha])`, `Mish`, `GLU([dim])`, `GeGLU([dim])` |
| `nn.transformer` | `MultiHeadAttention(dModel, heads[, dropout, causal[, batchFirst]])`, `TransformerEncoderLayer(dModel, heads[, dFF, dropout])`, `TransformerEncoder(layers, dModel, heads[, dFF, dropout])`, `TransformerDecoderLayer`, `TransformerDecoder`, `RotaryPE(dModel, maxSeqLen)`, `ALiBiPE(dModel, maxSeqLen, heads)`, `SinusoidalPE(dModel, maxSeqLen)`, `LearnedPE(dModel, maxSeqLen)`, `KVCache`, `MixtureOfExperts(dModel, dFF, experts, topK[, noiseStd, loadBalancing])`, `FlashAttentionOp` |
| `nn.recurrent` | `RNN`, `LSTM`, `GRU` (`(in, hidden, layers[, bidirectional])`), `RNNCell`, `LSTMCell`, `GRUCell`, `StateSpaceModel(in, hidden[, stateDim])` |
| `nn.vision` | `ResNet.resnet18/34/50/101/152(classes)`, `UNet(inCh, outCh[, depths, batchNorm])`, `ResidualBlock`, `BottleneckBlock`, `DepthwiseSeparableConv` |
| `nn.graph` | `GCNConv(in, out)`, `GATConv(in, out[, heads, concat])`, `GraphSAGEConv(in, out)`, `GINConv(in, out[, eps, trainEps])`, `GlobalMeanPool`, `GlobalMaxPool`, `TopKPool(in, ratio[, minScore][, nonlinearity])`, over `GraphData(nodeFeatures, edgeIndex)` |
| `nn.generative` | `VAE`, `GANTrainer`, `DDPMTrainer`, `DDIMSampler`, noise schedulers, `SpectralNorm`, `AutoregressiveGenerator`, `CFGWrapper` |
| `nn.multimodal` | `VisionEncoder`, `TextEncoder`, `AudioEncoder`, `CrossAttentionFusion`, `VisionLanguageModel` |
| `nn.structured` | `NeuralODE`, `NeuralRadianceField`, `GaussianSplatting`, `GraphTransformer` |
| `nn.selfsupervised` | `MaskedAutoencoder`, `CLIPTrainer`, `DINOTrainer`, `ContrastiveLearner`, `MaskedLanguageModel` |

Recurrent states returned by a cell are outputs of recorded nodes; to carry a state past the backward pass that closes them (truncated back-propagation through time), take its `detach()` first.

## Optimizers and schedulers

Optimizers take `(List<Parameter> params, float lr, ...)` and update the parameters in place on their backend:

| Optimizer | Short constructor defaults | Full constructor |
|---|---|---|
| `SGD` | momentum 0, weight decay 0, no Nesterov | `(params, lr, momentum, weightDecay, nesterov)` |
| `Adam` | betas 0.9 and 0.999, eps 1e-8, weight decay 0 | `(params, lr, beta1, beta2, eps, weightDecay)` |
| `AdamW` | as Adam, decoupled weight decay 0.01 | `(params, lr, beta1, beta2, eps, weightDecay)` |
| `RMSprop` | alpha 0.9, eps 1e-8, weight decay 0, not centered, momentum 0 | `(params, lr, alpha, eps, weightDecay, centered, momentum)` |
| `Adagrad` | eps 1e-10, weight decay 0, initial accumulator 0 | `(params, lr, eps, weightDecay, initialAccumulatorValue)` |
| `LAMB` | betas 0.9 and 0.999, eps 1e-6, weight decay 0, bias correction | `(params, lr, beta1, beta2, eps, weightDecay, biasCorrection)` |

```java
import com.eonmux.neuralfree.optim.*;

SGD   sgd   = new SGD(model.parameters(), 0.01f, /*momentum*/ 0.9f, /*weightDecay*/ 0f, /*nesterov*/ false);
AdamW adamw = new AdamW(model.parameters(), 1e-3f);

StepLR            step    = new StepLR(adamw, /*stepSize*/ 30, /*gamma*/ 0.1f);
CosineAnnealingLR cosine  = new CosineAnnealingLR(adamw, /*tMax*/ 100);
OneCycleLR        cycle   = new OneCycleLR(adamw, /*maxLr*/ 0.01f, /*totalSteps*/ 1000);
WarmupLR          warmup  = new WarmupLR(adamw, /*warmupSteps*/ 500, cosine);
ReduceOnPlateauLR plateau = new ReduceOnPlateauLR(adamw, "min", /*factor*/ 0.5f, /*patience*/ 5);
```

**Schedules.** `LRScheduler.step()` advances the per-batch step count and `stepEpoch()` the epoch count; each call applies the new rate to the optimizer at once. `StepLR` and `CosineAnnealingLR` follow the epoch count; `OneCycleLR` and `WarmupLR` follow the step count. The trainers call `step()` after every batch and `stepEpoch()` after every epoch. `OneCycleLR` throws `IllegalStateException` when stepped past `totalSteps`, so give it at least the number of batches the run trains. `ReduceOnPlateauLR` is driven by the caller with `step(metric)`.

**State dicts.** `optimizer.stateDict()` is a snapshot the caller owns: `step_count` (`Optimizer.STEP_COUNT_KEY`) as an `Integer`, the hyper-parameters except the learning rate, and one new FLOAT32 tensor copy per state buffer, keyed by family and parameter index (`velocity_0` for SGD; `moment_0` and `velocity_0` for Adam, AdamW and LAMB; `squared_grad_`, `momentum_buf_` and `avg_grad_` for RMSprop; `accumulator_` for Adagrad). Close its tensors with `Optimizer.closeStateDict(map)`. `loadStateDict(map)` accepts tensors or `float[]` values, checks every entry before writing, and ignores hyper-parameters. The learning rate is set with `setLearningRate` or a scheduler.

```java
Map<String, Object> state = adamw.stateDict();
try {
    other.loadStateDict(state);
} finally {
    Optimizer.closeStateDict(state);
}
```

## Loss functions

Losses implement `LossFunction.compute(prediction, target)` and return a tensor on the operands' backend; `LossFunction.Reduction` is `NONE`, `MEAN` (the default) or `SUM`. Read a scalar with `loss.toFloatArray()[0]`.

| Loss | Constructors |
|---|---|
| `CrossEntropyLoss` | `()`, `(reduction)`, `(reduction, dim, labelSmoothing)`, `(reduction, dim, labelSmoothing, ignoreIndex)` |
| `FocalLoss` | `()`, `(gamma)`, `(alpha, gamma)`, `(alpha, gamma, reduction, dim[, ignoreIndex])` |
| `KLDivLoss` | `()` (batchmean), `(reduction)`, `(reduction, logTarget)`, `KLDivLoss.batchMean(logTarget)` |
| `BCELoss`, `BCEWithLogitsLoss` | `()`, `(reduction)`; `BCEWithLogitsLoss(reduction, posWeight)` |
| `MSELoss`, `L1Loss` | `()`, `(reduction)` |
| `SmoothL1Loss`, `HuberLoss` | `()`, `(reduction)`, `(reduction, beta)` or `(reduction, delta)` |
| `ContrastiveLoss` | `()`, `(margin[, reduction])`; `compute(e1, e2, label)` |
| `TripletMarginLoss` | `()`, `(margin)`, `(margin, p, reduction)`; `compute(anchor, positive, negative)` |

**Cross-entropy** takes logits along `dim` (default the last). The target is either a FLOAT32 class distribution of the prediction's shape, or one class index per sample: `INT32`, `INT64`, or FLOAT32 values read as `Math.round(t)`. A sample whose class is `ignoreIndex` (default `CrossEntropyLoss.DEFAULT_IGNORE_INDEX = -100`) has loss 0 and is left out of the MEAN denominator; when every sample is ignored the mean is NaN. Any other class must lie in `[0, C)`: an out-of-range class throws `IndexOutOfBoundsException`, on the device at the next host synchronization. Integer targets support at most `2^24` classes.

```java
Tensor logits = model.forward(x);                                   // [batch, classes]
Tensor labels = Tensor.fromArray(backend, new int[]{2, 0, -100, 1});  // INT32; -100 is ignored
Tensor loss = new CrossEntropyLoss().compute(logits, labels);
```

**KL divergence** follows PyTorch's `kl_div`: the prediction holds log-probabilities (a `logSoftmax` output) and the target holds probabilities, or log-probabilities with `logTarget`. Per element it is `t * log(t) - t * x` (0 where `t == 0`). `new KLDivLoss()` uses `batchmean`, the sum divided by the batch size, which is the KL divergence per sample; `MEAN` divides by the element count instead.

## Data pipeline

`DataLoader<T>` is an `Iterable<List<T>>`: each element is one batch. Each `pass()` (and each `iterator()`) is an independent pass over a fresh sample order, and several may run at once.

```java
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;

TensorDataset dataset = new TensorDataset(inputs /* Tensor[] */, targets /* Tensor[] */);
DataLoader<TensorDataset.Sample> loader = DataLoader.builder(dataset)
        .batchSize(32).shuffle(true).numWorkers(4).build();
try (DataLoader.Pass<TensorDataset.Sample> pass = loader.pass()) {
    while (pass.hasNext()) {
        for (TensorDataset.Sample s : pass.next()) {
            Tensor x = s.input();
            Tensor y = s.target();
        }
    }
} finally {
    loader.close();
}
```

- **Loader options.** `batchSize`, `shuffle` or a `sampler` (`SequentialSampler`, `RandomSampler([seed])`, `WeightedRandomSampler(weights, numSamples, replacement[, seed])`), `numWorkers` (load ahead on background threads), `dropLast`, and `pinMemory` with a `backend`, which copies samples from another backend to the loader's.
- **Passes.** A consumer that leaves a pass before its last batch closes that pass (`Pass.close()`), which releases the batches loaded ahead; `loader.close()` ends every pass.
- **Sample ownership.** `new TensorDataset(Tensor[], Tensor[])` keeps the caller's tensors, which stay the caller's to close. `new TensorDataset(inputBatch, targetBatch)` creates one view per row, owned by the dataset and released by `close()`. Samples a `pinMemory` loader copied are owned by the sample: close each with `Sample.close()`, a no-op for a sample that owns nothing.
- **Other datasets.** `CSVDataset.builder()` (`csvPath`, `backend`, `hasHeader`, `delimiter`, `featureColumns`, `targetColumns`, `normalizeFeatures`, `oneHotEncodeTargets`), `ImageFolderDataset.builder()` (`rootPath`, `backend`, `targetSize`, `imagenetNormalization`, `grayscale`) and `TextFileDataset.builder()` (`filePath`, `backend`, tokenizer, `maxSequenceLength`). Their `get` creates new tensors the caller closes. The trainers take `DataLoader<TensorDataset.Sample>`, so build a `TensorDataset` from their tensors.
- **Image transforms** (`data.transforms.image`): `Resize`, `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`, `Normalize`, `ToTensor`.

## Training

`Trainer` runs the mini-batch loop. Per batch it stacks the samples into one batch on the device, runs the forward pass and the loss, and back-propagates the loss plus the auxiliary terms the forward pass left pending (`takeAuxiliaryLoss()`); the logged loss is the loss alone. At each accumulation boundary, under `noGrad()`:
1. the gradients are multiplied by `1 / n`, `n` being the number of batches back-propagated since the previous step (including a short last group), so the step applies their mean;
2. they are clipped, by norm when `gradientClipNorm > 0`, otherwise by value when `gradientClipValue > 0`;
3. the optimizer steps and the gradients are zeroed.

The first backward pass of each `train` call checks that the loss reached a trainable parameter of the optimizer; otherwise, or when the loss does not require grad, it throws `IllegalStateException` with `Trainer.NOT_CONNECTED` ("loss is not connected to any trainable parameter"). Every batch releases what it created, so the thread's graph is back to its size after each batch. Evaluation and `predict` run without recording.

```java
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.loss.CrossEntropyLoss;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.optim.CosineAnnealingLR;
import com.eonmux.neuralfree.training.*;
import java.util.List;

DataLoader<TensorDataset.Sample> trainLoader = DataLoader.builder(trainSet).batchSize(32).shuffle(true).build();
DataLoader<TensorDataset.Sample> valLoader = DataLoader.builder(valSet).batchSize(64).build();
Adam optimizer = new Adam(model.parameters(), 1e-3f);

TrainerConfig config = TrainerConfig.builder()
        .accumulationSteps(4)
        .gradientClipNorm(1.0f)
        .evalEveryNEpochs(1)
        .build();                                   // or TrainerConfig.defaults()
ModelCheckpoint checkpoints = ModelCheckpoint.builder()
        .directory("checkpoints").saveBestOnly("min").saveOptimizerState(true).keep(3)
        .build();

try (Trainer trainer = new Trainer(model, optimizer, new CrossEntropyLoss(), trainLoader, valLoader,
        List.of(new EarlyStopping(/*patience*/ 5), checkpoints, new ProgressLogger()),
        new CosineAnnealingLR(optimizer, /*tMax*/ 50), config)) {
    TrainingResult result = trainer.train(/*epochs*/ 50);
    System.out.println(result.totalSteps() + " steps, best validation loss " + result.bestValLoss()
            + " at epoch " + result.bestEpoch());
    checkpoints.restoreBest();                      // back to the best epoch's weights and optimizer state
} finally {
    trainLoader.close();
    valLoader.close();
    optimizer.close();
}
```

`TrainerConfig` defaults: `gradientClipNorm` 0 and `gradientClipValue` 0 (no clipping), `accumulationSteps` 1, `evalEveryNEpochs` 1 (validation runs after every `n`-th epoch when a validation loader is given), `verbose` true, `logEveryNSteps` 10 and `checkpointEveryNEpochs` 0.

`Trainer.builder()` builds the same trainer from named components. `TrainingResult` holds `trainLosses()`, `valLosses()`, `learningRates()` (one per epoch, recorded when a scheduler is set), `epochs()`, `totalSteps()`, `trainingTimeMs()`, `bestValLoss()`, `bestEpoch()`, `finalTrainLoss()` and `finalValLoss()`. An epoch's training loss is the mean of its batch losses, or NaN when it trained no batch because a stop was requested before its first one; NaN never counts as an improvement for `EarlyStopping` or `ModelCheckpoint`. `evaluate(loader)` returns an `EvalResult` (`loss()`, `accuracy()`, `metrics()`), with accuracy counted on the device for one-hot, probability or class-index targets. `predict(input)` runs the model in evaluation mode without recording. The trainer owns none of its components and closes none of them.

**Progress log.** With `verbose` the trainer logs its progress at INFO through the project logger (log4j), under its class name:
- a step line after every batch whose step count, the count `onBatchEnd` receives, is a multiple of `logEveryNSteps`: `epoch <e>/<epochs> step <s>: loss <loss>, lr <lr>`, with the batch's loss and the learning rate it trained with;
- an epoch line at the end of every epoch, before the callbacks' `onEpochEnd`: `epoch <e>/<epochs>: loss <train loss>`, followed by `, val_loss <val loss>` when validation ran.

Epochs and steps count from 1, each run counting its steps afresh, and values have six significant digits. The log is off when `verbose` is false, and also when a `ProgressLogger` callback with `verbose()` true is installed, since that callback reports the same progress. Nothing is written while INFO is disabled for the trainer's logger. The Gradle run tasks configure Log4j to log INFO to the console; an application configures it itself, since the library ships no logging configuration and Log4j's default prints errors only ([USAGE.md](USAGE.md#logging)).

**Checkpoint interval.** `checkpointEveryNEpochs(n)` with `n > 0` sets the periodic interval of every `ModelCheckpoint` callback, as Keras' `save_freq` and Lightning's `every_n_epochs` do: each epoch `e` with `(e + 1) % n == 0` is saved on top of the epochs the checkpoint's mode selects. Building a trainer with `n > 0` throws `IllegalArgumentException` when no callback is a `ModelCheckpoint`, or when one sets its own positive `saveEveryNEpochs` to a different value.

**Callbacks.** A `Callback` overrides any of `onTrainBegin(TrainingContext)`, `onTrainEnd(TrainingResult)`, `onEpochBegin`, `onEpochEnd(epoch, trainLoss, Float valLoss)`, `onBatchBegin`, `onBatchEnd(batch, loss)`, `onValidationBegin` and `onValidationEnd(epoch, EvalResult)`. The `TrainingContext` gives the model, optimizer, loss function and `TrainerConfig` (`config()`), and `requestStop()` ends the run before the next batch; the epoch in progress still steps, validates and reaches `onEpochEnd`. Built-in callbacks:
- `EarlyStopping` stops training when a monitored loss stops improving, as Keras' `EarlyStopping` does. `EarlyStopping.builder().monitor(m).patience(p).minDelta(d).mode(mode).restoreBestWeights(b).build()` sets it up. The defaults are monitor `"val_loss"`, patience 10, minDelta 0, mode `"min"` and no restore. The constructors `EarlyStopping([patience[, minDelta, mode, restoreBestWeights]])` are shorthands that monitor `"val_loss"`.
  - **Monitor.** `"val_loss"` is the epoch's validation loss and `"loss"` its training loss. An epoch without the monitored value, because no validation ran in it, is skipped: it neither improves nor counts toward the patience. When validation was due but did not run, as for a trainer without a validation loader, it logs one WARN per run and never stops training.
  - **Improvement.** In mode `"min"` a value improves when `value + minDelta < best`, in `"max"` when `value - minDelta > best`. NaN never improves.
  - **Stopping.** Each epoch with a value that does not improve adds one to the wait count, and an improvement resets it. When the count reaches the patience, at any epoch after the first, it calls `requestStop()`.
  - **Restore.** With `restoreBestWeights` it keeps a host copy of the model's `stateDict()`, every parameter and persistent buffer, from the end of the best epoch. When the run ends, stopped early or not, it loads that copy back with `loadStateDict(map, true)`, as Keras restores in `on_train_end`. The copy is taken under `noGrad()`; when no epoch improved it holds the first epoch that had a value.
  - **State.** `bestLoss()`, `bestEpoch()`, `stoppedEpoch()` (-1 when the run was not stopped) and `waitCount()` describe the current run, and `onTrainBegin` resets them.
- `ModelCheckpoint` (see [Checkpoints](#checkpoints-and-serialization)).
- `ProgressLogger([logEveryNEpochs, logEveryNSteps, logMetrics, verbose])` prints progress to standard output when `verbose`: a banner when the run starts and a summary when it ends; every `logEveryNEpochs`-th epoch its number, duration, training loss and validation loss; and the batch's position in its epoch and its loss after every batch whose step count is a multiple of `logEveryNSteps`. With `logMetrics` it also prints the validation results. The defaults are 1, 10, true and true. A verbose `ProgressLogger` turns the trainer's own progress log off.
- `MetricsTracker([trackAccuracy])`.

**Loader passes.** The trainer runs each epoch as one pass over the loader and closes a pass it leaves early (a stop request or a failure), which releases the batches loaded ahead without disturbing other passes over the same loader.

## Mixed precision

`MixedPrecisionTrainer` adds loss scaling and FP16 or BF16 emulation to the training loop. Every op computes in FLOAT32, so the `PrecisionMode` values `FP16` and `BF16` round the model input to the reduced format (round to nearest even) and widen it back before the forward pass, while `FP32` leaves it unchanged; parameters, gradients and optimizer state stay FLOAT32. It is built in one of two ways:

```java
import com.eonmux.neuralfree.precision.MixedPrecisionTrainer;

// Around an existing Trainer: takes its model, optimizer, loss, loaders, callbacks, scheduler and config
MixedPrecisionTrainer amp = MixedPrecisionTrainer.builder()
        .baseTrainer(trainer)
        .precisionMode(MixedPrecisionTrainer.PrecisionMode.BF16)
        .build();

// From components
MixedPrecisionTrainer amp2 = MixedPrecisionTrainer.builder()
        .model(model).optimizer(optimizer).lossFunction(new CrossEntropyLoss())
        .trainLoader(trainLoader).valLoader(valLoader)
        .addCallback(new EarlyStopping(5))
        .config(config)
        .precisionMode(MixedPrecisionTrainer.PrecisionMode.FP16)
        .dynamicLossScaling(true)
        .build();
TrainingResult result = amp2.train(20);
```

- **Base trainer.** Setting a component on a builder that has a base trainer makes `build()` throw `IllegalStateException`. The base trainer is not run and must not train at the same time, since the two share components.
- **Loss scaling.** The backward pass starts from the scaled loss plus the auxiliary terms. At each accumulation boundary the gradients are multiplied by `1 / (scale * n)` and checked for non-finite values, then clipped as true gradients, and the optimizer steps.
- **Overflow.** A non-finite loss or gradient zeroes the gradients and skips the step. With dynamic scaling the scale is divided by `lossScaleFactor`; without it a `RuntimeException` is thrown. After `lossScaleWindow` overflow-free batches the scale grows by the factor once the next step has completed. An epoch's training loss is the mean of its finite batch losses, or NaN when it has none, as when every batch overflowed.
- **Defaults.** Precision `FP16`, dynamic scaling on, initial scale 65536, factor 2, window 2000, scale range 1 to `2^30`. `currentLossScale()` reports the scale.
- Callbacks, the scheduler, the progress log, the checkpoint interval, `NOT_CONNECTED`, `evaluate` and `predict` behave as in `Trainer`. The progress log is written under `MixedPrecisionTrainer`'s class name, and the step line of a batch whose loss overflowed shows that non-finite loss. `build()` checks the checkpoint interval against the callbacks. `close()` closes neither the components nor a base trainer.

## Checkpoints and serialization

**`ModelCheckpoint`** is a callback that saves the model, and optionally the optimizer, at the end of selected epochs. The monitored value is the validation loss, or the training loss when no validation ran.

| Builder setting | Meaning | Default |
|---|---|---|
| `directory(String or Path)` | Where checkpoint files go (created when missing); none keeps checkpoints in memory | none |
| `prefix(name)` | File name prefix | `checkpoint` |
| `saveBestOnly(mode)` | `"min"` or `"max"` saves the improving epochs and tracks the best; `"none"` saves every epoch | `"min"` |
| `saveEveryNEpochs(n)` | Also saves every `n`-th epoch; a trainer's positive `checkpointEveryNEpochs` sets this interval for the run | 0 (off) |
| `saveOptimizerState(boolean)` | Includes the optimizer state | false |
| `keep(n)` | Keeps at most `n` checkpoints of the run, the best and the latest always among them | 0 (all) |

The constructors `ModelCheckpoint()`, `ModelCheckpoint(dir)` and `ModelCheckpoint(dir, saveBestOnly, saveEveryNEpochs, saveOptimizerState)` are shorthands. On disk each checkpoint is `<dir>/<prefix>-epoch<NNNN>.nfc`, written atomically through a temporary file. It is a `ModelSerializer` archive plus a `checkpoint.properties` entry holding the epoch, the losses, the monitored value, the mode, whether optimizer state is included and of which optimizer class, and the model's parameter and buffer names, which `ModelCheckpoint.readMetadata(path)` reads. A NaN value never becomes the best. `restore(epoch)` and `restoreBest()` load a checkpoint back into the trainer's model and optimizer, all or nothing, also from files an earlier process wrote. File system failures surface as `UncheckedIOException` naming the file.

**`ModelSerializer`** writes a zip archive:
- `model_state.bin`: the model's `stateDict()` (parameters and persistent buffers) as little-endian FLOAT32 records with their shapes;
- `optimizer_state.bin`: the optimizer's `stateDict()` as kind-tagged records, so state tensors keep their shapes and the step count and hyper-parameters keep their types (files whose records are all FLOAT32 tensors load as well);
- `model_meta.json`: parameter descriptions for `inspect`;
- `training_state.json`: optimizer class, step count and learning rate.

```java
import com.eonmux.neuralfree.io.ModelSerializer;

ModelSerializer.save(model, "model.nf");                         // weights and persistent buffers
ModelSerializer.save(model, optimizer, "checkpoint.nf");         // plus optimizer state
ModelSerializer.load("model.nf", model, backend);                // into an identically shaped model
ModelSerializer.load("checkpoint.nf", model, optimizer, backend);
ModelSerializer.CheckpointInfo info = ModelSerializer.inspect("checkpoint.nf");
```

A load reads and checks every entry before writing, so a refused file leaves the model and optimizer unchanged. The model state must hold every parameter with the model's shape and nothing the model lacks; buffers may be absent, and then keep their values. All methods throw `IOException`.

## Distributed training

`distributed` provides data, pipeline and federated parallelism. Collectives run over a `DistributedBackend`:
- `SharedMemoryBackend(rank, worldSize[, shmPrefix])` for ranks on one host, through memory-mapped files in `/dev/shm`;
- `TCPBackend(rank, worldSize, String[] addresses, int[] ports[, String bindAddress])` across hosts, one socket per pair of ranks.

**`SharedMemoryBackend`.** The ranks of one job find each other through the prefix, so pass the same prefix, unique to the job, to every rank; the two-argument form derives a prefix from the current time, which suits a single rank only. On file systems with POSIX permissions the segment files are created readable and writable by their owner only, so every rank must run as the same user, and a segment file other users can access is refused. Closing the backend deletes its segment files; a segment left by a job that ended without closing must be removed before its prefix is used again.

**`TCPBackend`.** Rank `r` listens on `addresses[r]:ports[r]` and connects to every lower rank. Pass `bindAddress` when the address the other ranks connect to is not a local interface of the rank, for example `0.0.0.0` behind NAT or a port forward. The protocol is neither authenticated nor encrypted: any host that can reach a rank's port can join as a rank, read the exchanged values and inject its own. Run it only on a trusted, private network, and firewall the rank ports from other hosts.

Every rank calls the same collectives in the same order with tensors of the same element count: `allReduce(tensor, ReduceOp)` (`SUM`, `MEAN`, `MAX`, `MIN`, `PRODUCT`), `broadcast`, `allGather` and `barrier`. Ranks first agree on each call's kind, op and size, and a mismatch throws. Results are reduced in rank order, so every rank holds bit-identical values, written into the caller's FLOAT32 tensor in place. Values cross through the host.

**`DistributedDataParallel(module, comm[, gradientBucketing, bucketSizeMb[, broadcastBuffers]])`**, with `comm` the rank's `DistributedBackend`, keeps one replica per rank. Bucketing is on by default with 25 MB buckets, and buffers are broadcast by default.
- **Construction** overwrites every rank's parameters and persistent buffers with rank 0's. With `broadcastBuffers`, every training-mode `forward` first broadcasts rank 0's buffers.
- **A step** is `ddp.backward(loss)` (backward, then the MEAN all-reduce of the gradients) followed by `optimizer.step()`, or `loss.backward()` followed by `ddp.step(optimizer)`.
- **Unused parameters.** A trainable parameter without a gradient on one rank contributes zeros, so all ranks issue the same collectives.

```java
import com.eonmux.neuralfree.distributed.DistributedDataParallel;
import com.eonmux.neuralfree.distributed.SharedMemoryBackend;

try (SharedMemoryBackend comm = new SharedMemoryBackend(rank, worldSize, "job42");
     DistributedDataParallel ddp = new DistributedDataParallel(model, comm)) {
    Tensor loss = lossFn.compute(ddp.forward(shardInputs), shardTargets);
    ddp.backward(loss);
    optimizer.step();
    ddp.zeroGrad();
}
```

**`PipelineParallel(stages, backends, numMicroBatches)`** splits the batch along dimension 0 into micro-batches that flow through the stage modules, each stage on its own worker thread and backend. `forward` records when grad mode is on and every stage is training. After the caller's backward pass, `pipeline.backward(output.grad())` backpropagates every micro-batch through the stages in reverse, including the stages' auxiliary loss terms, accumulates every stage's parameter gradients and returns the gradient of the input (null when the input does not require grad). Calls are serialized, stages that share a backend are serialized by its lock, and tensors cross between backends through the host.

**`FederatedTrainer(globalModel, optimizer, numClients, numRounds, localEpochs, lr[, strategy, proximalMu])`** simulates clients that share the global model and its backend. Each round every client starts from the global weights and persistent buffers, trains `localEpochs` steps on its data with the MSE loss (plus auxiliary terms), and the global state becomes the clients' mean (`FederatedTrainer.FederatedStrategy.FED_AVG`, the default). `FED_PROX` adds `proximalMu * (w - w_global)` to each gradient. `train(clientData, clientLabels)` takes one tensor of each per client and returns a `FederatedResult` with `roundLosses` and `finalLoss()`.

## Reinforcement learning

`rl` provides `DQN`, `A2C`, `PPO`, `SAC` and `RLHFTrainer`, each built with `builder()`. Every network of `DQN`, `A2C`, `PPO` and `SAC` is a perceptron with two hidden layers of width `hiddenSize`, `Linear(in, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, out)`, with He-normal hidden layers, a Glorot-normal output layer and zero biases, initialized from the algorithm's seed so a run is reproducible on every backend.

| Algorithm | Action space | `hiddenSize` | Other defaults |
|---|---|---|---|
| `DQN` | Discrete | 128 | gamma 0.99, epsilon 1.0 to 0.01 by 0.995, target update every 100 steps, batch 32 |
| `A2C` | Discrete | 64 | gamma 0.99, n-steps 5, entropy 0.01, value 0.5 |
| `PPO` | Discrete | 64 | gamma 0.99, lambda 0.95, clip 0.2, 4 epochs, batch 64 |
| `SAC` | Continuous | 256 | gamma 0.99, tau 0.005, alpha 0.2, automatic entropy tuning, batch 256 |

These four build their networks themselves, so their optimizers are passed as factories over the network's parameters, `Function<List<Parameter>, ? extends Optimizer>`. A non-positive `hiddenSize` is rejected.
- **`DQN`** takes a `ReplayBuffer(capacity, stateSize, backend[, prioritized, ...])`; transitions are added with `replayBuffer.add(state, action, reward, nextState, done)`. The target network starts as a frozen copy of the Q-network.
- **`A2C`** and **`PPO`** collect transitions with `storeTransition` and update with `trainStep()`.
- **`SAC`** has a tanh-squashed Gaussian actor and twin critics with Polyak-averaged targets. Its replay buffer stores the continuous actions as given (`ReplayBuffer.add(float[] state, float[] action, ...)`), and the critics train on exactly those actions. With `autoEntropy` the temperature is learned towards `targetEntropy = -actionSize`. `trainStep()` returns 0 until the buffer holds a batch.
- **`RLHFTrainer`** trains the caller's models (`policyModel`, `rewardModel`, `referenceModel`) with the caller's optimizers (`policyOptimizer`, `rewardOptimizer`). It fits the reward model to preference pairs with the Bradley-Terry loss and runs PPO with a KL penalty (`klCoeff`, default 0.02) against a frozen reference model, applied through the rewards.

```java
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.rl.ReplayBuffer;
import com.eonmux.neuralfree.rl.SAC;

SAC sac = SAC.builder()
        .backend(backend).stateSize(8).actionSize(2).hiddenSize(256)
        .actorOptimizer(params -> new Adam(params, 3e-4f))
        .criticOptimizer(params -> new Adam(params, 3e-4f))
        .alphaOptimizer(params -> new Adam(params, 3e-4f))
        .replayBuffer(new ReplayBuffer(100_000, 8, backend))
        .seed(1)
        .build();
float[] action = sac.selectAction(state, true);       // explore
sac.storeTransition(state, action, reward, nextState, done);
float criticLoss = sac.trainStep();
```

Environments implement `Environment<StateT, ActionT>` (`reset`, `step`, `actionSpaceSize`, `observationShape`); the algorithms implement `Policy`.

## Meta-learning and distillation

**`MAML(model, innerLr, outerLr, innerSteps, lossFunction[, firstOrder])`**, first order by default. Each task (`MAML.Task(supportInputs, supportTargets, queryInputs, queryTargets)`) adapts the weights with `innerSteps` SGD steps on its support set; the meta update applies the mean query gradient. The second-order path substitutes the adapted weights (`Module.substitute`) and differentiates through the adaptation with `Autograd.grad(..., createGraph = true)`, so every op of the model and loss must support create-graph. `metaTrain(tasks, iterations)`, `adapt(task)` and `evaluate(task)` run the loop.

**`KnowledgeDistiller(teacher, student, temperature, alpha, hardTargetLoss)`** trains a student on a frozen teacher (evaluation mode, gradients off, outputs computed under `noGrad`). With `t = softmax(teacherLogits / T)` and `log s = logSoftmax(studentLogits / T)` along the last dimension:

```
distillation = T^2 * (1 / B) * sum over all elements of t * (log t - log s)
total        = alpha * distillation + (1 - alpha) * hardTargetLoss(studentOutput, target)
```

`B` is the batch size, `dim(0)` for outputs of rank 2 or more and 1 otherwise. The distillation term is `KLDivLoss` with `batchmean`, scaled by `T^2` so its gradient `(T / B) * (softmax(studentLogits / T) - t)` keeps the scale of the hard-target gradient. `train(input, target, lr, steps)` takes SGD steps `w -= lr * g` on the student.

`meta` also holds `NASController`, `NASSearchSpace` and `FederatedMetaLearner`.

## Inference, quantization and export

```java
import com.eonmux.neuralfree.inference.InferenceEngine;
import com.eonmux.neuralfree.io.ONNXExporter;

try (InferenceEngine engine = new InferenceEngine(model, backend)) {   // switches the model to eval mode
    Tensor out = engine.infer(input);                                  // runs without recording
}
try (Tensor sample = Tensor.randn(backend, Shape.of(1, 3, 224, 224))) {
    new ONNXExporter().export(model, sample, "model.onnx");             // throws IOException
}
```

`inference` also holds `ContinuousBatcher`, `PagedAttention`, `SpeculativeDecoder`, `GraphOptimizer` and the configs `QuantizationConfig`, `KVCacheConfig` and `InferenceConfig`.

### Post-training quantization

`Quantizer` codes every parameter of a model, keyed by its dotted path. Types are `INT8_SYMMETRIC` (the default), `INT8_ASYMMETRIC`, `INT4_SYMMETRIC`, `INT4_ASYMMETRIC`, `FP8_E4M3` and `FP8_E5M2`. The FP8 types store the OCP 8-bit float pattern of `x / scale` with `scale = amax / fmax` (`fmax` 448 for E4M3, 57344 for E5M2), rounded to nearest-even and saturated.

| Calibration method | Clip range |
|---|---|
| `MIN_MAX` (default) | The running min and max, observed on the device |
| `PERCENTILE` | The given percentile of `\|x\|` (symmetric) or the `(100 - p, p)` percentiles of `x` (asymmetric; `p >= 50`) |
| `MSE` | Minimizes the expected squared quantization error |
| `ENTROPY` | TensorRT's KL-divergence calibration; integer types only |

The histogram methods read each parameter back once (or a sample of at most `numCalibrationSamples` values) and bin it in 2048 bins on the host. `calibrate(model)` merges observations across calls; `perChannel(true)` calibrates every slice of dimension 0 on its own.

```java
import com.eonmux.neuralfree.precision.Quantizer;
import java.util.Map;

Quantizer quantizer = Quantizer.builder()
        .quantType(Quantizer.QuantizationType.FP8_E4M3)
        .calibrationMethod(Quantizer.CalibrationMethod.PERCENTILE)
        .percentile(99.9f)
        .perChannel(true)
        .build();
quantizer.calibrate(model);
Map<String, Quantizer.QuantizedTensor> codes = quantizer.quantizeModel(model);
Tensor restored = quantizer.dequantizeTensor(codes.get("fc1.weight"));
```

### Quantization-aware training

`QATModule` wraps a base module (registered as its child `base`) and fake-quantizes, in each forward pass, every floating-point parameter of rank 2 or more (biases and normalization parameters stay full precision) and the input and output activations. The builder's `quantizeWeights` and `quantizeActivations` (both on by default) select these, `quantType` defaults to `INT8_SYMMETRIC`, and `perChannelQuantization` (off by default) gives each output channel of a weight its own range; activations are always quantized per tensor. Weights are substituted with their fake-quantized values, so the clipped straight-through gradient reaches them. FP8 types fake-quantize to `decode(encode(x / scale)) * scale`.

```java
import com.eonmux.neuralfree.precision.QATModule;

QATModule qat = QATModule.builder()
        .baseModule(model)
        .quantType(Quantizer.QuantizationType.INT8_ASYMMETRIC)
        .perChannelQuantization(true)
        .build();
// ... train qat like any module ...
qat.freezeObservers();
Quantizer exported = qat.toQuantizer();            // carries the observed scales and zero points
Map<String, Quantizer.QuantizedTensor> codes = exported.quantizeModel(qat.baseModule());
```

- **Observers.** Running min/max observers update whenever they are enabled (`enableObservers(boolean)`, `freezeObservers()`), independently of `enableFakeQuant(boolean)`. Their state is a set of persistent buffers (`min_val`, `max_val`, `scale`, `zero_point`) under `weight_fake_quant.<weight path with '.' written as '_'>` and `activation_fake_quant.input` / `.output`, so `stateDict()`, `ModelSerializer` and checkpoints save the calibration and `loadStateDict` restores it.
- **Export.** `toQuantizer()` returns a quantizer that codes exactly the fake-quantized weights with the observed parameters, leaves the other parameters in full precision and does not calibrate.

## Classical ML and other packages

| Package | Contents |
|---|---|
| `ml` | `KMeans`, `PCA`, `SVM`, `GaussianProcess`, `LogisticRegression`, `Perceptron` and `ForecastUnit` run on the train kernels of their backend and fall back to their CPU references. `RandomForest`, `GradientBoostedTrees` and `HiddenMarkovModel` run on the CPU; their backend argument must be non-null and is not used. |
| `acceleration.pipeline` | `PipelineScheduler` runs dense-network passes on the train kernels of one backend; it has no CPU path and throws `UnsupportedKernelException` when the library is unavailable. `new PipelineScheduler(backend, config)` takes an `acceleration.pipeline.schedule.PipelineConfig.builder()` (`enablePrefetch`, `prefetchDepth`, `enableKernelCache`, `enableMemoryPool`) configuration. |
| `acceleration.memory` | `GpuMemoryPool`, configured with `PoolConfig.builder()` (`maxPoolSizeBytes`, `minBlockSizeBytes`, `defaultBlockSizeBytes`) |
