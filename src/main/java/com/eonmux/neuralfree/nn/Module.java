package com.eonmux.neuralfree.nn;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorStorage;
import com.eonmux.neuralfree.tensor.TensorValues;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * Base class of the neural-network modules: a tree of named {@link Parameter}s, named buffers (non-trainable tensor
 * state such as running statistics) and named child modules.
 *
 * <p>{@link #to(ComputeBackend)}, {@link #to(DType)} and {@link #loadStateDict(Map)} update every parameter in place:
 * the {@code Parameter} objects keep their identity, so subclass fields and optimizers that hold them stay valid.
 * Buffers may be replaced by moved, converted or loaded tensors; subclasses read them through {@link #buffer(String)},
 * never from a cached field. These conversions and loads run under {@link ComputationGraph#noGrad()}, record nothing
 * and keep every parameter tensor a leaf with its trainable flag ({@link Parameter}).</p>
 *
 * <p><b>State.</b> {@link #stateDict()} holds every parameter and every persistent buffer (PyTorch's
 * {@code state_dict}); buffers registered with {@code persistent == false} are caches and stay out
 * ({@link #registerBuffer(String, Tensor, boolean)}). Values of every dtype are carried as {@code float[]}.
 * {@link #loadStateDict(Map, boolean)} restores both, in strict mode reporting missing and unexpected keys.</p>
 *
 * <p><b>Tree.</b> Children are registered under names ({@link #registerModule}, {@link #replaceModule}); a child's
 * parameters and buffers are keyed by the dotted path of names from this module ({@code "layer_0.fc1.weight"}) in
 * {@link #namedParameters()}, {@link #namedBuffers()} and {@link #stateDict()}, which list every path. As in PyTorch,
 * {@link #parameters()} and {@link #modules()} list each parameter and module once, by identity, even when one is
 * reachable under several names (weight sharing), so an optimizer built from {@link #parameters()} steps each shared
 * parameter once.</p>
 *
 * <p>{@link #setRequiresGrad(boolean)} freezes or unfreezes every parameter; {@link #substitute(Map)} lets a forward
 * pass read other tensors in place of parameters. A module whose forward computes an auxiliary loss term (a
 * load-balancing or KL term) hands it to {@link #setAuxiliaryLoss(Tensor)}, where the terms of every forward pass
 * accumulate; the training loop collects the pending terms of the whole tree with {@link #takeAuxiliaryLoss()} and
 * adds them to its loss before its backward pass.</p>
 */
public abstract class Module implements AutoCloseable {
    private final Map<String, Parameter> parameters = new LinkedHashMap<>();
    private final Map<String, Tensor> buffers = new LinkedHashMap<>();
    private final Map<String, Module> children = new LinkedHashMap<>();
    private final Set<String> parameterNames = new LinkedHashSet<>();
    /** The names of this module's buffers registered with {@code persistent == false}, kept out of the state dict. */
    private final Set<String> nonPersistentBuffers = new LinkedHashSet<>();
    /** The names of this module's buffers registered with {@code keepDtype == true}, which {@link #to(DType)} skips. */
    private final Set<String> dtypeKeptBuffers = new LinkedHashSet<>();

    private boolean training = true;
    /** The pending auxiliary loss terms of this module, in the order they were set. */
    private final List<PendingTerm> auxiliaryTerms = new ArrayList<>();
    private volatile boolean closed = false;

    protected Module() {
    }

    public abstract Tensor forward(Tensor input);

    public Tensor call(Tensor input) {
        return forward(input);
    }

    /**
     * Applies {@code step} to {@code intermediate}, a result this module's forward pass computed, then releases
     * {@code intermediate} ({@link #releaseIntermediate}) unless the step returned it; when the step fails,
     * {@code intermediate} is released as well. For chaining sub-module calls inside {@code forward} without leaking
     * the results in between, since device buffers have no GC-based release.
     */
    protected final Tensor forwardStep(Tensor intermediate, Tensor input, UnaryOperator<Tensor> step) {
        Tensor next;
        try {
            next = step.apply(intermediate);
        } catch (RuntimeException e) {
            try {
                releaseIntermediate(intermediate, input);
            } catch (RuntimeException releaseFailure) {
                e.addSuppressed(releaseFailure);
            }
            throw e;
        }
        if (next != intermediate) {
            releaseIntermediate(intermediate, input);
        }
        return next;
    }

    /**
     * Closes {@code intermediate} (null is ignored), a result this module's forward pass computed and no longer needs,
     * unless it is {@code input} (the caller's tensor), a tensor this module tree holds (a parameter's tensor or a
     * buffer, which a sub-module may return), or the output of a recorded autograd node, which the graph owns: the
     * release that follows the backward pass through it closes it (or {@link ComputationGraph#release(Tensor)} when no
     * backward pass runs), and leaves are never closed by the graph ({@link ComputationGraph}).
     */
    protected final void releaseIntermediate(Tensor intermediate, Tensor input) {
        if (intermediate == null || intermediate == input
                || ComputationGraph.current().getNode(intermediate) != null || holds(intermediate)) {
            return;
        }
        intermediate.close();
    }

    /** Whether {@code t} is a parameter's tensor or a buffer of this module or a descendant. */
    private boolean holds(Tensor t) {
        for (Parameter p : parameters()) {
            if (p.tensor() == t) {
                return true;
            }
        }
        return containsIdentity(namedBuffers().values(), t);
    }

    private static boolean containsIdentity(Iterable<?> items, Object t) {
        for (Object candidate : items) {
            if (candidate == t) {
                return true;
            }
        }
        return false;
    }

    protected Parameter registerParameter(String name, Parameter parameter) {
        ensureOpen();
        Objects.requireNonNull(name, "Parameter name cannot be null");
        
        if (parameterNames.contains(name)) {
            throw new IllegalArgumentException("Parameter '" + name + "' already registered");
        }
        if (buffers.containsKey(name)) {
            throw new IllegalArgumentException("Parameter '" + name + "' would share its state dict key with the"
                + " buffer of that name");
        }

        if (parameter == null) {
            throw new NullPointerException("Parameter '" + name + "' cannot be null");
        }
        
        parameters.put(name, parameter);
        parameterNames.add(name);
        
        return parameter;
    }

    /**
     * Registers {@code buffer} under {@code name} as persistent state: {@link #stateDict()} lists it and
     * {@link #loadStateDict(Map)} restores it. Equivalent to {@code registerBuffer(name, buffer, true)}.
     *
     * @throws IllegalArgumentException when a buffer or parameter of that name is already registered
     */
    protected Tensor registerBuffer(String name, Tensor buffer) {
        return registerBuffer(name, buffer, true);
    }

    /**
     * Registers {@code buffer} under {@code name}. The module takes ownership: it closes the buffer when the buffer
     * is replaced by {@link #to(ComputeBackend)}, {@link #to(DType)} or {@link #loadStateDict(Map, boolean)}, and on
     * {@link #close()}.
     *
     * <p>A persistent buffer is part of the module's state (running statistics, power-iteration vectors, EMA
     * centers): {@link #stateDict()} lists it and {@link #loadStateDict(Map, boolean)} restores it. A non-persistent
     * buffer ({@code persistent == false}, PyTorch's {@code persistent=False}) is a cache the module derives from its
     * configuration (a positional table, a window or transform matrix); it is moved and converted with the module and
     * listed by {@link #namedBuffers()}, but stays out of the state dict.</p>
     *
     * @throws IllegalArgumentException when a buffer or parameter of that name is already registered
     */
    protected Tensor registerBuffer(String name, Tensor buffer, boolean persistent) {
        return registerBuffer(name, buffer, persistent, false);
    }

    /**
     * {@link #registerBuffer(String, Tensor, boolean)}, where {@code keepDtype == true} keeps the buffer at its dtype
     * through {@link #to(DType)}: for state that is logically an integer but stored as a floating-point tensor (a
     * batch counter), which PyTorch holds as an integer tensor and therefore never converts. It is still moved by
     * {@link #to(ComputeBackend)}, and {@link #loadStateDict(Map, boolean)} writes it at its own dtype.
     *
     * @throws IllegalArgumentException when a buffer or parameter of that name is already registered
     */
    protected Tensor registerBuffer(String name, Tensor buffer, boolean persistent, boolean keepDtype) {
        ensureOpen();
        Objects.requireNonNull(name, "Buffer name cannot be null");
        Objects.requireNonNull(buffer, "Buffer '" + name + "' cannot be null");
        if (buffers.containsKey(name)) {
            throw new IllegalArgumentException("Buffer '" + name + "' already registered");
        }
        if (parameterNames.contains(name)) {
            throw new IllegalArgumentException("Buffer '" + name + "' would share its state dict key with the"
                + " parameter of that name");
        }
        buffers.put(name, buffer);
        if (!persistent) {
            nonPersistentBuffers.add(name);
        }
        if (keepDtype) {
            dtypeKeptBuffers.add(name);
        }
        return buffer;
    }

    /**
     * The current tensor of the buffer registered under {@code name}.
     *
     * @throws IllegalArgumentException when no buffer of that name is registered
     */
    protected Tensor buffer(String name) {
        ensureOpen();
        Tensor buffer = buffers.get(name);
        if (buffer == null) {
            throw new IllegalArgumentException("No buffer registered as '" + name + "'");
        }
        return buffer;
    }

    /** Every buffer of this module and its descendants, keyed by dotted path, in registration order. */
    public Map<String, Tensor> namedBuffers() {
        ensureOpen();
        return Collections.unmodifiableMap(namedBuffers(""));
    }

    private Map<String, Tensor> namedBuffers(String prefix) {
        Map<String, Tensor> result = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> entry : buffers.entrySet()) {
            result.put(prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey(), entry.getValue());
        }
        for (Map.Entry<String, Module> entry : children.entrySet()) {
            String childPrefix = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            result.putAll(entry.getValue().namedBuffers(childPrefix));
        }
        return result;
    }

    /**
     * Registers {@code module} as the child {@code name}, after every child registered so far; this module takes it
     * over and closes it on {@link #close()}. A null {@code module} registers nothing.
     *
     * @throws IllegalArgumentException when a child of that name is already registered
     */
    protected <T extends Module> T registerModule(String name, T module) {
        ensureOpen();
        Objects.requireNonNull(name, "Module name cannot be null");

        if (children.containsKey(name)) {
            throw new IllegalArgumentException("Child module '" + name + "' already registered");
        }

        if (module != null) {
            children.put(name, module);
        }

        return module;
    }

    /**
     * Makes {@code module} the child {@code name} whatever was registered there, and returns the child it replaces
     * ({@code module} itself when it already was that child, null when the name was free). A null {@code module}
     * unregisters {@code name}.
     *
     * <p><b>Order.</b> A replaced name keeps its position in the registration order, so {@link #parameters()},
     * {@link #namedParameters()}, {@link #namedBuffers()} and {@link #stateDict()} list the new child where the old one
     * was; a free name is appended after every child registered so far. The child's parameters and buffers are keyed
     * {@code name + "." + key}, as for {@link #registerModule}.</p>
     *
     * <p><b>Ownership.</b> This module takes {@code module} over as {@link #registerModule} does: {@link #to},
     * {@link #train(boolean)}, {@link #zeroGrad()}, {@link #setRequiresGrad(boolean)}, {@link #takeAuxiliaryLoss()} and
     * {@link #close()} reach it. The replaced child is unregistered without being closed and goes back to the caller,
     * with its parameters, buffers and pending auxiliary terms unchanged.</p>
     *
     * @throws IllegalArgumentException when {@code module} is this module or has it as a descendant (a cycle)
     * @throws IllegalStateException    when this module or {@code module} is closed
     */
    protected final Module replaceModule(String name, Module module) {
        ensureOpen();
        Objects.requireNonNull(name, "Module name cannot be null");
        if (module == null) {
            return children.remove(name);
        }
        if (containsIdentity(module.modules(), this)) {
            throw new IllegalArgumentException("Registering " + module.moduleName() + " as child '" + name
                + "' of " + moduleName() + " would make a module its own descendant");
        }
        return children.put(name, module);
    }

    protected <T extends Module> T addModule(String name, T module) {
        return registerModule(name, module);
    }

    public Parameter getParameter(String name) {
        ensureOpen();
        Objects.requireNonNull(name, "Parameter name cannot be null");
        
        if (parameters.containsKey(name)) {
            return parameters.get(name);
        }
        
        int dotIndex = name.indexOf('.');
        if (dotIndex > 0) {
            String childName = name.substring(0, dotIndex);
            String restName = name.substring(dotIndex + 1);
            Module child = children.get(childName);
            if (child != null) {
                return child.getParameter(restName);
            }
        }
        
        return null;
    }

    /**
     * Every parameter of this module and its descendants, each once by identity, in registration order (a module's own
     * parameters, then each child's in turn, depth first). A parameter reachable under several names (a shared module)
     * is listed at its first occurrence only.
     */
    public List<Parameter> parameters() {
        ensureOpen();
        Set<Parameter> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Parameter> result = new ArrayList<>();
        for (Module module : modules()) {
            for (Parameter param : module.parameters.values()) {
                if (seen.add(param)) {
                    result.add(param);
                }
            }
        }
        return Collections.unmodifiableList(result);
    }

    public Map<String, Parameter> namedParameters() {
        ensureOpen();
        return namedParameters("");
    }

    private Map<String, Parameter> namedParameters(String prefix) {
        Map<String, Parameter> result = new LinkedHashMap<>();
        
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            String name = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            result.put(name, entry.getValue());
        }
        
        for (Map.Entry<String, Module> entry : children.entrySet()) {
            String childPrefix = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            result.putAll(entry.getValue().namedParameters(childPrefix));
        }
        
        return result;
    }

    public List<Module> children() {
        ensureOpen();
        return new ArrayList<>(children.values());
    }

    public Map<String, Module> namedChildren() {
        ensureOpen();
        return new LinkedHashMap<>(children);
    }

    /**
     * This module and its descendants, each once by identity, depth first in registration order (this module first).
     */
    public List<Module> modules() {
        ensureOpen();
        List<Module> result = new ArrayList<>();
        collectModules(Collections.newSetFromMap(new IdentityHashMap<>()), result);
        return result;
    }

    private void collectModules(Set<Module> seen, List<Module> result) {
        if (!seen.add(this)) {
            return;
        }
        result.add(this);
        for (Module child : children.values()) {
            child.collectModules(seen, result);
        }
    }

    public Map<String, Module> namedModules() {
        ensureOpen();
        return namedModules("");
    }

    private Map<String, Module> namedModules(String prefix) {
        Map<String, Module> result = new LinkedHashMap<>();
        
        if (!prefix.isEmpty()) {
            result.put(prefix, this);
        }
        
        for (Map.Entry<String, Module> entry : children.entrySet()) {
            String childPrefix = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            result.putAll(entry.getValue().namedModules(childPrefix));
        }
        
        return result;
    }

    public void zeroGrad() {
        ensureOpen();
        
        for (Parameter param : parameters.values()) {
            param.zeroGrad();
        }
        
        for (Module child : children.values()) {
            child.zeroGrad();
        }
    }

    /**
     * Makes every parameter of this module and its descendants trainable ({@code true}) or frozen ({@code false}),
     * as {@link Parameter#setRequiresGrad(boolean)} does; freezing closes their gradients. Returns this module.
     */
    public Module setRequiresGrad(boolean requiresGrad) {
        ensureOpen();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (Parameter param : parameters()) {
                param.setRequiresGrad(requiresGrad);
            }
        }
        return this;
    }

    /**
     * Opens one scope that substitutes each key of {@code values}, a parameter of this module or a descendant, by its
     * value, as {@link Parameter#substitute(Tensor)} does for one parameter; closing the scope restores all of them.
     * The scope nests over substitutions already open (see {@link Parameter#substitute(Tensor)}). When one
     * substitution is refused, those already made are restored before the exception propagates.
     *
     * @throws NullPointerException     when {@code values}, a key or a value is null
     * @throws IllegalArgumentException when a key is not a parameter of this module tree, or a value differs from its
     *                                  parameter in shape, dtype or backend
     */
    public Parameter.Substitution substitute(Map<Parameter, Tensor> values) {
        ensureOpen();
        Objects.requireNonNull(values, "Substitution map cannot be null");
        Set<Parameter> owned = Collections.newSetFromMap(new IdentityHashMap<>());
        owned.addAll(parameters());
        for (Parameter param : values.keySet()) {
            Objects.requireNonNull(param, "Substituted parameter cannot be null");
            if (!owned.contains(param)) {
                throw new IllegalArgumentException("Parameter '" + param.name() + "' is not a parameter of "
                    + moduleName() + " or its descendants");
            }
        }
        Parameter.Substitution scope = new Parameter.Substitution();
        try {
            for (Map.Entry<Parameter, Tensor> entry : values.entrySet()) {
                entry.getKey().beginSubstitution(entry.getValue(), scope);
                scope.add(entry.getKey());
            }
        } catch (RuntimeException e) {
            scope.close();
            throw e;
        }
        return scope;
    }

    /**
     * Adds {@code term}, a scalar (one element) computed by a forward pass of this module, to the module's pending
     * auxiliary loss, taking it over. Terms accumulate until {@link #takeAuxiliaryLoss()} hands their sum to the
     * training loop, so a module applied several times before one backward pass (weight sharing, several forwards per
     * step) contributes the term of every application, as a PyTorch loop adding each forward's auxiliary loss does.
     *
     * <p>A term that requires grad is kept as it is and summed by {@link #takeAuxiliaryLoss()}. A term that does not
     * (a forward under {@link ComputationGraph#noGrad()}) is a constant: it is added at once to the module's pending
     * constant with an unrecorded {@link TensorOps#add} and closed, so constants never pile up.</p>
     *
     * <p><b>Dropped terms.</b> A pending term whose node left this thread's graph before it was taken (a
     * {@link ComputationGraph.StepScope} or {@link ComputationGraph#discardRecordedSince} discarded it, or
     * {@link ComputationGraph#release} or {@link ComputationGraph#clear()} removed it) is a released non-leaf that no
     * gradient can flow through. It is dropped: never summed, and closed by this module's next
     * {@code setAuxiliaryLoss}, {@link #takeAuxiliaryLoss()}, {@link #clearAuxiliaryLoss()} or {@link #close()}.</p>
     *
     * @throws NullPointerException     when {@code term} is null
     * @throws IllegalArgumentException when {@code term} has more than one element or is already pending; it is then
     *                                  not taken over. Any later failure closes {@code term} before it propagates.
     */
    protected final void setAuxiliaryLoss(Tensor term) {
        ensureOpen();
        Objects.requireNonNull(term, moduleName() + " auxiliary loss term cannot be null");
        if (term.numElements() != 1) {
            throw new IllegalArgumentException(moduleName() + " auxiliary loss must be a scalar, got shape "
                + term.shape());
        }
        for (PendingTerm pending : auxiliaryTerms) {
            if (pending.term == term) {
                throw new IllegalArgumentException(moduleName() + " auxiliary loss term is already pending");
            }
        }
        try {
            dropReleasedTerms();
            int constant = term.requiresGrad() ? -1 : pendingConstantIndex();
            if (constant < 0) {
                auxiliaryTerms.add(new PendingTerm(term));
                return;
            }
            Tensor previous = auxiliaryTerms.get(constant).term;
            auxiliaryTerms.set(constant, new PendingTerm(TensorOps.add(previous, term)));
            previous.close();
            term.close();
        } catch (RuntimeException e) {
            try {
                term.close();
            } catch (RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /**
     * The sum of the pending auxiliary loss terms of this module and its descendants, or null when none is pending;
     * every pending term is cleared. The terms are summed in {@link #modules()} order, each module's in the order they
     * were set; dropped terms ({@link #setAuxiliaryLoss}) are left out and closed. The caller owns the result; a single
     * term is handed over as is. The sum is computed with {@link TensorOps#add} in the caller's grad mode, so it is
     * recorded when a term requires grad, and adding it to the loss before the step's single backward pass delivers
     * the gradient of every term.
     *
     * <p>A consumed term (or partial sum) that is the output of a recorded operation stays in the graph when the
     * addition consuming it was recorded, and is released with the result's backward pass or
     * {@link ComputationGraph#release(Tensor)}; every other consumed tensor is released from the graph and closed, so a
     * sum of several terms taken under {@link ComputationGraph#noGrad()} leaves no node behind (a single recorded term
     * handed over as is keeps its node; the caller releases it). When the sum fails, every term not yet consumed and
     * the partial sum are released and closed before the exception propagates.</p>
     *
     * @throws IllegalStateException when a pending term is the output of a node recorded on another thread (the graph
     *                               is per thread, so the terms are taken on the thread that ran the forward passes);
     *                               nothing is changed then
     */
    public Tensor takeAuxiliaryLoss() {
        ensureOpen();
        List<Module> tree = modules();
        for (Module module : tree) {
            module.requireTermsOfThisThread();
        }
        List<Tensor> terms = new ArrayList<>();
        List<Tensor> dropped = new ArrayList<>();
        for (Module module : tree) {
            for (PendingTerm pending : module.auxiliaryTerms) {
                (pending.released() ? dropped : terms).add(pending.term);
            }
            module.auxiliaryTerms.clear();
        }
        RuntimeException dropFailure = discardAuxiliaryTerms(dropped, null);
        if (dropFailure != null) {
            throw discardAuxiliaryTerms(terms, dropFailure);
        }
        if (terms.isEmpty()) {
            return null;
        }
        Tensor sum = terms.get(0);
        int next = 1;
        try {
            while (next < terms.size()) {
                Tensor term = terms.get(next);
                Tensor added = TensorOps.add(sum, term);
                next++;
                Tensor previous = sum;
                sum = added;
                boolean recorded = ComputationGraph.current().getNode(added) != null;
                dropConsumedTerm(previous, added, recorded);
                dropConsumedTerm(term, added, recorded);
            }
            return sum;
        } catch (RuntimeException e) {
            List<Tensor> unconsumed = new ArrayList<>();
            unconsumed.add(sum);
            unconsumed.addAll(terms.subList(next, terms.size()));
            throw discardAuxiliaryTerms(unconsumed, e);
        }
    }

    /**
     * Drops the pending auxiliary loss terms of this module and its descendants without summing them: each is released
     * from this thread's graph ({@link ComputationGraph#release(Tensor)}) and closed, as {@link #close()} does. For a
     * caller that runs forward passes in training mode outside a training step, or abandons a step before its backward
     * pass. A term recorded on another thread is only closed. Every term is tried; the first failure is thrown with
     * later ones suppressed.
     */
    public void clearAuxiliaryLoss() {
        ensureOpen();
        List<Tensor> pending = new ArrayList<>();
        for (Module module : modules()) {
            module.drainAuxiliaryTerms(pending);
        }
        RuntimeException failure = discardAuxiliaryTerms(pending, null);
        if (failure != null) {
            throw failure;
        }
    }

    /** Moves this module's pending terms, dropped ones included, into {@code into} and clears them. */
    private void drainAuxiliaryTerms(List<Tensor> into) {
        for (PendingTerm pending : auxiliaryTerms) {
            into.add(pending.term);
        }
        auxiliaryTerms.clear();
    }

    /** Removes this module's dropped terms and closes them. */
    private void dropReleasedTerms() {
        List<Tensor> dropped = new ArrayList<>();
        Iterator<PendingTerm> it = auxiliaryTerms.iterator();
        while (it.hasNext()) {
            PendingTerm pending = it.next();
            if (pending.released()) {
                it.remove();
                dropped.add(pending.term);
            }
        }
        RuntimeException failure = discardAuxiliaryTerms(dropped, null);
        if (failure != null) {
            throw failure;
        }
    }

    /** The position of this module's pending constant term (one that does not require grad), or -1. */
    private int pendingConstantIndex() {
        for (int i = 0; i < auxiliaryTerms.size(); i++) {
            if (!auxiliaryTerms.get(i).term.requiresGrad()) {
                return i;
            }
        }
        return -1;
    }

    /** Refuses, before anything changes, when a pending term of this module was recorded on another thread. */
    private void requireTermsOfThisThread() {
        for (PendingTerm pending : auxiliaryTerms) {
            if (pending.recordedElsewhere()) {
                throw new IllegalStateException(moduleName() + " has an auxiliary loss term recorded on another"
                    + " thread; take it on the thread that ran the forward pass");
            }
        }
    }

    /**
     * Disposes of {@code consumed}, an input of {@code result}: nothing when it is {@code result}, or when
     * {@code result} was recorded and {@code consumed} is a recorded node's output (the graph then owns it);
     * otherwise it is released from the graph and closed.
     */
    private static void dropConsumedTerm(Tensor consumed, Tensor result, boolean resultRecorded) {
        if (consumed == result) {
            return;
        }
        if (resultRecorded && ComputationGraph.current().getNode(consumed) != null) {
            return;
        }
        discardAuxiliaryTerm(consumed);
    }

    /** Releases {@code term} from this thread's graph (nothing when it has no node there) and closes it. */
    private static void discardAuxiliaryTerm(Tensor term) {
        try {
            ComputationGraph.current().release(term);
        } finally {
            term.close();
        }
    }

    /**
     * Discards each of {@code terms} ({@link #discardAuxiliaryTerm}); every term is tried. Returns {@code failure}
     * with the new failures added as suppressed, or, when {@code failure} is null, the first new failure with the later
     * ones suppressed (null when none failed).
     */
    private static RuntimeException discardAuxiliaryTerms(List<Tensor> terms, RuntimeException failure) {
        RuntimeException result = failure;
        for (Tensor t : terms) {
            try {
                discardAuxiliaryTerm(t);
            } catch (RuntimeException e) {
                if (result == null) {
                    result = e;
                } else {
                    result.addSuppressed(e);
                }
            }
        }
        return result;
    }

    public boolean isTraining() {
        return training;
    }

    public void train() {
        train(true);
    }

    public void train(boolean training) {
        ensureOpen();
        this.training = training;
        
        for (Module child : children.values()) {
            child.train(training);
        }
    }

    public void eval() {
        train(false);
    }

    /**
     * Moves every parameter (in place) and buffer of this module and its descendants to {@code backend}. Values
     * cross through the host; gradients are dropped. Runs under {@link ComputationGraph#noGrad()}.
     *
     * @throws IllegalStateException when a parameter of the tree is substituted ({@link #substitute(Map)}); nothing
     *                               is moved then
     */
    public Module to(ComputeBackend backend) {
        ensureOpen();
        Objects.requireNonNull(backend, "Backend cannot be null");
        ensureNoSubstitution("moved to another backend");

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (Parameter param : parameters.values()) {
                param.moveTo(backend);
            }
            for (Map.Entry<String, Tensor> entry : buffers.entrySet()) {
                Tensor current = entry.getValue();
                if (current.backend() != backend) {
                    entry.setValue(Parameter.copyToBackend(current, backend));
                    current.close();
                }
            }
            
            for (Module child : children.values()) {
                child.to(backend);
            }
        }
        
        return this;
    }

    /**
     * Converts every floating-point parameter (in place) and every floating-point buffer of this module and its
     * descendants to the floating-point {@code dtype} ({@link Tensor#to(DType)}); gradients are dropped. As in
     * PyTorch, integer and boolean parameters and buffers keep their dtype, and so do buffers registered with
     * {@code keepDtype} ({@link #registerBuffer(String, Tensor, boolean, boolean)}). A conversion between two dtypes
     * other than FLOAT32 passes through FLOAT32, which holds every FLOAT16 and BFLOAT16 value exactly; a FLOAT64 value
     * is rounded to FLOAT32 on the way. Runs under {@link ComputationGraph#noGrad()}, so every parameter tensor stays
     * a leaf.
     *
     * @throws IllegalArgumentException when {@code dtype} is not floating-point; nothing is converted then
     * @throws IllegalStateException    when a parameter of the tree is substituted ({@link #substitute(Map)}); nothing
     *                                  is converted then
     */
    public Module to(DType dtype) {
        ensureOpen();
        Objects.requireNonNull(dtype, "DType cannot be null");
        if (!dtype.isFloatingPoint()) {
            throw new IllegalArgumentException("Module.to only accepts floating-point dtypes, got " + dtype);
        }
        ensureNoSubstitution("converted to another dtype");

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (Parameter param : parameters.values()) {
                DType current = param.dtype();
                if (current.isFloatingPoint() && current != dtype) {
                    if (current != DType.FLOAT32 && dtype != DType.FLOAT32) {
                        param.convertTo(DType.FLOAT32);
                    }
                    param.convertTo(dtype);
                }
            }
            for (Map.Entry<String, Tensor> entry : buffers.entrySet()) {
                Tensor current = entry.getValue();
                if (current.dtype().isFloatingPoint() && current.dtype() != dtype
                        && !dtypeKeptBuffers.contains(entry.getKey())) {
                    entry.setValue(convertFloating(current, dtype));
                    current.close();
                }
            }
            
            for (Module child : children.values()) {
                child.to(dtype);
            }
        }
        
        return this;
    }

    /**
     * {@code source} converted to the floating-point {@code dtype}, through FLOAT32 when neither dtype is FLOAT32
     * ({@link #to(DType)}); a new tensor. The caller holds a no-grad scope.
     */
    private static Tensor convertFloating(Tensor source, DType dtype) {
        if (source.dtype() == DType.FLOAT32 || dtype == DType.FLOAT32) {
            return source.to(dtype);
        }
        try (Tensor floats = source.to(DType.FLOAT32)) {
            return floats.to(dtype);
        }
    }

    public ComputeBackend backend() {
        ensureOpen();
        
        for (Parameter param : parameters.values()) {
            return param.backend();
        }
        
        for (Module child : children.values()) {
            ComputeBackend backend = child.backend();
            if (backend != null) {
                return backend;
            }
        }
        
        return null;
    }

    /**
     * The state of this module and its descendants as {@code float[]} values (row-major), keyed by dotted path, with
     * PyTorch's {@code state_dict} contents and order: for each module, depth first in registration order, its
     * parameters, then its persistent buffers ({@link #registerBuffer(String, Tensor, boolean)}); non-persistent
     * buffers are left out. Like {@link #namedParameters()}, a module reachable under several names is listed under
     * each. The parameters' own values are read, also while a substitution is open. Values of every dtype are
     * returned as {@code float[]}: a FLOAT32 tensor's own values, a FLOAT16, BFLOAT16 or FLOAT64 tensor's values
     * converted to FLOAT32, integer values rounded to the nearest float and booleans as {@code 0} or {@code 1}.
     * Values cross through the host.
     */
    public Map<String, Object> stateDict() {
        ensureOpen();
        Map<String, Object> state = new LinkedHashMap<>();
        for (StateEntry entry : stateEntries()) {
            state.put(entry.path, entry.read());
        }
        return state;
    }

    /**
     * {@link #loadStateDict(Map, boolean)} in non-strict mode: keys missing from {@code stateDict}, keys that name no
     * state of this module tree, and keys whose value is not a {@code float[]} are ignored.
     *
     * @throws IllegalArgumentException when an entry's length differs from its parameter's or buffer's element count
     */
    public void loadStateDict(Map<String, Object> stateDict) {
        loadStateDict(stateDict, false);
    }

    /**
     * Restores every parameter and persistent buffer whose dotted path ({@link #stateDict()}) maps to a
     * {@code float[]} in {@code stateDict} (row-major values). Parameters are overwritten in place, keeping the
     * {@code Parameter} objects and their trainable flags ({@link Parameter}). A FLOAT32 buffer is overwritten in
     * place; a buffer of another dtype is replaced by a tensor converted from the values, and the old one is closed.
     * An assignment under {@link ComputationGraph#noGrad()} that records nothing.
     *
     * <p>Every entry is checked before anything is written, so a refused load changes nothing. In strict mode the
     * keys of {@code stateDict} must be exactly the keys of {@link #stateDict()}, each with a {@code float[]} value;
     * otherwise the missing keys, the unexpected keys and the keys holding another value type are reported together.
     * In non-strict mode those keys are ignored.</p>
     *
     * <p>A value is converted to its tensor's dtype as {@link Tensor#to(DType)} converts FLOAT32: rounded to a
     * floating-point dtype, truncated toward zero to an integer dtype, and nonzero to {@code true} for BOOL.</p>
     *
     * @throws NullPointerException          when {@code stateDict} is null
     * @throws IllegalArgumentException      when an entry's length differs from its parameter's or buffer's element
     *                                       count, or, in strict mode, when the keys do not match
     * @throws UnsupportedOperationException when an entry names an INT8, INT16 or INT64 parameter, which
     *                                       {@code float[]} values cannot be loaded into
     */
    public void loadStateDict(Map<String, Object> stateDict, boolean strict) {
        ensureOpen();
        Objects.requireNonNull(stateDict, "State dict cannot be null");

        List<StateEntry> entries = stateEntries();
        Set<String> expected = new HashSet<>();
        List<StateEntry> loaded = new ArrayList<>();
        List<String> missing = new ArrayList<>();
        List<String> wrongType = new ArrayList<>();
        for (StateEntry entry : entries) {
            expected.add(entry.path);
            if (!stateDict.containsKey(entry.path)) {
                missing.add(entry.path);
                continue;
            }
            Object value = stateDict.get(entry.path);
            if (!(value instanceof float[])) {
                wrongType.add(entry.path);
                continue;
            }
            entry.requireLoadable((float[]) value);
            loaded.add(entry);
        }
        if (strict) {
            List<String> unexpected = new ArrayList<>();
            for (String key : stateDict.keySet()) {
                if (!expected.contains(key)) {
                    unexpected.add(key);
                }
            }
            if (!missing.isEmpty() || !unexpected.isEmpty() || !wrongType.isEmpty()) {
                throw new IllegalArgumentException(strictMismatch(missing, unexpected, wrongType));
            }
        }

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (StateEntry entry : loaded) {
                entry.write((float[]) stateDict.get(entry.path));
            }
        }
    }

    /** The message of a strict {@link #loadStateDict(Map, boolean)} whose keys do not match the module's state. */
    private String strictMismatch(List<String> missing, List<String> unexpected, List<String> wrongType) {
        StringBuilder message = new StringBuilder("State dict does not match ").append(moduleName()).append(':');
        if (!missing.isEmpty()) {
            message.append(" missing keys ").append(missing).append(';');
        }
        if (!unexpected.isEmpty()) {
            message.append(" unexpected keys ").append(unexpected).append(';');
        }
        if (!wrongType.isEmpty()) {
            message.append(" keys without a float[] value ").append(wrongType).append(';');
        }
        message.setLength(message.length() - 1);
        return message.toString();
    }

    /** The parameters and persistent buffers of this module tree in {@link #stateDict()} order. */
    private List<StateEntry> stateEntries() {
        List<StateEntry> entries = new ArrayList<>();
        collectStateEntries("", entries);
        return entries;
    }

    private void collectStateEntries(String prefix, List<StateEntry> into) {
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            into.add(new StateEntry(dottedPath(prefix, entry.getKey()), this, entry.getKey(), entry.getValue()));
        }
        for (String name : buffers.keySet()) {
            if (!nonPersistentBuffers.contains(name)) {
                into.add(new StateEntry(dottedPath(prefix, name), this, name, null));
            }
        }
        for (Map.Entry<String, Module> entry : children.entrySet()) {
            entry.getValue().collectStateEntries(dottedPath(prefix, entry.getKey()), into);
        }
    }

    private static String dottedPath(String prefix, String name) {
        return prefix.isEmpty() ? name : prefix + "." + name;
    }

    /**
     * Overwrites the buffer {@code name} with {@code values}: in place for FLOAT32, otherwise by a tensor of the
     * buffer's dtype, shape and backend converted from the values ({@link #fromFloatValues}), which replaces the
     * closed old one. The caller checked the length and holds a no-grad scope.
     */
    private void loadBuffer(String name, float[] values) {
        Tensor current = buffers.get(name);
        if (current.dtype() == DType.FLOAT32) {
            GpuOptimizerOps.storeFloats(current, values);
            return;
        }
        Tensor converted = fromFloatValues(current.backend(), current.shape(), current.dtype(), values);
        buffers.put(name, converted);
        current.close();
    }

    /**
     * A tensor of {@code dtype}, {@code shape} and {@code backend} holding {@code values} converted as
     * {@link #loadStateDict(Map, boolean)} describes: a floating-point dtype through {@link Tensor#to(DType)}, an
     * integer or boolean dtype on the host with Java's narrowing casts, which {@link Tensor#to(DType)} also applies.
     */
    private static Tensor fromFloatValues(ComputeBackend backend, Shape shape, DType dtype, float[] values) {
        if (dtype.isFloatingPoint()) {
            Tensor floats = Tensor.fromStorage(backend, shape, DType.FLOAT32,
                TensorStorage.fromHostFloat(backend, values));
            if (dtype == DType.FLOAT32) {
                return floats;
            }
            try {
                return floats.to(dtype);
            } finally {
                floats.close();
            }
        }
        ByteBuffer bytes = ByteBuffer.allocate(Math.multiplyExact(values.length, dtype.byteSize()))
            .order(ByteOrder.nativeOrder());
        for (int i = 0; i < values.length; i++) {
            putInteger(bytes, dtype, i, values[i]);
        }
        return Tensor.fromStorage(backend, shape, dtype, TensorStorage.fromHost(backend, bytes, dtype));
    }

    /** Stores {@code value} as element {@code i} of the integer or BOOL {@code bytes}. */
    private static void putInteger(ByteBuffer bytes, DType dtype, int i, float value) {
        switch (dtype) {
            case BOOL: bytes.put(i, (byte) (value != 0f ? 1 : 0)); break;
            case INT8: bytes.put(i, (byte) (int) value); break;
            case INT16: bytes.putShort(i * Short.BYTES, (short) (int) value); break;
            case INT32: bytes.putInt(i * Integer.BYTES, (int) value); break;
            case INT64: bytes.putLong(i * Long.BYTES, (long) value); break;
            default: throw new IllegalStateException("Not an integer or boolean dtype: " + dtype);
        }
    }

    /** The number of elements of {@link #parameters()}, so a shared parameter is counted once. */
    public int parameterCount() {
        ensureOpen();
        int count = 0;
        
        for (Parameter param : parameters()) {
            count += param.numElements();
        }
        
        return count;
    }

    public String moduleName() {
        return this.getClass().getSimpleName();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Module has been closed");
        }
    }

    /** Refuses {@code action} before anything changes when a parameter of the tree is substituted. */
    private void ensureNoSubstitution(String action) {
        for (Parameter param : parameters()) {
            param.ensureNotSubstituted(action);
        }
    }

    /**
     * Releases and closes this module's pending auxiliary loss terms ({@link #clearAuxiliaryLoss()}), then closes the
     * parameters, buffers and child modules; idempotent. Everything is closed even when a step fails; the first
     * failure is thrown afterwards with later ones suppressed.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            
            List<Tensor> pending = new ArrayList<>();
            drainAuxiliaryTerms(pending);
            RuntimeException failure = discardAuxiliaryTerms(pending, null);
            
            for (Parameter param : parameters.values()) {
                failure = closeCollecting(param::close, failure);
            }
            parameters.clear();
            parameterNames.clear();

            for (Tensor buffer : buffers.values()) {
                failure = closeCollecting(buffer::close, failure);
            }
            buffers.clear();
            nonPersistentBuffers.clear();
            dtypeKeptBuffers.clear();

            for (Module child : children.values()) {
                failure = closeCollecting(child::close, failure);
            }
            children.clear();
            
            if (failure != null) {
                throw failure;
            }
        }
    }

    /**
     * Runs {@code close}; returns {@code failure} when it succeeds, otherwise {@code failure} with the new failure
     * added as suppressed, or the new failure itself when {@code failure} is null.
     */
    private static RuntimeException closeCollecting(Runnable close, RuntimeException failure) {
        try {
            close.run();
            return failure;
        } catch (RuntimeException e) {
            if (failure == null) {
                return e;
            }
            failure.addSuppressed(e);
            return failure;
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return moduleName() + "{closed=true}";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append(moduleName()).append("{");
        sb.append("parameters=").append(parameters.size());
        sb.append(", children=").append(children.size());
        sb.append(", training=").append(training);
        sb.append("}");
        return sb.toString();
    }

    /** One parameter ({@code parameter != null}) or persistent buffer of the state dict, under its dotted path. */
    private static final class StateEntry {
        private final String path;
        private final Module owner;
        private final String name;
        private final Parameter parameter;

        StateEntry(String path, Module owner, String name, Parameter parameter) {
            this.path = path;
            this.owner = owner;
            this.name = name;
            this.parameter = parameter;
        }

        private Tensor tensor() {
            return parameter != null ? parameter.ownedTensor() : owner.buffers.get(name);
        }

        float[] read() {
            return TensorValues.toFloats(tensor());
        }

        /**
         * @throws IllegalArgumentException      when {@code values} differs in length from the element count
         * @throws UnsupportedOperationException for an INT8, INT16 or INT64 parameter, which
         *                                       {@link Parameter#load(float[])} cannot convert FLOAT32 values to
         */
        void requireLoadable(float[] values) {
            Tensor target = tensor();
            if (values.length != target.numElements()) {
                throw new IllegalArgumentException((parameter != null ? "Parameter '" : "Buffer '") + path
                    + "' of shape " + target.shape() + " holds " + target.numElements() + " values, got "
                    + values.length);
            }
            DType dtype = target.dtype();
            if (parameter != null && (dtype == DType.INT8 || dtype == DType.INT16 || dtype == DType.INT64)) {
                throw new UnsupportedOperationException("Parameter '" + path + "' of dtype " + dtype
                    + " cannot be loaded from float[] values");
            }
        }

        void write(float[] values) {
            if (parameter != null) {
                parameter.load(values);
            } else {
                owner.loadBuffer(name, values);
            }
        }
    }

    /** A pending auxiliary loss term and the graph of the thread that set it. */
    private static final class PendingTerm {
        private final Tensor term;
        private final ComputationGraph graph;

        PendingTerm(Tensor term) {
            this.term = term;
            this.graph = ComputationGraph.current();
        }

        /** Whether the term is the output of a node recorded in another thread's graph. */
        boolean recordedElsewhere() {
            return term.gradFn() != null && graph != ComputationGraph.current();
        }

        /** Whether the term's node left this thread's graph: a released non-leaf, dropped unsummed. */
        boolean released() {
            return term.gradFn() != null && graph == ComputationGraph.current() && graph.getNode(term) == null;
        }
    }
}
