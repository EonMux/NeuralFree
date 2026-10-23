package com.eonmux.neuralfree.nn.selfsupervised;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuTensorOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * DINO self-distillation with an exponential-moving-average teacher. Sharpening, centering, the cross-entropy loss,
 * the center update and the teacher update are {@link TensorOps} compositions and in-place
 * {@link GpuTensorOps#axpbyInPlace} updates, so they run on the device when the train kernels are available and on
 * the CPU reference otherwise; nothing is read back.
 *
 * <p>Gradients: only the student is trained. The teacher is frozen ({@link Module#setRequiresGrad(boolean)} with
 * {@code false} when it is attached), and its forward passes, the centering and sharpening of its outputs, the center
 * update and the teacher's EMA update run under {@link ComputationGraph#noGrad()}, so the teacher outputs are
 * constants of the loss. Unfreezing the whole trainer with {@code setRequiresGrad(true)} also unfreezes the
 * teacher.</p>
 *
 * <p>The center is a persistent module buffer named {@code center} ({@code [outputDim]}, zeros), so
 * {@link Module#stateDict()} saves it, {@link Module#loadStateDict} restores it and {@link Module#to(ComputeBackend)}
 * moves it. It is created at construction on the student's backend ({@link Module#backend()}); for a student without
 * parameters, whose backend is unknown until then, on the backend of the first teacher output. Ownership: the student
 * and teacher outputs are closed after use unless a module returns its input; the tensors of a {@link DINOOutput}
 * belong to the caller.</p>
 */
public final class DINOTrainer extends Module {

    private static final String CENTER = "center";

    /**
     * The result of {@link #forwardDINO}, owned by the caller. {@link #dinoLoss()} is the recorded loss to
     * back-propagate. {@link #studentOutput()} is an output of the same recorded graph, valid until the loss is
     * back-propagated (whose backward pass closes it); take a {@link Tensor#detach()} handle first to read it
     * afterwards. {@link #teacherOutput()} was computed without recording and stays valid.
     */
    public static final class DINOOutput {
        private final Tensor studentOutput;
        private final Tensor teacherOutput;
        private final Tensor dinoLoss;

        public DINOOutput(Tensor studentOutput, Tensor teacherOutput, Tensor dinoLoss) {
            this.studentOutput = Objects.requireNonNull(studentOutput);
            this.teacherOutput = Objects.requireNonNull(teacherOutput);
            this.dinoLoss = Objects.requireNonNull(dinoLoss);
        }

        public Tensor studentOutput() { return studentOutput; }
        public Tensor teacherOutput() { return teacherOutput; }
        public Tensor dinoLoss() { return dinoLoss; }
    }

    private final Module student;
    private Module teacher;
    private final int outputDim;
    private final float emaMomentum;
    private final float centerMomentum;
    private boolean centerRegistered;
    private final float teacherTemp;
    private final float studentTemp;
    private Random random;
    private volatile boolean closed = false;

    public DINOTrainer(Module student, int outputDim) {
        this(student, outputDim, 0.996f, 0.9f, 0.04f, 0.1f);
    }

    public DINOTrainer(Module student, int outputDim, float emaMomentum, float centerMomentum,
                        float teacherTemp, float studentTemp) {
        Objects.requireNonNull(student, "Student cannot be null");
        if (outputDim <= 0) throw new IllegalArgumentException("Output dim must be positive");
        if (emaMomentum <= 0 || emaMomentum >= 1) {
            throw new IllegalArgumentException("EMA momentum must be in (0, 1)");
        }

        this.student = registerModule("student", student);
        this.outputDim = outputDim;
        this.emaMomentum = emaMomentum;
        this.centerMomentum = centerMomentum;
        this.teacherTemp = teacherTemp;
        this.studentTemp = studentTemp;
        this.random = new Random();
        ComputeBackend studentBackend = student.backend();
        if (studentBackend != null) {
            registerCenter(studentBackend);
        }
    }

    /**
     * Creates the teacher through the student class's no-argument constructor and copies the student's parameter
     * values into it by name ({@link #initializeTeacher(Module)}).
     *
     * @throws IllegalStateException when the student class has no accessible no-argument constructor, or a teacher
     *                               is already set
     */
    public void initializeTeacher() {
        ensureOpen();
        if (teacher != null) {
            throw new IllegalStateException("Teacher already initialized");
        }
        Module copy;
        try {
            copy = student.getClass().getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Cannot instantiate a teacher: " + student.getClass().getName()
                + " has no accessible no-argument constructor; pass a teacher of the same architecture to"
                + " initializeTeacher(Module)", e);
        }
        try {
            attachTeacher(copy);
        } catch (RuntimeException e) {
            copy.close();
            throw e;
        }
    }

    /**
     * Uses {@code teacher}, a module of the student's architecture, as the teacher: every teacher parameter whose
     * name the student also has receives the student's values ({@link Parameter#copyFrom}, an assignment that records
     * nothing), the teacher is frozen, and it becomes the child module {@code teacher}.
     *
     * @throws IllegalArgumentException when {@code teacher} is the student, or a named pair differs in shape, dtype
     *                                  or backend
     * @throws IllegalStateException    when a teacher is already set
     */
    public void initializeTeacher(Module teacher) {
        ensureOpen();
        Objects.requireNonNull(teacher, "Teacher cannot be null");
        if (teacher == student) {
            throw new IllegalArgumentException("The teacher must be a separate module from the student");
        }
        attachTeacher(teacher);
    }

    private void attachTeacher(Module candidate) {
        if (teacher != null) {
            throw new IllegalStateException("Teacher already initialized");
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            copyParameters(student, candidate);
            candidate.setRequiresGrad(false);
        }
        this.teacher = registerModule("teacher", candidate);
    }

    /**
     * {@code teacher = emaMomentum * teacher + (1 - emaMomentum) * student} for every parameter pair in
     * {@link Module#parameters()} order, in place on the teacher's tensors (one axpby launch per parameter, or the
     * CPU in-place path), under {@link ComputationGraph#noGrad()}.
     *
     * @throws IllegalStateException when no teacher is set or the parameter lists differ in length or shapes
     */
    public void updateTeacher() {
        ensureOpen();
        if (teacher == null) {
            throw new IllegalStateException("Teacher not initialized. Call initializeTeacher() first.");
        }

        List<Parameter> studentParams = student.parameters();
        List<Parameter> teacherParams = teacher.parameters();

        if (studentParams.size() != teacherParams.size()) {
            throw new IllegalStateException("Student and teacher parameter count mismatch");
        }

        for (int i = 0; i < studentParams.size(); i++) {
            Shape studentShape = studentParams.get(i).shape();
            Shape teacherShape = teacherParams.get(i).shape();
            if (!studentShape.equals(teacherShape)) {
                throw new IllegalStateException("Student parameter " + i + " of shape " + studentShape
                    + " does not match the teacher's " + teacherShape);
            }
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (int i = 0; i < studentParams.size(); i++) {
                ema(teacherParams.get(i).tensor(), studentParams.get(i).tensor(), emaMomentum);
            }
        }
    }

    /**
     * {@code center = centerMomentum * center + (1 - centerMomentum) * mean_b(teacherOutput[b, :])}, in place, under
     * {@link ComputationGraph#noGrad()}.
     *
     * @throws IllegalArgumentException when the output does not hold {@code [batch, outputDim]} values
     */
    public void updateCenter(Tensor teacherOutput) {
        ensureOpen();
        Objects.requireNonNull(teacherOutput, "Teacher output cannot be null");
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor rows = rows(teacherOutput);
             Tensor batchMean = TensorOps.mean(rows, 0)) {
            ema(center(teacherOutput.backend()), batchMean, centerMomentum);
        }
    }

    /**
     * Student and teacher outputs of both global views, the cross-view loss {@code (H(s1, t2) + H(s2, t1)) / 2}, plus
     * {@code (H(s_l, t1) + H(s_l, t2)) / 2} per local view, then the center update from the first global teacher
     * output. The returned outputs are the sharpened student and the centered, sharpened teacher of the first view.
     * The student path is recorded; the teacher forwards, their centering and sharpening, and the center update run
     * under {@link ComputationGraph#noGrad()}.
     */
    public DINOOutput forwardDINO(Tensor globalView1, Tensor globalView2,
                                   Tensor... localViews) {
        ensureOpen();

        if (teacher == null) {
            throw new IllegalStateException("Teacher not initialized. Call initializeTeacher() first.");
        }
        Objects.requireNonNull(globalView1, "Global view 1 cannot be null");
        Objects.requireNonNull(globalView2, "Global view 2 cannot be null");

        Tensor[] inputs = inputsOf(globalView1, globalView2, localViews);
        List<Tensor> temporaries = new ArrayList<>();
        Tensor studentOut1 = null;
        Tensor teacherOut1 = null;
        Tensor loss = null;
        try {
            Tensor studentGlobal1 = track(temporaries, student.forward(globalView1));
            Tensor studentGlobal2 = track(temporaries, student.forward(globalView2));
            Tensor teacherGlobal1;
            Tensor teacherGlobal2;
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                teacherGlobal1 = track(temporaries, teacher.forward(globalView1));
                teacherGlobal2 = track(temporaries, teacher.forward(globalView2));
            }

            studentOut1 = sharpen(studentGlobal1, studentTemp);
            Tensor studentOut2 = track(temporaries, sharpen(studentGlobal2, studentTemp));
            Tensor teacherOut2;
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                teacherOut1 = centerAndSharpen(teacherGlobal1, teacherTemp);
                teacherOut2 = track(temporaries, centerAndSharpen(teacherGlobal2, teacherTemp));
            }

            loss = pairLoss(studentOut1, teacherOut2, studentOut2, teacherOut1);
            for (Tensor localView : localViews) {
                Tensor studentLocal = track(temporaries, student.forward(localView));
                Tensor studentLocalOut = track(temporaries, sharpen(studentLocal, studentTemp));
                try (Tensor localLoss = pairLoss(studentLocalOut, teacherOut1, studentLocalOut, teacherOut2)) {
                    Tensor total = TensorOps.add(loss, localLoss);
                    loss.close();
                    loss = total;
                }
            }

            updateCenter(teacherGlobal1);
            return new DINOOutput(studentOut1, teacherOut1, loss);
        } catch (RuntimeException e) {
            release(studentOut1, inputs);
            release(teacherOut1, inputs);
            release(loss, inputs);
            throw e;
        } finally {
            for (Tensor t : temporaries) {
                release(t, inputs);
            }
        }
    }

    /** {@code (H(s1, t1) + H(s2, t2)) / 2}; halving is the exact multiplication by 0.5. */
    private Tensor pairLoss(Tensor s1, Tensor t1, Tensor s2, Tensor t2) {
        try (Tensor loss1 = crossEntropy(s1, t1);
             Tensor loss2 = crossEntropy(s2, t2);
             Tensor sum = TensorOps.add(loss1, loss2)) {
            return TensorOps.mul(sum, 0.5f);
        }
    }

    /** Row softmax of {@code (x - max_row(x)) / temperature}, in the input's shape. */
    private Tensor sharpen(Tensor input, float temperature) {
        try (Tensor rows = rows(input)) {
            return sharpenRows(rows, temperature, input.shape());
        }
    }

    /** {@link #sharpen} of {@code x - center}, with the center before this step's update. */
    private Tensor centerAndSharpen(Tensor input, float temperature) {
        try (Tensor rows = rows(input);
             Tensor centered = TensorOps.sub(rows, center(input.backend()))) {
            return sharpenRows(centered, temperature, input.shape());
        }
    }

    /**
     * {@code softmax((x - max) / temperature)} along dim 1 of {@code [batch, outputDim]} rows. The scaled row's
     * maximum is exactly 0, so the softmax evaluates {@code exp((x - max) / T) / sum} as the CPU formula does.
     */
    private static Tensor sharpenRows(Tensor rows, float temperature, Shape outShape) {
        try (Tensor max = TensorOps.max(rows, 1);
             Tensor maxColumn = max.unsqueeze(1);
             Tensor shifted = TensorOps.sub(rows, maxColumn);
             Tensor temp = constant(rows.backend(), temperature);
             Tensor scaled = TensorOps.div(shifted, temp);
             Tensor probs = TensorOps.softmax(scaled, 1)) {
            return probs.reshape(outShape);
        }
    }

    /**
     * {@code -sum(max(1e-10, t) * log(max(1e-10, s))) / batch} over {@code [batch, outputDim]} rows, with the batch
     * of the student output.
     */
    private Tensor crossEntropy(Tensor studentOut, Tensor teacherOut) {
        int batchSize = studentOut.shape().dim(0);
        try (Tensor s = rows(studentOut);
             Tensor t = teacherOut.reshape(s.shape());
             Tensor sClamped = TensorOps.clamp(s, 1e-10f, Float.POSITIVE_INFINITY);
             Tensor tClamped = TensorOps.clamp(t, 1e-10f, Float.POSITIVE_INFINITY);
             Tensor logS = TensorOps.log(sClamped);
             Tensor terms = TensorOps.mul(tClamped, logS);
             Tensor total = TensorOps.sum(terms);
             Tensor negated = TensorOps.neg(total);
             Tensor batch = constant(studentOut.backend(), batchSize)) {
            return TensorOps.div(negated, batch);
        }
    }

    /** Registers the persistent {@code [outputDim]} center buffer as zeros on {@code backend}. */
    private void registerCenter(ComputeBackend backend) {
        registerBuffer(CENTER, Tensor.zeros(backend, Shape.vector(outputDim)));
        centerRegistered = true;
    }

    /**
     * The center buffer; for a student without parameters it is created as zeros on {@code backend} at first use.
     *
     * @throws IllegalArgumentException when the center lives on another backend than {@code backend}
     */
    private Tensor center(ComputeBackend backend) {
        if (!centerRegistered) {
            registerCenter(backend);
        }
        Tensor center = buffer(CENTER);
        if (center.backend() != backend) {
            throw new IllegalArgumentException("The DINO center lives on another backend than the teacher output;"
                + " move the trainer with to(backend)");
        }
        return center;
    }

    /** A {@code [batch, outputDim]} view of a module output. */
    private Tensor rows(Tensor output) {
        int batchSize = output.shape().dim(0);
        if (output.numElements() != (long) batchSize * outputDim) {
            throw new IllegalArgumentException("Output of shape " + output.shape() + " does not hold [" + batchSize
                + ", " + outputDim + "] values");
        }
        return output.reshape(batchSize, outputDim);
    }

    /**
     * {@code target = momentum * target + (1 - momentum) * source} in place: one axpby launch, or the CPU in-place
     * path with the same two roundings and sum. Callers run it under {@link ComputationGraph#noGrad()}.
     */
    private static void ema(Tensor target, Tensor source, float momentum) {
        float take = 1 - momentum;
        if (!GpuTensorOps.axpbyInPlace(target, take, source, momentum)) {
            TensorOps.mulInPlace(target, momentum);
            TensorOps.addInPlace(target, source, take);
        }
    }

    /**
     * Copies the student's values into every teacher parameter of the same name with {@link Parameter#copyFrom} (on
     * the device when the train kernels are available, else through the host).
     *
     * @throws IllegalArgumentException when a named pair differs in shape, dtype or backend
     */
    private static void copyParameters(Module source, Module target) {
        Map<String, Parameter> sourceParams = source.namedParameters();
        for (Map.Entry<String, Parameter> entry : target.namedParameters().entrySet()) {
            Parameter from = sourceParams.get(entry.getKey());
            if (from == null) {
                continue;
            }
            Parameter to = entry.getValue();
            if (!to.shape().equals(from.shape()) || to.dtype() != from.dtype() || to.backend() != from.backend()) {
                throw new IllegalArgumentException("Teacher parameter '" + entry.getKey() + "' (" + to.shape() + ", "
                    + to.dtype() + ") cannot take the student's values (" + from.shape() + ", " + from.dtype()
                    + ") or lives on another backend");
            }
            to.copyFrom(from.tensor());
        }
    }

    /** A rank-0 tensor holding {@code value}, filled on the device when the train kernels are available. */
    private static Tensor constant(ComputeBackend backend, float value) {
        try (Tensor zero = Tensor.zeros(backend, Shape.scalar())) {
            return TensorOps.add(zero, value);
        }
    }

    private static Tensor[] inputsOf(Tensor view1, Tensor view2, Tensor[] localViews) {
        Tensor[] inputs = new Tensor[2 + localViews.length];
        inputs[0] = view1;
        inputs[1] = view2;
        System.arraycopy(localViews, 0, inputs, 2, localViews.length);
        return inputs;
    }

    private static Tensor track(List<Tensor> temporaries, Tensor t) {
        temporaries.add(t);
        return t;
    }

    /** Closes {@code t} unless it is one of {@code keep}. */
    private static void release(Tensor t, Tensor... keep) {
        if (t == null) {
            return;
        }
        for (Tensor k : keep) {
            if (t == k) {
                return;
            }
        }
        t.close();
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        return student.forward(input);
    }

    public Module student() { return student; }
    public Module teacher() { return teacher; }
    public int outputDim() { return outputDim; }
    public float emaMomentum() { return emaMomentum; }
    public float teacherTemp() { return teacherTemp; }
    public float studentTemp() { return studentTemp; }

    public void setRandom(Random random) {
        this.random = Objects.requireNonNull(random);
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("DINOTrainer has been closed");
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            super.close();
        }
    }

    @Override
    public String toString() {
        if (closed) return "DINOTrainer{closed=true}";
        return "DINOTrainer{outputDim=" + outputDim +
               ", emaMomentum=" + emaMomentum +
               ", teacherTemp=" + teacherTemp + "}";
    }
}
