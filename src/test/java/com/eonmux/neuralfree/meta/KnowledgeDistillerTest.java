package com.eonmux.neuralfree.meta;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.MSELoss;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The distillation term of {@link KnowledgeDistiller} on the CPU path: Hinton's forward KL from teacher to student,
 * {@code T^2 * (1/B) * sum t * (log t - log s)} with {@code t = softmax(teacher / T)} and
 * {@code log s = log_softmax(student / T)}, as PyTorch's {@code T * T * kl_div(..., reduction='batchmean')}.
 * Each model is {@code input @ W} on an identity input, so its output is its weight and the student's weight gradient
 * is the gradient with respect to its logits; one SGD step with learning rate {@code lr} moves the weight by
 * {@code -lr * gradient}.
 */
public class KnowledgeDistillerTest {

    private static final double LN3 = Math.log(3.0);

    private ComputeBackend host;

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        host = TrainParity.hostBackend();
    }

    @After
    public void tearDown() {
        ComputationGraph.current().clear();
        TrainGpu.release(host);
    }

    /**
     * T = 2, alpha = 0.5, B = 2. Row 0: t = (1/2, 1/2), s = (1/4, 3/4); row 1: t = (1/4, 3/4), s = (1/2, 1/2). The
     * divergence sum is {@code (1/2) ln(4/3) + (1/4) ln(1/2) + (3/4) ln(3/2)}, so the distillation term is
     * {@code 4 / 2 * sum = (1/2) ln 3}. The hard target is MSE against zeros, {@code (2 ln 3)^2 / 4 = (ln 3)^2}. The
     * distillation gradient {@code (T / B) * (s - t)} is {@code [[-1/4, 1/4], [1/4, -1/4]]}, the MSE gradient
     * {@code 2 * logits / 4} is {@code [[0, ln 3], [0, 0]]}, and each enters the step with weight 1/2.
     */
    @Test
    public void handComputedLossesAndStepAtTemperatureTwo() {
        float[][] teacherLogits = {{0f, 0f}, {0f, (float) (2 * LN3)}};
        float[][] studentLogits = {{0f, (float) (2 * LN3)}, {0f, 0f}};
        float learningRate = 0.5f;

        Run run = distill(teacherLogits, studentLogits, Shape.of(2, 2), 2.0f, 0.5f, learningRate);

        assertEquals("distillation", 0.5 * LN3, run.distillation, 1e-5);
        assertEquals("hard target", LN3 * LN3, run.hard, 1e-5);
        assertEquals("total", 0.5 * (0.5 * LN3) + 0.5 * (LN3 * LN3), run.total, 1e-5);

        double[] gradient = {-0.125, 0.125 + 0.5 * LN3, 0.125, -0.125};
        double[] before = {0.0, 2 * LN3, 0.0, 0.0};
        for (int i = 0; i < gradient.length; i++) {
            assertEquals("student weight " + i, before[i] - learningRate * gradient[i], run.weightsAfter[i], 1e-5);
        }
    }

    /**
     * With alpha = 1 only the distillation term drives the student. Its gradient with respect to the student's logits
     * (read from one SGD step at learning rate 1) matches the closed form {@code (T / B) * (softmax(s / T) - t)} and
     * central finite differences of the reported distillation value.
     */
    @Test
    public void distillationGradientMatchesClosedFormAndFiniteDifferences() {
        float temperature = 2.5f;
        int rows = 3;
        int cols = 4;
        float[][] teacherLogits = matrix(TrainParity.uniform(71, rows * cols, -2f, 2f), rows, cols);
        float[][] studentLogits = matrix(TrainParity.uniform(72, rows * cols, -2f, 2f), rows, cols);
        Shape shape = Shape.of(rows, cols);

        Run run = distill(teacherLogits, studentLogits, shape, temperature, 1.0f, 1.0f);
        assertEquals("distillation value", reference(studentLogits, teacherLogits, temperature, rows),
                run.distillation, 1e-5);

        double[] expected = referenceGradient(studentLogits, teacherLogits, temperature, rows);
        double h = 1e-2;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int i = r * cols + c;
                double applied = studentLogits[r][c] - run.weightsAfter[i];
                assertEquals("closed-form gradient at " + i, expected[i], applied, 1e-5);

                float[][] plus = copy(studentLogits);
                float[][] minus = copy(studentLogits);
                plus[r][c] += (float) h;
                minus[r][c] -= (float) h;
                double lossPlus = distill(teacherLogits, plus, shape, temperature, 1.0f, 1.0f).distillation;
                double lossMinus = distill(teacherLogits, minus, shape, temperature, 1.0f, 1.0f).distillation;
                double numeric = (lossPlus - lossMinus) / ((plus[r][c] - studentLogits[r][c])
                        + (studentLogits[r][c] - minus[r][c]));
                assertEquals("finite-difference gradient at " + i, numeric, applied, 1e-3);
            }
        }
    }

    /**
     * A teacher logit 400 below the other gives, at T = 2, a probability of {@code exp(-200)}, which is 0 in float:
     * that entry contributes 0 (PyTorch's {@code xlogy}) instead of {@code 0 * log 0}, so the term is
     * {@code 4 * (1 * (log 1 - log(1/2))) = 4 ln 2}, and the student's step stays finite.
     */
    @Test
    public void zeroTeacherProbabilityContributesNothing() {
        float[][] teacherLogits = {{0f, -400f}};
        float[][] studentLogits = {{0f, 0f}};

        Run run = distill(teacherLogits, studentLogits, Shape.of(1, 2), 2.0f, 1.0f, 1.0f);

        assertEquals("distillation", 4 * Math.log(2.0), run.distillation, 1e-5);
        double[] gradient = {(2.0 / 1) * (0.5 - 1.0), (2.0 / 1) * (0.5 - 0.0)};
        for (int i = 0; i < gradient.length; i++) {
            assertEquals("student weight " + i, -gradient[i], run.weightsAfter[i], 1e-5);
        }
    }

    /** A rank-1 output is one sample: the divergence is not divided by its length. */
    @Test
    public void rankOneOutputIsOneSample() {
        float[][] teacherLogits = {{0.3f, -1.1f, 0.8f}};
        float[][] studentLogits = {{-0.4f, 0.9f, 0.2f}};
        float temperature = 3.0f;

        Run run = distill(teacherLogits, studentLogits, Shape.of(3), temperature, 1.0f, 1.0f);

        assertEquals("distillation", reference(studentLogits, teacherLogits, temperature, 1), run.distillation, 1e-5);
        double[] expected = referenceGradient(studentLogits, teacherLogits, temperature, 1);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("gradient at " + i, expected[i], studentLogits[0][i] - run.weightsAfter[i], 1e-5);
        }
    }

    /** An output of rank 3 holds {@code dim(0)} samples: the batchmean divisor is the first dimension. */
    @Test
    public void higherRankOutputDividesByTheFirstDimension() {
        float[][] teacherLogits = matrix(TrainParity.uniform(81, 12, -1f, 1f), 4, 3);
        float[][] studentLogits = matrix(TrainParity.uniform(82, 12, -1f, 1f), 4, 3);
        float temperature = 2.0f;

        Run run = distill(teacherLogits, studentLogits, Shape.of(2, 2, 3), temperature, 1.0f, 1.0f);

        assertEquals("distillation", reference(studentLogits, teacherLogits, temperature, 2), run.distillation, 1e-5);
        double[] expected = referenceGradient(studentLogits, teacherLogits, temperature, 2);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("gradient at " + i, expected[i], studentLogits[i / 3][i % 3] - run.weightsAfter[i], 1e-5);
        }
    }

    @Test
    public void mismatchedOutputShapesAreRejected() {
        Logits teacher = new Logits(host, matrix(TrainParity.uniform(91, 6, -1f, 1f), 2, 3), Shape.of(2, 3));
        Logits student = new Logits(host, matrix(TrainParity.uniform(92, 6, -1f, 1f), 2, 3), Shape.of(3, 2));
        int graphBefore = ComputationGraph.current().size();
        try (Tensor input = identity(2);
             Tensor target = Tensor.zeros(host, Shape.of(3, 2));
             KnowledgeDistiller distiller = new KnowledgeDistiller(teacher, student, 2.0f, 0.5f, new MSELoss())) {
            distiller.train(input, target, 0.1f, 1);
            fail("a student output of another shape than the teacher's is rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("teacher output shape"));
        } finally {
            teacher.close();
            student.close();
        }
        assertEquals("graph size after the failed step", graphBefore, ComputationGraph.current().size());
    }

    /** The reported losses and the student's weights after one distillation step. */
    private static final class Run {
        final float distillation;
        final float hard;
        final float total;
        final float[] weightsAfter;

        Run(KnowledgeDistiller.DistillationResult result, float[] weightsAfter) {
            this.distillation = result.finalDistillationLoss();
            this.hard = result.finalHardTargetLoss();
            this.total = result.finalTotalLoss();
            this.weightsAfter = weightsAfter;
        }
    }

    /**
     * One distillation step of a student whose output is {@code studentLogits} viewed as {@code outputShape}, against
     * a teacher whose output is {@code teacherLogits} viewed the same way, with MSE against zeros as the hard target.
     */
    private Run distill(float[][] teacherLogits, float[][] studentLogits, Shape outputShape, float temperature,
                        float alpha, float learningRate) {
        Logits teacher = new Logits(host, teacherLogits, outputShape);
        Logits student = new Logits(host, studentLogits, outputShape);
        try (Tensor input = identity(studentLogits.length);
             Tensor target = Tensor.zeros(host, outputShape);
             KnowledgeDistiller distiller = new KnowledgeDistiller(teacher, student, temperature, alpha,
                     new MSELoss())) {
            int graphBefore = ComputationGraph.current().size();
            KnowledgeDistiller.DistillationResult result = distiller.train(input, target, learningRate, 1);
            assertEquals("graph size after the step", graphBefore, ComputationGraph.current().size());
            return new Run(result, student.weights());
        } finally {
            teacher.close();
            student.close();
        }
    }

    /** {@code T^2 / batch * sum t * (log t - log s)} over rows of the logits, in double; 0 where {@code t == 0}. */
    private static double reference(float[][] student, float[][] teacher, double temperature, int batch) {
        double sum = 0.0;
        for (int r = 0; r < student.length; r++) {
            double[] t = softmax(teacher[r], temperature);
            double[] logS = logSoftmax(student[r], temperature);
            for (int c = 0; c < t.length; c++) {
                if (t[c] != 0.0) {
                    sum += t[c] * (Math.log(t[c]) - logS[c]);
                }
            }
        }
        return temperature * temperature * sum / batch;
    }

    /** {@code (T / batch) * (softmax(student / T) - softmax(teacher / T))} in double, row-major. */
    private static double[] referenceGradient(float[][] student, float[][] teacher, double temperature, int batch) {
        int cols = student[0].length;
        double[] out = new double[student.length * cols];
        for (int r = 0; r < student.length; r++) {
            double[] t = softmax(teacher[r], temperature);
            double[] s = softmax(student[r], temperature);
            for (int c = 0; c < cols; c++) {
                out[r * cols + c] = temperature / batch * (s[c] - t[c]);
            }
        }
        return out;
    }

    private static double[] softmax(float[] logits, double temperature) {
        double[] logP = logSoftmax(logits, temperature);
        double[] p = new double[logP.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp(logP[i]);
        }
        return p;
    }

    private static double[] logSoftmax(float[] logits, double temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v / temperature);
        }
        double sum = 0.0;
        for (float v : logits) {
            sum += Math.exp(v / temperature - max);
        }
        double logSum = Math.log(sum) + max;
        double[] out = new double[logits.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = logits[i] / temperature - logSum;
        }
        return out;
    }

    private Tensor identity(int n) {
        float[][] m = new float[n][n];
        for (int i = 0; i < n; i++) {
            m[i][i] = 1f;
        }
        return Tensor.fromArray(host, m);
    }

    private static float[][] matrix(float[] values, int rows, int cols) {
        float[][] m = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(values, r * cols, m[r], 0, cols);
        }
        return m;
    }

    private static float[][] copy(float[][] m) {
        float[][] out = new float[m.length][];
        for (int r = 0; r < m.length; r++) {
            out[r] = m[r].clone();
        }
        return out;
    }

    /**
     * {@code input @ W} recorded by TensorOps, viewed as {@code outputShape}; with an identity input the output holds
     * the values of the parameter {@code weight}.
     */
    private static final class Logits extends Module {
        private final Parameter weight;
        private final Shape outputShape;

        Logits(ComputeBackend b, float[][] values, Shape outputShape) {
            this.weight = registerParameter("weight", new Parameter(Tensor.fromArray(b, values), "weight"));
            this.outputShape = outputShape;
        }

        float[] weights() {
            return weight.tensor().toFloatArray();
        }

        @Override
        public Tensor forward(Tensor input) {
            Tensor product = TensorOps.matmul(input, weight.tensor());
            if (product.shape().equals(outputShape)) {
                return product;
            }
            try (Tensor owned = product) {
                return owned.reshape(outputShape);
            }
        }
    }
}
