package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.nn.loss.LossFunction.Reduction;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Collection;
import java.util.Random;

import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.in;
import static com.eonmux.neuralfree.tensor.TensorOpsGradCheckTest.GradCheck.uniform;

/**
 * Gradient checks of the eleven losses: the gradients a recorded backward pass gives the
 * prediction and every differentiable target or embedding against central finite differences
 * ({@link GradCheck#inputs}), on the host reference backend and on every available GPU API, with the SUM, MEAN and
 * NONE reductions (and {@code batchmean} for {@link KLDivLoss}). Class indices (FLOAT32 and INT32, with ignored
 * samples), labels and other non-differentiable targets are constants. Every input lies in {@code [-1, 1]}.
 *
 * <p>Kinks: the elementwise regression losses read a prediction {@code p} and a target {@code t} whose difference
 * {@code p - t} keeps a distance of at least {@link GradCheck#KINK_MARGIN} from 0 ({@link L1Loss}) and from the switch
 * points {@code +-delta} ({@link HuberLoss}) or {@code +-beta} ({@link SmoothL1Loss}) ({@link #targetsAwayFrom});
 * probabilities stay inside {@code [0.1, 0.9)}, away from the clamp of {@link BCELoss} and the zero-target guard of
 * {@link KLDivLoss}; class-distribution targets stay at least 0.1 above their {@code t > 0} test; the margins of
 * {@link ContrastiveLoss} and {@link TripletMarginLoss} exceed every distance the inputs can produce, so their hinges
 * stay active, and the triplet differences keep a distance from 0, where the norm's absolute value has its kink.</p>
 */
@RunWith(Parameterized.class)
public class LossGradCheckTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> data() {
        return GradCheck.targets();
    }

    private final GradCheck.Target target;
    private GradCheck fx;

    public LossGradCheckTest(GradCheck.Target target) {
        this.target = target;
    }

    @Before
    public void setUp() {
        fx = GradCheck.open(target);
    }

    @After
    public void tearDown() {
        if (fx != null) {
            fx.close();
        }
    }

    @Test
    public void meanSquaredError() {
        GradCheck.Input p = in(uniform(601, 12, -1f, 1f), 3, 4);
        GradCheck.Input t = in(uniform(602, 12, -1f, 1f), 3, 4);
        fx.inputs("MSELoss mean", a -> new MSELoss().compute(a.get(0), a.get(1)), p, t);
        fx.inputs("MSELoss none", a -> new MSELoss(Reduction.NONE).compute(a.get(0), a.get(1)), p, t);
    }

    @Test
    public void absoluteErrors() {
        float[] prediction = uniform(611, 12, -1f, 1f);
        GradCheck.Input p = in(prediction, 3, 4);
        GradCheck.Input awayFromZero = in(targetsAwayFrom(612, prediction, 0f), 3, 4);
        GradCheck.Input awayFromSwitch = in(targetsAwayFrom(613, prediction, 0f, -0.5f, 0.5f), 3, 4);
        fx.inputs("L1Loss sum", a -> new L1Loss(Reduction.SUM).compute(a.get(0), a.get(1)), p, awayFromZero);
        fx.inputs("HuberLoss mean", a -> new HuberLoss(Reduction.MEAN, 0.5f).compute(a.get(0), a.get(1)), p,
                awayFromSwitch);
        fx.inputs("SmoothL1Loss sum", a -> new SmoothL1Loss(Reduction.SUM, 0.5f).compute(a.get(0), a.get(1)), p,
                awayFromSwitch);
        fx.inputs("SmoothL1Loss none", a -> new SmoothL1Loss(Reduction.NONE, 0.5f).compute(a.get(0), a.get(1)), p,
                awayFromSwitch);
    }

    @Test
    public void binaryCrossEntropy() {
        GradCheck.Input probabilities = in(uniform(621, 12, 0.1f, 0.9f), 3, 4);
        GradCheck.Input targets = in(uniform(622, 12, 0f, 1f), 3, 4);
        GradCheck.Input logits = in(uniform(623, 12, -1f, 1f), 3, 4);
        fx.inputs("BCELoss mean", a -> new BCELoss().compute(a.get(0), a.get(1)), probabilities, targets);
        fx.inputs("BCEWithLogitsLoss sum", a -> new BCEWithLogitsLoss(Reduction.SUM).compute(a.get(0), a.get(1)),
                logits, targets);
        fx.inputs("BCEWithLogitsLoss positive weight",
                a -> new BCEWithLogitsLoss(Reduction.MEAN, 2.0f).compute(a.get(0), a.get(1)), logits, targets);
    }

    /**
     * Log-probabilities in {@code [-1, -0.1)} against probabilities in {@code [0.1, 0.9)}, or against
     * log-probabilities in {@code [-1, 0)} with {@code logTarget}; both operands get a gradient.
     */
    @Test
    public void klDivergence() {
        GradCheck.Input logProbabilities = in(uniform(631, 12, -1f, -0.1f), 3, 4);
        GradCheck.Input probabilities = in(uniform(632, 12, 0.1f, 0.9f), 3, 4);
        fx.inputs("KLDivLoss batchmean", a -> new KLDivLoss().compute(a.get(0), a.get(1)), logProbabilities,
                probabilities);
        fx.inputs("KLDivLoss sum", a -> new KLDivLoss(Reduction.SUM).compute(a.get(0), a.get(1)), logProbabilities,
                probabilities);
        fx.inputs("KLDivLoss log target, none", a -> new KLDivLoss(Reduction.NONE, true).compute(a.get(0), a.get(1)),
                logProbabilities, in(uniform(633, 12, -1f, 0f), 3, 4));
    }

    @Test
    public void crossEntropy() {
        Tensor classes = fx.constant(new float[]{2, 0, 4, 1}, 4);
        fx.inputs("CrossEntropyLoss class values", a -> new CrossEntropyLoss().compute(a.get(0), classes),
                in(uniform(641, 20, -1f, 1f), 4, 5));
        Tensor channelClasses = fx.constant(new float[]{3, 0, -100, 1, -100, 2}, 2, 3);
        fx.inputs("CrossEntropyLoss class dim 1, ignored classes",
                a -> new CrossEntropyLoss(Reduction.SUM, 1, 0f).compute(a.get(0), channelClasses),
                in(uniform(642, 24, -1f, 1f), 2, 4, 3));
        Tensor intClasses = fx.int32(new int[]{2, -100, 4, 1}, 4);
        fx.inputs("CrossEntropyLoss INT32 classes, label smoothing, ignored class",
                a -> new CrossEntropyLoss(Reduction.MEAN, -1, 0.1f).compute(a.get(0), intClasses),
                in(uniform(645, 20, -1f, 1f), 4, 5));
        GradCheck.Input logits = in(uniform(643, 12, -1f, 1f), 3, 4);
        GradCheck.Input distribution = in(uniform(644, 12, 0.1f, 1f), 3, 4);
        fx.inputs("CrossEntropyLoss distribution",
                a -> new CrossEntropyLoss(Reduction.MEAN).compute(a.get(0), a.get(1)), logits, distribution);
        fx.inputs("CrossEntropyLoss label smoothing",
                a -> new CrossEntropyLoss(Reduction.NONE, 1, 0.1f).compute(a.get(0), a.get(1)), logits,
                distribution);
    }

    @Test
    public void focal() {
        Tensor classes = fx.constant(new float[]{1, 3, 0, 2}, 4);
        fx.inputs("FocalLoss class values",
                a -> new FocalLoss(0.25f, 2.0f, Reduction.SUM, -1).compute(a.get(0), classes),
                in(uniform(651, 16, -1f, 1f), 4, 4));
        Tensor intClasses = fx.int32(new int[]{1, -100, 0, 2}, 4);
        fx.inputs("FocalLoss INT32 classes, ignored class",
                a -> new FocalLoss(0.25f, 2.0f, Reduction.MEAN, -1).compute(a.get(0), intClasses),
                in(uniform(653, 16, -1f, 1f), 4, 4));
        Tensor distribution = fx.constant(new float[]{0.7f, 0.1f, 0.2f, 0.15f, 0.65f, 0.2f}, 2, 3);
        fx.inputs("FocalLoss distribution",
                a -> new FocalLoss(0.25f, 1.5f, Reduction.MEAN, 1).compute(a.get(0), distribution),
                in(uniform(652, 6, -1f, 1f), 2, 3));
    }

    /** Embeddings in {@code [-0.5, 0.5)} are at most {@code sqrt(3)} apart, below the margin 3. */
    @Test
    public void contrastive() {
        Tensor labels = fx.constant(new float[]{1, 0, 1, 0}, 4);
        fx.inputs("ContrastiveLoss", a -> new ContrastiveLoss(3.0f, Reduction.MEAN).compute(a.get(0), a.get(1),
                labels), in(uniform(661, 12, -0.5f, 0.5f), 4, 3), in(uniform(662, 12, -0.5f, 0.5f), 4, 3));
    }

    /**
     * The positive and negative lie in {@code [-1, 1)} and differ from the anchor by at least
     * {@link GradCheck#KINK_MARGIN} and less than 2 per element, so both distances over the four features lie in
     * {@code [0.1, 4)} and {@code dist(a, p) - dist(a, n) + 5} stays above 1.
     */
    @Test
    public void tripletMargin() {
        float[] anchor = uniform(671, 12, -1f, 1f);
        GradCheck.Input a = in(anchor, 3, 4);
        GradCheck.Input positive = in(targetsAwayFrom(672, anchor, 0f), 3, 4);
        GradCheck.Input negative = in(targetsAwayFrom(673, anchor, 0f), 3, 4);
        TripletMarginLoss loss = new TripletMarginLoss(5.0f, 2.0f, Reduction.SUM);
        fx.inputs("TripletMarginLoss", t -> loss.compute(t.get(0), t.get(1), t.get(2)), a, positive, negative);
        fx.inputs("TripletMarginLoss weighted",
                t -> loss.compute(t.get(0), t.get(1), t.get(2), new float[]{0.5f, 2.0f, 1.5f}), a, positive,
                negative);
    }

    /**
     * One value per element of {@code reference}, uniform in {@code [-1, 1)} from a seeded stream and redrawn until
     * the difference {@code reference[i] - value} (in float, as the losses compute it) is at least
     * {@link GradCheck#KINK_MARGIN} from every one of {@code kinks}.
     */
    private static float[] targetsAwayFrom(long seed, float[] reference, float... kinks) {
        Random r = new Random(seed);
        float[] out = new float[reference.length];
        for (int i = 0; i < out.length; i++) {
            float v;
            do {
                v = -1f + 2f * r.nextFloat();
            } while (nearKink(reference[i] - v, kinks));
            out[i] = v;
        }
        return out;
    }

    private static boolean nearKink(float difference, float[] kinks) {
        for (float k : kinks) {
            if (Math.abs(difference - k) < GradCheck.KINK_MARGIN) {
                return true;
            }
        }
        return false;
    }
}
