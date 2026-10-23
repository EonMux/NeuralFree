package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.autograd.CheckpointFunction;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorStorage;

import java.util.Objects;
import java.util.Random;

public final class Dropout extends Module {
    private final float dropRate;
    private Random random;
    private Tensor lastMask;

    public Dropout() {
        this(0.5f);
    }

    public Dropout(float dropRate) {
        if (!(dropRate >= 0.0f && dropRate < 1.0f)) {
            throw new IllegalArgumentException("dropRate must be in [0.0, 1.0): " + dropRate);
        }
        this.dropRate = dropRate;
        this.random = new Random();
    }

    public void setSeed(long seed) {
        this.random = new Random(seed);
    }

    /**
     * In training mode, zeroes each element with probability {@code dropRate} and scales the kept elements by
     * {@code 1 / keepProb}; in evaluation mode, or with a zero drop rate, returns {@code input} itself.
     *
     * <p>The keep decisions are drawn from this module's {@link Random} on the host, one {@code nextFloat()} per
     * element in row-major order, so a seeded module reproduces the same mask on every backend.
     * The mask holds {@code 1 / keepProb} for kept and 0 for dropped elements and is uploaded once per forward. The
     * draw goes through {@link CheckpointFunction#replayable}, so the recompute of a checkpointed segment reuses the
     * mask of its forward pass instead of drawing a new one. The output is
     * {@link TensorOps#maskScale TensorOps.maskScale(input, mask, 1)}, which in grad mode records {@code MaskScaleGrad}
     * ({@code gx = maskScale(gy, mask, 1)}); that node saves its own handle of the mask.</p>
     */
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        
        if (!isTraining() || dropRate == 0.0f) {
            return input;
        }
        
        Tensor mask = CheckpointFunction.replayable(() -> drawMask(input));
        Tensor previous = lastMask;
        lastMask = mask;
        if (previous != null) {
            previous.close();
        }
        return TensorOps.maskScale(input, mask, 1.0f);
    }

    /** A new {@code input}-shaped FLOAT32 mask on {@code input}'s backend: {@code 1 / keepProb} kept, 0 dropped. */
    private Tensor drawMask(Tensor input) {
        float keepProb = 1.0f - dropRate;
        float scale = 1.0f / keepProb;
        float[] maskData = new float[Math.toIntExact(input.numElements())];
        for (int i = 0; i < maskData.length; i++) {
            maskData[i] = random.nextFloat() < keepProb ? scale : 0.0f;
        }
        return Tensor.fromStorage(input.backend(), input.shape(), DType.FLOAT32,
            TensorStorage.fromHostFloat(input.backend(), maskData));
    }

    /**
     * The mask of the latest training-mode forward (owned by this module, not requiring grad), or null before the
     * first one. It stays valid until the next training-mode forward or {@link #close()}, also across a backward.
     */
    public Tensor getLastMask() {
        return lastMask;
    }

    public float dropRate() {
        return dropRate;
    }

    public float keepProb() {
        return 1.0f - dropRate;
    }

    @Override
    public void close() {
        if (lastMask != null) {
            lastMask.close();
            lastMask = null;
        }
        super.close();
    }

    @Override
    public String toString() {
        return "Dropout{" +
            "dropRate=" + dropRate +
            ", training=" + isTraining() +
            '}';
    }
}
