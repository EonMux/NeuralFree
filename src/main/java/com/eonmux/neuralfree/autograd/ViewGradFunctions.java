package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Backward functions of the {@link Tensor} views that {@link GradFunctions} does not cover: {@code expand},
 * {@code contiguous} and the floating-point casts of {@code to(DType)}. They save nothing and follow the ownership
 * contract of {@link GradFunctions}: {@code gradOutput} is never closed, and each returned gradient is owned by the
 * caller unless it is {@code gradOutput} itself, which the graph then takes a new reference of. Every class supports
 * create-graph ({@code supportsCreateGraph()}): its backward passes {@code gradOutput} through unchanged or composes
 * recorded ops, and never takes a detached handle.
 */
public final class ViewGradFunctions {

    private ViewGradFunctions() {
    }

    /**
     * {@code expand(src, target)}: {@code sumToShape(gy, src.shape)} (the prepended dims and the dims where the source
     * has 1 and the target more are summed); {@code gy} itself when the expand kept the shape, and zeros of the source
     * shape when the target has no elements (a size-1 dim expanded to 0). Saved: nothing. Tolerance: reduction,
     * K = the number of target elements per source element.
     */
    public static class ExpandGrad implements GradientFunction {
        private final Shape sourceShape;

        /** @param sourceShape the shape of the expanded tensor */
        public ExpandGrad(Shape sourceShape) {
            this.sourceShape = Objects.requireNonNull(sourceShape, "sourceShape");
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            if (gradOutput.shape().equals(sourceShape)) {
                return Collections.singletonList(gradOutput);
            }
            if (gradOutput.numElements() == 0) {
                return Collections.singletonList(Tensor.zeros(gradOutput.backend(), sourceShape, gradOutput.dtype()));
            }
            return Collections.singletonList(TensorOps.sumToShape(gradOutput, sourceShape));
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code to(DType)} between FLOAT32, FLOAT16 and BFLOAT16: the gradient of the source is {@code gy} as FLOAT32,
     * whatever the source dtype (gradients are FLOAT32): {@code gy} itself when it is FLOAT32, else a contiguous
     * FLOAT32 copy ({@code gy.to(FLOAT32)}, an exact widening). Saved: nothing. Tolerance: bit-identical.
     */
    public static class CastGrad implements GradientFunction {
        private final DType sourceDType;
        private final DType targetDType;

        /**
         * @param sourceDType the dtype of the cast tensor
         * @param targetDType the dtype it was cast to
         * @throws IllegalArgumentException when either dtype is not FLOAT32, FLOAT16 or BFLOAT16, or both are equal
         *                                  (a cast to the same dtype returns its operand and records nothing)
         */
        public CastGrad(DType sourceDType, DType targetDType) {
            this.sourceDType = requireFloatingCast(sourceDType, "source");
            this.targetDType = requireFloatingCast(targetDType, "target");
            if (sourceDType == targetDType) {
                throw new IllegalArgumentException("a cast from " + sourceDType
                        + " to itself has no gradient function");
            }
        }

        public DType sourceDType() {
            return sourceDType;
        }

        public DType targetDType() {
            return targetDType;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            DType dtype = gradOutput.dtype();
            if (dtype == DType.FLOAT32) {
                return Collections.singletonList(gradOutput);
            }
            if (dtype == DType.FLOAT16 || dtype == DType.BFLOAT16) {
                return Collections.singletonList(gradOutput.to(DType.FLOAT32));
            }
            throw new IllegalArgumentException("the gradient of a " + sourceDType + " to " + targetDType
                    + " cast must be FLOAT32, FLOAT16 or BFLOAT16, got " + dtype);
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }

        private static DType requireFloatingCast(DType dtype, String role) {
            Objects.requireNonNull(dtype, role + " dtype");
            if (dtype != DType.FLOAT32 && dtype != DType.FLOAT16 && dtype != DType.BFLOAT16) {
                throw new IllegalArgumentException("differentiable casts are among FLOAT32, FLOAT16 and BFLOAT16; "
                        + role + " dtype is " + dtype);
            }
            return dtype;
        }
    }

    /**
     * {@code contiguous()}: {@code gy} itself (the graph takes the new reference). Saved: nothing. Tolerance:
     * bit-identical.
     */
    public static class IdentityGrad implements GradientFunction {

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(gradOutput);
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }
}
