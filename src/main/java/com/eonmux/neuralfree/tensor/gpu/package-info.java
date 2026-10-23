/**
 * The train kernel library runtime: {@link com.eonmux.neuralfree.tensor.gpu.TrainGpu} (kernel lookup, launches,
 * ordering against host transfers, deferred index errors), {@link com.eonmux.neuralfree.tensor.gpu.TrainOps}
 * (operation ids shared with the kernel sources) and {@link com.eonmux.neuralfree.tensor.gpu.StridedView} (strided
 * operand descriptors). The tensor dispatchers ({@code tensor/Gpu*Ops}) and the classical-ML runtime
 * ({@code ml/gpu}) run on it; each falls back to its CPU reference when the library is unavailable on a backend.
 *
 * <h2>Numerical contract</h2>
 *
 * <p>Every device op uses the formula and constants of its CPU reference. Sequential loops run in ascending index
 * order; work-group reductions use a fixed 256-wide tree; every choice between kernel variants or chunk counts depends
 * only on the shapes, so a result never depends on the device. Multiply-add accumulations use an explicit
 * {@code fma}; other floating-point contraction is disabled. Ties and NaN follow the CPU code: the Java
 * {@code Math.max}/{@code Math.min} rules for max/min, first index on ties for arg-reductions and max-pool.</p>
 *
 * <p>Vulkan without {@code VK_KHR_shader_float_controls} and OpenCL 1.2 allow inexact division and square root,
 * looser transcendentals and flushed denormals, so device results agree with the CPU reference within one of these
 * <b>tolerance classes</b>, which each dispatcher names and the parity tests assert:</p>
 * <ul>
 *   <li><b>Bit-identical:</b> data movement (fill, copy, strided copy, every cast, gather, scatter overwrite,
 *       index-select, where), neg, abs, round, floor, the Java max/min family, compare, arg-reductions, max-pool and
 *       its backward.</li>
 *   <li><b>Single-rounding</b> (add, sub, mul, axpby, scale, mask-scale, scalar add and multiply): bit-identical when
 *       the operands and the result are normal; a value below {@code FLT_MIN} may flush to a signed zero.</li>
 *   <li><b>Division</b> (div, sqrt and ops built from one of them: average pools, mean, segment mean):
 *       {@code |gpu - ref| <= 4 ulp(ref)} or {@code <= FLT_MIN}.</li>
 *   <li><b>Transcendental</b> (exp, log, tanh, sigmoid, gelu, silu, elu, softplus, mish, pow, their backwards,
 *       softmax, log-softmax): {@code |gpu - ref| <= 1e-5 * max(1, |ref|)}; NaN and infinities match exactly.</li>
 *   <li><b>Reduction</b> (reductions, GEMM, convolutions, norms, losses, optimizer steps):
 *       {@code |gpu - ref64| <= 1e-5 * sqrt(K) * sum|terms| + K * FLT_MIN} against a double-precision reference,
 *       with {@code K} the reduction length and {@code sum|terms|} the sum of the absolute terms. The GEMM class is
 *       this class with the products as terms.</li>
 * </ul>
 * <p>A composition inherits the class of its last operation; a composition of several reductions uses the reduction
 * class with {@code K} the longest reduction.</p>
 *
 * <h2>Host readbacks and host paths</h2>
 *
 * <p>Tensor math stays on the tensors' backend. Data crosses to the host only for these reasons, which the classes
 * that do it state:</p>
 * <ul>
 *   <li><b>Reproducible random streams:</b> masks and noise are drawn from a seeded {@link java.util.Random} on the
 *       host, so every backend sees the same sequence, and uploaded once.</li>
 *   <li><b>Java control decisions:</b> a value Java control flow consumes (a logged loss or metric, an early-stopping
 *       decision, a sampled action or token, routing indices that select which modules run, a convergence test) is
 *       read back as the smallest tensor that carries it. Each such read is a synchronization point.</li>
 *   <li><b>Cross-process and serialization I/O:</b> state dicts, checkpoints, model export, collectives, and moving a
 *       tensor between two backends.</li>
 *   <li><b>Sequential factorizations:</b> Cholesky, triangular solves and Jacobi iterations; the matrix products that
 *       feed them run on the device.</li>
 *   <li><b>Algorithms without a device formulation:</b> tree ensembles, hidden Markov models and the Gaussian
 *       splatting rasterizer run on the CPU.</li>
 *   <li><b>Graph structure metadata:</b> CSR offsets and degrees are built once per graph from its integer edge index
 *       and cached as device tensors.</li>
 *   <li><b>FLOAT64</b> conversions, since the library does not require fp64 device support.</li>
 * </ul>
 * <p>The kernel ABI addresses at most {@code 2^31-1} elements per tensor, as a Java array does; larger tensors are
 * rejected with {@link java.lang.IllegalArgumentException}.</p>
 */
package com.eonmux.neuralfree.tensor.gpu;
