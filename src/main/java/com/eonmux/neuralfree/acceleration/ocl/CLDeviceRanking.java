package com.eonmux.neuralfree.acceleration.ocl;

import com.eonmux.neuralfree.datatype.structures.DataMap;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.IntToDoubleFunction;

import static org.jocl.CL.*;

/**
 * Chooses which OpenCL device to run on when the host offers more than one.
 *
 * <h2>Why this exists</h2>
 * <p>Device enumeration order is the order the ICD loader happens to list
 * platforms in, and it has nothing to do with speed. A laptop with an integrated
 * GPU on platform 0 and a discrete card on platform 2 hands back the integrated
 * one first, so taking element 0 quietly runs every kernel on the slowest device
 * present. {@link com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend}
 * has always preferred {@code VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU} for exactly
 * this reason; this is the same policy for OpenCL.</p>
 *
 * <h2>Why the obvious ranking does not work</h2>
 * <p>The published attributes cannot be compared across vendors. Measured on an
 * Arrow Lake laptop with an RTX 5080:</p>
 * <pre>
 *   Intel(R) Graphics               CU=64  clock=1900MHz  global=120072MB    244 GFLOP/s
 *   NVIDIA GeForce RTX 5080 Laptop  CU=60  clock=1500MHz  global= 15880MB  30048 GFLOP/s
 * </pre>
 * <p>The integrated part wins on <em>every single published number</em> and
 * loses the actual work by a factor of 123. {@code CL_DEVICE_MAX_COMPUTE_UNITS}
 * counts Xe vector engines on Intel and streaming multiprocessors on NVIDIA —
 * units that differ by a factor of sixteen in lane width — and {@code
 * CL_DEVICE_GLOBAL_MEM_SIZE} on an integrated GPU reports system RAM. Any
 * ranking built on those numbers alone picks the wrong device here, which is why
 * the ordering below leads with a class test and settles ties by running a
 * kernel rather than by reading a spec sheet.</p>
 *
 * <h2>The order</h2>
 * <ol>
 *   <li><b>An explicit choice wins.</b> {@code -Dneuralfree.opencl.device=<i>substring</i>}
 *       selects the first device whose {@code CL_DEVICE_NAME} contains that text,
 *       case-insensitively. Nothing else is consulted.</li>
 *   <li><b>Discrete beats integrated.</b> If any device reports {@code
 *       CL_DEVICE_HOST_UNIFIED_MEMORY = false}, only those compete. A device that
 *       would not answer the query is treated as unknown and stays in the running
 *       rather than being discarded on a missing attribute.</li>
 *   <li><b>Measured throughput.</b> With more than one candidate left, each runs
 *       the same dependent-FMA kernel and the fastest wins. This is the only step
 *       that compares devices on what they actually do. Skip it with {@code
 *       -Dneuralfree.opencl.benchmark.disable=true}.</li>
 *   <li><b>Published attributes, last.</b> Only if the probe is disabled or does
 *       not complete on every candidate. See {@link #lanesPerComputeUnit}.</li>
 * </ol>
 *
 * <p>Step 3 costs nothing in the common case: one device, or one discrete device
 * among integrated ones, is decided by step 2 and no kernel is ever compiled. It
 * cost 390 ms across both devices on the machine above, and it only runs when
 * the class test leaves a genuine tie.</p>
 *
 * <h2>What the probe deliberately does not measure</h2>
 * <p>Compute throughput with buffers already resident — not transfer. That is
 * the right metric for the work this library exists to do, where weights live on
 * the device across many dispatches, but it overstates the margin for a caller
 * that uploads a large buffer, runs one cheap pass over it and reads it back:
 * an integrated GPU pays nothing to move data and a discrete one pays PCIe both
 * ways. On the machine above the 123x FLOP ratio narrowed to 1.7x for exactly
 * such a workload (1.49M x 16 projected once: 103 ms on the discrete card
 * against 168 ms on the integrated one, where the same pass costs 12.2 s on the
 * CPU). Same winner, and both devices still beat the CPU by 73-118x — but a
 * ranking is an ordering, not a promise about any particular kernel.</p>
 */
public final class CLDeviceRanking {

    private static final Logger LOGGER = LoggingUtil.getLogger(CLDeviceRanking.class);

    /** Forces a device by {@code CL_DEVICE_NAME} substring, case-insensitive. */
    public static final String DEVICE_PROPERTY = "neuralfree.opencl.device";

    /** Set true to rank on published attributes only and never dispatch a kernel. */
    public static final String NO_BENCHMARK_PROPERTY = "neuralfree.opencl.benchmark.disable";

    private CLDeviceRanking() {}

    /**
     * What the ranking knows about one candidate, separated from {@link CLDevice}
     * so the policy can be exercised without a GPU attached.
     */
    static final class Profile {
        final String name;
        final String vendor;
        /** {@code null} when the ICD declined the query — unknown, not integrated. */
        final Boolean integrated;
        final int computeUnits;
        final int clockMhz;

        Profile(String name, String vendor, Boolean integrated, int computeUnits, int clockMhz) {
            this.name = name == null ? "<unnamed>" : name;
            this.vendor = vendor == null ? "" : vendor;
            this.integrated = integrated;
            this.computeUnits = Math.max(1, computeUnits);
            this.clockMhz = Math.max(1, clockMhz);
        }
    }

    /**
     * The device to run on, or {@code null} when the list is empty.
     *
     * <p>Never throws. A device that cannot be scored for any reason ranks below
     * every device that could be, because "we failed to measure it" is not
     * evidence that it is fast.</p>
     */
    public static CLDevice fastest(List<CLDevice> devices) {
        if (devices == null || devices.isEmpty()) {
            return null;
        }
        if (devices.size() == 1) {
            return devices.get(0);
        }

        List<Profile> profiles = new ArrayList<>(devices.size());
        for (CLDevice device : devices) {
            profiles.add(profileOf(device));
        }

        int winner = rank(profiles, index -> probe(devices.get(index)));
        return winner < 0 ? devices.get(0) : devices.get(winner);
    }

    /**
     * Index of the winning profile, or {@code -1} if there is nothing to choose
     * between.
     *
     * @param probe measures profile {@code i} in GFLOP/s, or returns {@code NaN}
     *              when it could not; called at most once per candidate and never
     *              at all when the earlier steps already decide
     */
    static int rank(List<Profile> profiles, IntToDoubleFunction probe) {
        if (profiles == null || profiles.isEmpty()) {
            return -1;
        }

        int forced = forcedChoice(profiles);
        if (forced >= 0) {
            return forced;
        }

        List<Integer> candidates = discreteOnly(profiles);
        if (candidates.size() == 1) {
            int only = candidates.get(0);
            LOGGER.info("Selected OpenCL device '{}': the only discrete GPU of {} available",
                    profiles.get(only).name, profiles.size());
            return only;
        }

        int measured = fastestMeasured(profiles, candidates, probe);
        if (measured >= 0) {
            return measured;
        }

        int best = candidates.get(0);
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int index : candidates) {
            double score = attributeScore(profiles.get(index));
            if (score > bestScore) {
                bestScore = score;
                best = index;
            }
        }
        LOGGER.info("Selected OpenCL device '{}' on published attributes "
                + "(no timed comparison was available)", profiles.get(best).name);
        return best;
    }

    // ---------------------------------------------------------------- step 1

    private static int forcedChoice(List<Profile> profiles) {
        String wanted = System.getProperty(DEVICE_PROPERTY);
        if (wanted == null || wanted.trim().isEmpty()) {
            return -1;
        }
        String needle = wanted.trim().toLowerCase(Locale.ROOT);
        for (int i = 0; i < profiles.size(); i++) {
            if (profiles.get(i).name.toLowerCase(Locale.ROOT).contains(needle)) {
                LOGGER.info("Selected OpenCL device '{}': matched -D{}={}",
                        profiles.get(i).name, DEVICE_PROPERTY, wanted);
                return i;
            }
        }
        // Falling through to the ranking would silently run on a device the
        // caller explicitly did not ask for, so say so rather than pretend.
        LOGGER.warn("-D{}={} matched none of {} OpenCL device(s); ranking normally",
                DEVICE_PROPERTY, wanted, profiles.size());
        return -1;
    }

    // ---------------------------------------------------------------- step 2

    /**
     * Indices of the discrete devices, or of every device when none is known to
     * be discrete.
     *
     * <p>Absent attribute means unknown, not integrated. Dropping a device
     * because its ICD declined a deprecated query would be a worse failure than
     * ranking it on its merits.</p>
     */
    private static List<Integer> discreteOnly(List<Profile> profiles) {
        List<Integer> discrete = new ArrayList<>();
        for (int i = 0; i < profiles.size(); i++) {
            Boolean integrated = profiles.get(i).integrated;
            if (integrated != null && !integrated) {
                discrete.add(i);
            }
        }
        if (!discrete.isEmpty()) {
            return discrete;
        }
        List<Integer> all = new ArrayList<>(profiles.size());
        for (int i = 0; i < profiles.size(); i++) {
            all.add(i);
        }
        return all;
    }

    // ---------------------------------------------------------------- step 3

    /** Index of the fastest by measurement, or {@code -1} if any went unmeasured. */
    private static int fastestMeasured(List<Profile> profiles, List<Integer> candidates,
                                       IntToDoubleFunction probe) {
        if (probe == null || Boolean.getBoolean(NO_BENCHMARK_PROPERTY)) {
            return -1;
        }

        int best = -1;
        double bestGflops = Double.NEGATIVE_INFINITY;
        for (int index : candidates) {
            double gflops;
            try {
                gflops = probe.applyAsDouble(index);
            } catch (Throwable t) {
                gflops = Double.NaN;
            }
            if (Double.isNaN(gflops) || Double.isInfinite(gflops) || gflops <= 0) {
                // One unmeasured candidate makes the whole comparison unsound:
                // the rest would be ranked against a device with no number.
                LOGGER.info("OpenCL device '{}' could not be timed; falling back to "
                        + "published attributes for all candidates", profiles.get(index).name);
                return -1;
            }
            LOGGER.info("OpenCL device '{}' measured {} GFLOP/s",
                    profiles.get(index).name, String.format(Locale.ROOT, "%.1f", gflops));
            if (gflops > bestGflops) {
                bestGflops = gflops;
                best = index;
            }
        }
        if (best >= 0) {
            LOGGER.info("Selected OpenCL device '{}': fastest of {} candidates at {} GFLOP/s",
                    profiles.get(best).name, candidates.size(),
                    String.format(Locale.ROOT, "%.1f", bestGflops));
        }
        return best;
    }

    // ---------------------------------------------------------------- step 4

    /**
     * A crude FLOP-rate estimate: lanes x clock.
     *
     * <p>Reached only when no timed comparison was possible, and it is a guess —
     * {@link #lanesPerComputeUnit} is a hand-maintained table that will be wrong
     * about hardware newer than it is. It exists so that a host which cannot run
     * the probe still gets a better answer than enumeration order, not because
     * the number means anything on its own.</p>
     */
    private static double attributeScore(Profile profile) {
        return (double) profile.computeUnits
                * lanesPerComputeUnit(profile.vendor, profile.name)
                * profile.clockMhz;
    }

    /**
     * How many fp32 lanes one {@code CL_DEVICE_MAX_COMPUTE_UNITS} is worth.
     *
     * <p>Without this the counts are not comparable: NVIDIA reports streaming
     * multiprocessors (128 lanes each on Ampere and later), AMD reports compute
     * units (64), Intel reports Xe vector engines (8). The unknown-vendor value
     * is deliberately mid-range so an unrecognised device is neither
     * automatically preferred nor automatically dismissed.</p>
     */
    private static int lanesPerComputeUnit(String vendor, String deviceName) {
        String haystack = (vendor + " " + deviceName).toLowerCase(Locale.ROOT);
        if (haystack.contains("nvidia")) return 128;
        if (haystack.contains("advanced micro devices") || haystack.contains("amd")) return 64;
        if (haystack.contains("intel")) return 8;
        if (haystack.contains("apple")) return 128;
        return 32;
    }

    // ------------------------------------------------------------ the device

    private static Profile profileOf(CLDevice device) {
        DataMap<String, Object> info = device == null ? null : device.getDeviceInfo();
        if (info == null) {
            return new Profile(null, null, null, 1, 1);
        }
        Object unified = info.get("CL_DEVICE_HOST_UNIFIED_MEMORY");
        return new Profile(
                asString(info.get("CL_DEVICE_NAME")),
                asString(info.get("CL_DEVICE_VENDOR")),
                unified instanceof Boolean ? (Boolean) unified : null,
                (int) asDouble(info.get("CL_DEVICE_MAX_COMPUTE_UNITS"), 1),
                (int) asDouble(info.get("CL_DEVICE_MAX_CLOCK_FREQUENCY"), 1));
    }

    // ------------------------------------------------------------- the probe

    /**
     * One dependent FMA chain per work item. The chain is dependent on purpose:
     * each step needs the previous result, so the device cannot hide the latency
     * by widening, and {@code iters} arrives as an argument so the loop cannot be
     * unrolled away at build time. Float FMA is neither associative nor exact, so
     * no conforming compiler closes the loop into a constant.
     */
    private static final String PROBE_SOURCE =
            "__kernel void nf_rank_probe(__global const float* in,\n"
            + "                           __global float* out,\n"
            + "                           const int iters) {\n"
            + "    int i = get_global_id(0);\n"
            + "    float v = in[i];\n"
            + "    for (int k = 0; k < iters; k++) {\n"
            + "        v = fma(v, 1.0000001f, 1.0e-7f);\n"
            + "    }\n"
            + "    out[i] = v;\n"
            + "}\n";

    /** 2M work items x 1024 iterations = 4.3 GFLOP: 0.14 ms on an RTX 5080 and
     *  17.6 ms on an Arrow Lake iGPU, both far enough above launch overhead
     *  (~0.1 ms) that the ratio between them is signal rather than noise. */
    private static final int PROBE_ITEMS = 1 << 21;
    private static final int PROBE_ITERS = 1024;

    /** Best-of, not mean: scheduling noise only ever adds time, never removes it. */
    private static final int PROBE_REPEATS = 3;

    /**
     * Sustained FMA throughput in GFLOP/s, or {@code NaN} if anything went wrong.
     *
     * <p>Compilation and the host-to-device copy sit outside the timed region: a
     * device is not slow because its compiler is, and the buffer is uploaded once
     * and reused. Only the dispatch is timed.</p>
     */
    static double probe(CLDevice device) {
        if (device == null) {
            return Double.NaN;
        }
        cl_context context = device.getDeviceContext();
        cl_command_queue queue = device.getDeviceCommandQueue();
        if (context == null || queue == null) {
            return Double.NaN;
        }

        long bytes = (long) PROBE_ITEMS * Sizeof.cl_float;
        float[] host = new float[PROBE_ITEMS];
        for (int i = 0; i < PROBE_ITEMS; i++) {
            host[i] = 1.0f + (i & 0xFF) * 1.0e-3f;
        }

        cl_program program = null;
        cl_kernel kernel = null;
        cl_mem in = null;
        cl_mem out = null;
        try {
            program = clCreateProgramWithSource(context, 1, new String[]{PROBE_SOURCE}, null, null);
            clBuildProgram(program, 0, null, null, null, null);
            kernel = clCreateKernel(program, "nf_rank_probe", null);

            in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    bytes, Pointer.to(host), null);
            out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, null, null);

            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(in));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(out));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{PROBE_ITERS}));

            long[] globalWorkSize = {PROBE_ITEMS};

            // Warm-up. The first dispatch pays for driver-side kernel setup and
            // for paging the buffers in, neither of which is throughput.
            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, null, 0, null, null);
            clFinish(queue);

            long bestNanos = Long.MAX_VALUE;
            for (int r = 0; r < PROBE_REPEATS; r++) {
                long start = System.nanoTime();
                clEnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, null, 0, null, null);
                clFinish(queue);
                bestNanos = Math.min(bestNanos, System.nanoTime() - start);
            }
            if (bestNanos <= 0) {
                return Double.NaN;
            }

            // fma is two flops; nanoseconds cancel the 1e9 in GFLOP/s.
            return 2.0 * PROBE_ITEMS * PROBE_ITERS / bestNanos;
        } catch (Throwable t) {
            // Throwable: a broken ICD surfaces as CLException here but as
            // UnsatisfiedLinkError one frame down, and neither is worth losing a
            // device over when there is a fallback ranking.
            LOGGER.debug("OpenCL probe failed on '{}': {}", nameOf(device), t.toString());
            return Double.NaN;
        } finally {
            release(out);
            release(in);
            if (kernel != null) {
                try { clReleaseKernel(kernel); } catch (Throwable ignored) { }
            }
            if (program != null) {
                try { clReleaseProgram(program); } catch (Throwable ignored) { }
            }
        }
    }

    private static void release(cl_mem mem) {
        if (mem != null) {
            try { clReleaseMemObject(mem); } catch (Throwable ignored) { }
        }
    }

    // ---------------------------------------------------------------- shared

    /** {@code CL_DEVICE_NAME}, or a placeholder — this only ever feeds log lines. */
    static String nameOf(CLDevice device) {
        if (device == null || device.getDeviceInfo() == null) {
            return "<none>";
        }
        return asString(device.getDeviceInfo().get("CL_DEVICE_NAME"));
    }

    private static String asString(Object value) {
        return value == null ? "<unnamed>" : String.valueOf(value);
    }

    private static double asDouble(Object value, double fallback) {
        return value instanceof Number ? ((Number) value).doubleValue() : fallback;
    }
}
