package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuAttnKernels;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Runs one attention call twice in the JVM the tests launch, so the kernel lookup is observed on a JVM whose
 * properties the test chooses and whose engine no earlier test resolved. Output lines are {@code call1=} and
 * {@code call2=}: {@code ok}, or the simple class name and message of what the call threw. When both calls succeed,
 * {@code kernels=} gives the {@link CpuAttnKernels#describe} of the kernels the core selected, and {@code blocks=} and
 * {@code single=} the {@link #bitsHash} of {@link GqaCase}'s query-block and single-row outputs.
 */
public final class AttentionKernelProbe {

    private AttentionKernelProbe() {
    }

    public static void main(String[] args) {
        boolean ok = true;
        for (int call = 1; call <= 2; call++) {
            String result = attendOnce();
            ok &= result.equals("ok");
            System.out.println("call" + call + "=" + result);
        }
        if (ok) {
            GqaCase c = new GqaCase();
            System.out.println("kernels=" + CpuSimd.attention().describe());
            System.out.println("blocks=" + bitsHash(c.queryBlocks()));
            System.out.println("single=" + bitsHash(c.singleRows()));
        }
    }

    /** A hash of the raw bits of {@code f}, equal for two arrays exactly when they are (almost surely) bit-identical. */
    static int bitsHash(float[] f) {
        int[] bits = new int[f.length];
        for (int i = 0; i < f.length; i++) {
            bits[i] = Float.floatToRawIntBits(f[i]);
        }
        return Arrays.hashCode(bits);
    }

    /**
     * Grouped-query attention of 24 query rows (the last 24 of 200 causal keys) with 12 query heads on 4 KV heads,
     * {@code dK} 64 and {@code dV} 48, from fixed random inputs.
     */
    static final class GqaCase {
        final int nHead = 12;
        final int nHeadKv = 4;
        final int dK = 64;
        final int dV = 48;
        final int keys = 200;
        final int nt = 24;
        final int first = keys - nt;
        final float scale = 0.125f;
        final float[] q;
        final float[] k;
        final float[] v;
        final int[] qPos = new int[nt];
        private final CpuAttentionCore.DenseKv kv;
        private final CpuAttentionCore.Params p;

        GqaCase() {
            Random rnd = new Random(5);
            q = floats(rnd, nt * nHead * dK);
            k = floats(rnd, keys * nHeadKv * dK);
            v = floats(rnd, keys * nHeadKv * dV);
            int[] pos = new int[keys];
            for (int r = 0; r < keys; r++) {
                pos[r] = r;
            }
            for (int t = 0; t < nt; t++) {
                qPos[t] = first + t;
            }
            kv = CpuAttentionCore.DenseKv.of(k, 0, nHeadKv * dK, dK, v, 0, nHeadKv * dV, dV, nHeadKv, dK, dV, pos, 0);
            p = CpuAttentionCore.Params.builder(nHead).scale(scale).build();
        }

        private static float[] floats(Random rnd, int n) {
            float[] f = new float[n];
            for (int i = 0; i < n; i++) {
                f[i] = rnd.nextFloat() * 4f - 2f;
            }
            return f;
        }

        /** The rows attended as query blocks ({@link CpuAttentionCore#attendRows}). */
        float[] queryBlocks() {
            float[] rows = new float[nt * nHead * dV];
            CpuAttentionCore.attendRows(p, nt, q, 0, nHead * dK, qPos, 0, kv, 0, keys, rows, 0, nHead * dV);
            return rows;
        }

        /** The rows attended one at a time ({@link CpuAttentionCore#attend}). */
        float[] singleRows() {
            float[] single = new float[nt * nHead * dV];
            CpuAttentionCore.Scratch scratch = new CpuAttentionCore.Scratch();
            for (int t = 0; t < nt; t++) {
                CpuAttentionCore.attend(p, q, t * nHead * dK, qPos[t], kv, 0, keys, single, t * nHead * dV, scratch);
            }
            return single;
        }
    }

    private static String attendOnce() {
        final int dK = 8;
        float[] k = new float[2 * dK];
        float[] v = new float[2 * dK];
        float[] q = new float[dK];
        float[] out = new float[dK];
        try {
            CpuAttentionCore.DenseKv kv = CpuAttentionCore.DenseKv.of(k, 0, dK, dK, v, 0, dK, dK, 1, dK, dK,
                    new int[]{0, 1}, 0);
            CpuAttentionCore.attend(CpuAttentionCore.Params.builder(1).build(), q, 0, 1, kv, 0, 2, out, 0,
                    new CpuAttentionCore.Scratch());
            return "ok";
        } catch (Throwable t) {
            return t.getClass().getSimpleName() + ":" + t.getMessage();
        }
    }

    /** Runs the probe in a fresh JVM with the incubator module and {@code jvmArgs}, and returns its output. */
    static String run(String... jvmArgs) throws Exception {
        List<String> cmd = new ArrayList<>();
        cmd.add(Paths.get(System.getProperty("java.home"), "bin", "java").toString());
        if (Runtime.version().feature() >= 17) {
            cmd.add("--add-modules=jdk.incubator.vector");
        }
        cmd.addAll(Arrays.asList(jvmArgs));
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(AttentionKernelProbe.class.getName());
        ProcessBuilder pb = new ProcessBuilder(cmd).redirectErrorStream(true);
        pb.environment().remove("JAVA_TOOL_OPTIONS");
        pb.environment().remove("_JAVA_OPTIONS");
        pb.environment().remove("JDK_JAVA_OPTIONS");
        Process p = pb.start();
        StringBuilder out = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(),
                StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                out.append(line).append('\n');
            }
        }
        assertTrue("probe timed out: " + out, p.waitFor(60, TimeUnit.SECONDS));
        assertEquals("probe exit: " + out, 0, p.exitValue());
        return out.toString();
    }
}
