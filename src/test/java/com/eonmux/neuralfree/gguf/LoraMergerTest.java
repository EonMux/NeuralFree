package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * A LoRA merge ({@link LoraMerger}, llama.cpp tools/export-lora/export-lora.cpp) over a synthetic F32 base
 * and GGUF adapter files: the scaled {@code B A} delta of a projection, the transposed-A delta of a token
 * embedding, the alpha-0 and user-scale folds, the F32-into-F32 and quantized-into-F16 output types, the
 * byte-exact copies of the unadapted tensors, the {@code general.file_type} marker and the refusals (a split
 * base, adapters of different tensor lists, an adapter without pairs, a half pair, an unknown target, a
 * wrong shape, an activated aLoRA and an existing output).
 */
public class LoraMergerTest {

    private static final int IN = 32;
    private static final int OUT = 24;
    private static final int EMBD = 16;
    private static final int VOCAB = 12;
    private static final int RANK = 4;
    private static final int QIN = 32;
    private static final int QOUT = 48;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path out;

    @Before
    public void newOut() {
        out = tmp.getRoot().toPath().resolve("merged.gguf");
    }

    // ------------------------------------------------------------------ fixtures

    /** Base tensor value: row tags over columns, small enough for F16 rounding to stay negligible. */
    private static float[] rows(int rows, int cols) {
        float[] v = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                v[r * cols + c] = 0.5f + 0.1f * r - 0.03f * c;
            }
        }
        return v;
    }

    /** The A factor values of a projection, stored {@code {in, r}}: {@code A[k][i]} at {@code k*in + i}. */
    private static float[] aProj() {
        float[] v = new float[RANK * IN];
        for (int k = 0; k < RANK; k++) {
            for (int i = 0; i < IN; i++) {
                v[k * IN + i] = 0.1f * k - 0.02f * i;
            }
        }
        return v;
    }

    /** The B factor values of a projection, stored {@code {r, out}}: {@code B[o][k]} at {@code o*r + k}. */
    private static float[] bProj() {
        float[] v = new float[OUT * RANK];
        for (int o = 0; o < OUT; o++) {
            for (int k = 0; k < RANK; k++) {
                v[o * RANK + k] = 0.05f * o + 0.03f * k;
            }
        }
        return v;
    }

    /** The transposed A factor of the embedding, stored {@code {r, vocab}}: {@code A[t][k]} at {@code t*r + k}. */
    private static float[] aEmb() {
        float[] v = new float[VOCAB * RANK];
        for (int t = 0; t < VOCAB; t++) {
            for (int k = 0; k < RANK; k++) {
                v[t * RANK + k] = 0.02f * t + 0.1f * k;
            }
        }
        return v;
    }

    /** The B factor of the embedding, stored {@code {r, embd}}: {@code B[e][k]} at {@code e*r + k}. */
    private static float[] bEmb() {
        float[] v = new float[EMBD * RANK];
        for (int e = 0; e < EMBD; e++) {
            for (int k = 0; k < RANK; k++) {
                v[e * RANK + k] = 0.04f * e + 0.03f * k;
            }
        }
        return v;
    }

    /** The scaled projection delta {@code s * B A} of the fixture factors, evaluated in double. */
    private static double deltaProj(int o, int i, double s) {
        double acc = 0;
        for (int k = 0; k < RANK; k++) {
            acc += (double) bProj()[o * RANK + k] * aProj()[k * IN + i];
        }
        return s * acc;
    }

    /** The scaled embedding delta {@code s * A^T B} of the fixture factors, evaluated in double. */
    private static double deltaEmb(int t, int e, double s) {
        double acc = 0;
        for (int k = 0; k < RANK; k++) {
            acc += (double) aEmb()[t * RANK + k] * bEmb()[e * RANK + k];
        }
        return s * acc;
    }

    /** The effective merge scale of an adapter at user scale {@code u} (llama-adapter.h:53-57). */
    private static double scaleOf(double alpha, double u) {
        return alpha != 0 ? u * alpha / RANK : u;
    }

    /** An F32 base with an embedding, a projection and a norm tensor the adapters leave alone. */
    private Path base() throws IOException {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.u32(w.archKey("context_length"), 512);
        w.tensorF32("token_embd.weight", new long[]{EMBD, VOCAB}, rows(VOCAB, EMBD));
        w.tensorF32("blk.0.attn_q.weight", new long[]{IN, OUT}, rows(OUT, IN));
        w.tensorF32("blk.0.attn_norm.weight", new long[]{IN}, rows(1, IN));
        return w.writeTempFile();
    }

    /**
     * A GGUF adapter file over the projection and the embedding, at {@code alpha}: pass {@code withEmb}
     * false to adapt the projection only, or {@code qOut}/{@code qIn} overrides for a shape-refusal fixture.
     */
    private Path adapter(double alpha, boolean withEmb) throws IOException {
        return adapter(alpha, withEmb, OUT, IN);
    }

    private Path adapter(double alpha, boolean withEmb, int qOut, int qIn) throws IOException {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.str("general.type", "adapter");
        w.str("adapter.type", "lora");
        w.f32("adapter.lora.alpha", (float) alpha);
        if (qOut == OUT && qIn == IN) {
            w.tensorF32("blk.0.attn_q.weight.lora_a", new long[]{qIn, RANK}, aProj());
            w.tensorF32("blk.0.attn_q.weight.lora_b", new long[]{RANK, qOut}, bProj());
        } else {
            float[] a = new float[RANK * qIn];
            Arrays.fill(a, 0.1f);
            float[] b = new float[qOut * RANK];
            Arrays.fill(b, 0.1f);
            w.tensorF32("blk.0.attn_q.weight.lora_a", new long[]{qIn, RANK}, a);
            w.tensorF32("blk.0.attn_q.weight.lora_b", new long[]{RANK, qOut}, b);
        }
        if (withEmb) {
            w.tensorF32("token_embd.weight.lora_a", new long[]{RANK, VOCAB}, aEmb());
            w.tensorF32("token_embd.weight.lora_b", new long[]{RANK, EMBD}, bEmb());
        }
        return w.writeTempFile();
    }

    private Path merge(Path baseFile, Path... adapterFiles) throws IOException {
        return mergeAt(baseFile, out, 1.0f, adapterFiles);
    }

    private Path mergeAt(Path baseFile, Path target, float userScale, Path... adapterFiles) throws IOException {
        List<LoraMerger.Adapter> adapters = new java.util.ArrayList<>();
        for (Path p : adapterFiles) {
            adapters.add(new LoraMerger.Adapter(p.toString(), userScale));
        }
        try (GGUFFileParser src = new GGUFFileParser(baseFile.toString())) {
            return LoraMerger.merge(src, adapters, target, ModelSourceOptions.defaults(), false);
        }
    }

    private static float[] floats(GGUFFileParser p, String name) throws IOException {
        return p.readTensorDataAsFloat(p.getTensor(name));
    }

    // ------------------------------------------------------------------ the merge

    @Test
    public void mergeAddsTheScaledDeltasAndCopiesTheRest() throws Exception {
        Path merged = merge(base(), adapter(8.0, true));
        try (GGUFFileParser in = new GGUFFileParser(base().toString());
             GGUFFileParser m = new GGUFFileParser(merged.toString())) {
            // one output file, same tensor table, the F32 base stays F32
            assertEquals(in.getTensors().size(), m.getTensors().size());
            assertEquals(GGMLType.F32, m.getTensor("blk.0.attn_q.weight").type());
            assertEquals(GGMLType.F32, m.getTensor("token_embd.weight").type());
            double s = scaleOf(8.0, 1.0);
            float[] q = floats(m, "blk.0.attn_q.weight");
            float[] qBase = rows(OUT, IN);
            for (int o = 0; o < OUT; o++) {
                for (int i = 0; i < IN; i++) {
                    assertEquals("q[" + o + "][" + i + "]", qBase[o * IN + i] + deltaProj(o, i, s),
                            q[o * IN + i], 1e-3);
                }
            }
            float[] e = floats(m, "token_embd.weight");
            float[] eBase = rows(VOCAB, EMBD);
            for (int t = 0; t < VOCAB; t++) {
                for (int c = 0; c < EMBD; c++) {
                    assertEquals("emb[" + t + "][" + c + "]", eBase[t * EMBD + c] + deltaEmb(t, c, s),
                            e[t * EMBD + c], 1e-3);
                }
            }
            // the unadapted norm tensor is byte-exact
            assertArrayEquals(in.readTensorData(in.getTensor("blk.0.attn_norm.weight")),
                    m.readTensorData(m.getTensor("blk.0.attn_norm.weight")));
            // the output is marked mostly-F16, the base keys stay
            assertEquals(1L, ((Number) m.getMetadata().get(GgufKeys.GENERAL_FILE_TYPE)).longValue());
            assertEquals("llama", m.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertEquals(512L, ((Number) m.getMetadata().get("llama.context_length")).longValue());
            assertEquals(in.getAlignment(), m.getAlignment());
        }
    }

    @Test
    public void alphaZeroAppliesTheUserScaleAlone() throws Exception {
        Path merged = merge(base(), adapter(0.0, true));
        try (GGUFFileParser m = new GGUFFileParser(merged.toString())) {
            double s = scaleOf(0.0, 1.0);
            float[] q = floats(m, "blk.0.attn_q.weight");
            float[] qBase = rows(OUT, IN);
            assertEquals(qBase[0] + deltaProj(0, 0, s), q[0], 1e-4);
            float[] e = floats(m, "token_embd.weight");
            assertEquals(rows(VOCAB, EMBD)[0] + deltaEmb(0, 0, s), e[0], 1e-4);
        }
    }

    @Test
    public void theUserScaleMultipliesTheDelta() throws Exception {
        Path scaled = tmp.getRoot().toPath().resolve("scaled.gguf");
        mergeAt(base(), scaled, 2.5f, adapter(8.0, true));
        try (GGUFFileParser m = new GGUFFileParser(scaled.toString())) {
            double s = scaleOf(8.0, 2.5);
            float[] q = floats(m, "blk.0.attn_q.weight");
            assertEquals(rows(OUT, IN)[3 * IN + 5] + deltaProj(3, 5, s), q[3 * IN + 5], 1e-3);
            float[] e = floats(m, "token_embd.weight");
            assertEquals(rows(VOCAB, EMBD)[2 * EMBD + 7] + deltaEmb(2, 7, s), e[2 * EMBD + 7], 1e-3);
        }
    }

    @Test
    public void aQuantizedBaseTensorMergesIntoF16() throws Exception {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.tensorFromFloats("blk.1.ffn_down.weight", GGMLType.Q8_0, new long[]{QIN, QOUT}, rows(QOUT, QIN));
        w.tensorF32("blk.1.ffn_norm.weight", new long[]{QIN}, rows(1, QIN));
        Path baseFile = w.writeTempFile();
        GgufTestWriter aw = new GgufTestWriter();
        aw.architecture("llama");
        aw.str("general.type", "adapter");
        aw.str("adapter.type", "lora");
        aw.f32("adapter.lora.alpha", 4.0f);
        float[] a = new float[RANK * QIN];
        Arrays.fill(a, 0.05f);
        float[] b = new float[QOUT * RANK];
        Arrays.fill(b, -0.2f);
        aw.tensorF32("blk.1.ffn_down.weight.lora_a", new long[]{QIN, RANK}, a);
        aw.tensorF32("blk.1.ffn_down.weight.lora_b", new long[]{RANK, QOUT}, b);
        Path adapterFile = aw.writeTempFile();
        Path merged = merge(baseFile, adapterFile);
        try (GGUFFileParser in = new GGUFFileParser(baseFile.toString());
             GGUFFileParser m = new GGUFFileParser(merged.toString())) {
            assertEquals(GGMLType.F16, m.getTensor("blk.1.ffn_down.weight").type());
            float[] qBase = in.readTensorDataAsFloat(in.getTensor("blk.1.ffn_down.weight"));
            float[] q = floats(m, "blk.1.ffn_down.weight");
            double s = scaleOf(4.0, 1.0);
            for (int o = 0; o < QOUT; o++) {
                for (int i = 0; i < QIN; i++) {
                    double delta = 0;
                    for (int k = 0; k < RANK; k++) {
                        delta += (double) b[o * RANK + k] * a[k * QIN + i];
                    }
                    assertEquals("f16 q[" + o + "][" + i + "]", qBase[o * QIN + i] + s * delta,
                            q[o * QIN + i], 0.05);
                }
            }
            assertArrayEquals(in.readTensorData(in.getTensor("blk.1.ffn_norm.weight")),
                    m.readTensorData(m.getTensor("blk.1.ffn_norm.weight")));
        }
    }

    @Test
    public void twoAdaptersSumTheirDeltas() throws Exception {
        Path first = adapter(4.0, true);
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.str("general.type", "adapter");
        w.str("adapter.type", "lora");
        w.f32("adapter.lora.alpha", 12.0f);
        w.tensorF32("blk.0.attn_q.weight.lora_a", new long[]{IN, RANK}, aProj());
        w.tensorF32("blk.0.attn_q.weight.lora_b", new long[]{RANK, OUT}, bProj());
        w.tensorF32("token_embd.weight.lora_a", new long[]{RANK, VOCAB}, aEmb());
        w.tensorF32("token_embd.weight.lora_b", new long[]{RANK, EMBD}, bEmb());
        Path second = w.writeTempFile();
        Path merged = merge(base(), first, second);
        try (GGUFFileParser m = new GGUFFileParser(merged.toString())) {
            double s = scaleOf(4.0, 1.0) + scaleOf(12.0, 1.0);
            float[] q = floats(m, "blk.0.attn_q.weight");
            assertEquals(rows(OUT, IN)[1 * IN + 2] + deltaProj(1, 2, s), q[1 * IN + 2], 1e-3);
        }
    }

    @Test
    public void fileTypeIsAppendedWhenTheBaseHasNoneAndReplacedWhenItCarriesOne() throws Exception {
        Path merged = merge(base(), adapter(8.0, true));
        try (GGUFFileParser m = new GGUFFileParser(merged.toString())) {
            assertEquals(GGUFValueType.UINT32, m.getValueType(GgufKeys.GENERAL_FILE_TYPE));
        }
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.u32("general.file_type", 0);
        w.tensorF32("blk.0.attn_q.weight", new long[]{IN, OUT}, rows(OUT, IN));
        Path typed = w.writeTempFile();
        Path replaced = tmp.getRoot().toPath().resolve("typed.gguf");
        mergeAt(typed, replaced, 1.0f, adapter(8.0, false));
        try (GGUFFileParser m = new GGUFFileParser(replaced.toString())) {
            assertEquals(GGMLType.F32, m.getTensor("blk.0.attn_q.weight").type());
            assertEquals(GGUFValueType.UINT32, m.getValueType(GgufKeys.GENERAL_FILE_TYPE));
            assertEquals(1L, ((Number) m.getMetadata().get(GgufKeys.GENERAL_FILE_TYPE)).longValue());
        }
    }

    @Test
    public void anExistingOutputRequiresOverwrite() throws Exception {
        Path baseFile = base();
        Path adapterFile = adapter(8.0, true);
        Path merged = merge(baseFile, adapterFile);
        IOException e = assertThrows(IOException.class, () -> merge(baseFile, adapterFile));
        assertTrue(e.getMessage(), e.getMessage().contains("already exists"));
        List<LoraMerger.Adapter> adapters = Collections.singletonList(
                new LoraMerger.Adapter(adapterFile.toString(), 1.0f));
        try (GGUFFileParser src = new GGUFFileParser(baseFile.toString())) {
            assertEquals(merged, LoraMerger.merge(src, adapters, merged, ModelSourceOptions.defaults(), true));
        }
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void aSplitBaseIsRefused() throws Exception {
        Path dir = tmp.getRoot().toPath();
        GgufTestWriter s0 = new GgufTestWriter();
        s0.architecture("llama");
        s0.u16("split.no", 0);
        s0.u16("split.count", 2);
        s0.i32("split.tensors.count", 2);
        s0.tensorF32("blk.0.attn_q.weight", new long[]{IN, OUT}, rows(OUT, IN));
        s0.writeTo(dir.resolve("split-00001-of-00002.gguf"));
        GgufTestWriter s1 = new GgufTestWriter();
        s1.alignment(32);
        s1.u16("split.no", 1);
        s1.u16("split.count", 2);
        s1.i32("split.tensors.count", 2);
        s1.tensorF32("blk.0.attn_norm.weight", new long[]{IN}, rows(1, IN));
        s1.writeTo(dir.resolve("split-00002-of-00002.gguf"));
        Path split = dir.resolve("split-00001-of-00002.gguf");
        IOException e = assertThrows(IOException.class, () -> merge(split, adapter(8.0, true)));
        assertTrue(e.getMessage(), e.getMessage().contains("split model is not yet supported"));
    }

    @Test
    public void adaptersOfDifferentTensorListsAreRefused() throws Exception {
        Path full = adapter(8.0, true);
        Path partial = adapter(8.0, false);
        IOException e = assertThrows(IOException.class, () -> merge(base(), full, partial));
        assertTrue(e.getMessage(), e.getMessage().contains("same list of tensors"));
    }

    @Test
    public void anAdapterWithoutPairsIsRefused() throws Exception {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.str("general.type", "adapter");
        w.str("adapter.type", "lora");
        w.f32("adapter.lora.alpha", 8.0f);
        Path empty = w.writeTempFile();
        IOException e = assertThrows(IOException.class, () -> merge(base(), empty));
        assertTrue(e.getMessage(), e.getMessage().contains("nothing to merge"));
    }

    @Test
    public void aHalfPairIsRefused() throws Exception {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.str("general.type", "adapter");
        w.str("adapter.type", "lora");
        w.f32("adapter.lora.alpha", 8.0f);
        w.tensorF32("blk.0.attn_q.weight.lora_a", new long[]{IN, RANK}, aProj());
        Path half = w.writeTempFile();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> merge(base(), half));
        assertTrue(e.getMessage(), e.getMessage().contains("missing one component"));
    }

    @Test
    public void anUnknownTargetTensorIsRefused() throws Exception {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.str("general.type", "adapter");
        w.str("adapter.type", "lora");
        w.f32("adapter.lora.alpha", 8.0f);
        w.tensorF32("blk.9.attn_q.weight.lora_a", new long[]{IN, RANK}, aProj());
        w.tensorF32("blk.9.attn_q.weight.lora_b", new long[]{RANK, OUT}, bProj());
        Path unknown = w.writeTempFile();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> merge(base(), unknown));
        assertTrue(e.getMessage(), e.getMessage().contains("does not exist in base model"));
    }

    @Test
    public void aWrongShapePairIsRefused() throws Exception {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.str("general.type", "adapter");
        w.str("adapter.type", "lora");
        w.f32("adapter.lora.alpha", 8.0f);
        float[] a = new float[RANK * (IN + 4)];
        Arrays.fill(a, 0.1f);
        float[] b = new float[OUT * RANK];
        Arrays.fill(b, 0.1f);
        w.tensorF32("blk.0.attn_q.weight.lora_a", new long[]{IN + 4, RANK}, a);
        w.tensorF32("blk.0.attn_q.weight.lora_b", new long[]{RANK, OUT}, b);
        Path wrong = w.writeTempFile();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> merge(base(), wrong));
        assertTrue(e.getMessage(), e.getMessage().contains("incorrect shape"));
    }

    @Test
    public void anArchMismatchIsRefused() throws Exception {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("qwen");
        w.str("general.type", "adapter");
        w.str("adapter.type", "lora");
        w.f32("adapter.lora.alpha", 8.0f);
        w.tensorF32("blk.0.attn_q.weight.lora_a", new long[]{IN, RANK}, aProj());
        w.tensorF32("blk.0.attn_q.weight.lora_b", new long[]{RANK, OUT}, bProj());
        Path other = w.writeTempFile();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> merge(base(), other));
        assertTrue(e.getMessage(), e.getMessage().contains("mismatch"));
    }

    @Test
    public void anActivatedLoraIsRefused() throws Exception {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("llama");
        w.str("general.type", "adapter");
        w.str("adapter.type", "lora");
        w.f32("adapter.lora.alpha", 8.0f);
        w.u32Array("adapter.alora.invocation_tokens", 1, 2, 3);
        w.tensorF32("blk.0.attn_q.weight.lora_a", new long[]{IN, RANK}, aProj());
        w.tensorF32("blk.0.attn_q.weight.lora_b", new long[]{RANK, OUT}, bProj());
        Path alora = w.writeTempFile();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> merge(base(), alora));
        assertTrue(e.getMessage(), e.getMessage().contains("activated LoRA"));
    }
}
