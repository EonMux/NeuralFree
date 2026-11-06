package com.eonmux.neuralfree.gguf.testutil;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * GGUF LoRA adapter files for {@link SyntheticModels#llama()} (llama-adapter.cpp:203-375 layout): every layer's
 * {@code attn_q/k/v/output} and {@code ffn_gate/up/down}, {@code output.weight} and {@code token_embd.weight},
 * with random factors of one seed; plain, activated ({@link #activated}) or with PEFT task keys ({@link #tasked}),
 * and the projections of {@link SyntheticModels#bert()} ({@link #bert}). A projection {@code W {in, out}} takes {@code A {in, r}} and {@code B {r, out}},
 * the token table {@code E {n_embd, n_vocab}} takes {@code A {r, n_vocab}} and {@code B {r, n_embd}}.
 */
public final class LoraAdapterFiles {

    private static final String EMBD = "token_embd.weight";
    private static final String[] LAYER_PROJECTIONS = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate",
            "ffn_up", "ffn_down"};

    private LoraAdapterFiles() {
    }

    /** Writes an adapter of rank {@code rank} and alpha {@code alpha} with factors of {@code seed} to a temp file. */
    public static Path llama(long seed, int rank, float alpha) throws IOException {
        return writer(seed, rank, alpha).writeTempFile();
    }

    /** {@link #llama(long, int, float)} for the synthetic llama of {@code config}. */
    public static Path llama(SyntheticModels.Config config, long seed, int rank, float alpha) throws IOException {
        return writer(SyntheticModels.llama(config), "llama", targets(config.nLayer()), seed, rank, alpha)
                .writeTempFile();
    }

    /**
     * {@link #llama} as an activated LoRA ({@code adapter.alora.invocation_tokens} = {@code invocation}), whose
     * delta applies from the invocation sequence's last occurrence in a prompt on.
     */
    public static Path activated(long seed, int rank, float alpha, int... invocation) throws IOException {
        long[] ids = new long[invocation.length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = invocation[i];
        }
        return writer(seed, rank, alpha).u32Array("adapter.alora.invocation_tokens", ids).writeTempFile();
    }

    /**
     * {@link #activated} for the synthetic qwen2vl of {@link SyntheticModels.Config#llama()} (the same projections,
     * an M-RoPE model whose media prompts hold media rows).
     */
    public static Path activatedQwen2vl(long seed, int rank, float alpha, int... invocation) throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        long[] ids = new long[invocation.length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = invocation[i];
        }
        return writer(SyntheticModels.qwen2vl(c), "qwen2vl", targets(c.nLayer()), seed, rank, alpha)
                .u32Array("adapter.alora.invocation_tokens", ids).writeTempFile();
    }

    /** {@link #llama} carrying the PEFT task keys {@code adapter.lora.task_name} and {@code prompt_prefix}. */
    public static Path tasked(long seed, int rank, float alpha, String taskName, String promptPrefix)
            throws IOException {
        return writer(seed, rank, alpha).str("adapter.lora.task_name", taskName)
                .str("adapter.lora.prompt_prefix", promptPrefix).writeTempFile();
    }

    /**
     * An adapter of {@link SyntheticModels#bert()} (and its {@link SyntheticModels#rank} classifier): every layer's
     * {@code attn_q/k/v/output} and {@code ffn_up/down}, rank {@code rank}, alpha {@code alpha}, factors of
     * {@code seed}.
     */
    public static Path bert(long seed, int rank, float alpha) throws IOException {
        List<String> names = new ArrayList<>();
        for (int il = 0; il < SyntheticModels.Config.bert().nLayer(); il++) {
            for (String p : new String[]{"attn_q", "attn_k", "attn_v", "attn_output", "ffn_up", "ffn_down"}) {
                names.add("blk." + il + "." + p + ".weight");
            }
        }
        return writer(SyntheticModels.bert(), "bert", names, seed, rank, alpha).writeTempFile();
    }

    private static GgufTestWriter writer(long seed, int rank, float alpha) {
        return writer(SyntheticModels.llama(), "llama", targets(SyntheticModels.Config.llama().nLayer()), seed, rank,
                alpha);
    }

    private static GgufTestWriter writer(GgufTestWriter base, String arch, List<String> targets, long seed, int rank,
                                         float alpha) {
        GgufTestWriter w = new GgufTestWriter().architecture(arch);
        w.str("general.type", "adapter").str("adapter.type", "lora").f32("adapter.lora.alpha", alpha);
        long next = seed;
        for (String name : targets) {
            long[] ne = base.tensorShape(name);
            int rows = Math.toIntExact(ne[1]);
            int cols = Math.toIntExact(ne[0]);
            if (EMBD.equals(name)) {
                w.tensorF32(name + ".lora_a", new long[]{rank, ne[1]}, random(next++, rows * rank));
                w.tensorF32(name + ".lora_b", new long[]{rank, ne[0]}, random(next++, cols * rank));
            } else {
                w.tensorF32(name + ".lora_a", new long[]{ne[0], rank}, random(next++, rank * cols));
                w.tensorF32(name + ".lora_b", new long[]{rank, ne[1]}, random(next++, rows * rank));
            }
        }
        return w;
    }

    private static List<String> targets(int nLayer) {
        List<String> names = new ArrayList<>();
        for (int il = 0; il < nLayer; il++) {
            for (String p : LAYER_PROJECTIONS) {
                names.add("blk." + il + "." + p + ".weight");
            }
        }
        names.add("output.weight");
        names.add(EMBD);
        return names;
    }

    private static float[] random(long seed, int n) {
        Random r = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = 0.2f * (r.nextFloat() - 0.5f);
        }
        return out;
    }
}
