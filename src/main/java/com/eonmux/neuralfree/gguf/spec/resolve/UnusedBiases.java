package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * Optional biases that llama.cpp loads ({@code TENSOR_NOT_REQUIRED}) but whose graph passes {@code NULL} in their
 * place, so a file that carries them computes as if they were absent. The resolvers clear the matching spec flags
 * ({@code AttentionSpec.inBias}, {@code AttentionSpec.outBias}, {@code FfnSpec.biases}) and the executors then
 * never read the tensors.
 *
 * <ul>
 *   <li>{@code attn_qkv.bias}, {@code attn_q.bias}, {@code attn_k.bias}, {@code attn_v.bias}: mimo2
 *       ({@code create_tensor_qkv} at mimo2.cpp:47; the graph projects with {@code build_lora_mm} alone, :121-148,
 *       and so does its MTP block, :323) and qwen4exp ({@code create_tensor_qkv} at qwen4exp.cpp:243; :958-972).</li>
 *   <li>{@code attn_output.bias}: falcon-h1 (falcon-h1.cpp:152-154), ernie4_5 and ernie4_5-moe (ernie4-5.cpp:45,
 *       118-120; ernie4-5-moe.cpp:50-52, which reuses the ernie4_5 loader), llada (llada.cpp:42, 110) and refact
 *       (refact.cpp:38, 111). paddleocr shares ernie4_5's tensors and adds the bias (paddleocr.cpp:61-63).</li>
 *   <li>{@code ffn_gate.bias}, {@code ffn_up.bias}, {@code ffn_down.bias}: llada (llada.cpp:53-58 loaded,
 *       :124-128 {@code build_ffn} with every bias {@code NULL}) and refact (refact.cpp:55-58 loaded, :128-133).
 *       minicpm and granite-moe load them too but run the Granite graph, which adds them (granite.cpp:263-265).
 *       nomic-bert loads {@code ffn_up.bias} and {@code ffn_down.bias} (nomic-bert.cpp:37, 39) and runs the bert
 *       graph's last FFN branch, whose {@code build_ffn} takes no bias (bert.cpp:202-207); nomic-bert-moe's dense
 *       layers take the biased branch (:186-192).</li>
 * </ul>
 */
public final class UnusedBiases {

    private static final Set<Arch> ATTN_INPUT = Collections.unmodifiableSet(EnumSet.of(Arch.MIMO2, Arch.QWEN4EXP));

    private static final Set<Arch> ATTN_OUTPUT = Collections.unmodifiableSet(EnumSet.of(
            Arch.FALCON_H1, Arch.ERNIE4_5, Arch.ERNIE4_5_MOE, Arch.LLADA, Arch.REFACT));

    private static final Set<Arch> FFN = Collections.unmodifiableSet(EnumSet.of(
            Arch.LLADA, Arch.REFACT, Arch.NOMIC_BERT));

    private UnusedBiases() {
    }

    /** True when the graph of {@code arch} ignores {@code attn_qkv.bias} and {@code attn_q/k/v.bias}. */
    public static boolean attnInputUnused(Arch arch) {
        return ATTN_INPUT.contains(Objects.requireNonNull(arch, "arch"));
    }

    /** True when the graph of {@code arch} ignores {@code attn_output.bias}. */
    public static boolean attnOutputUnused(Arch arch) {
        return ATTN_OUTPUT.contains(Objects.requireNonNull(arch, "arch"));
    }

    /** True when the graph of {@code arch} ignores the dense FFN's {@code ffn_{gate,up,down}.bias}. */
    public static boolean ffnUnused(Arch arch) {
        return FFN.contains(Objects.requireNonNull(arch, "arch"));
    }

    /** Architectures whose graph ignores the Q/K/V projection biases. */
    public static Set<Arch> attnInput() {
        return ATTN_INPUT;
    }

    /** Architectures whose graph ignores {@code attn_output.bias}. */
    public static Set<Arch> attnOutput() {
        return ATTN_OUTPUT;
    }

    /** Architectures whose graph ignores the dense FFN biases. */
    public static Set<Arch> ffn() {
        return FFN;
    }
}
