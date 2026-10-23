package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Reads a LoRA adapter GGUF file against its base model (llama_adapter_lora_init_impl, src/llama-adapter.cpp:151-427):
 * <ul>
 *   <li>metadata: {@code general.type = adapter}, {@code general.architecture} equal to the base model's,
 *       {@code adapter.type = lora}, {@code adapter.lora.alpha} (0 when absent) and the jina task keys
 *       {@code adapter.lora.task_name} / {@code adapter.lora.prompt_prefix} (:203-242). An activated LoRA
 *       ({@code adapter.alora.invocation_tokens}, :225-241) is refused: its delta applies only to the tokens after
 *       an invocation sequence, a per-token mask the forward passes do not carry;</li>
 *   <li>tensors: {@code <base>.lora_a} and {@code <base>.lora_b} pairs; {@code *_norm.weight} entries are skipped and
 *       any other name is an error (:268-298); both halves must exist and the base tensor must exist (under its
 *       canonical name or a catalog alias) and not be a PRISM Hadamard-folded weight (:323-340);</li>
 *   <li>shapes (:361-375): a projection {@code W {in, out}} needs {@code A {in, r}} and {@code B {r, out}}; an expert
 *       tensor {@code W {in, out, n_expert}} needs {@code A {in, r, n_expert}} and {@code B {r, out, n_expert}}
 *       (llama.cpp checks {@code ne[0]} / {@code ne[1]} and lets {@code ggml_mul_mat_id} reject an expert-count
 *       mismatch at graph time; it is checked here); a token embedding {@code E {n_embd, n_vocab}} needs
 *       {@code A {r, n_vocab}} and {@code B {r, n_embd}}.</li>
 * </ul>
 * Every half is decoded to F32 into the returned {@link LoraAdapter}; the file is closed before returning.
 */
public final class LoraAdapterLoader {

    static final String SUFFIX_A = ".lora_a";
    static final String SUFFIX_B = ".lora_b";
    static final String SKIPPED_SUFFIX = "_norm.weight";
    static final String GENERAL_TYPE_ADAPTER = "adapter";
    static final String ADAPTER_TYPE_LORA = "lora";

    private LoraAdapterLoader() {
    }

    /**
     * Loads the adapter at {@code path} for the model whose weights are {@code base}.
     *
     * @throws IOException         when the file cannot be read
     * @throws GgufFormatException           when the metadata, the tensor names or the shapes do not form an adapter of
     *                                       the base model
     * @throws UnsupportedOperationException for an activated LoRA
     */
    public static LoraAdapter load(String path, CpuWeights base) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(base, "base");
        GGUFFileParser parser = new GGUFFileParser(path);
        try {
            GgufKv kv = GgufKv.of(parser);
            String generalType = kv.str(GgufKeys.GENERAL_TYPE, "");
            if (!GENERAL_TYPE_ADAPTER.equals(generalType)) {
                throw new GgufFormatException(path + ": expect general.type to be 'adapter', but got: " + generalType);
            }
            String arch = kv.archName();
            if (arch == null || !arch.equals(base.archName())) {
                throw new GgufFormatException(path + ": model arch " + base.archName() + " and LoRA arch " + arch
                        + " mismatch");
            }
            String adapterType = kv.str(GgufKeys.ADAPTER_TYPE, "");
            if (!ADAPTER_TYPE_LORA.equals(adapterType)) {
                throw new GgufFormatException(path + ": expect adapter.type to be 'lora', but got: " + adapterType);
            }
            float alpha = kv.f32(GgufKeys.ADAPTER_LORA_ALPHA, 0.0f);
            int[] invocation = null;
            if (kv.has(GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS)) {
                if (!kv.isArray(GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS)) {
                    throw new GgufFormatException(path + ": invalid gguf type for "
                            + GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS);
                }
                invocation = kv.u32Array(GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS);
                if (invocation.length > 0) {
                    throw new UnsupportedOperationException(path + ": activated LoRA ("
                            + GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS + ", " + invocation.length + " tokens) is not"
                            + " supported: its delta applies only after the invocation sequence, a per-token mask the"
                            + " forward passes do not carry; use a plain LoRA adapter");
                }
            }
            String taskName = kv.str(GgufKeys.ADAPTER_LORA_TASK_NAME, null);
            String promptPrefix = kv.str(GgufKeys.ADAPTER_LORA_PROMPT_PREFIX, null);
            Map<String, LoraAdapter.Pair> pairs = readPairs(path, parser, base);
            return new LoraAdapter(path, arch, alpha, taskName, promptPrefix, invocation, pairs, scalars(parser));
        } finally {
            parser.close();
        }
    }

    /** Scalar metadata as strings, in key order (array values are left out, as llama.cpp does). */
    private static Map<String, String> scalars(GGUFFileParser parser) {
        Map<String, String> out = new TreeMap<>();
        for (Map.Entry<String, Object> e : parser.getMetadata().entrySet()) {
            Object v = e.getValue();
            if (!(v instanceof List) && !(v != null && v.getClass().isArray())) {
                out.put(e.getKey(), String.valueOf(v));
            }
        }
        return out;
    }

    private static Map<String, LoraAdapter.Pair> readPairs(String path, GGUFFileParser parser, CpuWeights base)
            throws IOException {
        Map<String, GGUFTensorInfo[]> halves = new LinkedHashMap<>();
        for (GGUFTensorInfo info : parser.getTensors()) {
            String name = info.name();
            if (name.endsWith(SUFFIX_A)) {
                halves.computeIfAbsent(strip(name, SUFFIX_A), k -> new GGUFTensorInfo[2])[0] = info;
            } else if (name.endsWith(SUFFIX_B)) {
                halves.computeIfAbsent(strip(name, SUFFIX_B), k -> new GGUFTensorInfo[2])[1] = info;
            } else if (!name.endsWith(SKIPPED_SUFFIX)) {
                throw new GgufFormatException(path + ": LoRA tensor '" + name + "' has unexpected suffix");
            }
        }
        Map<String, LoraAdapter.Pair> pairs = new LinkedHashMap<>();
        try (CpuWeights adapter = CpuWeights.open(parser, TensorCatalog.of(parser), null)) {
            for (Map.Entry<String, GGUFTensorInfo[]> e : halves.entrySet()) {
                String target = e.getKey();
                GGUFTensorInfo a = e.getValue()[0];
                GGUFTensorInfo b = e.getValue()[1];
                if (a == null || b == null) {
                    throw new GgufFormatException(path + ": LoRA tensor pair for '" + target
                            + "' is missing one component");
                }
                pairs.put(target, pair(path, target, a, b, base, adapter));
            }
        }
        return pairs;
    }

    private static LoraAdapter.Pair pair(String path, String target, GGUFTensorInfo a, GGUFTensorInfo b,
                                         CpuWeights base, CpuWeights adapter) {
        TensorRef model = resolve(base.catalog(), target);
        if (model == null) {
            throw new GgufFormatException(path + ": LoRA tensor '" + target
                    + "' does not exist in base model (hint: maybe wrong base model?)");
        }
        CpuTensorRef view = base.get(model);
        if (base.of(view).input() != null || base.inverseHadamard(view) != null) {
            throw new GgufFormatException(path + ": LoRA tensor '" + target
                    + "' targets a prism.hadamard folded weight, which is not supported");
        }
        boolean experts = model.rank() == 3;
        if (model.rank() != 2 && !experts) {
            throw new GgufFormatException(path + ": LoRA tensor '" + target + "' targets a " + model.rank()
                    + "-D base tensor; only matrices and 3-D expert tensors take LoRA deltas");
        }
        if (a.rank() != model.rank() || b.rank() != model.rank()) {
            throw new GgufFormatException(path + ": LoRA tensors of '" + target + "' are " + a.rank() + "-D / "
                    + b.rank() + "-D, the base tensor is " + model.rank() + "-D (hint: maybe wrong base model?)");
        }
        boolean embedding = target.endsWith(TensorNames.weight(TensorNames.TOKEN_EMBD));
        if (embedding && experts) {
            throw new GgufFormatException(path + ": LoRA tensor '" + target + "' is a 3-D token embedding table");
        }
        long mIn = model.ne(0);
        long mOut = model.ne(1);
        if (experts && (a.dim(2) != model.ne(2) || b.dim(2) != model.ne(2))) {
            throw new GgufFormatException(path + ": LoRA tensors of '" + target + "' hold " + a.dim(2) + " / "
                    + b.dim(2) + " experts, the base tensor " + model.ne(2) + " (hint: maybe wrong base model?)");
        }
        if (embedding) {
            if (mIn != b.dim(1) || mOut != a.dim(1) || a.dim(0) != b.dim(0)) {
                throw new GgufFormatException(path + ": tensor '" + target
                        + "' has incorrect shape (hint: maybe wrong base model?)");
            }
        } else {
            if (mIn != a.dim(0) || mOut != b.dim(1)) {
                throw new GgufFormatException(path + ": tensor '" + target
                        + "' has incorrect shape (hint: maybe wrong base model?)");
            }
            if (a.dim(1) != b.dim(0)) {
                throw new GgufFormatException(path + ": lora_a tensor of '" + target + "' is not transposed (hint:"
                        + " adapter from \"finetune\" example is no longer supported)");
            }
        }
        int rank = HeadsBlocks.toInt(b.dim(0), target + SUFFIX_B);
        int in = HeadsBlocks.toInt(embedding ? a.dim(1) : a.dim(0), target + SUFFIX_A);
        int out = HeadsBlocks.toInt(b.dim(1), target + SUFFIX_B);
        LoraAdapter.Target kind = embedding ? LoraAdapter.Target.EMBEDDING
                : experts ? LoraAdapter.Target.EXPERTS : LoraAdapter.Target.PROJECTION;
        int nExpert = experts ? HeadsBlocks.toInt(model.ne(2), target) : 1;
        return new LoraAdapter.Pair(target, kind, in, out, rank, nExpert, decode(adapter, a), decode(adapter, b));
    }

    /**
     * The base tensor an adapter name targets: the stored tensor or its model-level alias, else, for a main-stack
     * name {@code blk.<il>.<suffix>}, its Hugging Face layer alias; null when absent.
     */
    static TensorRef resolve(TensorCatalog catalog, String target) {
        TensorRef r = catalog.get(target);
        if (r != null) {
            return r;
        }
        String prefix = TensorNames.STACK_BLK + ".";
        if (!target.startsWith(prefix)) {
            return null;
        }
        int dot = target.indexOf('.', prefix.length());
        if (dot < 0 || dot == target.length() - 1) {
            return null;
        }
        int il;
        try {
            il = Integer.parseInt(target.substring(prefix.length(), dot));
        } catch (NumberFormatException e) {
            return null;
        }
        return il < 0 ? null : catalog.layer(il, target.substring(dot + 1));
    }

    private static float[] decode(CpuWeights adapter, GGUFTensorInfo info) {
        CpuTensorRef t = adapter.find(info.name());
        if (t == null) {
            throw new GgufFormatException("LoRA tensor '" + info.name() + "' has no data");
        }
        long n = t.rows() * t.cols();
        if (n > Integer.MAX_VALUE - 8L) {
            throw new GgufFormatException("LoRA tensor '" + info.name() + "' has " + n
                    + " elements, too many for one float array");
        }
        float[] out = new float[(int) n];
        t.decodeRows(0L, (int) t.rows(), out, 0);
        return out;
    }

    private static String strip(String name, String suffix) {
        return name.substring(0, name.length() - suffix.length());
    }
}
