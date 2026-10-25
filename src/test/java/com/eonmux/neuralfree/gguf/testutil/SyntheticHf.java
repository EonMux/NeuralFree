package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.Assume;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystemException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Writes small Hugging Face checkpoints for tests: {@code config.json}, the tokenizer files and the safetensors files
 * (one {@code model.safetensors}, or {@code model-0000i-of-0000n.safetensors} shards with a
 * {@code model.safetensors.index.json}), optionally in the Hugging Face cache layout
 * ({@code snapshots/<rev>/<file>} links to {@code blobs/<name>}). JSON text may use single quotes, which become double
 * quotes.
 */
public final class SyntheticHf {

    /** The snapshot revision of a cache layout. */
    public static final String REVISION = "0123456789abcdef0123456789abcdef01234567";

    /** A byte-level BPE tokenizer.json of 7 tokens ({@code <s>}, {@code </s>}, {@code a}, {@code b}, {@code ab}, ...). */
    public static final String GPT2_TOKENIZER = json("{'added_tokens':["
            + "{'id':0,'content':'<s>','special':true,'normalized':false},"
            + "{'id':1,'content':'</s>','special':true,'normalized':false},"
            + "{'id':5,'content':'<|begin|>','special':true,'normalized':false},"
            + "{'id':6,'content':'Ġhi','special':false,'normalized':false,'lstrip':true}],"
            + "'normalizer':null,"
            + "'pre_tokenizer':{'type':'ByteLevel','add_prefix_space':false,'trim_offsets':true,'use_regex':true},"
            + "'post_processor':null,"
            + "'decoder':{'type':'ByteLevel','add_prefix_space':true,'trim_offsets':true,'use_regex':true},"
            + "'model':{'type':'BPE','unk_token':null,'byte_fallback':false,'ignore_merges':false,"
            + "'vocab':{'<s>':0,'</s>':1,'a':2,'b':3,'ab':4},'merges':['a b']}}");

    /** The tokenizer_config.json of {@link #GPT2_TOKENIZER}. */
    public static final String GPT2_TOKENIZER_CONFIG = json("{'bos_token':'<s>','eos_token':'</s>',"
            + "'add_bos_token':false,'clean_up_tokenization_spaces':false}");

    private static final Map<StDType, String> TORCH_DTYPES = torchDtypes();

    private SyntheticHf() {
    }

    private static Map<StDType, String> torchDtypes() {
        Map<StDType, String> m = new LinkedHashMap<>();
        m.put(StDType.BF16, "bfloat16");
        m.put(StDType.F16, "float16");
        m.put(StDType.F32, "float32");
        return m;
    }

    /** {@code singleQuoted} with every single quote turned into a double quote. */
    public static String json(String singleQuoted) {
        return singleQuoted.replace('\'', '"');
    }

    /** An empty builder: no configuration, the GPT-2 tokenizer, one shard. */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * A Llama checkpoint ({@code LlamaForCausalLM}) of {@code layers} layers, hidden size {@code dim}, {@code heads}
     * query and {@code kvHeads} key/value heads, an FFN of {@code 2 * dim} and {@code vocab} embedding rows (at least
     * the 7 tokens of {@link #GPT2_TOKENIZER}); BF16 matrices and norms, values from {@link #values}.
     */
    public static Builder llama(int layers, int dim, int heads, int kvHeads, int vocab) {
        return llama(layers, dim, heads, kvHeads, vocab, StDType.BF16);
    }

    /** {@link #llama(int, int, int, int, int)} with every tensor stored as {@code dtype} (BF16, F16 or F32). */
    public static Builder llama(int layers, int dim, int heads, int kvHeads, int vocab, StDType dtype) {
        if (dtype != StDType.BF16 && dtype != StDType.F16 && dtype != StDType.F32) {
            throw new IllegalArgumentException("a synthetic Llama is stored as BF16, F16 or F32, not " + dtype);
        }
        if (layers < 1 || dim < 1 || heads < 1 || kvHeads < 1 || dim % heads != 0 || heads % kvHeads != 0
                || vocab < 7) {
            throw new IllegalArgumentException("invalid Llama geometry: layers=" + layers + ", dim=" + dim
                    + ", heads=" + heads + ", kvHeads=" + kvHeads + ", vocab=" + vocab);
        }
        int headDim = dim / heads;
        int kvDim = kvHeads * headDim;
        int ffn = 2 * dim;
        Builder b = builder().config(json("{'architectures':['LlamaForCausalLM'],'model_type':'llama',"
                + "'hidden_size':" + dim + ",'intermediate_size':" + ffn + ",'num_hidden_layers':" + layers
                + ",'num_attention_heads':" + heads + ",'num_key_value_heads':" + kvHeads
                + ",'max_position_embeddings':128,'rms_norm_eps':1e-05,'rope_theta':10000.0,"
                + "'vocab_size':" + vocab + ",'torch_dtype':'" + TORCH_DTYPES.get(dtype) + "','bos_token_id':0,"
                + "'eos_token_id':1}"));
        b.floats(dtype, "model.embed_tokens.weight", vocab, dim);
        for (int l = 0; l < layers; l++) {
            llamaLayer(b, "model.layers." + l + ".", dim, kvDim, ffn, dtype);
        }
        b.floats(dtype, "model.norm.weight", dim);
        b.floats(dtype, "lm_head.weight", vocab, dim);
        return b;
    }

    /** Adds the BF16 tensors of one Llama decoder layer under {@code prefix} (ending with a dot). */
    public static void llamaLayer(Builder b, String prefix, int dim, int kvDim, int ffn) {
        llamaLayer(b, prefix, dim, kvDim, ffn, StDType.BF16);
    }

    /** Adds the {@code dtype} tensors of one Llama decoder layer under {@code prefix} (ending with a dot). */
    public static void llamaLayer(Builder b, String prefix, int dim, int kvDim, int ffn, StDType dtype) {
        b.floats(dtype, prefix + "input_layernorm.weight", dim);
        b.floats(dtype, prefix + "self_attn.q_proj.weight", dim, dim);
        b.floats(dtype, prefix + "self_attn.k_proj.weight", kvDim, dim);
        b.floats(dtype, prefix + "self_attn.v_proj.weight", kvDim, dim);
        b.floats(dtype, prefix + "self_attn.o_proj.weight", dim, dim);
        b.floats(dtype, prefix + "post_attention_layernorm.weight", dim);
        b.floats(dtype, prefix + "mlp.gate_proj.weight", ffn, dim);
        b.floats(dtype, prefix + "mlp.up_proj.weight", ffn, dim);
        b.floats(dtype, prefix + "mlp.down_proj.weight", dim, ffn);
    }

    /** Deterministic values in [-0.75, 0.75] that differ per tensor ({@code seed}) and per element. */
    public static float[] values(int seed, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (((i * 7 + seed * 13) % 13) - 6) / 8f;
        }
        return v;
    }

    /** {@code v} as little-endian F32 bytes. */
    public static byte[] f32(float... v) {
        ByteBuffer b = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            b.putFloat(x);
        }
        return b.array();
    }

    /** {@code v} as little-endian F16 bytes (rounded to nearest; the test values are exact in F16). */
    public static byte[] f16(float... v) {
        ByteBuffer b = ByteBuffer.allocate(2 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            b.putShort((short) HalfFloats.floatToHalf(x));
        }
        return b.array();
    }

    /** {@code v} as little-endian BF16 bytes (truncated; the test values are exact in BF16). */
    public static byte[] bf16(float... v) {
        ByteBuffer b = ByteBuffer.allocate(2 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) {
            b.putShort((short) (Float.floatToRawIntBits(x) >>> 16));
        }
        return b.array();
    }

    /** {@code v} as little-endian I32 bytes. */
    public static byte[] i32(int... v) {
        ByteBuffer b = ByteBuffer.allocate(4 * v.length).order(ByteOrder.LITTLE_ENDIAN);
        for (int x : v) {
            b.putInt(x);
        }
        return b.array();
    }

    /** The checkpoint being described; {@link #write} puts it on disk. */
    public static final class Builder {
        private String config;
        private final Map<String, Tensor> tensors = new LinkedHashMap<>();
        private final Map<String, String> files = new LinkedHashMap<>();
        private int shards = 1;
        private boolean symlinkSnapshot;

        private Builder() {
            files.put("tokenizer.json", GPT2_TOKENIZER);
            files.put("tokenizer_config.json", GPT2_TOKENIZER_CONFIG);
        }

        /** The config.json text (JSON). */
        public Builder config(String json) {
            this.config = Objects.requireNonNull(json, "json");
            return this;
        }

        /** Adds or replaces the tensor {@code name}. */
        public Builder tensor(String name, StDType dtype, long[] shape, byte[] bytes) {
            tensors.put(Objects.requireNonNull(name, "name"), new Tensor(dtype, shape.clone(), bytes.clone()));
            return this;
        }

        /**
         * Adds or replaces the tensor {@code name} as the first tensor, so that the first shard holds it (each file
         * lists its tensors in name order; files come in shard order).
         */
        public Builder first(String name, StDType dtype, long[] shape, byte[] bytes) {
            Map<String, Tensor> rest = new LinkedHashMap<>(tensors);
            rest.remove(Objects.requireNonNull(name, "name"));
            tensors.clear();
            tensors.put(name, new Tensor(dtype, shape.clone(), bytes.clone()));
            tensors.putAll(rest);
            return this;
        }

        /** Adds a BF16 tensor of {@code shape} with {@link #values} seeded by the tensor's position. */
        public Builder bf16(String name, long... shape) {
            return tensor(name, StDType.BF16, shape, SyntheticHf.bf16(values(tensors.size(), count(shape))));
        }

        /** Adds a {@code dtype} (BF16, F16 or F32) tensor of {@code shape} with {@link #values}. */
        public Builder floats(StDType dtype, String name, long... shape) {
            float[] v = values(tensors.size(), count(shape));
            switch (dtype) {
                case BF16:
                    return tensor(name, dtype, shape, SyntheticHf.bf16(v));
                case F16:
                    return tensor(name, dtype, shape, SyntheticHf.f16(v));
                case F32:
                    return tensor(name, dtype, shape, SyntheticHf.f32(v));
                default:
                    throw new IllegalArgumentException("not a float dtype: " + dtype);
            }
        }

        /** Adds an F32 tensor of {@code shape} with {@link #values} seeded by the tensor's position. */
        public Builder f32(String name, long... shape) {
            return tensor(name, StDType.F32, shape, SyntheticHf.f32(values(tensors.size(), count(shape))));
        }

        /** Removes the tensor {@code name}. */
        public Builder remove(String name) {
            tensors.remove(name);
            return this;
        }

        /** The tokenizer.json text; null writes none. */
        public Builder tokenizerJson(String json) {
            return file("tokenizer.json", json);
        }

        /** The tokenizer_config.json text; null writes none. */
        public Builder tokenizerConfigJson(String json) {
            return file("tokenizer_config.json", json);
        }

        /** A text file of the model directory at {@code relPath}; null writes none. */
        public Builder file(String relPath, String content) {
            if (content == null) {
                files.remove(relPath);
            } else {
                files.put(relPath, content);
            }
            return this;
        }

        /** The number of safetensors files; more than one writes an index. */
        public Builder shards(int n) {
            if (n < 1) {
                throw new IllegalArgumentException("at least one shard, got " + n);
            }
            this.shards = n;
            return this;
        }

        /** Whether to write the Hugging Face cache layout, whose snapshot directory {@link #write} returns. */
        public Builder symlinkSnapshot(boolean v) {
            this.symlinkSnapshot = v;
            return this;
        }

        /**
         * Writes the checkpoint under {@code root} and returns the model directory: {@code root} itself, or the
         * snapshot directory of the cache layout (the test is skipped where symbolic links are unavailable).
         */
        public Path write(Path root) throws IOException {
            if (config == null) {
                throw new IllegalStateException("no config.json");
            }
            if (tensors.size() < shards) {
                throw new IllegalStateException(tensors.size() + " tensors cannot fill " + shards + " shards");
            }
            Path out = symlinkSnapshot ? root.resolve("blobs") : root;
            Files.createDirectories(out);
            List<String> written = new ArrayList<>();
            writeText(out, "config.json", config, written);
            for (Map.Entry<String, String> f : files.entrySet()) {
                writeText(out, f.getKey(), f.getValue(), written);
            }
            writeShards(out, written);
            return symlinkSnapshot ? link(root, written) : root;
        }

        private void writeShards(Path out, List<String> written) throws IOException {
            List<String> names = new ArrayList<>(tensors.keySet());
            Map<String, String> weightMap = new LinkedHashMap<>();
            int per = (names.size() + shards - 1) / shards;
            for (int s = 0; s < shards; s++) {
                String file = shards == 1 ? "model.safetensors"
                        : String.format("model-%05d-of-%05d.safetensors", s + 1, shards);
                SafetensorsWriter.Builder w = SafetensorsWriter.builder();
                for (int i = s * per; i < Math.min(names.size(), (s + 1) * per); i++) {
                    Tensor t = tensors.get(names.get(i));
                    w.add(names.get(i), t.dtype, t.shape, t.bytes);
                    weightMap.put(names.get(i), file);
                }
                w.metadata("format", "pt");
                w.build().write(out.resolve(file));
                written.add(file);
            }
            if (shards > 1) {
                StringBuilder sb = new StringBuilder("{\"metadata\":{\"total_size\":0},\"weight_map\":{");
                boolean first = true;
                for (Map.Entry<String, String> e : weightMap.entrySet()) {
                    sb.append(first ? "" : ",").append('"').append(e.getKey()).append("\":\"").append(e.getValue())
                            .append('"');
                    first = false;
                }
                writeText(out, "model.safetensors.index.json", sb.append("}}").toString(), written);
            }
        }

        private static void writeText(Path dir, String rel, String content, List<String> written) throws IOException {
            Path p = dir.resolve(rel);
            Files.createDirectories(p.getParent());
            Files.write(p, content.getBytes(StandardCharsets.UTF_8));
            written.add(rel);
        }

        /** Links {@code snapshots/<rev>/<file>} to {@code ../../blobs/<file>} for every written file. */
        private static Path link(Path root, List<String> written) throws IOException {
            Path snapshot = root.resolve("snapshots").resolve(REVISION);
            Files.createDirectories(snapshot);
            for (String rel : written) {
                Path at = snapshot.resolve(rel);
                Files.createDirectories(at.getParent());
                Path target = Paths.get("..", "..");
                for (int i = 1; i < Paths.get(rel).getNameCount(); i++) {
                    target = target.resolve("..");
                }
                try {
                    Files.createSymbolicLink(at, target.resolve("blobs").resolve(rel));
                } catch (UnsupportedOperationException | FileSystemException e) {
                    Assume.assumeNoException("symbolic links are not available here", e);
                }
            }
            return snapshot;
        }

        private static int count(long[] shape) {
            long n = 1;
            for (long d : shape) {
                n *= d;
            }
            return Math.toIntExact(n);
        }
    }

    /** One tensor to write. */
    private static final class Tensor {
        final StDType dtype;
        final long[] shape;
        final byte[] bytes;

        Tensor(StDType dtype, long[] shape, byte[] bytes) {
            this.dtype = Objects.requireNonNull(dtype, "dtype");
            this.shape = shape;
            this.bytes = bytes;
        }
    }
}
