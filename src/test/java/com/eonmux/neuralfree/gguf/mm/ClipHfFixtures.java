package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Map;

/**
 * Hugging Face {@code CLIPModel} / {@code SiglipModel} checkpoints holding the values of a {@link ClipDualFixtures}
 * two-tower fixture under the checkpoint names (the inverse of the legacy converter's {@code get_tensor_name}),
 * each tensor in F32 with the torch shape (the GGUF dimensions reversed), plus the buffers the converter skips
 * ({@code position_ids}, CLIP's {@code logit_scale}). Test support of the {@code mm} tests.
 */
final class ClipHfFixtures {

    private ClipHfFixtures() {
    }

    /** The checkpoint name of a converter tensor name. */
    static String hfName(String gguf) {
        if (gguf.endsWith("_projection.weight") || gguf.startsWith("logit_")) {
            return gguf;
        }
        String tower = gguf.startsWith("t.") ? "text_model." : "vision_model.";
        String rest = gguf.substring(2);
        if (rest.startsWith("blk.")) {
            String[] parts = rest.split("\\.", 4);
            return tower + "encoder.layers." + parts[1] + "." + blockPart(parts[2]) + "." + parts[3];
        }
        String[] fixed = {
                "token_embd.", "embeddings.token_embedding.",
                "position_embd.", "embeddings.position_embedding.",
                "patch_embd.", "embeddings.patch_embedding.",
                "class_embd", "embeddings.class_embedding",
                "pre_ln.", "pre_layrnorm.",
                "head.attention.in_weight", "head.attention.in_proj_weight",
                "head.attention.in_bias", "head.attention.in_proj_bias",
                "head.attention.out.", "head.attention.out_proj.",
                "head.ln.", "head.layernorm.",
                "head.ffn_down.", "head.mlp.fc1.",
                "head.ffn_up.", "head.mlp.fc2.",
                "head.", "head.",
        };
        for (int i = 0; i < fixed.length; i += 2) {
            if (rest.startsWith(fixed[i])) {
                return tower + fixed[i + 1] + rest.substring(fixed[i].length());
            }
        }
        if (rest.startsWith("post_ln.")) {
            String norm = gguf.startsWith("t.") ? "final_layer_norm." : "post_layernorm.";
            return tower + norm + rest.substring("post_ln.".length());
        }
        throw new IllegalArgumentException("no checkpoint name for " + gguf);
    }

    private static String blockPart(String part) {
        switch (part) {
            case "ln1":
                return "layer_norm1";
            case "ln2":
                return "layer_norm2";
            case "attn_q":
                return "self_attn.q_proj";
            case "attn_k":
                return "self_attn.k_proj";
            case "attn_v":
                return "self_attn.v_proj";
            case "attn_out":
                return "self_attn.out_proj";
            case "ffn_down":
                return "mlp.fc1";
            case "ffn_up":
                return "mlp.fc2";
            default:
                throw new IllegalArgumentException("no checkpoint name for block tensor " + part);
        }
    }

    /**
     * The checkpoint of {@code f} with {@code config} as its {@code config.json}, written under {@code dir}; a
     * CLIP checkpoint also carries {@code preprocessor_config.json} with the fixture's mean and std, and the CLIP
     * tokenizer files; a SigLIP checkpoint only the {@code spiece.model} bytes {@code spiece}, when not null.
     */
    static Path write(MmprojFixture f, String config, boolean siglip, byte[] spiece, Path dir) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config);
        for (Map.Entry<String, long[]> e : f.shapes.entrySet()) {
            long[] ne = e.getValue();
            long[] shape = new long[ne.length];
            for (int i = 0; i < ne.length; i++) {
                shape[i] = ne[ne.length - 1 - i];
            }
            b.tensor(hfName(e.getKey()), StDType.F32, shape, SyntheticHf.f32(f.get(e.getKey())));
        }
        b.tensor("text_model.embeddings.position_ids", StDType.I64, new long[]{1, ClipDualFixtures.CONTEXT},
                positions(ClipDualFixtures.CONTEXT));
        if (siglip) {
            b.tokenizerJson(null).tokenizerConfigJson(null);
            if (spiece != null) {
                b.binaryFile("spiece.model", spiece);
            }
        } else {
            b.tensor("logit_scale", StDType.F32, new long[0], SyntheticHf.f32(2.6592f));
            b.file("preprocessor_config.json", "{\"image_mean\": [0.48, 0.46, 0.41],"
                    + " \"image_std\": [0.27, 0.26, 0.28]}");
            b.tokenizerJson(ClipDualFixtures.TOKENIZER_JSON);
            b.tokenizerConfigJson("{\"bos_token\": \"<|startoftext|>\", \"eos_token\": \"<|endoftext|>\","
                    + " \"unk_token\": \"<|endoftext|>\", \"pad_token\": \"<|endoftext|>\"}");
        }
        return b.write(dir);
    }

    private static byte[] positions(int n) {
        ByteBuffer buf = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            buf.putLong(i);
        }
        return buf.array();
    }

    /** The {@code config.json} of a dual encoder with the fixture's towers; {@code visionLayers} blocks of vision. */
    static String config(boolean siglip, int visionLayers) {
        String act = siglip ? "gelu_pytorch_tanh" : "quick_gelu";
        String common = "\"hidden_size\": 8, \"intermediate_size\": 12, \"num_attention_heads\": 2,"
                + " \"hidden_act\": \"" + act + "\", \"layer_norm_eps\": 1e-05";
        return "{\"architectures\": [\"" + (siglip ? "SiglipModel" : "CLIPModel") + "\"],"
                + " \"model_type\": \"" + (siglip ? "siglip" : "clip") + "\","
                + (siglip ? "" : " \"projection_dim\": 6,")
                + " \"text_config\": {" + common + ", \"num_hidden_layers\": 2, \"max_position_embeddings\": 8,"
                + " \"vocab_size\": 16},"
                + " \"vision_config\": {" + common + ", \"num_hidden_layers\": " + visionLayers + ","
                + " \"image_size\": 8, \"patch_size\": 4}}";
    }
}
