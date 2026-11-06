package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hf.fixtures.HfSpecialFixturesT5;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Synthetic CLIP two-tower GGUF files in the layout of llama.cpp's legacy {@code convert_image_encoder_to_gguf.py}
 * (the {@code t.*} / {@code v.*} tensors, the swapped {@code ffn_up} / {@code ffn_down} names, the token list
 * without merges) and a CLIP-format {@code tokenizer.json} over the same tiny vocabulary. The text tower is 8
 * wide with 2 heads, 2 blocks and 8 positions over 16 tokens; the vision tower reads an 8x8 tile of 4x4 patches
 * (5 positions with the class token); both project to {@link #DIM}. Test support shared with the server tests.
 */
public final class ClipDualFixtures {

    /** The embedding width of both towers. */
    public static final int DIM = 6;
    /** The text positions. */
    public static final int CONTEXT = 8;
    /** The tile side of the vision tower. */
    public static final int IMAGE = 8;
    /** The vocabulary of {@link #TOKENIZER_JSON} in id order. */
    public static final String[] VOCAB = {"a", "c", "t", "d", "o", "g", "a</w>", "t</w>", "g</w>", "o</w>",
            "ca", "cat</w>", "do", "dog</w>", "<|startoftext|>", "<|endoftext|>"};
    /** The start and end token ids. */
    public static final int BOS = 14;
    public static final int EOS = 15;
    /** The pad token of the SigLIP fixtures (HF {@code SiglipTextConfig.pad_token_id}'s default). */
    static final int SIGLIP_PAD = 1;

    static final int E = 8;
    static final int FF = 12;
    static final int LAYERS = 2;
    static final int HEADS = 2;
    static final float EPS = 1e-5f;

    private ClipDualFixtures() {
    }

    /** The tensors of a two-tower file: the text tower when {@code text}, the vision tower when {@code vision}. */
    static MmprojFixture tensors(long seed, boolean text, boolean vision) {
        MmprojFixture f = new MmprojFixture(seed);
        if (text) {
            f.weight("t.token_embd.weight", E, VOCAB.length);
            f.weight("t.position_embd.weight", E, CONTEXT);
            blocks(f, "t");
            f.norm("t.post_ln.weight", "t.post_ln.bias", E);
            f.weight("text_projection.weight", E, DIM);
        }
        if (vision) {
            f.weight("v.patch_embd.weight", 4, 4, 3, E);
            f.weight("v.class_embd", E);
            f.weight("v.position_embd.weight", E, 5);
            f.norm("v.pre_ln.weight", "v.pre_ln.bias", E);
            blocks(f, "v");
            f.norm("v.post_ln.weight", "v.post_ln.bias", E);
            f.weight("visual_projection.weight", E, DIM);
        }
        return f;
    }

    /** CLIP encoder blocks under the converter's names: separate Q/K/V with biases, fc1 stored as ffn_down. */
    private static void blocks(MmprojFixture f, String prefix) {
        for (int il = 0; il < LAYERS; il++) {
            block(f, prefix, il);
        }
    }

    /** Encoder block {@code il} of the {@code prefix} tower. */
    static void block(MmprojFixture f, String prefix, int il) {
        String p = prefix + ".blk." + il + ".";
        f.norm(p + "ln1.weight", p + "ln1.bias", E);
        f.norm(p + "ln2.weight", p + "ln2.bias", E);
        for (String w : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
            f.weight(p + w + ".weight", E, E);
            f.weight(p + w + ".bias", E);
        }
        f.weight(p + "ffn_down.weight", E, FF);
        f.weight(p + "ffn_down.bias", FF);
        f.weight(p + "ffn_up.weight", FF, E);
        f.weight(p + "ffn_up.bias", E);
    }

    /**
     * The tensors of a SigLIP two-tower file under the converter's names: the text tower with its head
     * ({@code t.head}, {@code E} wide like the image side) and no projection, the vision tower with a patch bias, no
     * class token and no pre-norm, its attention pooling head ({@code v.head.*}), and the logit scale and bias.
     */
    static MmprojFixture siglipTensors(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("t.token_embd.weight", E, VOCAB.length);
        f.weight("t.position_embd.weight", E, CONTEXT);
        blocks(f, "t");
        f.norm("t.post_ln.weight", "t.post_ln.bias", E);
        f.weight("t.head.weight", E, E);
        f.weight("t.head.bias", E);
        f.weight("v.patch_embd.weight", 4, 4, 3, E);
        f.weight("v.patch_embd.bias", E);
        f.weight("v.position_embd.weight", E, 4);
        blocks(f, "v");
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        f.weight("v.head.probe", E, 1, 1);
        f.weight("v.head.attention.in_weight", E, 3 * E);
        f.weight("v.head.attention.in_bias", 3 * E);
        f.weight("v.head.attention.out.weight", E, E);
        f.weight("v.head.attention.out.bias", E);
        f.norm("v.head.ln.weight", "v.head.ln.bias", E);
        f.weight("v.head.ffn_down.weight", E, FF);
        f.weight("v.head.ffn_down.bias", FF);
        f.weight("v.head.ffn_up.weight", FF, E);
        f.weight("v.head.ffn_up.bias", E);
        f.put("logit_scale", 2.3f, 2.3f, 1);
        f.put("logit_bias", -2f, -2f, 1);
        return f;
    }

    /** The metadata of a SigLIP two-tower file: the family, no projections, tanh GELU, pad token 1. */
    static Map<String, Object> siglipMeta() {
        Map<String, Object> m = meta(true, true);
        m.put(ClipFamily.KEY, ClipFamily.SIGLIP.keyName());
        m.put("clip.text.projection_dim", 0);
        m.put("clip.vision.projection_dim", 0);
        m.put("clip.vision.image_mean", new float[]{0.5f, 0.5f, 0.5f});
        m.put("clip.vision.image_std", new float[]{0.5f, 0.5f, 0.5f});
        m.put("clip.use_gelu", true);
        m.put("tokenizer.ggml.padding_token_id", SIGLIP_PAD);
        return m;
    }

    /** The metadata of a two-tower file, without a pooling type or tokenizer model. */
    static Map<String, Object> meta(boolean text, boolean vision) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("clip.has_text_encoder", text);
        m.put("clip.has_vision_encoder", vision);
        m.put("clip.has_llava_projector", false);
        m.put("general.name", "synthetic-clip");
        if (text) {
            m.put("clip.text.context_length", CONTEXT);
            tower(m, "clip.text.");
        }
        if (vision) {
            m.put("clip.vision.image_size", IMAGE);
            m.put("clip.vision.patch_size", 4);
            m.put("clip.vision.image_mean", new float[]{0.48f, 0.46f, 0.41f});
            m.put("clip.vision.image_std", new float[]{0.27f, 0.26f, 0.28f});
            tower(m, "clip.vision.");
        }
        m.put("clip.use_gelu", false);
        return m;
    }

    private static void tower(Map<String, Object> m, String prefix) {
        m.put(prefix + "embedding_length", E);
        m.put(prefix + "feed_forward_length", FF);
        m.put(prefix + "projection_dim", DIM);
        m.put(prefix + "attention.head_count", HEADS);
        m.put(prefix + "attention.layer_norm_epsilon", EPS);
        m.put(prefix + "block_count", LAYERS);
    }

    /** The GGUF writer of a fixture: the metadata, the legacy token list when there is a text tower, the tensors. */
    static GgufTestWriter writer(MmprojFixture f, Map<String, Object> meta) {
        GgufTestWriter w = VisionFamilyFixtures.gguf(f, meta);
        if (Boolean.TRUE.equals(meta.get("clip.has_text_encoder"))) {
            w.strArray("tokenizer.ggml.tokens", VOCAB);
        }
        return w;
    }

    /** A two-tower file with both towers. */
    public static Path twoTower() throws IOException {
        return writer(tensors(31, true, true), meta(true, true)).writeTempFile();
    }

    /** A two-tower file with the vision tower only. */
    public static Path visionOnly() throws IOException {
        return writer(tensors(32, false, true), meta(false, true)).writeTempFile();
    }

    /**
     * A Hugging Face {@code SiglipModel} checkpoint of the {@link #siglipTensors SigLIP fixture} under {@code dir},
     * with the Unigram {@code spiece.model} of {@link HfSpecialFixturesT5#unigramModel()} as its tokenizer.
     */
    public static Path siglipCheckpoint(Path dir) throws IOException {
        return ClipHfFixtures.write(siglipTensors(71), ClipHfFixtures.config(true, 2), true,
                HfSpecialFixturesT5.unigramModel(), dir);
    }

    /** A directory holding a CLIP-format {@code tokenizer.json} over {@link #VOCAB}. */
    public static Path tokenizerDir() throws IOException {
        Path dir = Files.createTempDirectory("clip-tokenizer");
        dir.toFile().deleteOnExit();
        Path file = dir.resolve("tokenizer.json");
        Files.write(file, TOKENIZER_JSON.getBytes(StandardCharsets.UTF_8));
        file.toFile().deleteOnExit();
        return dir;
    }

    /** The ids the CLIP tokenizer gives {@code "A cat  DOG"}: start, a, cat, dog, end. */
    public static int[] catDogIds() {
        return new int[]{BOS, 6, 11, 13, EOS};
    }

    /** The CLIP tokenizer.json layout (openai/clip-vit-base-patch32) over the tiny vocabulary. */
    static final String TOKENIZER_JSON;

    static {
        List<String> vocab = new ArrayList<>();
        for (int i = 0; i < VOCAB.length; i++) {
            vocab.add("\"" + VOCAB[i] + "\": " + i);
        }
        TOKENIZER_JSON = "{\"version\": \"1.0\", \"truncation\": null, \"padding\": null,"
                + " \"added_tokens\": ["
                + "{\"id\": 14, \"content\": \"<|startoftext|>\", \"single_word\": false, \"lstrip\": false,"
                + " \"rstrip\": false, \"normalized\": true, \"special\": true},"
                + "{\"id\": 15, \"content\": \"<|endoftext|>\", \"single_word\": false, \"lstrip\": false,"
                + " \"rstrip\": false, \"normalized\": false, \"special\": true}],"
                + " \"normalizer\": {\"type\": \"Sequence\", \"normalizers\": [{\"type\": \"NFC\"},"
                + " {\"type\": \"Replace\", \"pattern\": {\"Regex\": \"\\\\s+\"}, \"content\": \" \"},"
                + " {\"type\": \"Lowercase\"}]},"
                + " \"pre_tokenizer\": {\"type\": \"Sequence\", \"pretokenizers\": ["
                + "{\"type\": \"Split\", \"pattern\": {\"Regex\": \"<\\\\|startoftext\\\\|>|<\\\\|endoftext\\\\|>|'s|'t|"
                + "'re|'ve|'m|'ll|'d|[\\\\p{L}]+|[\\\\p{N}]|[^\\\\s\\\\p{L}\\\\p{N}]+\"}, \"behavior\": \"Removed\","
                + " \"invert\": true},"
                + " {\"type\": \"ByteLevel\", \"add_prefix_space\": false, \"trim_offsets\": true,"
                + " \"use_regex\": false}]},"
                + " \"post_processor\": {\"type\": \"RobertaProcessing\", \"sep\": [\"<|endoftext|>\", 15],"
                + " \"cls\": [\"<|startoftext|>\", 14], \"trim_offsets\": false, \"add_prefix_space\": false},"
                + " \"decoder\": {\"type\": \"BPEDecoder\", \"suffix\": \"</w>\"},"
                + " \"model\": {\"type\": \"BPE\", \"dropout\": null, \"unk_token\": \"<|endoftext|>\","
                + " \"continuing_subword_prefix\": \"\", \"end_of_word_suffix\": \"</w>\", \"fuse_unk\": false,"
                + " \"byte_fallback\": false, \"vocab\": {" + String.join(", ", vocab) + "},"
                + " \"merges\": [\"c a\", \"ca t</w>\", \"d o\", \"do g</w>\"]}}";
    }
}
