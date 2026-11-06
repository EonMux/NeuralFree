package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * A tiny synthetic pockettts generator file: the flow network and streaming mimi decoder of
 * tools/mtmd/clip-impl.h with a geometry small enough to transcribe by hand. Values are deterministic per fixture
 * seed, so the reference implementations in the tests read exactly what {@link PocketTtsClip} reads.
 *
 * <p>Geometry: a conditioning width of 8, 3-value latents, a 4-wide velocity network with 2-value time frequencies
 * and two AdaLN blocks; a 4-wide mimi decoder with one head of dim 4, one transformer layer, an FFN of 8, three
 * SEANet stages of 4 channels whose transposed convolutions stride 6, 5 and 4, and a 32-tap depthwise upsampler.
 * One latent decodes to {@value #HOP} samples.
 */
public final class PocketTtsFixtures {

    static final int EMB = 8;
    static final int LATENT = 3;
    static final int FLOW = 4;
    static final int FREQ = 2;
    static final int FLOW_BLOCKS = 2;
    static final int WIDTH = 4;
    static final int TFM_HEADS = 1;
    static final int TFM_DHEAD = 4;
    static final int TFM_FF = 8;
    static final int TFM_LAYERS = 1;
    static final int[] STAGE_STRIDES = {6, 5, 4};
    static final int UP_K = 2 * PocketTtsMimiDecoder.DOWNSAMPLE;
    static final int CONV_K = 3;
    static final int RES_K = 3;
    static final float EPS = 1e-5f;
    /** Samples one latent decodes to: {@code 16 * 6 * 5 * 4}. */
    public static final int HOP = 1920;

    private final Map<String, long[]> shapes = new LinkedHashMap<>();
    private final Map<String, float[]> values = new LinkedHashMap<>();
    private final Random rng;
    private final String variant;
    private final boolean withSpeakerEncoder;

    public PocketTtsFixtures(long seed) {
        this(seed, "english", false);
    }

    public PocketTtsFixtures(long seed, String variant) {
        this(seed, variant, false);
    }

    /** @param speakerEncoder whether the file also carries the mimi speaker encoder and its metadata */
    public PocketTtsFixtures(long seed, String variant, boolean speakerEncoder) {
        this.rng = new Random(seed);
        this.variant = variant;
        this.withSpeakerEncoder = speakerEncoder;
        flow();
        mimi();
        if (speakerEncoder) {
            speakerEncoder();
        }
    }

    /** Whether this fixture declares a speaker encoder. */
    public boolean hasSpeakerEncoder() {
        return withSpeakerEncoder;
    }

    private void flow() {
        random("a.gen.flow.input_proj.weight", LATENT, FLOW);
        bias("a.gen.flow.input_proj.bias", FLOW);
        random("a.gen.flow.cond_embd.weight", EMB, FLOW);
        bias("a.gen.flow.cond_embd.bias", FLOW);
        for (int t = 0; t < 2; t++) {
            String p = "a.gen.flow.time." + t + ".";
            float[] freqs = new float[FREQ];
            for (int i = 0; i < FREQ; i++) {
                freqs[i] = 0.3f + rng.nextFloat();
            }
            put(p + "freqs", new long[]{FREQ}, freqs);
            random(p + "up.weight", 2L * FREQ, FLOW);
            bias(p + "up.bias", FLOW);
            random(p + "down.weight", FLOW, FLOW);
            bias(p + "down.bias", FLOW);
            norm(p + "norm", FLOW);
        }
        for (int b = 0; b < FLOW_BLOCKS; b++) {
            String p = "a.gen.flow.blk." + b + ".";
            norm(p + "norm.weight", FLOW);
            bias(p + "norm.bias", FLOW);
            random(p + "up.weight", FLOW, FLOW);
            bias(p + "up.bias", FLOW);
            random(p + "down.weight", FLOW, FLOW);
            bias(p + "down.bias", FLOW);
            random(p + "ada.weight", FLOW, 3L * FLOW);
            bias(p + "ada.bias", 3 * FLOW);
        }
        random("a.gen.flow.final.ada.weight", FLOW, 2L * FLOW);
        bias("a.gen.flow.final.ada.bias", 2 * FLOW);
        random("a.gen.flow.final.proj.weight", FLOW, LATENT);
        bias("a.gen.flow.final.proj.bias", LATENT);
        random("a.gen.out_eos.weight", EMB, 1);
        bias("a.gen.out_eos.bias", 1);
        random("a.gen.input_linear.weight", LATENT, EMB);
    }

    private void mimi() {
        float[] mean = new float[LATENT];
        float[] std = new float[LATENT];
        for (int i = 0; i < LATENT; i++) {
            mean[i] = (rng.nextFloat() * 2f - 1f) * 0.3f;
            std[i] = 0.5f + rng.nextFloat() * 0.5f;
        }
        put("a.gen.emb_mean", new long[]{LATENT}, mean);
        put("a.gen.emb_std", new long[]{LATENT}, std);
        random("a.gen.wav.quant_out.weight", LATENT, WIDTH);
        random("a.gen.wav.upsample.weight", UP_K, 1, WIDTH);
        random("a.gen.wav.seanet.conv_in.weight", CONV_K, WIDTH, WIDTH);
        bias("a.gen.wav.seanet.conv_in.bias", WIDTH);
        for (int s = 0; s < STAGE_STRIDES.length; s++) {
            String p = "a.gen.wav.seanet.blk." + s + ".";
            int stride = STAGE_STRIDES[s];
            random(p + "scale_conv.weight", 2L * stride, WIDTH, WIDTH);
            bias(p + "scale_conv.bias", WIDTH);
            random(p + "res_conv1.weight", RES_K, WIDTH, WIDTH);
            bias(p + "res_conv1.bias", WIDTH);
            random(p + "res_conv2.weight", 1, WIDTH, WIDTH);
            bias(p + "res_conv2.bias", WIDTH);
        }
        random("a.gen.wav.seanet.conv_out.weight", CONV_K, WIDTH, 1);
        bias("a.gen.wav.seanet.conv_out.bias", 1);
        for (int l = 0; l < TFM_LAYERS; l++) {
            String p = "a.gen.wav.tfm.blk." + l + ".";
            norm(p + "ln1.weight", WIDTH);
            bias(p + "ln1.bias", WIDTH);
            random(p + "attn_q.weight", WIDTH, WIDTH);
            random(p + "attn_k.weight", WIDTH, WIDTH);
            random(p + "attn_v.weight", WIDTH, WIDTH);
            random(p + "attn_out.weight", WIDTH, WIDTH);
            scale(p + "ls1.weight", WIDTH);
            norm(p + "ln2.weight", WIDTH);
            bias(p + "ln2.bias", WIDTH);
            random(p + "ffn_up.weight", WIDTH, TFM_FF);
            random(p + "ffn_down.weight", TFM_FF, WIDTH);
            scale(p + "ls2.weight", WIDTH);
        }
    }

    /** The mimi speaker encoder tensors of tools/mtmd/clip-impl.h:525-532. */
    private void speakerEncoder() {
        int[] strides = {4, 5, 6};
        random("a.seanet.conv_in.weight", CONV_K, 1, WIDTH);
        bias("a.seanet.conv_in.bias", WIDTH);
        for (int s = 0; s < strides.length; s++) {
            String p = "a.seanet.blk." + s + ".";
            random(p + "scale_conv.weight", 2L * strides[s], WIDTH, WIDTH);
            bias(p + "scale_conv.bias", WIDTH);
            random(p + "res_conv1.weight", RES_K, WIDTH, WIDTH);
            bias(p + "res_conv1.bias", WIDTH);
            random(p + "res_conv2.weight", 1, WIDTH, WIDTH);
            bias(p + "res_conv2.bias", WIDTH);
        }
        random("a.seanet.conv_out.weight", CONV_K, WIDTH, WIDTH);
        bias("a.seanet.conv_out.bias", WIDTH);
        random("a.downsample.conv.weight", 2L * PocketTtsSpeakerEncoder.DOWNSAMPLE, WIDTH, LATENT);
        random("a.speaker_proj.weight", LATENT, EMB);
        for (int l = 0; l < TFM_LAYERS; l++) {
            String p = "a.blk." + l + ".";
            norm(p + "ln1.weight", WIDTH);
            bias(p + "ln1.bias", WIDTH);
            random(p + "attn_q.weight", WIDTH, WIDTH);
            random(p + "attn_k.weight", WIDTH, WIDTH);
            random(p + "attn_v.weight", WIDTH, WIDTH);
            random(p + "attn_out.weight", WIDTH, WIDTH);
            scale(p + "ls1.weight", WIDTH);
            norm(p + "ln2.weight", WIDTH);
            bias(p + "ln2.bias", WIDTH);
            random(p + "ffn_up.weight", WIDTH, TFM_FF);
            random(p + "ffn_down.weight", TFM_FF, WIDTH);
            scale(p + "ls2.weight", WIDTH);
        }
    }

    private void random(String name, long... ne) {
        int n = (int) elements(ne);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rng.nextFloat() * 2f - 1f) * 0.3f;
        }
        put(name, ne, v);
    }

    private void bias(String name, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rng.nextFloat() * 2f - 1f) * 0.2f;
        }
        put(name, new long[]{n}, v);
    }

    /** A LayerNorm weight around 1 (never zero), so the reference divides by a real variance. */
    private void norm(String name, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = 0.7f + rng.nextFloat() * 0.6f;
        }
        put(name, new long[]{n}, v);
    }

    /** A layer scale around 1. */
    private void scale(String name, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = 0.7f + rng.nextFloat() * 0.6f;
        }
        put(name, new long[]{n}, v);
    }

    private void put(String name, long[] ne, float[] v) {
        shapes.put(name, ne);
        values.put(name, v);
    }

    private static long elements(long[] ne) {
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        return n;
    }

    /** The raw F32 values of {@code name}, in ggml order ({@code ne[0]} fastest). */
    public float[] get(String name) {
        float[] v = values.get(name);
        if (v == null) {
            throw new IllegalArgumentException("no fixture tensor " + name);
        }
        return v;
    }

    public long[] ne(String name) {
        long[] ne = shapes.get(name);
        if (ne == null) {
            throw new IllegalArgumentException("no fixture tensor " + name);
        }
        return ne;
    }

    /** The fixture as a clip generator GGUF with the pockettts generator metadata. */
    public GgufTestWriter writer() {
        return writer(null);
    }

    /**
     * The fixture with {@code a.gen.out_eos.bias} replaced by one value, so a test can make the end-of-speech
     * probe fire immediately or never.
     */
    public GgufTestWriter writerWithEosBias(float eosBias) {
        return writer(eosBias);
    }

    private GgufTestWriter writer(Float eosBias) {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("clip");
        w.bool("clip.has_gen_audio_encoder", true);
        w.str("clip.gen.audio.projector_type", PocketTtsFlow.PROJECTOR_TYPE);
        w.u32("clip.gen.audio.embedding_length", WIDTH);
        w.u32("clip.gen.audio.block_count", TFM_LAYERS);
        w.u32("clip.gen.audio.attention.head_count", TFM_HEADS);
        w.u32("clip.gen.audio.attention.head_count_kv", TFM_HEADS);
        w.u32("clip.gen.audio.feed_forward_length", TFM_FF);
        w.u32("clip.gen.audio.projection_dim", EMB);
        w.f32("clip.gen.audio.attention.layer_norm_epsilon", EPS);
        w.str("clip.gen.audio.model_variant", variant);
        if (withSpeakerEncoder) {
            w.bool("clip.has_audio_encoder", true);
            w.str("clip.audio.projector_type", PocketTtsSpeakerEncoder.PROJECTOR_TYPE);
            w.u32("clip.audio.projection_dim", EMB);
            w.u32("clip.audio.num_mel_bins", 1);
            w.u32("clip.audio.block_count", TFM_LAYERS);
            w.u32("clip.audio.embedding_length", WIDTH);
            w.u32("clip.audio.feed_forward_length", TFM_FF);
            w.u32("clip.audio.attention.head_count", TFM_HEADS);
            w.f32("clip.audio.attention.layer_norm_epsilon", EPS);
        }
        for (Map.Entry<String, long[]> e : shapes.entrySet()) {
            float[] v = values.get(e.getKey());
            if (eosBias != null && e.getKey().equals("a.gen.out_eos.bias")) {
                v = new float[]{eosBias};
            }
            w.tensorF32(e.getKey(), e.getValue(), v);
        }
        return w;
    }

    public Path write() throws IOException {
        return writer().writeTempFile();
    }

    public Path writeWithEosBias(float eosBias) throws IOException {
        return writerWithEosBias(eosBias).writeTempFile();
    }
}
