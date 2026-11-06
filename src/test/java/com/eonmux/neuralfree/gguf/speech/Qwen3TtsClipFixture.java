package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * A tiny synthetic qwen3tts generator file: the code predictor and code2wav tensors of
 * tools/mtmd/clip-impl.h with a geometry small enough to transcribe by hand. Values are deterministic per
 * fixture seed, so the reference implementations in the tests read exactly what {@link Qwen3TtsClip} reads.
 *
 * <p>Geometry: talker 6, predictor 8 (so the projection is exercised), 2 heads / 1 KV head of dim 4, one
 * layer, 2 codebook pages of vocabulary 4, codec-0 vocabulary 4; decoder width 4, first/rest codebook dims 3,
 * pre-transformer 4 wide with 2 heads of dim 2, one layer and a 3-frame window (the override key); one
 * upsample stage and one DAC block whose ConvTranspose narrows 4 to 3 channels, residual units of 3.
 */
public final class Qwen3TtsClipFixture {

    static final int TALKER = 6;
    static final int PROJ = 8;
    static final int PRED_HEADS = 2;
    static final int PRED_KV = 1;
    static final int PRED_DHEAD = 4;
    static final int PRED_FF = 12;
    static final int PRED_LAYERS = 1;
    static final int PAGES = 2;
    static final int VOCAB = 4;
    static final int VOCAB0 = 4;
    static final float EPS = 1e-5f;

    static final int FIRST_DIM = 3;
    static final int FIRST_VOCAB = 4;
    static final int REST_DIM = 3;
    static final int REST_VOCAB = 4;
    static final int DECODER = 4;
    static final int TFM_HEADS = 2;
    static final int TFM_KV = 2;
    static final int TFM_DHEAD = 2;
    static final int TFM_FF = 8;
    static final int TFM_LAYERS = 1;
    static final int WINDOW = 3;
    static final int UP_FF = 8;
    static final int DAC_CH = 3;
    static final int DAC_STRIDE = 2;
    static final int RES_K = 3;
    /** Samples per frame of the decoded waveform. */
    public static final int HOP = 4;

    /** Speaker-encoder geometry: 8 channels split into 8 Res2Net chunks, a 2-value squeeze bottleneck. */
    static final int SPK_CHANNELS = 8;
    static final int SPK_SE = 2;
    static final int SPK_ASP = 4;
    static final int SPK_STEM_K = 5;
    static final int SPK_RES2_K = 3;

    private final Map<String, long[]> shapes = new LinkedHashMap<>();
    private final Map<String, float[]> values = new LinkedHashMap<>();
    private final Random rng;
    private final int talker;
    private final boolean withSpeakerEncoder;

    public Qwen3TtsClipFixture(long seed) {
        this(seed, TALKER, false);
    }

    /**
     * @param talkerWidth the talker hidden width {@code proj_in} takes, i.e. the backbone's embedding width when the
     *                    fixture is paired with a backbone
     */
    public Qwen3TtsClipFixture(long seed, int talkerWidth) {
        this(seed, talkerWidth, false);
    }

    /**
     * @param speakerEncoder whether the file also carries the ECAPA-TDNN speaker encoder and its
     *                       {@code clip.audio.*} metadata
     */
    public Qwen3TtsClipFixture(long seed, int talkerWidth, boolean speakerEncoder) {
        if (talkerWidth <= 0) {
            throw new IllegalArgumentException("talker width must be positive, got " + talkerWidth);
        }
        this.talker = talkerWidth;
        this.withSpeakerEncoder = speakerEncoder;
        rng = new Random(seed);
        predictor();
        code2Wav();
        if (speakerEncoder) {
            speakerEncoder();
        }
    }

    /** Whether this fixture declares a speaker encoder. */
    public boolean hasSpeakerEncoder() {
        return withSpeakerEncoder;
    }

    /** The talker hidden width of this fixture. */
    public int talkerWidth() {
        return talker;
    }

    private void predictor() {
        random("a.gen.code.proj_in.weight", 0.5f, talker, PROJ);
        bias("a.gen.code.proj_in.bias", PROJ, 0.2f);
        random("a.gen.code.embd.weight", 0.5f, talker, VOCAB, PAGES);
        random("a.gen.code.head.weight", 0.5f, PROJ, VOCAB, PAGES);
        random("a.gen.code.out_embd.weight", 0.5f, talker, VOCAB0);
        ones("a.gen.code.output_norm.weight", PROJ);
        for (int il = 0; il < PRED_LAYERS; il++) {
            String p = "a.gen.code.blk." + il + ".";
            ones(p + "ln1.weight", PROJ);
            random(p + "attn_q.weight", 0.4f, PROJ, PRED_HEADS * PRED_DHEAD);
            ones(p + "attn_q_norm.weight", PRED_DHEAD);
            random(p + "attn_k.weight", 0.4f, PROJ, PRED_KV * PRED_DHEAD);
            ones(p + "attn_k_norm.weight", PRED_DHEAD);
            random(p + "attn_v.weight", 0.4f, PROJ, PRED_KV * PRED_DHEAD);
            random(p + "attn_out.weight", 0.4f, PRED_HEADS * PRED_DHEAD, PROJ);
            ones(p + "ln2.weight", PROJ);
            random(p + "ffn_gate.weight", 0.4f, PROJ, PRED_FF);
            random(p + "ffn_up.weight", 0.4f, PROJ, PRED_FF);
            random(p + "ffn_down.weight", 0.4f, PRED_FF, PROJ);
        }
    }

    private void code2Wav() {
        random("a.gen.wav.quant.first.codebook.weight", 0.5f, FIRST_DIM, FIRST_VOCAB);
        random("a.gen.wav.quant.first.out_proj.weight", 0.4f, FIRST_DIM, DECODER);
        random("a.gen.wav.quant.rest.codebook.weight", 0.5f, REST_DIM, REST_VOCAB, PAGES);
        random("a.gen.wav.quant.rest.out_proj.weight", 0.4f, REST_DIM, DECODER);
        random("a.gen.wav.pre_conv.weight", 0.4f, 2, DECODER, DECODER);
        bias("a.gen.wav.pre_conv.bias", DECODER, 0.2f);
        random("a.gen.wav.tfm.in_proj.weight", 0.4f, DECODER, DECODER);
        bias("a.gen.wav.tfm.in_proj.bias", DECODER, 0.2f);
        random("a.gen.wav.tfm.out_proj.weight", 0.4f, DECODER, DECODER);
        bias("a.gen.wav.tfm.out_proj.bias", DECODER, 0.2f);
        ones("a.gen.wav.tfm.output_norm.weight", DECODER);
        for (int il = 0; il < TFM_LAYERS; il++) {
            String p = "a.gen.wav.tfm.blk." + il + ".";
            ones(p + "ln1.weight", DECODER);
            random(p + "attn_q.weight", 0.4f, DECODER, TFM_HEADS * TFM_DHEAD);
            random(p + "attn_k.weight", 0.4f, DECODER, TFM_KV * TFM_DHEAD);
            random(p + "attn_v.weight", 0.4f, DECODER, TFM_KV * TFM_DHEAD);
            random(p + "attn_out.weight", 0.4f, TFM_HEADS * TFM_DHEAD, DECODER);
            ones(p + "ls1.weight", DECODER);
            ones(p + "ln2.weight", DECODER);
            random(p + "ffn_gate.weight", 0.4f, DECODER, TFM_FF);
            random(p + "ffn_up.weight", 0.4f, DECODER, TFM_FF);
            random(p + "ffn_down.weight", 0.4f, TFM_FF, DECODER);
            ones(p + "ls2.weight", DECODER);
        }
        random("a.gen.wav.up.blk.0.conv.weight", 0.4f, 2, DECODER, DECODER);
        bias("a.gen.wav.up.blk.0.conv.bias", DECODER, 0.2f);
        random("a.gen.wav.up.blk.0.dwconv.weight", 0.4f, 2, 1, DECODER);
        bias("a.gen.wav.up.blk.0.dwconv.bias", DECODER, 0.2f);
        ones("a.gen.wav.up.blk.0.norm.weight", DECODER);
        zero("a.gen.wav.up.blk.0.norm.bias", DECODER);
        random("a.gen.wav.up.blk.0.pw1.weight", 0.4f, DECODER, UP_FF);
        bias("a.gen.wav.up.blk.0.pw1.bias", UP_FF, 0.2f);
        random("a.gen.wav.up.blk.0.pw2.weight", 0.4f, UP_FF, DECODER);
        bias("a.gen.wav.up.blk.0.pw2.bias", DECODER, 0.2f);
        ones("a.gen.wav.up.blk.0.gamma", DECODER);
        random("a.gen.wav.dac.entry.weight", 0.4f, 2, DECODER, DECODER);
        bias("a.gen.wav.dac.entry.bias", DECODER, 0.2f);
        snake("a.gen.wav.dac.blk.0.snake.alpha", "a.gen.wav.dac.blk.0.snake.beta", DECODER);
        random("a.gen.wav.dac.blk.0.conv.weight", 0.4f, 4, DAC_CH, DECODER);
        bias("a.gen.wav.dac.blk.0.conv.bias", DAC_CH, 0.2f);
        snake("a.gen.wav.dac.blk.0.res.0.act1.alpha", "a.gen.wav.dac.blk.0.res.0.act1.beta", DAC_CH);
        random("a.gen.wav.dac.blk.0.res.0.conv1.weight", 0.4f, RES_K, DAC_CH, DAC_CH);
        bias("a.gen.wav.dac.blk.0.res.0.conv1.bias", DAC_CH, 0.2f);
        snake("a.gen.wav.dac.blk.0.res.0.act2.alpha", "a.gen.wav.dac.blk.0.res.0.act2.beta", DAC_CH);
        random("a.gen.wav.dac.blk.0.res.0.conv2.weight", 0.4f, 1, DAC_CH, DAC_CH);
        bias("a.gen.wav.dac.blk.0.res.0.conv2.bias", DAC_CH, 0.2f);
        snake("a.gen.wav.dac.post_snake.alpha", "a.gen.wav.dac.post_snake.beta", DAC_CH);
        random("a.gen.wav.dac.post_conv.weight", 0.4f, 2, DAC_CH, 1);
        bias("a.gen.wav.dac.post_conv.bias", 1, 0.2f);
    }

    /** The ECAPA-TDNN speaker encoder tensors of tools/mtmd/clip-impl.h:217-245. */
    private void speakerEncoder() {
        random("a.conv1d.0.weight", 0.3f, SPK_STEM_K, Qwen3TtsSpeakerEncoder.N_MEL, SPK_CHANNELS);
        bias("a.conv1d.0.bias", SPK_CHANNELS, 0.2f);
        int chunk = SPK_CHANNELS / Qwen3TtsSpeakerEncoder.RES2_SCALE;
        for (int il = 0; il < Qwen3TtsSpeakerEncoder.BLOCKS; il++) {
            String p = "a.blk." + (il + 1) + ".";
            random(p + "conv_pw1.weight", 0.3f, 1, SPK_CHANNELS, SPK_CHANNELS);
            bias(p + "conv_pw1.bias", SPK_CHANNELS, 0.2f);
            random(p + "conv_pw2.weight", 0.3f, 1, SPK_CHANNELS, SPK_CHANNELS);
            bias(p + "conv_pw2.bias", SPK_CHANNELS, 0.2f);
            random(p + "se_conv1.weight", 0.3f, 1, SPK_CHANNELS, SPK_SE);
            bias(p + "se_conv1.bias", SPK_SE, 0.2f);
            random(p + "se_conv2.weight", 0.3f, 1, SPK_SE, SPK_CHANNELS);
            bias(p + "se_conv2.bias", SPK_CHANNELS, 0.2f);
            for (int xid = 0; xid < Qwen3TtsSpeakerEncoder.RES2_SCALE - 1; xid++) {
                random(p + "res2." + xid + ".weight", 0.3f, SPK_RES2_K, chunk, chunk);
                bias(p + "res2." + xid + ".bias", chunk, 0.2f);
            }
        }
        random("a.conv_out.weight", 0.3f, 1, Qwen3TtsSpeakerEncoder.BLOCKS * SPK_CHANNELS, SPK_CHANNELS);
        bias("a.conv_out.bias", SPK_CHANNELS, 0.2f);
        random("a.asp_tdnn.weight", 0.3f, 1, 3 * SPK_CHANNELS, SPK_ASP);
        bias("a.asp_tdnn.bias", SPK_ASP, 0.2f);
        random("a.asp_attn.weight", 0.3f, 1, SPK_ASP, SPK_CHANNELS);
        bias("a.asp_attn.bias", SPK_CHANNELS, 0.2f);
        random("mm.a.fc.weight", 0.3f, 1, 2 * SPK_CHANNELS, talker);
        bias("mm.a.fc.bias", talker, 0.2f);
    }

    private void random(String name, float scale, long... ne) {
        float[] v = new float[(int) elements(ne)];
        for (int i = 0; i < v.length; i++) {
            v[i] = (rng.nextFloat() * 2f - 1f) * scale;
        }
        put(name, ne, v);
    }

    private void bias(String name, int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rng.nextFloat() * 2f - 1f) * scale;
        }
        put(name, new long[]{n}, v);
    }

    private void ones(String name, int n) {
        float[] v = new float[n];
        java.util.Arrays.fill(v, 1f);
        put(name, new long[]{n}, v);
    }

    private void zero(String name, int n) {
        put(name, new long[]{n}, new float[n]);
    }

    private void snake(String alpha, String beta, int n) {
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = 0.5f + rng.nextFloat();
            b[i] = 0.1f + rng.nextFloat() * 0.4f;
        }
        put(alpha, new long[]{n}, a);
        put(beta, new long[]{n}, b);
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

    /** The raw F32 values of {@code name}, in ggml order (ne[0] fastest). */
    float[] get(String name) {
        float[] v = values.get(name);
        if (v == null) {
            throw new IllegalArgumentException("no fixture tensor " + name);
        }
        return v;
    }

    long[] ne(String name) {
        long[] ne = shapes.get(name);
        if (ne == null) {
            throw new IllegalArgumentException("no fixture tensor " + name);
        }
        return ne;
    }

    /** The fixture as a clip generator GGUF: the metadata of a qwen3tts file and every tensor. */
    public GgufTestWriter writer() {
        GgufTestWriter w = new GgufTestWriter();
        w.architecture("clip");
        w.bool("clip.has_gen_audio_encoder", true);
        w.str("clip.gen.audio.projector_type", Qwen3TtsCodePredictor.PROJECTOR_TYPE);
        w.u32("clip.gen.audio.embedding_length", PROJ);
        w.u32("clip.gen.audio.block_count", PRED_LAYERS);
        w.u32("clip.gen.audio.attention.head_count", PRED_HEADS);
        w.u32("clip.gen.audio.attention.head_count_kv", PRED_KV);
        w.u32("clip.gen.audio.feed_forward_length", PRED_FF);
        w.u32("clip.gen.audio.projection_dim", PROJ);
        w.f32("clip.gen.audio.attention.layer_norm_epsilon", EPS);
        w.u32(Qwen3TtsCode2Wav.HEAD_COUNT_KEY, TFM_HEADS);
        w.u32(Qwen3TtsCode2Wav.HEAD_COUNT_KV_KEY, TFM_KV);
        w.u32(Qwen3TtsCode2Wav.WINDOW_KEY, WINDOW);
        w.f32(Qwen3TtsCode2Wav.ROPE_THETA_KEY, 10000f);
        if (withSpeakerEncoder) {
            w.bool("clip.has_audio_encoder", true);
            w.str("clip.audio.projector_type", Qwen3TtsSpeakerEncoder.PROJECTOR_TYPE);
            w.u32("clip.audio.projection_dim", talker);
            w.u32("clip.audio.num_mel_bins", Qwen3TtsSpeakerEncoder.N_MEL);
            w.u32("clip.audio.block_count", Qwen3TtsSpeakerEncoder.BLOCKS);
            w.u32("clip.audio.embedding_length", 1536);
            w.u32("clip.audio.attention.head_count", 1);
            w.u32("clip.audio.feed_forward_length", 1536);
            w.f32("clip.audio.attention.layer_norm_epsilon", 1e-5f);
        }
        for (Map.Entry<String, long[]> e : shapes.entrySet()) {
            float[] v = values.get(e.getKey());
            w.tensorF32(e.getKey(), e.getValue(), v);
        }
        return w;
    }

    public Path write() throws IOException {
        return writer().writeTempFile();
    }
}
