package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The mel flavors of {@link MelFrontend} against the frame arithmetic of the mtmd preprocessors: the whisper
 * 3000-frame chunks, the qwen3a 100-frame padding, the granite frame-pair stack, the lfm3a frame alignment and the
 * gemma3n 30 s windows.
 */
public class MelFrontendTest {

    private static MelFrontend.Config cfg(int nMel, int nFft, int window, int hop) {
        return MelFrontend.Config.builder(nMel, 16000, nFft, window, hop).build();
    }

    private static AudioClip sine(int seconds, int extraFrames) {
        int n = seconds * 16000 + extraFrames;
        float[] s = new float[n];
        for (int i = 0; i < n; i++) {
            s[i] = (float) Math.sin(2.0 * Math.PI * 220 * i / 16000.0);
        }
        return new AudioClip(s, 16000);
    }

    @Test
    public void whisperChunksAre3000Frames() {
        // 10 s gives one 3000-frame chunk (the padded tail is dropped); 30 s gives two
        List<MelChunk> one = MelFrontend.chunks(MelFrontend.Flavor.WHISPER, sine(10, 0),
                cfg(128, 400, 400, 160));
        assertEquals(1, one.size());
        assertEquals(3000, one.get(0).nLen());
        assertEquals(128, one.get(0).nMel());
        List<MelChunk> two = MelFrontend.chunks(MelFrontend.Flavor.WHISPER, sine(30, 0),
                cfg(128, 400, 400, 160));
        assertEquals(2, two.size());
        for (MelChunk chunk : two) {
            assertEquals(3000, chunk.nLen());
        }
    }

    @Test
    public void whisperClampFloorsAtMaxMinusEight() {
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.WHISPER, sine(1, 0),
                cfg(80, 400, 400, 160));
        MelChunk chunk = chunks.get(0);
        float min = Float.MAX_VALUE;
        for (float v : chunk.data()) {
            min = Math.min(min, v);
        }
        // the audio ends after 1 s, so the chunk's late frames come from the zero tail; those were clamped to
        // (full max - 8) before the (v + 4) / 4 scale, so they carry the chunk minimum
        assertEquals(min, chunk.data()[0 * chunk.nLen() + 2900], 1e-6);
        assertTrue(min < 0.0f);
    }

    @Test
    public void qwen3aWindowsPadToChunkMultiples() {
        // 1 s of audio: center-padded framing gives 101 frames, one 800-frame window padded to 2 x 100
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.QWEN3A, sine(1, 0), cfg(128, 400, 400, 160));
        assertEquals(1, chunks.size());
        assertEquals(200, chunks.get(0).nLen());
        // 8 s gives 801 effective frames -> windows of 800 and 1, the latter padded to 100
        List<MelChunk> eight = MelFrontend.chunks(MelFrontend.Flavor.QWEN3A, sine(8, 0), cfg(128, 400, 400, 160));
        assertEquals(2, eight.size());
        assertEquals(800, eight.get(0).nLen());
        assertEquals(100, eight.get(1).nLen());
        // 16 s: 1601 frames -> windows of 800, 800 and 1->100
        List<MelChunk> sixteen = MelFrontend.chunks(MelFrontend.Flavor.QWEN3A, sine(16, 0),
                cfg(128, 400, 400, 160));
        assertEquals(3, sixteen.size());
        assertEquals(800, sixteen.get(0).nLen());
        assertEquals(800, sixteen.get(1).nLen());
        assertEquals(100, sixteen.get(2).nLen());
    }

    @Test
    public void graniteStacksFramePairsIntoDoubleBins() {
        // 0.5 s: reflect-padded framing gives 51 frames; the odd last frame drops and pairs stack to 25
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.GRANITE_SPEECH, sine(0, 8000),
                cfg(160, 512, 400, 160));
        assertEquals(1, chunks.size());
        MelChunk chunk = chunks.get(0);
        assertEquals(160, chunk.nMel());
        assertEquals(25, chunk.nLen());
        for (float v : chunk.data()) {
            assertTrue(Double.isFinite(v));
        }
        // the stacked second half mirrors the odd frames, so the halves are not equal
        boolean differs = false;
        for (int t = 0; t < chunk.nLen(); t++) {
            if (chunk.data()[t] != chunk.data()[80 * chunk.nLen() + t]) {
                differs = true;
            }
        }
        assertTrue(differs);
    }

    @Test
    public void lfm3aPadsTheLastFrame() {
        // 1 s: padded to the next hop boundary, 99 frames, values scaled by (v + 4) / 4
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.LFM3A, sine(1, 0), cfg(128, 400, 400, 160));
        assertEquals(1, chunks.size());
        assertEquals(99, chunks.get(0).nLen());
        float min = Float.MAX_VALUE;
        for (float v : chunks.get(0).data()) {
            min = Math.min(min, v);
        }
        assertTrue("lfm3a floor should exceed -3 after the (v + 4) / 4 scale, got " + min, min > -3.0f);
    }

    @Test
    public void conformerNormalizesPerFeature() {
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.CONFORMER, sine(2, 0), cfg(128, 512, 400, 160));
        assertEquals(1, chunks.size());
        MelChunk chunk = chunks.get(0);
        int effective = 2 * 16000 / 160;
        assertTrue(chunk.nLen() > effective);
        for (int m = 0; m < 4; m++) {
            double mean = 0.0;
            for (int t = 0; t < effective; t++) {
                mean += chunk.data()[m * chunk.nLen() + t];
            }
            mean /= effective;
            assertEquals(0.0, mean, 1e-3);
        }
    }

    @Test
    public void gemma4aTrimsToPytorchFrames() {
        // 1 s chunk: pt_frames = (16000 + 160 - 321) / 160 + 1 = 99
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.GEMMA4A, sine(1, 0), cfg(128, 512, 320, 160));
        assertEquals(1, chunks.size());
        assertEquals(99, chunks.get(0).nLen());
        // 30 s: pt_frames = 2999 in one chunk
        List<MelChunk> long_ = MelFrontend.chunks(MelFrontend.Flavor.GEMMA4A, sine(30, 0), cfg(128, 512, 320, 160));
        assertEquals(1, long_.size());
        assertEquals(2999, long_.get(0).nLen());
    }

    @Test
    public void gemma3nSplitsThirtySecondWindowsPaddedTo128Samples() {
        MelFrontend.Config c = cfg(128, 1024, 512, 160);
        // 31 s: 480000 samples give (480000 - 513) / 160 + 1 = 2997 frames, the last 16000 give 97
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.GEMMA3N, sine(31, 0), c);
        assertEquals(2, chunks.size());
        assertEquals(2997, chunks.get(0).nLen());
        assertEquals(97, chunks.get(1).nLen());
        // 16100 samples pad to 16128: (16128 - 513) / 160 + 1 = 98 frames
        assertEquals(98, MelFrontend.chunks(MelFrontend.Flavor.GEMMA3N, sine(1, 100), c).get(0).nLen());
        // a clip shorter than one frame pads to one
        assertEquals(1, MelFrontend.chunks(MelFrontend.Flavor.GEMMA3N, sine(0, 100), c).get(0).nLen());
    }

    @Test
    public void gemma3nSilenceSitsOnTheMelFloor() {
        MelChunk chunk = MelFrontend.chunks(MelFrontend.Flavor.GEMMA3N, AudioClip.silent(8000, 16000),
                cfg(128, 1024, 512, 160)).get(0);
        for (float v : chunk.data()) {
            assertEquals(Math.log(1e-5), v, 1e-6);
        }
    }

    @Test
    public void parakeetNeedsGgufWindowAndFilters() {
        MelFrontend.Config c = cfg(128, 512, 400, 160);
        assertThrows(IllegalArgumentException.class,
                () -> MelFrontend.chunks(MelFrontend.Flavor.PARAKEET, sine(1, 0), c));
        int nBins = 512 / 2 + 1;
        float[] filters = new float[128 * nBins];
        float[] window = new float[400];
        for (int i = 0; i < window.length; i++) {
            window[i] = 0.5f - 0.5f * (float) Math.cos(2.0 * Math.PI * i / 400);
        }
        MelFrontend.Config full = MelFrontend.Config.builder(128, 16000, 512, 400, 160)
                .melFilterRows(filters)
                .windowVector(window)
                .build();
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.PARAKEET, sine(1, 0), full);
        assertEquals(1, chunks.size());
        // centered zero padding: n_len = n / hop + 1 = 101
        assertEquals(101, chunks.get(0).nLen());
    }

    @Test
    public void parakeetReadsTheFileFilterbankAndNormalizesEveryFrame() {
        int nMel = 8;
        int nBins = 512 / 2 + 1;
        java.util.Random r = new java.util.Random(3);
        float[] filters = new float[nMel * nBins];
        for (int i = 0; i < filters.length; i++) {
            filters[i] = r.nextFloat() * 0.05f;
        }
        float[] swapped = filters.clone();
        System.arraycopy(filters, 2 * nBins, swapped, 5 * nBins, nBins);
        System.arraycopy(filters, 5 * nBins, swapped, 2 * nBins, nBins);
        float[] window = new float[400];
        java.util.Arrays.fill(window, 1f);
        AudioClip clip = sine(1, 0);
        MelChunk a = parakeet(nMel, filters, window, clip);
        MelChunk b = parakeet(nMel, swapped, window, clip);
        // 16000 samples: 100 valid frames of 160, plus the frame centered past the end
        assertEquals(101, a.nLen());
        for (int t = 0; t < a.nLen(); t++) {
            assertEquals(a.value(2, t), b.value(5, t), 0f);
            assertEquals(a.value(5, t), b.value(2, t), 0f);
        }
        for (int m = 0; m < nMel; m++) {
            double mean = 0;
            for (int t = 0; t < 100; t++) {
                mean += a.value(m, t);
            }
            assertEquals(0.0, mean / 100, 1e-4);
            // the last frame shares the valid frames' statistics instead of being zeroed
            assertTrue(Math.abs(a.value(m, 100)) > 1e-3);
        }
    }

    private static MelChunk parakeet(int nMel, float[] filters, float[] window, AudioClip clip) {
        MelFrontend.Config c = MelFrontend.Config.builder(nMel, 16000, 512, 400, 160)
                .melFilterRows(filters)
                .windowVector(window)
                .build();
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.PARAKEET, clip, c);
        assertEquals(1, chunks.size());
        return chunks.get(0);
    }

    @Test
    public void emptyClipsAreRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> MelFrontend.chunks(MelFrontend.Flavor.WHISPER, AudioClip.silent(0, 16000),
                        cfg(128, 400, 400, 160)));
    }

    @Test
    public void d1omniClipsToThirtySeconds() {
        List<MelChunk> chunks = MelFrontend.chunks(MelFrontend.Flavor.D1OMNI, sine(40, 0), cfg(128, 512, 400, 160));
        assertEquals(1, chunks.size());
        // frames = min(mel_len, 30 * 16000 / 160) = 3000
        assertEquals(3000, chunks.get(0).nLen());
    }
}
