package com.eonmux.neuralfree.gguf.engine.batch;

import org.junit.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertArrayEquals;

/**
 * {@link Xxh3_128} against Python's {@code xxhash.xxh3_128_digest} (xxhash 0.16, the library vLLM's
 * {@code xxhash} hash functions call): every length class of the dispatch — 0, 1–3, 4–8, 9–16, 17–128, 129–240 and
 * the long-input accumulator loop from 241 on — over a fixed byte pattern.
 */
public class Xxh3_128Test {

    /** The digest of {@code bytes((i * 7 + 3) % 256 for i in range(n))}, keyed by {@code n}. */
    private static final String[] DIGESTS = {
            "0:99aa06d3014798d86001c324468d497f",
            "1:22bbb76b211a39ba13e608bc156defed",
            "2:a6b9dd53b14487221c9074b93943b86c",
            "3:ce31763cbf8245a5a9088dda485b481c",
            "4:47197970590746b1788a609154b0fe20",
            "8:e3bc8a5f461715553cd024e3d63a1588",
            "9:c72c88247a9a56d7eafab1c7f123109f",
            "16:ce0b9647ab24f88460d75c5e47d40a24",
            "17:bfd327edcc2fbd12eeed7654312a26d7",
            "32:4f130f27ab6baf45d761fd22e8ad5262",
            "100:2207ed96998d91f20cc97f05750182b2",
            "128:1b1962a096bac78bc580008b6c92ac53",
            "129:293e4968c4619023bd91ce7ace4d385b",
            "200:32200a52a918beaf380142cdd5843bbd",
            "240:ad46c1021b076bc704e0b5f034bee80b",
            "241:ac6c3492c3d6b45d8beadd3a8874fe17",
            "256:77f21db933350c7e3c38817f6d79c0da",
            "1000:6bcc7eff62da44c26c4f14bd97bd9e82",
            "4096:1546867423105cd5d7428746842be37e",
            "16384:9bcfdb9efa4dafd727dfea7e2deaa657",
    };

    @Test
    public void everyLengthClassDigestsAsPythonXxhashDoes() {
        for (String vector : DIGESTS) {
            int at = vector.indexOf(':');
            int n = Integer.parseInt(vector.substring(0, at));
            byte[] input = new byte[n];
            for (int i = 0; i < n; i++) {
                input[i] = (byte) ((i * 7 + 3) % 256);
            }
            assertArrayEquals("xxh3_128 of " + n + " bytes", hex(vector.substring(at + 1)),
                    Xxh3_128.digest(input));
        }
    }

    /** The digest of a non-ASCII string's UTF-8 bytes reads the same bytes Python does. */
    @Test
    public void aNonAsciiStringDigestsItsUtf8Bytes() {
        byte[] utf8 = "vllm-none-hash ✓".getBytes(StandardCharsets.UTF_8);
        assertArrayEquals(hex("81a612ab50a3db714a9ffc2c23d622a4"), Xxh3_128.digest(utf8));
    }

    private static byte[] hex(String hex) {
        byte[] out = new byte[hex.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
        }
        return out;
    }
}
