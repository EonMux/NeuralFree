package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * {@link BlockHasher} against the values vLLM computes: the {@code NONE_HASH} per seed for every algorithm
 * ({@code init_none_hash}, the seed hashed as the string it is) and the block hashes of
 * {@code hash_block_tokens(hash_fn, parent, token_ids, extra_keys)} — plain token chains, a trailing partial
 * block, every tagged extra key, and seeds long enough for the long pickle and CBOR string forms. The reference
 * values come from vLLM's own hash functions ({@code pickle.dumps(..., protocol=HIGHEST)},
 * {@code cbor2.dumps(..., canonical=True)}, {@code xxhash.xxh3_128_digest}).
 */
public class BlockHasherTest {

    /** {@code hash_fn(seed)} per algorithm and seed: {@code algo:seed:digest}. */
    private static final String[] NONE = {
            "sha256:vllm-none-hash:066b1e8987e027138e98701c46128929378935f9b6cd1549105ae6b4f0d71198",
            "sha256:123:8508caabf15277770fec9d07df755897d40add4758a1ca33355c8dc7e8525c69",
            "sha256:0:1973e23848344dc43a988a9b478663803cfffe1243480253f9a3cf004b14aa7c",
            "sha256:deadbeef:14fccf6a8f20d48b94985a2f093455d0a06524684deb31d28037cbc8d57554ba",
            "sha256:cafecafecafecafecafecafecafecafe:767c5f325ff4d82c563f7f93bc42aa87c9a3be049b6393f3ce5f403347cf8fd8",
            "sha256_cbor:vllm-none-hash:9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0",
            "sha256_cbor:123:9bf5bb3e4bd40e48b2a893d8928a13c0289f9a74404437ffc44070135c95f290",
            "sha256_cbor:0:4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e",
            "sha256_cbor:deadbeef:163455e4fd0d6409471012223d3da451b79e44f33bee14dc0a137928a535f405",
            "sha256_cbor:cafecafecafecafecafecafecafecafe:eca9f2cc65f5c87a1663cba290b83882959980906ba061bf0a49f5409e166c6b",
            "xxhash:vllm-none-hash:d98f72b9c430d985211489af307f0229",
            "xxhash:123:c0e497e5a2b5a20079ef3915e470c163",
            "xxhash:0:6fb31cc38bba6066183f7f2104b9c051",
            "xxhash:deadbeef:ddd9b2b2fd068df5c80833dc5a7e27ec",
            "xxhash:cafecafecafecafecafecafecafecafe:cee0960b61e4eb94de5c6cf7ad9d7fc4",
            "xxhash_cbor:vllm-none-hash:a4e364baa0791a6f63bbf917db9b700d",
            "xxhash_cbor:123:1bc65fa7fe8d6593254f67eab640b26d",
            "xxhash_cbor:0:1ebe36576dcb573f26b99533a20aaeca",
            "xxhash_cbor:deadbeef:3f17c4b3089ba43ab7965b25ada0312f",
            "xxhash_cbor:cafecafecafecafecafecafecafecafe:4739c59ee17004a280f48f28b608c100",
    };

    /**
     * A chain of two full blocks then the trailing 3-token partial one, per algorithm:
     * {@code algo:first,second,partial}.
     */
    private static final String[] CHAINS = {
            "sha256:701b70154804aedf84141ee185a5899bca8c5082abb6656a123a2538a89dde89,"
                    + "f9ce8de97636ec9ae5b6aa0536621929570d984d17a61a71da2ff16fc613cba1,"
                    + "4bd1393329bda8bff11fea671771c79e38fba9c113d9dea81e1919d5962879b8",
            "sha256_cbor:76099cee84b69cd4376c34a25a5a93533a350c81bee03b044193ef6506a11f9c,"
                    + "7c91819edf5db30314871fa379962937eb9e52155d12e0f2c3124b870275e2d2,"
                    + "75f23f0d294e9486921c2164c9e8ada628d7696b2d60b8af2cdf8307353be685",
            "xxhash:0aa23b24da4cd2dc38804e86b5253998,763c79fb487d8edba9f43597d6173e47,"
                    + "3dd03920d1055d5f8b3f9fb53f90e9a8",
            "xxhash_cbor:0c0fe2f7484c1152012c77b67d1d6767,ef7b4c839fc7070b827f42ed8a677eaf,"
                    + "2c1e8a12544996075eb9c55128beeda2",
    };

    /** A fresh chain's first block under every tagged extra key, per algorithm: {@code algo:case:digest}. */
    private static final String[] EXTRA = {
            "sha256:salt:e92df6d9351bff34a17197414886e482afedfea5cb6dd3494bc8a468e29cea62",
            "sha256:lora:0ca41a916e7abd2745c2984308127eb371626e3d976136aed321a251895e7e5b",
            "sha256:mm:8e27ac612ff18a485615b60def8bed0b5ae432c1ccafa0df5c89e309f6e58bcc",
            "sha256:mm_negative:fc14854f36c17d6bc520263073592f4db7a9e4c68f59d10920909e99d25d4db4",
            "sha256:mm_two:e25f9047a75539fe4fad92c5a8eabd02bb44c6a933138b215cf3943abaf86706",
            "sha256:embeds:a9ef93dad15a9980bd5d13f41b0372c0ef9b30ebfb12d9dceb37e49c316c54df",
            "sha256:all:e1a73cb537c408f4243f1576f4000b6097a3e8636b60771435f42b0e19394f12",
            "sha256_cbor:salt:bf26a0327fa055104875a4bed1822a3d4d72e6e92198871d86320d00666b7a9d",
            "sha256_cbor:lora:7a9cdc2dba3c8caf37d6e419608c70bdfea10b2733cccd93a32f3d23b7e7c519",
            "sha256_cbor:mm:fdc92ab236167658f838897e55d1f753d4642879ce0633991bc2b6e9e7eea332",
            "sha256_cbor:mm_negative:fb35392b83ee00054d2d8b1eb5c3c393be10c4f154135d19679aebb3313fe37f",
            "sha256_cbor:mm_two:2c8690723fee1aac8883ff283764b4ef6ea77fdbb080e4adfdac686ed50c0014",
            "sha256_cbor:embeds:a57cad701f058b2331775727fce42975f764ee801e172132b0c7d2e5502c052e",
            "sha256_cbor:all:b43c7111e620799292eda4d5d4393903c9db40e05f92a186411dc5043c51995b",
            "xxhash:salt:775ff4498d6d636ba1b702c0291b0c5d",
            "xxhash:lora:5c886560ee1b379b29f92e500637b0f7",
            "xxhash:mm:3d4cb0f6df99c9e9eedf5660edd4a1e9",
            "xxhash:mm_negative:a0d5b9331c7fff0c5273acef6480c756",
            "xxhash:mm_two:77ad9d8ae193dbed0f75cc20e54c9613",
            "xxhash:embeds:64272b49e1ea8279cf7eeda67d6795ef",
            "xxhash:all:bf82c38bbd76a888d68544fc69a2ad70",
            "xxhash_cbor:salt:443abb167294cfcb75ed9333204cbd52",
            "xxhash_cbor:lora:f52afb30e74fbd950896955dc1538f37",
            "xxhash_cbor:mm:c669be30481979d48163f46870585b25",
            "xxhash_cbor:mm_negative:89a79493630e0a2e7b99b6489e46e2ce",
            "xxhash_cbor:mm_two:695bc1df381da68ab71ae64ada2e9e7d",
            "xxhash_cbor:embeds:c1b039b8de4a346f6845eb819a96794c",
            "xxhash_cbor:all:eb10e44d0d3502f017e9e4b20430753f",
    };

    /** A 300-byte seed string, the long pickle and CBOR string forms, per algorithm. */
    private static final String[] LONG_SEEDS = {
            "sha256:2b711bc62cb1e211c9d3ca5c41635662fdb84f7e1a2adac99d838824df265e0c",
            "sha256_cbor:97c60376ca7845a0f09ac0e670d3324bcc230567cb711f4aef4bb44c81a3980c",
            "xxhash:a30e8788e818bb32f300376ccbf6750b",
            "xxhash_cbor:9eb2fde68246d9b396bb2c1e3d87e2f7",
    };

    /** A 400-byte LoRA identity, the long string forms again, per algorithm. */
    private static final String[] LONG_LORA = {
            "sha256:b50cabe89b5b763dfc748b6d7223ee6270ab5fa3c95d8c0cf60d8be8b4901836",
            "sha256_cbor:73385058d72a8f687aae9dfda6647382f0530045ff0553083dd010320d03cc65",
            "xxhash:490b143e8d5a3f4bcda5a0c05f250546",
            "xxhash_cbor:24f156271bd95bc77f280970430aaa99",
    };

    /** The tokens of a chain's first and second block, the partial block's three tokens. */
    private static final int[] TOKENS_A = {5, 17, 300, 65536, 2147483647, 0, 1, 99, 254, 255, 256, 100000, 7, 15,
            63, 64};
    private static final int[] TOKENS_B = {1234, -1, 5, 6, -1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    private static final int[] TOKENS_PARTIAL = {7, 8, 9};

    @Test
    public void noneHashesMatchVllmsPerSeed() {
        for (String vector : NONE) {
            String[] parts = vector.split(":", 3);
            BlockHasher hasher = BlockHasher.of(PrefixHashAlgo.parse(parts[0], "test"), parts[1]);
            assertArrayEquals("none hash of " + parts[0] + "/" + parts[1], hex(parts[2]), hasher.noneHash());
        }
    }

    @Test
    public void blockChainsMatchVllmsHashes() {
        for (String vector : CHAINS) {
            String[] parts = vector.split(":", 2);
            PrefixHashAlgo algo = PrefixHashAlgo.parse(parts[0], "test");
            BlockHasher hasher = BlockHasher.of(algo, BlockHasher.DEFAULT_NONE_HASH_SEED);
            byte[] first = hasher.hash(null, TOKENS_A, null);
            byte[] second = hasher.hash(first, TOKENS_B, null);
            byte[] partial = hasher.hash(first, TOKENS_PARTIAL, null);
            String[] expected = parts[1].split(",");
            assertArrayEquals(algo + " first block", hex(expected[0]), first);
            assertArrayEquals(algo + " second block", hex(expected[1]), second);
            assertArrayEquals(algo + " partial block", hex(expected[2]), partial);
        }
    }

    @Test
    public void taggedExtraKeysMatchVllmsHashes() {
        byte[] embeds = sha256Of("embeds");
        for (String vector : EXTRA) {
            String[] parts = vector.split(":", 3);
            PrefixHashAlgo algo = PrefixHashAlgo.parse(parts[0], "test");
            BlockHasher hasher = BlockHasher.of(algo, BlockHasher.DEFAULT_NONE_HASH_SEED);
            List<BlockExtraKey> keys = extraKeysOf(parts[1], hexOf(embeds));
            assertArrayEquals(algo + " with the " + parts[1] + " extra key", hex(parts[2]),
                    hasher.hash(null, TOKENS_A, keys));
        }
    }

    @Test
    public void longSeedsUseTheLongStringForms() {
        String longSeed = repeat('s', 300);
        for (String vector : LONG_SEEDS) {
            String[] parts = vector.split(":", 2);
            BlockHasher hasher = BlockHasher.of(PrefixHashAlgo.parse(parts[0], "test"), longSeed);
            assertArrayEquals(parts[0] + " long seed", hex(parts[1]), hasher.noneHash());
        }
    }

    @Test
    public void longLoraIdentitiesUseTheLongStringForms() {
        String longLora = repeat('l', 400);
        for (String vector : LONG_LORA) {
            String[] parts = vector.split(":", 2);
            BlockHasher hasher = BlockHasher.of(PrefixHashAlgo.parse(parts[0], "test"),
                    BlockHasher.DEFAULT_NONE_HASH_SEED);
            assertArrayEquals(parts[0] + " long lora identity", hex(parts[1]),
                    hasher.hash(null, TOKENS_A, Collections.singletonList(BlockExtraKey.lora(longLora))));
        }
    }

    @Test
    public void anEmptyExtraKeyListHashesLikeNone() {
        BlockHasher hasher = BlockHasher.of(PrefixHashAlgo.SHA256_CBOR, BlockHasher.DEFAULT_NONE_HASH_SEED);
        assertArrayEquals(hasher.hash(null, TOKENS_A, null),
                hasher.hash(null, TOKENS_A, Collections.emptyList()));
    }

    @Test
    public void aDifferentSeedChangesEveryHash() {
        BlockHasher seeded = BlockHasher.of(PrefixHashAlgo.SHA256, "vllm-none-hash");
        BlockHasher other = BlockHasher.of(PrefixHashAlgo.SHA256, "123");
        assertArrayEquals(hex(NONE[0].split(":")[2]), seeded.noneHash());
        assertArrayEquals(hex(NONE[1].split(":")[2]), other.noneHash());
        assertFalse(Arrays.equals(seeded.hash(null, TOKENS_A, null), other.hash(null, TOKENS_A, null)));
    }

    @Test
    public void theDigestSizeFollowsTheAlgorithm() {
        assertEquals(32, BlockHasher.of(PrefixHashAlgo.SHA256, "s").noneHash().length);
        assertEquals(32, BlockHasher.of(PrefixHashAlgo.SHA256_CBOR, "s").noneHash().length);
        assertEquals(16, BlockHasher.of(PrefixHashAlgo.XXHASH, "s").noneHash().length);
        assertEquals(16, BlockHasher.of(PrefixHashAlgo.XXHASH_CBOR, "s").noneHash().length);
    }

    private static List<BlockExtraKey> extraKeysOf(String name, String embedsDigest) {
        switch (name) {
            case "salt":
                return Collections.singletonList(BlockExtraKey.salt("tenant-a"));
            case "lora":
                return Collections.singletonList(BlockExtraKey.lora("lora|1a2b*3f800000"));
            case "mm":
                return Collections.singletonList(BlockExtraKey.media("uuid-one", 0));
            case "mm_negative":
                return Collections.singletonList(BlockExtraKey.media("uuid-two", -4));
            case "mm_two":
                return Arrays.asList(BlockExtraKey.media("uuid-one", 0), BlockExtraKey.media("uuid-two", 5));
            case "embeds":
                return Collections.singletonList(BlockExtraKey.promptEmbeds(embedsDigest));
            case "all":
                return Arrays.asList(BlockExtraKey.lora("lora|1a2b*3f800000"), BlockExtraKey.media("uuid-one", 2),
                        BlockExtraKey.salt("tenant-a"), BlockExtraKey.promptEmbeds(embedsDigest));
            default:
                throw new IllegalStateException("no extra-key vector named " + name);
        }
    }

    private static byte[] sha256Of(String text) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static String repeat(char c, int n) {
        char[] chars = new char[n];
        java.util.Arrays.fill(chars, c);
        return new String(chars);
    }

    private static String hexOf(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(Character.forDigit((b >> 4) & 0xf, 16)).append(Character.forDigit(b & 0xf, 16));
        }
        return sb.toString();
    }

    private static byte[] hex(String hex) {
        byte[] out = new byte[hex.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(hex.substring(i * 2, i * 2 + 2), 16);
        }
        return out;
    }
}
