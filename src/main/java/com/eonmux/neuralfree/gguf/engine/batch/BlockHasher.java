package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.kvevents.BlockExtraKey;

import java.security.SecureRandom;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * The prefix cache's block hasher, vLLM's {@code hash_block_tokens} over the hash function its
 * {@code --prefix-caching-hash-algo} selects: a block's hash is the algorithm's digest over the serialization of
 * {@code (parent_block_hash, token_ids, extra_keys)} — the parent's 32- or 16-byte digest ({@link #noneHash()} for
 * a chain's first block, vLLM's {@code NONE_HASH}), the block's token ids, and the block's extra keys in vLLM's
 * tagged shapes ({@code ("lora", identity)}, {@code ("mm", identifier, offset)}, {@code ("cache_salt", salt)},
 * {@code ("prompt_embeds", digest)}), so a router recomputing the hash from the published {@code BlockStored} data
 * gets the same bytes.
 *
 * <p>The {@code NONE_HASH} seed follows vLLM's {@code init_none_hash}: the {@code PYTHONHASHSEED} environment
 * variable when it is set, vLLM's fixed {@code vllm-none-hash} for the cryptographic algorithms, and fresh random
 * bytes for the XXH3 ones (with a warning), whose per-process seed keeps collisions unpredictable. Immutable.
 */
public final class BlockHasher {

    private static final Logger LOGGER = Logger.getLogger(BlockHasher.class.getName());

    /** vLLM's deterministic seed for the cryptographic algorithms ({@code DEFAULT_NONE_HASH_SEED}). */
    public static final String DEFAULT_NONE_HASH_SEED = "vllm-none-hash";

    private final PrefixHashAlgo algo;
    private final String seed;
    private final byte[] noneHash;

    private BlockHasher(PrefixHashAlgo algo, String seed) {
        this.algo = Objects.requireNonNull(algo, "algo");
        this.seed = Objects.requireNonNull(seed, "seed");
        // NONE_HASH = hash_fn(seed), the seed hashed as the string it is
        byte[] input;
        if (algo.pickleSerialization()) {
            PickleWriter w = new PickleWriter();
            w.string(seed);
            input = w.finish();
        } else {
            CborWriter w = new CborWriter();
            w.string(seed);
            input = w.finish();
        }
        this.noneHash = algo.digest(input);
    }

    /**
     * The hasher of {@code algo} with the seed {@code seed} (the {@code PYTHONHASHSEED} value itself, as vLLM hashes
     * the string).
     */
    public static BlockHasher of(PrefixHashAlgo algo, String seed) {
        return new BlockHasher(algo, seed);
    }

    /**
     * The hasher of {@code algo} over the environment's seed: {@code PYTHONHASHSEED} when set, else vLLM's fixed
     * default for the cryptographic algorithms and fresh random bytes for the XXH3 ones, warning that their block
     * hashes repeat across processes only under a shared {@code PYTHONHASHSEED}.
     */
    public static BlockHasher forEnvironment(PrefixHashAlgo algo) {
        String envSeed = System.getenv("PYTHONHASHSEED");
        if (envSeed != null) {
            return new BlockHasher(algo, envSeed);
        }
        if (algo.cryptographic()) {
            return new BlockHasher(algo, DEFAULT_NONE_HASH_SEED);
        }
        LOGGER.warning("Using a random per-process NONE_HASH seed because " + algo.algoName() + " is not collision"
                + " resistant. Block hashes are therefore not reproducible across processes; set PYTHONHASHSEED to"
                + " a shared value to reuse the prefix cache across instances, or use sha256.");
        byte[] random = new byte[32];
        new SecureRandom().nextBytes(random);
        StringBuilder hex = new StringBuilder(64);
        for (byte b : random) {
            hex.append(Character.forDigit((b >> 4) & 0xf, 16)).append(Character.forDigit(b & 0xf, 16));
        }
        return new BlockHasher(algo, hex.toString());
    }

    /** The algorithm. */
    public PrefixHashAlgo algo() {
        return algo;
    }

    /** The seed the {@code NONE_HASH} was derived from. */
    public String seed() {
        return seed;
    }

    /** The first block's parent hash (vLLM's {@code NONE_HASH}), {@code digestSize} bytes. */
    public byte[] noneHash() {
        return noneHash.clone();
    }

    /**
     * The hash of a block of {@code tokenIds} after the parent digest {@code parent} (null for a chain's first
     * block, which hashes over {@link #noneHash()}), the block's extra keys {@code extraKeys} — vLLM's untagged
     * event form, tagged with their source for the hash input, empty or null for none.
     */
    public byte[] hash(byte[] parent, int[] tokenIds, List<BlockExtraKey> extraKeys) {
        byte[] root = parent == null ? noneHash : parent;
        Objects.requireNonNull(tokenIds, "tokenIds");
        return algo.digest(algo.pickleSerialization() ? pickleInput(root, tokenIds, extraKeys)
                : cborInput(root, tokenIds, extraKeys));
    }

    /** The pickle serialization of {@code (parent, (token ids...), extra keys or None)}. */
    private static byte[] pickleInput(byte[] parent, int[] tokenIds, List<BlockExtraKey> extraKeys) {
        PickleWriter w = new PickleWriter();
        w.tupleOpen(3);
        w.bytes(parent);
        w.tupleOpen(tokenIds.length);
        for (int token : tokenIds) {
            w.integer(token);
        }
        w.tupleClose(tokenIds.length);
        int n = extraKeys == null ? 0 : extraKeys.size();
        if (n == 0) {
            w.nullValue();
        } else {
            w.tupleOpen(n);
            for (int i = 0; i < n; i++) {
                pickleKey(w, extraKeys.get(i));
            }
            w.tupleClose(n);
        }
        w.tupleClose(3);
        return w.finish();
    }

    /** The canonical CBOR serialization of {@code (parent, (token ids...), extra keys or null)}. */
    private static byte[] cborInput(byte[] parent, int[] tokenIds, List<BlockExtraKey> extraKeys) {
        CborWriter w = new CborWriter();
        w.array(3);
        w.bytes(parent);
        w.array(tokenIds.length);
        for (int token : tokenIds) {
            w.integer(token);
        }
        int n = extraKeys == null ? 0 : extraKeys.size();
        if (n == 0) {
            w.nullValue();
        } else {
            w.array(n);
            for (int i = 0; i < n; i++) {
                cborKey(w, extraKeys.get(i));
            }
        }
        return w.finish();
    }

    /** One extra key in vLLM's tagged pickle shape: a tuple whose first element names the source. */
    private static void pickleKey(PickleWriter w, BlockExtraKey key) {
        if (key.kind() == BlockExtraKey.Kind.MEDIA) {
            w.tupleOpen(3);
            w.tagString("mm");
            w.string(key.value());
            w.integer(key.offset());
            w.tupleClose(3);
            return;
        }
        w.tupleOpen(2);
        w.tagString(tag(key.kind()));
        pickleValue(w, key);
        w.tupleClose(2);
    }

    /** {@link #pickleKey(PickleWriter, BlockExtraKey)} in canonical CBOR. */
    private static void cborKey(CborWriter w, BlockExtraKey key) {
        if (key.kind() == BlockExtraKey.Kind.MEDIA) {
            w.array(3);
            w.string("mm");
            w.string(key.value());
            w.integer(key.offset());
            return;
        }
        w.array(2);
        w.string(tag(key.kind()));
        cborValue(w, key);
    }

    private static String tag(BlockExtraKey.Kind kind) {
        switch (kind) {
            case LORA:
                return "lora";
            case SALT:
                return "cache_salt";
            case PROMPT_EMBEDS:
                return "prompt_embeds";
            default:
                throw new IllegalStateException("no tag for the extra key kind " + kind);
        }
    }

    /** A tagged key's value: a string, or the raw digest bytes of a prompt-embedding key. */
    private static void pickleValue(PickleWriter w, BlockExtraKey key) {
        if (key.kind() == BlockExtraKey.Kind.PROMPT_EMBEDS) {
            w.bytes(hexBytes(key.value()));
        } else {
            w.string(key.value());
        }
    }

    /** {@link #pickleValue(PickleWriter, BlockExtraKey)} in canonical CBOR. */
    private static void cborValue(CborWriter w, BlockExtraKey key) {
        if (key.kind() == BlockExtraKey.Kind.PROMPT_EMBEDS) {
            w.bytes(hexBytes(key.value()));
        } else {
            w.string(key.value());
        }
    }

    /** The bytes of a hex digest. */
    static byte[] hexBytes(String hex) {
        if (hex.length() % 2 != 0) {
            throw new IllegalArgumentException("a prompt-embeds digest is an even-length hex string, got " + hex);
        }
        byte[] bytes = new byte[hex.length() / 2];
        for (int i = 0; i < bytes.length; i++) {
            int hi = Character.digit(hex.charAt(2 * i), 16);
            int lo = Character.digit(hex.charAt(2 * i + 1), 16);
            if (hi < 0 || lo < 0) {
                throw new IllegalArgumentException("a prompt-embeds digest is a hex string, got " + hex);
            }
            bytes[i] = (byte) ((hi << 4) | lo);
        }
        return bytes;
    }

    @Override
    public String toString() {
        return "BlockHasher{" + algo + ", seed=" + seed + "}";
    }
}
