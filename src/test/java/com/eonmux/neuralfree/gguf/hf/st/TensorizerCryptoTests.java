package com.eonmux.neuralfree.gguf.hf.st;

import org.junit.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The XSalsa20-Poly1305 secretbox agrees with libsodium: a vector produced by PyNaCl's
 * {@code crypto_secretbox_detached} (key bytes 0..31, nonce bytes 0..23, message bytes 1..199) decrypts to the
 * message, and a changed ciphertext is rejected before any plaintext is served.
 */
public class TensorizerCryptoTests {

    private static final byte[] KEY = sequence(0, 32);
    private static final byte[] NONCE = sequence(0, 24);
    private static final byte[] MESSAGE = sequence(1, 199);
    private static final String MAC_HEX = "dd384564223674569594ee3afee5765d";
    private static final String CIPHER_HEX =
            "5ffd3b4bc2cca518b23684326581458743b755cb993328f70eb87c8848d9bec1"
            + "a2f9d6f16096d22a90a7cbb55748847312735babc6a1eb7785251c81275fa8b"
            + "59dc2bc5d40508ea55db8ca9d873dc7d2a78181afc74879a0c062c5567f65b8"
            + "cee03867e23ae435004600ba23f591307852be51e9a754e911e4b861fd00b72"
            + "b3b1a4b8e68be1d5437a91109567a07bb1acf867fc0befb7913f984e5f17e"
            + "5c158df61486434f15e470428a2e1a5b2ca6e8288793f097d6d7d96b375237"
            + "57e6107f7b6fb3c75ddf67";

    @Test
    public void decryptsALibsodiumSecretboxVector() {
        byte[] cipher = hex(CIPHER_HEX);
        byte[] mac = hex(MAC_HEX);
        TensorizerCrypto.openDetached(KEY, NONCE, mac, cipher);
        assertArrayEquals(MESSAGE, cipher);
    }

    @Test
    public void rejectsAChangedCiphertext() {
        byte[] cipher = hex(CIPHER_HEX);
        cipher[10] ^= 1;
        try {
            TensorizerCrypto.openDetached(KEY, NONCE, hex(MAC_HEX), cipher);
            fail("a changed ciphertext must not authenticate");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("Poly1305"));
        }
    }

    @Test
    public void poly1305MatchesItsOwnTagAndRejectsAnother() {
        byte[] key = sequence(0, 32);
        byte[] message = "hello tensorizer".getBytes(StandardCharsets.UTF_8);
        byte[] tag = TensorizerCrypto.poly1305(message, message.length, key);
        assertTrue(TensorizerCrypto.constantTimeEquals(tag,
                TensorizerCrypto.poly1305(message, message.length, key)));
        message[0] ^= 1;
        assertFalse(TensorizerCrypto.constantTimeEquals(tag,
                TensorizerCrypto.poly1305(message, message.length, key)));
    }

    private static byte[] sequence(int from, int length) {
        byte[] out = new byte[length];
        for (int i = 0; i < length; i++) {
            out[i] = (byte) (from + i);
        }
        return out;
    }

    private static byte[] hex(String text) {
        byte[] out = new byte[text.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(text.substring(2 * i, 2 * i + 2), 16);
        }
        return out;
    }
}
