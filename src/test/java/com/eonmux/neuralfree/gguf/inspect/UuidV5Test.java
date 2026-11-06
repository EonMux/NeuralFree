package com.eonmux.neuralfree.gguf.inspect;

import org.junit.Test;

import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link UuidV5} against UUIDv5's published example: the URL namespace and {@code en.wikipedia.org/wiki/Llama.cpp}
 * give llama.cpp's namespace {@code ef001206-dadc-5f6d-a15f-3359e577d4e5}, the same derivation the Python tool
 * makes with {@code uuid.uuid5(uuid.NAMESPACE_URL, 'en.wikipedia.org/wiki/Llama.cpp')}.
 */
public class UuidV5Test {

    private static final UUID NAMESPACE_URL = UUID.fromString("6ba7b811-9dad-11d1-80b4-00c04fd430c8");

    @Test
    public void llamaCppNamespaceIsTheUrlNameOfWikipedia() {
        assertEquals(UuidV5.NAMESPACE_LLAMA_CPP,
                UuidV5.of(NAMESPACE_URL, "en.wikipedia.org/wiki/Llama.cpp"));
    }

    @Test
    public void versionAndVariantBitsAreSet() {
        assertEquals(5, UuidV5.NAMESPACE_LLAMA_CPP.version());
        assertEquals(2, UuidV5.NAMESPACE_LLAMA_CPP.variant());
    }

    @Test
    public void namespaceBytesAreBigEndian() {
        byte[] bytes = UuidV5.namespaceBytes(UuidV5.NAMESPACE_LLAMA_CPP);
        assertEquals(16, bytes.length);
        assertEquals((byte) 0xef, bytes[0]);
        assertEquals((byte) 0x00, bytes[1]);
        assertEquals((byte) 0xe5, bytes[15]);
    }

    @Test
    public void aDigestThatIsTooShortIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> UuidV5.fromDigest(new byte[15]));
    }
}
