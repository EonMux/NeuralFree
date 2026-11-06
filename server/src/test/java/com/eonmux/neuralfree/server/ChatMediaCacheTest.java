package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.media.MediaConnector;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaIoOptions;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import java.io.IOException;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's media {@code uuid} over {@link ChatMedia}: a part whose uuid the encoded-media cache holds is served from
 * the cache without loading its URL, a uuid may come alone ({@code image_url} null or absent) once cached, a uuid
 * the cache does not hold must bring its data, a malformed uuid is refused by name, a zero cache budget caches
 * nothing, a released projector refuses new encodes as the sleeping engine does while cached encodings serve, the
 * chunks carry the item key the cache counts its queries and hits under, and the server's loader options
 * ({@code --media-io-kwargs}) shape the decode and hash into every image key.
 */
class ChatMediaCacheTest {

    private static final int WIDTH = 16;
    /** A URL the connector refuses to load: loading it fails the request, so a served part proves no load ran. */
    private static final String UNLOADABLE = "http://forbidden.example/cat.png";

    private static String dataUrl;
    private static String otherDataUrl;
    private static ChatMedia media;

    @BeforeAll
    static void open() throws IOException {
        dataUrl = "data:image/png;base64,"
                + Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(8, 8));
        otherDataUrl = "data:image/png;base64,"
                + Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(9, 7));
        media = open(MediaOptions.defaultCacheBytes());
    }

    @AfterAll
    static void close() throws IOException {
        media.close();
    }

    /** A llava projector whose connector loads data URLs and http(s) URLs of {@code localhost} only. */
    private static ChatMedia open(long cacheBytes) throws IOException {
        return open(cacheBytes, MediaIoOptions.defaults());
    }

    /** {@link #open(long)} loading media under the loader options {@code io}. */
    private static ChatMedia open(long cacheBytes, MediaIoOptions io) throws IOException {
        MediaConnector.Options connector = MediaConnector.Options.builder()
                .allowedDomains(List.of("localhost")).build();
        return ChatMedia.open(MediaOptions.of(MmprojTestFiles.llava(WIDTH).toString()).withConnector(connector)
                .withEncoder(MmprojOptions.cpu()).withCacheBytes(cacheBytes).withIo(io), WIDTH, false, t -> false);
    }

    /** The one media chunk of {@code chunks}. */
    private static MediaChunk mediaChunk(List<MediaChunk> chunks) {
        List<MediaChunk> media = chunks.stream().filter(c -> !c.isText()).toList();
        assertEquals(1, media.size(), "one image is one media chunk here: " + chunks);
        return media.get(0);
    }

    /** An {@code image_url} part of {@code url} (null for none) with {@code uuid} (null for none). */
    private static Map<String, Object> image(String url, Object uuid) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "image_url");
        part.put("image_url", url == null ? null : Map.of("url", url));
        if (uuid != null) {
            part.put("uuid", uuid);
        }
        return part;
    }

    private static ChatRequest chat(Object... parts) {
        return ChatRequest.builder()
                .messages(List.of(Map.of("role", "user", "content", List.of(parts))))
                .build();
    }

    /** The media chunk of a one-image prompt over {@code parts}. */
    private static List<MediaChunk> prompt(ChatMedia over, Object part) {
        List<ChatMedia.Item> items = over.items(chat(part));
        return over.prompt(ChatMessages.MEDIA_MARKER, items, List.of(), -1, ChatMediaCacheTest::chars).chunks();
    }

    private static int[] chars(String text) {
        int[] out = new int[text.length()];
        for (int i = 0; i < out.length; i++) {
            out[i] = text.charAt(i);
        }
        return out;
    }

    private static void assertRefused(Executable call, String... fragments) {
        MediaInputException e = assertThrows(MediaInputException.class, call);
        for (String fragment : fragments) {
            assertTrue(e.getMessage().contains(fragment), e.getMessage());
        }
    }

    @Test
    void aCachedUuidServesThePartWithoutLoadingItsUrl() {
        assertRefused(() -> media.items(chat(image(UNLOADABLE, null))), "forbidden.example");
        List<MediaChunk> first = prompt(media, image(dataUrl, "cat-url"));
        assertEquals(first, prompt(media, image(UNLOADABLE, "cat-url")),
                "the cached uuid serves the part; its URL would have been refused had it loaded");
    }

    @Test
    void aCachedUuidMayComeAloneWithANullOrAbsentMediaValue() {
        List<MediaChunk> first = prompt(media, image(dataUrl, "cat-alone"));
        assertEquals(first, prompt(media, image(null, "cat-alone")));
        assertEquals(first, prompt(media, Map.of("type", "image_url", "uuid", "cat-alone")));
        assertEquals(first, prompt(media, Map.of("type", "image_url", "image_url", Map.of(), "uuid", "cat-alone")));
        ChatMedia.Item item = media.items(chat(image(null, "cat-alone"))).get(0);
        assertEquals("cat-alone", item.uuid());
        assertNull(item.bytes(), "a uuid sent alone carries no data");
    }

    @Test
    void theUuidKeysTheItemOverItsBytes() {
        List<MediaChunk> first = prompt(media, image(dataUrl, "same-key"));
        assertEquals(first, prompt(media, image(otherDataUrl, "same-key")),
                "vLLM takes the client's uuid as the item's identity");
    }

    @Test
    void aUuidTheCacheDoesNotHoldMustBringItsData() {
        assertRefused(() -> media.items(chat(Map.of("type", "text", "text", "hi"), image(null, "never-sent"))),
                "messages[0].content[1].image_url", "uuid 'never-sent'", "not in the media cache");
        assertRefused(() -> media.items(chat(image(UNLOADABLE, "uncached"))), "forbidden.example");
    }

    @Test
    void aMalformedUuidIsRefusedByName() {
        assertRefused(() -> media.items(chat(image(dataUrl, Map.of("id", 1)))),
                "messages[0].content[0].image_url.uuid must be a string");
        assertRefused(() -> media.items(chat(image(dataUrl, " "))), "uuid must not be blank");
        List<MediaChunk> numbered = prompt(media, image(dataUrl, 42));
        assertEquals(numbered, prompt(media, image(null, "42")), "a numeric uuid is its text, as vLLM's str()");
    }

    @Test
    void aZeroCacheBudgetCachesNothing() throws IOException {
        try (ChatMedia uncached = open(0)) {
            prompt(uncached, image(dataUrl, "zero"));
            assertEquals(0, uncached.cachedBytes());
            assertRefused(() -> uncached.items(chat(image(null, "zero"))), "uuid 'zero'");
        }
    }

    @Test
    void aReleasedProjectorRefusesNewEncodesAsTheSleepingEngineDoes() throws IOException {
        try (ChatMedia own = open(MediaOptions.defaultCacheBytes())) {
            List<MediaChunk> cached = prompt(own, image(dataUrl, "kept"));
            own.release();
            assertTrue(own.released());
            assertTrue(own.vision(), "the capabilities stay for /props while released");
            EngineUnavailableException e = assertThrows(EngineUnavailableException.class,
                    () -> prompt(own, image(otherDataUrl, null)));
            assertTrue(e.getMessage().contains("projector is released"), e.getMessage());
            assertEquals(cached, prompt(own, image(null, "kept")), "a cached encoding needs no projector");

            own.reopen();
            assertFalse(own.released());
            prompt(own, image(otherDataUrl, null));
            own.clearCache();
            assertEquals(0, own.cachedBytes());
        }
    }

    @Test
    void theChunksCarryTheItemKeyTheCacheCountsUnder() throws IOException {
        try (ChatMedia own = open(MediaOptions.defaultCacheBytes())) {
            MediaKey uuid = mediaChunk(prompt(own, image(dataUrl, "counted"))).item();
            assertEquals("counted", uuid.id(), "a uuid without loader options is the item's id, as vLLM's mm_hash");
            assertEquals(MediaKey.Source.UUID, uuid.source());
            MediaKey content = mediaChunk(prompt(own, image(otherDataUrl, null))).item();
            assertEquals(MediaKey.Source.CONTENT, content.source());
            assertEquals(content, mediaChunk(prompt(own, image(otherDataUrl, null))).item());
            assertEquals(3, own.cacheQueries(), "one query per item");
            assertEquals(1, own.cacheHits(), "the repeated item hits");
        }
    }

    @Test
    void theServersLoaderOptionsShapeTheDecodeAndKeyTheItems() throws IOException {
        MediaIoOptions gray = MediaIoOptions.parse("{\"image\": {\"image_mode\": \"L\"}}");
        try (ChatMedia rgb = open(MediaOptions.defaultCacheBytes());
             ChatMedia luma = open(MediaOptions.defaultCacheBytes(), gray)) {
            MediaChunk plain = mediaChunk(prompt(rgb, image(dataUrl, "io-cat")));
            MediaChunk grayed = mediaChunk(prompt(luma, image(dataUrl, "io-cat")));
            assertEquals("io-cat", plain.item().id());
            assertNotEquals(plain.item(), grayed.item(), "the options are hash factors of the uuid");
            assertTrue(grayed.item().id().matches("[0-9a-f]{64}"), grayed.item().id());
            assertEquals(grayed.item(), mediaChunk(prompt(luma, image(null, "io-cat"))).item(),
                    "the same uuid under the same options names the cached item");
            assertFalse(Arrays.deepEquals(plain.embeddings(), grayed.embeddings()),
                    "a grayscale decode encodes another image");
            assertNotEquals(mediaChunk(prompt(rgb, image(otherDataUrl, null))).item(),
                    mediaChunk(prompt(luma, image(otherDataUrl, null))).item(),
                    "the options are hash factors of the content digest too");
        }
    }

    @Test
    void withoutAProjectorTheLifecycleIsANoOp() throws IOException {
        try (ChatMedia none = ChatMedia.disabled()) {
            none.release();
            none.reopen();
            none.clearCache();
            assertFalse(none.released());
            assertEquals(0, none.cachedBytes());
        }
    }
}
