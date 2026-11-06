package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.media.MediaConnector;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.mm.EncoderCacheConfig;
import com.eonmux.neuralfree.gguf.mm.EncoderCacheConnector;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The encoder cache transfer of a disaggregated deployment over {@link ChatMedia}: the encoder server (an
 * {@code ec_producer}) publishes the encoder output of a media part and the decode server (an {@code ec_consumer})
 * appends the same item from its uuid alone, without the item's bytes and without running its encoder.
 */
class EncoderCacheTransferTest {

    private static final int WIDTH = 16;

    @TempDir
    Path dir;

    private static String dataUrl() throws IOException {
        return "data:image/png;base64,"
                + Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(8, 8));
    }

    /** A llava projector sharing its encoder cache through {@code shared} as {@code role}. */
    private static ChatMedia open(Path shared, EncoderCacheConnector.Role role) throws IOException {
        MediaConnector.Options connector = MediaConnector.Options.builder()
                .allowedDomains(List.of("localhost")).build();
        return ChatMedia.open(MediaOptions.of(MmprojTestFiles.llava(WIDTH).toString()).withConnector(connector)
                .withEncoder(MmprojOptions.cpu())
                .withEncoderCache(EncoderCacheConfig.of(shared, role)), WIDTH, false, t -> false);
    }

    /** An {@code image_url} part of {@code url} (null for none) with {@code uuid}. */
    private static Map<String, Object> image(String url, String uuid) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "image_url");
        part.put("image_url", url == null ? null : Map.of("url", url));
        part.put("uuid", uuid);
        return part;
    }

    private static ChatRequest chat(Map<String, Object> part) {
        return ChatRequest.builder()
                .messages(List.of(Map.of("role", "user", "content", List.of(part))))
                .build();
    }

    private static List<MediaChunk> prompt(ChatMedia over, Map<String, Object> part) {
        List<ChatMedia.Item> items = over.items(chat(part));
        return over.prompt(ChatMessages.MEDIA_MARKER, items, List.of(), -1, EncoderCacheTransferTest::chars)
                .chunks();
    }

    private static int[] chars(String text) {
        int[] out = new int[text.length()];
        for (int i = 0; i < out.length; i++) {
            out[i] = text.charAt(i);
        }
        return out;
    }

    @Test
    void anEncoderServerPublishesWhatADecodeServerAppendsFromTheUuidAlone() throws IOException {
        Path shared = dir.resolve("shared");
        List<MediaChunk> produced;
        try (ChatMedia producer = open(shared, EncoderCacheConnector.Role.PRODUCER)) {
            produced = prompt(producer, image(dataUrl(), "cat"));
            try (Stream<Path> entries = Files.list(shared)) {
                assertEquals(1, entries.count(), "the producer published the item");
            }
        }
        try (ChatMedia consumer = open(shared, EncoderCacheConnector.Role.CONSUMER)) {
            List<MediaChunk> loaded = prompt(consumer, image(null, "cat"));
            assertEquals(produced, loaded);
        }
    }

    @Test
    void aConsumerWithoutTheEntryRefusesAUuidSentAlone() throws IOException {
        Path shared = dir.resolve("shared");
        try (ChatMedia consumer = open(shared, EncoderCacheConnector.Role.CONSUMER)) {
            MediaInputException e = assertThrows(MediaInputException.class,
                    () -> consumer.items(chat(image(null, "never-published"))));
            assertTrue(e.getMessage().contains("uuid 'never-published'"), e.getMessage());
            assertTrue(e.getMessage().contains("not in the media cache"), e.getMessage());
        }
    }

    @Test
    void aBothRoleServerServesItsOwnRepeatFromAnotherContext() throws IOException {
        Path shared = dir.resolve("shared");
        try (ChatMedia first = open(shared, EncoderCacheConnector.Role.BOTH)) {
            List<MediaChunk> produced = prompt(first, image(dataUrl(), "dog"));
            try (ChatMedia second = open(shared, EncoderCacheConnector.Role.BOTH)) {
                assertEquals(produced, prompt(second, image(null, "dog")));
            }
        }
    }
}
