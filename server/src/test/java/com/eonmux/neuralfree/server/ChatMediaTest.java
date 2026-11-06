package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.gguf.media.MediaLimits.Modality;
import com.eonmux.neuralfree.gguf.mm.MmEncoderTpMode;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link ChatMedia}: media content parts are collected in message and part order and checked against the
 * per-prompt limits before anything loads, {@code multimodal_data} entries are sniffed as image or audio, the
 * prompt splices each item where its marker sits (empty text between markers adds no tokens, as in mtmd), and the
 * markers that do not pair with the items, or media without a projector, are refused by name. The vision encoder
 * follows the engine's backend unless a device is named.
 */
class ChatMediaTest {

    private static final int WIDTH = 16;
    private static String png64;
    private static ChatMedia media;

    @BeforeAll
    static void open() throws IOException {
        png64 = Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(8, 8));
        media = ChatMedia.open(MediaOptions.of(MmprojTestFiles.llava(WIDTH).toString())
                .withLimits(MediaLimits.parse("{\"image\": 2}")).withEncoder(MmprojOptions.cpu()),
                WIDTH, false, t -> false);
    }

    @AfterAll
    static void close() throws IOException {
        media.close();
    }

    private static Map<String, Object> image(String url) {
        return Map.of("type", "image_url", "image_url", Map.of("url", url));
    }

    private static ChatRequest chat(Object... parts) {
        return ChatRequest.builder()
                .messages(List.of(Map.of("role", "user", "content", List.of(parts))))
                .build();
    }

    private static void assertRefused(Executable call, String fragment) {
        MediaInputException e = assertThrows(MediaInputException.class, call);
        assertTrue(e.getMessage().contains(fragment), e.getMessage());
    }

    /** One token per character. */
    private static int[] chars(String text) {
        int[] out = new int[text.length()];
        for (int i = 0; i < out.length; i++) {
            out[i] = text.charAt(i);
        }
        return out;
    }

    @Test
    void partsLoadInOrderWithTheirFieldNames() {
        List<ChatMedia.Item> items = media.items(chat(Map.of("type", "text", "text", "hi"),
                image("data:image/png;base64," + png64)));
        assertEquals(1, items.size());
        assertEquals(Modality.IMAGE, items.get(0).modality());
        assertEquals("messages[0].content[1].image_url", items.get(0).field());
        assertArrayEquals(Base64.getDecoder().decode(png64), items.get(0).bytes());
        assertTrue(media.items(chat(Map.of("type", "text", "text", "plain"))).isEmpty());
    }

    @Test
    void theLimitIsCheckedBeforeAnyPartLoads() {
        // the third part's URL would be refused if it loaded; the limit refuses first
        assertRefused(() -> media.items(chat(image("data:image/png;base64," + png64),
                image("data:image/png;base64," + png64), image("http://unreachable.invalid/x.png"))),
                "At most 2 image(s) may be provided in one prompt.");
    }

    @Test
    void audioWithoutAnAudioEncoderIsRefusedWithLlamaCppsHint() {
        assertTrue(media.vision());
        assertFalse(media.audio());
        assertRefused(() -> media.items(chat(Map.of("type", "input_audio",
                "input_audio", Map.of("data", "AAAA", "format", "wav")))), "audio input is not supported - hint");
        assertRefused(() -> media.items(List.of("UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="),
                "multimodal_data"), "audio input is not supported");
    }

    @Test
    void withoutAProjectorEveryMediaPartIsRefused() throws IOException {
        try (ChatMedia none = ChatMedia.disabled()) {
            assertFalse(none.enabled());
            assertRefused(() -> none.items(chat(image("data:image/png;base64," + png64))),
                    "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
        }
    }

    @Test
    void multimodalDataEntriesMustBeBase64Strings() {
        assertEquals(Modality.IMAGE, media.items(List.of(png64), "prompt.multimodal_data").get(0).modality());
        assertRefused(() -> media.items(List.of(5), "prompt.multimodal_data"),
                "prompt.multimodal_data[0] must be a base64 string");
        assertRefused(() -> media.items(List.of(png64, png64, png64), "prompt.multimodal_data"), "At most 2 image(s)");
    }

    @Test
    void thePromptSplicesEachItemAtItsMarker() {
        List<ChatMedia.Item> items = media.items(List.of(png64, png64), "multimodal_data");
        String m = ChatMessages.MEDIA_MARKER;
        List<String> seen = new ArrayList<>();
        MediaPrompt prompt = media.prompt("ab" + m + m + "c", items, List.of(), 1, text -> {
            seen.add(text);
            return chars(text);
        });
        assertEquals(List.of("ab", "c"), seen);
        assertEquals(4, prompt.chunks().size());
        assertArrayEquals(new int[] {1, 'a', 'b'}, prompt.chunks().get(0).tokens());
        assertFalse(prompt.chunks().get(1).isText());
        assertFalse(prompt.chunks().get(2).isText());
        assertEquals(3 + prompt.chunks().get(1).nTokens(), prompt.chunks().get(2).positions().pos0());
        assertArrayEquals(new int[] {'c'}, prompt.chunks().get(3).tokens());
    }

    @Test
    void theSplicedItemsCarryTheirContentDigestsSoTheEngineCacheCanShareThem() {
        List<ChatMedia.Item> items = media.items(List.of(png64, png64), "multimodal_data");
        String m = ChatMessages.MEDIA_MARKER;
        MediaPrompt prompt = media.prompt(m + m, items, List.of(), -1, ChatMediaTest::chars);
        assertEquals(2, prompt.chunks().size());
        assertEquals(prompt.chunks().get(0).id(), prompt.chunks().get(1).id(),
                "the same bytes digested to the same id");
        String id = prompt.chunks().get(0).id();
        assertTrue(id.matches("image:content:[0-9a-f]{64}#0"), id);
    }

    @Test
    void embeddingRowGroupsTakeOnePositionPerRow() {
        float[][] rows = new float[3][WIDTH];
        MediaPrompt prompt = media.prompt("x" + ChatMessages.PROMPT_EMBEDS_MARKER, List.of(), Collections.<float[][]>singletonList(rows), -1,
                ChatMediaTest::chars);
        assertEquals(2, prompt.chunks().size());
        assertEquals(3, prompt.chunks().get(1).nTokens());
        assertEquals(1, prompt.chunks().get(1).positions().pos0());
    }

    @Test
    void markersThatDoNotPairWithTheItemsAreRefused() {
        List<ChatMedia.Item> one = media.items(List.of(png64), "multimodal_data");
        assertRefused(() -> media.prompt("no marker", one, List.of(), -1, ChatMediaTest::chars),
                "0 media marker(s) for 1 media item(s)");
        String m = ChatMessages.MEDIA_MARKER;
        assertRefused(() -> media.prompt(m + m, one, List.of(), -1, ChatMediaTest::chars),
                "2 media marker(s) for 1 media item(s)");
        assertRefused(() -> media.prompt("x", List.of(), Collections.<float[][]>singletonList(new float[1][WIDTH]), -1, ChatMediaTest::chars),
                "0 embedding placeholders for 1 embedding parts");
    }

    @Test
    void theVisionEncoderFollowsTheEngineBackendUnlessADeviceIsNamed() {
        MediaOptions offloaded = MediaOptions.of("mmproj.gguf");
        assertEquals(MmprojOptions.cpu().withBackend(EngineConfig.BackendMode.CPU),
                EngineService.onEngineBackend(offloaded, EngineConfig.builder()
                        .backend(EngineConfig.BackendMode.CPU).build()).encoder());
        MmprojOptions cuda = EngineService.onEngineBackend(offloaded, EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CUDA).build()).encoder();
        assertTrue(cuda.offload());
        assertEquals(EngineConfig.BackendMode.CUDA, cuda.backend());
        MediaOptions named = offloaded.withEncoder(MmprojOptions.onDevice("1"));
        assertEquals(MmprojOptions.onDevice("1"), EngineService.onEngineBackend(named, EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).build()).encoder());
    }

    @Test
    void aDataParallelEncoderTakesTheEnginesDeviceRanks() {
        MediaOptions data = MediaOptions.of("mmproj.gguf")
                .withEncoder(MmprojOptions.defaults().withTpMode(MmEncoderTpMode.DATA));
        EngineConfig twoGpus = EngineConfig.builder()
                .gpuDevices(java.util.Arrays.asList("0", "1")).build();
        MmprojOptions resolved = EngineService.onEngineBackend(data, twoGpus).encoder();
        assertEquals(MmEncoderTpMode.DATA, resolved.tpMode());
        assertEquals(java.util.Arrays.asList("0", "1"), resolved.devices());
        // an encoder with its own rank group keeps it
        MediaOptions own = data.withEncoder(data.encoder().withDevices(java.util.Collections.singletonList("2")));
        assertEquals(java.util.Collections.singletonList("2"),
                EngineService.onEngineBackend(own, twoGpus).encoder().devices());
        // weights mode never takes the ranks: the encoder runs whole on one device
        assertTrue(EngineService.onEngineBackend(MediaOptions.of("mmproj.gguf"), twoGpus).encoder().devices()
                .isEmpty());
    }
}
