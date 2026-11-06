package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The LFM3-Omni mmproj fixture: one file carrying the LFM3-VL vision encoder and the LFM3-Audio tower, both
 * writing rows of {@code clip.vision.projection_dim} / {@code clip.audio.projection_dim} (6, the LFM3-VL
 * projector's width). The modality-specific {@code clip.vision.projector_type} / {@code clip.audio.projector_type}
 * keys name each encoder, as a mixed-modality file does. Package-private test support of the {@code mm} tests.
 */
final class Lfm3OmniFixture {

    /** The row width both projectors write. */
    static final int WIDTH = 6;

    private Lfm3OmniFixture() {
    }

    /** The fixture written as an F32 mmproj GGUF. */
    static Path write() throws IOException {
        MmprojFixture audio = Lfm3aTowerTest.fixture();
        Map<String, Object> meta = new LinkedHashMap<>(Lfm3aTowerTest.meta());
        meta.remove("clip.projector_type");
        meta.put("clip.audio.projector_type", "lfm3a");
        Map<String, Object> vision = new LinkedHashMap<>(VisionFamiliesFFixtures.lfm3vlMeta());
        vision.remove("clip.projector_type");
        vision.put("clip.vision.projector_type", "lfm3vl");
        meta.putAll(vision);

        GgufTestWriter w = VisionFamilyFixtures.gguf(audio, meta);
        MmprojFixture visionTensors = VisionFamiliesFFixtures.lfm3vl(902);
        for (Map.Entry<String, long[]> e : visionTensors.shapes.entrySet()) {
            w.tensorFromFloats(e.getKey(), GGMLType.F32, e.getValue(), visionTensors.get(e.getKey()));
        }
        return w.writeTempFile();
    }
}
