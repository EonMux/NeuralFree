package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

/**
 * {@link HfFileType}: the converter's {@code LlamaFileType} of an explicit {@code --outtype}
 * ({@code add_file_type(self.ftype)}, gguf-py constants.py) and the reverse lookup from a file type to its tensor type.
 */
public class HfFileTypeTest {

    @Test
    public void anExplicitOuttypeWritesItsOwnFileType() {
        assertEquals(0, HfFileType.forOutType(OutType.F32, null));
        assertEquals(1, HfFileType.forOutType(OutType.F16, null));
        assertEquals(32, HfFileType.forOutType(OutType.BF16, null));
        assertEquals(7, HfFileType.forOutType(OutType.Q8_0, null));
        assertEquals(36, HfFileType.forOutType(OutType.TQ1_0, null));
        assertEquals(37, HfFileType.forOutType(OutType.TQ2_0, null));
        // SOURCE and AUTO name no single type and guess from the plan they need
        assertThrows(NullPointerException.class, () -> HfFileType.forOutType(OutType.SOURCE, null));
        assertThrows(NullPointerException.class, () -> HfFileType.forOutType(OutType.AUTO, null));
    }

    @Test
    public void typeOfIsTheReverseOfOf() {
        for (GGMLType t : new GGMLType[]{GGMLType.F32, GGMLType.F16, GGMLType.BF16, GGMLType.Q8_0, GGMLType.TQ1_0,
                GGMLType.TQ2_0, GGMLType.Q4_K}) {
            assertEquals(t, HfFileType.typeOf(HfFileType.of(t)));
        }
        assertNull(HfFileType.typeOf(HfFileType.GUESSED));
        assertNull(HfFileType.typeOf(5));
    }
}
