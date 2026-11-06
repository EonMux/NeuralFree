package com.eonmux.neuralfree.gguf.ops.rope;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link MediaRopePositions} against the llama.cpp reference it mirrors:
 * {@code mtmd_image_tokens_get_decoder_pos} (tools/mtmd/mtmd.cpp:3078) fills the decoder position of image row
 * {@code i} as {@code t = pos0}, {@code x = pos0 + i % nx}, {@code y = pos0 + i / nx}, {@code z = 0}, and
 * {@code decode_embd_batch.set_position_mrope_2d} (tools/mtmd/mtmd-helper-common.h:101) lays them out section
 * major as {@code [t, y, x, z]}; {@code set_position_mrope_1d} and {@code set_position_normal} put
 * {@code pos0 + i} in every section; {@code mtmd_input_chunk_get_n_pos} counts {@code max(nx, ny)} positions for
 * an image and one per row otherwise. A MiniCPM-V 4.7 canvas follows {@code mtmd_canvas_decoder_pos}: the values
 * below are worked by hand from it.
 */
public class MediaRopePositionsTest {

    @Test
    public void anImageGridCarriesTheReferenceDecoderPositions() {
        // a 3 x 2 grid at base 7: 6 rows, temporal position 7 for every row
        MediaRopePositions p = MediaRopePositions.image2d(7, 3, 2);
        assertEquals(MediaRopePositions.Kind.IMAGE_2D, p.kind());
        assertEquals(6, p.nTokens());
        assertEquals(3, p.nPositions());
        int[][] s = p.sections();
        assertEquals(4, s.length);
        // section 0 = t: pos0 for every row; section 3 = z: 0 for every row
        assertArrayEquals(new int[]{7, 7, 7, 7, 7, 7}, s[0]);
        assertArrayEquals(new int[]{0, 0, 0, 0, 0, 0}, s[3]);
        // section 2 = x = pos0 + i % nx, section 1 = y = pos0 + i / nx (set_position_mrope_2d)
        assertArrayEquals(new int[]{7, 8, 9, 7, 8, 9}, s[2]);
        assertArrayEquals(new int[]{7, 7, 7, 8, 8, 8}, s[1]);
    }

    @Test
    public void imageNPosFollowsMaxNxNy() {
        assertEquals(3, MediaRopePositions.image2d(0, 3, 2).nPositions());
        assertEquals(4, MediaRopePositions.image2d(0, 2, 4).nPositions());
        assertEquals(5, MediaRopePositions.image2d(0, 5, 5).nPositions());
    }

    @Test
    public void audioRowsCarryPosPlusIInEverySection() {
        MediaRopePositions p = MediaRopePositions.audio1d(11, 4);
        assertEquals(MediaRopePositions.Kind.AUDIO_1D, p.kind());
        assertEquals(4, p.nTokens());
        assertEquals(4, p.nPositions());
        for (int s = 0; s < 4; s++) {
            assertArrayEquals(new int[]{11, 12, 13, 14}, p.section(s));
        }
    }

    @Test
    public void normalRowsCarryPosPlusIInEverySection() {
        MediaRopePositions p = MediaRopePositions.normal(2, 3);
        assertEquals(MediaRopePositions.Kind.NORMAL, p.kind());
        assertEquals(3, p.nTokens());
        assertEquals(3, p.nPositions());
        for (int s = 0; s < 4; s++) {
            assertArrayEquals(new int[]{2, 3, 4}, p.section(s));
        }
    }

    @Test
    public void accessorsReturnCopiesAndReadSingleCells() {
        MediaRopePositions p = MediaRopePositions.image2d(0, 2, 2);
        int[][] s = p.sections();
        s[0][0] = 99;
        assertEquals(0, p.position(0, 0));
        int[] t = p.section(0);
        t[1] = 99;
        assertEquals(0, p.position(0, 1));
    }

    @Test
    public void validationRefusesBadShapesByName() {
        assertThrows(IllegalArgumentException.class, () -> MediaRopePositions.image2d(-1, 2, 2));
        assertThrows(IllegalArgumentException.class, () -> MediaRopePositions.image2d(0, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> MediaRopePositions.image2d(0, 2, 0));
        assertThrows(IllegalArgumentException.class, () -> MediaRopePositions.audio1d(0, 0));
        assertThrows(IllegalArgumentException.class, () -> MediaRopePositions.normal(0, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> MediaRopePositions.image2d(0, 2, 2).position(4, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> MediaRopePositions.image2d(0, 2, 2).position(0, 4));
    }

    @Test
    public void aCanvasSpreadsTheOverviewAndPlacesEachSliceInItsCell() {
        // pos0 10 (the <image> token at 9 is the origin), a 2x1 slice grid of 2x2-token slices: a 4x2 canvas;
        // the 3x2 overview spreads its columns to round(linspace(0, 3, 3)) = 0, 2 (1.5 to even), 3
        MediaRopePositions p = MediaRopePositions.canvas(10, 2, 1, Arrays.asList(
                new MediaRopePositions.CanvasTile(3, 2, 2), new MediaRopePositions.CanvasTile(2, 2, 2),
                new MediaRopePositions.CanvasTile(2, 2, 1)));
        assertEquals(MediaRopePositions.Kind.CANVAS, p.kind());
        assertEquals(19, p.nTokens());
        assertEquals("one position per tile and per separator", 8, p.nPositions());
        assertEquals(2, p.nx());
        assertEquals(2, p.ny());
        assertArrayEquals(new int[]{10, 10, 10, 10, 10, 10, 11, 12, 13, 13, 13, 13, 14, 15, 16, 16, 16, 16, 17},
                p.section(0));
        assertArrayEquals(new int[]{9, 9, 9, 10, 10, 10, 11, 12, 9, 9, 10, 10, 14, 15, 9, 9, 10, 10, 17},
                p.section(1));
        assertArrayEquals(new int[]{9, 11, 12, 9, 11, 12, 11, 12, 9, 10, 9, 10, 14, 15, 11, 12, 11, 12, 17},
                p.section(2));
        assertArrayEquals(new int[]{9, 9, 9, 9, 9, 9, 11, 12, 9, 9, 9, 9, 14, 15, 9, 9, 9, 9, 17},
                p.section(3));
    }

    @Test
    public void aCanvasRoundsTheOverviewTiesToEvenAndAnOverviewAloneIsItsOwnCanvas() {
        // a 6x1 canvas: the 3x1 overview's columns land on round(0, 2.5, 5) = 0, 2, 5
        MediaRopePositions p = MediaRopePositions.canvas(1, 3, 1, Arrays.asList(
                new MediaRopePositions.CanvasTile(3, 1, 0), new MediaRopePositions.CanvasTile(2, 1, 0),
                new MediaRopePositions.CanvasTile(2, 1, 0), new MediaRopePositions.CanvasTile(2, 1, 0)));
        assertArrayEquals(new int[]{0, 2, 5, 0, 1, 2, 3, 4, 5}, p.section(2));
        assertArrayEquals(new int[]{1, 1, 1, 2, 2, 3, 3, 4, 4}, p.section(0));
        MediaRopePositions alone = MediaRopePositions.canvas(4, 0, 0,
                Collections.singletonList(new MediaRopePositions.CanvasTile(2, 2, 1)));
        assertArrayEquals(new int[]{4, 4, 4, 4, 5}, alone.section(0));
        assertArrayEquals(new int[]{3, 3, 4, 4, 5}, alone.section(1));
        assertArrayEquals(new int[]{3, 4, 3, 4, 5}, alone.section(2));
        assertEquals(2, alone.nPositions());
    }

    @Test
    public void aCanvasRefusesAMissingOriginAndTilesThatDoNotFillTheGrid() {
        MediaRopePositions.CanvasTile tile = new MediaRopePositions.CanvasTile(2, 2, 0);
        assertThrows(IllegalArgumentException.class,
                () -> MediaRopePositions.canvas(0, 0, 0, Collections.singletonList(tile)));
        assertThrows(IllegalArgumentException.class,
                () -> MediaRopePositions.canvas(1, 2, 1, Arrays.asList(tile, tile)));
        assertThrows(IllegalArgumentException.class,
                () -> MediaRopePositions.canvas(1, 1, 0, Arrays.asList(tile, tile)));
        assertThrows(IllegalArgumentException.class, () -> new MediaRopePositions.CanvasTile(0, 2, 0));
        assertThrows(IllegalArgumentException.class, () -> new MediaRopePositions.CanvasTile(2, 2, -1));
    }

    @Test
    public void equalContentsAreEqual() {
        assertEquals(MediaRopePositions.image2d(3, 2, 3), MediaRopePositions.image2d(3, 2, 3));
        assertNotEquals(MediaRopePositions.image2d(3, 2, 3), MediaRopePositions.image2d(4, 2, 3));
        assertNotEquals(MediaRopePositions.image2d(3, 2, 3), MediaRopePositions.audio1d(3, 6));
    }
}
