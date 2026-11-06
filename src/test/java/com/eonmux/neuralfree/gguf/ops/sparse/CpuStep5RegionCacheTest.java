package com.eonmux.neuralfree.gguf.ops.sparse;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * The Step5 region-summary cache against a direct transcription of vLLM {@code summary_update.py}: an online softmax
 * over each region's BF16 ring rows in position order, summarized as BF16 then FP8 E4M3, with the ring aliasing every
 * {@code capacity} positions.
 */
public class CpuStep5RegionCacheTest {

    private static final int PROXY = 8;
    private static final int REGION = 8;

    @Test
    public void closedRegionSummaryIsTheWeightedMeanOfItsBf16Rows() {
        int capacity = 8;
        float[] summary = new float[4 * PROXY];
        float[] ring = new float[capacity * 2 * PROXY];
        CpuStep5RegionCache cache = new CpuStep5RegionCache(REGION, capacity, PROXY, summary, 0, ring, 0);
        float[][] keys = new float[16][PROXY];
        float[][] z = new float[16][PROXY];
        Random rnd = new Random(7);
        for (int p = 0; p < 16; p++) {
            for (int d = 0; d < PROXY; d++) {
                keys[p][d] = (float) rnd.nextGaussian();
                z[p][d] = (float) rnd.nextGaussian() * 0.5f;
            }
            cache.update(p, keys[p], 0, z[p], 0);
            if (p % REGION == REGION - 1) {
                float[] expected = expected(keys, z, p - REGION + 1, p);
                for (int d = 0; d < PROXY; d++) {
                    assertEquals("region " + (p / REGION) + " dim " + d,
                            Step5Dtypes.e4m3(Step5Dtypes.bf16(expected[d])),
                            summary[(p / REGION) * PROXY + d], 1e-7f);
                }
            }
        }
    }

    @Test
    public void ringAliasesAfterItsCapacityAndRebuildsTheSummary() {
        int capacity = 8;
        float[] summary = new float[2 * PROXY];
        float[] ring = new float[capacity * 2 * PROXY];
        CpuStep5RegionCache cache = new CpuStep5RegionCache(REGION, capacity, PROXY, summary, 0, ring, 0);
        float[][] keys = new float[8][PROXY];
        float[][] z = new float[8][PROXY];
        for (int p = 0; p < 8; p++) {
            for (int d = 0; d < PROXY; d++) {
                keys[p][d] = p + d * 0.25f;
                z[p][d] = 0.1f * p - 0.05f * d;
            }
            cache.update(p, keys[p], 0, z[p], 0);
        }
        float[] first = summary.clone();
        for (int p = 0; p < 8; p++) {
            for (int d = 0; d < PROXY; d++) {
                keys[p][d] = -keys[p][d];
            }
            cache.update(p, keys[p], 0, z[p], 0);
        }
        for (int d = 0; d < PROXY; d++) {
            assertEquals("rewritten summary dim " + d,
                    Step5Dtypes.e4m3(Step5Dtypes.bf16(-first[d])), summary[d], 1e-7f);
        }
        float[] next = new float[PROXY];
        float[] nextZ = new float[PROXY];
        for (int p = 8; p < 16; p++) {
            for (int d = 0; d < PROXY; d++) {
                next[d] = 0.5f * p + d;
                nextZ[d] = 0.01f * d;
            }
            cache.update(p, next, 0, nextZ, 0);
        }
        // region 1's rows occupy the ring slots of region 0's rows; the written summary must use only its own rows
        float[][] region = new float[8][PROXY];
        float[][] regionZ = new float[8][PROXY];
        for (int i = 0; i < 8; i++) {
            for (int d = 0; d < PROXY; d++) {
                region[i][d] = 0.5f * (8 + i) + d;
                regionZ[i][d] = 0.01f * d;
            }
        }
        float[] expected = expected(region, regionZ, 0, 7);
        for (int d = 0; d < PROXY; d++) {
            assertEquals("region 1 dim " + d, Step5Dtypes.e4m3(Step5Dtypes.bf16(expected[d])),
                    summary[PROXY + d], 1e-7f);
        }
    }

    @Test
    public void refusesARingSmallerThanARegion() {
        assertThrows(IllegalArgumentException.class,
                () -> new CpuStep5RegionCache(8, 7, PROXY, new float[PROXY], 0, new float[7 * 2 * PROXY], 0));
    }

    @Test
    public void refusesARegionOutsideTheSummaryArray() {
        CpuStep5RegionCache cache = new CpuStep5RegionCache(REGION, 8, PROXY, new float[PROXY], 0,
                new float[8 * 2 * PROXY], 0);
        float[] row = new float[PROXY];
        assertThrows(IllegalArgumentException.class, () -> {
            for (int p = 0; p < 16; p++) {
                cache.update(p, row, 0, row, 0);
            }
        });
    }

    @Test
    public void summaryViewStartsAtItsOffset() {
        float[] summary = new float[2 * PROXY];
        float[] ring = new float[8 * 2 * PROXY];
        CpuStep5RegionCache cache = new CpuStep5RegionCache(REGION, 8, PROXY, summary, PROXY, ring, 0);
        float[] key = new float[PROXY];
        float[] z = new float[PROXY];
        for (int p = 0; p < 8; p++) {
            key[0] = p;
            cache.update(p, key, 0, z, 0);
        }
        assertEquals(0f, summary[0], 0f);
        assertArrayEquals(new float[]{3.5f}, new float[]{summary[PROXY]}, 1e-6f);
    }

    @Test
    public void shiftRowsKeepsThePrefixAndMovesTheTailDown() {
        int capacity = 16;
        float[] summary = new float[2 * PROXY];
        float[] ring = new float[capacity * 2 * PROXY];
        CpuStep5RegionCache cache = new CpuStep5RegionCache(REGION, capacity, PROXY, summary, 0, ring, 0);
        for (int p = 0; p < capacity; p++) {
            for (int d = 0; d < PROXY; d++) {
                cache.ringArray()[cache.ringOffset(p) + d] = p;
                cache.ringArray()[cache.ringOffset(p) + PROXY + d] = -p;
            }
        }
        // positions [4, 12) leave: position 4 takes old 12, 5 takes 13, ... 7 takes 15
        cache.shiftRows(4, 8, 8);
        for (int p = 0; p < 8; p++) {
            int source = p < 4 ? p : p + 8;
            for (int d = 0; d < PROXY; d++) {
                assertEquals("position " + p + " key dim " + d, source,
                        cache.ringArray()[cache.ringOffset(p) + d], 0f);
                assertEquals("position " + p + " z dim " + d, -source,
                        cache.ringArray()[cache.ringOffset(p) + PROXY + d], 0f);
            }
        }
    }

    @Test
    public void rebuiltSummariesUseTheRowsAtTheirNewPositions() {
        int capacity = 16;
        float[] summary = new float[2 * PROXY];
        float[] ring = new float[capacity * 2 * PROXY];
        CpuStep5RegionCache cache = new CpuStep5RegionCache(REGION, capacity, PROXY, summary, 0, ring, 0);
        float[][] keys = new float[capacity][PROXY];
        float[][] z = new float[capacity][PROXY];
        Random rnd = new Random(11);
        for (int p = 0; p < capacity; p++) {
            for (int d = 0; d < PROXY; d++) {
                keys[p][d] = (float) rnd.nextGaussian();
                z[p][d] = (float) rnd.nextGaussian() * 0.5f;
            }
            cache.update(p, keys[p], 0, z[p], 0);
        }
        cache.shiftRows(4, 8, 8);
        cache.rebuildSummaries(1);
        cache.clearSummariesFrom(1);
        float[][] movedKeys = new float[8][PROXY];
        float[][] movedZ = new float[8][PROXY];
        for (int p = 0; p < 8; p++) {
            int source = p < 4 ? p : p + 8;
            System.arraycopy(keys[source], 0, movedKeys[p], 0, PROXY);
            System.arraycopy(z[source], 0, movedZ[p], 0, PROXY);
        }
        float[] expected = expected(movedKeys, movedZ, 0, 7);
        for (int d = 0; d < PROXY; d++) {
            assertEquals("rebuilt summary dim " + d, Step5Dtypes.e4m3(Step5Dtypes.bf16(expected[d])),
                    summary[d], 0f);
        }
        for (int r = 1; r < 2; r++) {
            for (int d = 0; d < PROXY; d++) {
                assertEquals("region " + r + " beyond the rebuilt ones is zero", 0f, summary[r * PROXY + d], 0f);
            }
        }
    }

    @Test
    public void clearSummariesFromZeroesTheRegionsAhead() {
        int capacity = 16;
        float[] summary = new float[2 * PROXY];
        float[] ring = new float[capacity * 2 * PROXY];
        CpuStep5RegionCache cache = new CpuStep5RegionCache(REGION, capacity, PROXY, summary, 0, ring, 0);
        float[] key = new float[PROXY];
        float[] z = new float[PROXY];
        for (int p = 0; p < capacity; p++) {
            key[0] = p + 1;
            cache.update(p, key, 0, z, 0);
        }
        cache.clearSummariesFrom(1);
        assertArrayEquals(new float[]{4.5f}, new float[]{summary[0]}, 1e-6f);
        for (int d = 0; d < PROXY; d++) {
            assertEquals("cleared region dim " + d, 0f, summary[PROXY + d], 0f);
        }
    }

    /** Direct weighted mean of ring positions {@code [first, last]}: one global maximum over all {@code z}. */
    private static float[] expected(float[][] keys, float[][] z, int first, int last) {
        float[] out = new float[PROXY];
        float max = Float.NEGATIVE_INFINITY;
        for (int p = first; p <= last; p++) {
            for (int d = 0; d < PROXY; d++) {
                max = Math.max(max, Step5Dtypes.bf16(z[p][d]));
            }
        }
        for (int d = 0; d < PROXY; d++) {
            double num = 0.0;
            double den = 0.0;
            for (int p = first; p <= last; p++) {
                double w = Math.exp(Step5Dtypes.bf16(z[p][d]) - max);
                num += w * Step5Dtypes.bf16(keys[p][d]);
                den += w;
            }
            out[d] = (float) (num / Math.max(den, 1e-20));
        }
        return out;
    }
}
