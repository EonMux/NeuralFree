package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * The {@code random-mm} dataset, vLLM's {@code RandomMultiModalDataset}: the random dataset's
 * synthetic text plus synthetic images. Each request draws its item count from
 * {@code [floor(n(1-r)), ceil(n(1+r))]} around {@code --random-mm-base-items-per-request}, samples
 * each item's shape from {@code --random-mm-bucket-config} (a bucket of one frame is an image), and
 * enforces {@code --random-mm-limit-mm-per-prompt}. The text prompt is decoded through the
 * tokenizer and every image is a random-pixel JPEG data URL, the reference's synthetic RGB image.
 * Video buckets are refused by name: the harness has no MP4 encoder.
 */
final class RandomMultiModalDataset implements ServeBenchDataset {

    private final ServeBenchOptions opts;
    private final BenchTokenizer tokenizer;
    private final int[] allowed;

    RandomMultiModalDataset(ServeBenchOptions opts, BenchTokenizer tokenizer, int vocabSize, int[] skipIds) {
        this.opts = opts;
        this.tokenizer = tokenizer;
        this.allowed = SyntheticSampler.allowed(vocabSize, skipIds);
    }

    @Override
    public List<BenchSample> sample() {
        int numRequests = opts.numPrompts();
        Random random = new Random(opts.seed());
        int inputLen = opts.inputLen() != null ? opts.inputLen() : opts.randomInputLen();
        int outputLen = opts.outputLen() != null ? opts.outputLen() : opts.randomOutputLen();
        SyntheticSampler.Draw[] draws = SyntheticSampler.draws(opts, random, numRequests, inputLen, outputLen);
        SyntheticSampler.drawOffsets(draws, random, allowed.length);

        int prefixLen = opts.randomPrefixLen();
        int[] prefix = new int[prefixLen];
        for (int i = 0; i < prefixLen; i++) {
            prefix[i] = allowed[random.nextInt(allowed.length)];
        }

        Buckets buckets = Buckets.of(opts);
        List<BenchSample> out = new ArrayList<>(numRequests);
        for (int i = 0; i < numRequests; i++) {
            int[] body = SyntheticSampler.walk(allowed, draws[i].offset, i, draws[i].inputLen);
            int[] tokens = new int[prefix.length + body.length];
            System.arraycopy(prefix, 0, tokens, 0, prefix.length);
            System.arraycopy(body, 0, tokens, prefix.length, body.length);
            String prompt = SyntheticSampler.text(tokenizer, tokens);

            List<Object> content = new ArrayList<>();
            content.add(BenchDatasets.textPart(prompt));
            for (Bucket bucket : buckets.draw(random)) {
                content.add(BenchDatasets.imagePart(
                        BenchDatasets.syntheticImageDataUrl(bucket.width, bucket.height, random), null));
            }
            out.add(BenchSample.ofChat(content, tokens.length, draws[i].outputLen,
                    opts.requestIdPrefix() + i));
        }
        return out;
    }

    /** One item shape and its probability. */
    private static final class Bucket {

        final int height;
        final int width;
        final int frames;
        final double probability;

        Bucket(int height, int width, int frames, double probability) {
            this.height = height;
            this.width = width;
            this.frames = frames;
            this.probability = probability;
        }

        boolean video() {
            return frames > 1;
        }
    }

    /** The normalized bucket set and the per-request item drawing over it. */
    private static final class Buckets {

        private final List<Bucket> buckets;
        private final Map<String, Integer> limits;
        private final int minItems;
        private final int maxItems;

        private Buckets(List<Bucket> buckets, Map<String, Integer> limits, int minItems, int maxItems) {
            this.buckets = buckets;
            this.limits = limits;
            this.minItems = minItems;
            this.maxItems = maxItems;
        }

        static Buckets of(ServeBenchOptions opts) {
            double ratio = opts.randomMmNumMmItemsRangeRatio();
            if (!(ratio >= 0.0 && ratio <= 1.0)) {
                throw new IllegalArgumentException("--random-mm-num-mm-items-range-ratio must be in"
                        + " [0, 1], got " + ratio);
            }
            Map<String, Integer> limits = new LinkedHashMap<>(opts.randomMmLimitMmPerPrompt());
            List<Bucket> all = new ArrayList<>();
            double total = 0.0;
            for (Map.Entry<String, Double> entry : opts.randomMmBucketConfig().entrySet()) {
                String[] parts = entry.getKey().split(",");
                if (parts.length != 3) {
                    throw new IllegalArgumentException("a --random-mm-bucket-config key is"
                            + " 'height,width,frames', not '" + entry.getKey() + "'");
                }
                int height;
                int width;
                int frames;
                try {
                    height = Integer.parseInt(parts[0].trim());
                    width = Integer.parseInt(parts[1].trim());
                    frames = Integer.parseInt(parts[2].trim());
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("a --random-mm-bucket-config key is"
                            + " 'height,width,frames', not '" + entry.getKey() + "'");
                }
                double probability = entry.getValue();
                if (probability < 0.0) {
                    throw new IllegalArgumentException("a --random-mm-bucket-config probability cannot"
                            + " be negative, got " + probability + " for '" + entry.getKey() + "'");
                }
                String modality = frames == 1 ? "image" : frames > 1 ? "video" : null;
                if (modality == null) {
                    throw new IllegalArgumentException("a random-mm bucket needs at least one frame,"
                            + " got " + frames);
                }
                if (probability > 0.0 && !limits.containsKey(modality)) {
                    throw new IllegalArgumentException("the modality '" + modality + "' of a"
                            + " --random-mm-bucket-config entry is not in"
                            + " --random-mm-limit-mm-per-prompt");
                }
                if (probability > 0.0 && frames > 1) {
                    throw new IllegalArgumentException("random-mm video buckets are not supported:"
                            + " the harness has no MP4 encoder");
                }
                if (probability > 0.0) {
                    all.add(new Bucket(height, width, frames, probability));
                    total += probability;
                }
            }
            if (all.isEmpty()) {
                throw new IllegalArgumentException("--random-mm-bucket-config needs at least one"
                        + " positive image bucket");
            }
            List<Bucket> normalized = new ArrayList<>();
            for (Bucket bucket : all) {
                normalized.add(new Bucket(bucket.height, bucket.width, bucket.frames,
                        bucket.probability / total));
            }
            int allowedLimit = 0;
            java.util.Set<String> modalities = new java.util.HashSet<>();
            for (Bucket bucket : normalized) {
                modalities.add(bucket.video() ? "video" : "image");
            }
            for (String modality : modalities) {
                allowedLimit += Math.max(0, limits.getOrDefault(modality, 0));
            }
            int base = opts.randomMmBaseItemsPerRequest();
            int max = Math.min(allowedLimit,
                    (int) Math.ceil(base * (1.0 + ratio)));
            int min = Math.max(0, (int) Math.floor(base * (1.0 - ratio)));
            if (min > max) {
                throw new IllegalArgumentException("the random-mm item count is empty: min " + min
                        + " > max " + max);
            }
            Map<String, Integer> kept = new LinkedHashMap<>();
            for (String modality : modalities) {
                kept.put(modality, limits.get(modality));
            }
            return new Buckets(normalized, kept, min, max);
        }

        /** The buckets one request draws, at most one per slot and never past a modality's limit. */
        List<Bucket> draw(Random random) {
            int count = minItems + random.nextInt(maxItems - minItems + 1);
            List<Bucket> out = new ArrayList<>();
            if (count == 0) {
                return out;
            }
            Map<String, Integer> used = new LinkedHashMap<>();
            for (String modality : limits.keySet()) {
                used.put(modality, 0);
            }
            List<Bucket> active = new ArrayList<>(buckets);
            while (out.size() < count) {
                Bucket picked = pick(active, random);
                String modality = picked.video() ? "video" : "image";
                if (used.getOrDefault(modality, 0) < limits.getOrDefault(modality, 0)) {
                    used.merge(modality, 1, Integer::sum);
                    out.add(picked);
                } else {
                    List<Bucket> remaining = new ArrayList<>();
                    for (Bucket bucket : active) {
                        if (!(bucket.video() ? "video" : "image").equals(modality)) {
                            remaining.add(bucket);
                        }
                    }
                    if (remaining.isEmpty()) {
                        break;
                    }
                    active = normalize(remaining);
                }
            }
            return out;
        }

        private static Bucket pick(List<Bucket> buckets, Random random) {
            double draw = random.nextDouble();
            double cumulative = 0.0;
            for (Bucket bucket : buckets) {
                cumulative += bucket.probability;
                if (draw < cumulative) {
                    return bucket;
                }
            }
            return buckets.get(buckets.size() - 1);
        }

        private static List<Bucket> normalize(List<Bucket> buckets) {
            double total = 0.0;
            for (Bucket bucket : buckets) {
                total += bucket.probability;
            }
            List<Bucket> out = new ArrayList<>();
            for (Bucket bucket : buckets) {
                out.add(new Bucket(bucket.height, bucket.width, bucket.frames, bucket.probability / total));
            }
            return out;
        }
    }
}
