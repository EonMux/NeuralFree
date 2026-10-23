package com.eonmux.neuralfree.data;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class ImageFolderDataset implements Dataset<ImageFolderDataset.Sample> {
    private final Path rootPath;
    private final ComputeBackend backend;
    private final int targetWidth;
    private final int targetHeight;
    private final boolean normalize;
    private final float meanR;
    private final float meanG;
    private final float meanB;
    private final float stdR;
    private final float stdG;
    private final float stdB;
    private final boolean grayscale;

    private final List<ImageEntry> entries;
    private final Map<String, Integer> classToIndex;
    private final List<String> classNames;

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(
        "jpg", "jpeg", "png", "bmp", "gif"
    );

    public ImageFolderDataset(Path rootPath, ComputeBackend backend) {
        this(rootPath, backend, 224, 224, false, 0f, 0f, 0f, 1f, 1f, 1f, false);
    }

    public ImageFolderDataset(
            Path rootPath,
            ComputeBackend backend,
            int targetWidth,
            int targetHeight,
            boolean normalize,
            float meanR, float meanG, float meanB,
            float stdR, float stdG, float stdB,
            boolean grayscale) {
        
        this.rootPath = Objects.requireNonNull(rootPath, "Root path cannot be null");
        this.backend = Objects.requireNonNull(backend, "Backend cannot be null");
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
        this.normalize = normalize;
        this.meanR = meanR;
        this.meanG = meanG;
        this.meanB = meanB;
        this.stdR = stdR;
        this.stdG = stdG;
        this.stdB = stdB;
        this.grayscale = grayscale;

        if (!Files.isDirectory(rootPath)) {
            throw new IllegalArgumentException("Root path must be a directory: " + rootPath);
        }
        if (targetWidth <= 0 || targetHeight <= 0) {
            throw new IllegalArgumentException("Target dimensions must be positive");
        }

        this.entries = new ArrayList<>();
        this.classToIndex = new HashMap<>();
        this.classNames = new ArrayList<>();

        scanDirectory();
    }

    private void scanDirectory() {
        try {
            List<Path> classDirs = new ArrayList<>();
            try (var stream = Files.list(rootPath)) {
                stream.filter(Files::isDirectory)
                      .sorted(Comparator.comparing(Path::getFileName))
                      .forEach(classDirs::add);
            }

            int[] classIndexRef = {0};
            for (Path classDir : classDirs) {
                String className = classDir.getFileName().toString();
                classToIndex.put(className, classIndexRef[0]);
                classNames.add(className);

                final int currentClassIndex = classIndexRef[0];
                try (var stream = Files.walk(classDir)) {
                    stream.filter(Files::isRegularFile)
                          .filter(this::isImageFile)
                          .forEach(imagePath -> entries.add(new ImageEntry(imagePath, currentClassIndex)));
                }

                classIndexRef[0]++;
            }

            if (entries.isEmpty()) {
                throw new IllegalStateException("No images found in directory: " + rootPath);
            }

        } catch (IOException e) {
            throw new RuntimeException("Failed to scan directory: " + rootPath, e);
        }
    }

    private boolean isImageFile(Path path) {
        String fileName = path.getFileName().toString().toLowerCase(Locale.ROOT);
        int dotIndex = fileName.lastIndexOf('.');
        if (dotIndex < 0) return false;
        String extension = fileName.substring(dotIndex + 1);
        return SUPPORTED_EXTENSIONS.contains(extension);
    }

    @Override
    public int size() {
        return entries.size();
    }

    @Override
    public Sample get(int index) {
        if (index < 0 || index >= entries.size()) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " out of bounds for dataset size " + entries.size());
        }

        ImageEntry entry = entries.get(index);
        float[] imageData = loadImage(entry.imagePath);
        Tensor imageTensor = createTensor(imageData);

        return new Sample(imageTensor, entry.labelIndex, classNames.get(entry.labelIndex));
    }

    private float[] loadImage(Path imagePath) {
        try {
            BufferedImage image = ImageIO.read(imagePath.toFile());
            if (image == null) {
                throw new IOException("Unsupported image format: " + imagePath);
            }

            BufferedImage resized = resizeImage(image, targetWidth, targetHeight);

            int channels = grayscale ? 1 : 3;
            float[] data = new float[targetWidth * targetHeight * channels];

            int idx = 0;
            for (int y = 0; y < targetHeight; y++) {
                for (int x = 0; x < targetWidth; x++) {
                    int rgb = resized.getRGB(x, y);
                    
                    if (grayscale) {
                        int r = (rgb >> 16) & 0xFF;
                        int g = (rgb >> 8) & 0xFF;
                        int b = rgb & 0xFF;
                        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                        data[idx++] = normalizeValue(gray / 255f, 0.5f, 1f);
                    } else {
                        float r = ((rgb >> 16) & 0xFF) / 255f;
                        float g = ((rgb >> 8) & 0xFF) / 255f;
                        float b = (rgb & 0xFF) / 255f;

                        if (normalize) {
                            data[idx++] = normalizeValue(r, meanR, stdR);
                            data[idx++] = normalizeValue(g, meanG, stdG);
                            data[idx++] = normalizeValue(b, meanB, stdB);
                        } else {
                            data[idx++] = r;
                            data[idx++] = g;
                            data[idx++] = b;
                        }
                    }
                }
            }

            return data;
        } catch (IOException e) {
            throw new RuntimeException("Failed to load image: " + imagePath, e);
        }
    }

    private float normalizeValue(float value, float mean, float std) {
        return (value - mean) / std;
    }

    private BufferedImage resizeImage(BufferedImage original, int targetWidth, int targetHeight) {
        if (original.getWidth() == targetWidth && original.getHeight() == targetHeight) {
            return original;
        }

        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = resized.createGraphics();
        g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                          java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(original, 0, 0, targetWidth, targetHeight, null);
        g.dispose();
        return resized;
    }

    private Tensor createTensor(float[] data) {
        int channels = grayscale ? 1 : 3;
        Tensor flat = Tensor.fromArray(backend, data);
        return flat.reshape(channels, targetHeight, targetWidth);
    }

    public List<String> classNames() {
        return Collections.unmodifiableList(classNames);
    }

    public int numClasses() {
        return classNames.size();
    }

    public Map<String, Integer> classToIndex() {
        return Collections.unmodifiableMap(classToIndex);
    }

    public int targetWidth() {
        return targetWidth;
    }

    public int targetHeight() {
        return targetHeight;
    }

    @Override
    public String toString() {
        return "ImageFolderDataset{size=" + entries.size() + 
               ", numClasses=" + classNames.size() + 
               ", dimensions=" + targetWidth + "x" + targetHeight + "}";
    }

    private static final class ImageEntry {
        final Path imagePath;
        final int labelIndex;

        ImageEntry(Path imagePath, int labelIndex) {
            this.imagePath = imagePath;
            this.labelIndex = labelIndex;
        }
    }

    public static final class Sample {
        private final Tensor image;
        private final int labelIndex;
        private final String labelName;

        public Sample(Tensor image, int labelIndex, String labelName) {
            this.image = Objects.requireNonNull(image, "Image cannot be null");
            this.labelIndex = labelIndex;
            this.labelName = Objects.requireNonNull(labelName, "Label name cannot be null");
        }

        public Tensor image() {
            return image;
        }

        public int labelIndex() {
            return labelIndex;
        }

        public String labelName() {
            return labelName;
        }

        @Override
        public String toString() {
            return "Sample{image=" + image.shape() + ", label=" + labelName + "(" + labelIndex + ")}";
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Path rootPath;
        private ComputeBackend backend;
        private int targetWidth = 224;
        private int targetHeight = 224;
        private boolean normalize = false;
        private float meanR = 0.485f, meanG = 0.456f, meanB = 0.406f;
        private float stdR = 0.229f, stdG = 0.224f, stdB = 0.225f;
        private boolean grayscale = false;

        public Builder rootPath(Path rootPath) {
            this.rootPath = rootPath;
            return this;
        }

        public Builder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }

        public Builder targetSize(int width, int height) {
            this.targetWidth = width;
            this.targetHeight = height;
            return this;
        }

        public Builder normalize(boolean normalize) {
            this.normalize = normalize;
            return this;
        }

        public Builder mean(float r, float g, float b) {
            this.meanR = r;
            this.meanG = g;
            this.meanB = b;
            return this;
        }

        public Builder std(float r, float g, float b) {
            this.stdR = r;
            this.stdG = g;
            this.stdB = b;
            return this;
        }

        public Builder imagenetNormalization() {
            this.normalize = true;
            this.meanR = 0.485f;
            this.meanG = 0.456f;
            this.meanB = 0.406f;
            this.stdR = 0.229f;
            this.stdG = 0.224f;
            this.stdB = 0.225f;
            return this;
        }

        public Builder grayscale(boolean grayscale) {
            this.grayscale = grayscale;
            return this;
        }

        public ImageFolderDataset build() {
            return new ImageFolderDataset(rootPath, backend, targetWidth, targetHeight,
                    normalize, meanR, meanG, meanB, stdR, stdG, stdB, grayscale);
        }
    }
}
