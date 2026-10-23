package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaKernelSignatures;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.Proxy;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Map;
import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The CUDA library's use of the kernel inventory: parsing each kernel's argument signature from the
 * count and kind columns, and registering it in {@link CudaKernelSignatures} when a kernel is
 * resolved, so the CUDA backend checks the kernel's launches. Uses a classpath built in a temporary
 * directory and a stand-in backend that only reports its type; no CUDA device or driver is used.
 */
public class CudaKernelLibrarySignatureTest {

    private static final String BUILD = "nvcc.version=13.3\ncuda.driver.min=13000\nsass.archs=75,120\nptx.archs=75\n";

    private Path dir;
    private String sourceRoot;
    private String cudaRoot;

    @Before
    public void createClasspath() throws IOException {
        dir = Files.createTempDirectory("nf-cuda-sig");
        sourceRoot = "nftest/" + UUID.randomUUID().toString().replace("-", "");
        cudaRoot = sourceRoot + "/cuda";
    }

    @After
    public void deleteClasspath() throws IOException {
        Files.walkFileTree(dir, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path d, IOException e) throws IOException {
                Files.delete(d);
                return FileVisitResult.CONTINUE;
            }
        });
    }

    @Test
    public void parsesCountAndKindColumns() {
        Map<String, KernelSignature> s = CudaKernelLibrary.parseSignatures(
                "# comment\n"
                        + "BF16_BatchMatMul\tmatvec_f16_bf16.cl\t3\t3\ti,i,i\n"
                        + "LLM_Fill\teltwise.cl\t1\t0\t\n"
                        + "\n"
                        + "LLM_Scale\teltwise.cl\t2\t2\tf,u\n",
                "test");

        assertEquals(3, s.size());
        assertEquals(KernelSignature.parseInventory("3", "3", "i,i,i"), s.get("BF16_BatchMatMul"));
        assertEquals(1, s.get("LLM_Fill").bufferCount());
        assertEquals(0, s.get("LLM_Fill").scalarCount());
        assertEquals('u', s.get("LLM_Scale").pushKind(1));
    }

    @Test
    public void parsesTheLayoutColumn() {
        Map<String, KernelSignature> s = CudaKernelLibrary.parseSignatures(
                "LLM_Mixed\tmixed.cl\t2\t2\ti,f\tiBfB\n", "test");

        KernelSignature sig = s.get("LLM_Mixed");
        assertTrue(sig.hasLayout());
        assertEquals(KernelSignature.parseInventory("2", "2", "i,f", "iBfB"), sig);
        expectIllegal("LLM_A\tsrc.cl\t1\t1\ti\tBB\n", "test:1: kernel LLM_A: layout 'BB' has 2 buffers");
    }

    @Test
    public void nameOnlyLinesCarryNoSignature() {
        Map<String, KernelSignature> s = CudaKernelLibrary.parseSignatures("LLM_A\nLLM_B extra words\n", "test");

        assertTrue(s.isEmpty());
    }

    @Test
    public void malformedLinesNameTheOriginAndLine() {
        expectIllegal("LLM_A\tsrc.cl\t1\t1\n", "test:1: expected kernel_name<TAB>source<TAB>buffers<TAB>scalars<TAB>kinds");
        expectIllegal("LLM_A\tsrc.cl\t1\t2\ti\n", "test:1: kernel LLM_A: scalar count 2 disagrees");
        expectIllegal("LLM_A\tsrc.cl\t1\t0\t\nLLM_A\tsrc.cl\t1\t0\t\n", "test:2: duplicate kernel 'LLM_A'");
        expectIllegal("9bad\tsrc.cl\t1\t0\t\n", "test:1: '9bad' is not a kernel name");
    }

    @Test
    public void resolveRegistersTheInventorySignature() throws IOException {
        writeBuild("LLM_Sig_A\tmod_a.fatbin\n", "LLM_Sig_A\tmod_a.cl\t2\t3\ti,f,u\n");

        try (URLClassLoader cl = loader()) {
            CudaKernelLibrary lib = library(cl);
            KernelArtifact a = lib.resolve("LLM_Sig_A", cudaBackend());

            assertEquals(KernelArtifact.Kind.CUDA_IMAGE, a.kind());
            assertEquals(cudaRoot + "/mod_a.fatbin", a.resourcePath());
            assertEquals(KernelSignature.parseInventory("2", "3", "i,f,u"),
                    CudaKernelSignatures.lookup(a.resourcePath(), "LLM_Sig_A"));
        }
    }

    @Test
    public void kernelWithoutInventoryColumnsResolvesUnchecked() throws IOException {
        writeBuild("LLM_Sig_B\tmod_b.fatbin\n", "LLM_Sig_B\n");

        try (URLClassLoader cl = loader()) {
            KernelArtifact a = library(cl).resolve("LLM_Sig_B", cudaBackend());

            assertNull(CudaKernelSignatures.lookup(a.resourcePath(), "LLM_Sig_B"));
        }
    }

    @Test
    public void conflictingRegistrationMakesTheKernelUnavailable() throws IOException {
        writeBuild("LLM_Sig_C\tmod_c.fatbin\n", "LLM_Sig_C\tmod_c.cl\t1\t1\ti\n");
        CudaKernelSignatures.register(cudaRoot + "/mod_c.fatbin", "LLM_Sig_C", KernelSignature.parseInventory("1", "1", "f"));

        try (URLClassLoader cl = loader()) {
            library(cl).resolve("LLM_Sig_C", cudaBackend());
            fail("a signature that contradicts the registered one must be refused");
        } catch (UnsupportedKernelException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("different builds"));
        }
    }

    @Test
    public void malformedInventoryMakesTheLibraryUnusable() throws IOException {
        writeBuild("LLM_Sig_D\tmod_d.fatbin\n", "LLM_Sig_D\tmod_d.cl\tx\t0\t\n");

        try (URLClassLoader cl = loader()) {
            CudaKernelLibrary lib = library(cl);
            assertTrue(lib.kernelNames().isEmpty());
            lib.resolve("LLM_Sig_D", cudaBackend());
            fail("a malformed inventory must make the CUDA build unusable");
        } catch (UnsupportedKernelException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("buffer count 'x' is not an integer"));
        }
    }

    @Test
    public void registryNormalisesLeadingSlashes() {
        KernelSignature s = KernelSignature.parseInventory("1", "0", "");
        String resource = sourceRoot + "/cuda/slash.fatbin";

        CudaKernelSignatures.register("/" + resource, "LLM_Slash", s);
        CudaKernelSignatures.register(resource, "LLM_Slash", s);

        assertEquals(s, CudaKernelSignatures.lookup("//" + resource, "LLM_Slash"));
        assertNull(CudaKernelSignatures.lookup(resource, "LLM_Other"));
    }

    private void writeBuild(String index, String inventory) throws IOException {
        write(cudaRoot + "/" + CudaKernelLibrary.INDEX_FILE, index);
        write(cudaRoot + "/" + CudaKernelLibrary.BUILD_FILE, BUILD);
        write(sourceRoot + "/" + KernelManifest.INVENTORY_FILE, inventory);
        for (String line : index.split("\n")) {
            String[] cols = line.split("\t");
            if (cols.length == 2) {
                write(cudaRoot + "/" + cols[1], "fatbin");
            }
        }
    }

    private void write(String resource, String text) throws IOException {
        Path p = dir.resolve(resource);
        Files.createDirectories(p.getParent());
        Files.write(p, text.getBytes(StandardCharsets.UTF_8));
    }

    private URLClassLoader loader() throws IOException {
        return new URLClassLoader(new URL[]{dir.toUri().toURL()}, null);
    }

    private CudaKernelLibrary library(ClassLoader cl) {
        return new CudaKernelLibrary(new KernelLibraryConfig("sigtest", sourceRoot, cudaRoot, sourceRoot + "/spv",
                false), cl);
    }

    private static ComputeBackend cudaBackend() {
        return (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                new Class<?>[]{ComputeBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("backendType")) {
                        return ComputeBackend.BackendType.CUDA;
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
    }

    private static void expectIllegal(String text, String fragment) {
        try {
            CudaKernelLibrary.parseSignatures(text, "test");
            fail("expected IllegalArgumentException containing '" + fragment + "'");
        } catch (IllegalArgumentException e) {
            assertTrue("message '" + e.getMessage() + "' lacks '" + fragment + "'", e.getMessage().contains(fragment));
        }
    }
}
