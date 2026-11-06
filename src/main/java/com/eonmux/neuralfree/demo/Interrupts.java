package com.eonmux.neuralfree.demo;

import java.io.PrintStream;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.IntConsumer;

/**
 * The Ctrl-C (SIGINT) state of a console session, with the llama.cpp CLI semantics (tools/cli/cli.cpp
 * {@code signal_handler}, tools/completion/completion.cpp {@code sigint_handler}): the first press during a
 * generation sets the flag, which cancels the turn through the {@code GenerationRequest} cancellation hook or
 * closes a streamed server answer; a second press before the session clears the flag, or any press while the
 * session waits for input, ends the process with exit code {@value #EXIT_CODE}.
 *
 * <p>{@link #system()} installs the process handler through reflection on the JDK-internal {@code sun.misc.Signal},
 * which is not reachable at compile time; on a runtime without it Ctrl-C keeps the JVM's default behavior of
 * ending the process.</p>
 */
final class Interrupts {

    /** The exit code of a process ended by Ctrl-C (128 + SIGINT), as llama.cpp exits. */
    static final int EXIT_CODE = 130;

    private static Interrupts system;

    private final AtomicBoolean requested = new AtomicBoolean(false);
    private final AtomicBoolean awaitingInput = new AtomicBoolean(false);
    private final IntConsumer exit;
    private final PrintStream out;

    /**
     * Interrupt state whose ending presses call {@code exit} with {@value #EXIT_CODE} after a newline on
     * {@code out}.
     */
    Interrupts(IntConsumer exit, PrintStream out) {
        if (exit == null || out == null) {
            throw new IllegalArgumentException("exit and out are required");
        }
        this.exit = exit;
        this.out = out;
    }

    /** The process-wide state, its SIGINT handler installed on first use. */
    static synchronized Interrupts system() {
        if (system == null) {
            system = new Interrupts(System::exit, System.out);
            installHandler(system);
        }
        return system;
    }

    /** One Ctrl-C press. */
    void press() {
        boolean second = !requested.compareAndSet(false, true);
        if (second || awaitingInput.get()) {
            out.println();
            out.flush();
            exit.accept(EXIT_CODE);
        }
    }

    /** Whether Ctrl-C was pressed since the last {@link #clear}. */
    boolean requested() {
        return requested.get();
    }

    /** Clears the flag once the session acted on it. */
    void clear() {
        requested.set(false);
    }

    /** Marks the session as blocked on user input, where a press ends the process. */
    void awaitingInput(boolean waiting) {
        awaitingInput.set(waiting);
    }

    private static void installHandler(Interrupts target) {
        try {
            Class<?> signal = Class.forName("sun.misc.Signal");
            Class<?> handler = Class.forName("sun.misc.SignalHandler");
            Object sig = signal.getConstructor(String.class).newInstance("INT");
            Object handlerInstance = Proxy.newProxyInstance(Interrupts.class.getClassLoader(),
                    new Class<?>[]{handler},
                    (proxy, method, args) -> {
                        switch (method.getName()) {
                            case "handle":
                                target.press();
                                return null;
                            case "hashCode":
                                return System.identityHashCode(proxy);
                            case "equals":
                                return proxy == args[0];
                            default:
                                return "Interrupts.SignalHandler";
                        }
                    });
            signal.getMethod("handle", signal, handler).invoke(null, sig, handlerInstance);
        } catch (ReflectiveOperationException | RuntimeException | LinkageError e) {
            System.err.println("Ctrl-C keeps its default behavior (no signal API on this runtime: " + e + ")");
        }
    }
}
