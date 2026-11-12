# The CPU SIMD kernels run on jdk.incubator.vector, which Java 17 and later runtimes carry unless their image was
# linked without it. For a command that runs the engine in this JVM the module is resolved at start-up, so the
# launcher need not relaunch itself to add it: when the runtime's release file ($JAVA_HOME/release, or the one two
# directories above the resolved java executable) names Java 17 or later and, where it lists the image's modules,
# names jdk.incubator.vector among them. collect-env resolves it too, since it reports the CPU kernels the engine
# commands select. The other commands (serve, run-batch, chat, complete, version, help, cache-list, --list-devices)
# never load the CPU kernels here and start without it.
case ${1-} in
    ''|serve|--serve|server|--server|run-batch|--run-batch|chat|--chat|complete|--complete|version|--version|\
    help|--help|-h|cache-list|--cache-list|-cl|--list-devices)
        NEURALFREE_VECTOR_COMMAND= ;;
    *)
        NEURALFREE_VECTOR_COMMAND=1 ;;
esac
NEURALFREE_JAVA_RELEASE_FILE=
if [ -z "$NEURALFREE_VECTOR_COMMAND" ] ; then
    :
elif [ -n "$JAVA_HOME" ] && [ -r "$JAVA_HOME/release" ] ; then
    NEURALFREE_JAVA_RELEASE_FILE=$JAVA_HOME/release
else
    NEURALFREE_JAVA_PATH=$( command -v "$JAVACMD" 2>/dev/null ) || NEURALFREE_JAVA_PATH=
    if [ -n "$NEURALFREE_JAVA_PATH" ] ; then
        NEURALFREE_JAVA_PATH=$( readlink -f "$NEURALFREE_JAVA_PATH" 2>/dev/null || printf '%s' "$NEURALFREE_JAVA_PATH" )
    fi
    case $NEURALFREE_JAVA_PATH in
        /*/*/*) NEURALFREE_JAVA_RELEASE_FILE=${NEURALFREE_JAVA_PATH%/*/*}/release ;;
    esac
fi
NEURALFREE_JAVA_FEATURE=
NEURALFREE_JAVA_MODULES=
if [ -n "$NEURALFREE_JAVA_RELEASE_FILE" ] && [ -r "$NEURALFREE_JAVA_RELEASE_FILE" ] ; then
    NEURALFREE_JAVA_FEATURE=$( sed -n 's/^JAVA_VERSION="\{0,1\}\([0-9][0-9]*\).*/\1/p' "$NEURALFREE_JAVA_RELEASE_FILE" |
        head -n 1 )
    # an "=" marks a listed line, so an image without a MODULES line is told apart from one listing nothing
    NEURALFREE_JAVA_MODULES=$( sed -n 's/^MODULES=/=/p' "$NEURALFREE_JAVA_RELEASE_FILE" | head -n 1 | tr -d '"' )
fi
NEURALFREE_JAVA_VECTOR=
case $NEURALFREE_JAVA_MODULES in
    '') NEURALFREE_JAVA_VECTOR=1 ;;
    *)
        case " ${NEURALFREE_JAVA_MODULES#=} " in
            *" jdk.incubator.vector "*) NEURALFREE_JAVA_VECTOR=1 ;;
        esac
        ;;
esac
case $NEURALFREE_JAVA_FEATURE in
    ''|*[!0-9]*) ;;
    *)
        if [ "$NEURALFREE_JAVA_FEATURE" -ge 17 ] && [ -n "$NEURALFREE_JAVA_VECTOR" ] ; then
            DEFAULT_JVM_OPTS='"--add-modules=jdk.incubator.vector" '"$DEFAULT_JVM_OPTS"
        fi
        ;;
esac
