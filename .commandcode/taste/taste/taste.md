# Taste
- When the user says to continue/resume a previous session, don't tell them to restart via CLI flags — locate and read that session's transcript on disk (e.g. under ~/.claude/projects/) and pick up the work in-session, even if it came from a different tool (e.g. a Claude Code session rather than command code). Confidence: 0.8
- Never run tests, demos, apps or models directly; verify correctness by compiling (including kernel compilers where relevant) and tracing logic through code paths instead. Confidence: 0.95
- Keep an existing codebase's design patterns when extending it; new work must be performant and well organized. Confidence: 0.85
- When reference repos are provided (e.g. vllm, llama.cpp), use them as the semantic source of truth for feature parity. Confidence: 0.85
- Give compound instructions ("do X, then a full review pass, and commit") and expect them driven end-to-end to completion without follow-up questions; large work is executed in explicit stages with adversarial review before delivery. Confidence: 0.85
- Commit only when explicitly asked; when a commit is requested it comes after the full work (including the review pass), never before. Confidence: 0.9
- Before making a repo public: clean junk files, refresh/rewrite documentation, add a quick-start guide, and sweep for secrets and hardcoded credentials, removing anything found. Confidence: 0.9
- License preference: Apache 2.0 with a commercial-usage limitation — organizations above $20M revenue require a commercial licence; copyright attributed to EONMUX LTD and Amir Farhad Eslampanah. Confidence: 0.95
- Publishing pipeline: reset commit history and make the current state the first commit, set the remote to an SSH URL (git@github.com:EonMux/NeuralFree.git), then push. Confidence: 0.85
