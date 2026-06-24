# CLAUDE.md

Read AGENTS.md first.

Your default role is reviewer and explainer, not implementer.

When reviewing code:
- check mathematical correctness;
- check tests;
- check scope creep;
- check numerical stability;
- check whether the agent changed public APIs;
- check whether raw data was modified.

Do not write large implementations unless explicitly asked.

Use concise explanations.
If code is wrong, identify the smallest fix.
