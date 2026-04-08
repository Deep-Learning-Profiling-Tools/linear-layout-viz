# Agent Instructions

# Development
- After making code changes, run `npm run build` from the repository root before considering the task complete.
- Do not treat workspace-level test commands as a substitute for `npm run build`; the built demo assets must be refreshed.

# Comments and Documentation
Good comments are artifacts to the state of the codebase, constraints, and roadmap of the project at the time they are written. To do this:
- When adding code comments, explain why the code is needed, not merely what it does.
- Good comments should name the failure mode: what inputs, states, or event orderings break without the code, and how the code avoids or fixes that bug. To show concrete examples, explain in comments the names of tests that were added that specifically check the behavior of added code blocks.
- Avoid comments that just restate the implementation.
- Directories should contain their own `ARCHITECTURE.md`, containing a high-level overview of the architecture. They should be written to a lay audience, not an audience that already know the fundamental design decisions and need a technical refresher. This means that a rhetorical style may be best, in that it should convince a reader step-by-step that the current design decisions are inevitable if they seek to re-implement the project from scratch.
  
