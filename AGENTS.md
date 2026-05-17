# Agent Instructions

# Development
- After making code changes, run `npm run build` from the repository root before considering the task complete.
- Do not treat workspace-level test commands as a substitute for `npm run build`; the built demo assets must be refreshed.

# Comments and Documentation
Good comments are artifacts to the state of the codebase, constraints, and roadmap of the project at the time they are written. They should be written liberally throughout the codebase to aid future readers. To do this:
- Edge cases always get their own comment. An edge case test: "Think of the most obvious singular use case of this code. Any other use case is an edge case and added logic must be commented".
- Good comments should name the failure mode: what inputs, states, or event orderings break without the code, and how the code avoids or fixes that bug. To show concrete examples, explain in comments the names of tests that were added that specifically check the behavior of added code blocks.
- Avoid comments that just restate the implementation.
- Functions should contain docstrings explaining 1) their use, 2) input arguments, 3) return values and error codes, and 4) examples of how the function may be used to accomplish some larger task.
- Use short comments to indicate code sections. If a function has three logical sections, each section should have a short comment implying start of section.
  - NOTE: Avoid helper functions to separate code logic. Only add a helper function if the logic is used 3 times or more.
- When introducing an abstraction, add a docstring explaining 1) the purpose of the abstraction, 2) **at least** two examples of instances of the abstraction (add more examples if two examples don't adequately cover the whole abstraction surface), and 3) how the abstraction handles shared attributes of the instances.
- Directories should contain their own `ARCHITECTURE.md`, containing a high-level overview of the architecture (what responsibility each file handles, describe any new abstractions accordingly). They should be written to a lay audience, not an audience that already know the fundamental design decisions and need a technical refresher. This means that a rhetorical style may be best, in that it should convince a reader step-by-step that the current design decisions are inevitable if they seek to re-implement the project from scratch.

# Code Quality
The test for good code quality is: "If I throw a reader in a random part of the code, in 30 seconds they can 1) understand what the code immediately does, 2) how the code directly affects the end user, and 3) the code is necessary". A reader should be able to go to a random point of code and roughly think "Ah, so _this_ how feature XXX in the app works". Examples of bad code:
- uncommented code (violates point 1)
- code that don't work on strict types, e.g. function only operates on ints but input type is any object (violates point 1)
- needlessly complicated chained operations (violates point 1)
- extensive helper functions (violates point 2)
- overabstraction (violates point 2)
- over-defensive code, e.g. validation checks that aren't specified in interface specs (violates point 3)
- dead/hard-to-test code (violates point 3)
