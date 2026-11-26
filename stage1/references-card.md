# Human Regressions Reference Set v1

Two ordinary Python regressions from two nontraining repository families provide historical references for blind evaluation. Each source contributes one reference. Two derived Factory variants per source verify an independent control and share the base case's provenance.

Each actor folder contains an issue-style ticket with three fields, the original affected module at the faulty revision and its copyright notice. Provide that folder alone to a participant. Keep the complete dataset and the grader archive outside the participant's environment. Grader folders contain issue or merged PR URLs, the faulty and fixed revisions, the merged commit, the corrected module, verifier code and original notices. Public upstream projects can be found independently; blinding applies to the provided task materials.

Run `python stage1/build.py` from the corresponding fault-injector commit with Git, Node/npm and Python 3.12. It fetches the frozen Git sources and tools, runs the same trusted assertion at both revisions through the pinned Factory, retains both reset fingerprints and validates every exported case through RunCase. The release archives are deterministic. The per-host receipts contain the actual observations including case IDs, test output, execution mode and fingerprints.

The dataset selection and tickets are © 2026 RedinGhost and fault-injector contributors under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution and identify changes. Original sources retain MIT licenses. License files accompany each task and grader. Project-authored verifier code uses the enclosed Apache-2.0 license. [Project and frozen inputs](https://github.com/rrrrrredy/fault-injector)
