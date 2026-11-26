---
license: cc-by-4.0
language:
  - en
tags:
  - python
  - software-testing
---

# fi-python-seeds-v1

Six healthy, frozen Python source slices support functional testing on previously seen and unseen repository families. Four training families cover terminal colors, ordered sets, time parsing and signal delivery. Two evaluation families cover nested mappings and command-line parsing. Each slice was selected by hand. These counts describe this release.

Each split has separate `actor.zip` and `grader.zip` files. Supply only the selected input folder from the actor archive to a participant. Give a training process access only to the training split. Keep the complete repository and the grader archives under trusted control. A repository family split cannot establish what a pretrained model has already seen.

Actor folders include the original source, its license, a task, one public pytest node and the exact rendered Qwen3-14B input in `prompt.txt`. Grader folders include the independent oracle node, import root, tests, source identity, license path and hashes. Inputs from src-layout projects use `src` as their Python import root. Each test suite runs in its own process. Both exercise functions in the same frozen business module. The release verifier requires two independently reset copies to match the original slice bytes.

Use `python stage1/build.py` from the frozen fault-injector checkout with Python 3.12, Git and Node/npm. Its receipt records the upstream SHAs, actual run times and family comparisons. The tokenizer files, template and revision are fixed in the project manifest. SHA256SUMS covers every other file in this directory.

The dataset selection, task wording and annotations are © 2026 RedinGhost and fault-injector contributors, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the dataset authors, link the license and identify changes. Original Python sources retain MIT licenses with the copyright notices supplied in each actor folder. Project-authored test code uses Apache-2.0, supplied in LICENSE and NOTICE. Source URLs and original license paths accompany each grader record. [Project and frozen inputs](https://github.com/rrrrrredy/fault-injector)
