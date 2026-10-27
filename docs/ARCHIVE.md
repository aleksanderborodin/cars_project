# Previous project archive

[Index](../README.md) · [Current specification](EXPERIMENT.md)

The previous project is preserved in the local Git branch:

```text
archive/pre-belief-research-2026-10-27
commit: 5ca42b107e9441e04b135b985dcfec6427d17171
```

The repository had no commits before this migration. This root commit snapshots
the original source, tests, dependency lockfile, task documents, and the original
experiment specification. The old specification is `docs/EXPERIMENT.md` **on that
branch**; its content was verified against the pre-migration copy. The replacement
research proposal was not included in the archive commit.

## Retired from the active tree

- Original experiment plan, task instructions/results, and old implementation reports.
- Rust controller template/loader/build command and its integration test.
- Empty evolution package and the previous evolution research direction.
- Historical checkpoint-specific demo, manual baseline tuner, and video pipeline.
- The video pipeline's unused Playwright dependency.

Reusable environment, pure pursuit, learning, logging, and evaluation components
remain on `main`. Retention is not an endorsement of historical experimental
claims under the new information protocol.

## Inspect or recover

These commands inspect the archive without changing `main`:

```bash
git show archive/pre-belief-research-2026-10-27:docs/EXPERIMENT.md
git ls-tree -r --name-only archive/pre-belief-research-2026-10-27
git worktree add ../cars_project_archive archive/pre-belief-research-2026-10-27
```

The optional worktree provides an isolated old checkout. Use its own locked
environment to reproduce historical code. Restoring individual files is possible
with `git restore --source=archive/pre-belief-research-2026-10-27 -- <path>`, but
review the research contract before reintroducing retired interfaces.

## Local artifacts

Historical `runs/` artifacts stay at their existing paths and remain gitignored;
they have not been reclassified or used as new-paper evidence. Runtime task logs
and the old top-level `logs/` directory were moved, without reading their contents,
under `runs/_archive/pre-belief-research-2026-10-27/`.

Git protects the source snapshot, not these ignored local outputs. Preserve that
directory separately when migrating the workstation. No held-out run results
were opened for this migration. The archive and active commits are local; this
work does not publish or push them to the remote.
