#set page(
  paper: "us-letter",
  margin: (x: 0.8in, y: 0.8in),
)

#set text(
  font: "New Computer Modern",
  size: 10pt,
)

#set par(
  justify: true,
  leading: 0.62em,
)

#set heading(numbering: "1.")

#let status(name) = {
  let fill = if name == "Pass" {
    rgb("#dff3e3")
  } else if name == "Warn" {
    rgb("#fff0c2")
  } else if name == "Fail" {
    rgb("#ffd8d2")
  } else {
    rgb("#e8e8e8")
  }

  box(
    fill: fill,
    stroke: rgb("#444444") + 0.3pt,
    radius: 3pt,
    inset: (x: 5pt, y: 2pt),
  )[*#name*]
}

#align(center)[
  #text(17pt, weight: "bold")[IsoPEPS.jl Repository Hygiene Audit]

  #v(4pt)
  #text(9pt)[Prepared for `IsoPEPS.jl`]
]

#v(10pt)

== Executive Summary

#table(
  columns: (1.4fr, 4fr),
  inset: 7pt,
  stroke: rgb("#d0d0d0") + 0.5pt,
  [*Overall*], [#status("Warn")],
  [*Best signals*], [
    Julia package metadata is solid, tests exist by module, CI is present,
    `.gitignore` protects generated data, and the `gates` smoke test passed.
  ],
  [*Highest-risk gaps*], [
    The public README was too sparse, contribution and release processes were
    not visible, branch and commit history were noisy, and even focused tests
    load a heavy dependency graph.
  ],
  [*Recommended first fixes*], [
    Improve onboarding documentation, add contribution and PR templates,
    document quick test commands, make generated-output policy explicit, and
    inventory stale branches before deleting anything.
  ],
)

== Scope and Evidence

This audit used local, read-only repository evidence: root files, package
metadata, test layout, CI configuration, ignored generated outputs, and Git
history. One focused verification command was run:

```bash
make test-gates
```

The focused `gates` test group passed with 296 tests. A full test suite was not
run because the repository's CI timeout is 180 minutes and the package has a
large precompile surface.

== Scorecard

#table(
  columns: (1.1fr, 0.85fr, 2.4fr, 2.2fr, 2.4fr),
  inset: 5pt,
  stroke: rgb("#d0d0d0") + 0.45pt,
  table.header(
    [*Area*],
    [*Status*],
    [*Evidence*],
    [*Why it matters*],
    [*Next step*],
  ),
  [README],
  [#status("Fail")],
  [`README.md` originally contained only the project title and CI badge.],
  [New users could not tell what the package does or how to start.],
  [Add purpose, installation, quick start, tests, layout, and links to scripts.],

  [Setup],
  [#status("Warn")],
  [Setup commands existed in `AGENTS.md`, not in the public README.],
  [GitHub readers and new contributors would miss the bootstrap path.],
  [Mirror the essential setup commands in `README.md`.],

  [Tests],
  [#status("Pass")],
  [`test/runtests.jl` defines named test groups; `make test-gates` passed.],
  [Contributors can verify focused changes without running everything.],
  [Document named test groups and keep adding focused tests.],

  [CI],
  [#status("Pass")],
  [GitHub Actions runs Julia package tests on pull requests and `master`.],
  [Regression checks exist for collaborative work.],
  [Keep CI aligned with `Project.toml` compat bounds.],

  [Formatting],
  [#status("Warn")],
  [Style conventions were documented, but no formatter config was visible.],
  [Contributors could not mechanically check style.],
  [Add JuliaFormatter config and Makefile targets.],

  [Docs and examples],
  [#status("Warn")],
  [`project/README.md` has useful script guidance, but root docs were sparse.],
  [Project knowledge was present but buried.],
  [Link research scripts and common workflows from the root README.],

  [Project layout],
  [#status("Warn")],
  [Generated output directories include `data/`, `image/`, `states/`, and `project/results/`.],
  [Large local outputs can obscure source files and confuse new contributors.],
  [Document generated-output policy and ignore output paths explicitly.],

  [Dependency hygiene],
  [#status("Warn")],
  [`Project.toml` has compat bounds, but focused tests still precompile heavy optional packages.],
  [Slow smoke tests discourage frequent verification.],
  [Consider moving plotting/reference dependencies behind lighter extension paths.],

  [Security basics],
  [#status("Warn")],
  [No obvious secrets were found, but no security policy was visible.],
  [A public repo should have a safe reporting path for sensitive issues.],
  [Add `SECURITY.md`.],

  [Contribution path],
  [#status("Warn")],
  [No `CONTRIBUTING.md`, issue template, or PR template was visible.],
  [New contributors had to guess the workflow.],
  [Add contribution docs and GitHub templates.],

  [Release/versioning],
  [#status("Warn")],
  [`Project.toml` is `1.0.0-DEV`; TagBot exists, but no local tags were visible.],
  [Users cannot tell what is released or stable.],
  [Add a changelog and release checklist before tagging.],

  [Git history],
  [#status("Warn")],
  [Recent history includes many commits titled `up` or `update`.],
  [Vague history makes regressions and decisions harder to trace.],
  [Use descriptive PR titles and squash summaries going forward.],

  [Branch hygiene],
  [#status("Warn")],
  [Several topic branches are old or ahead/behind their upstreams.],
  [Stale branches hide active work.],
  [Inventory branches as keep, merge, archive, or delete after confirmation.],
)

== Remediation Applied

The following changes address the fail and most beginner-facing warn items:

- Expanded `README.md` with installation, quick start, testing, layout, and
  generated-output policy.
- Added `CONTRIBUTING.md` with branch, test, formatting, PR, and generated-file
  guidance.
- Added `SECURITY.md` with a private reporting policy.
- Added `CHANGELOG.md` with an unreleased section and release checklist.
- Added `.github/PULL_REQUEST_TEMPLATE.md` and structured issue templates.
- Added `.JuliaFormatter.toml`.
- Updated the Makefile with `help`, `test-%`, `format`, and `format-check`.
- Updated `.gitignore` for coverage, local settings, and generated result
  directories.
- Updated CI to include Julia `1.10` and use `actions/checkout@v5`.

== Remaining Risks

Some warnings should not be fixed automatically:

- *Git history:* old vague commits should not be rewritten without an explicit
  history policy.
- *Branches:* stale branches need a human inventory before deletion.
- *Heavy dependencies:* improving test startup time likely requires package
  architecture changes, especially around plotting and reference dependencies.

== Suggested Next Actions

1. Open a hygiene PR with the documentation, template, Makefile, `.gitignore`,
   formatter, and CI changes.
2. Add a branch inventory issue that lists active, stale, and archival branches.
3. Add a follow-up technical issue for reducing eager dependency loading.
4. Keep future commit and PR titles descriptive, using behavior-focused
   summaries rather than `up` or `update`.
