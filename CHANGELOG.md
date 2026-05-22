# Changelog

This project does not have tagged releases yet. Notable changes should be
recorded here before the first release.

## Unreleased

- Add contributor-facing repository hygiene docs and templates.
- Document setup, quick-start usage, test commands, generated-output policy,
  and release expectations.

## Release Process

Before tagging a release:

1. Update `version` in `Project.toml`.
2. Summarize user-facing changes in this changelog.
3. Run the full test suite with `make test`.
4. Create a Git tag matching the package version.
5. Let TagBot handle Julia registry tag automation when applicable.
