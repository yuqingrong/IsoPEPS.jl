# Deferred public-release checklist

This checklist is intentionally deferred. Do not perform any item until every
author approves the public release.

- [ ] Confirm final author approval for the code, curated data, figures, and
  citation information.
- [ ] Run the full test suite in the tracked root environment and record any
  version-specific known failures.
- [ ] Rebuild the local staging package, verify `MANIFEST.sha256`, restore the
  archive figures, and cross-check the figure manifest against `main.tex`.
- [ ] Review `CITATION.cff` and complete `repro/zenodo-metadata-template.json`.
- [ ] Bump the package version from `1.0.0-DEV` and update the root manifest if
  the dependency resolution changes.
- [ ] Create a signed/reviewed Git tag and GitHub release only after approval.
- [ ] Create versioned Zenodo records for the MIT-licensed code and the
  CC-BY-4.0 data package, as appropriate.
- [ ] Add the newly assigned DOI(s) to release metadata, citation material, and
  manuscript only after they exist and the authors approve those edits.
- [ ] Publish the GitHub release and Zenodo records, then archive the exact
  release manifests and checksums.
