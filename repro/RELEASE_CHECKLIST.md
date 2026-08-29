# Public-release checklist

The `1.0.0` release candidate includes `.zenodo.json` for the code record and
`zenodo-data-metadata-template.json` for the curated data record. Do not
perform any public-release item until every author approves the release.

- [ ] Confirm final author approval for the code, curated data, figures, and
  citation information.
- [ ] Run the full test suite in the tracked root environment and record any
  version-specific known failures.
- [ ] Rebuild the local staging package, verify `MANIFEST.sha256`, restore the
  archive figures, and cross-check the figure manifest against `main.tex`.
- [ ] Review `.zenodo.json`, `CITATION.cff`, and
  `repro/zenodo-data-metadata-template.json`; add only DOI values that have
  been reserved or assigned for these exact records.
- [ ] Confirm that the root `Project.toml` still matches the `1.0.0` release
  candidate and that no root `Manifest.toml` is tracked.
- [ ] Create a signed/reviewed Git tag and GitHub release only after approval.
- [ ] Create versioned Zenodo records for the MIT-licensed code and the
  CC-BY-4.0 data package, as appropriate.
- [ ] Add the newly assigned DOI(s) to release metadata, citation material, and
  manuscript only after they exist and the authors approve those edits.
- [ ] Publish the GitHub release and Zenodo records, then archive the exact
  release manifests and checksums.
