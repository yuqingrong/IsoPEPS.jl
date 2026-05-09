# release

Create a new package release for IsoPEPS.jl.

## Usage
```
/release [version] [options]
```

## Options
- `version`: Semantic version (e.g., 1.2.0) or bump type (major, minor, patch)
- `--dry-run`: Preview changes without committing
- `--skip-tests`: Skip test suite (not recommended)
- `--no-tag`: Don't create git tag

## Workflow

1. **Pre-release checks**
   - Verify on correct branch (master)
   - Check working directory is clean
   - Ensure all tests pass
   - Verify CI is passing
   - Check for uncommitted changes

2. **Determine version**
   - Parse current version from Project.toml
   - Calculate new version based on input
   - Validate semantic versioning rules
   - Check version doesn't already exist

3. **Update version**
   - Modify Project.toml
   - Update version number
   - Commit version change

4. **Run tests**
   - Execute full test suite
   - Verify all tests pass
   - Check for deprecation warnings
   - Ensure examples run

5. **Create release**
   - Create git tag with version
   - Push tag to remote
   - Generate release notes
   - Create GitHub release (if applicable)

6. **Post-release**
   - Verify tag was created
   - Check GitHub release
   - Update documentation if needed
   - Announce release (optional)

## Version Bumping Rules

- **Patch** (1.0.0 → 1.0.1): Bug fixes, no API changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Major** (1.0.0 → 2.0.0): Breaking changes

## Example Usage

```bash
# Bump patch version
/release patch

# Bump minor version
/release minor

# Specific version
/release 1.2.0

# Dry run
/release minor --dry-run
```

## Release Checklist

- [ ] All tests pass locally
- [ ] CI is green
- [ ] Documentation is up to date
- [ ] CHANGELOG is updated (if exists)
- [ ] No uncommitted changes
- [ ] Version number follows SemVer
- [ ] Breaking changes are documented
- [ ] Examples still work

## Git Commands

```bash
# Update version in Project.toml
# (done programmatically)

# Commit version change
git add Project.toml
git commit -m "Release v1.2.0"

# Create and push tag
git tag v1.2.0
git push origin master --tags
```

## Release Notes Template

```markdown
# IsoPEPS.jl v1.2.0

## New Features
- Added new quantum gates: Rx, Ry, Rz
- Implemented parallel optimization

## Improvements
- 30% faster tensor contractions
- Better memory efficiency in training

## Bug Fixes
- Fixed edge case in observable computation
- Corrected entanglement entropy calculation

## Breaking Changes
- None

## Dependencies
- Updated ITensors to 0.7
- Added CMAEvolutionStrategy 0.3
```

## Success Criteria
- Version is updated in Project.toml
- All tests pass
- Git tag is created and pushed
- Release notes are generated
- No errors during release process

## Rollback Procedure

If release fails:
```bash
# Delete local tag
git tag -d v1.2.0

# Delete remote tag (if pushed)
git push origin :refs/tags/v1.2.0

# Reset version change
git reset --hard HEAD~1
```
