# Security Policy

IsoPEPS is a research-oriented Julia package. Please do not open public issues
for suspected vulnerabilities that include sensitive details, credentials, or
private data.

## Reporting

Report security concerns privately to the repository owner or maintainer. If
GitHub private vulnerability reporting is enabled for this repository, use that
path. Otherwise, contact the maintainer directly through the account listed on
the repository.

Please include:

- A short description of the concern.
- Steps to reproduce, if applicable.
- Affected files, commands, or workflows.
- Whether any credentials, generated data, or private research artifacts may be
  involved.

## Supported Versions

The package is currently marked `1.0.0-DEV`, so security fixes are handled on
the active `master` branch until formal releases begin.

## Secrets and Data

Do not commit access tokens, private keys, `.env` files, local settings, or
large generated research outputs. Generated outputs should stay in ignored
directories such as `data/`, `states/`, `image/`, `project/results/`, and
`simulations/results/`.
