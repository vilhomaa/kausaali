# Contributing to Kausaali

Thanks for your interest in contributing.

## Getting started

```bash
sbt compile      # build
sbt test         # run the ScalaTest suites
```

Scala 3.3 (LTS), sbt 1.10. No other toolchain is required for the library
itself; the cross-framework benchmarks live in a separate repository
(`kausaali-benchmarks`) and need R + Python.

## Pull requests

- Keep changes focused; one logical change per PR.
- Add or update tests under `src/test/scala/` for any behavioural change.
- Match the surrounding code style (the existing files are the reference —
  naming, comment density, formatting).
- Update `CHANGELOG.md` under `[Unreleased]`.
- Make sure `sbt test` passes before opening the PR.

## Reporting bugs

Open an issue with a minimal reproduction: the config used, the shape of the
input data, and what you expected versus what happened.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, the same license that covers this project (see `LICENSE`).
