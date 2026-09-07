# Contributing to Kausaali

Thanks for your interest in contributing.

Possible next steps & further implementations:
- iv/quantile forests from Generalized Random Forests (Athey, Tibshirani and Wager 2018)
- Local linear forests (Athey, Friedberg, Tibshirani and Wager, 2021)

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

## Releasing (maintainers)

Releases go to Maven Central from GitHub Actions via `sbt-ci-release`. The build
version is derived from the git tag by `sbt-dynver` — it is never set in
`build.sbt`. Pushing to `main` publishes a `-SNAPSHOT`; pushing a `vX.Y.Z` tag
publishes the release.

To cut a release:

1. Move the `CHANGELOG.md` `[Unreleased]` entries under a new `[X.Y.Z]` heading
   with today's date; commit and push to `main`.
2. Tag and push — the version must never have been published before, Central
   rejects re-publishing a coordinate:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

3. Watch the **Publish** job in the Actions tab. It runs `publishSigned` then
   `sonatypeBundleRelease`; the artifact is searchable on Central ~10–30 min later.

### One-time / rotation

Publishing needs four repo secrets (Settings → Secrets and variables → Actions):

| Secret              | Value                                                |
| ------------------- | --------------------------------------------------- |
| `PGP_SECRET`        | `gpg --armor --export-secret-keys <KEYID> \| base64` |
| `PGP_PASSPHRASE`    | the signing key passphrase                           |
| `SONATYPE_USERNAME` | username half of the Sonatype Central user token     |
| `SONATYPE_PASSWORD` | password half of the Sonatype Central user token     |

The signing key is RSA-4096 with a 2-year expiry. To rotate it: generate a new
key (`gpg --quick-generate-key "Lassi Vilhomaa <lassi.vilhomaa@gmail.com>"
rsa4096 sign 2y`), push the public half to `keyserver.ubuntu.com` and
`keys.openpgp.org`, and update `PGP_SECRET` / `PGP_PASSPHRASE`.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, the same license that covers this project (see `LICENSE`).
