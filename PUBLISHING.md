# Publishing

## Status

The build is set up for **local publishing** today:

```bash
sbt publishLocal        # writes to ~/.ivy2/local
sbt +publishLocal       # if/when cross-building is added
```

Other projects (e.g. `kausaali-benchmarks`) can then depend on it with:

```scala
libraryDependencies += "io.github.vilhomaa" %% "kausaali" % "0.1.0-SNAPSHOT"
```

## Before the first public release

1. **Pick real Maven coordinates.** In `build.sbt`, replace every `vilhomaa`:
   - `organization` — the group id. The no-domain path is
     `io.github.<your-github-username>` (works with Sonatype's automatic
     namespace verification for GitHub accounts).
   - `homepage`, `scmInfo`, `developers` URLs.
2. **Set a release version.** Drop `-SNAPSHOT` (e.g. `0.1.0`) and tag it
   `v0.1.0`. Update `CHANGELOG.md`.
3. **Add the Copyright line** to source file headers if you want per-file
   notices (optional; `LICENSE` + `NOTICE` cover the project as a whole).

## Maven Central (Sonatype) — when ready

Add to `project/plugins.sbt`:

```scala
addSbtPlugin("com.github.sbt" % "sbt-pgp"      % "2.3.1")
addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "3.11.3")
// or, for a one-command CI release:
// addSbtPlugin("com.github.sbt" % "sbt-ci-release" % "1.9.2")
```

Then:

- Create a Sonatype Central account and verify the `io.github.<user>` namespace.
- Generate a GPG key, publish it to a keyserver, and store the secret key +
  passphrase (locally in `~/.sbt/gpg/` or as CI secrets).
- `sbt clean test publishSigned sonatypeBundleRelease`.

`versionScheme := Some("early-semver")` is already set so downstream eviction
warnings behave sensibly while the library is on `0.x`.
