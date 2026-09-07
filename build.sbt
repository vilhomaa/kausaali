val circeVersion = "0.14.10"

// ─────────────────────────────────────────────────────────────────────────────
// Project metadata for publishing. `sbt publishLocal` works as-is;
// Maven Central release steps are in CONTRIBUTING.md.
// ─────────────────────────────────────────────────────────────────────────────
ThisBuild / organization     := "io.github.vilhomaa"
ThisBuild / organizationName := "Lassi Vilhomaa"
ThisBuild / scalaVersion     := "3.3.5"
ThisBuild / startYear        := Some(2025)
ThisBuild / versionScheme    := Some("early-semver")

ThisBuild / licenses := Seq(
  "Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0")
)
ThisBuild / homepage := Some(url("https://github.com/vilhomaa/kausaali"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/vilhomaa/kausaali"),
    "scm:git:https://github.com/vilhomaa/kausaali.git",
    "scm:git:git@github.com:vilhomaa/kausaali.git"
  )
)
ThisBuild / developers := List(
  Developer(
    id    = "vilhomaa",
    name  = "Lassi Vilhomaa",
    email = "lassi.vilhomaa@gmail.com",
    url   = url("https://github.com/vilhomaa")
  )
)

// Maven Central via the Sonatype Central Portal. sbt-ci-release derives the
// version from git tags and sets `publishTo`; `sbt ci-release` in CI signs and
// uploads. Snapshots on push to main, full releases on a `v*` tag.
ThisBuild / sonatypeCredentialHost := "central.sonatype.com"
ThisBuild / sonatypeProfileName    := "io.github.vilhomaa"

lazy val root = (project in file("."))
  .settings(
    name        := "kausaali",
    description := "Generalized Random Forests and Causal Forests for CATE estimation, in Scala 3.",

    libraryDependencies ++= Seq(
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.2.0",
      "io.circe"               %% "circe-core"                 % circeVersion,
      "io.circe"               %% "circe-generic"              % circeVersion,
      "io.circe"               %% "circe-parser"               % circeVersion,
      "org.scalatest"          %% "scalatest"                  % "3.2.19" % Test
    ),

    // `sbt publishLocal` writes to ~/.ivy2/local; `sbt ci-release` publishes to
    // Maven Central (see PUBLISHING.md).
    publishMavenStyle      := true,
    Test / publishArtifact := false,
    pomIncludeRepository    := { _ => false }
  )
