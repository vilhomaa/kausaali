package treebased.testutils

/** Unwrap the `Either[Throwable, A]` that the forest APIs return, failing the test on `Left`. */
object EitherOps {
  def orThrow[A](e: Either[Throwable, A]): A =
    e.fold(err => throw new RuntimeException(err.toString), identity)
}
