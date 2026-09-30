"""The measure stage: served and in-process scoring, grounding snapshot, probe and slices.

* :mod:`~jev_factory.measure.run` -- measure a scorer on a split (predictions, metrics, page);
* :mod:`~jev_factory.measure.serve` -- serve a model dir (vLLM) or GGUF (llama-server);
* :mod:`~jev_factory.measure.preflight` -- the served model and context must be right;
* :mod:`~jev_factory.measure.once` -- the test and held-out sides are measured once;
* :mod:`~jev_factory.measure.snapshot` -- one fixed grounding world;
* :mod:`~jev_factory.measure.probe` -- the permutation probe;
* :mod:`~jev_factory.measure.slices` -- the missing-candidate slice;
* :mod:`~jev_factory.measure.corpus` / :mod:`~jev_factory.measure.predictions` -- entries in,
  predictions lines out.
"""
