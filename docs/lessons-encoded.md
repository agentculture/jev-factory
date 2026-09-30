# Lessons encoded

Every build of a jev-like model costs days of GPU and reviewer time, and
nvsh's Tool-Jev build showed that the expensive failures were *silent*: the
process lived in prose, in scripts and in a person's head. jev-factory
encodes each lesson as a check in code with a test that fails when the check
is violated. This page maps each failure to its nvsh record (by issue and
ledger id), what the before state looked like, why it mattered, and the
jev-factory tests that now prevent it.

Test references are `tests/<file>::<test name>`. A test marked
`@pytest.mark.behavioral("oN")` asserts the named behavioural obligation.

## Mid-run rule change (nvsh #46 b1-rule)

- **Before.** The selection rule and the bars were prose. An overfit-prone
  rule was chosen after seeing results, and nothing stopped a later edit.
  Frozen hashes appeared only as 16-hex prefixes in the decision ledger.
- **Why it matters.** A rule picked after the numbers are known selects the
  noise, so the final figures are optimistic and cannot be trusted.
- **Now.** The bars and the rule are a machine-readable, sha256-hashed
  pre-registration. Training, selection and `jev decide` refuse a run whose
  pre-registration is missing or changed, unless a deviation id is recorded.
  The rule's order and candidates must match the registered ones.
- **Tests.**
  - `tests/test_prereg.py::test_each_bar_records_stock_minimum_and_stricter` (o12)
  - `tests/test_prereg.py::test_register_hashes_file_into_run` (o11)
  - `tests/test_prereg.py::test_register_refuses_invalid_and_train_refuses_unregistered` (o11)
  - `tests/test_prereg.py::test_changed_file_refused_until_deviation_id` (o11)
  - `tests/test_causal_lm_train.py::test_training_refuses_without_a_registered_preregistration`
  - `tests/test_decide_rules.py::test_rule_order_must_match_the_registered_one`
  - `tests/test_decide_rules.py::test_unregistered_candidate_is_refused`
  - `tests/test_decide_records.py::test_load_detects_a_rewritten_earlier_record`
  - `tests/test_decide_records.py::test_human_override_is_a_new_record_and_never_an_edit`

## Leakage into the protected side (nvsh #39 l5)

- **Before.** Protected-side rows (held-out and test) could reappear in
  training data as exact or near duplicates, and the check was a script
  that a run could skip or that printed the leaked text it found.
- **Why it matters.** A model that has seen the protected set scores
  perfectly on it, so the sealed measurement proves nothing.
- **Now.** Every protected file is checked by every rule (exact and
  near-duplicate). An unreadable or malformed protected file, or an entry
  with no usable text, fails closed, and only ids are ever printed. The sealed
  set is exposed as ids, counts and a hash, never text, and is measured once
  per side and slice unless a deviation is recorded.
- **Tests.**
  - `tests/test_core_leakage.py::test_every_protected_file_is_checked_by_every_rule_and_only_ids_are_printed` (o16)
  - `tests/test_core_leakage.py::test_unreadable_or_malformed_protected_file_fails_closed_without_echoing_content` (o16)
  - `tests/test_core_leakage.py::test_a_text_that_normalises_to_nothing_is_refused_not_passed`
  - `tests/test_core_sealed.py::test_sealed_loader_exposes_only_ids_counts_and_sha256` (o10)
  - `tests/test_measure_run.py::test_a_second_sealed_measurement_without_a_deviation_exits_non_zero` (o10)
  - `tests/test_release_bundle.py::test_the_dataset_bundle_refuses_a_train_row_repeating_a_held_apart_entry`

## Silent merge of the adapter (nvsh #46 l4)

- **Before.** A PEFT merge could load without applying the LoRA weights
  (missing adapter keys, or a vision-language wrapper whose keys did not
  map). The script succeeded and returned the base model.
- **Why it matters.** The run then "trained" for 26 minutes and every
  downstream number described the untuned base model.
- **Now.** The merge is verified, never trusted. A missing-adapter-key
  warning (in any spelling, from warnings or log records), an adapter tensor
  that did not load, a sampled merged weight equal to the base, or a merge
  that changed nothing all fail the stage. The merge must be the text-only
  causal LM.
- **Tests.**
  - `tests/test_causal_lm_train.py::test_a_clean_merge_passes_and_reports_zero_missing_keys` (o15)
  - `tests/test_causal_lm_train.py::test_a_missing_adapter_key_warning_fails_the_merge` (o15)
  - `tests/test_causal_lm_train.py::test_every_spelling_of_the_missing_key_warning_is_caught` (o15)
  - `tests/test_causal_lm_train.py::test_a_sampled_merged_weight_equal_to_the_base_fails_the_merge` (o15)
  - `tests/test_causal_lm_train.py::test_a_merge_that_changed_nothing_at_all_fails` (o15)
  - `tests/test_causal_lm_train.py::test_an_adapter_tensor_that_did_not_load_fails_even_without_a_warning` (o15)

## Hand-copied bundle files (nvsh issue #4 stage 16)

- **Before.** `calibration.json`, `gate.json` and the training set actually
  used were copied into the bundle by hand, so bundles shipped without them,
  with symlinks, or with an absolute path baked into the GGUF imatrix
  metadata.
- **Why it matters.** A bundle without its calibration and gate cannot be
  used safely, a path leaks the builder's machine, and an unverified upload
  cannot be trusted to be the file that was measured.
- **Now.** The bundle stage builds the bundle from the run and refuses it
  without `calibration.json`, `gate.json` and `scorer-train.json`, or with a
  symlink or an absolute imatrix path. The bundle scans clean, uploads
  privately and is fetched back and hash-verified.
- **Tests.**
  - `tests/test_release_bundle.py::test_a_bundle_without_calibration_gate_or_scorer_train_is_refused` (o18)
  - `tests/test_release_bundle.py::test_a_finished_bundle_missing_a_required_file_fails_the_check` (o18)
  - `tests/test_release_bundle.py::test_a_symlink_in_a_bundle_refuses_it_at_build_and_at_check` (o18)
  - `tests/test_release_bundle.py::test_an_absolute_path_in_gguf_imatrix_metadata_refuses_the_bundle` (o18)
  - `tests/test_release_bundle.py::test_the_model_bundle_carries_everything_and_the_card_comes_from_domain_and_config`
  - `tests/test_release_bundle.py::test_a_built_bundle_scans_clean_and_uploads_through_the_fake_hub`

## Stale snapshot and stale stage state

The record for this failure is nvsh #53 lapse l9: a stale grounding
snapshot was missing 23 services and 11 containers (issue #4, stage 6),
because nothing tied a measurement to the snapshot it used. Behind it is
the absence of any stage state in nvsh's pipeline (issue #4, stages and
"Shell and automation traps"): a stage could be re-used after its inputs,
knobs or grounding snapshot had changed, and a frozen input was identified
only by a 16-hex prefix.

- **Before.** The pipeline kept no stage-state file, so a finished stage
  looked fresh even when its inputs had moved. The grounding snapshot a
  measurement used was not recorded, and a bundle was not tied to the
  candidate surface it was trained on.
- **Why it matters.** A stale stage or a different world makes two
  measurements incomparable while looking identical.
- **Now.** Every stage writes a manifest with full sha256 input hashes.
  Changing an input or a knob marks the stage and everything downstream
  stale, while a rerun with unchanged inputs is a no-op. The grounding
  snapshot is schema-checked and its hash is recorded in the measurement
  page. A bundle records the domain's surface hash, and a mismatch with the
  running surface is reported.
- **Tests.**
  - `tests/test_factory_stages.py::test_rerun_unchanged_is_noop` (o23)
  - `tests/test_factory_stages.py::test_input_change_marks_stage_and_downstream_stale` (o23)
  - `tests/test_factory_stages.py::test_midchain_change_leaves_upstream_fresh` (o23)
  - `tests/test_factory_stages.py::test_knob_change_is_stale_and_failure_recorded`
  - `tests/test_measure_snapshot.py::test_load_returns_the_snapshot_and_its_hash`
  - `tests/test_measure_snapshot.py::test_a_malformed_snapshot_is_refused`
  - `tests/test_measure_run.py::test_a_ground_snapshot_grounds_every_proposal_and_its_hash_is_recorded`
  - `tests/test_release_bundle.py::test_a_bundle_records_the_jev_cli_surface_hash_and_ask_reports_a_mismatch` (o36)
  - `tests/test_jev_cli_domain.py::test_surface_hash_is_domain_hash_and_tracks_the_surface`

## Parser blacklist for reviewer verdicts (nvsh issue #4 stage 3)

- **Before.** The reviewer's free-text reply was parsed by a blacklist of
  hedge words. It misread "no-argument", "ambiguous" and a mid-sentence
  "but" as rejections and dropped good training rows.
- **Why it matters.** Silent false rejects starve the classes the model most
  needs, and a teacher that returns junk looked the same as one that said no.
- **Now.** Reviewers return a JSON verdict parsed against a schema. A
  malformed or empty reply is retried twice and then recorded as an error,
  never as a reject. Deterministic guards still reject whatever a reviewer
  said.
- **Tests.**
  - `tests/test_teachers.py::test_verdict_parses_from_json` (o20)
  - `tests/test_teachers.py::test_malformed_or_empty_is_a_teacher_error_not_a_reject` (o20)
  - `tests/test_teachers.py::test_reviewer_retries_twice_then_records_an_error` (o20)
  - `tests/test_teachers.py::test_reviewer_recovers_on_retry` (o20)
  - `tests/test_data_augment.py::test_no_argument_ambiguous_and_but_in_a_reason_do_not_reject`
  - `tests/test_data_augment.py::test_a_malformed_verdict_is_an_error_not_a_reject`
  - `tests/test_data_augment.py::test_deterministic_guards_reject_whatever_the_reviewers_said`
