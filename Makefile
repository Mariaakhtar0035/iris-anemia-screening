# Run the survey analysis in order. Each step runs only if its output file is missing, so finished steps are skipped.
# A step is NOT rerun just because an earlier file changed: that could start hours of work, or score a locked test
# again, by accident. To redo a step, delete its results file (or run its script directly).
#
#   make test        unit tests (no data needed)
#   make baseline      Stage 7  (README 3.2)
#   make transfer      Stage 8  (README 3.5a)
#   make zimbabwe     Stage 29 (README 3.1, 3.3, 3.4, 3.7); Step 8a takes several hours, Step 7 needs TABPFN_TOKEN
#   make countries     Stage 30 (README 3.1, 3.4-3.6); about 1.5 hours
#   make severity    severity curve from the saved Stage 30 predictions (README 3.4); a few minutes
#   make trisha      Trisha's checks: data funnel, within-country overfit test, rules vs models (README 1, 3.5d, 3.7)
#   make covariates     village covariates, Zimbabwe (needs ZWGC72FL in data/raw); reads Stage 30 predictions
#   make all         everything above
#   make -n countries  show what would run, without running it

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
RUN = $(PYTHON) -m src.survey
R = results

.PHONY: all test baseline transfer zimbabwe countries severity trisha covariates help

help:
	@sed -n "5,14p" Makefile

all: baseline transfer zimbabwe countries severity trisha covariates

test:
	$(PYTHON) -m pytest -q

# Stages 7-8: the 6-question pipeline
data/processed/zimbabwe.parquet:
	$(RUN).build_dataset
$(R)/survey_audit.json:
	$(RUN).audit
$(R)/six_question_baseline.json: | data/processed/zimbabwe.parquet
	$(RUN).train_baselines
$(R)/zimbabwe_transfer.json: | data/processed/zimbabwe.parquet
	$(RUN).transfer

baseline: $(R)/survey_audit.json $(R)/six_question_baseline.json
transfer: $(R)/zimbabwe_transfer.json

# Stage 29: Zimbabwe, every question (order: later steps read what earlier ones save)
$(R)/zimbabwe_inventory.json:
	$(RUN).select_features
$(R)/zimbabwe_screening.json: | $(R)/zimbabwe_inventory.json
	$(RUN).screen_features
$(R)/zimbabwe_models.json: | $(R)/zimbabwe_screening.json
	$(RUN).select_models
$(R)/zimbabwe_final_test.json: | $(R)/zimbabwe_models.json
	$(RUN).final_test
$(R)/zimbabwe_tabpfn.json: | $(R)/zimbabwe_final_test.json
	$(RUN).tabpfn_cv
$(R)/zimbabwe_repeated_cv.json: | $(R)/zimbabwe_tabpfn.json
	$(RUN).repeated_cv
$(R)/zimbabwe_severity.json: | $(R)/zimbabwe_repeated_cv.json
	$(RUN).severity
$(R)/zimbabwe_area_check.json: | $(R)/zimbabwe_repeated_cv.json $(R)/zimbabwe_final_test.json
	$(RUN).area_check
$(R)/zimbabwe_blank_sensitivity.json: | $(R)/zimbabwe_repeated_cv.json
	$(RUN).blank_sensitivity
$(R)/zimbabwe_hb_sources.json: | $(R)/zimbabwe_severity.json $(R)/zimbabwe_final_test.json
	$(RUN).hb_sources
$(R)/zimbabwe_hb_vs_class.json: | $(R)/zimbabwe_severity.json $(R)/zimbabwe_final_test.json
	$(RUN).hb_vs_class
$(R)/zimbabwe_exploratory.json: | $(R)/zimbabwe_hb_vs_class.json
	$(RUN).exploratory

zimbabwe: $(R)/zimbabwe_severity.json $(R)/zimbabwe_area_check.json $(R)/zimbabwe_blank_sensitivity.json \
         $(R)/zimbabwe_hb_sources.json $(R)/zimbabwe_hb_vs_class.json $(R)/zimbabwe_exploratory.json

# Stage 30: Malawi and Zambia cleaned with the Stage 29 rules, then every country on its own
$(R)/%_inventory.json:
	$(RUN).select_features $*
$(R)/%_screening.json: | $(R)/%_inventory.json
	$(RUN).screen_features $*
$(R)/three_country_comparison.json: | $(R)/zimbabwe_screening.json $(R)/malawi_screening.json \
                               $(R)/zambia_screening.json
	$(RUN).countries

countries: $(R)/three_country_comparison.json

$(R)/severity_curve.json: | $(R)/three_country_comparison.json
	$(RUN).severity_curve

severity: $(R)/severity_curve.json

# Trisha's checks (2026-10-26/27), on the 6-question pipeline of Stages 7-8
$(R)/funnel.csv: | data/processed/zimbabwe.parquet
	$(RUN).funnel
$(R)/within_country.json: | data/processed/zimbabwe.parquet
	$(RUN).within_country
$(R)/rules_transfer.json: | data/processed/zimbabwe.parquet
	$(RUN).rules_transfer

trisha: $(R)/funnel.csv $(R)/within_country.json $(R)/rules_transfer.json

# Stage 31: village geospatial covariates (Zimbabwe)
$(R)/covariates.json: | $(R)/three_country_comparison.json
	$(RUN).covariates

covariates: $(R)/covariates.json
