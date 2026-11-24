# Run the survey analysis in order. Each step runs only if its output file is missing, so finished steps are skipped.
# A step is NOT rerun just because an earlier file changed: that could start hours of work, or score a locked test
# again, by accident. To redo a step, delete its results file (or run its script directly).
#
#   make test        unit tests (no data needed)
#   make stage7      Stage 7  (README 3.2)
#   make stage8      Stage 8  (README 3.5a)
#   make stage29     Stage 29 (README 3.1, 3.3, 3.4, 3.7); Step 8a takes several hours, Step 7 needs TABPFN_TOKEN
#   make stage30     Stage 30 (README 3.1, 3.4-3.6); about 1.5 hours
#   make all         everything above
#   make -n stage30  show what would run, without running it

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
RUN = $(PYTHON) -m src.survey
R = results

.PHONY: all test stage7 stage8 stage29 stage30 help

help:
	@sed -n "5,11p" Makefile

all: stage7 stage8 stage29 stage30

test:
	$(PYTHON) -m pytest -q

# Stages 7-8: the 6-question pipeline
data/processed/zimbabwe.parquet:
	$(RUN).build_dataset
$(R)/survey_audit.json:
	$(RUN).audit
$(R)/stage7_survey_baselines.json: | data/processed/zimbabwe.parquet
	$(RUN).train_baselines
$(R)/stage8_survey_transfer.json: | data/processed/zimbabwe.parquet
	$(RUN).transfer

stage7: $(R)/survey_audit.json $(R)/stage7_survey_baselines.json
stage8: $(R)/stage8_survey_transfer.json

# Stage 29: Zimbabwe, every question (order: later steps read what earlier ones save)
$(R)/stage29_inventory.json:
	$(RUN).select_features
$(R)/stage29_screening.json: | $(R)/stage29_inventory.json
	$(RUN).screen_features
$(R)/stage29_models.json: | $(R)/stage29_screening.json
	$(RUN).select_models
$(R)/stage29_final_test.json: | $(R)/stage29_models.json
	$(RUN).final_test
$(R)/stage29_tabpfn.json: | $(R)/stage29_final_test.json
	$(RUN).tabpfn_cv
$(R)/stage29_repeated_cv.json: | $(R)/stage29_tabpfn.json
	$(RUN).repeated_cv
$(R)/stage29_severity.json: | $(R)/stage29_repeated_cv.json
	$(RUN).severity
$(R)/stage29_area_check.json: | $(R)/stage29_repeated_cv.json $(R)/stage29_final_test.json
	$(RUN).area_check
$(R)/stage29_blank_sensitivity.json: | $(R)/stage29_repeated_cv.json
	$(RUN).blank_sensitivity
$(R)/stage29_hb_sources.json: | $(R)/stage29_severity.json $(R)/stage29_final_test.json
	$(RUN).hb_sources
$(R)/stage29_hb_vs_class.json: | $(R)/stage29_severity.json $(R)/stage29_final_test.json
	$(RUN).hb_vs_class
$(R)/stage29_exploratory.json: | $(R)/stage29_hb_vs_class.json
	$(RUN).exploratory

stage29: $(R)/stage29_severity.json $(R)/stage29_area_check.json $(R)/stage29_blank_sensitivity.json \
         $(R)/stage29_hb_sources.json $(R)/stage29_hb_vs_class.json $(R)/stage29_exploratory.json

# Stage 30: Malawi and Zambia cleaned with the Stage 29 rules, then every country on its own
$(R)/stage30_%_inventory.json:
	$(RUN).select_features $*
$(R)/stage30_%_screening.json: | $(R)/stage30_%_inventory.json
	$(RUN).screen_features $*
$(R)/stage30_countries.json: | $(R)/stage29_screening.json $(R)/stage30_malawi_screening.json \
                               $(R)/stage30_zambia_screening.json
	$(RUN).countries

stage30: $(R)/stage30_countries.json
