# Plot targets for postprocess.jl
# Run from the project/ directory: `cd project && make <target>`
#
# Figure outputs are routed by model:
#   - TFIM figures      -> results/tfim_abc/figures/
#   - Heisenberg figures -> results/heisenberg/figures/
#   - Model-agnostic    -> results/figures/   (only circuit-block, circuit-full)

JL    = julia --project=..
SC    = postprocess.jl
RES   = results
FIG   = $(RES)/figures
HRES  = $(RES)/heisenberg
HFIG  = $(HRES)/figures
TRES  = $(RES)/tfim
ABC   = $(RES)/tfim_abc
AFIG  = $(ABC)/figures

# ---------------------------------------------------------------------------
# Paths that contain '=' MUST be defined as variables; otherwise GNU make
# parses any '=' before the first ':' as a variable assignment, swallowing
# both the target and its prerequisites.
# ---------------------------------------------------------------------------

# Heisenberg circuit JSONs used as prerequisites
H_J20  = $(HRES)/circuit_heisenberg_j1j2_J1=1.0_J2=0.0_row=4_p=3_nqubits=3_2x2.json
H_J205 = $(HRES)/circuit_heisenberg_j1j2_J1=1.0_J2=0.5_row=4_p=3_nqubits=3_2x2.json
H_J21  = $(HRES)/circuit_heisenberg_j1j2_J1=1.0_J2=1.0_row=4_p=3_nqubits=3_2x2.json

# Heisenberg sampling JSONs (inputs for structure-factors plot)
H_SAMP_J20  = $(HRES)/samples_heisenberg_J2=0.0.json
H_SAMP_J205 = $(HRES)/samples_heisenberg_J2=0.5.json
H_SAMP_J21  = $(HRES)/samples_heisenberg_J2=1.0.json

# TFIM g=3 circuit JSON (under tfim_abc per the model->dir convention)
T_G3_ABC    = $(ABC)/circuit_tfim_J=1.0_g=3.0_row=3_p=3_nqubits=3_1x1.json

# Heisenberg figure outputs
M2_PDF         = $(HFIG)/M2_comparison.pdf
SF_PDF         = $(HFIG)/structure_factors_combined.pdf
BE_EXACT_PDF   = $(HFIG)/bond_energy_exact.pdf
H_E_PDF        = $(HFIG)/heisenberg_energy_vs_J2.pdf
VAR_PDF        = $(HFIG)/heisenberg_variance_vs_samples_J2=0.5.pdf
ANALYZE_STAMP  = $(HFIG)/.analyze-heisenberg.stamp

# TFIM figure outputs
TFIM_E_PDF     = $(AFIG)/tfim_energy_vs_g.pdf
INV_PDF        = $(AFIG)/tfim_energy_vs_inv_samples_g=3.0.pdf
NNC_PDF        = $(AFIG)/NNconnected_corr_vs_g.pdf
CORR_PDF       = $(AFIG)/corr_length_vs_g_row=3.pdf
MAG_PDF        = $(AFIG)/magnetization_vs_g.pdf
DYN_TFIM_PDF   = $(AFIG)/energy_dynamics_vs_g.pdf
READOUT_E_PDF  = $(AFIG)/readout_noise_energ_g=3.0.pdf
READOUT_B_PDF  = $(AFIG)/readout_noise_energ_error_g=3.0.pdf

# Model-agnostic figure outputs
BLOCK_PDF      = $(FIG)/circuit_block_3x5.pdf
FULL_PDF       = $(FIG)/circuit_full_3x3x5.pdf

# Data-only outputs
VAR_JSON       = $(HRES)/heisenberg_variance_vs_samples.json
READOUT_JSON   = $(ABC)/readout_noise_energ_g=3.0.json

.DEFAULT_GOAL := help

help:  ## list available targets
	@echo "Phony / grouping targets (use 'make <name>'):"
	@echo "  all-plots           build every plot"
	@echo "  energy-plots        TFIM + Heisenberg energy-vs-parameter"
	@echo "  correlation-plots   NN correlation, corr length, magnetization"
	@echo "  dynamics-plots      TFIM training-dynamics plot"
	@echo "  circuit-plots       circuit-block + full channel diagrams"
	@echo "  readout-plots       readout-noise energy + bias"
	@echo "  sf-plots            structure factors / M^2 / bond-energy"
	@echo "  variance-plots      variance vs samples + energy vs 1/samples"
	@echo "  analysis-plots      training-history + expectation-values pair"
	@echo "  analyze-heisenberg  (alias for analysis-plots)"
	@echo "  clean-figures       delete generated PDFs"
	@echo
	@echo "Single-plot aliases (use 'make <alias>'):"
	@echo "  m2-comparison           structure-factors       bond-energy-exact"
	@echo "  energy-vs-g-tfim        energy-vs-J2-heisenberg"
	@echo "  variance-data           variance-vs-samples     energy-vs-inv-samples"
	@echo "  connected-corr-vs-g     corr-length-vs-g        magnetization-vs-g"
	@echo "  energy-dynamics-tfim"
	@echo "  circuit-block           circuit-full"
	@echo "  readout-data            readout-energy          readout-bias"
	@echo
	@echo "Individual figure outputs (use 'make <path>'):"
	@echo "  Heisenberg:"
	@echo "    $(M2_PDF)"
	@echo "    $(SF_PDF)"
	@echo "    $(BE_EXACT_PDF)"
	@echo "    $(H_E_PDF)"
	@echo "    $(VAR_PDF)"
	@echo "  TFIM:"
	@echo "    $(TFIM_E_PDF)"
	@echo "    $(INV_PDF)"
	@echo "    $(NNC_PDF)"
	@echo "    $(CORR_PDF)"
	@echo "    $(MAG_PDF)"
	@echo "    $(DYN_TFIM_PDF)"
	@echo "    $(READOUT_E_PDF)"
	@echo "    $(READOUT_B_PDF)"
	@echo "  Model-agnostic:"
	@echo "    $(BLOCK_PDF)"
	@echo "    $(FULL_PDF)"
	@echo
	@echo "Data-only outputs:"
	@echo "  $(VAR_JSON)"
	@echo "  $(READOUT_JSON)"

# Output directories (order-only prereqs)
$(FIG):
	mkdir -p $@
$(HFIG):
	mkdir -p $@
$(AFIG):
	mkdir -p $@

# ---------------------------------------------------------------------------
# Single-figure rules: <output> : <inputs...>
# ---------------------------------------------------------------------------

# Heisenberg
$(M2_PDF): $(HRES)/M2_sampling.json $(SC) | $(HFIG)
	$(JL) $(SC) m2-comparison

$(SF_PDF): $(HRES)/sf.json $(H_SAMP_J20) $(H_SAMP_J205) $(H_SAMP_J21) $(SC) | $(HFIG)
	$(JL) $(SC) structure-factors

$(BE_EXACT_PDF): $(H_J20) $(SC) | $(HFIG)
	$(JL) $(SC) bond-energy-exact

$(H_E_PDF): $(SC) | $(HFIG)
	$(JL) $(SC) energy-vs-J2-heisenberg

$(VAR_JSON): $(H_J205) $(SC)
	$(JL) $(SC) variance-data

$(VAR_PDF): $(VAR_JSON) $(SC) | $(HFIG)
	$(JL) $(SC) variance-vs-samples

# analyze-heisenberg writes two PDFs in a single run; use a stamp file under HFIG.
$(ANALYZE_STAMP): $(H_J205) $(SC) | $(HFIG)
	$(JL) $(SC) analyze-heisenberg
	@touch $@

analyze-heisenberg analysis-plots: $(ANALYZE_STAMP)

# TFIM
$(TFIM_E_PDF): $(SC) | $(AFIG)
	$(JL) $(SC) energy-vs-g-tfim

$(INV_PDF): $(T_G3_ABC) $(SC) | $(AFIG)
	$(JL) $(SC) energy-vs-inv-samples

$(NNC_PDF): $(SC) | $(AFIG)
	$(JL) $(SC) connected-corr-vs-g

$(CORR_PDF): $(SC) | $(AFIG)
	$(JL) $(SC) corr-length-vs-g

$(MAG_PDF): $(SC) | $(AFIG)
	$(JL) $(SC) magnetization-vs-g

$(DYN_TFIM_PDF): $(SC) | $(AFIG)
	$(JL) $(SC) energy-dynamics-tfim

$(READOUT_JSON): $(T_G3_ABC) $(SC)
	$(JL) $(SC) readout-data

$(READOUT_E_PDF): $(READOUT_JSON) $(SC) | $(AFIG)
	$(JL) $(SC) readout-energy

$(READOUT_B_PDF): $(READOUT_JSON) $(SC) | $(AFIG)
	$(JL) $(SC) readout-bias

# Model-agnostic
$(BLOCK_PDF): $(SC) | $(FIG)
	$(JL) $(SC) circuit-block

$(FULL_PDF): $(SC) | $(FIG)
	$(JL) $(SC) circuit-full

# ---------------------------------------------------------------------------
# Groupings
# ---------------------------------------------------------------------------

energy-plots: $(TFIM_E_PDF) $(H_E_PDF)
correlation-plots: $(NNC_PDF) $(CORR_PDF) $(MAG_PDF)
dynamics-plots: $(DYN_TFIM_PDF)
circuit-plots: $(BLOCK_PDF) $(FULL_PDF)
readout-plots: $(READOUT_E_PDF) $(READOUT_B_PDF)
sf-plots: $(SF_PDF) $(M2_PDF) $(BE_EXACT_PDF)
variance-plots: $(VAR_PDF) $(INV_PDF)

all-plots: energy-plots correlation-plots dynamics-plots circuit-plots readout-plots sf-plots variance-plots analysis-plots

# ---------------------------------------------------------------------------
# Short phony aliases for individual figures
# ---------------------------------------------------------------------------

m2-comparison: $(M2_PDF)
structure-factors: $(SF_PDF)
bond-energy-exact: $(BE_EXACT_PDF)
energy-vs-g-tfim: $(TFIM_E_PDF)
energy-vs-J2-heisenberg: $(H_E_PDF)
variance-data: $(VAR_JSON)
variance-vs-samples: $(VAR_PDF)
energy-vs-inv-samples: $(INV_PDF)
connected-corr-vs-g: $(NNC_PDF)
corr-length-vs-g: $(CORR_PDF)
magnetization-vs-g: $(MAG_PDF)
energy-dynamics-tfim: $(DYN_TFIM_PDF)
circuit-block: $(BLOCK_PDF)
circuit-full: $(FULL_PDF)
readout-data: $(READOUT_JSON)
readout-energy: $(READOUT_E_PDF)
readout-bias: $(READOUT_B_PDF)

clean-figures:
	rm -f $(FIG)/*.pdf $(HFIG)/*.pdf $(AFIG)/*.pdf $(ANALYZE_STAMP)

.PHONY: help all-plots energy-plots correlation-plots dynamics-plots circuit-plots readout-plots sf-plots variance-plots analysis-plots clean-figures analyze-heisenberg \
        m2-comparison structure-factors bond-energy-exact \
        energy-vs-g-tfim energy-vs-J2-heisenberg variance-data variance-vs-samples \
        energy-vs-inv-samples connected-corr-vs-g corr-length-vs-g magnetization-vs-g \
        energy-dynamics-tfim circuit-block circuit-full readout-data readout-energy readout-bias
