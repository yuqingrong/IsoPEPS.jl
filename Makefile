JL = julia --project

default: data_plots

init:
	$(JL) -e 'using Pkg; Pkg.precompile()'

update:
	$(JL) -e 'using Pkg; Pkg.update(); Pkg.precompile()'

test:
	$(JL) -e 'using Pkg; Pkg.test()'

coverage:
	$(JL) -e 'using Pkg; Pkg.test(; coverage=true)'

serve:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); using LiveServer; servedocs(;skip_dirs=["docs/src/assets", "docs/src/generated"])'

data_plots:
	@echo "Running data plots generation..."
	@mkdir -p data scripts
	$(JL) --startup-file=no scripts/run_data_plots.jl

clean:
	rm -rf docs/build
	rm -f data/energy_error_plot.png
	rm -f energy_errorvs_g.png
	rm -f cost_vs_iteration*.png
	rm -f new_cost_vs_iteration.png
	rm -f vqe_energies_p*.txt
	find . -name "*.cov" -type f -print0 | xargs -0 /bin/rm -f

.PHONY: init test coverage serve clean update data_plots
