JL = julia --project=.

default: init test

help:
	@printf "Common targets:\n"
	@printf "  make init              Precompile the project\n"
	@printf "  make test              Run the full test suite\n"
	@printf "  make test-gates        Run one named test group\n"
	@printf "  make coverage          Run tests with coverage\n"
	@printf "  make format            Format Julia files with JuliaFormatter\n"
	@printf "  make format-check      Check Julia formatting\n"
	@printf "  make clean             Remove Julia coverage files\n"

init:
	$(JL) -e 'using Pkg; Pkg.precompile()'

update:
	$(JL) -e 'using Pkg; Pkg.update(); Pkg.precompile()'

test:
	$(JL) -e 'using Pkg; Pkg.test()'

test-%:
	$(JL) -e 'using Pkg; Pkg.test(; test_args=["$*"])'

coverage:
	$(JL) -e 'using Pkg; Pkg.test(; coverage=true)'

format:
	julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")'

format-check:
	julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; @assert format(".", overwrite=false)'

clean:
	find . -name "*.cov" -type f -print0 | xargs -0 /bin/rm -f

.PHONY: default help init update test coverage format format-check clean
