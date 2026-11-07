.PHONY: format format-check

format:
	@bash scripts/ci/check_cpp_format.sh --fix

format-check:
	@bash scripts/ci/check_cpp_format.sh --check

