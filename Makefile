.DEFAULT_GOAL := all

# Keep each saver's compiler flags and generated-protocol rules in its own
# Makefile. Only discover immediate children; build/ contains optional tools.
NATIVE_SAVERS := $(patsubst %/Makefile,%,$(wildcard */Makefile))
ACTIVE_SAVERS := $(filter $(NATIVE_SAVERS),$(shell bash -c '. ./savers.sh; printf "%s\n" "$${SAVERS[@]}"'))
BENCHED_SAVERS := $(filter $(NATIVE_SAVERS),$(shell bash -c '. ./savers.sh; printf "%s\n" "$${BENCHED[@]}"'))
CLEAN_TARGETS := $(addprefix clean-,$(NATIVE_SAVERS))

# Saver names are directories, so these targets must always enter the child
# Makefile. The child decides whether its binary actually needs rebuilding.
.PHONY: all active benched clean check help $(NATIVE_SAVERS) $(CLEAN_TARGETS)

all: $(NATIVE_SAVERS)

active: $(ACTIVE_SAVERS)

benched: $(BENCHED_SAVERS)

# Recursive make preserves command-line overrides and the make -j jobserver.
$(NATIVE_SAVERS):
	+$(MAKE) -C $@

clean: $(CLEAN_TARGETS)

$(CLEAN_TARGETS): clean-%:
	+$(MAKE) -C $* clean

check:
	python3 -B -m unittest discover -s tests -v

help:
	@printf '%s\n' \
	  'make [-jN]         Build all native C savers (including benched)' \
	  'make <saver>       Build one saver, e.g. make papercut' \
	  'make active        Build native savers in the rotation' \
	  'make benched       Build native savers on the bench' \
	  'make clean         Remove all native binaries and generated protocols' \
	  'make clean-<saver> Clean one saver, e.g. make clean-papercut' \
	  'make check         Run Python tests' \
	  '' \
	  'pipes/maze and the optional tools in build/ are managed separately.' \
	  'Compiler overrides are forwarded, e.g. make papercut CC=clang.'
