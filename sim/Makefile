# =============================================================================
# Makefile -- NATIVE cocotb + Icarus Verilog sim for the ecp5/ port (no Docker).
# Adapted from sim/link/Makefile.gen (origin/refactor/unified-node-ss-link).
# Run from ecp5/sim/. Requires oss-cad-suite (iverilog) + `pip install cocotb` + make.
#
#   make                                   # default: gen_seq, test_gen_A
#   make MODULE=test_gen_F2                # pick a test
#   make -f Makefile.gen MODULE=test_gen_A # explicit gen-flow
#
# The .hex LUTs are auto-copied into cwd (the $readmemh relative-path trap:
# iverilog resolves $readmemh relative to the sim cwd, NOT the source file).
# =============================================================================
SIM           ?= icarus
TOPLEVEL_LANG ?= verilog
RTL := $(abspath ../src)
LUT := $(abspath ../lut)

# Auto-copy LUT .hex into cwd so $readmemh("silu_lut.hex") etc. resolve.
HEX_FILES := $(notdir $(wildcard $(LUT)/*.hex))
$(HEX_FILES): %: $(LUT)/%
	cp -f $< .
hex: $(HEX_FILES)

VERILOG_SOURCES += $(abspath models/Gowin_rPLL.v)
VERILOG_SOURCES += $(abspath models/MULTALU18X18.v)
VERILOG_SOURCES += $(abspath models/sdram_chip.v)
VERILOG_SOURCES += $(RTL)/mac18.v
VERILOG_SOURCES += $(RTL)/rmsnorm_op.v
VERILOG_SOURCES += $(RTL)/silu_op.v
VERILOG_SOURCES += $(RTL)/rope_op.v
VERILOG_SOURCES += $(RTL)/softmax_op.v
VERILOG_SOURCES += $(RTL)/attention_head_op.v
VERILOG_SOURCES += $(RTL)/sdram.v
VERILOG_SOURCES += $(abspath ../src/async_fifo.v)
VERILOG_SOURCES += $(abspath ../src/uart_bridge.v)
VERILOG_SOURCES += $(abspath ../src/vec_alu2.v)
VERILOG_SOURCES += $(RTL)/top.v
VERILOG_SOURCES += $(abspath ../src/gen_seq.v)
VERILOG_SOURCES += $(abspath ../src/gen_seq_top.v)

COMPILE_ARGS += -DLINK_SS
# Précharge sdram_preload.hex dans sdram_chip.v ($readmemh) au build. Nécessaire
# pour test_gen_F/F2 (cocotb VPI ne persiste pas l'écriture dans le grand reg mem
# sous icarus) -> sinon SDRAM vide -> argmax=0 -> tous tokens=0. Les tests A..E
# préchargent leurs propres poids par VPI qui écrase ces zones au runtime.
COMPILE_ARGS += -DSDRAM_PRELOAD
COMPILE_ARGS += -I$(RTL)

TOPLEVEL := gen_seq_top
MODULE   ?= test_gen_A

# cocotb needs to find local test modules + the .hex present in cwd.
export PYTHONPATH := $(abspath .):$(PYTHONPATH)

# Run `make hex` first (dependency), then the cocotb sim flow.
all: hex
	+$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

include $(shell cocotb-config --makefiles)/Makefile.sim