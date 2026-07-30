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

# node_top.v = module `top` ss-link+KV (path silicium, sim-compatible via mocks).
# L'ancien monolithe UART-direct (src/legacy/top_gowin_monolith.v, ex-src/top.v,
# desormais hors de tout flow) n'a PAS les ports lk_rx/tx+kv_* que gen_seq_top.v
# connecte -> sim cassée depuis l'étape 1 KV SDRAM. On pointe sur node_top.v
# (mêmes ports ss-link+KV), cache controller inactif sans FLASH_BOOT (reads FFN ->
# fallback SDRAM = path sim historique, bit-exact vs SDRAM_PRELOAD). Voir mémoire
# sim-top-uart-direct-stale / rtl-dim-parametrable-validated.
# Gowin_rPLL.v -> ECP5_PLL_sim.v : node_top instancie `ecp5_pll` (mock sim), pas
# `Gowin_rPLL` (qui était l'ancienne ALU UART-direct top.v).
VERILOG_SOURCES += $(abspath models/ECP5_PLL_sim.v)
VERILOG_SOURCES += $(abspath models/MULTALU18X18.v)
VERILOG_SOURCES += $(abspath models/sdram_chip.v)
VERILOG_SOURCES += $(RTL)/mac18.v
VERILOG_SOURCES += $(RTL)/rmsnorm_op.v
VERILOG_SOURCES += $(RTL)/silu_op.v
VERILOG_SOURCES += $(RTL)/rope_op.v
VERILOG_SOURCES += $(RTL)/softmax_op.v
VERILOG_SOURCES += $(RTL)/attention_head_op.v
VERILOG_SOURCES += $(RTL)/shared_lut.v
VERILOG_SOURCES += $(RTL)/sdram.v
VERILOG_SOURCES += $(abspath ../src/async_fifo.v)
VERILOG_SOURCES += $(abspath ../src/uart_bridge.v)
VERILOG_SOURCES += $(abspath ../src/vec_alu2.v)
VERILOG_SOURCES += $(RTL)/node_top.v
VERILOG_SOURCES += $(RTL)/wcache_ffn.v
VERILOG_SOURCES += $(abspath ../src/gen_seq.v)
VERILOG_SOURCES += $(abspath ../src/gen_seq_top.v)

COMPILE_ARGS += -DLINK_SS
# Précharge sdram_preload.hex dans sdram_chip.v ($readmemh) au build. Nécessaire
# pour test_gen_F/F2 (cocotb VPI ne persiste pas l'écriture dans le grand reg mem
# sous icarus) -> sinon SDRAM vide -> argmax=0 -> tous tokens=0. Les tests A..E
# préchargent leurs propres poids par VPI qui écrase ces zones au runtime.
COMPILE_ARGS += -DSDRAM_PRELOAD
COMPILE_ARGS += -I$(RTL)

# WAVES=1 : active le dump VCD dans gen_seq_top.v (ifdef WAVES) -> waves.vcd pour
# GTKWave. Inactif sinon (pas de overhead ni de fichier généré).
ifeq ($(WAVES),1)
    COMPILE_ARGS += -DWAVES
endif

TOPLEVEL := gen_seq_top
MODULE   ?= test_gen_A

# cocotb needs to find local test modules + the .hex present in cwd.
export PYTHONPATH := $(abspath .):$(PYTHONPATH)

# Run `make hex` first (dependency), then the cocotb sim flow.
all: hex
	+$(MAKE) -f $(shell cocotb-config --makefiles)/Makefile.sim

include $(shell cocotb-config --makefiles)/Makefile.sim