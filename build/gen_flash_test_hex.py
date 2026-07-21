#!/usr/bin/env python3
# gen_flash_test_hex.py -- produce a $readmemh hex file for the W25Q64 mock.
#
# Layout (flash byte addresses):
#   0x120000 .. 0x18003F  : int8 model image  (FLASH_LEN = 0x060040 bytes)
#   0x180040 .. 0x180FFF  : unused gap        (zero filled)
#   0x181000 .. 0x18102E  : 47-byte shift trailer (SHIFT_BASE)
#
# The hex file lists bytes in increasing address order starting at 0x120000,
# one byte per line as 2-hex-digit.  iverilog's $readmemh with the 3rd arg
# `start_address` then loads them into mem[0x120000 .. ].
#
# Pattern is deterministic-but-non-trivial so an order/endianness bug in the
# DMA cannot hide:  byte = (i*7 + 13) & 0xFF for the model region, and
# 200+i for the 47 shift bytes (so they are visually distinct from model
# bytes which stay in 0..255 but with a different stride).

import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "model_flash.hex"

FLASH_BASE = 0x120000
MODEL_LEN  = 0x060040      # match flash_boot.v default FLASH_LEN
SHIFT_BASE = 0x181000
SHIFT_LEN  = 47

TOTAL = SHIFT_BASE + SHIFT_LEN - FLASH_BASE   # bytes to emit

def byte_for(i):
    # i is the byte index relative to FLASH_BASE
    abs_addr = FLASH_BASE + i
    if SHIFT_BASE <= abs_addr < SHIFT_BASE + SHIFT_LEN:
        return (200 + (abs_addr - SHIFT_BASE)) & 0xFF
    if i < MODEL_LEN:
        return (i * 7 + 13) & 0xFF
    return 0x00  # gap

with open(OUT, "w") as f:
    for i in range(TOTAL):
        f.write(f"{byte_for(i):02x}\n")

print(f"wrote {OUT}: {TOTAL} bytes starting at 0x{FLASH_BASE:06x}")
print(f"  model region : 0x{FLASH_BASE:06x} .. 0x{FLASH_BASE+MODEL_LEN-1:06x} ({MODEL_LEN} bytes)")
print(f"  shift trailer: 0x{SHIFT_BASE:06x} .. 0x{SHIFT_BASE+SHIFT_LEN-1:06x} ({SHIFT_LEN} bytes)")