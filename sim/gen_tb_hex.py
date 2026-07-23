"""gen_tb_hex.py -- exporte un gen_model_dump JSON en fichiers hex pour le TB
Verilog pur tb_gen_seq_top_fb.v (flot sans cocotb, iverilog ou Verilator).

Ecrit (dans --out-dir, default = ce rep) :
  tb_meta.hex     : 1 ligne  = NTOK (nb de tokens attendus)
  tb_shifts.hex   : 11 lignes = ports sw_* packed (layer0 dans [7:0])
                    ordre : sw_rms, swq, swk, swv, swo,
                           sw_rmsf, sw1, sw3, sw2, sw_rmsfinal, sw_emb
  tb_expected.hex : NTOK lignes = tokens attendus (node_tokens[:NTOK])

Usage :
  python gen_tb_hex.py --dump ../host/gen_model_dump.json --ntok 17
  python gen_tb_hex.py --dump ../host/gen_model_dump_hs16_rand.json --ntok 17
"""
import json, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ntok", type=int, default=17)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    with open(a.dump) as f:
        m = json.load(f)
    NL = int(m["cfg"]["n_layers"])
    tok = m["node_tokens"][:a.ntok]
    assert len(tok) == a.ntok, f"dump n'a que {len(m['node_tokens'])} tokens, NTOK={a.ntok}"

    def pack(wk):
        # layer0 -> [7:0], layer1 -> [15:8], ... (identique a cocotb drive_shifts.pack)
        return sum((int(m["layers"][l][wk + "_s"]) & 0xFF) << (8 * l) for l in range(NL))

    nl_ports = [("rms_att", "sw_rms"), ("wq", "swq"), ("wk", "swk"), ("wv", "swv"),
                ("wo", "swo"), ("rms_ffn", "sw_rmsf"), ("w1", "sw1"), ("w3", "sw3"),
                ("w2", "sw2")]
    shifts = [pack(wk) for wk, _ in nl_ports]
    shifts.append(int(m["rms_final_s"]) & 0xFF)   # sw_rmsfinal
    shifts.append(int(m["tok_emb_s"]) & 0xFF)     # sw_emb

    od = a.out_dir
    with open(os.path.join(od, "tb_meta.hex"), "w") as f:
        f.write(f"{a.ntok:x}\n")
    with open(os.path.join(od, "tb_shifts.hex"), "w") as f:
        for v in shifts:
            f.write(f"{v:x}\n")
    with open(os.path.join(od, "tb_expected.hex"), "w") as f:
        for t in tok:
            f.write(f"{t:x}\n")

    print(f"[gen_tb_hex] NL={NL} NTOK={a.ntok} -> {od}")
    print(f"  shifts: {[hex(s) for s in shifts[:4]]} ...")
    print(f"  expect: {tok}")
    print(f"  fichiers: tb_meta.hex tb_shifts.hex tb_expected.hex")

if __name__ == "__main__":
    main()