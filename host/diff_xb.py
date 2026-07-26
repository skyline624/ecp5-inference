import numpy as np, sys, json
HERE = r'C:\Users\User\Downloads\ecp5-inference\host'
sys.path.insert(0, HERE)
import diag_oracle_dim128 as D

def parse_xb(line):
    i = line.index('XB=') + 3
    words = line[i:].split()
    b = []
    for w in words:
        v = int(w, 16)
        b += [v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff]
    a = np.array(b, dtype=np.int16)
    a = np.where(a > 127, a - 256, a).astype(np.int8)
    return a

# parse RTL : "tok[N]=V OK ... XB=..." ou "MISMATCH tok[N]=V expect=E ... XB=..."
rtl = {}
for line in open(HERE + r'\rtl_xb.txt'):
    line = line.strip()
    if line.startswith('tok['):
        t = int(line[4:line.index(']')])
        rtl[t] = parse_xb(line)
    elif line.startswith('MISMATCH tok['):
        t = int(line[line.index('[')+1:line.index(']')])
        rtl[t] = parse_xb(line)

with open(HERE + r'\gen_model_dump_dim128.json') as f:
    m = json.load(f)
def to_i8(v): return np.array(v, dtype=np.int8)
m['tok_emb'] = to_i8(m['tok_emb']); m['rms_final'] = to_i8(m['rms_final'])
for l in range(len(m['layers'])):
    for k in ('rms_att','rms_ffn','wq','wk','wv','wo','w1','w2','w3'):
        m['layers'][l][k] = to_i8(m['layers'][l][k])

out, xbs = D.forward_diag(m, 17)
print(f"oracle tokens = {out}")
for t in range(min(len(xbs), len(rtl))):
    r = rtl[t]; o = xbs[t][0].astype(np.int8)
    diff = r.astype(int) - o.astype(int)
    nz = np.nonzero(diff)[0]
    mx = int(np.abs(diff).max()) if len(nz) else 0
    print(f"tok[{t:>2}] rtl_sxb? ora_sxb={xbs[t][1]:>3} : {len(nz):>3}/128 differ, max|d|={mx}")
    if len(nz) and t <= 2:
        print(f"   first diff idx={nz[0]}: rtl={int(r[nz[0]])} ora={int(o[nz[0]])} d={int(diff[nz[0]])}")
        print(f"   diff[0:16] = {list(diff[:16])}")