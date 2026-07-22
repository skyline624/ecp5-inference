// timeline.js — modèle de timing partagé du pipeline (demo synthétique).
//
// Factored depuis AnimateView pour que la vue Cluster anime au même rythme que
// le pipeline Animate (synchronisation cross-tab via window.ptState).
// Représente la pleine traversée d'un token : boot (1×) → prompt → embed →
// 5 couches (rms→QKV→rope→kvwrite→attn→WO→resid→rms→FFN→resid) → rms_final →
// lm_head → argmax → token, bouclé ×17.

export const NLAYERS = 5;
export const NTOKENS = 17;

// (étape, ticks d'animation) pour 1 couche ; répétée ×5.
export const LAYER = [
  ['rms_att', 6], ['qkv', 12], ['rope', 4], ['kvwrite', 3], ['attn', 10],
  ['wo', 10], ['resid1', 2], ['rms_ffn', 6], ['ffn_up', 12], ['silu', 4],
  ['ffn_down', 10], ['resid2', 2],
];

export const ONE_TOKEN = [
  ['prompt', 3], ['embed', 8],
  ...Array(NLAYERS).fill(LAYER).flat(),
  ['rms_final', 6], ['lm_head', 16], ['argmax', 8], ['token', 3],
];

// Déroule la démo : stepAt[i] + meta (tokAt, layAt) par cycle. Boot (1×) en tête.
export function buildDemo() {
  const stepAt = [];
  const tokAt = [];
  const BOOT_TICKS = 30;
  for (let k = 0; k < BOOT_TICKS; k++) { stepAt.push('boot'); tokAt.push(-1); }
  for (let t = 0; t < NTOKENS; t++) {
    for (const [step, ticks] of ONE_TOKEN) {
      for (let k = 0; k < ticks; k++) { stepAt.push(step); tokAt.push(t); }
    }
  }
  // couche = -1 hors-bloc, sinon 0..4 ; on compte les passages rms_att.
  const layAt = new Array(stepAt.length);
  let layer = -1;
  for (let i = 0; i < stepAt.length; i++) {
    const s = stepAt[i];
    if (s === 'boot' || s === 'prompt' || s === 'embed') layer = -1;
    else if (s === 'rms_att') layer = (layer + 1) % NLAYERS;
    layAt[i] = layer;
  }
  return { stepAt, tokAt, layAt };
}