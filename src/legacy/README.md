# `src/legacy/` — code historique, non compilé

Ce répertoire contient du RTL conservé pour la **traçabilité de la provenance**
uniquement. Aucun fichier ici n'est listé dans une liste de sources de build ou de
simulation — les ajouter casserait la compilation (collision de noms de module).

## `top_gowin_monolith.v` (anciennement `src/top.v`)

Portage ECP5 du monolithe Gowin d'origine (`dsp_inference`, branche
`ecp5-g4-nomem2reg-silicon`) : datapath du nœud **plus** le FSM de génération `GG`
monolithique (~340 états) qui exécutait toute la couche transformer.

**Pourquoi il est écarté du build :**

1. **Il ne route pas.** C'est la raison d'être du projet : le monolithe `GG` est
   congestion-bound (372–789 nets non routés sur GW2AR-18), d'où la bascule vers la
   lignée `gen_seq` séquencée (cf. `docs/REFACTOR_UNIFIED_NODE.md`).
2. **Collision de nom.** Il déclare `module top`, exactement comme `src/node_top.v`.
   Les deux tops instanciés (`src/gen_seq_top.v:161` et
   `board/gen_seq_silicon.v:485`) résolvent `top` vers **`node_top.v`**, qui est le
   seul listé dans `build/build_silicon.ps1` et `sim/Makefile`.
3. **Ports incompatibles.** Il n'a pas les ports `lk_rx`/`lk_tx` + `kv_*` que
   `gen_seq_top.v` connecte — la sim est cassée avec lui depuis l'étape KV SDRAM
   (cf. commentaire `sim/Makefile:24-26`).

**Ce à quoi il sert encore :** référence du portage des primitives vendor
(`Gowin_rPLL`→`EHXPLLL`, `MULTALU18X18`→`MULT18X18D` inféré,
`syn_ramstyle`→`ram_style`, `.cst`→`.lpf`), documenté dans le tableau de portage du
`README.md`. Les numéros de ligne cités là-bas (`top.v:240,250,258,267,274,282` pour
`ram_style="block"`, `top.v:866` pour `ram_style="registers"`) référencent ce
fichier.
