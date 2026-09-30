# Addendum (27 September 2026): added after the main simulated results

A review of the draft pointed out that removing MSWEA also removes almost half of the parameters on the 8-channel
simulated series (3.26M to 1.73M). A parameter-matched MSWEA-off variant was therefore added: d_model raised to 354
(3.29M parameters, see param_match.txt), with 30 runs by run_synth_pm.sh. This control was not part of design.md,
and it is reported separately, with its own Holm correction over the two conditions.
