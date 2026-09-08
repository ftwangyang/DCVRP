# Frozen-state vehicle-module permutation test

The identity rollout is stopped at intervals 3, 5, and 7. The complete physical environment state is cloned, and only the mapping between the four independent selector subnetworks and the four fixed vehicle-state slots is changed. All 24 mappings are evaluated from each identical snapshot.

| Dynamic rate | Instances | Frozen intervals | Mappings | Vehicle-choice change (%) | Mean score RMS $\Delta$ | Mean $\Delta$Cost | Mean $|\Delta$Cost| | Max $|\Delta$Cost| | Mean $\Delta$QoS (pp) | Max $|\Delta$QoS|$ (pp) |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 100 | 3,5,7 | 24 | 74.18 | 9.5781 | +0.1009 | 0.1443 | 2.2613 | +0.0000 | 0.0000 |
| 25% | 100 | 3,5,7 | 24 | 69.96 | 9.6905 | +0.1409 | 0.2813 | 2.6094 | +0.0000 | 0.0000 |
| 50% | 100 | 3,5,7 | 24 | 66.62 | 9.7614 | +0.1175 | 0.3891 | 3.4706 | -0.0087 | 5.0000 |
| 75% | 100 | 3,5,7 | 24 | 67.97 | 9.7531 | -0.0091 | 0.4494 | 2.9071 | -0.0145 | 5.0000 |
