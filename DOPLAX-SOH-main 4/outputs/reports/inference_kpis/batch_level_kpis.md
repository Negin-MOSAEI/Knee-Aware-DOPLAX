# Batch Level Inference KPIs

| Dataset   | Batch                     | Method          |       RMSE |        MAE |       MAPE |
|:----------|:--------------------------|:----------------|-----------:|-----------:|-----------:|
| HUST      | default                   | DEEPOPINN (U_1) |  1.21002   |  1.2054    |  111.769   |
| HUST      | default                   | KADOPLAX        |  0.0821205 |  0.0678842 |    6.51135 |
| HUST      | default                   | LAX (U_2)       |  0.188644  |  0.167797  |   16.2502  |
| HUST      | default                   | POST_KADOPLAX   |  0.0821616 |  0.0678876 |    6.51206 |
| MIT       | 2017-05-12                | DEEPOPINN (U_1) |  0.521679  |  0.51932   |   49.7415  |
| MIT       | 2017-05-12                | KADOPLAX        |  0.467633  |  0.466076  |   44.5554  |
| MIT       | 2017-05-12                | LAX (U_2)       | 21.624     | 21.6128    | 2074.49    |
| MIT       | 2017-05-12                | POST_KADOPLAX   |  0.469369  |  0.467819  |   44.7153  |
| MIT       | 2017-06-30                | DEEPOPINN (U_1) |  0.520715  |  0.519576  |   50.4694  |
| MIT       | 2017-06-30                | KADOPLAX        |  0.520715  |  0.519576  |   50.4694  |
| MIT       | 2017-06-30                | LAX (U_2)       |  9.02149   |  9.02127   |  878.202   |
| MIT       | 2017-06-30                | POST_KADOPLAX   |  0.528872  |  0.527815  |   51.2314  |
| MIT       | 2018-04-12                | DEEPOPINN (U_1) |  0.0819245 |  0.0780015 |    7.45235 |
| MIT       | 2018-04-12                | KADOPLAX        |  0.0314982 |  0.0205782 |    2.05412 |
| MIT       | 2018-04-12                | LAX (U_2)       |  0.0356809 |  0.0236667 |    2.36448 |
| MIT       | 2018-04-12                | POST_KADOPLAX   |  0.0319982 |  0.0206299 |    2.06188 |
| TJU       | Dataset_1_NCA_battery     | DEEPOPINN (U_1) |  0.307774  |  0.281326  |    9.23253 |
| TJU       | Dataset_1_NCA_battery     | KADOPLAX        |  0.201618  |  0.173074  |    5.84327 |
| TJU       | Dataset_1_NCA_battery     | LAX (U_2)       |  0.198178  |  0.167875  |    5.75219 |
| TJU       | Dataset_1_NCA_battery     | POST_KADOPLAX   |  0.201357  |  0.172687  |    5.83699 |
| TJU       | Dataset_2_NCM_battery     | DEEPOPINN (U_1) |  0.162049  |  0.138348  |    4.87957 |
| TJU       | Dataset_2_NCM_battery     | KADOPLAX        |  0.162098  |  0.138397  |    4.88086 |
| TJU       | Dataset_2_NCM_battery     | LAX (U_2)       |  4.32704   |  4.24826   |  144.625   |
| TJU       | Dataset_2_NCM_battery     | POST_KADOPLAX   |  0.154028  |  0.131393  |    4.6411  |
| TJU       | Dataset_3_NCM_NCA_battery | DEEPOPINN (U_1) |  0.404766  |  0.34923   |   16.5071  |
| TJU       | Dataset_3_NCM_NCA_battery | KADOPLAX        |  0.225963  |  0.195085  |    9.70148 |
| TJU       | Dataset_3_NCM_NCA_battery | LAX (U_2)       |  1.80503   |  1.79061   |   91.7605  |
| TJU       | Dataset_3_NCM_NCA_battery | POST_KADOPLAX   |  0.224543  |  0.19437   |    9.68887 |
| XJTU      | 2C                        | DEEPOPINN (U_1) |  1.49575   |  1.49021   |   79.7228  |
| XJTU      | 2C                        | KADOPLAX        |  0.0959605 |  0.0729998 |    4.08291 |
| XJTU      | 2C                        | LAX (U_2)       |  1.02737   |  1.02372   |   55.1248  |
| XJTU      | 2C                        | POST_KADOPLAX   |  0.0969024 |  0.0734191 |    4.10958 |
| XJTU      | 3C                        | DEEPOPINN (U_1) |  2.02545   |  2.02381   |  107.855   |
| XJTU      | 3C                        | KADOPLAX        |  0.0825828 |  0.0660613 |    3.58264 |
| XJTU      | 3C                        | LAX (U_2)       |  0.0825611 |  0.0652356 |    3.54512 |
| XJTU      | 3C                        | POST_KADOPLAX   |  0.0825309 |  0.0660034 |    3.5797  |
| XJTU      | R2.5                      | DEEPOPINN (U_1) |  1.99896   |  1.99734   |  106.336   |
| XJTU      | R2.5                      | KADOPLAX        |  0.114224  |  0.0986787 |    5.2489  |
| XJTU      | R2.5                      | LAX (U_2)       |  0.114103  |  0.0985406 |    5.24219 |
| XJTU      | R2.5                      | POST_KADOPLAX   |  0.11424   |  0.0986919 |    5.24957 |
| XJTU      | R3                        | DEEPOPINN (U_1) |  1.13562   |  1.13248   |   60.131   |
| XJTU      | R3                        | KADOPLAX        |  0.334897  |  0.325296  |   17.1526  |
| XJTU      | R3                        | LAX (U_2)       |  0.343292  |  0.333611  |   17.593   |
| XJTU      | R3                        | POST_KADOPLAX   |  0.338861  |  0.329831  |   17.3999  |
| XJTU      | RW                        | DEEPOPINN (U_1) |  1.28471   |  1.26299   |   68.3878  |
| XJTU      | RW                        | KADOPLAX        |  0.369356  |  0.361369  |   19.3168  |
| XJTU      | RW                        | LAX (U_2)       |  0.368557  |  0.36037   |   19.2608  |
| XJTU      | RW                        | POST_KADOPLAX   |  0.371275  |  0.363161  |   19.4116  |
| XJTU      | Sim_satellite             | DEEPOPINN (U_1) |  1.29304   |  1.28985   |   68.0989  |
| XJTU      | Sim_satellite             | KADOPLAX        |  0.125628  |  0.114829  |    6.02136 |
| XJTU      | Sim_satellite             | LAX (U_2)       |  0.125615  |  0.114816  |    6.02071 |
| XJTU      | Sim_satellite             | POST_KADOPLAX   |  0.12563   |  0.114831  |    6.02146 |