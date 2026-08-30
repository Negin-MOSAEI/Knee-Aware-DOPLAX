# Batch Level Inference KPIs

| Dataset   | Batch                     | Method          |      RMSE |       MAE |      MAPE |
|:----------|:--------------------------|:----------------|----------:|----------:|----------:|
| HUST      | default                   | DEEPOPINN (U_1) | 0.856038  | 0.852187  |  78.6388  |
| HUST      | default                   | KADOPLAX        | 0.0839336 | 0.0724842 |   6.85985 |
| HUST      | default                   | LAX (U_2)       | 0.0839336 | 0.0724842 |   6.85985 |
| HUST      | default                   | POST_KADOPLAX   | 0.0839409 | 0.0724909 |   6.86041 |
| MIT       | 2017-05-12                | DEEPOPINN (U_1) | 0.0397371 | 0.035787  |   3.45043 |
| MIT       | 2017-05-12                | KADOPLAX        | 0.0520972 | 0.049355  |   4.70822 |
| MIT       | 2017-05-12                | LAX (U_2)       | 0.194696  | 0.191846  |  18.2868  |
| MIT       | 2017-05-12                | POST_KADOPLAX   | 0.0516973 | 0.0488905 |   4.6652  |
| MIT       | 2017-06-30                | DEEPOPINN (U_1) | 0.0532823 | 0.0483714 |   4.648   |
| MIT       | 2017-06-30                | KADOPLAX        | 0.065371  | 0.0603493 |   5.75508 |
| MIT       | 2017-06-30                | LAX (U_2)       | 0.411028  | 0.409975  |  39.7592  |
| MIT       | 2017-06-30                | POST_KADOPLAX   | 0.0648277 | 0.0596074 |   5.68833 |
| MIT       | 2018-04-12                | DEEPOPINN (U_1) | 0.064011  | 0.0597817 |   5.72228 |
| MIT       | 2018-04-12                | KADOPLAX        | 0.0338849 | 0.0228872 |   2.2808  |
| MIT       | 2018-04-12                | LAX (U_2)       | 0.0353839 | 0.0237456 |   2.36903 |
| MIT       | 2018-04-12                | POST_KADOPLAX   | 0.0342242 | 0.0231064 |   2.30313 |
| TJU       | Dataset_1_NCA_battery     | DEEPOPINN (U_1) | 0.294918  | 0.269709  |   8.80866 |
| TJU       | Dataset_1_NCA_battery     | KADOPLAX        | 0.298926  | 0.273437  |   8.92953 |
| TJU       | Dataset_1_NCA_battery     | LAX (U_2)       | 4.61148   | 4.60874   | 156.716   |
| TJU       | Dataset_1_NCA_battery     | POST_KADOPLAX   | 0.307391  | 0.279916  |   9.14649 |
| TJU       | Dataset_2_NCM_battery     | DEEPOPINN (U_1) | 0.178153  | 0.154234  |   5.32108 |
| TJU       | Dataset_2_NCM_battery     | KADOPLAX        | 0.16457   | 0.145288  |   4.88012 |
| TJU       | Dataset_2_NCM_battery     | LAX (U_2)       | 0.838556  | 0.833011  |  28.1248  |
| TJU       | Dataset_2_NCM_battery     | POST_KADOPLAX   | 0.146642  | 0.126454  |   4.26276 |
| TJU       | Dataset_3_NCM_NCA_battery | DEEPOPINN (U_1) | 0.554403  | 0.526683  |  25.661   |
| TJU       | Dataset_3_NCM_NCA_battery | KADOPLAX        | 0.215702  | 0.185887  |   9.22802 |
| TJU       | Dataset_3_NCM_NCA_battery | LAX (U_2)       | 1.00241   | 0.976576  |  50.6204  |
| TJU       | Dataset_3_NCM_NCA_battery | POST_KADOPLAX   | 0.214812  | 0.18593   |   9.27379 |
| XJTU      | 2C                        | DEEPOPINN (U_1) | 1.58983   | 1.58659   |  84.7522  |
| XJTU      | 2C                        | KADOPLAX        | 0.67095   | 0.664415  |  35.3129  |
| XJTU      | 2C                        | LAX (U_2)       | 0.416235  | 0.404314  |  21.3553  |
| XJTU      | 2C                        | POST_KADOPLAX   | 0.670721  | 0.663829  |  35.2731  |
| XJTU      | 3C                        | DEEPOPINN (U_1) | 1.15816   | 1.13643   |  60.4399  |
| XJTU      | 3C                        | KADOPLAX        | 0.324818  | 0.316481  |  16.6753  |
| XJTU      | 3C                        | LAX (U_2)       | 0.119323  | 0.110351  |   5.79324 |
| XJTU      | 3C                        | POST_KADOPLAX   | 0.330336  | 0.321857  |  16.9613  |
| XJTU      | R2.5                      | DEEPOPINN (U_1) | 1.50791   | 1.50443   |  79.9256  |
| XJTU      | R2.5                      | KADOPLAX        | 0.889532  | 0.884154  |  46.8603  |
| XJTU      | R2.5                      | LAX (U_2)       | 0.805161  | 0.799098  |  42.3166  |
| XJTU      | R2.5                      | POST_KADOPLAX   | 0.874453  | 0.868905  |  46.0442  |
| XJTU      | R3                        | DEEPOPINN (U_1) | 0.694199  | 0.688769  |  36.6056  |
| XJTU      | R3                        | KADOPLAX        | 0.125745  | 0.114431  |   6.02194 |
| XJTU      | R3                        | LAX (U_2)       | 0.0946848 | 0.0803894 |   4.28836 |
| XJTU      | R3                        | POST_KADOPLAX   | 0.124829  | 0.113756  |   5.98849 |
| XJTU      | RW                        | DEEPOPINN (U_1) | 2.8371    | 2.83594   | 152.801   |
| XJTU      | RW                        | KADOPLAX        | 2.51536   | 2.51407   | 135.468   |
| XJTU      | RW                        | LAX (U_2)       | 2.34044   | 2.33923   | 126.083   |
| XJTU      | RW                        | POST_KADOPLAX   | 2.31379   | 2.31238   | 124.57    |
| XJTU      | Sim_satellite             | DEEPOPINN (U_1) | 0.631433  | 0.625593  |  32.9005  |
| XJTU      | Sim_satellite             | KADOPLAX        | 0.538121  | 0.532027  |  27.9223  |
| XJTU      | Sim_satellite             | LAX (U_2)       | 0.526738  | 0.51956   |  27.245   |
| XJTU      | Sim_satellite             | POST_KADOPLAX   | 0.538743  | 0.532829  |  27.9712  |