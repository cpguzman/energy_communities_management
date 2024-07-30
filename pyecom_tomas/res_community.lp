\* Source Pyomo model name=unknown *\

min 
objFn:
+0.4 storDischarge(1_24)
+0.08 storDischarge(2_24)
+0.04 storDischarge(3_24)
+1 storRelax(1_24)
+1 storRelax(2_24)
+1 storRelax(3_24)
+0.08 genActPower(1_24)
+0.08 genActPower(2_24)
+0.08 genActPower(3_24)
+0.08 genActPower(4_24)
+0.08 genActPower(5_24)
+4.5 genExcPower(1_24)
+4.5 genExcPower(2_24)
+4.5 genExcPower(3_24)
+4.5 genExcPower(4_24)
+4.5 genExcPower(5_24)
+0.5 loadRedActPower(1_24)
+0.5 loadRedActPower(2_24)
+0.5 loadRedActPower(3_24)
+0.5 loadRedActPower(4_24)
+0.5 loadRedActPower(5_24)
+0.6 loadCutActPower(1_24)
+0.6 loadCutActPower(2_24)
+0.6 loadCutActPower(3_24)
+0.6 loadCutActPower(4_24)
+0.6 loadCutActPower(5_24)
+4.5 loadENS(1_24)
+4.5 loadENS(2_24)
+4.5 loadENS(3_24)
+4.5 loadENS(4_24)
+4.5 loadENS(5_24)
+0.05 v2gDischarge(1_24)
+0.06 v2gDischarge(2_24)
+0.05 v2gDischarge(3_24)
+0.05 v2gDischarge(4_24)
+0.06 v2gDischarge(5_24)
+0.05 v2gDischarge(6_24)
+0.06 v2gDischarge(7_24)
+0.05 v2gDischarge(8_24)
+0.05 v2gDischarge(9_24)
+0.06 v2gDischarge(10_24)
+200 v2gRelax(1_24)
+200 v2gRelax(2_24)
+200 v2gRelax(3_24)
+200 v2gRelax(4_24)
+200 v2gRelax(5_24)
+200 v2gRelax(6_24)
+200 v2gRelax(7_24)
+200 v2gRelax(8_24)
+200 v2gRelax(9_24)
+200 v2gRelax(10_24)
+0.0643 imports(24)
-0.04 exports(24)
+ [
+0.004 storScheduleChRelax(1_24) ^ 2
+0.004 storScheduleChRelax(2_24) ^ 2
+0.004 storScheduleChRelax(3_24) ^ 2
+0.004 storScheduleDchRelax(1_24) ^ 2
+0.004 storScheduleDchRelax(2_24) ^ 2
+0.004 storScheduleDchRelax(3_24) ^ 2
+0.02 v2gScheduleChRelax(1_24) ^ 2
+0.02 v2gScheduleChRelax(2_24) ^ 2
+0.02 v2gScheduleChRelax(3_24) ^ 2
+0.02 v2gScheduleChRelax(4_24) ^ 2
+0.02 v2gScheduleChRelax(5_24) ^ 2
+0.02 v2gScheduleChRelax(6_24) ^ 2
+0.02 v2gScheduleChRelax(7_24) ^ 2
+0.02 v2gScheduleChRelax(8_24) ^ 2
+0.02 v2gScheduleChRelax(9_24) ^ 2
+0.02 v2gScheduleChRelax(10_24) ^ 2
+0.02 v2gScheduleDchRelax(1_24) ^ 2
+0.02 v2gScheduleDchRelax(2_24) ^ 2
+0.02 v2gScheduleDchRelax(3_24) ^ 2
+0.02 v2gScheduleDchRelax(4_24) ^ 2
+0.02 v2gScheduleDchRelax(5_24) ^ 2
+0.02 v2gScheduleDchRelax(6_24) ^ 2
+0.02 v2gScheduleDchRelax(7_24) ^ 2
+0.02 v2gScheduleDchRelax(8_24) ^ 2
+0.02 v2gScheduleDchRelax(9_24) ^ 2
+0.02 v2gScheduleDchRelax(10_24) ^ 2
] / 2

s.t.

c_u_impMaxEq(24)_:
+1 imports(24)
<= 20000.0

c_u_expMaxEq(24)_:
+1 exports(24)
<= 50.0

c_e_genActMaxEq(1_24)_:
+1 genActPower(1_24)
+1 genExcPower(1_24)
= 1.0117854799318593e-05

c_e_genActMaxEq(2_24)_:
+1 genActPower(2_24)
+1 genExcPower(2_24)
= 0.81666100025177

c_e_genActMaxEq(3_24)_:
+1 genActPower(3_24)
+1 genExcPower(3_24)
= 0.0

c_e_genActMaxEq(4_24)_:
+1 genActPower(4_24)
+1 genExcPower(4_24)
= 0.0016335057616743143

c_e_genActMaxEq(5_24)_:
+1 genActPower(5_24)
+1 genExcPower(5_24)
= 0.0

c_u_loadReactEq(1_24)_:
+1 loadRedActPower(1_24)
<= 224.00833129882812

c_u_loadReactEq(2_24)_:
+1 loadRedActPower(2_24)
<= 57.4666633605957

c_u_loadReactEq(3_24)_:
+1 loadRedActPower(3_24)
<= 1389.0400085449219

c_u_loadReactEq(4_24)_:
+1 loadRedActPower(4_24)
<= 104.35666179656982

c_u_loadReactEq(5_24)_:
+1 loadRedActPower(5_24)
<= 580.8100128173828

c_e_loadCutActEq(1_24)_:
+1 loadCutActPower(1_24)
-224.00833129882812 loadXo(1_24)
= 0

c_e_loadCutActEq(2_24)_:
+1 loadCutActPower(2_24)
-57.4666633605957 loadXo(2_24)
= 0

c_e_loadCutActEq(3_24)_:
+1 loadCutActPower(3_24)
-1389.0400085449219 loadXo(3_24)
= 0

c_e_loadCutActEq(4_24)_:
+1 loadCutActPower(4_24)
-104.35666179656982 loadXo(4_24)
= 0

c_e_loadCutActEq(5_24)_:
+1 loadCutActPower(5_24)
-580.8100128173828 loadXo(5_24)
= 0

c_u_loadENSEq(1_24)_:
+1 loadRedActPower(1_24)
+1 loadCutActPower(1_24)
+1 loadENS(1_24)
<= 2240.0833129882812

c_u_loadENSEq(2_24)_:
+1 loadRedActPower(2_24)
+1 loadCutActPower(2_24)
+1 loadENS(2_24)
<= 574.666633605957

c_u_loadENSEq(3_24)_:
+1 loadRedActPower(3_24)
+1 loadCutActPower(3_24)
+1 loadENS(3_24)
<= 13890.400085449219

c_u_loadENSEq(4_24)_:
+1 loadRedActPower(4_24)
+1 loadCutActPower(4_24)
+1 loadENS(4_24)
<= 1043.5666179656982

c_u_loadENSEq(5_24)_:
+1 loadRedActPower(5_24)
+1 loadCutActPower(5_24)
+1 loadENS(5_24)
<= 5808.100128173828

c_u_storDchRateEq(1_24)_:
+1 storDischarge(1_24)
-6000.0 storDchXo(1_24)
<= 0

c_u_storDchRateEq(2_24)_:
+1 storDischarge(2_24)
-12000.0 storDchXo(2_24)
<= 0

c_u_storDchRateEq(3_24)_:
+1 storDischarge(3_24)
-2400.0 storDchXo(3_24)
<= 0

c_u_storChRateEq(1_24)_:
+1 storCharge(1_24)
-6000.0 storChXo(1_24)
<= 0

c_u_storChRateEq(2_24)_:
+1 storCharge(2_24)
-12000.0 storChXo(2_24)
<= 0

c_u_storChRateEq(3_24)_:
+1 storCharge(3_24)
-2400.0 storChXo(3_24)
<= 0

c_u_storMaxEq(1_24)_:
+1 storState(1_24)
<= 30000.0

c_u_storMaxEq(2_24)_:
+1 storState(2_24)
<= 60000.0

c_u_storMaxEq(3_24)_:
+1 storState(3_24)
<= 9000.0

c_u_storRelaxEq(1_24)_:
-1 storRelax(1_24)
-1 storState(1_24)
<= -6000.0

c_u_storRelaxEq(2_24)_:
-1 storRelax(2_24)
-1 storState(2_24)
<= -12000.0

c_u_storRelaxEq(3_24)_:
-1 storRelax(3_24)
-1 storState(3_24)
<= -1800.0

c_e_storBalanceEq(1_24)_:
+1.0526315789473684 storDischarge(1_24)
-0.9499999999999998 storCharge(1_24)
+1 storState(1_24)
= 8515.460434773971

c_e_storBalanceEq(2_24)_:
+1.0416666666666667 storDischarge(2_24)
-0.96 storCharge(2_24)
+1 storState(2_24)
= 33114.09149241935

c_e_storBalanceEq(3_24)_:
+1.0526315789473684 storDischarge(3_24)
-0.9499999999999998 storCharge(3_24)
+1 storState(3_24)
= 4326.315789473684

c_u_storChDchEq(1_24)_:
+1 storDchXo(1_24)
+1 storChXo(1_24)
<= 1

c_u_storChDchEq(2_24)_:
+1 storDchXo(2_24)
+1 storChXo(2_24)
<= 1

c_u_storChDchEq(3_24)_:
+1 storDchXo(3_24)
+1 storChXo(3_24)
<= 1

c_u_v2gDchRateEq(1_24)_:
+1 v2gDischarge(1_24)
<= 0

c_u_v2gDchRateEq(2_24)_:
+1 v2gDischarge(2_24)
-120.0 v2gDchXo(2_24)
<= 0

c_u_v2gDchRateEq(3_24)_:
+1 v2gDischarge(3_24)
<= 0

c_u_v2gDchRateEq(4_24)_:
+1 v2gDischarge(4_24)
<= 0

c_u_v2gDchRateEq(5_24)_:
+1 v2gDischarge(5_24)
<= 0

c_u_v2gDchRateEq(6_24)_:
+1 v2gDischarge(6_24)
<= 0

c_u_v2gDchRateEq(7_24)_:
+1 v2gDischarge(7_24)
-120.0 v2gDchXo(7_24)
<= 0

c_u_v2gDchRateEq(8_24)_:
+1 v2gDischarge(8_24)
<= 0

c_u_v2gDchRateEq(9_24)_:
+1 v2gDischarge(9_24)
<= 0

c_u_v2gDchRateEq(10_24)_:
+1 v2gDischarge(10_24)
<= 0

c_u_v2gChRateEq(1_24)_:
+1 v2gCharge(1_24)
<= 0

c_u_v2gChRateEq(2_24)_:
+1 v2gCharge(2_24)
-120.0 v2gChXo(2_24)
<= 0

c_u_v2gChRateEq(3_24)_:
+1 v2gCharge(3_24)
<= 0

c_u_v2gChRateEq(4_24)_:
+1 v2gCharge(4_24)
<= 0

c_u_v2gChRateEq(5_24)_:
+1 v2gCharge(5_24)
<= 0

c_u_v2gChRateEq(6_24)_:
+1 v2gCharge(6_24)
<= 0

c_u_v2gChRateEq(7_24)_:
+1 v2gCharge(7_24)
-120.0 v2gChXo(7_24)
<= 0

c_u_v2gChRateEq(8_24)_:
+1 v2gCharge(8_24)
<= 0

c_u_v2gChRateEq(9_24)_:
+1 v2gCharge(9_24)
<= 0

c_u_v2gChRateEq(10_24)_:
+1 v2gCharge(10_24)
<= 0

c_u_v2gMaxEq(1_24)_:
+1 v2gState(1_24)
<= 40.0

c_u_v2gMaxEq(2_24)_:
+1 v2gState(2_24)
<= 60.0

c_u_v2gMaxEq(3_24)_:
+1 v2gState(3_24)
<= 40.0

c_u_v2gMaxEq(4_24)_:
+1 v2gState(4_24)
<= 40.0

c_u_v2gMaxEq(5_24)_:
+1 v2gState(5_24)
<= 60.0

c_u_v2gMaxEq(6_24)_:
+1 v2gState(6_24)
<= 40.0

c_u_v2gMaxEq(7_24)_:
+1 v2gState(7_24)
<= 60.0

c_u_v2gMaxEq(8_24)_:
+1 v2gState(8_24)
<= 40.0

c_u_v2gMaxEq(9_24)_:
+1 v2gState(9_24)
<= 40.0

c_u_v2gMaxEq(10_24)_:
+1 v2gState(10_24)
<= 60.0

c_u_v2gRelaxEq(2_24)_:
-1 v2gRelax(2_24)
-1 v2gState(2_24)
<= -60.0

c_u_v2gRelaxEq(7_24)_:
-1 v2gRelax(7_24)
-1 v2gState(7_24)
<= -60.0

c_e_v2gStateEq(1_24)_:
+1 v2gState(1_24)
= 0

c_e_v2gStateEq(2_24)_:
+1.0101010101010102 v2gDischarge(2_24)
-0.9899999999999999 v2gCharge(2_24)
+1 v2gState(2_24)
= 0

c_e_v2gStateEq(3_24)_:
+1 v2gState(3_24)
= 0

c_e_v2gStateEq(4_24)_:
+1 v2gState(4_24)
= 0

c_e_v2gStateEq(5_24)_:
+1 v2gState(5_24)
= 0

c_e_v2gStateEq(6_24)_:
+1 v2gState(6_24)
= 0

c_e_v2gStateEq(7_24)_:
+1.0101010101010102 v2gDischarge(7_24)
-0.9899999999999999 v2gCharge(7_24)
+1 v2gState(7_24)
= 0

c_e_v2gStateEq(8_24)_:
+1 v2gState(8_24)
= 0

c_e_v2gStateEq(9_24)_:
+1 v2gState(9_24)
= 0

c_e_v2gStateEq(10_24)_:
+1 v2gState(10_24)
= 0

c_u_v2gChDchEq(1_24)_:
+1 v2gDischarge(1_24)
+1 v2gCharge(1_24)
<= 1

c_u_v2gChDchEq(2_24)_:
+1 v2gDischarge(2_24)
+1 v2gCharge(2_24)
<= 1

c_u_v2gChDchEq(3_24)_:
+1 v2gDischarge(3_24)
+1 v2gCharge(3_24)
<= 1

c_u_v2gChDchEq(4_24)_:
+1 v2gDischarge(4_24)
+1 v2gCharge(4_24)
<= 1

c_u_v2gChDchEq(5_24)_:
+1 v2gDischarge(5_24)
+1 v2gCharge(5_24)
<= 1

c_u_v2gChDchEq(6_24)_:
+1 v2gDischarge(6_24)
+1 v2gCharge(6_24)
<= 1

c_u_v2gChDchEq(7_24)_:
+1 v2gDischarge(7_24)
+1 v2gCharge(7_24)
<= 1

c_u_v2gChDchEq(8_24)_:
+1 v2gDischarge(8_24)
+1 v2gCharge(8_24)
<= 1

c_u_v2gChDchEq(9_24)_:
+1 v2gDischarge(9_24)
+1 v2gCharge(9_24)
<= 1

c_u_v2gChDchEq(10_24)_:
+1 v2gDischarge(10_24)
+1 v2gCharge(10_24)
<= 1

c_e_v2gFollowScheduleChEq(1_24)_:
-1 v2gScheduleChRelax(1_24)
+1 v2gCharge(1_24)
= 0.0

c_e_v2gFollowScheduleChEq(2_24)_:
-1 v2gScheduleChRelax(2_24)
+1 v2gCharge(2_24)
= 1.0

c_e_v2gFollowScheduleChEq(3_24)_:
-1 v2gScheduleChRelax(3_24)
+1 v2gCharge(3_24)
= 0.0

c_e_v2gFollowScheduleChEq(4_24)_:
-1 v2gScheduleChRelax(4_24)
+1 v2gCharge(4_24)
= 0.0

c_e_v2gFollowScheduleChEq(5_24)_:
-1 v2gScheduleChRelax(5_24)
+1 v2gCharge(5_24)
= 0.0

c_e_v2gFollowScheduleChEq(6_24)_:
-1 v2gScheduleChRelax(6_24)
+1 v2gCharge(6_24)
= 0.0

c_e_v2gFollowScheduleChEq(7_24)_:
-1 v2gScheduleChRelax(7_24)
+1 v2gCharge(7_24)
= 1.0

c_e_v2gFollowScheduleChEq(8_24)_:
-1 v2gScheduleChRelax(8_24)
+1 v2gCharge(8_24)
= 0.0

c_e_v2gFollowScheduleChEq(9_24)_:
-1 v2gScheduleChRelax(9_24)
+1 v2gCharge(9_24)
= 0.0

c_e_v2gFollowScheduleChEq(10_24)_:
-1 v2gScheduleChRelax(10_24)
+1 v2gCharge(10_24)
= 0.0

c_e_v2gFollowScheduleDchEq(1_24)_:
+1 v2gDischarge(1_24)
-1 v2gScheduleDchRelax(1_24)
= 0.0

c_e_v2gFollowScheduleDchEq(2_24)_:
+1 v2gDischarge(2_24)
-1 v2gScheduleDchRelax(2_24)
= 0.0

c_e_v2gFollowScheduleDchEq(3_24)_:
+1 v2gDischarge(3_24)
-1 v2gScheduleDchRelax(3_24)
= 0.0

c_e_v2gFollowScheduleDchEq(4_24)_:
+1 v2gDischarge(4_24)
-1 v2gScheduleDchRelax(4_24)
= 0.0

c_e_v2gFollowScheduleDchEq(5_24)_:
+1 v2gDischarge(5_24)
-1 v2gScheduleDchRelax(5_24)
= 0.0

c_e_v2gFollowScheduleDchEq(6_24)_:
+1 v2gDischarge(6_24)
-1 v2gScheduleDchRelax(6_24)
= 0.0

c_e_v2gFollowScheduleDchEq(7_24)_:
+1 v2gDischarge(7_24)
-1 v2gScheduleDchRelax(7_24)
= 0.0

c_e_v2gFollowScheduleDchEq(8_24)_:
+1 v2gDischarge(8_24)
-1 v2gScheduleDchRelax(8_24)
= 0.0

c_e_v2gFollowScheduleDchEq(9_24)_:
+1 v2gDischarge(9_24)
-1 v2gScheduleDchRelax(9_24)
= 0.0

c_e_v2gFollowScheduleDchEq(10_24)_:
+1 v2gDischarge(10_24)
-1 v2gScheduleDchRelax(10_24)
= 0.0

c_u_csMaxEq(1_24)_:
+1 csCharge(1_24)
<= 7.2

c_u_csMaxEq(2_24)_:
+1 csCharge(2_24)
<= 7.2

c_u_csMaxEq(3_24)_:
+1 csCharge(3_24)
<= 7.2

c_u_csMaxEq(4_24)_:
+1 csCharge(4_24)
<= 7.2

c_u_csMaxEq(5_24)_:
+1 csCharge(5_24)
<= 7.2

c_u_csMaxEq(6_24)_:
+1 csCharge(6_24)
<= 7.2

c_u_csMaxEq(7_24)_:
+1 csCharge(7_24)
<= 7.2

c_u_csMaxEq(8_24)_:
+1 csCharge(8_24)
<= 7.2

c_u_csMaxEq(9_24)_:
+1 csCharge(9_24)
<= 7.2

c_l_csMinEq(1_24)_:
+1 csCharge(1_24)
>= -7.2

c_l_csMinEq(2_24)_:
+1 csCharge(2_24)
>= -7.2

c_l_csMinEq(3_24)_:
+1 csCharge(3_24)
>= -7.2

c_l_csMinEq(4_24)_:
+1 csCharge(4_24)
>= -7.2

c_l_csMinEq(5_24)_:
+1 csCharge(5_24)
>= -7.2

c_l_csMinEq(6_24)_:
+1 csCharge(6_24)
>= -7.2

c_l_csMinEq(7_24)_:
+1 csCharge(7_24)
>= -7.2

c_l_csMinEq(8_24)_:
+1 csCharge(8_24)
>= -7.2

c_l_csMinEq(9_24)_:
+1 csCharge(9_24)
>= -7.2

c_e_csPowerEq(1_24)_:
+1 csCharge(1_24)
= 0

c_e_csPowerEq(2_24)_:
+1 v2gDischarge(2_24)
-1 v2gCharge(2_24)
+1 csCharge(2_24)
= 0

c_e_csPowerEq(3_24)_:
+1 csCharge(3_24)
= 0

c_e_csPowerEq(4_24)_:
+1 csCharge(4_24)
= 0

c_e_csPowerEq(5_24)_:
+1 csCharge(5_24)
= 0

c_e_csPowerEq(6_24)_:
+1 v2gDischarge(7_24)
-1 v2gCharge(7_24)
+1 csCharge(6_24)
= 0

c_e_csPowerEq(7_24)_:
+1 csCharge(7_24)
= 0

c_e_csPowerEq(8_24)_:
+1 csCharge(8_24)
= 0

c_e_csPowerEq(9_24)_:
+1 csCharge(9_24)
= 0

c_e_csNetChargeEq(1_24)_:
+1 csNetCharge(1_24)
= 0

c_e_csNetChargeEq(2_24)_:
+1.0101010101010102 v2gDischarge(2_24)
-0.99 v2gCharge(2_24)
+1 csNetCharge(2_24)
= 0

c_e_csNetChargeEq(3_24)_:
+1 csNetCharge(3_24)
= 0

c_e_csNetChargeEq(4_24)_:
+1 csNetCharge(4_24)
= 0

c_e_csNetChargeEq(5_24)_:
+1 csNetCharge(5_24)
= 0

c_e_csNetChargeEq(6_24)_:
+1.0101010101010102 v2gDischarge(7_24)
-0.99 v2gCharge(7_24)
+1 csNetCharge(6_24)
= 0

c_e_csNetChargeEq(7_24)_:
+1 csNetCharge(7_24)
= 0

c_e_csNetChargeEq(8_24)_:
+1 csNetCharge(8_24)
= 0

c_e_csNetChargeEq(9_24)_:
+1 csNetCharge(9_24)
= 0

c_e_balanceEq(24)_:
+1 storDischarge(1_24)
+1 storDischarge(2_24)
+1 storDischarge(3_24)
+1 genActPower(1_24)
+1 genActPower(2_24)
+1 genActPower(3_24)
+1 genActPower(4_24)
+1 genActPower(5_24)
-1 genExcPower(1_24)
-1 genExcPower(2_24)
-1 genExcPower(3_24)
-1 genExcPower(4_24)
-1 genExcPower(5_24)
+1 loadRedActPower(1_24)
+1 loadRedActPower(2_24)
+1 loadRedActPower(3_24)
+1 loadRedActPower(4_24)
+1 loadRedActPower(5_24)
+1 loadCutActPower(1_24)
+1 loadCutActPower(2_24)
+1 loadCutActPower(3_24)
+1 loadCutActPower(4_24)
+1 loadCutActPower(5_24)
+1 loadENS(1_24)
+1 loadENS(2_24)
+1 loadENS(3_24)
+1 loadENS(4_24)
+1 loadENS(5_24)
+1 v2gDischarge(1_24)
+1 v2gDischarge(2_24)
+1 v2gDischarge(3_24)
+1 v2gDischarge(4_24)
+1 v2gDischarge(5_24)
+1 v2gDischarge(6_24)
+1 v2gDischarge(7_24)
+1 v2gDischarge(8_24)
+1 v2gDischarge(9_24)
+1 v2gDischarge(10_24)
+1 imports(24)
-1 exports(24)
-1 storCharge(1_24)
-1 storCharge(2_24)
-1 storCharge(3_24)
-1 v2gCharge(1_24)
-1 v2gCharge(2_24)
-1 v2gCharge(3_24)
-1 v2gCharge(4_24)
-1 v2gCharge(5_24)
-1 v2gCharge(6_24)
-1 v2gCharge(7_24)
-1 v2gCharge(8_24)
-1 v2gCharge(9_24)
-1 v2gCharge(10_24)
-1 csNetCharge(1_24)
-1 csNetCharge(2_24)
-1 csNetCharge(3_24)
-1 csNetCharge(4_24)
-1 csNetCharge(5_24)
-1 csNetCharge(6_24)
-1 csNetCharge(7_24)
-1 csNetCharge(8_24)
-1 csNetCharge(9_24)
= 23556.816778182983

bounds
   0 <= storDischarge(1_24) <= +inf
   0 <= storDischarge(2_24) <= +inf
   0 <= storDischarge(3_24) <= +inf
   0 <= storScheduleChRelax(1_24) <= +inf
   0 <= storScheduleChRelax(2_24) <= +inf
   0 <= storScheduleChRelax(3_24) <= +inf
   0 <= storScheduleDchRelax(1_24) <= +inf
   0 <= storScheduleDchRelax(2_24) <= +inf
   0 <= storScheduleDchRelax(3_24) <= +inf
   0 <= storRelax(1_24) <= +inf
   0 <= storRelax(2_24) <= +inf
   0 <= storRelax(3_24) <= +inf
   0 <= genActPower(1_24) <= +inf
   0 <= genActPower(2_24) <= +inf
   0 <= genActPower(3_24) <= +inf
   0 <= genActPower(4_24) <= +inf
   0 <= genActPower(5_24) <= +inf
   0 <= genExcPower(1_24) <= +inf
   0 <= genExcPower(2_24) <= +inf
   0 <= genExcPower(3_24) <= +inf
   0 <= genExcPower(4_24) <= +inf
   0 <= genExcPower(5_24) <= +inf
   0 <= loadRedActPower(1_24) <= +inf
   0 <= loadRedActPower(2_24) <= +inf
   0 <= loadRedActPower(3_24) <= +inf
   0 <= loadRedActPower(4_24) <= +inf
   0 <= loadRedActPower(5_24) <= +inf
   0 <= loadCutActPower(1_24) <= +inf
   0 <= loadCutActPower(2_24) <= +inf
   0 <= loadCutActPower(3_24) <= +inf
   0 <= loadCutActPower(4_24) <= +inf
   0 <= loadCutActPower(5_24) <= +inf
   0 <= loadENS(1_24) <= +inf
   0 <= loadENS(2_24) <= +inf
   0 <= loadENS(3_24) <= +inf
   0 <= loadENS(4_24) <= +inf
   0 <= loadENS(5_24) <= +inf
   0 <= v2gDischarge(1_24) <= +inf
   0 <= v2gDischarge(2_24) <= +inf
   0 <= v2gDischarge(3_24) <= +inf
   0 <= v2gDischarge(4_24) <= +inf
   0 <= v2gDischarge(5_24) <= +inf
   0 <= v2gDischarge(6_24) <= +inf
   0 <= v2gDischarge(7_24) <= +inf
   0 <= v2gDischarge(8_24) <= +inf
   0 <= v2gDischarge(9_24) <= +inf
   0 <= v2gDischarge(10_24) <= +inf
   -inf <= v2gScheduleChRelax(1_24) <= +inf
   -inf <= v2gScheduleChRelax(2_24) <= +inf
   -inf <= v2gScheduleChRelax(3_24) <= +inf
   -inf <= v2gScheduleChRelax(4_24) <= +inf
   -inf <= v2gScheduleChRelax(5_24) <= +inf
   -inf <= v2gScheduleChRelax(6_24) <= +inf
   -inf <= v2gScheduleChRelax(7_24) <= +inf
   -inf <= v2gScheduleChRelax(8_24) <= +inf
   -inf <= v2gScheduleChRelax(9_24) <= +inf
   -inf <= v2gScheduleChRelax(10_24) <= +inf
   -inf <= v2gScheduleDchRelax(1_24) <= +inf
   -inf <= v2gScheduleDchRelax(2_24) <= +inf
   -inf <= v2gScheduleDchRelax(3_24) <= +inf
   -inf <= v2gScheduleDchRelax(4_24) <= +inf
   -inf <= v2gScheduleDchRelax(5_24) <= +inf
   -inf <= v2gScheduleDchRelax(6_24) <= +inf
   -inf <= v2gScheduleDchRelax(7_24) <= +inf
   -inf <= v2gScheduleDchRelax(8_24) <= +inf
   -inf <= v2gScheduleDchRelax(9_24) <= +inf
   -inf <= v2gScheduleDchRelax(10_24) <= +inf
   0 <= v2gRelax(1_24) <= +inf
   0 <= v2gRelax(2_24) <= +inf
   0 <= v2gRelax(3_24) <= +inf
   0 <= v2gRelax(4_24) <= +inf
   0 <= v2gRelax(5_24) <= +inf
   0 <= v2gRelax(6_24) <= +inf
   0 <= v2gRelax(7_24) <= +inf
   0 <= v2gRelax(8_24) <= +inf
   0 <= v2gRelax(9_24) <= +inf
   0 <= v2gRelax(10_24) <= +inf
   0 <= imports(24) <= +inf
   0 <= exports(24) <= +inf
   0 <= loadXo(1_24) <= 1
   0 <= loadXo(2_24) <= 1
   0 <= loadXo(3_24) <= 1
   0 <= loadXo(4_24) <= 1
   0 <= loadXo(5_24) <= 1
   0 <= storDchXo(1_24) <= 1
   0 <= storDchXo(2_24) <= 1
   0 <= storDchXo(3_24) <= 1
   0 <= storCharge(1_24) <= +inf
   0 <= storCharge(2_24) <= +inf
   0 <= storCharge(3_24) <= +inf
   0 <= storChXo(1_24) <= 1
   0 <= storChXo(2_24) <= 1
   0 <= storChXo(3_24) <= 1
   0 <= storState(1_24) <= +inf
   0 <= storState(2_24) <= +inf
   0 <= storState(3_24) <= +inf
   0 <= v2gDchXo(2_24) <= 1
   0 <= v2gDchXo(7_24) <= 1
   0 <= v2gCharge(1_24) <= +inf
   0 <= v2gCharge(2_24) <= +inf
   0 <= v2gCharge(3_24) <= +inf
   0 <= v2gCharge(4_24) <= +inf
   0 <= v2gCharge(5_24) <= +inf
   0 <= v2gCharge(6_24) <= +inf
   0 <= v2gCharge(7_24) <= +inf
   0 <= v2gCharge(8_24) <= +inf
   0 <= v2gCharge(9_24) <= +inf
   0 <= v2gCharge(10_24) <= +inf
   0 <= v2gChXo(2_24) <= 1
   0 <= v2gChXo(7_24) <= 1
   0 <= v2gState(1_24) <= +inf
   0 <= v2gState(2_24) <= +inf
   0 <= v2gState(3_24) <= +inf
   0 <= v2gState(4_24) <= +inf
   0 <= v2gState(5_24) <= +inf
   0 <= v2gState(6_24) <= +inf
   0 <= v2gState(7_24) <= +inf
   0 <= v2gState(8_24) <= +inf
   0 <= v2gState(9_24) <= +inf
   0 <= v2gState(10_24) <= +inf
   0 <= csCharge(1_24) <= +inf
   0 <= csCharge(2_24) <= +inf
   0 <= csCharge(3_24) <= +inf
   0 <= csCharge(4_24) <= +inf
   0 <= csCharge(5_24) <= +inf
   0 <= csCharge(6_24) <= +inf
   0 <= csCharge(7_24) <= +inf
   0 <= csCharge(8_24) <= +inf
   0 <= csCharge(9_24) <= +inf
   0 <= csNetCharge(1_24) <= +inf
   0 <= csNetCharge(2_24) <= +inf
   0 <= csNetCharge(3_24) <= +inf
   0 <= csNetCharge(4_24) <= +inf
   0 <= csNetCharge(5_24) <= +inf
   0 <= csNetCharge(6_24) <= +inf
   0 <= csNetCharge(7_24) <= +inf
   0 <= csNetCharge(8_24) <= +inf
   0 <= csNetCharge(9_24) <= +inf
binary
  loadXo(1_24)
  loadXo(2_24)
  loadXo(3_24)
  loadXo(4_24)
  loadXo(5_24)
  storDchXo(1_24)
  storDchXo(2_24)
  storDchXo(3_24)
  storChXo(1_24)
  storChXo(2_24)
  storChXo(3_24)
  v2gDchXo(2_24)
  v2gDchXo(7_24)
  v2gChXo(2_24)
  v2gChXo(7_24)
end
