from ocelot import * 

# Drifts
dr_lh_8 = Drift(l=0.4975, eid='DR_LH.8')
mscr_lh_01 = Drift(l=0.08, eid='MSCR_LH.01')
dr_lh_9 = Drift(l=0.081, eid='DR_LH.9')
dr_lh_10 = Drift(l=0.176, eid='DR_LH.10')
bpm_dr = Drift(l=0.075, eid='BPM_DR')
dr_lh_11 = Drift(l=0.4072, eid='DR_LH.11')
dr_lh_2 = Drift(l=0.05, eid='DR_LH.2')
dr_lh_12 = Drift(l=0.531402, eid='DR_LH.12')
dr_lh_13 = Drift(l=0.421398, eid='DR_LH.13')
dr_lh_24 = Drift(l=0.3, eid='DR_LH.24')
dr_lh_25 = Drift(l=0.1266, eid='DR_LH.25')
dr_lh_26 = Drift(l=0.4819, eid='DR_LH.26')
und_lh_01 = Drift(l=0.48432, eid='UND_LH.01')
dr_lh_28 = Drift(l=0.246, eid='DR_LH.28')
dr_lh_29 = Drift(l=0.777, eid='DR_LH.29')
dr_lh_14 = Drift(l=0.271, eid='DR_LH.14')
dr_lh_15 = Drift(l=0.281, eid='DR_LH.15')
dr_lh_15a = Drift(l=0.0075, eid='DR_LH.15A')
dr_lh_16 = Drift(l=1.2882, eid='DR_LH.16')
dr_lh_17 = Drift(l=0.1195, eid='DR_LH.17')
dr_lh_18 = Drift(l=0.18716, eid='DR_LH.18')
dr_lh_19 = Drift(l=0.1676, eid='DR_LH.19')
dr_lh_20 = Drift(l=0.3175, eid='DR_LH.20')
dr_lh_21 = Drift(l=0.0485, eid='DR_LH.21')
dr_lh_23 = Drift(l=0.1675, eid='DR_LH.23')
dr_acc_c_1_1 = Drift(l=0.124001, eid='DR_ACC_C_1_1')
dr_acc_c_1_2 = Drift(l=0.102005, eid='DR_ACC_C_1_2')
dr_l1_sis1 = Drift(l=0.1296, eid='DR_L1.SIS1')
dr_l1_sis2 = Drift(l=0.0545, eid='DR_L1.SIS2')
dr_l1_sis3 = Drift(l=0.1348, eid='DR_L1.SIS3')
dr_l1_sis4 = Drift(l=0.1441, eid='DR_L1.SIS4')
dr_l1_sis5 = Drift(l=0.112, eid='DR_L1.SIS5')
dr_acc_c_2_1 = Drift(l=0.124002, eid='DR_ACC_C_2_1')
dr_acc_c_2_2 = Drift(l=0.102006, eid='DR_ACC_C_2_2')
dr_l1_8a = Drift(l=0.12959, eid='DR_L1.8A')
dr_l1_8b = Drift(l=0.0645, eid='DR_L1.8B')
dr_l1_8c = Drift(l=0.24764, eid='DR_L1.8C')
dr_l1_9a = Drift(l=0.18277, eid='DR_L1.9A')
dr_l1_9b = Drift(l=0.1075, eid='DR_L1.9B')
dr_l1_10 = Drift(l=0.048, eid='DR_L1.10')
dr_acc_c_3_1 = Drift(l=0.124003, eid='DR_ACC_C_3_1')
dr_acc_c_3_2 = Drift(l=0.102007, eid='DR_ACC_C_3_2')
dr_l1_sis7 = Drift(l=0.0759, eid='DR_L1.SIS7')
dr_l1_sis8 = Drift(l=0.1213, eid='DR_L1.SIS8')
dr_l1_sis9 = Drift(l=0.1269, eid='DR_L1.SIS9')
dr_l1_sis10 = Drift(l=0.0414, eid='DR_L1.SIS10')
dr_acc_c_4_1 = Drift(l=0.124004, eid='DR_ACC_C_4_1')
dr_acc_c_4_2 = Drift(l=0.102008, eid='DR_ACC_C_4_2')
dr_l1_sis12 = Drift(l=0.0599, eid='DR_L1.SIS12')
dr_l1_sis13 = Drift(l=0.1918, eid='DR_L1.SIS13')
dr_l1_sis14 = Drift(l=0.0883, eid='DR_L1.SIS14')
dr_l1_sis15 = Drift(l=0.0507, eid='DR_L1.SIS15')
dr_l1_14 = Drift(l=0.4236, eid='DR_L1.14')
dr_bc1_1a = Drift(l=0.4063, eid='DR_BC1.1A')
dr_bc1_1b = Drift(l=2.1851, eid='DR_BC1.1B')
dr_bc1_2a = Drift(l=0.24474, eid='DR_BC1.2A')
scrph_bc01_01 = Drift(l=0.093, eid='SCRPH_BC01.01')
dr_bc1_2b = Drift(l=0.0869, eid='DR_BC1.2B')
dr_bc1_2c = Drift(l=0.01654, eid='DR_BC1.2C')
dr_bc1_2d = Drift(l=0.17796, eid='DR_BC1.2D')
dr_bc1_2e = Drift(l=0.40916, eid='DR_BC1.2E')
dr_bc1_3a = Drift(l=2.1852, eid='DR_BC1.3A')
dr_bc1_3b = Drift(l=0.4062, eid='DR_BC1.3B')
dr_bc1_4 = Drift(l=0.017, eid='DR_BC1.4')
cblm_bc01_01 = Drift(l=0.25, eid='CBLM_BC01.01')
gblm_bc01_01 = Drift(l=0.1, eid='GBLM_BC01.01')
dr_bc1_4b = Drift(l=0.9724, eid='DR_BC1.4B')
dr_bc1_4d = Drift(l=0.1874, eid='DR_BC1.4D')
dr_bc1_5a = Drift(l=0.5205, eid='DR_BC1.5A')
lerf = Drift(eid='LERF')
dr_bc1_5b = Drift(l=0.1975, eid='DR_BC1.5B')
dr_bc1_6 = Drift(l=0.3737, eid='DR_BC1.6')
dr_bc1_6a = Drift(l=0.1022, eid='DR_BC1.6A')
dr_bc1_6b = Drift(l=0.3522, eid='DR_BC1.6B')
dr_bc1_7b = Drift(l=0.3982, eid='DR_BC1.7B')
dr_bc1_7c = Drift(l=0.0622, eid='DR_BC1.7C')
dr_bc1_7d = Drift(l=0.1872, eid='DR_BC1.7D')
dr_bc1_7e = Drift(l=0.3986, eid='DR_BC1.7E')
gcol_bc01_01 = Drift(l=0.5, eid='GCOL_BC01.01')
dr_bc1_7f = Drift(l=0.3124, eid='DR_BC1.7F')
dr_bc1_8b = Drift(l=4.9573, eid='DR_BC1.8B')
dr_bc1_9 = Drift(l=0.6031, eid='DR_BC1.9')
dr_bc1_9a = Drift(l=0.1592, eid='DR_BC1.9A')
dr_bc1_10 = Drift(l=0.2332, eid='DR_BC1.10')
dr_bc1_11 = Drift(l=0.2742, eid='DR_BC1.11')
dr_bc1_12 = Drift(l=0.1082, eid='DR_BC1.12')
dr_bc1_13 = Drift(l=0.15, eid='DR_BC1.13')
dr_bc1_14 = Drift(l=1.68816, eid='DR_BC1.14')
dr_bc1_15 = Drift(l=0.1819, eid='DR_BC1.15')
dr_bc1_15a = Drift(l=0.3661, eid='DR_BC1.15A')
dr_bc1_15b = Drift(l=0.8314, eid='DR_BC1.15B')
dr_bc1_15c = Drift(l=0.2842, eid='DR_BC1.15C')
dr_bc1_16a = Drift(l=0.0512, eid='DR_BC1.16A')
cm_lh_01 = Drift(l=0.04, eid='CM_LH.01')
dr_bc1_16b = Drift(l=0.10092, eid='DR_BC1.16B')
dr_l2_sis1 = Drift(l=0.12956, eid='DR_L2.SIS1')
dr_l2_sis2 = Drift(l=0.0076, eid='DR_L2.SIS2')
dr_l2_sis3 = Drift(l=0.22854, eid='DR_L2.SIS3')
dr_l2_sis4 = Drift(l=0.09455, eid='DR_L2.SIS4')
dr_l2_sis5 = Drift(l=0.12825, eid='DR_L2.SIS5')
dr_l2_sis7 = Drift(l=0.0725, eid='DR_L2.SIS7')
dr_l2_sis8 = Drift(l=0.2492, eid='DR_L2.SIS8')
dr_l2_sis9 = Drift(l=0.0741, eid='DR_L2.SIS9')
dr_l2_sis10 = Drift(l=0.1382, eid='DR_L2.SIS10')
dr_l2_sis12 = Drift(l=0.0625, eid='DR_L2.SIS12')
dr_l2_sis13 = Drift(l=0.2335, eid='DR_L2.SIS13')
dr_l2_sis14 = Drift(l=0.0002, eid='DR_L2.SIS14')
dr_l2_sis15 = Drift(l=0.2396, eid='DR_L2.SIS15')
dr_l2_sis18 = Drift(l=0.0984, eid='DR_L2.SIS18')
dr_l2_sis19 = Drift(l=0.1291, eid='DR_L2.SIS19')
dr_l2_sis20 = Drift(l=0.1293, eid='DR_L2.SIS20')
dr_acc_s_1_1 = Drift(l=0.175001, eid='DR_ACC_S_1_1')
dr_acc_s_1_2 = Drift(l=0.145001, eid='DR_ACC_S_1_2')
dr_l3_sis1 = Drift(l=0.1301, eid='DR_L3.SIS1')
dr_l3_sis2 = Drift(l=0.1399, eid='DR_L3.SIS2')
dr_l3_sis3 = Drift(l=0.098, eid='DR_L3.SIS3')
dr_l3_sis4 = Drift(l=0.1078, eid='DR_L3.SIS4')
dr_l3_sis5 = Drift(l=0.1029, eid='DR_L3.SIS5')
dr_acc_s_2_1 = Drift(l=0.175002, eid='DR_ACC_S_2_1')
dr_acc_s_2_2 = Drift(l=0.145002, eid='DR_ACC_S_2_2')
dr_l3_sis6 = Drift(l=0.164, eid='DR_L3.SIS6')
dr_l3_sis7 = Drift(l=0.0517, eid='DR_L3.SIS7')
dr_l3_sis8 = Drift(l=0.1229, eid='DR_L3.SIS8')
dr_l3_sis9 = Drift(l=0.1632, eid='DR_L3.SIS9')
dr_l3_sis10 = Drift(l=0.1911, eid='DR_L3.SIS10')
dr_bc2_1a = Drift(l=0.3932, eid='DR_BC2.1A')
dr_bc2_1b = Drift(l=2.1975, eid='DR_BC2.1B')
dr_bc2_2a = Drift(l=0.2327, eid='DR_BC2.2A')
dr_bc2_2c = Drift(l=0.1027, eid='DR_BC2.2C')
dr_bc2_2d = Drift(l=0.1787, eid='DR_BC2.2D')
dr_bc2_2e = Drift(l=0.422, eid='DR_BC2.2E')
dr_bc2_3a = Drift(l=2.1715, eid='DR_BC2.3A')
dr_bc2_3b = Drift(l=0.4195, eid='DR_BC2.3B')
dr_bc2_5a = Drift(l=0.004, eid='DR_BC2.5A')
dr_bc2_5b = Drift(l=0.621, eid='DR_BC2.5B')
dr_bc2_sis1 = Drift(l=0.0454, eid='DR_BC2.SIS1')
dr_bc2_sis2 = Drift(l=0.0652, eid='DR_BC2.SIS2')
dr_bc2_sis3 = Drift(l=0.1139, eid='DR_BC2.SIS3')
dr_bc2_sis4 = Drift(l=0.1181, eid='DR_BC2.SIS4')
dr_bc2_sis5 = Drift(l=0.1226, eid='DR_BC2.SIS5')
dr_l4_sis2 = Drift(l=0.0642, eid='DR_L4.SIS2')
dr_l4_sis3 = Drift(l=0.11, eid='DR_L4.SIS3')
dr_l4_sis4 = Drift(l=0.1552, eid='DR_L4.SIS4')
dr_l4_sis5 = Drift(l=0.1028, eid='DR_L4.SIS5')
dr_l4_sis6 = Drift(l=0.1294, eid='DR_L4.SIS6')
bpm_dr_l4_2 = Drift(l=0.055, eid='BPM_DR_L4.2')
dr_l4_sis7 = Drift(l=0.0012, eid='DR_L4.SIS7')
dr_l4_sis8 = Drift(l=0.2379, eid='DR_L4.SIS8')
dr_l4_sis9 = Drift(l=0.0791, eid='DR_L4.SIS9')
dr_l4_sis10 = Drift(l=0.1402, eid='DR_L4.SIS10')
dr_acc_s_3_1 = Drift(l=0.175003, eid='DR_ACC_S_3_1')
dr_acc_s_3_2 = Drift(l=0.145003, eid='DR_ACC_S_3_2')
dr_l4_sis11 = Drift(l=0.1297, eid='DR_L4.SIS11')
bpm_dr_l4_3 = Drift(l=0.068, eid='BPM_DR_L4.3')
dr_l4_sis14 = Drift(l=0.0781, eid='DR_L4.SIS14')
dr_l4_sis15 = Drift(l=0.1398, eid='DR_L4.SIS15')
dr_acc_s_4_1 = Drift(l=0.175004, eid='DR_ACC_S_4_1')
dr_acc_s_4_2 = Drift(l=0.145004, eid='DR_ACC_S_4_2')
dr_l4_sis16 = Drift(l=0.1299, eid='DR_L4.SIS16')
bpm_dr_l4_4 = Drift(l=0.03, eid='BPM_DR_L4.4')
dr_l4_sis17 = Drift(l=0.038, eid='DR_L4.SIS17')
dr_l4_sis19 = Drift(l=0.0799, eid='DR_L4.SIS19')
dr_l4_sis20 = Drift(l=0.1394, eid='DR_L4.SIS20')
dr_acc_s_5_1 = Drift(l=0.175005, eid='DR_ACC_S_5_1')
dr_acc_s_5_2 = Drift(l=0.145005, eid='DR_ACC_S_5_2')
dr_l4_sis24 = Drift(l=0.0787, eid='DR_L4.SIS24')
dr_l4_sis25 = Drift(l=0.2581, eid='DR_L4.SIS25')
dr_l4_sis26 = Drift(l=0.2452, eid='DR_L4.SIS26')
dr_l4_sis29 = Drift(l=0.0546, eid='DR_L4.SIS29')
dr_l4_sis30 = Drift(l=0.1288, eid='DR_L4.SIS30')
dr_l4_9 = Drift(l=0.1017, eid='DR_L4.9')
dr_l4_10 = Drift(l=0.2706, eid='DR_L4.10')
dr_l4_10a = Drift(l=0.0388, eid='DR_L4.10A')
dr_l4_10b = Drift(l=0.325, eid='DR_L4.10B')
dr_l4_11 = Drift(l=0.318, eid='DR_L4.11')
dr_tls_2 = Drift(l=0.1335, eid='DR_TLS.2')
dr_tls_3 = Drift(l=0.418, eid='DR_TLS.3')

# Quadrupoles
q_lh_04 = Quadrupole(l=0.065, k1=10.145619656515303, eid='Q_LH.04')
q_lh_05 = Quadrupole(l=0.065, k1=-4.943570504596247, eid='Q_LH.05')
q_lh_06 = Quadrupole(l=0.065, k1=4.494562317700287, eid='Q_LH.06')
q_lh_07 = Quadrupole(l=0.065, k1=0.443393459905855, eid='Q_LH.07')
q_l01_01 = Quadrupole(l=0.065, k1=-2.9518006327349338, eid='Q_L01.01')
q_l01_02 = Quadrupole(l=0.065, k1=1.1484912707889778, eid='Q_L01.02')
q_l01_03 = Quadrupole(l=0.065, k1=0.44303024707767913, eid='Q_L01.03')
q_l01_04 = Quadrupole(l=0.065, k1=1.5189054191237887, eid='Q_L01.04')
q_bc01_01 = Quadrupole(l=0.065, eid='Q_BC01.01')
q_bc01_02 = Quadrupole(l=0.065, eid='Q_BC01.02')
q_bc01_03 = Quadrupole(l=0.065, k1=-1.1847195817336877, eid='Q_BC01.03')
q_bc01_04 = Quadrupole(l=0.0876, k1=-4.281899345588231, eid='Q_BC01.04')
q_bc01_05 = Quadrupole(l=0.0876, k1=2.513071868174999, eid='Q_BC01.05')
q_bc01_06 = Quadrupole(l=0.065, k1=5.271312855361364, eid='Q_BC01.06')
q_bc01_07 = Quadrupole(l=0.0876, k1=0.08483188045405614, eid='Q_BC01.07')
q_bc01_08 = Quadrupole(l=0.1636, k1=3.06601735883316e-17, eid='Q_BC01.08')
q_bc01_09 = Quadrupole(l=0.1636, eid='Q_BC01.09')
q_bc01_10 = Quadrupole(l=0.0876, k1=6.283341707621023, eid='Q_BC01.10')
q_bc01_11 = Quadrupole(l=0.0876, k1=-6.28334759436536, eid='Q_BC01.11')
q_l02_01 = Quadrupole(l=0.065, eid='Q_L02.01')
q_l02_02 = Quadrupole(l=0.065, k1=1.6793230990120382, eid='Q_L02.02')
q_l02_03 = Quadrupole(l=0.065, eid='Q_L02.03')
q_l02_04 = Quadrupole(l=0.065, k1=-1.8522913694301653, eid='Q_L02.04')
q_l03_01 = Quadrupole(l=0.0876, k1=1.44635865963236, eid='Q_L03.01')
q_l03_02 = Quadrupole(l=0.0876, k1=-1.052392277337335, eid='Q_L03.02')
q_bc02_01 = Quadrupole(l=0.065, eid='Q_BC02.01')
q_bc02_02 = Quadrupole(l=0.065, eid='Q_BC02.02')
q_bc02_03 = Quadrupole(l=0.0876, k1=0.7115732699561141, eid='Q_BC02.03')
q_l04_01 = Quadrupole(l=0.0876, k1=-0.8849857110564808, eid='Q_L04.01')
q_l04_02 = Quadrupole(l=0.0876, k1=1.1447074983321686, eid='Q_L04.02')
q_l04_03 = Quadrupole(l=0.065, k1=-1.2678921011081394, eid='Q_L04.03')
q_l04_04 = Quadrupole(l=0.065, k1=-0.000465549277857299, eid='Q_L04.04')
q_l04_05 = Quadrupole(l=0.065, k1=-1.2697213068883173, eid='Q_L04.05')
q_l04_06 = Quadrupole(l=0.0876, k1=-0.08954282060903239, eid='Q_L04.06')
q_l04_07 = Quadrupole(l=0.1636, k1=1.2649231847022102, eid='Q_L04.07')
q_tls_01 = Quadrupole(l=0.282, k1=1.7570746346426755, eid='Q_TLS.01')
q_tls_02 = Quadrupole(l=0.282, k1=-1.7662843972876905, eid='Q_TLS.02')

# SBends
b_lh_01 = SBend(l=0.2, angle=0.0609, e2=0.0609, eid='B_LH.01')
b_lh_02 = SBend(l=0.2, angle=-0.0609, e1=-0.0609, eid='B_LH.02')
b_lh_03 = SBend(l=0.2, angle=-0.0609, e2=-0.0609, eid='B_LH.03')
b_lh_04 = SBend(l=0.2, angle=0.0609, e1=0.0609, eid='B_LH.04')
b_bc01_01 = SBend(l=0.366, angle=-0.08499976178059568, e2=-0.08499976178059568, eid='B_BC01.01')
b_bc01_02 = SBend(l=0.366, angle=0.08499976178059568, e1=0.08499976178059568, eid='B_BC01.02')
b_bc01_03 = SBend(l=0.366, angle=0.08499976178059568, e2=0.08499976178059568, eid='B_BC01.03')
b_bc01_04 = SBend(l=0.366, angle=-0.08499976178059568, e1=-0.08499976178059568, eid='B_BC01.04')
b_bc02_01 = SBend(l=0.366, eid='B_BC02.01')
b_bc02_02 = SBend(l=0.366, eid='B_BC02.02')
b_bc02_03 = SBend(l=0.366, eid='B_BC02.03')
b_bc02_04 = SBend(l=0.366, eid='B_BC02.04')

# Vcors
chv_lh_02 = Vcor(l=0.158, eid='CHV_LH.02')
chv_lh_03 = Vcor(l=0.158, eid='CHV_LH.03')
chv_lh_04 = Vcor(l=0.158, eid='CHV_LH.04')
chv_l01_01 = Vcor(l=0.158, eid='CHV_L01.01')
chv_l01_02 = Vcor(l=0.158, eid='CHV_L01.02')
chv_l01_03 = Vcor(l=0.158, eid='CHV_L01.03')
chv_l01_04 = Vcor(l=0.158, eid='CHV_L01.04')
chv_bc01_01 = Vcor(l=0.158, eid='CHV_BC01.01')
chv_bc01_02 = Vcor(l=0.158, eid='CHV_BC01.02')
chv_bc01_03 = Vcor(l=0.158, eid='CHV_BC01.03')
chv_bc01_04 = Vcor(l=0.158, eid='CHV_BC01.04')
chv_bc01_05 = Vcor(l=0.158, eid='CHV_BC01.05')
chv_bc01_06 = Vcor(l=0.158, eid='CHV_BC01.06')
chv_l02_01 = Vcor(l=0.0696, eid='CHV_L02.01')
chv_l02_02 = Vcor(l=0.0696, eid='CHV_L02.02')
chv_l02_03 = Vcor(l=0.0696, eid='CHV_L02.03')
chv_l02_04 = Vcor(l=0.0696, eid='CHV_L02.04')
chv_l03_01 = Vcor(l=0.0696, eid='CHV_L03.01')
chv_l03_02 = Vcor(l=0.0696, eid='CHV_L03.02')
chv_bc02_01 = Vcor(l=0.0696, eid='CHV_BC02.01')
chv_l04_01 = Vcor(l=0.0696, eid='CHV_L04.01')
chv_l04_02 = Vcor(l=0.0696, eid='CHV_L04.02')
chv_l04_03 = Vcor(l=0.0696, eid='CHV_L04.03')
chv_l04_04 = Vcor(l=0.0696, eid='CHV_L04.04')
chv_l04_05 = Vcor(l=0.0696, eid='CHV_L04.05')
chv_l04_06 = Vcor(l=0.0696, eid='CHV_L04.06')
chv_l04_07 = Vcor(l=0.0696, eid='CHV_L04.07')

# Cavitys
acct_l01_01 = Cavity(l=4.572, v=0.057148332287523196, phi=26.49971008252541, freq=2997924000.0, eid='ACCT_L01_01')
acct_l01_02 = Cavity(l=4.572, v=0.057148332287523196, phi=26.49971008252541, freq=2997924000.0, eid='ACCT_L01_02')
acch_l01 = Cavity(l=0.595, v=0.010825692018083305, phi=179.95213165618068, freq=11991600000.0, eid='ACCH_L01')
acct_l01_03 = Cavity(l=4.572, v=0.05529223260638997, phi=26.499023437017613, freq=2997924000.0, eid='ACCT_L01_03')
acct_l01_04 = Cavity(l=4.572, v=0.05529223260638997, phi=26.499023437017613, freq=2997924000.0, eid='ACCT_L01_04')
wdcav_bc1_1 = Cavity(l=0.25, freq=2997924000.0, eid='WDCAV_BC1.1')
acct_l02_1 = Cavity(l=4.572, v=0.05685433581082914, freq=2997924000.0, eid='ACCT_L02_1')
acct_l02_2 = Cavity(l=4.572, v=0.05685433581082914, freq=2997924000.0, eid='ACCT_L02_2')
acct_l02_3 = Cavity(l=4.572, v=0.04914527332800485, freq=2997924000.0, eid='ACCT_L02_3')
acct_l02_04 = Cavity(l=3.2, v=0.04914527332800485, freq=2997924000.0, eid='ACCT_L02.04')
accbt_l03_1 = Cavity(l=6.1125, v=0.12334499972518864, freq=2997924000.0, eid='ACCBT_L03_1')
accbt_l03_2 = Cavity(l=6.1125, v=0.12912679658730686, freq=2997924000.0, eid='ACCBT_L03_2')
accbt_l04_1 = Cavity(l=6.1125, v=0.1223813669148356, freq=2997924000.0, eid='ACCBT_L04_1')
accbt_l04_2 = Cavity(l=6.1125, v=0.1223813669148356, freq=2997924000.0, eid='ACCBT_L04_2')
accbt_l04_3 = Cavity(l=6.1125, v=0.1252722653458947, freq=2997924000.0, eid='ACCBT_L04_3')
accbt_l04_4 = Cavity(l=6.1125, v=0.12912679658730686, phi=20.0730514522713, freq=2997924000.0, eid='ACCBT_L04_4')
accbt_l04_5 = Cavity(l=6.1125, v=0.0001, phi=-100.49983978088528, freq=2997924000.0, eid='ACCBT_L04_5')
acct_l04_06 = Cavity(l=3.2, v=0.0001, freq=2997924000.0, eid='ACCT_L04.06')
wdcav_l4_1 = Cavity(l=1.25, freq=2997924000.0, eid='WDCAV_L4.1')
wdcav_l4_2 = Cavity(l=1.25, freq=2997924000.0, eid='WDCAV_L4.2')

# Monitors
bpm_pk_lh_1 = Monitor(eid='BPM_PK_LH.1')
bpm_cen_lh_1 = Monitor(eid='BPM_CEN_LH.1')
bpm_pk_lh_2 = Monitor(eid='BPM_PK_LH.2')
bpm_cen_lh_2 = Monitor(eid='BPM_CEN_LH.2')
bpm_pk_lh_3 = Monitor(eid='BPM_PK_LH.3')
bpm_cen_lh_3 = Monitor(eid='BPM_CEN_LH.3')
bpm_pk_lh_4 = Monitor(eid='BPM_PK_LH.4')
bpm_cen_lh_4 = Monitor(eid='BPM_CEN_LH.4')
bpm_pk_lh_5 = Monitor(eid='BPM_PK_LH.5')
bpm_cen_lh_5 = Monitor(eid='BPM_CEN_LH.5')
bpm_pk_l1_1 = Monitor(eid='BPM_PK_L1.1')
bpm_cen_l1_1 = Monitor(eid='BPM_CEN_L1.1')
bpm_pk_l1_2 = Monitor(eid='BPM_PK_L1.2')
bpm_cen_l1_2 = Monitor(eid='BPM_CEN_L1.2')
bpm_pk_l1_3 = Monitor(eid='BPM_PK_L1.3')
bpm_cen_l1_3 = Monitor(eid='BPM_CEN_L1.3')
bpm_pk_l1_4 = Monitor(eid='BPM_PK_L1.4')
bpm_cen_l1_4 = Monitor(eid='BPM_CEN_L1.4')
bpm_pk_l1_5 = Monitor(eid='BPM_PK_L1.5')
bpm_cen_l1_5 = Monitor(eid='BPM_CEN_L1.5')
bpm_pk_bc1_1 = Monitor(eid='BPM_PK_BC1.1')
bpm_cen_bc1_1 = Monitor(eid='BPM_CEN_BC1.1')
bpm_pk_bc1_2 = Monitor(eid='BPM_PK_BC1.2')
bpm_cen_bc1_2 = Monitor(eid='BPM_CEN_BC1.2')
bpm_pk_bc1_3 = Monitor(eid='BPM_PK_BC1.3')
bpm_cen_bc1_3 = Monitor(eid='BPM_CEN_BC1.3')
bpm_pk_bc1_4 = Monitor(eid='BPM_PK_BC1.4')
bpm_cen_bc1_4 = Monitor(eid='BPM_CEN_BC1.4')
bpm_pk_bc1_5 = Monitor(eid='BPM_PK_BC1.5')
bpm_cen_bc1_5 = Monitor(eid='BPM_CEN_BC1.5')
bpm_pk_l2_1 = Monitor(eid='BPM_PK_L2.1')
bpm_cen_l2_1 = Monitor(eid='BPM_CEN_L2.1')
bpm_pk_l2_2 = Monitor(eid='BPM_PK_L2.2')
bpm_cen_l2_2 = Monitor(eid='BPM_CEN_L2.2')
bpm_pk_l2_3 = Monitor(eid='BPM_PK_L2.3')
bpm_cen_l2_3 = Monitor(eid='BPM_CEN_L2.3')
bpm_pk_l2_4 = Monitor(eid='BPM_PK_L2.4')
bpm_cen_l2_4 = Monitor(eid='BPM_CEN_L2.4')
bpm_pk_l3_1 = Monitor(eid='BPM_PK_L3.1')
bpm_cen_l3_1 = Monitor(eid='BPM_CEN_L3.1')
bpm_pk_l3_2 = Monitor(eid='BPM_PK_L3.2')
bpm_cen_l3_2 = Monitor(eid='BPM_CEN_L3.2')
bpm_pk_bc2_1 = Monitor(eid='BPM_PK_BC2.1')
bpm_cen_bc2_1 = Monitor(eid='BPM_CEN_BC2.1')
bpm_pk_bc2_2 = Monitor(eid='BPM_PK_BC2.2')
bpm_cen_bc2_2 = Monitor(eid='BPM_CEN_BC2.2')
bpm_pk_l4_1 = Monitor(eid='BPM_PK_L4.1')
bpm_cen_l4_1 = Monitor(eid='BPM_CEN_L4.1')
bpm_pk_l4_2 = Monitor(eid='BPM_PK_L4.2')
bpm_cen_l4_2 = Monitor(eid='BPM_CEN_L4.2')
bpm_pk_l4_3 = Monitor(eid='BPM_PK_L4.3')
bpm_cen_l4_3 = Monitor(eid='BPM_CEN_L4.3')
bpm_pk_l4_4 = Monitor(eid='BPM_PK_L4.4')
bpm_cen_l4_4 = Monitor(eid='BPM_CEN_L4.4')
bpm_pk_l4_5 = Monitor(eid='BPM_PK_L4.5')
bpm_cen_l4_5 = Monitor(eid='BPM_CEN_L4.5')
bpm_pk_l4_6 = Monitor(eid='BPM_PK_L4.6')
bpm_cen_l4_6 = Monitor(eid='BPM_CEN_L4.6')
bpm_pk_l4_7 = Monitor(eid='BPM_PK_L4.7')
bpm_cen_l4_7 = Monitor(eid='BPM_CEN_L4.7')

# Markers
mkp_lh = Marker(eid='MKP_LH')
mkp_mscr_lh_3 = Marker(eid='MKP_MSCR_LH.3')
mk_acct_l01_01 = Marker(eid='MK_ACCT_L01_01')
mk_acct_l01_02 = Marker(eid='MK_ACCT_L01_02')
mk_acct_l01_03 = Marker(eid='MK_ACCT_L01_03')
mk_acct_l01_04 = Marker(eid='MK_ACCT_L01_04')
mk_bc1start = Marker(eid='MK_BC1START')
mk_bc1 = Marker(eid='MK_BC1')
mk_ledcav = Marker(eid='MK_LEDCAV')
mkp_q_bc1_3 = Marker(eid='MKP_Q_BC1.3')
mkp_q_bc1_4 = Marker(eid='MKP_Q_BC1.4')
mkp_q_bc1_5 = Marker(eid='MKP_Q_BC1.5')
mkp_q_bc1_6 = Marker(eid='MKP_Q_BC1.6')
mkp_q_bc1_7 = Marker(eid='MKP_Q_BC1.7')
mk_bc1end = Marker(eid='MK_BC1END')
mkp_bc1diag = Marker(eid='MKP_BC1DIAG')
mkp_mscr_bc1_3 = Marker(eid='MKP_MSCR_BC1.3')
mk_acct_l02_01 = Marker(eid='MK_ACCT_L02_01')
mk_acct_l02_02 = Marker(eid='MK_ACCT_L02_02')
mk_acct_l02_03 = Marker(eid='MK_ACCT_L02_03')
mk_acct_l02_04 = Marker(eid='MK_ACCT_L02_04')
mk_acct_l03_01 = Marker(eid='MK_ACCT_L03_01')
mk_l3 = Marker(eid='MK_L3')
mk_acct_l03_02 = Marker(eid='MK_ACCT_L03_02')
mk_bc2start = Marker(eid='MK_BC2START')
mk_bc2 = Marker(eid='MK_BC2')
mk_bc2end = Marker(eid='MK_BC2END')
mk_acct_l04_01 = Marker(eid='MK_ACCT_L04_01')
mk_l4 = Marker(eid='MK_L4')
mk_acct_l04_02 = Marker(eid='MK_ACCT_L04_02')
mk_l4_2 = Marker(eid='MK_L4_2')
mk_acct_l04_03 = Marker(eid='MK_ACCT_L04_03')
mk_l4_3 = Marker(eid='MK_L4_3')
mk_acct_l04_04 = Marker(eid='MK_ACCT_L04_04')
mk_l4_4 = Marker(eid='MK_L4_4')
mk_acct_l04_05 = Marker(eid='MK_ACCT_L04_05')
mk_l4_5 = Marker(eid='MK_L4_5')
mkp_q_l4_6 = Marker(eid='MKP_Q_L4.6')
mk_l4_6 = Marker(eid='MK_L4_6')
mkp_q_l4_7 = Marker(eid='MKP_Q_L4.7')
mkp_q_tls_2 = Marker(eid='MKP_Q_TLS.2')

# Lattice 
cell = (q_lh_04, dr_lh_8, mkp_lh, mscr_lh_01, dr_lh_9, chv_lh_02, dr_lh_10, bpm_pk_lh_1, bpm_dr, 
bpm_cen_lh_1, dr_lh_11, dr_lh_2, dr_lh_12, bpm_pk_lh_2, bpm_dr, bpm_cen_lh_2, dr_lh_13, b_lh_01, dr_lh_24, 
b_lh_02, dr_lh_25, mscr_lh_01, dr_lh_26, und_lh_01, dr_lh_26, mkp_mscr_lh_3, mscr_lh_01, dr_lh_28, bpm_pk_lh_3, 
bpm_dr, bpm_cen_lh_3, dr_lh_29, b_lh_03, dr_lh_24, b_lh_04, dr_lh_14, chv_lh_03, dr_lh_15, bpm_pk_lh_4, 
bpm_dr, bpm_cen_lh_4, dr_lh_15a, q_lh_05, dr_lh_16, dr_lh_17, bpm_pk_lh_5, bpm_dr, bpm_cen_lh_5, dr_lh_18, 
mscr_lh_01, dr_lh_19, q_lh_06, dr_lh_20, dr_lh_21, chv_lh_04, dr_lh_21, q_lh_07, dr_lh_23, dr_acc_c_1_1, 
mk_acct_l01_01, acct_l01_01, dr_acc_c_1_2, dr_l1_sis1, bpm_pk_l1_1, bpm_cen_l1_1, dr_l1_sis2, q_l01_01, dr_l1_sis3, dr_l1_sis4, 
chv_l01_01, dr_l1_sis5, dr_acc_c_2_1, mk_acct_l01_02, acct_l01_02, dr_acc_c_2_2, dr_l1_8a, bpm_pk_l1_2, bpm_cen_l1_2, dr_l1_8b, 
q_l01_02, dr_l1_8c, chv_l01_02, dr_l1_9a, acch_l01, dr_l1_9b, bpm_pk_l1_3, bpm_dr, bpm_cen_l1_3, dr_l1_10, 
dr_acc_c_3_1, mk_acct_l01_03, acct_l01_03, dr_acc_c_3_2, dr_l1_sis1, bpm_pk_l1_4, bpm_cen_l1_4, dr_l1_sis7, q_l01_03, dr_l1_sis8, 
mscr_lh_01, dr_l1_sis9, chv_l01_03, dr_l1_sis10, dr_acc_c_4_1, mk_acct_l01_04, acct_l01_04, dr_acc_c_4_2, dr_l1_sis1, bpm_pk_l1_5, 
bpm_cen_l1_5, dr_l1_sis12, q_l01_04, dr_l1_sis13, dr_l1_sis14, chv_l01_04, dr_l1_sis15, dr_l1_14, mk_bc1start, b_bc01_01, 
dr_bc1_1a, q_bc01_01, dr_bc1_1b, b_bc01_02, dr_bc1_2a, scrph_bc01_01, dr_bc1_2b, dr_bc1_2c, bpm_pk_bc1_1, bpm_dr, 
bpm_cen_bc1_1, dr_bc1_2d, mscr_lh_01, dr_bc1_2e, b_bc01_03, mk_bc1, dr_bc1_3a, q_bc01_02, dr_bc1_3b, b_bc01_04, 
mk_bc1, dr_bc1_4, cblm_bc01_01, gblm_bc01_01, dr_bc1_4b, chv_bc01_01, dr_lh_10, dr_lh_2, dr_bc1_4d, bpm_pk_bc1_2, 
bpm_dr, bpm_cen_bc1_2, dr_bc1_5a, mk_ledcav, wdcav_bc1_1, lerf, wdcav_bc1_1, dr_bc1_5b, mkp_q_bc1_3, q_bc01_03, 
dr_bc1_6, mkp_q_bc1_4, q_bc01_04, dr_bc1_6a, chv_bc01_02, dr_bc1_6b, mkp_q_bc1_5, q_bc01_05, dr_bc1_6, mkp_q_bc1_6, 
q_bc01_06, dr_bc1_7b, bpm_pk_bc1_3, bpm_cen_bc1_3, dr_bc1_7c, mkp_q_bc1_7, q_bc01_07, mk_bc1end, dr_bc1_7d, chv_bc01_03, 
dr_bc1_7e, gcol_bc01_01, dr_bc1_7f, mkp_bc1diag, mscr_lh_01, gblm_bc01_01, dr_bc1_8b, mkp_mscr_bc1_3, mscr_lh_01, dr_bc1_9, 
chv_bc01_04, dr_bc1_9a, q_bc01_08, dr_bc1_10, chv_bc01_05, dr_bc1_11, q_bc01_09, dr_bc1_12, bpm_pk_bc1_4, bpm_dr, 
bpm_cen_bc1_4, dr_bc1_13, dr_bc1_14, bpm_pk_bc1_5, bpm_dr, bpm_cen_bc1_5, dr_bc1_15, gcol_bc01_01, dr_bc1_15a, mscr_lh_01, 
mkp_bc1diag, dr_bc1_15b, q_bc01_10, dr_bc1_15c, chv_bc01_06, dr_bc1_6a, q_bc01_11, dr_bc1_16a, cm_lh_01, dr_bc1_16b, 
dr_acc_c_1_1, mk_acct_l02_01, acct_l02_1, dr_acc_c_1_2, dr_l2_sis1, bpm_pk_l2_1, bpm_dr, bpm_cen_l2_1, dr_l2_sis2, q_l02_01, 
dr_l2_sis3, dr_l2_sis4, chv_l02_01, dr_l2_sis5, dr_acc_c_2_1, mk_acct_l02_02, acct_l02_2, dr_acc_c_2_2, dr_l1_sis1, bpm_pk_l2_2, 
bpm_cen_l2_2, dr_l2_sis7, q_l02_02, dr_l2_sis8, dr_l2_sis9, chv_l02_02, dr_l2_sis10, dr_acc_c_3_1, mk_acct_l02_03, acct_l02_3, 
dr_acc_c_3_2, dr_l1_sis1, bpm_pk_l2_3, bpm_cen_l2_3, dr_l2_sis12, q_l02_03, dr_l2_sis13, dr_l2_sis14, chv_l02_03, dr_l2_sis15, 
mk_acct_l02_04, acct_l02_04, dr_lh_28, bpm_pk_l2_4, bpm_cen_l2_4, dr_l2_sis12, q_l02_04, dr_l2_sis18, dr_l2_sis19, chv_l02_04, 
dr_l2_sis20, dr_acc_s_1_1, mk_acct_l03_01, accbt_l03_1, dr_acc_s_1_2, dr_l3_sis1, bpm_pk_l3_1, bpm_cen_l3_1, mk_l3, q_l03_01, 
dr_l3_sis2, dr_l3_sis3, dr_l3_sis4, chv_l03_01, dr_l3_sis5, dr_acc_s_2_1, mk_acct_l03_02, accbt_l03_2, dr_acc_s_2_2, dr_l3_sis6, 
bpm_pk_l3_2, bpm_cen_l3_2, dr_l3_sis7, q_l03_02, dr_l3_sis8, mscr_lh_01, dr_l3_sis9, chv_l03_02, dr_l3_sis10, mk_bc2start, 
b_bc02_01, dr_bc2_1a, q_bc02_01, dr_bc2_1b, b_bc02_02, dr_bc2_2a, scrph_bc01_01, lerf, dr_bc2_2c, bpm_pk_bc2_1, 
bpm_dr, bpm_cen_bc2_1, dr_bc2_2d, mscr_lh_01, dr_bc2_2e, b_bc02_03, mk_bc2, dr_bc2_3a, q_bc02_02, dr_bc2_3b, 
b_bc02_04, mk_bc2, dr_bc2_5a, cblm_bc01_01, gblm_bc01_01, dr_bc2_5b, dr_lh_2, dr_bc2_sis1, bpm_pk_bc2_2, bpm_cen_bc2_2, 
dr_bc2_sis2, mk_bc2end, q_bc02_03, mk_l3, dr_bc2_sis3, cm_lh_01, dr_bc2_sis4, chv_bc02_01, dr_bc2_sis5, dr_acc_s_1_1, 
mk_acct_l04_01, accbt_l04_1, dr_acc_s_1_2, dr_l3_sis1, bpm_pk_l4_1, bpm_cen_l4_1, dr_l4_sis2, q_l04_01, mk_l4, dr_l4_sis3, 
mscr_lh_01, dr_l4_sis4, chv_l04_01, dr_l4_sis5, dr_acc_s_2_1, mk_acct_l04_02, accbt_l04_2, dr_acc_s_2_2, dr_l4_sis6, bpm_pk_l4_2, 
bpm_dr_l4_2, bpm_cen_l4_2, dr_l4_sis7, q_l04_02, mk_l4_2, dr_l4_sis8, dr_l4_sis9, chv_l04_02, dr_l4_sis10, dr_acc_s_3_1, 
mk_acct_l04_03, accbt_l04_3, dr_acc_s_3_2, dr_l4_sis11, bpm_pk_l4_3, bpm_dr_l4_3, bpm_cen_l4_3, dr_l2_sis14, q_l04_03, mk_l4_3, 
dr_l2_sis8, dr_l4_sis14, chv_l04_03, dr_l4_sis15, dr_acc_s_4_1, mk_acct_l04_04, accbt_l04_4, dr_acc_s_4_2, dr_l4_sis16, bpm_pk_l4_4, 
bpm_dr_l4_4, bpm_cen_l4_4, dr_l4_sis17, q_l04_04, mk_l4_4, dr_l2_sis8, dr_l4_sis19, chv_l04_04, dr_l4_sis20, dr_acc_s_5_1, 
mk_acct_l04_05, accbt_l04_5, dr_acc_s_5_2, dr_l4_sis16, bpm_pk_l4_5, bpm_dr_l4_3, bpm_cen_l4_5, lerf, q_l04_05, mk_l4_5, 
dr_l2_sis8, dr_l4_sis24, chv_l04_05, dr_l4_sis25, acct_l04_06, dr_l4_sis26, bpm_pk_l4_6, bpm_dr_l4_2, bpm_cen_l4_6, dr_l4_sis7, 
mkp_q_l4_6, q_l04_06, mk_l4_6, dr_l4_sis8, dr_l4_sis29, chv_l04_06, dr_l4_sis30, dr_l4_9, wdcav_l4_1, lerf, 
wdcav_l4_1, dr_l4_10, mkp_q_l4_7, q_l04_07, mk_l4, dr_l4_10a, bpm_pk_l4_7, bpm_dr, bpm_cen_l4_7, dr_l4_10b, 
chv_l04_07, dr_l4_11, wdcav_l4_2, lerf, wdcav_l4_2, lerf, dr_tls_2, q_tls_01, dr_tls_3, mkp_q_tls_2, 
q_tls_02)