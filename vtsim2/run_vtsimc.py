import numpy as np
import pandas as pd
import time

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import vtsimc as vt

STEP_P      = 1e-6        #偏微分時の圧力変化
VENT_ERR    = 1e-6        #換気回路網の許容残差
STEP_T      = 1e-6        #偏微分時の温度変化
THRM_ERR    = 1e-6        #熱回路網の許容残差
SOR_RATIO   = 0.9         #SOR法の緩和係数
SOR_ERR     = 1e-6        #SOR法の許容残差

SOLVE_LU    = 0
SOLVE_SOR   = 1

FLAG_NONE   = 0           #計算しない
FLAG_CALC   = 1           #計算する
FLAG_FIX    = 2           #固定値（計算には利用するが、更新しない）
FLAG_DLY    = 3           #遅延（熱容量計算用）

VN_SIMPLE   = 0           #換気回路網：単純開口
VN_GAP      = 1           #換気回路網：隙間
VN_FIX      = 2           #換気回路網：風量固定
VN_AIRCON   = 3           #換気回路網：エアコン=風量固定、換気による熱移動=0
VN_FAN      = 4           #換気回路網：送風ファン、PQ特性

TH_SIMPLE   = 0           #熱回路網：単純熱回路
TH_AIRCON   = 1           #熱回路網：エアコン、熱量収支付け替え
TH_SOLAR    = 2           #熱回路網：日射取得
TH_GROUND   = 3           #熱回路網：地盤

get_rho = lambda sita:   353.25 / (sita + 273.15)       #空気の密度rhoを返す                                                                                                             #重力加速度 kg/s2
Rho20 = get_rho(20)                                     #空気の密度 kg/m3
Air_Cp = 1006      

n_trans = {None: FLAG_NONE, 'CALC': FLAG_CALC, 'FIX': FLAG_FIX, 'DLY': FLAG_DLY}
v_trans = {'simple': VN_SIMPLE, 'gap': VN_GAP, 'fix': VN_FIX, 'aircon': VN_AIRCON, 'fan': VN_FAN}
t_trans = {'simple': TH_SIMPLE, 'aircon': TH_AIRCON, 'solar': TH_SOLAR, 'ground': TH_GROUND}

node = lambda name, v_flag, c_flag, t_flag: {'name': name, 'v_flag': v_flag, 'c_flag': c_flag, 't_flag': t_flag}
net  = lambda name1, name2, tpe:{'name1': name1, 'name2': name2, 'type': tpe}
r_df = lambda fn:     pd.read_csv(fn, index_col = 0, parse_dates = True).fillna(method = 'bfill')
nc   = lambda id, v:  np.array([v] * len(id)) 
nd   = lambda df, cl: np.array(df[cl])
ix   = lambda length:  pd.date_range(datetime(2021, 1, 1, 0, 0, 0), datetime(2021, 1, 1, 0, 0, 0) + timedelta(seconds = length), freq='1s')

d_node  = lambda name: name + '_c'                                                                                      #遅延ノードの名前作成
cap     = lambda v, t_step: v * Rho20 * Air_Cp / t_step

def run_calc(ix, sn, **kwargs):

    sts    = kwargs['sts'] if 'sts' in kwargs else [SOLVE_LU, STEP_P, VENT_ERR, STEP_T, THRM_ERR, SOR_RATIO, SOR_ERR]
    vn  = kwargs['vn']     if 'vn'  in kwargs else []
    tn  = kwargs['tn']     if 'tn'  in kwargs else [] 
    opf = kwargs['output'] if 'output' in kwargs else 2

    t_step = (ix[1] - ix[0]).seconds + (ix[1] - ix[0]).microseconds / 1000000 

    length  = len(ix)

    node, length, nodes, v_nets, t_nets,\
    sn_P_set, sn_C_set, sn_T_set, sn_h_sr_set, sn_h_inp_set,\
    vn_v_set, vn_capa_set, vn_m_set, vn_beta_set,\
    vn_simple_set, vn_gap_set, vn_fix_set, vn_fan_set, vn_eta_set,\
    tn_simple_set, tn_solar_set, tn_ground_set = make_calc(length, t_step, sn, vn, tn)

    print('sts          : ', sts)

    print('nodes        : ', nodes)
    print('sn_P_set     : ', sn_P_set)
    print('sn_C_set     : ', sn_C_set)
    print('sn_T_set     : ', sn_T_set)
    print('sn_h_sr      : ', sn_h_sr_set)
    print('sn_h_inp     : ', sn_h_inp_set)

    print('v_nets       : ', v_nets)
    print('vn_simple_set: ', vn_simple_set)
    print('vn_gap_set   : ', vn_gap_set)
    print('vn_fix_set   : ', vn_fix_set)
    print('vn_fan_set   : ', vn_fan_set)
    print('vn_eta_set   : ', vn_eta_set)

    print('t_nets       : ', t_nets)
    print('tn_simple_set: ', tn_simple_set)
    print('tn_solar_set ; ', tn_solar_set)
    print('tn_ground_set: ', tn_ground_set)

    print('start vtsim calc')
    start = time.time()
    p, c, t, qv, qt1, qt2 = vt.calc(sts, length, t_step, nodes, v_nets, t_nets, 
                                    sn_P_set, sn_C_set, sn_T_set, sn_h_sr_set, sn_h_inp_set,
                                    vn_v_set, vn_capa_set, vn_m_set, vn_beta_set,
                                    vn_simple_set, vn_gap_set, vn_fix_set, vn_fan_set, vn_eta_set,
                                    tn_simple_set, tn_solar_set, tn_ground_set)
    print('finish vtsim calc')
    e_time = time.time() - start
    print("calc time:{0}".format(e_time  * 1000) + "[ms]")

    node_swap = {v: k for k, v in node.items()}

    n_columns   = [node_swap[i] for i in range(len(nodes))]
    v_columns  = [str(i) + " " + node_swap[v_nets[i][0]] + "->" + node_swap[v_nets[i][1]] for i in range(len(v_nets))]
    t_columns = [str(i) + " " + node_swap[t_nets[i][0]] + "->" + node_swap[t_nets[i][1]] for i in range(len(t_nets))]

    df_p, df_c, df_t, df_qv, df_qt1, df_qt2 = output_calc(opf, p, c, t, qv, qt1, qt2, ix, n_columns, v_columns, t_columns)

    return(df_p, df_c, df_t, df_qv, df_qt1, df_qt2)

def output_calc(opf, p, c, t, qv, qt1, qt2, ix, n_columns, v_columns, t_columns):
    
    if len(v_columns) == 0: v_columns = [0]
    if len(t_columns) == 0: t_columns = [0]

    if len(p)   == 0: p   = [[0.0] * len(ix)] * len(n_columns)
    if len(c)   == 0: c   = [[0.0] * len(ix)] * len(n_columns)
    if len(t)   == 0: t   = [[0.0] * len(ix)] * len(n_columns)
    if len(qv)  == 0: qv  = [[0.0] * len(ix)] * len(v_columns)
    if len(qt1) == 0: qt1 = [[0.0] * len(ix)] * len(v_columns)
    if len(qt2) == 0: qt2 = [[0.0] * len(ix)] * len(t_columns)

    df_p   = pd.DataFrame(np.array(p).T,     index = ix, columns = n_columns)
    df_c   = pd.DataFrame(np.array(c).T,     index = ix, columns = n_columns)
    df_t   = pd.DataFrame(np.array(t).T,     index = ix, columns = n_columns)
    df_qv  = pd.DataFrame(np.array(qv).T,    index = ix, columns = v_columns)
    df_qt1 = pd.DataFrame(np.array(qt1).T,   index = ix, columns = v_columns)
    df_qt2 = pd.DataFrame(np.array(qt2).T,   index = ix, columns = t_columns)

    if opf > 0:

        df_p.to_csv('vent_p.csv', encoding = 'utf_8_sig')
        df_c.to_csv('vent_c.csv', encoding = 'utf_8_sig')
        df_t.to_csv('thrm_t.csv', encoding = 'utf_8_sig')
        df_qv.to_csv('vent_qv.csv', encoding = 'utf_8_sig')
        df_qt1.to_csv('thrm_qt1.csv', encoding = 'utf_8_sig')
        df_qt2.to_csv('thrm_qt2.csv', encoding = 'utf_8_sig')

    if opf > 1:
        graphlist = [df_p, df_c, df_t, df_qv, df_qt1, df_qt2]
        fig = plt.figure(facecolor = 'w', figsize = (18, len(graphlist) * 4))
        fig.subplots_adjust(wspace = -0.1, hspace=0.7)

        for i, graph in enumerate(graphlist):
            a = fig.add_subplot(len(graphlist), 1, i + 1)
            for cl in graph.columns:
                a.plot(graph[cl], linewidth = 1.0, label = cl)
            a.legend(ncol = 5, bbox_to_anchor = (0, 1.05, 1, 0), loc = 'lower left', borderaxespad = 0, facecolor = 'w', edgecolor = 'k')

    return df_p, df_c, df_t, df_qv, df_qt1, df_qt2

def make_calc(length, t_step, sn, vn, tn):
    node = {}

    nodes, v_nets, t_nets                             = [], [], []
    sn_P_set, sn_C_set, sn_T_set                      = [], [], []
    sn_h_sr_set, sn_h_inp_set                         = [], []
    sn_v_set, sn_capa_set, sn_m_set, sn_beta_set      = [], [], [], []
    vn_simple_set, vn_gap_set, vn_fix_set, vn_fan_set = [], [], [], []
    vn_eta_set                                        = []
    tn_simple_set, tn_solar_set, tn_ground_set        = [], [], []

    for i, n in enumerate(sn):    
        node[n['name']] = i
        nodes.append([n_trans[n['v_flag']], n_trans[n['c_flag']], n_trans[n['t_flag']]])
        if 'p' in n:     sn_P_set.append([i, n['p']]) 
        if 'c' in n:     sn_C_set.append([i, n['c']]) 
        if 't' in n:     sn_T_set.append([i, n['t']])
        if 'h_sr' in n:  sn_h_sr_set.append([i, n['h_sr']])
        if 'h_inp' in n: sn_h_inp_set.append([i, n['h_inp']])
        if 'v' in n:     sn_v_set.append([i, n['v']])
        if 'm' in n:     sn_m_set.append([i, n['m']])
        if 'beta' in n:  sn_beta_set.append([i, n['beta']])

    for i, nt in enumerate(vn):
        h1 = nt['h1'] if 'h1' in nt else 0.0
        h2 = nt['h1'] if 'h1' in nt else 0.0
        v_nets.append([node[nt['name1']], node[nt['name2']], v_trans[nt['type']], h1, h2])
        if nt['type'] == 'simple':          vn_simple_set.append([i, nt['alpha'], nt['area']])
        if nt['type'] == 'gap':             vn_gap_set.append([i, nt['a'], nt['n']])
        if nt['type'] == 'fan':             vn_fan_set.append([i, nt['qmax'], nt['pmax'], nt['q1'], nt['p1']])
        if 'vol' in nt: vn_fix_set.append([i, nt['vol']])
        if 'eta' in nt: vn_eta_set.append([i, nt['eta']]) 
        else: vn_eta_set.append([i, [0.0] * length])

    for i, nt in enumerate(tn):
        t_nets.append([node[nt['name1']], node[nt['name2']], t_trans[nt['type']]])
        if nt['type'] == 'simple':          tn_simple_set.append([i, nt['cdtc']])
        if nt['type'] == 'solar':           tn_solar_set.append([i, nt['ms']])
        if nt['type'] == 'ground':          tn_ground_set.append([i, nt['area'], nt['rg'], nt['phi_0'], nt['cof_r'], nt['cof_phi']])
        
    for i, n in enumerate([n for n in sn if 'capa' in n]):
        node[d_node(n['name'])] = len(sn) + i
        nodes.append([FLAG_NONE, FLAG_NONE, FLAG_DLY])

        t_nets.append([node[n['name']], node[d_node(n['name'])], TH_SIMPLE])
        tn_simple_set.append([len(tn) + i, n['capa'] / t_step])

        sn_capa_set.append([node[d_node(n['name'])], node[n['name']], n['capa']])

    return node, length, nodes, v_nets, t_nets,\
           sn_P_set, sn_C_set, sn_T_set, sn_h_sr_set, sn_h_inp_set,\
           sn_v_set, sn_capa_set, sn_m_set, sn_beta_set,\
           vn_simple_set, vn_gap_set, vn_fix_set, vn_fan_set, vn_eta_set,\
           tn_simple_set, tn_solar_set, tn_ground_set
