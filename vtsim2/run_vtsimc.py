###############################################################################
# import
###############################################################################

import numpy as np
import pandas as pd
import time

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas.core.indexing import convert_from_missing_indexer_tuple

import vtsimc as vtc
import vtsim2.archenvlib as lib

###############################################################################
# define const
###############################################################################

STEP_P         = 1e-6           #偏微分時の圧力変化
VENT_ERR       = 1e-6           #換気回路網の許容残差
STEP_T         = 1e-6           #偏微分時の温度変化
THRM_ERR       = 1e-6           #熱回路網の許容残差
CONV_ERR       = 1e-6           #収束の許容誤差
SOR_RATIO      = 0.9            #SOR法の緩和係数
SOR_ERR        = 1e-6           #SOR法の許容残差

SOLVE_LU:  int = 0              #LU分解法で計算  
SOLVE_SOR: int = 1              #SOR法で計算

SN_NONE:   int = 0              #計算しない
SN_CALC:   int = 1              #計算する
SN_FIX:    int = 2              #固定値（計算には利用するが、更新しない）
SN_DLY:    int = 3              #遅延（熱容量計算用）

VN_SIMPLE: int = 0              #換気回路網：単純開口
VN_GAP:    int = 1              #換気回路網：隙間
VN_FIX:    int = 2              #換気回路網：風量固定
VN_AIRCON: int = 3              #換気回路網：エアコン=風量固定、換気による熱移動=0
VN_FAN:    int = 4              #換気回路網：送風ファン、PQ特性

TN_SIMPLE: int = 0              #熱回路網：単純熱回路
TN_AIRCON: int = 1              #熱回路網：エアコン、熱量収支付け替え
TN_SOLAR:  int = 2              #熱回路網：日射取得
TN_GROUND: int = 3              #熱回路網：地盤
TN_HEATER: int = 4              #熱回路網：発熱

OPT_DF:    int = 0              #DataFrameを出力
OPT_CSV:   int = 1              #上記に加え、csvファイルを出力
OPT_GRAPH: int = 2              #上記に加えグラフを描画

###############################################################################
# define lambda
###############################################################################

node = lambda name, v_flag, c_flag, t_flag: {'name':   name, 
                                             'v_flag': v_flag, 'c_flag': c_flag, 't_flag': t_flag}      #ノードの設定
net  = lambda name1, name2, tp:             {'name1': name1, 'name2': name2, 'type': tp}                #ネットワークの設定
r_df = lambda fn:                           pd.read_csv(fn, index_col = 0, 
                                                        parse_dates = True).fillna(method = 'bfill')    #csvファイルの読み込み
nc   = lambda id, v:                        np.array([v] * len(id))                                     #idの長さ分の値value
nd   = lambda df, cl:                       np.array(df[cl])                                            #dfの列clを設定
ix   = lambda length:                       pd.date_range(datetime(2021, 1, 1, 0, 0, 0), 
                                                          datetime(2021, 1, 1, 0, 0, 0) + timedelta(seconds = length), 
                                                          freq='1s')                                    #長さlength、1s毎の時刻
d_node  = lambda name:                      name + '_c'                                                 #遅延ノードの名前作成

#リストかnp.ndarrayでなければlength分の長さのリストにする
to_list = lambda v, length:                 [float(v)] * length if type(v) != list and type(v) != np.ndarray else v  

###############################################################################
# define function
###############################################################################

def run_calc(ix, sn, **kwargs):                                                                            #はじめに呼び出される関数
    inp = vtc.InputData()

    inp.sts    = kwargs['sts']    if 'sts' in kwargs else [SOLVE_LU, STEP_P, VENT_ERR, 
                                                           STEP_T, THRM_ERR, CONV_ERR,
                                                           SOR_RATIO, SOR_ERR]                             #計算ステータスの読み込み
    inp.length = len(ix)
    inp.t_step = (ix[1] - ix[0]).seconds + (ix[1] - ix[0]).microseconds / 1000000                          #t_stepの読み込み

    vn         = kwargs['vn']     if 'vn'  in kwargs else []                                               #vnの読み込み
    tn         = kwargs['tn']     if 'tn'  in kwargs else []                                               #tnの読み込み
    opt        = kwargs['output'] if 'output' in kwargs else OPT_GRAPH                                     #出力フラグ                        

    node = {}
    nodes, v_nets, t_nets                                         = [], [], []
    sn_P_set, sn_C_set, sn_T_set, sn_h_sr_set, sn_h_inp_set       = [], [], [], [], []
    sn_v_set, sn_capa_set, sn_m_set, sn_beta_set                  = [], [], [], []
    vn_simple_set, vn_gap_set, vn_fix_set, vn_fan_set, vn_eta_set = [], [], [], [], []
    tn_simple_set, tn_solar_set, tn_h_inp_set, tn_ground_set      = [], [], [], []

    for i, n in enumerate(sn):                                                                              #sn
        node[n['name']] = i                                                                                 #ノード番号
        nodes.append([n['v_flag'], n['c_flag'], n['t_flag']])                                               #計算フラグ

        if 'p' in n:            sn_P_set.append([i, to_list(n['p'],     inp.length)])                       #圧力、行列で設定可能                                                 
        if 'c' in n:            sn_C_set.append([i, to_list(n['c'],     inp.length)])                       #濃度、行列で設定可能
        if 't' in n:            sn_T_set.append([i, to_list(n['t'],     inp.length)])                       #温度、行列で設定可能
        if 'h_sr' in n:      sn_h_sr_set.append([i, to_list(n['h_sr'],  inp.length)])                       #日射量、行列で設定可能
        if 'h_inp' in n:    sn_h_inp_set.append([i, to_list(n['h_inp'], inp.length)])                       #発熱、行列で設定可能
        if 'v' in n:            sn_v_set.append([i, to_list(n['v'],     inp.length)])                       #気積、行列で設定可能
        if 'm' in n:            sn_m_set.append([i, to_list(n['m'],     inp.length)])                       #発生量、行列で設定可能
        if 'beta' in n:      sn_beta_set.append([i, to_list(n['beta'],  inp.length)])                       #濃度減少率、行列で設定可能

    for i, nt in enumerate(vn):                                                                             #vn
        h1 = to_list(nt['h1'], inp.length) if 'h1' in nt else to_list(0.0, inp.length)                      #高さ1、行列設定不可
        h2 = to_list(nt['h2'], inp.length) if 'h2' in nt else to_list(0.0, inp.length)                      #高さ2、行列設定不可
        v_nets.append([node[nt['name1']], node[nt['name2']], nt['type'], h1, h2])                           #ネットワークタイプ＆高さ
        
        if nt['type'] == VN_SIMPLE:     vn_simple_set.append([i, to_list(nt['alpha'], inp.length), 
                                                                 to_list(nt['area'],  inp.length)])         #単純開口、行列で設定可能
        if nt['type'] == VN_GAP:           vn_gap_set.append([i, to_list(nt['a'],     inp.length), 
                                                                 to_list(nt['n'],     inp.length)])         #隙間、行列で設定可能
        if nt['type'] == VN_FAN:           vn_fan_set.append([i, to_list(nt['qmax'],  inp.length), 
                                                                 to_list(nt['pmax'],  inp.length), 
                                                                 to_list(nt['q1'],    inp.length),
                                                                 to_list(nt['p1'],    inp.length)])         #ファン、行列で設定可能

        if 'vol' in nt:                    vn_fix_set.append([i, to_list(nt['vol'],   inp.length)])         #風量固定値、行列で設定可能
        if 'eta' in nt:                    vn_eta_set.append([i, to_list(nt['eta'],   inp.length)])               
        else:                              vn_eta_set.append([i, to_list(0.0,         inp.length)])         #粉じん除去率、行列で設定可能

    for i, nt in enumerate(tn):                                                                             #tn
        t_nets.append([node[nt['name1']], node[nt['name2']], nt['type']])                                   #ネットワークタイプ
        if nt['type'] == TN_SIMPLE:     tn_simple_set.append([i, to_list(nt['cdtc'],  inp.length)])         #コンダクタンス、行列設定可能
        if nt['type'] == TN_SOLAR:       tn_solar_set.append([i, to_list(nt['ms'],    inp.length)])         #日射熱取得率、行列設定可能
        if nt['type'] == TN_HEATER:      tn_h_inp_set.append([i, to_list(nt['h_inp'], inp.length)])
        if nt['type'] == TN_GROUND:     tn_ground_set.append([i, to_list(nt['area'],  inp.length),           
                                                                 to_list(nt['rg'],    inp.length), 
                                                                 nt['phi_0'], nt['cof_r'], nt['cof_phi']])  #地盤熱応答、行列設定不可（面積と断熱性能はOK）
        
    for i, n in enumerate([n for n in sn if 'capa' in n]):                                                  #熱容量の設定のあるノード
        node[d_node(n['name'])] = len(sn) + i                                                               #時間遅れノードのノード番号
        nodes.append([SN_NONE, SN_NONE, SN_DLY])                                                            #計算フラグ
        sn_capa_set.append([node[d_node(n['name'])], node[n['name']]])                                      #熱容量の設定
        t_nets.append([node[n['name']], node[d_node(n['name'])], TN_SIMPLE])                                #ネットワークの設定
        tn_simple_set.append([len(tn) + i, to_list(n['capa'] / inp.t_step, inp.length)])                    #コンダクタンス（熱容量）

    inp.nodes, inp.v_nets, inp.t_nets                                                 = nodes, v_nets, t_nets
    inp.sn_P_set, inp.sn_C_set, inp.sn_T_set, inp.sn_h_sr_set, inp.sn_h_inp_set       = sn_P_set, sn_C_set, sn_T_set, sn_h_sr_set, sn_h_inp_set  
    inp.sn_v_set, inp.sn_capa_set, inp.sn_m_set, inp.sn_beta_set                      = sn_v_set, sn_capa_set, sn_m_set, sn_beta_set
    inp.vn_simple_set, inp.vn_gap_set, inp.vn_fix_set, inp.vn_fan_set, inp.vn_eta_set = vn_simple_set, vn_gap_set, vn_fix_set, vn_fan_set, vn_eta_set
    inp.tn_simple_set, inp.tn_solar_set, inp.tn_h_inp_set, inp.tn_ground_set          = tn_simple_set, tn_solar_set, tn_h_inp_set, tn_ground_set      

    print('sts          : ', inp.sts)

    print('nodes        : ', inp.nodes)
    print('sn_P_set     : ', inp.sn_P_set)
    print('sn_C_set     : ', inp.sn_C_set)
    print('sn_T_set     : ', inp.sn_T_set)
    print('sn_h_sr      : ', inp.sn_h_sr_set)
    print('sn_h_inp     : ', inp.sn_h_inp_set)

    print('v_nets       : ', inp.v_nets)
    print('vn_simple_set: ', inp.vn_simple_set)
    print('vn_gap_set   : ', inp.vn_gap_set)
    print('vn_fix_set   : ', inp.vn_fix_set)
    print('vn_fan_set   : ', inp.vn_fan_set)
    print('vn_eta_set   : ', inp.vn_eta_set)

    print('t_nets       : ', inp.t_nets)
    print('tn_simple_set: ', inp.tn_simple_set)
    print('tn_solar_set: ',  inp.tn_solar_set)
    print('tn_h_inp_set:',   inp.tn_h_inp_set)
    print('tn_ground_set: ', inp.tn_ground_set)

    print('start vtsim calc')
    s_time = time.time()
    p, c, t, qv, qt1, qt2 = vtc.calc(inp)
    e_time = time.time() - s_time
    print('finish vtsim calc')
    print("calc time:{0}".format(e_time * 1000) + "[ms]")

    node_swap = {v: k for k, v in node.items()}
    n_columns = [node_swap[i] for i in range(len(inp.nodes))]                                                                           #出力用カラムの作成（ノード）
    v_columns = [str(i) + " " + node_swap[inp.v_nets[i][0]] + "->" + node_swap[inp.v_nets[i][1]] for i in range(len(inp.v_nets))]       #出力用カラムの作成（換気回路網）
    t_columns = [str(i) + " " + node_swap[inp.t_nets[i][0]] + "->" + node_swap[inp.t_nets[i][1]] for i in range(len(inp.t_nets))]       #出力用カラムの作成（熱回路網）

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

    if opt > 0:
        df_p.to_csv('vent_p.csv',     encoding = 'utf_8_sig')
        df_c.to_csv('vent_c.csv',     encoding = 'utf_8_sig')
        df_t.to_csv('thrm_t.csv',     encoding = 'utf_8_sig')
        df_qv.to_csv('vent_qv.csv',   encoding = 'utf_8_sig')
        df_qt1.to_csv('thrm_qt1.csv', encoding = 'utf_8_sig')
        df_qt2.to_csv('thrm_qt2.csv', encoding = 'utf_8_sig')

    if opt > 1:
        graph_list  =  [df_p, df_c, df_t, df_qv, df_qt1, df_qt2]
        graph_title =  ['圧力', '濃度', '温度', '風量', '熱量1', '熱量2']
        graph_ylabel = ['[Pa]', '[個/L]', '[℃]', '[m3/s]', '[W]', '[W]']
        fig = plt.figure(facecolor = 'w', figsize = (18, len(graph_list) * 4))
        fig.subplots_adjust(wspace = -0.1, hspace=0.9)

        for i, graph in enumerate(graph_list):
            a = fig.add_subplot(len(graph_list), 1, i + 1)
            for cl in graph.columns:
                a.plot(graph[cl], linewidth = 1.0, label = cl)
            a.legend(ncol = 5, bbox_to_anchor = (0, 1.05, 1, 0), 
                     loc = 'lower right', borderaxespad = 0, facecolor = 'w', edgecolor = 'k')
            a.set_title(graph_title[i], loc='left')
            a.set_ylabel(graph_ylabel[i])

    return df_p, df_c, df_t, df_qv, df_qt1, df_qt2