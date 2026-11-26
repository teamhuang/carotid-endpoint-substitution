"""Shared data preparation for the released cohorts. No analysis runs on import."""
from pathlib import Path
import os
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('CAROTID_DATA_DIR',ROOT/'data'))
FIG=DATA/'DataSheet_1_Development and validation of a nomogram for evaluating the incident risk of carotid atherosclerosis in patients with type 2 diabetes.csv'
XLS=DATA/'PROLOGUE_raw data for submission 20160506.xlsx'
if not XLS.is_file():
    XLS=DATA/'PROLOGUE_raw+data+for+submission+20160506.xlsx'
Chol2,TG2,GLU2=1/38.67,1/88.57,1/18.018
PRED=['age','male','BMI','SBP','smoke_ever','TC','HDL','TG','FPG','HbA1c']

def build_A():
    d = pd.read_csv(FIG, encoding='utf-8-sig')
    a = pd.DataFrame()
    a['age'] = d['age'].astype(float)
    a['male'] = (d['gender'] == 1).astype(int)
    a['BMI'] = d['BMI'].astype(float); a['SBP'] = d['SBP'].astype(float)
    a['smoke_ever'] = (d['smoke'] == 1).astype(int)
    a['TC'] = d['CHOL'].astype(float); a['HDL'] = d['HDL.C'].astype(float)
    a['TG'] = d['TG'].astype(float); a['FPG'] = d['OGTT0H'].astype(float)
    a['HbA1c'] = d['HbA1c'].astype(float)
    a['NAFLD'] = d['NAFLD'].astype(float); a['LDL'] = d['LDL.C'].astype(float)
    a['outcome_CAD'] = d['CAD'].astype(int)
    return a

def build_B():
    al = pd.read_excel(XLS, sheet_name='1. Allocation')
    assert al['ID'].is_unique, 'Allocation IDs must be unique'
    imt = pd.read_excel(XLS, sheet_name='2. IMT', na_values=['.'])
    oth = pd.read_excel(XLS, sheet_name='3. Other data', na_values=['.'])
    assert imt['ID'].is_unique and oth['ID'].is_unique
    assert pd.to_numeric(al['sex'], errors='coerce').isin([1,2]).all()
    m = al.merge(imt.drop(columns=[c for c in ['allocation', 'age', 'sex'] if c in imt.columns]), on='ID', how='left') \
           .merge(oth, on='ID', how='left')
    b = pd.DataFrame(index=m.index)
    b['age'] = pd.to_numeric(m['age'], errors='coerce')
    b['male'] = (pd.to_numeric(m['sex'], errors='coerce') == 1).astype(int)
    b['BMI'] = pd.to_numeric(m['body_weight_0M'], errors='coerce') / ((pd.to_numeric(m['height_background'], errors='coerce') / 100) ** 2)
    b['SBP'] = pd.to_numeric(m['sbp_0M'], errors='coerce').fillna(pd.to_numeric(m['al_SBP'], errors='coerce'))
    smk = pd.to_numeric(m['smoke_background'], errors='coerce')
    # Reproduce the original binary indicator and preserve unknown codes for sensitivity analysis.
    b['smoke_ever'] = smk.isin([1, 2]).astype(int)
    b['smoke_unknown'] = ~smk.isin([0, 1, 2])
    b['TC'] = pd.to_numeric(m['t_cholesterol_0M'], errors='coerce') * Chol2
    b['HDL'] = pd.to_numeric(m['hdl_cholesterol_0M'], errors='coerce') * Chol2
    b['TG'] = pd.to_numeric(m['tg_0M'], errors='coerce') * TG2
    b['FPG'] = pd.to_numeric(m['fbs_0M'], errors='coerce') * GLU2
    hba = pd.to_numeric(m['hba1c_ngsp_0M'], errors='coerce')
    b['HbA1c'] = hba.fillna(pd.to_numeric(m['al_HbA1c（JDS）'], errors='coerce') + 0.4)
    segmax = [c for c in imt.columns if c.startswith('0M_') and c.endswith('_max_IMT')]
    b['IMT_seg'] = pd.concat([pd.to_numeric(m[c], errors='coerce') for c in segmax], axis=1).max(axis=1)
    b['maxIMT'] = pd.to_numeric(m['al_max_IMT'], errors='coerce')
    b['statin'] = pd.to_numeric(m['Statin_0M'], errors='coerce')
    b['outcome_imt11'] = (b['maxIMT'] >= 1.1).astype(int)
    b['outcome_seg11'] = (b['IMT_seg'] >= 1.1).astype(float).where(b['IMT_seg'].notna())
    b['plaque'] = pd.to_numeric(m['0M_plaque_area'], errors='coerce')
    return b
