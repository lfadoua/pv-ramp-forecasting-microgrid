"""Check the scientific indexing and leakage boundary on deterministic examples."""
import numpy as np
import pandas as pd
from pv_ramp_automation.config import load_config
from pv_ramp_automation.target_features import construct_target
from pv_ramp_automation.physics_features import engineer_features
from pv_ramp_automation.windowing import build_direct_supervised_windows
from pv_ramp_automation.splits import chronological_split_indices
from pv_ramp_automation.metrics import official_nrmse

def test_direct_future_increment_is_not_net_horizon_change():
    df=pd.DataFrame({'power':[0.,2.,8.,9.,5.,10.,3.,7.,8.,4.]})
    target,_=construct_target(df,'power','ramp',10.)
    x,y=build_direct_supervised_windows(target,['power'],'ramp',lookback=2,horizon=3)
    assert x[0,-1,0]==8.
    assert y[0]==(10.-5.)/10.
    assert y[0]!=(10.-8.)/10.

def test_windows_are_built_inside_chronological_splits():
    df=pd.DataFrame({'x':np.arange(100.),'ramp':np.arange(100.)})
    split=chronological_split_indices(len(df))
    for start,end in [(split.train_start,split.train_end),(split.validation_start,split.validation_end),(split.test_start,split.test_end)]:
        x,y=build_direct_supervised_windows(df.iloc[start:end],['x'],'ramp',4,3)
        assert len(y)==end-start-4-3+1
        assert x.min()>=start and y.max()<end
        np.testing.assert_allclose(y,x[:,-1,0]+3)
    assert official_nrmse(.04)==.02

def test_physics_features_do_not_change_when_future_changes():
    cfg=load_config()
    n=220
    t=np.arange(n,dtype=float)
    df=pd.DataFrame({'timestamp':pd.date_range('2020-01-01',periods=n,freq='min')})
    for col in ['P_Solar[kW]','POA Irr[kW1m2]','GHI[kW1m2]','Elevation[deg]','Azimuth[deg]','TEMPERATURE[degC]','HUMIDITY[%]','WIND_SPEED[m1s]','WIND_DIR[deg]']:
        df[col]=2.+.1*np.sin(t/15.)
    df['ramp_rate_pu']=np.r_[0.,np.diff(df['P_Solar[kW]'])/10.]
    first,_,_,_=engineer_features(df,'ramp_rate_pu',cfg,'selected_physics')
    altered=df.copy()
    numeric=[c for c in df.columns if c not in ['timestamp','ramp_rate_pu']]
    altered.loc[160:,numeric]=1000.
    second,_,_,_=engineer_features(altered,'ramp_rate_pu',cfg,'selected_physics')
    pd.testing.assert_frame_equal(first.loc[first.timestamp<df.timestamp.iloc[160]],second.loc[second.timestamp<df.timestamp.iloc[160]])
    assert first.shape[1]-2==83
