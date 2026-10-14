"""Verify retained tables and provenance without training or external services."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'evidence/provenance.json').read_text())
    for item in manifest:
        p=ROOT/item['publication_path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['publication_sha256'], p
    stage=pd.read_csv(ROOT/'evidence/solete/stage_c_summary_canonical.csv')
    assert len(stage)==30 and stage['config_id'].is_unique
    assert stage['status'].eq('success').all()
    for split in ['validation','test']:
        np.testing.assert_allclose(stage[f'{split}_nrmse'],stage[f'{split}_rmse']/2,atol=1e-12)
    winners={}
    for resolution,model in [('one_minute','HistGradientBoosting'),('one_hour','ExtraTrees')]:
        rows=stage.loc[(stage.resolution==resolution)&(stage.horizon==1)]
        row=rows.loc[rows.validation_rmse.idxmin()]
        assert row.model==model and row.lookback==8
        winners[resolution]={'model':model,'validation_rmse':float(row.validation_rmse),'test_rmse':float(row.test_rmse)}
    unsw=ROOT/'evidence/unsw'
    perday=pd.read_csv(unsw/'per_day_metrics.csv')
    perday=perday.loc[perday.model=='HistGradientBoosting']
    assert len(perday)==16
    means=pd.read_csv(unsw/'mean_std_across_days.csv')
    means=means.loc[means.model=='HistGradientBoosting']
    for _,row in means.iterrows():
        group=perday.loc[(perday.scenario==row.scenario)&(perday.arm==row.arm)]
        assert len(group)==4
        for metric in ['MAE','RMSE','NRMSE','persistence_skill','directional_accuracy']:
            np.testing.assert_allclose([group[metric].mean(),group[metric].std(ddof=1)],[row[metric+'_mean'],row[metric+'_std']],atol=1e-10)
    print(json.dumps({'status':'passed','provenance_files':len(manifest),'stage_c_configs':len(stage),'official_h1_winners':winners,'unsw_heldout_fits':len(perday)},indent=2))

if __name__=='__main__': main()
