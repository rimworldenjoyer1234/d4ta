# d4ta

Modular research codebase for graph-construction experiments in intrusion-detection GNNs.

## Phase 1: dataset profiling

```bash
python scripts/profile_datasets.py \
  --unsw "C:\\Users\\yo\\Documents\\Datasets\\NF-UNSW-NB15-v2.csv" \
  --ton "C:\\Users\\yo\\Documents\\Datasets\\NF-ToN-IoT-v2.csv" \
  --kdd_train "C:\\Users\\yo\\Documents\\Datasets\\KDDTrain+.txt" \
  --kdd_test "C:\\Users\\yo\\Documents\\Datasets\\KDDTest+.txt" \
  --out_dir ./artifacts \
  --seed 123
```

Outputs:
- `artifacts/profiles/{dataset}.json`
- `artifacts/schemas/{dataset}.yaml`
