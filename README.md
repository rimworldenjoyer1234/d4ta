# d4ta

Modular research codebase for graph-construction experiments in intrusion-detection GNNs.

## 1) Profile datasets

```bash
python scripts/profile_datasets.py \
  --unsw "C:\\Users\\yo\\Documents\\Datasets\\NF-UNSW-NB15-v2.csv" \
  --ton "C:\\Users\\yo\\Documents\\Datasets\\NF-ToN-IoT-v2.csv" \
  --kdd_train "C:\\Users\\yo\\Documents\\Datasets\\KDDTrain+.txt" \
  --kdd_test "C:\\Users\\yo\\Documents\\Datasets\\KDDTest+.txt" \
  --out_dir .\\artifacts \
  --seed 123
```

## 2) Prepare train/val/test pools

```bash
python scripts/prepare_pools.py \
  --unsw "C:\\Users\\yo\\Documents\\Datasets\\NF-UNSW-NB15-v2.csv" \
  --ton "C:\\Users\\yo\\Documents\\Datasets\\NF-ToN-IoT-v2.csv" \
  --kdd_train "C:\\Users\\yo\\Documents\\Datasets\\KDDTrain+.txt" \
  --kdd_test "C:\\Users\\yo\\Documents\\Datasets\\KDDTest+.txt" \
  --out_dir .\\artifacts \
  --seed 123
```

## 3) Run RQ1

```bash
python scripts/run_rq1.py --out_dir .\\artifacts --seeds 123 456 --device cpu
```

## 4) Plot RQ1 outputs

```bash
python scripts/plot_rq1.py --results .\\artifacts\\results\\rq1_results.csv --out_dir .\\artifacts\\plots
```
