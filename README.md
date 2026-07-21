## Installation

Python ≥ 3.9 with:

```
pip install -r requirements.txt
```

## Data setup

Place the ballot CSVs at `data/dataverse_files/<Election>.csv` (same files as
the original repository; not shipped here), or pass `--data-root` to the
runner and test. Results are written under `<out-root>/_raw/<election>/`
(default `results/_raw/`).

## Running

```bash
# 1. Verify the corrections (fast without Mult; a few minutes with it)
python _test_corrected_methods.py --skip-mult
python _test_corrected_methods.py

# 2. See the corpus
python run_experiments.py --list

# 3. Run everything (resumable), or slices of it
python run_experiments.py --all --skip-existing
python run_experiments.py --election Alaska_20221108_HouseDistrict18
python run_experiments.py --all --k-max 4 --methods exact,fft,multinomial
python run_experiments.py --all --methods ks --ks-sigmafactor 1.0 \
    --ks-bucket-sizes 100,400,700,1000
```
