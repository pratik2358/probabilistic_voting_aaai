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

## Third-party code

`irvprob.py` and `irvsimulator.py` are the reference implementation of the
KSIRV baseline, from "Ahead of the Count: An Algorithm for Probabilistic
Prediction of Instant Runoff (IRV) Elections" (Kapoor & Staecker,
arXiv:2405.09009), obtained from https://github.com/cstaecker/irvprob
(commit f140fe1) and included here for reproducibility of the KSIRV
experiments. Local modifications, kept minimal: (i) the internal
ranking-key delimiter was changed from "," to "||" so candidate names
containing commas are handled; (ii) the mpltern plotting dependency was
made optional; (iii) an int() cast fixes a float floor-division crash in
`ptfrombound`; (iv) a numerical-underflow fallback was added when the
±3σ-trimmed bucket list sums to ~0. All other code in this archive is by
the submission authors.
