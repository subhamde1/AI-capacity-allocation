##### Replication Files for *Who Gets Connected First?*



This repository contains the code, computational records, and supporting files used for the numerical and reproducibility work in the manuscript **“Who Gets Connected First? Incentive-Compatible Capacity Allocation for Flexible AI Data Centers.”**

The purpose of this repository is to keep the computational material used in the paper in one place so that the numerical results, robustness checks, trace-based analysis, and verification exercises can be inspected and reproduced without placing long program listings inside the manuscript itself.



##### Repository contents



The main files are grouped below according to their role in the paper.



###### Baseline numerical study

* `baseline\_code.py`  
Reproduces the original numerical design reported in the paper.
* `verify\_reproduction.py`  
Runs the main baseline reproduction checks and saves the corresponding numerical records.
* `check\_arithmetic.py`  
Contains the additional arithmetic checks used for selected analytical examples and counterexamples.
* `baseline\_results.json`  
Saved summary results from the baseline experiment.
* `baseline\_rows.npy`  
Instance-level baseline results used for the reported tables and figures.
* `arithmetic\_audit.json`  
Output from the arithmetic verification checks.



###### Larger computational experiments

* `extended\_experiments.py`  
Generates the larger finite-catalog experiments, structural checks, economic sensitivity calculations, and the time-limited scalability study.
* `economic\_results.json`  
Saved results from the economic-sensitivity calculations.
* `structural\_audit.json`  
Results from the structural verification exercises.



###### Paired computational-budget experiment

* `long\_budget.py`  
Runs the matched short-budget and tenfold-long-budget experiments.
* `paired\_tasks.jsonl`  
Task-level checkpoint file for the paired-budget calculations.
* `scale\_paired\_short.json`  
Results from the matched short-budget run.
* `scale\_paired\_long.json`  
Results from the matched long-budget run.
* `budget\_summary.json`  
Summary of the paired-budget comparison.
* `paired\_environment.json`  
Records the software environment and computational settings used for the paired experiment.
* `independent\_audit.py`  
Performs a separate implementation check on the stored optimization and payment records.
* `independent\_audit.json`  
Saved output from that audit.



###### Public workload trace analysis

* `trace\_robustness.py`  
Extracts the selected Philly workload records, constructs the operating profiles, and runs the chronological holdout analysis.
* `trace\_inputs.json`  
Frozen transformed inputs used in the trace exercise.
* `trace\_results.json`  
Complete machine-readable results from the trace experiment.
* `trace\_summary.json`  
Summary statistics reported in the manuscript.
* `philly\_provenance.json`  
Provenance information for the public workload trace.
* `Philly\_LICENSE`  
License information for the public Philly trace.
* `Philly\_README.md`  
Original source documentation retained with the trace materials.



###### Institutional sensitivity analysis

* `institutional\_sensitivity.py`  
Runs the main institutional sensitivity experiment and the recovery-only diagnostic.
* `audit\_institutional.py`  
Independently checks selected institutional-sensitivity calculations.
* `bundle\_physical\_audit.py`  
Verifies the broad-versus-narrow recovery-bundle construction using explicit physical load matrices.
* `summarize\_institutional.py`  
Produces the parameter-cell summaries used in the appendix tables and figures.
* `broad\_value\_grid.json`  
Records the parameter grid used for the bundle diagnostic.
* `strengthening\_protocol.json`  
Contains the frozen protocol information used for the institutional robustness exercise.
* `reference\_crosswalk.json`  
Records how the public institutional sources are mapped to the inputs used in the sensitivity analysis.



###### Additional records

The repository also contains several saved audit and environment files, including:

* `final\_document\_audit.json`
* `layout\_metrics.json`
* `original\_environment.json`

These files are retained as part of the computational record supporting the manuscript.



###### Software requirements

The reported computations were run with:

* Python 3.12
* NumPy 2.3.5
* SciPy 1.17.0
* HiGHS 1.8.0 through SciPy
* Linux

A minimal environment can be created with:

```bash
python -m venv .venv
```

After activating the environment, install the required numerical packages:

```bash
pip install numpy==2.3.5 scipy==1.17.0
```

###### Main reproduction commands

For the baseline numerical study and the principal verification checks:

```bash
python verify\_reproduction.py
```

The original baseline program can also be run directly:

```bash
python baseline\_code.py
```

For the larger time-limited experiment:

```bash
python extended\_experiments.py scale
```

For the paired short- and long-budget experiment:

```bash
python long\_budget.py
```

After that run, the separate implementation audit can be executed with:

```bash
python independent\_audit.py
```

###### Trace exercise

The workload-trace exercise uses the public Microsoft Research Philly job log. After placing `cluster\_job\_log.json` in the working directory, run:

```bash
python trace\_robustness.py extract cluster\_job\_log.json
```

and then:

```bash
python trace\_robustness.py analyze
```

The transformed inputs and the resulting output files are retained in the repository so that the exact records used in the paper can be inspected directly.

Institutional sensitivity analysis

Run the main sensitivity experiment with:

```bash
python institutional\_sensitivity.py
```

Run the recovery-only diagnostic with:

```bash
python institutional\_sensitivity.py --recovery
```

The associated verification programs are:

```bash
python audit\_institutional.py
python audit\_institutional.py --recovery
python bundle\_physical\_audit.py
python summarize\_institutional.py
```

###### How this repository relates to the manuscript



This repository is the computational companion to the paper.

The manuscript contains the model, proofs, derivations, interpretation, reported tables, figures, and methodological discussion. The repository contains the executable code and machine-readable records needed to reproduce and check the computational parts of that analysis.

In particular:

* Appendix C refers to `baseline\_code.py`.
* Appendix F refers to `extended\_experiments.py`, `check\_arithmetic.py`, and `verify\_reproduction.py`.
* Appendix G refers to `long\_budget.py` and `independent\_audit.py`.
* Appendix H refers to `trace\_robustness.py` and the trace-related input and output records.
* Appendix I refers to `institutional\_sensitivity.py`, `audit\_institutional.py`, `bundle\_physical\_audit.py`, and `summarize\_institutional.py`.



###### Reproducibility notes



Most of the constructed numerical experiments are deterministic once the stated seeds and parameters are fixed.

The time-limited mixed-integer optimization experiments can show small differences in run time, incumbent solutions, or remaining optimization bounds across machines because they depend on hardware, process scheduling, and solver behavior. For those experiments, the important reproduction checks are feasibility, objective accounting, the reported bound interpretation, and consistency with the stated computational protocol rather than identical wall-clock times.

The public Philly trace is used only for workload timing and GPU-reservation information. The financial preferences, deferral permissions, and shared resource-capacity assumptions used in the corresponding experiment are modeled inputs rather than observations from the trace.

The computational audits included here are intended to check the internal consistency of the reported calculations. They are not external certification of the optimization software or empirical validation of the private preference assumptions used in the model.

