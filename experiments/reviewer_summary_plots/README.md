# Reviewer summary plots for Tables I, II, and IV

Run from the repository root:

```powershell
python experiments/reviewer_summary_plots/plot_improvement_over_best_baseline.py
```

The script regenerates the three figures and the source-data CSV in `output/`.
Each figure is exported as editable SVG, PDF, 600-dpi PNG, and 600-dpi TIFF.

The plotted metric is

```text
improvement (%) = 100 * (best_baseline_cost - DVNDA_cost) / best_baseline_cost
```

Each single-axis forest plot uses one dark-blue point for the mean improvement
and one horizontal line for its approximate 95% confidence interval. Points to
the right of the dashed zero line indicate a lower DVNDA mean cost; points to
the left indicate a lower mean cost for the best baseline. An interval crossing
zero does not resolve which method is better under the summary-statistic
approximation.

The best baseline is the non-DVNDA method with the lowest reported mean cost in
the corresponding table cell; it is selected separately for every dynamic
rate, problem size, and distribution.

The manuscript tables provide means and standard deviations for 100 instances,
but not paired per-instance costs. The displayed 95% confidence intervals are
therefore approximate independent-sample delta-method intervals. If paired raw
costs become available, the final submission should replace these intervals
with paired bootstrap confidence intervals.

Recommended placement:

- `main_table_I_improvement.pdf`: single-column main-manuscript figure,
  immediately after Table I.
- `supp_table_II_improvement.pdf`: single-column Supplementary figure.
- `supp_table_IV_improvement.pdf`: single-column Supplementary figure.

Ready-to-paste reviewer response, captions, and concise interpretation are in
`reviewer_response_and_captions.tex`.
