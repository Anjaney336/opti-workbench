# Undergraduate Research Journal Submission Plan

## Best-Fit Venue

**Recommended first target: Journal of Young Investigators (JYI)**  
Reason: JYI is explicitly an undergraduate research journal and its site includes research submission and manuscript guideline pages. The current project is scientific/computational, which fits better here than a humanities or general campus-only outlet.

Useful pages:

- JYI home: https://www.jyi.org/
- Submit research: https://www.jyi.org/submit-research
- Manuscript guidelines: https://www.jyi.org/manuscript-guidelines

## Alternative Venues To Check

- American Journal of Undergraduate Research
- Midwest Journal of Undergraduate Research
- Butler Journal of Undergraduate Research
- Your university's undergraduate research journal or engineering journal
- A student research conference in computer science, data science, finance, operations research, or engineering

## What Must Be Completed Before Submission

1. Replace placeholders in `paper/research_paper_draft.md`:
   - Author name
   - Institution
   - Faculty mentor
   - Contact email

2. Confirm authorship:
   - Only list people who made real intellectual or implementation contributions.
   - Add helpers who do not qualify as authors to acknowledgments.

3. Add a short ethics and AI-use disclosure if the target journal requires it:
   - Example: "The authors used AI-assisted editing tools to help organize and revise manuscript language. All technical claims, code, experiments, and final content were reviewed and approved by the author."

4. Strengthen experiments before final submission:
   - Run at least 10 random seeds.
   - Add an equal-weight baseline.
   - Add a random feasible portfolio baseline.
   - Report statistical means and standard deviations.
   - Add one real-data experiment if possible.

5. Prepare figures:
   - Use high-resolution PNG/PDF exports.
   - Number captions as Figure 1, Figure 2, etc.
   - Make sure every figure is discussed in the text.
   - Newly generated manuscript figures are in `paper/figures/`.
   - Existing project figures are referenced from `results/`.

6. Format references:
   - Convert references to the target journal's style.
   - Verify every citation manually.

7. Prepare submission files:
   - Manuscript document, usually DOCX or PDF.
   - Figure files if uploaded separately.
   - Cover letter.
   - Any required faculty approval form.
   - Any required supplementary code/data link.

## Current Figure Package

The manuscript now includes ten graph references:

- `paper/figures/objective_runtime_tradeoff.png`
- `paper/figures/monte_carlo_risk_return.png`
- `paper/figures/robustness_objective_distribution.png`
- `paper/figures/aggregate_method_profile.png`
- `results/allocations.png`
- `results/comparison.png`
- `results/convergence.png`
- `results/risk_return.png`
- `results/sensitivity.png`
- `results/stress_summary.png`

If the target journal requires separate image uploads, upload these files individually and keep the captions in the manuscript.

## Cover Letter Draft

Dear Editors,

I am submitting the manuscript "Hybrid Portfolio Optimization Using QP-Seeded Swarm and Genetic Search for Real Estate Asset Allocation" for consideration as an undergraduate research article.

This work presents a Python-based computational research project that combines Quadratic Programming, Particle Swarm Optimization, and Genetic Algorithm refinement for constrained portfolio allocation. The system includes synthetic asset generation, feasibility projection, Monte Carlo evaluation, convergence analysis, stress/sensitivity testing, and an interactive dashboard. The manuscript compares QP, PSO, GA, and hybrid optimization methods and discusses the trade-off between convex solver speed and hybrid search flexibility.

The manuscript is original, has not been published elsewhere, and is not under consideration by another journal. All listed authors have reviewed and approved the submission. The project code and result artifacts can be made available as supplementary material.

Thank you for your consideration.

Sincerely,

[Your Name]  
[Institution]  
[Email]

## Practical Submission Status

I cannot complete the final journal upload from here because submission requires your personal account, author identity, affiliation, email, authorship confirmation, and agreement to journal policies. The prepared draft and cover letter are ready for you to adapt to the selected journal's portal.
