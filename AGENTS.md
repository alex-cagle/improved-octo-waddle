Read AGENTS.md and inspect the repository.

Do analysis only. Do not edit code yet.

Task:
Identify all direct and indirect dependencies on `BP._e_index` in this repository.

Repository-specific targets:
- `bp/_bp.pyx`
- `bp/_bp.pxd`
- `bp/tests/test_bp.py`
- `bp/tests/test_bp_cy.pyx`

What I need:
1. Every direct read of `_e_index`
2. Every method that depends on `_e_index` indirectly through `excess()`
3. Which of those methods are:
   - easy to refactor first
   - risky to refactor early
4. The smallest safe first step that reduces coupling to `_e_index` without changing behavior
5. The exact files that would need to change for that first step

Be specific:
- name exact methods
- name exact files
- distinguish direct vs indirect dependency
- distinguish paper-faithful refactor targets vs temporary transition steps

Do not propose a rewrite.
Do not change architecture.
Do not write code yet.