# 06 Error Handling Protocol

1. Stop and read the full error. Reproduce it with the smallest possible command.
2. Search `docs/05_ERROR_PLAYBOOK.md` and `docs/08_ISSUES_LOG.md` for the symptom. If an entry exists, follow its fix exactly.
3. If there is no entry, diagnose methodically: form a hypothesis, test it, change one thing at a time. Do not make random edits.
4. Maximum 3 distinct fix attempts per problem. If all fail, choose the best workaround that keeps the spec intact, record it in the Blockers section of the progress file marked BLOCKED or WORKAROUND, and continue with work that does not depend on it. Never loop.
5. Never fix a failure by weakening a test, deleting a check, or hiding an exception. If a test is wrong, explain why in DECISIONS first.
6. After every resolved problem: add a row to `docs/08_ISSUES_LOG.md` (date, symptom, root cause, fix, files changed), and if the problem was not already in the playbook, append a new playbook entry.
7. After any fix, rerun the full test suite and the relevant quality gate before moving on.
