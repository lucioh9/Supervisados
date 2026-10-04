# LYDH — Discover what you can't see

A small lab where AI agents look for a hidden planet. They only see part of a solar system, notice that something
doesn't add up, guess what could be causing it, choose the measurement that can prove a guess wrong, and only accept
an answer if it still works on new data. This is a simulation: no real planet was discovered.

## Run it
```bash
pip install -r requirements.txt     # numpy scipy pandas matplotlib plotly streamlit
python bench.py 100                 # the test: 4 cases x 100 repeats x 3 methods (about 1 minute)
python make_backup_run.py           # saves a real run to replay in the demo
streamlit run app.py                # the app
omnigent run attn_lab/attn_lab.yaml -p "Run the discovery"   # the AI-agent team (needs Omnigent + a model key)
```

## The three methods we compare
- **Random**: measures on a random date.
- **Biggest error**: measures where the current picture is most wrong.
- **LYDH**: measures where our competing guesses disagree the most.

All three use the same math and the same stopping rule. Only the choice of the next measurement changes.

## Result (4 cases × 100 repeats × 3 methods = 1,200 runs)
LYDH needed 1.33× fewer measurements than random (range 1.25–1.41), and 1.67× fewer in the hardest case.

## Rules the agents follow
The AI agents never see the real answer. A conclusion is only accepted after a double-check on new measurements.
Every idea is labeled as written by an AI agent. The code does the math; the AI only suggests and explains.
