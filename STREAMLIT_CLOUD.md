# Deploy On Streamlit Community Cloud

Use `rl_pendulum_activity` as the GitHub repo root.

## Files Streamlit Cloud Needs

- `pendulum_rl_live.py` as the app entrypoint
- `requirements.txt` for Python dependencies
- `runtime.txt` to request Python 3.11
- `drag_canvas/index.html` for the local drag/drop component
- `assets/demo_assets.pkl` so the intro slides load quickly

## Deploy Steps

1. Push this folder to a GitHub repo.
2. Open Streamlit Community Cloud.
3. Choose **New app**.
4. Select the GitHub repo.
5. Set the main file path to:

```text
pendulum_rl_live.py
```

6. Deploy and share the app URL.

## Notes For A Workshop

- Prefer Q-learning for live activities on Community Cloud.
- DQN works, but Community Cloud may not provide CUDA/GPU.
- The app caps DQN CPU parallelism for Network URL/shared-hosting sessions.
- The `submissions/` folder is ignored because generated GIF submissions should not be committed.
