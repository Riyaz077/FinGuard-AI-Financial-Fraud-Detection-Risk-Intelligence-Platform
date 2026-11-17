# Deployment guide — GitHub and Render

## GitHub
Create a public repository named `FinGuard-AI` and upload the project files. Keep `requirements.txt`, `app.py`, `src/`, `setup.py`, `render.yaml`, and `README.md` at the repository root. Do not upload credentials or real financial/customer data.

## Render
1. Sign in at https://render.com/.
2. Choose **New +** → **Blueprint**, connect GitHub and select `FinGuard-AI`.
3. Render reads the included `render.yaml`.
4. Alternatively create a Web Service with build command `pip install -r requirements.txt` and start command `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`.
5. Deploy, open the Render URL and test every dashboard page.

The app generates synthetic transactions and trains its models on first startup if generated files are absent, so first deployment can take longer. Render free services may spin down when inactive; generated local files should not be treated as permanent storage. This is a portfolio demo using synthetic data, not a production banking or payment system.

## LinkedIn
Add two Featured items: `FinGuard AI — Live Demo` (Render URL) and `FinGuard AI — GitHub Repository` (GitHub URL).
