# Streamlit deployment

This branch adds a Streamlit interface alongside the existing FastAPI and Vite application. The original backend and frontend remain in place.

## Run locally

From the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Configure Gemini

Set `GEMINI_API_KEY` as an environment variable, or enter it in the app's sidebar.

For Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

Do not commit a real API key to GitHub.

## Deploy on Streamlit Community Cloud

1. Push or merge the `streamlit-deployment` branch.
2. In Streamlit Community Cloud, create a new app and connect this GitHub repository.
3. Select the `streamlit-deployment` branch.
4. Set the main file path to `streamlit_app.py`.
5. Add `GEMINI_API_KEY` in the app's Secrets settings.
6. Deploy.

The first startup downloads the Sentence Transformers embedding model. The app builds a FAISS index from the uploaded ZIP in a temporary session workspace. Uploaded code and index data are not committed to the repository.

## Notes and limits

- ZIP uploads are limited to 200 MB compressed and 500 MB extracted.
- The app rejects ZIP paths that attempt directory traversal.
- The FAISS index is session-local and is not a persistent multi-user database.
- Streamlit hosting resource limits may constrain large repositories because the embedding model and FAISS index use memory.
- The existing FastAPI backend remains independently runnable using the instructions in the main README.
