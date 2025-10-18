# Deploy to Streamlit Cloud

## Files Required for Deployment ✅

All these files are now ready in your repo:

- ✅ `requirements.txt` - Python dependencies (with plotly!)
- ✅ `runtime.txt` - Python version (3.12)
- ✅ `.streamlit/config.toml` - App configuration
- ✅ `.gitignore` - Exclude large files from git
- ✅ `src/app/streamlit_app.py` - Your app
- ✅ `data/processed/` - Prediction data

## Step-by-Step Deployment

### 1. Commit and Push to GitHub

```bash
# Check git status
git status

# Add all files
git add requirements.txt runtime.txt .streamlit/ .gitignore
git add src/app/ data/processed/

# Commit
git commit -m "Add Streamlit deployment files with plotly dependency"

# Push to GitHub
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Configure:
   - **Repository**: Your GitHub repo (e.g., `aneesshaikh/ff_ppr`)
   - **Branch**: `main` (or `master`)
   - **Main file path**: `src/app/streamlit_app.py`
5. Click **"Deploy"**

### 3. Verify Python Version (Important!)

After deployment starts:

1. Click **"Manage app"** (⋮ menu)
2. Go to **Settings → Advanced settings**
3. Verify **Python version = 3.12** (should auto-detect from `runtime.txt`)
4. If it says 3.13, manually change to 3.12
5. Save and **reboot app**

### 4. Monitor Deployment

Watch the logs in real-time:
- Click **"Manage app" → Logs**
- You should see:
  ```
  Collecting plotly>=5.24.0
  Successfully installed plotly-5.24.0
  ```

### 5. Test on Mobile

Once deployed (usually 2-5 minutes):
- You'll get a URL like `https://your-app-name.streamlit.app`
- Open on your phone
- Test all features (season selection, week navigation, predictions)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'plotly'"

**Fix**: Make sure `plotly>=5.24.0` is in `requirements.txt` and redeploy

### Issue: Build takes forever or fails

**Fix**: The commented-out packages (torch, xgboost) are huge. They're already commented out in the optimized `requirements.txt`.

### Issue: "Data files not found"

**Fix**: Make sure `data/processed/*.parquet` files are committed to git:
```bash
git add data/processed/preds_week_*.parquet
git add data/processed/backtest_predictions.parquet
git commit -m "Add prediction data"
git push
```

### Issue: Python 3.13 incompatibility

**Fix**: Set Python version to 3.12 in Streamlit Cloud settings (see Step 3 above)

---

## File Size Limits

Streamlit Cloud has a **1GB total repository size limit**. Your prediction data is:

```bash
# Check size
du -sh data/processed/
```

If over 1GB, you'll need to either:
1. Use Git LFS for large files
2. Store data externally (S3, Google Drive) and download on app startup
3. Reduce data (e.g., only last 2 seasons)

Current setup (~20K predictions) should be **well under 100MB** ✅

---

## App Performance Tips

### Speed up deployment:
- Heavy packages are already commented out
- Keep `data/processed/` files committed
- Don't commit `data/interim/` (raw cached data)

### Speed up app loading:
The app already uses Streamlit's `@st.cache_data` decorator, so predictions load once and cache.

### Reduce memory:
If the app runs out of memory on Streamlit Cloud:
1. Reduce seasons (e.g., only 2024-2025)
2. Remove unused columns from prediction files
3. Use Parquet compression (already done)

---

## Update Predictions

To add new weeks as the season progresses:

```bash
# Locally: Run backtest with new data
python run_backtest_simple.py

# Prepare app data
python prepare_app_data.py

# Commit and push
git add data/processed/
git commit -m "Update predictions for week X"
git push

# Streamlit will auto-redeploy
```

---

## Estimated Build Time

- **First deployment**: 5-8 minutes (installing all packages)
- **Subsequent deployments**: 2-3 minutes (cached dependencies)

---

## Cost

Streamlit Cloud is **FREE** for public apps! 🎉

Limitations on free tier:
- 1 GB storage
- Sleep after 7 days of inactivity
- Public repository required

---

## Alternative: Streamlit Cloud Secrets

If you add API keys or secrets later, use Streamlit Secrets:

1. In Streamlit Cloud: **Manage app → Secrets**
2. Add secrets in TOML format:
   ```toml
   api_key = "your_key_here"
   ```
3. Access in code:
   ```python
   import streamlit as st
   api_key = st.secrets["api_key"]
   ```

---

## Success Checklist

Before going live:

- ✅ All files committed and pushed
- ✅ `plotly>=5.24.0` in requirements.txt
- ✅ Python 3.12 set in runtime.txt
- ✅ App deploys without errors
- ✅ Predictions load on mobile
- ✅ All seasons (2022-2025) accessible
- ✅ Player names display correctly
- ✅ Charts render (Plotly working)

---

## Your App URL

After deployment, share this URL:
```
https://[your-app-name].streamlit.app
```

Or use the custom subdomain if you upgrade to Streamlit Cloud Teams.

---

**Ready to deploy!** 🚀

Just run:
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push
```

Then follow steps 1-5 above.
