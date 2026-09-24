# V2.0 Admin Console Manual Draft

The Admin Console preserves the Streamlit full-function backend for managers and developers.

V2.0 current runtime status:

- React/Tauri user app can open the desktop HomePage;
- Tauri release executable build is verified;
- FastAPI `/api/health` is verified;
- backend single-fish HTTP loop is verified through upload, calibration, measurement, point update and export;
- browser-level SingleFishPage visual workflow and MeasurementCanvas drag validation are verified;
- Tauri installer/package generation is not enabled yet.

Admin functions include:

- specimen assignment
- image curation
- single image review
- batch measurement
- realworld review
- Advanced overlay
- developer tools
- model evaluation
- compressed-tail TL audit
- dataset export
- training dataset planning

Launch:

```powershell
streamlit run siganusmorph_v2/admin_streamlit/app.py
```

Legacy backend:

```powershell
streamlit run developer_tools/legacy_app.py
```

Review Queue:

```powershell
streamlit run pages/3_realworld_review.py
```

Admin outputs should use `results/v2_admin_outputs/` for future V2 admin workflows.

The Admin Console page also displays:

- availability checks for legacy backend and review pages;
- V2 backend command;
- V2 smoke test command;
- data safety boundaries for user outputs, admin outputs and historical V1.0 results.

Ordinary users should use the React/Tauri user app instead of this console.

Administrators should use this console for review, advanced overlay, model/debug inspection and dataset planning. It should not be exposed as the ordinary measurement entry point.
