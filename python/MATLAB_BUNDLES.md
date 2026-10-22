# Bundles from MATLAB MEA-NAP

In the Python GUI, select **Results → Make bundle…**, choose the MATLAB
`OutputData…` folder, and save the `.meanap` file. Open it with **Open bundle…**
or `meanap-viewer path/to/run.meanap`.

The folder must contain `Parameters_*.mat` and `ExperimentMatFiles/*.mat`
with saved step-4 electrophysiology results (`Info`, `NetMet`, `adjMs`, and
channel IDs). Classic MAT and v7.3/HDF5 files are supported. MATLAB suite2p
imports and spike-detection-only folders are not supported yet.

The importer copies saved results; it does not detect spikes or recalculate
network metrics. It converts MATLAB's one-based active-node indices to
zero-based indices for the viewer, and keeps the electrode order. Saved
spike times remain in seconds. Missing or inconsistent channel mappings cause
an error instead of guessing. Missing per-recording network results are
reported in the returned warnings.

The source folder is untouched. Conversion happens in a temporary folder.
The bundle carries CSVs and original figures, plus compatible network, activity,
and spike payloads where available; it omits the potentially large source MAT
files. Network figures can be drawn interactively from imported metrics. Some
MATLAB figure types have no interactive equivalent, so their originals remain
inside the bundle (which can also be opened as a ZIP). Bundles retaining images
can be substantially larger than Python express bundles.

The manifest labels MATLAB as the source, records the importer version, and
leaves the MATLAB code version unknown. `matlab_import.json` lists mapped source
settings and limitations. Unmapped viewer settings use display defaults; these
are not a complete recovery of the original analysis settings. This import is
intended for viewing/sharing, not as a complete MATLAB analysis backup.

Validation: `.venv/bin/python python/test_matlab_bundle.py` and
`.venv/bin/python python/test_bundle_from_folder.py`.
