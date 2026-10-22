"""Import saved MATLAB ephys results for viewing; never run analysis."""
from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from meanap.params import Params, save_params


def read_mat(path: Path, names: tuple[str, ...]) -> dict:
    """Read selected variables from classic MAT or MATLAB v7.3/HDF5 files."""
    import h5py
    from scipy.io import loadmat
    if not h5py.is_hdf5(path):
        return loadmat(path, variable_names=names, simplify_cells=True)
    with h5py.File(path, 'r') as f:
        def read(obj):
            if isinstance(obj, h5py.Group):
                return {k: read(v) for k, v in obj.items() if not k.startswith('#')}
            if obj.attrs.get('MATLAB_empty', 0):
                return np.array([])
            a = obj[()]
            if h5py.check_dtype(ref=obj.dtype) is not None:
                values = [read(f[r]) if r else np.array([]) for r in a.T.flat]
                return values[0] if len(values) == 1 else values
            kind = obj.attrs.get('MATLAB_class', b'')
            if kind == b'char':
                return ''.join(chr(int(x)) for x in a.T.flat)
            a = np.asarray(a).T
            return a.item() if a.size == 1 else a
        return {k: read(f[k]) for k in names if k in f}


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def is_matlab_output(root: Path) -> bool:
    return any(root.glob('Parameters_*.mat')) and (root / 'ExperimentMatFiles').is_dir()


def bundle_matlab_output(root: Path, dest: Path, *, log=print, verify=True):
    """Stage compatible data and original figures without touching the source."""
    from meanap.pipeline.bundle import build_manifest, write_bundle
    from meanap.pipeline.io import save_spike_times_npz
    from meanap.pipeline.pack import BundleResult, _verify
    from meanap.pipeline.spreadsheet import RecordingInfo

    results, activity, recordings, lags = {}, {}, [], set()
    warnings = ['Imported saved MATLAB results; no spike detection or metrics were recomputed. '
                'Viewer settings not imported from MATLAB use display defaults. Original figures are retained; some MATLAB figure types have no interactive viewer equivalent.']
    with tempfile.TemporaryDirectory(prefix='meanap-matlab-') as tmp:
        stage = Path(tmp)
        # Preserve figures and tabular outputs, but do not distribute huge chained MAT files.
        for path in root.rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.csv'}:
                target = stage / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        params = Params()
        first_params = None
        for path in sorted((root / 'ExperimentMatFiles').glob('*.mat')):
            if path.name.startswith('._'):
                continue
            log(f'Importing MATLAB results: {path.name}')
            data = read_mat(path, ('Info', 'Params', 'NetMet', 'adjMs', 'Ephys', 'coords', 'channels', 'spikeTimes'))
            info = data.get('Info', {})
            net = data.get('NetMet', {})
            adj = data.get('adjMs', {})
            if not isinstance(info, dict) or not isinstance(net, dict) or not net:
                warnings.append(f'{path.name}: no saved network metrics; skipped.')
                continue
            saved = data.get('Params', {})
            if saved.get('suite2pMode', 0):
                raise ValueError('MATLAB bundle import currently supports electrophysiology, not suite2p/CAT-NAP.')
            name = str(info.get('FN', ''))
            if not name or Path(name).name != name or name in results:
                raise ValueError(f'Invalid or duplicate recording name in {path.name}: {name!r}')
            first_params = first_params or saved
            channels = np.asarray(data.get('channels', info.get('channels', []))).reshape(-1).astype(int)
            if not channels.size:
                raise ValueError(f'{name}: saved channel IDs are missing.')
            arrays = {'channels': channels}
            if 'coords' in data:
                arrays['coords'] = np.asarray(data['coords']).reshape(-1, 2)
            per_lag = {}
            for key, metrics in net.items():
                match = re.fullmatch(r'adjM(\d+)mslag', key)
                if not match or not isinstance(metrics, dict):
                    continue
                if key not in adj:
                    raise ValueError(f'{name}: missing saved adjacency {key}.')
                matrix = np.asarray(adj[key]).reshape(len(channels), -1)
                if matrix.shape != (len(channels), len(channels)):
                    raise ValueError(f'{name}: adjacency/channel dimensions disagree for {key}.')
                converted = dict(metrics)
                raw_indices = np.asarray(metrics.get('activeNodeIndices', [])).reshape(-1)
                if np.any(~np.isfinite(raw_indices)) or np.any(raw_indices != np.floor(raw_indices)):
                    raise ValueError(f'{name}: non-integer MATLAB activeNodeIndices for {key}.')
                indices = raw_indices.astype(int) - 1
                if 'activeNodeIndices' not in metrics or np.any(indices < 0) or np.any(indices >= len(channels)) or len(np.unique(indices)) != len(indices):
                    raise ValueError(f'{name}: missing or invalid MATLAB activeNodeIndices for {key}.')
                converted['activeChannelIndex'] = indices.tolist()
                converted['aN'] = len(indices)
                for metric in ('ND', 'NS', 'BC', 'PC', 'Z', 'Eloc', 'Ci'):
                    if metric in converted:
                        converted[metric] = np.asarray(converted[metric]).reshape(-1)
                per_lag[f'{match[1]}mslag'] = _plain(converted)
                arrays[key] = matrix
                lags.add(int(match[1]))
            if not per_lag:
                raise ValueError(f'{name}: no supported saved network lag metrics.')
            results[name] = per_lag
            recordings.append(RecordingInfo(filename=name, group=str(info.get('Grp', '')), div=float(info.get('DIV', 0))))
            out = stage / 'ExperimentMatFiles'
            out.mkdir(exist_ok=True)
            np.savez_compressed(out / f'{name}_adjM.npz', **arrays)
            if isinstance(data.get('Ephys'), dict):
                activity[name] = _plain(data['Ephys'])
            spikes = data.get('spikeTimes')
            if spikes is not None:
                entries = [spikes] if isinstance(spikes, dict) else (spikes if isinstance(spikes, list) else list(np.asarray(spikes, dtype=object).reshape(-1)))
                if len(entries) == len(channels):
                    trains = {i: {k: np.asarray(v).reshape(-1) for k, v in entry.items()}
                              if isinstance(entry, dict) else {str(saved.get('SpikesMethod', params.spikes_method)): np.asarray(entry).reshape(-1)}
                              for i, entry in enumerate(entries)}
                    save_spike_times_npz(stage / '1_SpikeDetection/1A_SpikeDetectedData' / f'{name}_spikes.npz',
                                         trains, channels, float(saved.get('fs', params.fs)), duration_s=info.get('duration_s'))
                else:
                    warnings.append(f'{name}: spike/channel counts differ; spike payload omitted.')
        if not results:
            raise ValueError('No saved MATLAB network results found. Select an output folder that completed step 4.')
        saved = first_params or {}
        # Only use explicitly mapped settings; retain originals separately as provenance.
        for source, target in {'fs': 'fs', 'SpikesMethod': 'spikes_method', 'channelLayout': 'channel_layout',
                               'minActivityLevel': 'min_activity_level'}.items():
            if source in saved:
                setattr(params, target, saved[source])
        params.func_con_lag_val = sorted(lags)
        save_params(params, stage)
        netdir = stage / '4_NetworkActivity'
        netdir.mkdir(exist_ok=True)
        (netdir / 'netmet_results.json').write_text(json.dumps(results))
        # Keep the MATLAB table when available (including its full metric columns).
        table = netdir / 'NetworkActivity_RecordingLevel.csv'
        if not table.exists():
            with table.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['FileName', 'Grp', 'DIV'])
                writer.writerows((r.filename, r.group, r.div) for r in recordings)
        activitydir = stage / '2_NeuronalActivity'
        activitydir.mkdir(exist_ok=True)
        (activitydir / 'ephys_results.json').write_text(json.dumps(activity))
        manifest = build_manifest(params, recordings, mode='ephys', lags=sorted(lags))
        manifest['importer'] = {k: manifest[k] for k in ('pipeline_name', 'version', 'versions')}
        manifest.update(source_pipeline='matlab', pipeline_name='MATLAB MEA-NAP (imported)',
                        version='unknown', versions={}, reconstructable=['4A_individual_network', '4B_group_comparisons', '2B_activity_comparisons'],
                        embedded_figures=['matlab_original_figures'], import_warnings=warnings)
        (stage / 'matlab_import.json').write_text(json.dumps({'source_pipeline': 'matlab', 'settings': {k: _plain(saved[k]) for k in ('fs', 'SpikesMethod', 'FuncConLagval', 'channelLayout') if k in saved}, 'warnings': warnings}, default=str))
        write_bundle(stage, manifest, dest)
        if verify:
            _verify(dest)
        log(f'MATLAB bundle written: {dest}')
        return BundleResult(dest=dest, recordings=len(recordings), size_bytes=dest.stat().st_size, warnings=warnings)
