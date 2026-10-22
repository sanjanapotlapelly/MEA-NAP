"""MATLAB import integration checks (classic MAT and v7.3)."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import h5py
import numpy as np
from scipy.io import savemat
from meanap.pipeline.pack import bundle_output_folder, unbundlable_reason
from meanap.pipeline.bundle import open_bundle
from meanap.pipeline.render import load_context, available_figures, render_figure
from meanap.pipeline.matlab_bundle import read_mat

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'MatlabRun'
    exp = root / 'ExperimentMatFiles'
    exp.mkdir(parents=True)
    savemat(root / 'Parameters_MatlabRun.mat', {'Params': {'fs': 25000}})
    matrix = np.array([[0., .2, .3], [.2, 0., .4], [.3, .4, 0.]])
    metrics = {'activeNodeIndices': [1, 3], 'ND': [1, 1], 'NS': [.3, .3], 'Dens': 1., 'MEW': .3}
    data = {'Info': {'FN': 'recording', 'Grp': 'WT', 'DIV': 14., 'duration_s': 10.},
            'Params': {'fs': 25000., 'SpikesMethod': 'bior1p5'},
            'NetMet': {'adjM25mslag': metrics}, 'adjMs': {'adjM25mslag': matrix},
            'channels': [11, 12, 13], 'coords': [[0, 0], [1, 0], [2, 0]],
            'spikeTimes': np.array([{'bior1p5': [.1, .3]}, {'bior1p5': [.2]}, {'bior1p5': []}], dtype=object)}
    path = exp / 'recording_MatlabRun.mat'
    savemat(path, data)
    figure = root / '4_NetworkActivity/4A_IndividualNetworkAnalysis/WT/recording/original.png'
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b'original MATLAB figure')
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert unbundlable_reason(root) is None
    result = bundle_output_folder(root, log=lambda _: None)
    with open_bundle(result.dest) as bundle:
        assert bundle.manifest['source_pipeline'] == 'matlab'
        ctx = load_context(bundle)
        met = ctx.results['recording']['25mslag']
        np.testing.assert_array_equal(met['activeChannelIndex'], [0, 2])
        np.testing.assert_array_equal(met['NS'], [.3, .3])
        np.testing.assert_array_equal(met['adjMsub'], matrix[np.ix_([0, 2], [0, 2])])
        assert (bundle.root / figure.relative_to(root)).read_bytes() == b'original MATLAB figure'
        with np.load(bundle.root / '1_SpikeDetection/1A_SpikeDetectedData/recording_spikes.npz') as spikes:
            np.testing.assert_array_equal(spikes['spike_times_0_bior1p5'], [.1, .3])
        assert available_figures(ctx, 'recording', 25)
        rendered = render_figure(ctx, 'recording', 25, '2_MEA_NetworkPlot', Path(tmp) / 'rendered')
        assert rendered.is_file() and rendered.stat().st_size > 1000
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    # Write the same numeric/struct layout with MATLAB's HDF5 dimension order.
    def put(group, key, value):
        if isinstance(value, dict):
            child = group.create_group(key)
            for k, v in value.items(): put(child, k, v)
        elif isinstance(value, str):
            ds = group.create_dataset(key, data=np.array([ord(c) for c in value], dtype='uint16').reshape(-1, 1))
            ds.attrs['MATLAB_class'] = np.bytes_('char')
        else:
            group.create_dataset(key, data=np.atleast_2d(value).T)
    with h5py.File(path, 'w') as f:
        for k, v in data.items():
            if k != 'spikeTimes': put(f, k, v)
    imported = read_mat(path, ('Info', 'adjMs'))
    assert imported['Info']['FN'] == 'recording'
    np.testing.assert_array_equal(imported['adjMs']['adjM25mslag'], matrix)
    result = bundle_output_folder(root, log=lambda _: None)
    with open_bundle(result.dest) as bundle:
        assert load_context(bundle).recordings['recording'].group == 'WT'
    # Missing indices must fail instead of guessing which channels were analysed.
    del data['NetMet']['adjM25mslag']['activeNodeIndices']
    savemat(path, data)
    try:
        bundle_output_folder(root, log=lambda _: None)
    except ValueError as exc:
        assert 'activeNodeIndices' in str(exc)
    else:
        raise AssertionError('Invalid channel mapping accepted')
print('MATLAB import checks passed.')
