"""Optional state history is packed once, independently of image row count."""
import numpy as np
import pyarrow as pa
import pytest

from openarm_policy_runtime import ObservationReader
from dora_openarm_local_policy_server.shm_ring import SharedMemoryRingWriter


def observation(include=True):
    qpos = np.arange(80, dtype=np.float32).reshape(5,16)
    state = np.arange(112, dtype=np.float32).reshape(7,16)
    state[-1] = qpos[-1]
    values = [pa.array(qpos.tolist(), type=pa.list_(pa.float32())),
              pa.array([[1]*18]*5, type=pa.list_(pa.uint8()))]
    names = ['position','camera_ceiling']
    metadata = {'timestamp':1000, 'history_delta_indices':'-24,-18,-12,-6,0',
                'camera_ceiling.height':2, 'camera_ceiling.width':3}
    if include:
        values.append(pa.array([None]*4+[state.tolist()], type=pa.list_(pa.list_(pa.float32()))))
        names.append('state_history')
        metadata.update(state_history_delta_indices='-6,-5,-4,-3,-2,-1,0', state_history_hz=30.,
                        state_history_timestamps='400,500,600,700,800,900,1000')
    return pa.StructArray.from_arrays(values,names=names),metadata,state


@pytest.mark.parametrize('borrow', [False, True])
def test_roundtrip_independent_history_and_no_repeated_state_payload(tmp_path, borrow):
    legacy, meta, _ = observation(False)
    baseline = SharedMemoryRingWriter(tmp_path)
    old_size = baseline.write(legacy, meta).bytes_written
    baseline.close()
    arrow, metadata, state = observation()
    writer, reader = SharedMemoryRingWriter(tmp_path), ObservationReader()
    try:
        written = writer.write(arrow, metadata)
        assert written.bytes_written - old_size == state.nbytes
        assert written.descriptor['fields']['state_history']['shape'] == [7,16]
        assert written.descriptor['fields']['position']['shape'] == [5,16]
        obs = reader.from_ring(written.descriptor, metadata, copy=not borrow)
        np.testing.assert_array_equal(obs.state_history, state)
        assert obs.state_history.flags.owndata
        assert obs.state_history_delta_indices == tuple(range(-6,1))
        assert obs.qpos.shape == (5,16)
    finally:
        reader.close()
        writer.close()


def test_writer_rejects_partial_auxiliary_contract(tmp_path):
    arrow, metadata, _ = observation()
    del metadata['state_history_timestamps']
    writer = SharedMemoryRingWriter(tmp_path)
    try:
        with pytest.raises(ValueError, match='requires offsets, rate, and timestamps'):
            writer.write(arrow,metadata)
        assert writer.mapping is None
    finally:
        writer.close()
