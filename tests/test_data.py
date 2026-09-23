from openmoe.data.streams import make_synthetic_stream, split_class_ranges


def test_task_ranges_are_disjoint():
    specs = split_class_ranges(100, 5)
    assert len(specs) == 5
    flat = [c for spec in specs for c in spec.class_ids]
    assert len(flat) == len(set(flat)) == 100


def test_synthetic_stream_shapes():
    loaders = make_synthetic_stream(tasks=3, classes_per_task=2, samples_per_class=4, seed=0)
    x, y, task_id = next(iter(loaders[1]))
    assert x.shape[1:] == (3, 32, 32)
    assert y.ndim == 1
    assert int(task_id[0]) == 1
