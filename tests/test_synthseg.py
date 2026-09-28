from oasis2.data.synthseg import synthseg_available, extract_fsl_seg_volumes


def test_synthseg_available_returns_bool():
    assert isinstance(synthseg_available(), bool)


def test_extract_fsl_seg_volumes_on_missing_dir_returns_empty(tmp_path):
    result = extract_fsl_seg_volumes(tmp_path / "does_not_exist")
    assert result == {}
