"""Camera device selection/fallback logic — pure, no Qt enumeration."""

from identity_lab.camera.devices import CameraDeviceInfo, resolve_device


def dev(device_id, name="Cam", is_default=False, position=0):
    return CameraDeviceInfo(
        device_id=device_id, name=name, is_default=is_default, position=position
    )


FACETIME = dev("uid-facetime", "FaceTime HD Camera", is_default=True, position=0)
IPHONE = dev("uid-iphone", "iPhone Camera", is_default=False, position=1)


def test_empty_list_resolves_to_none():
    assert resolve_device("anything", []) == (None, "none")
    assert resolve_device(None, []) == (None, "none")


def test_saved_id_found():
    device, reason = resolve_device("uid-iphone", [FACETIME, IPHONE])
    assert device == IPHONE
    assert reason == "saved"


def test_missing_saved_id_falls_back_to_default_device():
    device, reason = resolve_device("uid-gone", [FACETIME, IPHONE])
    assert device == FACETIME
    assert reason == "default"


def test_no_saved_id_uses_default_device():
    device, reason = resolve_device(None, [IPHONE, FACETIME])
    assert device == FACETIME
    assert reason == "default"


def test_no_default_flag_falls_back_to_first():
    a, b = dev("a", position=0), dev("b", position=1)
    device, reason = resolve_device(None, [a, b])
    assert device == a
    assert reason == "first"


def test_position_is_capture_index():
    device, _ = resolve_device("uid-iphone", [FACETIME, IPHONE])
    assert device.position == 1
