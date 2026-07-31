"""Camera device discovery via Qt, mapped to OpenCV capture indices.

Qt (QMediaDevices) can enumerate cameras with real names and stable unique
ids without opening them. OpenCV's AVFoundation backend can only *open* a
camera by numeric index — it has no device-id API. Both enumerate the same
AVFoundation device list, so we map a device to its position in Qt's list
and use that position as the OpenCV index.

Verified on this Mac (single camera: Qt position 0 == OpenCV index 0, the
FaceTime HD Camera). With multiple cameras the identical-ordering assumption
is plausible but not guaranteed by either API — documented limitation; the
selector always shows which device name it believes it is opening.

Selection resolution is pure logic (testable without Qt): prefer the saved
device id, fall back to the system default device, then to the first device,
and report which fallback happened.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDeviceInfo:
    device_id: str
    name: str
    is_default: bool
    position: int  # index in enumeration order == OpenCV capture index


def snapshot_devices() -> list[CameraDeviceInfo]:
    """Enumerate video inputs via Qt (does not open any camera)."""
    from PySide6.QtMultimedia import QMediaDevices

    devices = []
    for pos, dev in enumerate(QMediaDevices.videoInputs()):
        devices.append(
            CameraDeviceInfo(
                device_id=bytes(dev.id()).decode("utf-8", errors="replace"),
                name=dev.description() or f"Camera {pos}",
                is_default=dev.isDefault(),
                position=pos,
            )
        )
    return devices


def resolve_device(
    saved_id: str | None, devices: list[CameraDeviceInfo]
) -> tuple[CameraDeviceInfo | None, str]:
    """Pick the device to use. Returns (device, reason).

    reason is one of: "saved" (saved id found), "default" (fell back to the
    system default), "first" (no default flagged; fell back to the first
    device), "none" (no cameras at all).
    """
    if not devices:
        return None, "none"
    if saved_id is not None:
        for dev in devices:
            if dev.device_id == saved_id:
                return dev, "saved"
    for dev in devices:
        if dev.is_default:
            return dev, "default"
    return devices[0], "first"
