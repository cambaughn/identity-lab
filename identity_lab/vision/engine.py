"""InsightFace wrapper — model loading, warm-up, and per-frame analysis.

Loads only the modules we use: detection, fine landmarks, and recognition
(the gender/age model in the buffalo pack is deliberately excluded). The
warm-up pass exercises every loaded ONNX session once so the multi-second
lazy first-inference cost (measured at CP3) happens during MODEL LOADING
instead of on the first real face.
"""

import numpy as np

MODEL_PACK = "buffalo_l"
DET_SIZE = (640, 640)
ALLOWED_MODULES = ["detection", "landmark_2d_106", "recognition"]


class VisionEngine:
    """Owns FaceAnalysis. Construct + initialize() on the inference thread."""

    def __init__(self) -> None:
        self._app = None
        self.model_id = MODEL_PACK

    def initialize(self) -> None:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=MODEL_PACK,
            allowed_modules=ALLOWED_MODULES,
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=DET_SIZE)
        self._app = app
        self._warm_up()

    def _warm_up(self) -> None:
        """Run every model's full per-face path once.

        ONNX Runtime sessions and the alignment/cropping Python paths both
        initialize lazily on first use (measured: multi-hundred-ms first-call
        cost). A blank frame warms the detector; a fabricated Face drives the
        landmark and recognition paths since a blank frame has no detections.
        """
        from insightface.app.common import Face

        blank = np.zeros((DET_SIZE[1], DET_SIZE[0], 3), dtype=np.uint8)
        self._app.get(blank)
        face = Face(
            bbox=np.array([200.0, 150.0, 440.0, 470.0]),
            kps=np.array(
                [
                    [260.0, 270.0], [380.0, 270.0], [320.0, 340.0],
                    [270.0, 400.0], [370.0, 400.0],
                ]
            ),
            det_score=np.float32(0.9),
        )
        for name, model in self._app.models.items():
            if name == "detection":
                continue
            try:
                model.get(blank, face)
            except Exception:
                # Warm-up is best-effort; worst case is one slower first frame.
                pass

    def analyze(self, frame_bgr: np.ndarray, with_embeddings: bool = False):
        """Detect faces and return DetectedFace tuples (frame coordinates).

        Mirrors FaceAnalysis.get() from the installed insightface, except the
        recognition (embedding) model only runs when with_embeddings is True —
        embeddings are unused before recognition (CP7+) and cost ~60ms/face.
        """
        from insightface.app.common import Face

        from identity_lab.vision.types import DetectedFace

        if self._app is None:
            raise RuntimeError("VisionEngine.initialize() was not called")
        bboxes, kpss = self._app.det_model.detect(
            frame_bgr, max_num=0, metric="default"
        )
        out = []
        for i in range(bboxes.shape[0]):
            face = Face(
                bbox=bboxes[i, 0:4],
                kps=None if kpss is None else kpss[i],
                det_score=bboxes[i, 4],
            )
            for taskname, model in self._app.models.items():
                if taskname == "detection":
                    continue
                if taskname == "recognition" and not with_embeddings:
                    continue
                model.get(frame_bgr, face)

            x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
            landmarks = getattr(face, "landmark_2d_106", None)
            embedding = (
                getattr(face, "normed_embedding", None) if with_embeddings else None
            )
            out.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    det_score=float(face.det_score),
                    landmarks=None if landmarks is None else np.asarray(landmarks),
                    kps=None if face.kps is None else np.asarray(face.kps),
                    embedding=None if embedding is None else np.asarray(embedding),
                )
            )
        return tuple(out)
