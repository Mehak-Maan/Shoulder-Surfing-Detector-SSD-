"""Command line entry point of the Shoulder Surfing Detector."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from benchmark import format_benchmark, run_benchmark
from capture import CameraSource, FramePacket, FrameSource, VideoSource
from config import AppConfig, ConfigError, load_config
from detector import IdentityVerifier, MediaPipeBackend, ModelNotFoundError
from evaluation import evaluate, format_report, load_windows, replay
from gui import run_gui
from headless import HeadlessRunner
from overlay import BlurOverlay, MssGrabber, NullOverlay, TkOverlayWindow
from pipeline import Pipeline, PipelineResult
from runtime import SingleInstance, install_signal_handlers, setup_logging
from security_actions import (
    ActionExecutor,
    DryRunPlatform,
    EvidenceSink,
    Overlay,
    ResponseHandler,
    SystemPlatform,
)
from tray import create_tray

logger = logging.getLogger("ssd")

BASE_DIR = Path(__file__).resolve().parent
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    p = argparse.ArgumentParser(description="Real-time shoulder surfing detector (offline, CPU only).")
    p.add_argument("--mode", choices=["gui", "headless"], default="gui", help="gui = debug window, headless = service")
    p.add_argument("--source", choices=["cam", "video"], default="cam", help="live camera or recorded video")
    p.add_argument("--video", type=Path, help="video file for --source video")
    p.add_argument("--camera-index", type=int, help="override camera.index from the config")
    p.add_argument("--config", type=Path, default=BASE_DIR / "config.yaml", help="YAML configuration file")
    p.add_argument("--benchmark", type=float, metavar="SECONDS", help="measure FPS/latency/CPU, then exit")
    p.add_argument("--bench-fps", type=float, metavar="FPS", help="pace the benchmark (e.g. 8 for headless load)")
    p.add_argument("--report", action="store_true", help="replay a video without a window and print metrics")
    p.add_argument("--labels", type=Path, help="JSON with intruder_windows for --report")
    p.add_argument("--realtime", action="store_true", help="play a video at its native speed in the GUI")
    p.add_argument(
        "--actions",
        choices=["off", "dry-run", "on"],
        help="off = detect only, dry-run = log actions, on = really notify/blur/lock "
        "(default: dry-run in gui mode, on in headless mode)",
    )
    p.add_argument("--no-tray", action="store_true", help="headless: do not create a tray icon")
    p.add_argument("--decrypt-evidence", type=Path, metavar="FILE", help="decrypt an evidence snapshot, then exit")
    p.add_argument("--out", type=Path, help="output path for --decrypt-evidence")
    p.add_argument("--enroll", action="store_true", help="record the owner's face embeddings (no photos), then exit")
    p.add_argument("--forget-owner", action="store_true", help="delete the enrolled owner profile, then exit")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _emit(text: str) -> None:
    sys.stdout.write(text + "\n")


def _build_source(args: argparse.Namespace, cfg: AppConfig) -> FrameSource:
    if args.source == "video":
        return VideoSource(args.video, realtime=args.realtime)
    return CameraSource(cfg.camera)


def _build_executor(mode: str, cfg: AppConfig) -> tuple[ActionExecutor | None, Overlay | None]:
    """Create the action executor for ``--actions`` (None when actions are off)."""
    if mode == "off":
        return None, None
    if mode == "dry-run":
        overlay: Overlay = NullOverlay()
        return ActionExecutor(cfg.actions, DryRunPlatform(), overlay), overlay
    real: Overlay = BlurOverlay(cfg.overlay, MssGrabber(), TkOverlayWindow())
    return ActionExecutor(cfg.actions, SystemPlatform(), real), real


def _decrypt(cfg: AppConfig, args: argparse.Namespace) -> int:
    from evidence import decrypt_file

    if args.out is None:
        logger.error("--decrypt-evidence needs --out PATH")
        return 2
    try:
        args.out.write_bytes(decrypt_file(Path(cfg.privacy.key_file), args.decrypt_evidence))
    except Exception as exc:  # noqa: BLE001 - wrong key, missing file, corrupt data: report them all the same way
        logger.error("Could not decrypt: %s", exc)
        return 2
    logger.info("Decrypted to %s", args.out)
    return 0


def _forget_owner(cfg: AppConfig) -> int:
    path = Path(cfg.identity.profile_file).expanduser()
    if not path.exists():
        logger.info("No owner profile at %s", path)
        return 0
    path.unlink()
    logger.info("Deleted owner profile %s", path)
    return 0


def _enroll(args: argparse.Namespace, cfg: AppConfig) -> int:
    from enrollment import run_enrollment
    from identity import ProfileError, create_embedder, save_profile

    try:
        embedder = create_embedder(cfg.identity, BASE_DIR)
        backend = MediaPipeBackend(cfg.detector)
    except (ModelNotFoundError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
    source = CameraSource(cfg.camera)
    _emit("Enrollment: sit where you normally sit, face the screen, alone in view, and slowly turn your head a little.")
    try:
        profile = run_enrollment(cfg, source, backend, embedder, _emit)
        save_profile(profile, Path(cfg.identity.profile_file), Path(cfg.privacy.key_file))
    except (ProfileError, OSError, ImportError) as exc:
        logger.error("%s", exc)
        return 2
    finally:
        source.close()
    _emit(f"Owner profile saved to {Path(cfg.identity.profile_file).expanduser()} ({profile.loo_hint}).")
    if not cfg.identity.enabled:
        _emit("Now set identity.enabled: true in config.yaml to use it.")
    return 0


def _build_verifier(cfg: AppConfig) -> IdentityVerifier | None:
    """Load the owner profile when ``identity.enabled``; raises ``ProfileError``/``FileNotFoundError``."""
    if not cfg.identity.enabled:
        return None
    from identity import OwnerVerifier, create_embedder, load_profile

    embedder = create_embedder(cfg.identity, BASE_DIR)
    profile = load_profile(Path(cfg.identity.profile_file), Path(cfg.privacy.key_file), embedder.name)
    logger.info("Owner profile loaded (%s)", profile.loo_hint)
    return OwnerVerifier(cfg.identity, profile, embedder)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program and return the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source == "video" and args.video is None:
        parser.error("--source video requires --video PATH")
    if args.report and args.source != "video":
        parser.error("--report requires --source video")
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)

    try:
        cfg = load_config(args.config if args.config.exists() else None)
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2
    if args.decrypt_evidence is not None:
        return _decrypt(cfg, args)
    if args.forget_owner:
        return _forget_owner(cfg)
    headless = args.mode == "headless" and not (args.benchmark is not None or args.report)
    if headless:
        setup_logging(cfg.runtime, args.log_level, console=sys.stderr is not None)
    if args.camera_index is not None:
        cfg = dataclasses.replace(cfg, camera=dataclasses.replace(cfg.camera, index=args.camera_index))
    model = Path(cfg.detector.model_path)
    if not model.is_absolute():
        model = BASE_DIR / model
    cfg = dataclasses.replace(cfg, detector=dataclasses.replace(cfg.detector, model_path=str(model)))

    if args.enroll:
        return _enroll(args, cfg)

    guard = SingleInstance(Path(cfg.runtime.lock_file)) if headless else None
    if guard is not None and not guard.acquire():
        logger.error("Another instance is already running (lock %s)", cfg.runtime.lock_file)
        return 3
    try:
        return _run(args, cfg, headless)
    finally:
        if guard is not None:
            guard.release()


def _run(args: argparse.Namespace, cfg: AppConfig, headless: bool) -> int:
    try:
        backend = MediaPipeBackend(cfg.detector)
    except ModelNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    try:
        verifier = _build_verifier(cfg)
    except (FileNotFoundError, ImportError) as exc:
        logger.error("%s", exc)
        backend.close()
        return 2
    except Exception as exc:  # noqa: BLE001 - ProfileError and any model/crypto failure: refuse to run unprotected
        logger.error("Owner identity is enabled but unusable: %s", exc)
        backend.close()
        return 2
    pipeline = Pipeline(cfg, backend, verifier)
    try:
        source = _build_source(args, cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        pipeline.close()
        return 2

    measuring = args.benchmark is not None or args.report
    mode = args.actions or ("on" if headless else "dry-run")
    evidence: EvidenceSink | None = None
    try:
        executor, overlay = (None, None) if measuring else _build_executor(mode, cfg)
        if cfg.privacy.evidence_enabled and not measuring:
            from evidence import EvidenceStore

            evidence = EvidenceStore(cfg.privacy)
    except (RuntimeError, ImportError, OSError) as exc:
        logger.error("%s", exc)
        source.close()
        pipeline.close()
        return 2
    handler = ResponseHandler(executor, evidence)

    try:
        if args.benchmark is not None:
            _emit(format_benchmark(run_benchmark(source, pipeline, args.benchmark, args.bench_fps)))
            return 0
        if args.report:
            windows = load_windows(args.labels) if args.labels else []
            run = replay(source, pipeline)
            _emit(format_report(evaluate(run.transitions, run.duration_s, windows)))
            return 0
        if headless:
            return _run_headless(args, cfg, source, pipeline, handler)

        def on_result(packet: FramePacket, result: PipelineResult) -> None:
            handler(packet, result)

        return run_gui(cfg.gui, source, pipeline, on_result)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 0
    finally:
        if overlay is not None:
            overlay.close()
        source.close()
        pipeline.close()


def _run_headless(
    args: argparse.Namespace, cfg: AppConfig, source: FrameSource, pipeline: Pipeline, handler: ResponseHandler
) -> int:
    stop = threading.Event()
    restore = install_signal_handlers(stop)
    tray = create_tray(cfg.runtime.tray_enabled and not args.no_tray)
    runner = HeadlessRunner(cfg.runtime, source, pipeline, handler, tray, stop, paced=args.source == "cam")
    try:
        return runner.run()
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(main())
