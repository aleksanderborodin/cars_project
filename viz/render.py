"""Render a race video in the Kiln game style from episode logs of several methods on one track.

    uv run python viz/render.py --run ppo "RL · PPO" PPO runs/.../episode_00000_seed_10000.json.gz \
        --run base "Baseline" PID runs/.../episode_00000_seed_10000.json.gz \
        --hot ppo --rail "Unseen track · seed 10000" --out videos/race

Writes <out>/game.html (plays live in a browser), one MP4 per shot and <out>/race.mp4 (all shots joined).
Default shots: wide opening, follow the ember run for its whole lap, and the crash close-up if there is one.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from convert import convert  # noqa: E402

KILN_TOOLS = Path(os.environ.get("KILN_TOOLS", Path.home() / "Desktop/Kiln/tools"))
THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0"
FONTS_CDN = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Onest:wght@300;400;500'
             '&family=JetBrains+Mono:wght@500;700&display=swap">')


def icon_models() -> tuple[str, str]:
    """The car and the cone from the Kiln icon set, exported fresh so the game matches the icons."""
    sys.path.insert(0, str(KILN_TOOLS))
    import icons3d as I
    tmp = HERE / "_icons"
    tmp.mkdir(exist_ok=True)
    car, cone = tmp / "car.json", tmp / "cone.json"
    I.export_parts(str(car))
    I.export_parts(str(cone), {"cone": (I.cone(), (0, 0, 0))}, {})
    out = car.read_text(), cone.read_text()
    car.unlink(); cone.unlink(); tmp.rmdir()
    return out


def build_page(data: dict, out_html: Path) -> Path:
    car, cone = icon_models()
    page = ((HERE / "scene.html").read_text()
            .replace("/*FONTS*/", "").replace("THREE_URL", THREE_CDN)
            .replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
            .replace("/*CAR*/null", car).replace("/*CONE*/null", cone)
            .replace("<style>", FONTS_CDN + "\n<style>", 1))
    out_html.write_text(page)
    return out_html


def default_shots(data: dict) -> list[tuple[str, float, float]]:
    hot = data["runs"][data["hot"]]
    shots = [("wide", 0.0, 3.5), ("follow", 0.0, hot["lap"] + 0.6)]
    if data["crash"]:
        r = data["runs"][data["crash"]["run"]]
        t = r["t"][data["crash"]["i"]]
        shots.append(("crash", max(0.0, t - 1.2), t + 1.6))
    shots.append(("wide", hot["lap"] + 0.6, hot["lap"] + 3.0))
    return shots


async def shoot(html: Path, shots, out: Path, fps: int, size=(1600, 900)) -> list[Path]:
    from playwright.async_api import async_playwright
    clips = []
    async with async_playwright() as p:
        # Try the GPU first; fall back to software WebGL.
        for args in (["--use-angle=vulkan", "--enable-gpu", "--ignore-gpu-blocklist"],
                     ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]):
            b = await p.chromium.launch(args=args + ["--allow-file-access-from-files"])
            pg = await b.new_page(viewport={"width": size[0], "height": size[1]})
            errs: list[str] = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            await pg.goto(html.resolve().as_uri())
            try:
                await pg.wait_for_function("window.READY === true", timeout=120000)
                break
            except Exception:
                print("renderer failed with", args[0], "; ".join(errs[:3]))
                await b.close()
        else:
            sys.exit("page failed to load")
        for k, (mode, t0, t1) in enumerate(shots):
            d = out / f"{k:02d}_{mode}"
            d.mkdir(parents=True, exist_ok=True)
            await pg.evaluate(f"window.SHOT0 = {t0}")      # the chase camera's spring starts here
            n = int(round((t1 - t0) * fps)) + 1
            for j in range(n):
                await pg.evaluate(f"frame('{mode}', {t0 + j / fps})")
                await pg.screenshot(path=str(d / f"f{j:04d}.png"))
            clip = out / f"{k:02d}_{mode}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(d / "f%04d.png"),
                            "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", str(clip)], check=True)
            print(f"{mode} {t0:.1f}-{t1:.1f}s: {n} frames", flush=True)
            clips.append(clip)
        await b.close()
    return clips


def join(clips: list[Path], out_file: Path) -> None:
    lst = out_file.with_suffix(".txt")
    lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out_file)], check=True)
    lst.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=4, action="append", metavar=("KEY", "NAME", "TAG", "LOG"), required=True)
    ap.add_argument("--hot", required=True, help="the run drawn in ember and followed by the camera")
    ap.add_argument("--rail", default="", help="top-left rail text")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--page-only", action="store_true")
    ap.add_argument("--preview", action="store_true", help="one still per shot (its middle frame) instead of video")
    a = ap.parse_args()
    runs = [{"key": k, "name": n, "tag": t, "log": log} for k, n, t, log in a.run]
    hot_name = next(r["name"] for r in runs if r["key"] == a.hot)
    meta = {"railL": a.rail, "railR": f"{len(runs)} methods · one lap", "big": f"{hot_name} · lap"}
    data = convert(runs, a.hot, meta)
    a.out.mkdir(parents=True, exist_ok=True)
    html = build_page(data, a.out / "game.html")
    print("page:", html)
    if a.page_only:
        return
    shots = default_shots(data)
    if a.preview:
        shots = [(m, (t0 + t1) / 2, (t0 + t1) / 2) for m, t0, t1 in shots]
        asyncio.run(shoot(html, shots, a.out / "preview", 1))
        return
    clips = asyncio.run(shoot(html, shots, a.out, a.fps))
    join(clips, a.out / "race.mp4")
    print("video:", a.out / "race.mp4")


if __name__ == "__main__":
    main()
