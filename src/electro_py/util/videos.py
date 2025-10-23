# FUNCTIONS FOR WORKING WITH VIDEO DATA


import numpy as np
import json, shlex, subprocess as sp
from fractions import Fraction


def probe_video(path):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,avg_frame_rate -of json "{path}"'
    info = json.loads(sp.run(shlex.split(cmd), capture_output=True, text=True).stdout)
    s = info["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    # nb_frames may be missing for some MP4s; fall back to duration*fps if so
    n = s.get("nb_frames")
    if n is None or n == "0":
        # duration: use -show_entries format=duration if you want exact; here use avg_frame_rate only
        # For brevity assume container reports nb_frames; if not, you can add a duration probe.
        raise RuntimeError(
            "nb_frames not available; probe duration or decode once to count."
        )
    n = int(n)
    return w, h, n


def load_video_ffmpeg(path):
    w, h, n = probe_video(path)
    # -vsync 0 keeps 1:1 frame mapping; -pix_fmt rgb24 gives (H,W,3) uint8
    cmd = f'ffmpeg -nostdin -i "{path}" -f rawvideo -pix_fmt rgb24 -vsync 0 -'
    p = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.DEVNULL)
    buf = p.stdout.read(w * h * 3 * n)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, h, w, 3)
    p.stdout.close()
    p.wait()
    return arr
