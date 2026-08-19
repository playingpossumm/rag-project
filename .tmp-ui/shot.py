"""Screenshot the inspector. usage: shot.py <outdir> [--query "..."] [--theme dark]"""
import sys, pathlib
from playwright.sync_api import sync_playwright

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
args = sys.argv[2:]
def opt(name, default=None):
    return args[args.index(name)+1] if name in args else default

QUERIES = [
    ("answer", "What BLEU score did the Transformer achieve on WMT 2014 English-to-German?"),
    ("refusal", "What is the airspeed velocity of an unladen swallow?"),
]
themes = (opt("--theme") or "light,dark").split(",")
widths = [int(w) for w in (opt("--widths") or "1440").split(",")]
only = opt("--only")

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    for theme in themes:
        for w in widths:
            pg = b.new_page(viewport={"width": w, "height": 1000},
                            color_scheme=theme, device_scale_factor=2)
            logs = []
            pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
            pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
            pg.goto("http://127.0.0.1:8000/", wait_until="networkidle")
            pg.wait_for_timeout(700)
            tag = f"{theme}-{w}"
            pg.screenshot(path=out / f"00-landing-{tag}.png", full_page=True)
            for name, q in QUERIES:
                if only and only != name: continue
                pg.fill("#q", q)
                pg.click("#go")
                pg.wait_for_timeout(9000)
                pg.screenshot(path=out / f"01-{name}-{tag}.png", full_page=True)
            for sel, name in [("#v-corpus", "corpus"), ("#v-eval", "eval")]:
                if pg.locator(sel).count():
                    pg.click(sel); pg.wait_for_timeout(1500)
                    pg.screenshot(path=out / f"02-{name}-{tag}.png", full_page=True)
            errs = [l for l in logs if "PAGEERROR" in l or l.startswith("error")]
            print(f"[{tag}] console: {errs if errs else 'clean'}")
            pg.close()
    b.close()
print("wrote", out)
