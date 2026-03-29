"""
Medical Triage RL – Hospital Ward Renderer  v5  (layout-perfect)
================================================================
Canvas  : 1280 × 800  (standard 16:10, fits most screens)
Layout  : everything is derived from a single set of constants
          so nothing ever overflows or disappears.

  HEADER  : 80px  (title bar + cross + windows)
  BED GRID: 4 cols × 2 rows, each cell 290 × 235 px
            total grid = 1160 × 470, centred in canvas
  AISLE   : 20px gap between the two bed rows
  HUD     : 90px footer (stats + action banner)
  SIDEBAR : 116px right margin for resource bars

  Doctor  : stands in the 20px aisle or beside active bed
  Patient : large figure lying in bed, clearly readable
  All text: sized to fit within its container

Author : Christian Ishimwe – ALU / Machine Learning & Robotics
"""

from __future__ import annotations
import os, math
import numpy as np

MAX_PATIENTS = 8

SEV = {
    0: (110,110,110),
    1: (225, 30, 30),   # Critical   – red
    2: (230,115,  0),   # Emergent   – orange
    3: (210,198, 18),   # Urgent     – yellow
    4: ( 38,128,222),   # Less-urgent– blue
    5: ( 28,185, 55),   # Non-urgent – green
}
SEV_NAME = {
    1:"CRITICAL", 2:"EMERGENT", 3:"URGENT",
    4:"LESS-URG", 5:"NON-URGENT",
}

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import pygame
    PG_OK = True
except ImportError:
    PG_OK = False

try:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import Texture, CardMaker, WindowProperties, PNMImage
    P3D_OK = True
except ImportError:
    P3D_OK = False


# ── colour helpers ──────────────────────────────────────────────
def lc(a, b, t):
    return tuple(int(a[i]+(b[i]-a[i])*t) for i in range(3))
def dk(c, f=0.72): return tuple(int(x*f) for x in c)
def lt(c, f=1.28): return tuple(min(255,int(x*f)) for x in c)


# ═══════════════════════════════════════════════════════════════
#  Layout constants  – single source of truth
# ═══════════════════════════════════════════════════════════════
class Layout:
    # Canvas
    W = 1280
    H = 800

    # Zones (px from top)
    HEADER_H = 78    # title + cross + windows
    FOOTER_H = 100   # stat boxes (48px) + 4px gap + action banner (~28px) + margins
    SIDEBAR_W = 118  # right sidebar for resource bars

    # Bed grid
    COLS    = 4
    ROWS    = 2
    CELL_W  = 285    # width of one cell  (4 × 285 + margins ≈ 1160)
    CELL_H  = 228    # height of one cell
    GAP_X   = 8      # horizontal gap between cells
    GAP_Y   = 12     # vertical gap (aisle) between rows

    # Grid top-left: centred horizontally, just below header
    GRID_CONTENT_W = COLS * CELL_W + (COLS-1) * GAP_X   # 1168
    GRID_X = (W - SIDEBAR_W - GRID_CONTENT_W) // 2 + 2  # ≈ 0
    GRID_Y = HEADER_H + 4                                # 82

    # Total grid height
    GRID_H = ROWS * CELL_H + (ROWS-1) * GAP_Y           # 468

    # Aisle Y (between rows, used for doctor when idle)
    AISLE_Y = GRID_Y + CELL_H + GAP_Y // 2              # 82+228+6 = 316

    # Footer starts
    FOOTER_Y = H - FOOTER_H                              # 710

    @classmethod
    def cell_origin(cls, idx):
        """Top-left pixel of cell idx (0-7)."""
        r, c = divmod(idx, cls.COLS)
        x = cls.GRID_X + c * (cls.CELL_W + cls.GAP_X)
        y = cls.GRID_Y + r * (cls.CELL_H + cls.GAP_Y)
        return x, y

L = Layout()   # singleton


# ═══════════════════════════════════════════════════════════════
#  ScenePainter
# ═══════════════════════════════════════════════════════════════
class ScenePainter:
    W = L.W
    H = L.H

    # ── main ──────────────────────────────────────────────────
    def render(self, env, action=0, reward=0.0, rh=None):
        if not PIL_OK:
            raise ImportError("Pillow required.")
        rh  = rh or []
        img = Image.new("RGB", (L.W, L.H), (12,16,28))
        d   = ImageDraw.Draw(img)

        self._bg(d)
        self._header(d)
        self._ceiling_lights(d)

        for i in range(MAX_PATIENTS):
            cx, cy = L.cell_origin(i)
            self._cell(d, cx, cy, i, env.patients[i], action, env)

        self._doctor(d, env, action)
        self._footer(d, env, action, reward)
        self._sidebar(d, env)
        self._mini_graph(d, rh)
        self._legend(d)

        return img

    # ── background ────────────────────────────────────────────
    def _bg(self, d):
        # Wall (above floor line)
        floor_y = L.GRID_Y + L.GRID_H + 18
        for y in range(L.HEADER_H, floor_y):
            t = (y - L.HEADER_H) / max(floor_y - L.HEADER_H, 1)
            d.line([(0,y),(L.W,y)], fill=lc((188,200,215),(162,176,196),t))
        # Floor
        for y in range(floor_y, L.FOOTER_Y):
            t = (y-floor_y)/max(L.FOOTER_Y-floor_y,1)
            d.line([(0,y),(L.W,y)], fill=lc((62,68,82),(48,52,64),t))
        # Floor tiles
        for x in range(0, L.W, 88):
            d.line([(x,floor_y),(x,L.FOOTER_Y)], fill=(72,78,92), width=1)
        for y in range(floor_y, L.FOOTER_Y, 66):
            d.line([(0,y),(L.W,y)], fill=(72,78,92), width=1)
        # Skirting
        d.rectangle([(0,floor_y-3),(L.W,floor_y+8)], fill=(132,140,155))
        # Dado rail
        d.rectangle([(0,L.HEADER_H+2),(L.W,L.HEADER_H+8)], fill=(155,162,175))

    def _header(self, d):
        d.rectangle([(0,0),(L.W, L.HEADER_H)], fill=(14,20,42))
        # Windows – left side
        for wx in [8, 148]:
            self._window(d, wx, 8, 120, 60)
        # Windows – right side
        for wx in [L.W - 140, L.W - 280]:
            self._window(d, wx, 8, 120, 60)
        # Green medical cross – centre
        cx = L.W // 2
        d.rectangle([(cx-7,10),(cx+7,60)],  fill=(28,172,65))
        d.rectangle([(cx-22,28),(cx+22,42)], fill=(28,172,65))
        # Title banner – two lines
        d.rectangle([(cx-390, 48),(cx+390, L.HEADER_H-2)],
                    fill=(22,32,68), outline=(52,78,148), width=1)
        # Top line: name
        self._t(d, "CHRISTIAN ISHIMWE WARD",
                cx, 57, 13, (255, 220, 80), True, True)
        # Bottom line: unit subtitle
        self._t(d, "AUTOMATED AFIYA BORA TRIAGE UNIT",
                cx, 72, 12, (118, 188, 255), True, False)

    def _window(self, d, x, y, w, h):
        d.rectangle([(x,y),(x+w,y+h)],
                    fill=(138,192,228), outline=(188,195,208), width=2)
        d.line([(x+w//2,y),(x+w//2,y+h)], fill=(188,195,208), width=1)
        d.line([(x,y+h//2),(x+w,y+h//2)], fill=(188,195,208), width=1)
        for qx,qy,qw,qh in [(x+2,y+2,w//2-3,h//2-3),(x+w//2+1,y+2,w//2-3,h//2-3),
                             (x+2,y+h//2+1,w//2-3,h//2-3),(x+w//2+1,y+h//2+1,w//2-3,h//2-3)]:
            d.rectangle([(qx,qy),(qx+qw,qy+qh)], fill=(118,172,215))
        d.ellipse([(x+6,y+4),(x+32,y+18)],  fill=(228,235,245))
        d.ellipse([(x+w//2+4,y+3),(x+w-6,y+16)], fill=(228,235,245))

    def _ceiling_lights(self, d):
        for lx in [180, 480, 780, 1080]:
            d.rectangle([(lx-40,0),(lx+40,10)], fill=(238,238,222))
            for r in range(22,0,-4):
                g = lc((255,255,215),(235,235,180), r/22)
                d.ellipse([(lx-r,6-r//2),(lx+r,6+r//2)], fill=g)
            for ang in range(-55,56,18):
                ex = int(lx + 62*math.sin(math.radians(ang)))
                ey = int(10  + 62*math.cos(math.radians(ang)))
                d.line([(lx,10),(ex,ey)], fill=(255,250,195), width=1)

    # ── bed cell ──────────────────────────────────────────────
    def _cell(self, d, cx, cy, idx, pat, action, env):
        sev     = int(pat[0])
        alive   = bool(pat[3])
        treated = bool(pat[2])
        active  = pat[0] > 0
        wait    = int(pat[1])
        det     = int(pat[4])
        tgt     = (action == idx+1)

        CW, CH = L.CELL_W, L.CELL_H

        # Panel colour
        if not active:
            pc, bc = (20,22,30), (48,52,65)
        elif not alive:
            pc, bc = (35,14,14), (175,28,28)
        elif treated:
            pc, bc = (14,38,20), (32,175,62)
        elif tgt:
            pc, bc = (32,48,28), (195,212,42)
        else:
            sr,sg,sb = SEV.get(sev,(100,100,100))
            pc = (max(16,sr//9),max(16,sg//9),max(16,sb//9))
            bc = SEV.get(sev,(90,90,90))

        # Panel
        d.rectangle([(cx+3,cy+3),(cx+CW-3,cy+CH-3)],
                    fill=pc, outline=bc, width=2)

        # Critical pulsing glow
        if sev==1 and alive and not treated and not tgt:
            for r in [12,8,4]:
                d.rectangle([(cx+3-r,cy+3-r),(cx+CW-3+r,cy+CH-3+r)],
                             outline=(218,28,28), width=1)

        # Top severity bar + bed number
        d.rectangle([(cx+5,cy+5),(cx+CW-5,cy+22)],
                    fill=SEV.get(sev,(55,58,70)) if active else (42,45,55))
        self._t(d, f"BED {idx+1}", cx+CW//2, cy+13, 11,
                (255,255,255), True, True)

        if not active:
            self._t(d, "[ EMPTY ]", cx+CW//2, cy+CH//2, 14,
                    (80,84,98), True)
            return

        # ── 3-D bed ──────────────────────────────────────────
        bx  = cx + 28
        by  = cy + 26
        bw  = CW - 72     # 185 px wide
        bh  = 138         # bed height

        # Shadow
        d.polygon([(bx+6,by+bh+3),(bx+bw+6,by+bh+3),
                   (bx+bw+2,by+bh+9),(bx+2,by+bh+9)], fill=(12,14,22))
        # Side face (3-D depth)
        side = dk((142,150,172), 0.62)
        d.polygon([(bx+bw,by),(bx+bw+10,by-7),
                   (bx+bw+10,by+bh-7),(bx+bw,by+bh)], fill=side)
        # Top lip
        top_col = lt((148,155,175),1.08)
        d.polygon([(bx,by),(bx+bw,by),
                   (bx+bw+10,by-7),(bx+10,by-7)], fill=top_col)
        # Front face
        bed_f = (162,168,190)
        d.rectangle([(bx,by),(bx+bw,by+bh)],
                    fill=bed_f, outline=(118,125,145), width=2)
        # Headboard
        hb = (98,110,138)
        d.rectangle([(bx,by),(bx+18,by+bh)], fill=hb, outline=(80,90,115), width=1)
        d.polygon([(bx,by),(bx+10,by-7),(bx+10,by+bh-7),(bx,by+bh)],
                  fill=dk(hb,0.78))
        d.polygon([(bx,by),(bx+18,by),(bx+28,by-7),(bx+10,by-7)],
                  fill=lt(hb,1.12))
        # Mattress
        mc = (222,226,242) if alive else (140,140,152)
        d.rectangle([(bx+20,by+4),(bx+bw-4,by+bh-5)], fill=mc)
        # Pillow
        pc2 = (238,242,252) if alive else (172,172,182)
        d.ellipse([(bx+22,by+7),(bx+72,by+42)],
                  fill=pc2, outline=(195,200,218))
        d.arc([(bx+22,by+7),(bx+72,by+42)], 210,360, fill=(205,210,228), width=1)
        # Blanket
        blk = (78,125,188) if alive else (72,74,88)
        d.rectangle([(bx+20,by+bh//2),(bx+bw-4,by+bh-5)], fill=blk)
        d.rectangle([(bx+20,by+bh//2),(bx+bw-4,by+bh//2+10)], fill=dk(blk,0.8))
        d.polygon([(bx+20,by+bh//2),(bx+bw-4,by+bh//2),
                   (bx+bw+4,by+bh//2-6),(bx+28,by+bh//2-6)],
                  fill=lt(blk,1.1))
        # Rails
        for rx in [bx+bw-5, bx+bw+1]:
            d.rectangle([(rx,by+30),(rx+4,by+bh-5)], fill=(130,138,158))

        # ── patient in bed ───────────────────────────────────
        self._patient(d, bx, by, bw, bh, sev, alive, treated)

        # ── equipment ────────────────────────────────────────
        if sev <= 2 and alive:
            self._ecg(d, cx+CW-62, cy+28, alive, sev)
        if sev <= 3:
            self._iv(d, bx+bw+14, by-8)

        # ── status text (fits in remaining cell space) ────────
        sy = cy + CH - 58
        if not alive:
            self._t(d,"PATIENT DECEASED",cx+CW//2,sy,12,(212,40,40),True,True)
            # Small skull
            skx,sky = cx+CW//2, sy+18
            d.ellipse([(skx-10,sky-12),(skx+10,sky+4)], fill=(75,75,78))
            d.rectangle([(skx-8,sky),(skx+8,sky+8)], fill=(75,75,78))
            for ex in [skx-5,skx+2]:
                d.ellipse([(ex,sky-8),(ex+5,sky-3)], fill=(18,18,22))
        elif treated:
            self._t(d,"TREATED  ✓",cx+CW//2,sy,12,(32,195,68),True,True)
        else:
            nm = SEV_NAME.get(sev,"?")
            self._t(d, f"ESI-{sev}  {nm}",
                    cx+CW//2, sy, 12, SEV.get(sev,(155,155,155)), True, True)
        if alive and not treated:
            self._t(d,f"Wait:{wait}s  Det:{det}",
                    cx+CW//2, sy+16, 10, (215,170,50), True)
        if tgt:
            d.rectangle([(cx+CW//2-52,cy+CH-20),(cx+CW//2+52,cy+CH-5)],
                        fill=(32,138,32), outline=(75,210,75))
            self._t(d,"▶ TREATING",cx+CW//2,cy+CH-12,9,(195,255,195),True,True)

    # ── patient figure ────────────────────────────────────────
    def _patient(self, d, bx, by, bw, bh, sev, alive, treated):
        skin  = (212,165,122) if alive else (142,135,132)
        gown  = (148,200,235)

        fy = by + bh//2 - 18   # torso centre Y

        # Torso / gown
        d.ellipse([(bx+78,fy-18),(bx+bw-28,fy+18)],
                  fill=gown, outline=dk(gown,0.82), width=1)

        # Arms
        d.ellipse([(bx+76,fy+8),(bx+bw-40,fy+22)],
                  fill=skin, outline=dk(skin,0.85))

        # Head (on pillow, left)
        hcx, hcy, hr = bx+46, fy-5, 24

        # Head sphere shading
        for r in range(hr,0,-1):
            t = 1 - r/hr
            hc = lc(lt(skin,1.15), dk(skin,0.8), t*0.5)
            d.ellipse([(hcx-r,hcy-r),(hcx+r,hcy+r)], fill=hc)
        d.ellipse([(hcx-hr,hcy-hr),(hcx+hr,hcy+hr)],
                  outline=dk(skin,0.8), width=2)

        # Hair
        hair = (42,28,14)
        d.ellipse([(hcx-hr,hcy-hr),(hcx+hr,hcy-hr+22)], fill=hair)
        d.arc([(hcx-hr,hcy-hr),(hcx+hr,hcy-hr+26)],
              start=200, end=340, fill=hair, width=4)

        if alive:
            # Eyes
            for ex in [hcx-12,hcx+3]:
                d.ellipse([(ex,hcy-9),(ex+9,hcy-1)], fill=(38,38,65))
                d.ellipse([(ex+1,hcy-9),(ex+4,hcy-6)], fill=(252,252,255))
            # Eyebrows
            if sev==1:
                d.arc([(hcx-13,hcy-20),(hcx-2,hcy-11)], 205,335, fill=(55,38,22),width=2)
                d.arc([(hcx+2,hcy-20),(hcx+13,hcy-11)], 205,335, fill=(55,38,22),width=2)
            # Nose
            d.arc([(hcx-4,hcy-2),(hcx+4,hcy+5)],0,180,fill=dk(skin,0.85),width=1)
            # Mouth
            if sev==1:
                d.arc([(hcx-8,hcy+5),(hcx+8,hcy+16)],0,180,fill=(125,38,38),width=2)
                for px_,py_ in [(hcx-16,hcy-14),(hcx+8,hcy-14)]:
                    d.line([(px_,py_),(px_+6,py_+7)],fill=(175,95,78),width=1)
            elif sev==2:
                d.arc([(hcx-6,hcy+6),(hcx+6,hcy+15)],0,180,fill=(112,65,45),width=2)
            else:
                d.line([(hcx-6,hcy+10),(hcx+6,hcy+10)],fill=(112,65,45),width=2)
            # O2 mask for ESI-1
            if sev==1:
                d.ellipse([(hcx-13,hcy-5),(hcx+16,hcy+16)],
                          fill=(162,212,250), outline=(82,138,192), width=2)
                self._t(d,"O2",hcx+2,hcy+5,8,(35,85,172),True,True)
                d.line([(hcx-13,hcy+4),(hcx-hr-2,hcy+4)],fill=(105,142,172),width=1)
                d.line([(hcx+16,hcy+4),(hcx+hr+2,hcy+4)],fill=(105,142,172),width=1)
        else:
            for ex in [hcx-12,hcx+3]:
                d.line([(ex,hcy-9),(ex+9,hcy-1)],fill=(85,85,85),width=2)
                d.line([(ex+9,hcy-9),(ex,hcy-1)],fill=(85,85,85),width=2)
            d.line([(hcx-6,hcy+10),(hcx+6,hcy+10)],fill=(88,80,80),width=1)

        # Neck
        d.rectangle([(hcx-5,hcy+hr-5),(hcx+5,hcy+hr+8)], fill=skin)

        # Wristband
        wb = SEV.get(sev,(128,128,128))
        d.rectangle([(bx+80,fy+6),(bx+96,fy+18)], fill=wb, outline=(25,25,25))
        self._t(d,str(sev),bx+88,fy+12,7,(255,255,255),True,True)

        # Blanket over lower body
        blk2 = (78,125,188) if alive else (68,70,82)
        d.rectangle([(bx+76,fy+8),(bx+bw-30,fy+25)], fill=blk2)

        # Treated badge
        if treated:
            tx,ty = bx+bw//2+20, fy-28
            d.ellipse([(tx-14,ty-14),(tx+14,ty+14)], fill=(255,255,255))
            d.ellipse([(tx-14,ty-14),(tx+14,ty+14)], outline=(195,0,0),width=2)
            d.rectangle([(tx-9,ty-3),(tx+9,ty+3)], fill=(200,0,0))
            d.rectangle([(tx-3,ty-9),(tx+3,ty+9)], fill=(200,0,0))

    # ── ECG monitor ───────────────────────────────────────────
    def _ecg(self, d, x, y, alive, sev):
        w,h = 50,38
        # 3-D box
        d.polygon([(x+w,y),(x+w+7,y-5),(x+w+7,y+h-5),(x+w,y+h)],fill=(10,16,25))
        d.polygon([(x,y),(x+w,y),(x+w+7,y-5),(x+7,y-5)],fill=(24,35,52))
        d.rectangle([(x,y),(x+w,y+h)],fill=(12,20,32),outline=(68,82,98),width=2)
        # Screen
        d.rectangle([(x+2,y+2),(x+w-2,y+h-10)],fill=(4,12,8))
        if alive:
            pts=[]
            for i in range(13):
                mx=x+3+i*3
                my=(y+8 if i==4 else y+26 if i==5 else y+6 if i==6 else y+16)
                pts.append((mx,my))
            ec=(0,225,75) if sev>2 else (225,50,50)
            d.line(pts,fill=ec,width=2)
            self._t(d,f"{60+sev*7}bpm",x+w//2,y+h-5,7,ec,True)
        else:
            d.line([(x+3,y+16),(x+w-3,y+16)],fill=(195,32,32),width=2)
            self._t(d,"FLAT",x+w//2,y+h-5,7,(195,32,32),True)
        # Stand
        d.line([(x+w//2,y+h),(x+w//2,y+h+18)],fill=(82,90,105),width=3)
        d.rectangle([(x+w//2-11,y+h+17),(x+w//2+12,y+h+24)],fill=(68,75,88))

    # ── IV stand ──────────────────────────────────────────────
    def _iv(self, d, x, y):
        d.line([(x,y),(x,y+128)],fill=(168,175,192),width=4)
        d.rectangle([(x-15,y+2),(x+15,y+8)],fill=(155,162,180))
        # Bag
        bag=(152,220,180)
        d.ellipse([(x-14,y+8),(x+14,y+52)],fill=bag,outline=(82,155,112),width=2)
        d.ellipse([(x-8,y+12),(x,y+26)],fill=lt(bag,1.18))
        self._t(d,"IV",x,y+30,8,(25,100,50),True,True)
        # Tube + drip
        d.line([(x,y+52),(x,y+95)],fill=(148,182,158),width=2)
        d.ellipse([(x-4,y+88),(x+4,y+96)],fill=(68,178,122))
        # Wheels
        for wx in [x-14,x-3,x+6]:
            d.ellipse([(wx,y+120),(wx+10,y+130)],fill=(95,100,115))

    # ── doctor (110px tall, clear face) ───────────────────────
    def _doctor(self, d, env, action):
        if 0 < action <= MAX_PATIENTS:
            cx_b, cy_b = L.cell_origin(action-1)
            # Place doctor just outside right edge of bed cell
            dx = cx_b + L.CELL_W + 2
            dy = cy_b + L.CELL_H - 30
            # If that overflows right, place inside the cell on right
            if dx + 35 > L.W - L.SIDEBAR_W:
                dx = cx_b - 38
        else:
            # Stand in the aisle between rows, centre
            dx = L.W // 2 - L.SIDEBAR_W // 2
            dy = L.AISLE_Y + 35

        self._draw_doctor(d, dx, dy, treating=(action > 0))

        # Speech bubble – always above doctor
        if action == 0:
            txt = "Assessing all patients..."
        else:
            p   = env.patients[action-1]
            sev = int(p[0])
            txt = f"Treating P{action} — ESI-{sev} {SEV_NAME.get(sev,'')}"
        self._bubble(d, dx, dy-118, txt)

    def _draw_doctor(self, d, cx, cy, treating=False):
        coat  = (238,244,252)
        coat2 = dk(coat,0.86)
        scrub = (65,118,192)
        scrb2 = dk(scrub,0.78)
        skin  = (212,168,125)
        skin2 = dk(skin,0.82)
        hair  = (40,26,12)

        # Shoes – white medical clogs
        shoe_w = (245, 248, 252)   # bright white
        shoe_s = (200, 205, 215)   # subtle shadow side
        # Left shoe
        d.ellipse([(cx-18,cy+94),(cx-1, cy+110)], fill=shoe_w, outline=(175,180,195), width=2)
        d.ellipse([(cx-16,cy+96),(cx-4, cy+104)], fill=(225,228,238))   # toe cap sheen
        d.line([(cx-18,cy+103),(cx-1,cy+103)], fill=shoe_s, width=1)    # sole line
        # Right shoe
        d.ellipse([(cx+1,  cy+94),(cx+18,cy+110)], fill=shoe_w, outline=(175,180,195), width=2)
        d.ellipse([(cx+4,  cy+96),(cx+16,cy+104)], fill=(225,228,238))
        d.line([(cx+1, cy+103),(cx+18,cy+103)], fill=shoe_s, width=1)
        # Trousers
        d.rectangle([(cx-14,cy+40),(cx-3, cy+98)], fill=scrub)
        d.polygon([(cx-14,cy+40),(cx-3,cy+40),(cx-5,cy+98),(cx-16,cy+98)],fill=scrb2)
        d.rectangle([(cx+3, cy+40),(cx+14,cy+98)], fill=scrub)
        d.polygon([(cx+3, cy+40),(cx+14,cy+40),(cx+16,cy+98),(cx+5, cy+98)],fill=scrb2)
        # Coat
        d.rectangle([(cx-26,cy+14),(cx+28,cy+52)], fill=coat2)
        d.rectangle([(cx-22,cy-12),(cx+22,cy+52)], fill=coat)
        d.polygon([(cx,cy-6),(cx-22,cy-12),(cx-13,cy+25)],fill=(222,228,238))
        d.polygon([(cx,cy-6),(cx+22,cy-12),(cx+13,cy+25)],fill=(214,220,232))
        d.rectangle([(cx-9,cy),(cx+9, cy+50)], fill=scrub)
        # Coat bottom edge
        d.polygon([(cx-22,cy+52),(cx+22,cy+52),(cx+28,cy+56),(cx-26,cy+56)],fill=coat2)
        # Stethoscope
        d.arc([(cx-13,cy+5),(cx+13,cy+30)], 218,322, fill=(40,40,52), width=4)
        d.ellipse([(cx-6,cy+26),(cx+6,cy+40)], fill=(35,35,48))
        d.ellipse([(cx-4,cy+29),(cx+4,cy+37)], fill=(82,85,100))
        d.line([(cx-13,cy+7),(cx-22,cy+1)],fill=(40,40,52),width=2)
        d.line([(cx+13,cy+7),(cx+22,cy+1)],fill=(40,40,52),width=2)
        d.ellipse([(cx-26,cy-4),(cx-18,cy+4)],fill=(40,40,52))
        d.ellipse([(cx+18,cy-4),(cx+26,cy+4)],fill=(40,40,52))
        # Pocket + pen
        d.rectangle([(cx+9,cy+12),(cx+21,cy+28)],fill=(222,228,240),outline=(182,190,205))
        d.line([(cx+12,cy+12),(cx+12,cy+31)],fill=(12,52,145),width=2)
        d.ellipse([(cx+9,cy+30),(cx+15,cy+36)],fill=(12,52,145))
        # ID badge
        d.rectangle([(cx-23,cy+8),(cx-9,cy+24)],fill=(188,220,252),outline=(82,138,195))
        self._t(d,"DR",cx-16,cy+16,7,(20,50,112),True,True)
        d.line([(cx-16,cy+8),(cx-16,cy-14)],fill=(82,40,40),width=1)
        # Arms
        if treating:
            d.line([(cx-20,cy+5),(cx-40,cy-22)],fill=coat,width=11)
            d.line([(cx-20,cy+5),(cx-40,cy-22)],fill=coat2,width=9)
            d.ellipse([(cx-48,cy-32),(cx-30,cy-14)],fill=(222,235,222))
            d.line([(cx+20,cy+5),(cx+38,cy+25)],fill=coat,width=11)
            d.line([(cx+20,cy+5),(cx+38,cy+25)],fill=coat2,width=9)
            self._clipboard(d,cx+36,cy+23)
        else:
            d.line([(cx-20,cy+5),(cx-36,cy+46)],fill=coat,width=11)
            d.line([(cx-20,cy+5),(cx-36,cy+46)],fill=coat2,width=9)
            d.ellipse([(cx-44,cy+38),(cx-27,cy+55)],fill=skin)
            d.line([(cx+20,cy+5),(cx+36,cy+46)],fill=coat,width=11)
            d.line([(cx+20,cy+5),(cx+36,cy+46)],fill=coat2,width=9)
            self._clipboard(d,cx+34,cy+44)
        # Head
        hr = 25
        hcy = cy-50
        d.rectangle([(cx-7,hcy+hr-4),(cx+7,hcy+hr+10)],fill=skin)
        for r in range(hr,0,-1):
            t=1-r/hr
            hc=lc(lt(skin,1.12),dk(skin,0.82),t*0.52)
            d.ellipse([(cx-r,hcy-r),(cx+r,hcy+r)],fill=hc)
        d.ellipse([(cx-hr,hcy-hr),(cx+hr,hcy+hr)],outline=skin2,width=2)
        # ── White surgical cap ──────────────────────────────────
        cap_w = (245, 248, 255)   # bright white
        cap_s = (192, 198, 218)   # soft shadow / fold
        cap_b = (52,  85, 168)    # blue brim band

        # Puffy dome sitting above the head
        d.ellipse([(cx-hr-3, hcy-hr-18),(cx+hr+3, hcy-hr+20)], fill=cap_w)
        # Shading creases on dome
        d.arc([(cx-hr+3, hcy-hr-14),(cx,      hcy-hr+16)], 195, 345, fill=cap_s, width=1)
        d.arc([(cx,      hcy-hr-14),(cx+hr-3, hcy-hr+16)], 195, 345, fill=cap_s, width=1)
        # Blue brim band at base of cap
        d.rectangle([(cx-hr-3, hcy-hr+16),(cx+hr+3, hcy-hr+24)], fill=cap_b)
        d.line([(cx-hr-2, hcy-hr+16),(cx+hr+2, hcy-hr+16)], fill=(175,192,235), width=1)
        # Red cross badge on cap front
        ccx, ccy = cx, hcy-hr+6
        d.rectangle([(ccx-6, ccy-2),(ccx+6, ccy+2)], fill=(215,12,12))
        d.rectangle([(ccx-2, ccy-6),(ccx+2, ccy+6)], fill=(215,12,12))
        # White halo so cross pops on white background
        d.rectangle([(ccx-7, ccy-3),(ccx+7, ccy+3)], outline=(245,248,255), width=1)
        d.rectangle([(ccx-3, ccy-7),(ccx+3, ccy+7)], outline=(245,248,255), width=1)
        # Eyes
        for ex in [cx-12,cx+3]:
            d.ellipse([(ex,hcy-10),(ex+9,hcy-2)],fill=(245,250,255))
            d.ellipse([(ex+1,hcy-10),(ex+8,hcy-3)],fill=(48,85,158))
            d.ellipse([(ex+2,hcy-9),(ex+6,hcy-5)],fill=(16,16,26))
            d.ellipse([(ex+2,hcy-9),(ex+5,hcy-7)],fill=(255,255,255))
        # Eyebrows
        brow=dk(hair,0.85)
        if treating:
            d.arc([(cx-14,hcy-23),(cx-1,hcy-14)],212,338,fill=brow,width=2)
            d.arc([(cx+1, hcy-23),(cx+14,hcy-14)],202,328,fill=brow,width=2)
        else:
            d.line([(cx-13,hcy-20),(cx-2,hcy-19)],fill=brow,width=2)
            d.line([(cx+2, hcy-20),(cx+13,hcy-19)],fill=brow,width=2)
        # Nose
        d.arc([(cx-4,hcy-2),(cx+4,hcy+6)],0,180,fill=skin2,width=2)
        d.ellipse([(cx-5,hcy+3),(cx-1,hcy+8)],fill=dk(skin,0.88))
        d.ellipse([(cx+1,hcy+3),(cx+5,hcy+8)],fill=dk(skin,0.88))
        # Mouth
        if treating:
            d.arc([(cx-8,hcy+10),(cx+8,hcy+20)],0,180,fill=(130,62,42),width=2)
        else:
            d.line([(cx-7,hcy+14),(cx+7,hcy+14)],fill=(122,58,40),width=2)
        # Surgical mask
        if treating:
            d.rectangle([(cx-18,hcy-6),(cx+18,hcy+18)],
                        fill=(162,202,225),outline=(122,165,195),width=2)
            for my in range(hcy-2,hcy+18,4):
                d.line([(cx-16,my),(cx+16,my)],fill=(142,182,208),width=1)
            d.line([(cx-18,hcy+4),(cx-26,hcy-10)],fill=(142,182,208),width=1)
            d.line([(cx+18,hcy+4),(cx+26,hcy-10)],fill=(142,182,208),width=1)
        # Ears
        for s in [-1,1]:
            ex_=cx+s*(hr-1)
            d.ellipse([(ex_-4,hcy-7),(ex_+4,hcy+7)],fill=skin)
            d.arc([(ex_-3,hcy-5),(ex_+3,hcy+5)],30+s*60,150+s*60,fill=skin2,width=1)

    def _clipboard(self, d, x, y):
        d.rectangle([(x+4,y-1),(x+28,y+36)],fill=(148,132,102))
        d.rectangle([(x,y),(x+26,y+36)],fill=(195,180,148),outline=(145,130,105))
        d.rectangle([(x+8,y-5),(x+18,y+3)],fill=(150,158,172))
        d.rectangle([(x+10,y-6),(x+16,y+1)],fill=(182,190,205))
        d.rectangle([(x+2,y+2),(x+24,y+34)],fill=(245,245,242))
        for i in range(6):
            lw=20 if i%2==0 else 14
            d.line([(x+3,y+5+i*5),(x+3+lw,y+5+i*5)],fill=(100,100,120),width=1)
        d.line([(x+3,y+8),(x+7,y+13)],fill=(190,0,0),width=2)
        d.line([(x+7,y+13),(x+13,y+5)],fill=(190,0,0),width=2)

    def _bubble(self, d, cx, cy, text):
        tw = max(len(text)*6+22, 130)
        th = 32
        x0,y0,x1,y1 = cx-tw//2,cy-th//2,cx+tw//2,cy+th//2
        # Clamp to canvas
        if x0 < 2: x0,x1 = 2, 2+tw
        if x1 > L.W-L.SIDEBAR_W-2: x1,x0 = L.W-L.SIDEBAR_W-2, L.W-L.SIDEBAR_W-2-tw
        # Shadow
        d.rectangle([(x0+3,y0+3),(x1+3,y1+3)],fill=(10,14,25))
        d.rectangle([(x0,y0),(x1,y1)],fill=(252,252,238),outline=(152,152,132),width=2)
        d.polygon([(cx-7,y1),(cx+10,y1),(cx+3,y1+15)],fill=(252,252,238))
        d.line([(cx-7,y1),(cx+3,y1+15)],fill=(152,152,132),width=1)
        d.line([(cx+10,y1),(cx+3,y1+15)],fill=(152,152,132),width=1)
        self._t(d,text,x0+(x1-x0)//2,cy,11,(30,30,50),True)

    # ── footer HUD ────────────────────────────────────────────
    def _footer(self, d, env, action, reward):
        fy = L.FOOTER_Y
        # Full footer background
        d.rectangle([(0, fy),(L.W, L.H)], fill=(8, 12, 28))

        # ── STAT BOXES – top of footer, always fully visible ──
        stats = [
            ("STEP",    f"{env.step_count}/200",        (160, 190, 255)),
            ("TREATED", str(env.total_treated),          ( 60, 208,  85)),
            ("DEATHS",  str(env.total_deaths),           (212,  62,  62)),
            ("REWARD",  f"{reward:+.1f}",                (255, 190,  40)),
            ("BEDS",    f"{env.beds_available}/6",       ( 85, 200, 252)),
            ("STAFF",   f"{env.staff_available}/4",      (132, 210, 252)),
            ("EQUIP",   f"{env.equipment_available}/3",  (182, 210, 252)),
        ]
        n       = len(stats)
        usable  = L.W - L.SIDEBAR_W - 16   # px available for boxes
        box_w   = usable // n               # width of each box
        box_h   = 48                        # fixed height – plenty of room
        box_top = fy + 4                    # 4px gap from top of footer

        for i, (lbl, val, col) in enumerate(stats):
            bx = 8 + i * box_w
            # Box background + coloured border
            d.rectangle([(bx,      box_top),
                          (bx+box_w-4, box_top+box_h)],
                        fill=(18, 28, 58), outline=col, width=2)
            # Label (small, centred, top of box)
            self._t(d, lbl,
                    bx + box_w//2 - 2, box_top + 10,
                    9, (140, 155, 195), True, False)
            # Value (large, bold, centred, bottom of box)
            self._t(d, val,
                    bx + box_w//2 - 2, box_top + 30,
                    16, col, True, True)

        # ── ACTION BANNER – below the stat boxes ──────────────
        banner_y = box_top + box_h + 4
        d.rectangle([(0, banner_y),(L.W, L.H)], fill=(14, 22, 48))
        if action == 0:
            atxt = "  ⬛  HOLDING  –  Assessing all patients in the ward"
            ac   = (105, 172, 255)
        else:
            p    = env.patients[action - 1]
            sev  = int(p[0])
            atxt = f"  ▶  TREATING Patient {action}  |  ESI-{sev}: {SEV_NAME.get(sev,'?')}"
            ac   = (255, 210, 45)
        # Vertical centre of remaining banner space
        banner_cy = banner_y + (L.H - banner_y) // 2
        self._t(d, atxt, L.W // 2, banner_cy, 13, ac, True, True)

    # ── right sidebar ─────────────────────────────────────────
    def _sidebar(self, d, env):
        sx = L.W - L.SIDEBAR_W + 4
        items=[
            ("BEDS",  env.beds_available,     6,  (50,145,250)),
            ("STAFF", env.staff_available,    4,  (60,205,105)),
            ("EQUIP", env.equipment_available,3,  (245,165,45)),
        ]
        for i,(lbl,val,mx,col) in enumerate(items):
            by_=L.GRID_Y+i*52
            d.rectangle([(sx,by_),(L.W-4,by_+46)],fill=(15,22,44),outline=(44,55,82))
            d.rectangle([(sx+4,by_+4),(L.W-8,by_+46-4)],fill=(18,26,50))
            fw=int((val/mx)*(L.SIDEBAR_W-18))
            if fw>0:
                d.rectangle([(sx+4,by_+22),(sx+4+fw,by_+38)],fill=col)
            self._t(d,lbl,sx+L.SIDEBAR_W//2,by_+11,10,(205,215,235),True,True)
            self._t(d,f"{val}/{mx}",sx+L.SIDEBAR_W//2,by_+34,12,col,True,True)

    # ── mini reward graph ─────────────────────────────────────
    def _mini_graph(self, d, rh):
        if len(rh)<2: return
        sx = L.W - L.SIDEBAR_W + 4
        gx,gy,gw,gh = sx, L.GRID_Y+165, L.SIDEBAR_W-8, 72
        d.rectangle([(gx,gy),(gx+gw,gy+gh)],fill=(10,15,32),outline=(44,55,82))
        self._t(d,"Reward",gx+gw//2,gy+8,8,(140,152,188),True)
        data=rh[-50:]
        lo,hi=min(data),max(data)
        if hi==lo: return
        pts=[]
        for i,v in enumerate(data):
            px_=gx+4+int(i*(gw-8)/max(len(data)-1,1))
            py_=gy+gh-4-int((v-lo)/(hi-lo)*(gh-18))
            pts.append((px_,py_))
        if len(pts)>1:
            for s in [(0,2),(0,1)]:
                d.line([(p[0]+s[0],p[1]+s[1]) for p in pts],fill=(22,78,102),width=3)
            d.line(pts,fill=(62,188,250),width=2)
        if lo<0<hi:
            zy=gy+gh-4-int((0-lo)/(hi-lo)*(gh-18))
            d.line([(gx+3,zy),(gx+gw-3,zy)],fill=(85,85,105),width=1)

    # ── ESI legend ────────────────────────────────────────────
    def _legend(self, d):
        sx = L.W - L.SIDEBAR_W + 4
        lx,ly = sx, L.GRID_Y+248
        lw = L.SIDEBAR_W - 8
        lh = 5*22+18
        d.rectangle([(lx,ly),(lx+lw,ly+lh)],fill=(10,15,32),outline=(44,55,82))
        self._t(d,"ESI SCALE",lx+lw//2,ly+9,8,(158,170,210),True,True)
        rows={1:"1 Critical",2:"2 Emergent",3:"3 Urgent",
              4:"4 Less-Urg",5:"5 Non-Urg"}
        for sev,lbl in rows.items():
            by_=ly+18+(sev-1)*22
            d.rectangle([(lx+4,by_+2),(lx+18,by_+16)],fill=SEV.get(sev,(100,100,100)))
            d.line([(lx+4,by_+2),(lx+18,by_+2)],fill=lt(SEV.get(sev,(100,100,100)),1.25),width=1)
            self._t(d,lbl,lx+22,by_+9,8,(200,208,228))

    # ── text helper ───────────────────────────────────────────
    def _t(self, d, txt, x, y, sz=12, col=(255,255,255), center=False, bold=False):
        try:
            font=ImageFont.truetype(
                f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'Bold' if bold else ''}.ttf",sz)
        except Exception:
            try:
                font=ImageFont.truetype(
                    f"/usr/share/fonts/truetype/liberation/"
                    f"LiberationSans-{'Bold' if bold else 'Regular'}.ttf",sz)
            except Exception:
                font=ImageFont.load_default()
        if center:
            bb=d.textbbox((0,0),txt,font=font)
            x-=(bb[2]-bb[0])//2; y-=(bb[3]-bb[1])//2
        d.text((x,y),txt,fill=col,font=font)


# ═══════════════════════════════════════════════════════════════
#  Renderer classes
# ═══════════════════════════════════════════════════════════════
class PILRenderer:
    def __init__(self, save_dir="results/frames"):
        if not PIL_OK: raise ImportError("Pillow required.")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir=save_dir; self.frame_idx=0
        self.reward_history: list=[]
        self._p=ScenePainter()

    def update(self, env, action=0, reward=0.0):
        self.reward_history.append(reward)
        img=self._p.render(env,action,reward,self.reward_history)
        img.save(os.path.join(self.save_dir,f"frame_{self.frame_idx:05d}.png"))
        self.frame_idx+=1
        return img

    def close(self): pass


class PygameRenderer:
    W,H=ScenePainter.W,ScenePainter.H
    def __init__(self):
        if not PG_OK:  raise ImportError("pygame not installed.")
        if not PIL_OK: raise ImportError("Pillow required.")
        pygame.init()
        self._screen=pygame.display.set_mode((self.W,self.H))
        pygame.display.set_caption("Medical Triage RL – Hospital Ward")
        self._clock=pygame.time.Clock()
        self._painter=ScenePainter()
        self.reward_history: list=[]

    def update(self, env, action=0, reward=0.0):
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: self.close(); return
        self.reward_history.append(reward)
        img=self._painter.render(env,action,reward,self.reward_history)
        surf=pygame.image.fromstring(img.tobytes(),(self.W,self.H),"RGB")
        self._screen.blit(surf,(0,0)); pygame.display.flip()
        self._clock.tick(12)

    def close(self): pygame.quit()


class Panda3DRenderer(ShowBase if P3D_OK else object):
    W,H=ScenePainter.W,ScenePainter.H
    def __init__(self):
        if not P3D_OK: raise ImportError("panda3d not installed.")
        ShowBase.__init__(self)
        props=WindowProperties()
        props.setTitle("Medical Triage RL – Hospital Ward")
        props.setSize(self.W,self.H)
        self.win.requestProperties(props)
        self.setBackgroundColor(0,0,0,1); self.disableMouse()
        cm=CardMaker("sc"); cm.setFrame(-1,1,1,-1)
        self._card=self.render2d.attachNewNode(cm.generate())
        self._tex=Texture("ward")
        self._tex.setup2dTexture(self.W,self.H,Texture.T_unsigned_byte,Texture.F_rgb)
        self._card.setTexture(self._tex)
        self._painter=ScenePainter(); self.reward_history: list=[]

    def update(self, env, action=0, reward=0.0):
        self.reward_history.append(reward)
        img=self._painter.render(env,action,reward,self.reward_history)
        arr=np.array(img)
        pnm=PNMImage(self.W,self.H)
        for y in range(self.H):
            for x in range(self.W):
                r,g,b=int(arr[y,x,0]),int(arr[y,x,1]),int(arr[y,x,2])
                pnm.setXel(x,y,r/255,g/255,b/255)
        self._tex.load(pnm); self.taskMgr.step()

    def close(self): self.destroy()


class TerminalRenderer:
    A={1:"\033[91m",2:"\033[33m",3:"\033[93m",4:"\033[94m",5:"\033[92m"}; R="\033[0m"
    def update(self,env,action=0,reward=0.0):
        print("\033[H\033[J",end="")
        print("╔"+"═"*64+"╗")
        print(f"║  MEDICAL TRIAGE RL  │  Step {env.step_count:3d}/200"+" "*33+"║")
        print(f"║  Beds:{env.beds_available}  Staff:{env.staff_available}  Equip:{env.equipment_available}"
              f"  Treated:{env.total_treated}  Deaths:{env.total_deaths}"+" "*18+"║")
        print("╠"+"─"*64+"╣")
        for i in range(MAX_PATIENTS):
            p=env.patients[i]
            if p[0]==0: continue
            sev=int(p[0]); col=self.A.get(sev,"")
            stat=("DECEASED" if p[3]==0 else "TREATED " if p[2]==1 else "WAITING ")
            mark="▶▶ " if action==i+1 else "   "
            print(f"║ {mark}P{i+1}  {col}{SEV_NAME.get(sev,'?'):9s}{self.R}"
                  f"  W:{int(p[1]):3d}s  Det:{int(p[4])}  {stat}  ║")
        print("╠"+"═"*64+"╣")
        act="HOLD" if action==0 else f"TREAT P{action}"
        print(f"║  {act:<28}  Reward:{reward:+.2f}"+" "*28+"║")
        print("╚"+"═"*64+"╝")
    def close(self): pass


class TriageRenderer:
    """Auto-selects: Pygame > Panda3D > PIL > Terminal."""
    def __init__(self):
        if PG_OK and PIL_OK:
            try: self._r=PygameRenderer(); self._mode="pygame"; return
            except Exception as e: print(f"[Renderer] pygame: {e}")
        if P3D_OK and PIL_OK:
            try: self._r=Panda3DRenderer(); self._mode="panda3d"; return
            except Exception as e: print(f"[Renderer] panda3d: {e}")
        if PIL_OK:
            self._r=PILRenderer(save_dir="results/frames"); self._mode="pil"
            print("[Renderer] PIL – frames → results/frames/"); return
        self._r=TerminalRenderer(); self._mode="terminal"

    def update(self,env,action=0,reward=0.0): self._r.update(env,action,reward)
    def close(self): self._r.close()

    @property
    def mode(self): return self._mode


def get_renderer(): return TriageRenderer()


# ── preview generator ──────────────────────────────────────────
if __name__=="__main__":
    class MockEnv:
        patients=np.zeros((8,5),dtype=np.float32)
        beds_available=3; staff_available=2; equipment_available=1
        step_count=47; total_treated=5; total_deaths=1

    env=MockEnv()
    env.patients[0]=[1.0,18.0,0.0,1.0,2.0]
    env.patients[1]=[2.0, 9.0,0.0,1.0,0.0]
    env.patients[2]=[3.0, 5.0,0.0,1.0,0.0]
    env.patients[3]=[5.0, 2.0,0.0,1.0,0.0]
    env.patients[4]=[1.0,22.0,1.0,1.0,0.0]
    env.patients[5]=[4.0, 7.0,0.0,1.0,1.0]
    env.patients[6]=[2.0,14.0,0.0,0.0,0.0]
    env.patients[7]=[0.0, 0.0,0.0,0.0,0.0]

    os.makedirs("results/frames",exist_ok=True)
    r=PILRenderer(save_dir="results/frames")
    r.reward_history=[-4,-1,2,5,8,5,10,8,3,7,9,12,8,6,10,7,11]
    for act,rew in [(0,-1.0),(1,10.0),(2,5.0),(3,2.0)]:
        img=r.update(env,action=act,reward=rew)
        img.save(f"results/frames/demo_a{act}.png")
        print(f"  action={act} reward={rew:+.1f}")
    print("Done – results/frames/")
