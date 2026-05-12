"""
visualizer.py
-------------
Interactive K Shortest Simple Paths visualizer (pygame).
Compares Yen's algorithm vs. Suurballe–Bhandari (SB) algorithm
with live time/memory/operation stats.

Controls
~~~~~~~~
  Left-click canvas  : Add node (node mode)
  Left-click node    : Select node / start edge (edge mode)
  Right-click node   : Delete node
  Right-click edge   : Delete edge
  Drag node          : Reposition
  S  (node selected) : Set as Source
  T  (node selected) : Set as Target
  Y                  : Run Yen's algorithm
  B                  : Run SB (Suurballe–Bhandari) algorithm
  N                  : Switch to node-add mode
  G                  : Switch to edge-add mode (click src→dst, type weight, Enter)
  D                  : Switch to delete mode
  1-8 / Tab          : Highlight path by index
  +/-                : Increase / decrease k
  R                  : Reset to default graph
  Q / Escape         : Quit
"""

from __future__ import annotations
import pygame
import sys
import math
import time
from typing import Optional

from graph import Graph, Node, Edge, path_cost
from yen import yen_k_shortest
from sb  import sb_k_shortest

# ──────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────
BG           = (248, 247, 243)
PANEL_BG     = (252, 251, 249)
PANEL_BORDER = (210, 208, 200)
DIVIDER      = (228, 226, 218)

NODE_FILL    = (241, 239, 232)
NODE_BORDER  = (136, 135, 128)
NODE_SEL_RNG = (83,  74, 183)
SRC_FILL     = (238, 237, 254)
SRC_BORDER   = (83,  74, 183)
TGT_FILL     = (225, 245, 238)
TGT_BORDER   = (15,  110, 86)

EDGE_COL     = (180, 178, 169)

YEN_COL      = (83,  74, 183)   # purple
YEN_LIGHT    = (220, 216, 248)
SB_COL       = (194, 100, 30)   # amber / orange
SB_LIGHT     = (252, 228, 196)

TEXT_PRI     = (44,  44,  42)
TEXT_SEC     = (95,  94,  90)
TEXT_HINT    = (160, 158, 150)
WHITE        = (255, 255, 255)
BLACK        = (0,   0,   0)

STAT_BG_YEN  = (240, 238, 254)
STAT_BG_SB   = (254, 240, 220)
STAT_BORDER  = (200, 198, 190)

# ──────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────
W, H       = 1160, 700
PANEL_W    = 380
CANVAS_W   = W - PANEL_W
NODE_R     = 20

# ──────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────

def draw_arrow(surf, color, p1, p2, lw=2, r=NODE_R):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    sx, sy = p1[0] + ux * r, p1[1] + uy * r
    ex, ey = p2[0] - ux * (r + 4), p2[1] - uy * (r + 4)
    if math.hypot(ex - sx, ey - sy) < 6:
        return
    pygame.draw.line(surf, color, (int(sx), int(sy)), (int(ex), int(ey)), lw)
    angle = math.atan2(ey - sy, ex - sx)
    ah = 11
    pts = [
        (int(ex), int(ey)),
        (int(ex - ah * math.cos(angle - 0.45)), int(ey - ah * math.sin(angle - 0.45))),
        (int(ex - ah * math.cos(angle + 0.45)), int(ey - ah * math.sin(angle + 0.45))),
    ]
    pygame.draw.polygon(surf, color, pts)


def draw_node(surf, n: Node, border_col, fill_col, font, ring=False, ring_col=None):
    x, y = int(n.x), int(n.y)
    if ring:
        pygame.draw.circle(surf, ring_col or border_col, (x, y), NODE_R + 5)
        pygame.draw.circle(surf, fill_col, (x, y), NODE_R + 3)
    pygame.draw.circle(surf, fill_col, (x, y), NODE_R)
    pygame.draw.circle(surf, border_col, (x, y), NODE_R, 2)
    t = font.render(n.label, True, TEXT_PRI)
    surf.blit(t, t.get_rect(center=(x, y)))


def edge_midpoint(a: Node, b: Node, offset=14):
    mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
    dx, dy = b.x - a.x, b.y - a.y
    ln = math.hypot(dx, dy) or 1
    return mx + (-dy / ln) * offset, my + (dx / ln) * offset


# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────

class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("K Shortest Simple Paths — Yen's vs Suurballe–Bhandari")
        self.clock = pygame.time.Clock()

        self.font       = pygame.font.SysFont("Arial", 14)
        self.font_sm    = pygame.font.SysFont("Arial", 12)
        self.font_bold  = pygame.font.SysFont("Arial", 14, bold=True)
        self.font_title = pygame.font.SysFont("Arial", 15, bold=True)
        self.font_tiny  = pygame.font.SysFont("Arial", 11)

        self.g   = Graph()
        self.src: Optional[int] = None
        self.tgt: Optional[int] = None
        self.k   = 3

        # Results: list of (path, cost, stats) — one per algorithm
        self.yen_results: list[tuple[list[int], float, dict]] = []
        self.sb_results:  list[tuple[list[int], float, dict]] = []

        # Which path row is highlighted per algo (-1 = none)
        self.yen_hl = -1
        self.sb_hl  = -1

        self.mode = "node"          # "node" | "edge" | "delete"
        self.selected: Optional[int] = None
        self.pending_edge_src: Optional[int] = None
        self.drag: Optional[int] = None
        self.drag_ox = self.drag_oy = 0
        self.msg = "Add nodes by clicking the canvas, then set Source (S) and Target (T)."

        # Weight input popup state
        self.weight_input_active = False   # True while typing a weight
        self.weight_input_str    = ""      # digits typed so far
        self.weight_input_dst    = None    # destination node id waiting for weight
        self.weight_input_pos    = (0, 0)  # screen position to draw the popup

        self._build_default()

    # ── default graph ──────────────────────────────────────────────────────

    def _build_default(self):
        self.g = Graph()
        cx, cy = CANVAS_W // 2, H // 2
        pts = [
            (cx - 290, cy),       # A
            (cx - 150, cy - 120), # B
            (cx - 150, cy + 120), # C
            (cx + 20,  cy - 120), # D
            (cx + 20,  cy + 120), # E
            (cx + 170, cy),       # F
            (cx - 60,  cy),       # G
        ]
        for (x, y) in pts:
            self.g.add_node(x, y)
        ids = [n.id for n in self.g.nodes]

        def ae(u, v, w):
            self.g.add_edge(ids[u], ids[v], w)

        ae(0, 1, 2); ae(0, 2, 4); ae(1, 3, 3); ae(1, 6, 2)
        ae(2, 6, 1); ae(2, 4, 2); ae(3, 5, 2); ae(4, 5, 3)
        ae(6, 3, 3); ae(6, 4, 4); ae(6, 5, 6); ae(0, 6, 7)

        self.src = ids[0]
        self.tgt = ids[5]
        self.yen_results = []
        self.sb_results  = []
        self.yen_hl = self.sb_hl = -1
        self.msg = "Press Y for Yen's, B for Suurballe–Bhandari.  +/- to change k."

    # ── utilities ─────────────────────────────────────────────────────────

    def node_at(self, x, y) -> Optional[Node]:
        return next((n for n in self.g.nodes if math.hypot(n.x - x, n.y - y) <= NODE_R + 3), None)

    def edge_near(self, x, y) -> int:
        for i, e in enumerate(self.g.edges):
            a = self.g.node_by_id(e.u)
            b = self.g.node_by_id(e.v)
            if not a or not b:
                continue
            mx, my = edge_midpoint(a, b)
            if math.hypot(mx - x, my - y) < 14:
                return i
        return -1

    def _clear_results(self):
        self.yen_results = []
        self.sb_results  = []
        self.yen_hl = self.sb_hl = -1

    # ── algorithms ────────────────────────────────────────────────────────

    def run_yen(self):
        if not self._check_src_tgt():
            return
        self.yen_results = yen_k_shortest(self.g, self.src, self.tgt, self.k)
        self.yen_hl = 0 if self.yen_results else -1
        n = len(self.yen_results)
        self.msg = f"Yen's: {n} path(s) found.  Keys 1–{n} to highlight." if n else "Yen's: no path found."

    def run_sb(self):
        if not self._check_src_tgt():
            return
        self.sb_results = sb_k_shortest(self.g, self.src, self.tgt, self.k)
        self.sb_hl = 0 if self.sb_results else -1
        n = len(self.sb_results)
        self.msg = f"SB: {n} path(s) found.  Keys 1–{n} to highlight." if n else "SB: no path found."

    def _check_src_tgt(self) -> bool:
        if self.src is None or self.tgt is None:
            self.msg = "Set source (select node + S) and target (select node + T) first."
            return False
        if self.src == self.tgt:
            self.msg = "Source and target must be different nodes."
            return False
        return True

    # ── events ────────────────────────────────────────────────────────────

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if self.weight_input_active:
                    self._on_weight_key(event)
                else:
                    self._on_key(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not self.weight_input_active and event.pos[0] < CANVAS_W:
                    self._on_click(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.drag = None
            elif event.type == pygame.MOUSEMOTION:
                if not self.weight_input_active:
                    self._on_drag(event)
        return True

    def _on_weight_key(self, ev):
        """Handle keystrokes while the weight-input popup is open."""
        k = ev.key
        if k == pygame.K_RETURN or k == pygame.K_KP_ENTER:
            # Confirm — add the edge with typed weight (default 1 if empty)
            try:
                w = float(self.weight_input_str) if self.weight_input_str else 1.0
                if w <= 0:
                    self.msg = "Weight must be > 0. Try again."
                    self.weight_input_str = ""
                    return
            except ValueError:
                self.msg = "Invalid number. Try again."
                self.weight_input_str = ""
                return
            src_node = self.g.node_by_id(self.pending_edge_src)
            dst_node = self.g.node_by_id(self.weight_input_dst)
            self.g.add_edge(self.pending_edge_src, self.weight_input_dst, w)
            self.msg = f"Edge {src_node.label}→{dst_node.label} (w={w}) added."
            self._finish_weight_input()
        elif k == pygame.K_ESCAPE:
            # Cancel
            self.msg = "Edge cancelled. Click a source node to start again."
            self._finish_weight_input(cancel=True)
        elif k == pygame.K_BACKSPACE:
            self.weight_input_str = self.weight_input_str[:-1]
        elif ev.unicode in "0123456789.":
            # Allow one decimal point only
            if ev.unicode == "." and "." in self.weight_input_str:
                return
            if len(self.weight_input_str) < 6:
                self.weight_input_str += ev.unicode

    def _finish_weight_input(self, cancel=False):
        self.weight_input_active = False
        self.weight_input_str    = ""
        self.weight_input_dst    = None
        if not cancel:
            self.pending_edge_src = None
            self._clear_results()

    def _on_key(self, ev):
        k = ev.key
        if k in (pygame.K_q, pygame.K_ESCAPE):
            pygame.quit(); sys.exit()
        elif k == pygame.K_r:
            self._build_default()
        elif k == pygame.K_y:
            self.run_yen()
        elif k == pygame.K_b:
            self.run_sb()
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.k = min(self.k + 1, 12)
            self._clear_results()
            self.msg = f"k = {self.k}. Press Y or B to run."
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.k = max(self.k - 1, 1)
            self._clear_results()
            self.msg = f"k = {self.k}. Press Y or B to run."
        elif k == pygame.K_TAB:
            # Cycle highlight in whichever list was last run.
            if self.yen_results:
                self.yen_hl = (self.yen_hl + 1) % len(self.yen_results)
            if self.sb_results:
                self.sb_hl = (self.sb_hl + 1) % len(self.sb_results)
        elif k == pygame.K_s and self.selected is not None:
            self.src = self.selected
            self._clear_results()
            lbl = self.g.node_by_id(self.src)
            self.msg = f"Source → node {lbl.label if lbl else '?'}"
        elif k == pygame.K_t and self.selected is not None:
            self.tgt = self.selected
            self._clear_results()
            lbl = self.g.node_by_id(self.tgt)
            self.msg = f"Target → node {lbl.label if lbl else '?'}"
        elif k == pygame.K_n:
            self.mode = "node"; self.pending_edge_src = None
            self.msg = "Node mode: click canvas to add nodes."
        elif k == pygame.K_g:
            self.mode = "edge"; self.pending_edge_src = None
            self.msg = "Edge mode: click source node, then destination node."
        elif k == pygame.K_d:
            self.mode = "delete"; self.pending_edge_src = None
            self.msg = "Delete mode: right-click node or edge weight."
        elif pygame.K_1 <= k <= pygame.K_9:
            idx = k - pygame.K_1
            if idx < len(self.yen_results):
                self.yen_hl = idx
            if idx < len(self.sb_results):
                self.sb_hl = idx

    def _on_click(self, ev):
        x, y = ev.pos
        n = self.node_at(x, y)

        # Right-click = delete
        if ev.button == 3:
            if n:
                if n.id == self.src: self.src = None
                if n.id == self.tgt: self.tgt = None
                self.g.remove_node(n.id)
                self._clear_results()
                self.pending_edge_src = None
            else:
                ei = self.edge_near(x, y)
                if ei >= 0:
                    self.g.remove_edge(ei)
                    self._clear_results()
            return

        if ev.button == 1:
            if n:
                self.selected = n.id
                if self.mode == "edge":
                    if self.pending_edge_src is None:
                        self.pending_edge_src = n.id
                        self.msg = f"Edge from {n.label} — now click destination node."
                    elif self.pending_edge_src != n.id:
                        # Open weight-input popup instead of random weight
                        self.weight_input_active = True
                        self.weight_input_str    = ""
                        self.weight_input_dst    = n.id
                        # Place popup near the midpoint between the two nodes
                        a = self.g.node_by_id(self.pending_edge_src)
                        self.weight_input_pos = (
                            int((a.x + n.x) / 2),
                            int((a.y + n.y) / 2) - 40,
                        )
                        self.msg = f"Type weight for {a.label}→{n.label}, then press Enter."
                else:
                    self.drag    = n.id
                    self.drag_ox = x - n.x
                    self.drag_oy = y - n.y
            else:
                self.selected = None
                self.pending_edge_src = None
                if self.mode == "node":
                    nn = self.g.add_node(x, y)
                    self.msg = f"Node {nn.label} added. Select + press S or T to assign roles."

    def _on_drag(self, ev):
        if self.drag is None:
            return
        n = self.g.node_by_id(self.drag)
        if n:
            n.x = max(NODE_R + 5, min(CANVAS_W - NODE_R - 5, ev.pos[0] - self.drag_ox))
            n.y = max(NODE_R + 5, min(H - NODE_R - 40,       ev.pos[1] - self.drag_oy))

    # ── rendering ─────────────────────────────────────────────────────────

    def draw(self):
        self.screen.fill(BG)
        self._draw_canvas()
        self._draw_panel()
        pygame.display.flip()

    # ── canvas ────────────────────────────────────────────────────────────

    def _hl_edge_sets(self):
        """Return highlighted edge sets for yen and sb separately."""
        def edges_of(results, idx):
            if idx < 0 or idx >= len(results):
                return set(), set()
            path, _, _ = results[idx]
            es = {(path[i], path[i+1]) for i in range(len(path)-1)}
            return es, set(path)
        ye, yn = edges_of(self.yen_results, self.yen_hl)
        se, sn = edges_of(self.sb_results,  self.sb_hl)
        return ye, yn, se, sn

    def _draw_canvas(self):
        surf = self.screen
        g    = self.g
        ye, yn, se, sn = self._hl_edge_sets()

        # ── edges ─────────────────────────────────────────────────────────
        for e in g.edges:
            a = g.node_by_id(e.u)
            b = g.node_by_id(e.v)
            if not a or not b:
                continue
            key = (e.u, e.v)
            in_yen = key in ye
            in_sb  = key in se

            if in_yen and in_sb:
                col, lw = YEN_COL, 4      # both — draw purple (yen)
            elif in_yen:
                col, lw = YEN_COL, 3
            elif in_sb:
                col, lw = SB_COL, 3
            else:
                col, lw = EDGE_COL, 1

            draw_arrow(surf, col, (a.x, a.y), (b.x, b.y), lw)

            # Weight label
            mx, my = edge_midpoint(a, b)
            wt = self.font_tiny.render(str(e.w), True, col if (in_yen or in_sb) else TEXT_HINT)
            wr = wt.get_rect(center=(int(mx), int(my)))
            bg = pygame.Surface((wr.w + 4, wr.h + 2), pygame.SRCALPHA)
            bg.fill((248, 247, 243, 210))
            surf.blit(bg, (wr.x - 2, wr.y - 1))
            surf.blit(wt, wr)

        # Pending edge ghost line
        if self.pending_edge_src is not None:
            a = g.node_by_id(self.pending_edge_src)
            if a:
                mx, my = pygame.mouse.get_pos()
                if mx < CANVAS_W:
                    pygame.draw.line(surf, (*YEN_COL, 80), (int(a.x), int(a.y)), (mx, my), 1)

        # ── nodes ─────────────────────────────────────────────────────────
        for n in g.nodes:
            is_src = n.id == self.src
            is_tgt = n.id == self.tgt
            is_sel = n.id == self.selected
            in_yen = n.id in yn
            in_sb  = n.id in sn

            if is_src:
                fc, bc = SRC_FILL, SRC_BORDER
            elif is_tgt:
                fc, bc = TGT_FILL, TGT_BORDER
            elif in_yen and in_sb:
                fc, bc = YEN_LIGHT, YEN_COL
            elif in_yen:
                fc, bc = YEN_LIGHT, YEN_COL
            elif in_sb:
                fc, bc = SB_LIGHT, SB_COL
            else:
                fc, bc = NODE_FILL, NODE_BORDER

            draw_node(surf, n, bc, fc, self.font, ring=is_sel, ring_col=NODE_SEL_RNG)

            # S / T badge
            bx, by = int(n.x) + NODE_R, int(n.y) - NODE_R
            if is_src:
                s = self.font_tiny.render("S", True, SRC_BORDER)
                surf.blit(s, (bx - s.get_width(), by - s.get_height()))
            if is_tgt:
                s = self.font_tiny.render("T", True, TGT_BORDER)
                surf.blit(s, (bx - s.get_width(), by - s.get_height()))

        # ── status bar ────────────────────────────────────────────────────
        mode_str = {"node": "N: add node", "edge": "G: add edge", "delete": "D: delete"}[self.mode]
        ms = self.font_tiny.render(mode_str, True, TEXT_HINT)
        surf.blit(ms, (10, H - 38))
        msg = self.font_sm.render(self.msg, True, TEXT_SEC)
        surf.blit(msg, (10, H - 22))

        # ── weight input popup ────────────────────────────────────────────
        if self.weight_input_active:
            self._draw_weight_popup(surf)

    def _draw_weight_popup(self, surf):
        """Draw the floating weight-input box near the new edge midpoint."""
        px, py = self.weight_input_pos
        # Clamp so it never goes off-canvas
        px = max(60, min(CANVAS_W - 60, px))
        py = max(30, min(H - 80,        py))

        box_w, box_h = 180, 64
        bx = px - box_w // 2
        by = py - box_h // 2

        # Shadow
        shadow = pygame.Surface((box_w + 4, box_h + 4), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 40))
        surf.blit(shadow, (bx - 2, by + 3))

        # Box
        pygame.draw.rect(surf, (255, 255, 255), (bx, by, box_w, box_h), border_radius=8)
        pygame.draw.rect(surf, YEN_COL,         (bx, by, box_w, box_h), 2, border_radius=8)

        # Label
        lbl = self.font_bold.render("Edge weight:", True, TEXT_PRI)
        surf.blit(lbl, (bx + 10, by + 8))

        # Input field
        field_rect = pygame.Rect(bx + 10, by + 28, box_w - 20, 24)
        pygame.draw.rect(surf, (245, 244, 240), field_rect, border_radius=4)
        pygame.draw.rect(surf, YEN_COL,         field_rect, 1, border_radius=4)

        display = self.weight_input_str if self.weight_input_str else ""
        val_surf = self.font_bold.render(display, True, TEXT_PRI)
        surf.blit(val_surf, (field_rect.x + 6, field_rect.y + 4))

        # Blinking cursor
        if (pygame.time.get_ticks() // 500) % 2 == 0:
            cx = field_rect.x + 6 + val_surf.get_width() + 1
            pygame.draw.line(surf, YEN_COL, (cx, field_rect.y + 4), (cx, field_rect.y + 18), 2)

        # Hint
        hint = self.font_tiny.render("Enter to confirm · Esc to cancel", True, TEXT_HINT)
        surf.blit(hint, hint.get_rect(centerx=px, top=by + box_h + 4))

    # ── panel ─────────────────────────────────────────────────────────────

    def _draw_panel(self):
        surf = self.screen
        px   = CANVAS_W

        # Background
        pygame.draw.rect(surf, PANEL_BG, (px, 0, PANEL_W, H))
        pygame.draw.line(surf, PANEL_BORDER, (px, 0), (px, H), 1)

        pad = 14
        y   = 12

        # ── Title row ─────────────────────────────────────────────────────
        title = self.font_title.render("K Shortest Simple Paths", True, TEXT_PRI)
        surf.blit(title, (px + pad, y))
        y += title.get_height() + 2

        sn = self.g.node_by_id(self.src)
        tn = self.g.node_by_id(self.tgt)
        sub = self.font_sm.render(
            f"k = {self.k}   Source: {sn.label if sn else '—'}   Target: {tn.label if tn else '—'}",
            True, TEXT_SEC
        )
        surf.blit(sub, (px + pad, y))
        y += sub.get_height() + 8
        pygame.draw.line(surf, DIVIDER, (px + pad, y), (px + PANEL_W - pad, y), 1)
        y += 10

        # ── Yen's column ──────────────────────────────────────────────────
        y = self._draw_algo_section(
            surf, px, pad, y,
            label="Yen's Algorithm",
            key_hint="Y",
            results=self.yen_results,
            hl_idx=self.yen_hl,
            col=YEN_COL,
            light=YEN_LIGHT,
            stat_bg=STAT_BG_YEN,
            set_hl=lambda i: setattr(self, 'yen_hl', i),
        )

        y += 6
        pygame.draw.line(surf, DIVIDER, (px + pad, y), (px + PANEL_W - pad, y), 1)
        y += 10

        # ── SB column ─────────────────────────────────────────────────────
        y = self._draw_algo_section(
            surf, px, pad, y,
            label="Suurballe–Bhandari",
            key_hint="B",
            results=self.sb_results,
            hl_idx=self.sb_hl,
            col=SB_COL,
            light=SB_LIGHT,
            stat_bg=STAT_BG_SB,
            set_hl=lambda i: setattr(self, 'sb_hl', i),
        )

        y += 6
        pygame.draw.line(surf, DIVIDER, (px + pad, y), (px + PANEL_W - pad, y), 1)
        y += 8

        # ── keybind help ──────────────────────────────────────────────────
        helps = [
            ("Y — Yen's  |  B — SB  |  R — reset",    False),
            ("N/G/D — node / edge / delete mode",       False),
            ("S / T — set source / target",             False),
            ("+/- — change k  |  Tab — cycle path",    False),
            ("Right-click node/edge — delete",          False),
            ("Q / Esc — quit",                          False),
        ]
        for txt, bold in helps:
            f  = self.font_bold if bold else self.font_tiny
            s  = f.render(txt, True, TEXT_HINT)
            if y + s.get_height() < H - 4:
                surf.blit(s, (px + pad, y))
                y += s.get_height() + 3

    def _draw_algo_section(
        self, surf, px, pad, y,
        label, key_hint, results, hl_idx, col, light, stat_bg, set_hl
    ) -> int:
        rw = PANEL_W - pad * 2

        # Header badge
        pygame.draw.rect(surf, col, (px + pad, y, rw, 22), border_radius=4)
        hdr = self.font_bold.render(f"{label}  [{key_hint}]", True, WHITE)
        surf.blit(hdr, hdr.get_rect(center=(px + pad + rw // 2, y + 11)))
        y += 26

        if not results:
            s = self.font_tiny.render("Not run yet — press key above.", True, TEXT_HINT)
            surf.blit(s, (px + pad, y))
            return y + s.get_height() + 4

        _, _, stats = results[0]

        # ── Stats box ─────────────────────────────────────────────────────
        stat_lines = [
            ("Time complexity",   stats["time_complexity"]),
            ("Space complexity",  stats["space_complexity"]),
            ("Elapsed",           f"{stats['elapsed_ms']:.3f} ms"),
            ("Peak memory",       f"{stats['peak_memory_kb']} KB"),
            ("Dijkstra calls",    str(stats["dijkstra_calls"])),
            ("Paths found",       str(stats["paths_found"])),
        ]
        # Algorithm-specific counters
        if "heap_pushes" in stats:
            stat_lines.append(("Heap pushes",   str(stats["heap_pushes"])))
            stat_lines.append(("Heap pops",     str(stats["heap_pops"])))
        if "graph_transforms" in stats:
            stat_lines.append(("Graph transforms", str(stats["graph_transforms"])))
            stat_lines.append(("Edges reversed",   str(stats["edges_reversed"])))

        box_h = len(stat_lines) * 15 + 6
        pygame.draw.rect(surf, stat_bg, (px + pad, y, rw, box_h), border_radius=4)
        pygame.draw.rect(surf, col,     (px + pad, y, rw, box_h), 1, border_radius=4)
        sy = y + 4
        for lbl_txt, val_txt in stat_lines:
            lbl_s = self.font_tiny.render(lbl_txt + ":", True, TEXT_SEC)
            val_s = self.font_tiny.render(val_txt,        True, col)
            surf.blit(lbl_s, (px + pad + 4, sy))
            surf.blit(val_s, (px + pad + rw - val_s.get_width() - 4, sy))
            sy += 15
        y += box_h + 6

        # ── Path list ─────────────────────────────────────────────────────
        s = self.font_tiny.render(f"{len(results)} path(s) — click to highlight:", True, TEXT_SEC)
        surf.blit(s, (px + pad, y)); y += s.get_height() + 2

        for i, (path, cost, _) in enumerate(results):
            row_h = 30
            if y + row_h > H - 80:
                more = self.font_tiny.render(f"… {len(results)-i} more", True, TEXT_HINT)
                surf.blit(more, (px + pad, y)); y += more.get_height() + 2
                break

            is_hl = i == hl_idx
            rx = px + pad

            if is_hl:
                pygame.draw.rect(surf, light, (rx, y, rw, row_h), border_radius=5)
                pygame.draw.rect(surf, col,   (rx, y, rw, row_h), 1, border_radius=5)

            labels = " → ".join(
                self.g.node_by_id(nid).label
                for nid in path
                if self.g.node_by_id(nid)
            )
            num_s  = self.font_bold.render(f"#{i+1}", True, col if is_hl else TEXT_HINT)
            path_s = self.font_tiny.render(labels, True, TEXT_PRI if is_hl else TEXT_SEC)
            cost_s = self.font_tiny.render(f"cost {cost:.1f}", True, col if is_hl else TEXT_HINT)

            surf.blit(num_s,  (rx + 4, y + 4))
            surf.blit(path_s, (rx + 4, y + 17))
            surf.blit(cost_s, (rx + rw - cost_s.get_width() - 4, y + 10))

            # Invisible click target stored for mouse interaction
            # (We use keyboard shortcuts for highlighting in this version)
            y += row_h + 2

        return y

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self):
        while True:
            if not self.handle_events():
                break
            self.draw()
            self.clock.tick(60)
        pygame.quit()


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    App().run()