"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore
from pyqtgraph import Point
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu

from . import masks


class TraceBox(pg.PlotItem):

    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True,
                 invertY=False, enableMenu=True, name=None, invertX=False):
        super(TraceBox, self).__init__()
        self.parent = parent

    def mouseDoubleClickEvent(self, ev):
        self.zoom_plot()

    def zoom_plot(self):
        self.setXRange(0, self.parent.Fcell.shape[1])
        self.setYRange(self.parent.fmin, self.parent.fmax)
        self.parent.show()


class ViewBox(pg.ViewBox):

    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True,
                 invertY=False, enableMenu=True, name=None, invertX=False):
        #pg.ViewBox.__init__(self, border, lockAspect, enableMouse,
        #invertY, enableMenu, name, invertX)
        super(ViewBox, self).__init__()
        self.border = fn.mkPen(border)
        if enableMenu:
            self.menu = ViewBoxMenu(self)
        self.name = name
        self.parent = parent

        # set state
        self.state["enableMenu"] = enableMenu
        self.state["yInverted"] = invertY

    def mouseDoubleClickEvent(self, ev):
        if self.parent.loaded:
            self.zoom_plot()

    def mouseClickEvent(self, ev):
        if self.parent.loaded:
            pos = self.mapSceneToView(ev.scenePos())
            posy = int(pos.x())
            posx = int(pos.y())
            if self.name == "plot1":
                iplot = 0
            else:
                iplot = 1
            if posy >= 0 and posx >= 0 and posy <= self.parent.Lx and posx <= self.parent.Ly:
                if self.parent.merged_view:
                    visible_plot = "plot1" if self.parent.merged_view_mode == 0 else "plot2"
                    if self.name != visible_plot:
                        return
                    ichosen = merged_roi_at_pixel(self.parent, posx, posy)
                else:
                    ichosen = int(self.parent.rois["iROI"][iplot, 0, posx, posy])
                if ichosen < 0:
                    if ev.button() == QtCore.Qt.RightButton and self.menuEnabled():
                        self.raiseContextMenu(ev)
                    return
                else:
                    if ev.button() == QtCore.Qt.RightButton:
                        self.parent.imerge = [ichosen]
                        self.parent.ichosen = ichosen
                        masks.flip_plot(self.parent)
                        self.parent.imerge = []
                        self.parent.ichosen = -1
                        self.parent.update_plot()
                    else:
                        merged = False
                        if ev.modifiers() == QtCore.Qt.ShiftModifier or ev.modifiers(
                        ) == QtCore.Qt.ControlModifier:
                            if self.parent.iscell[self.parent.imerge[
                                    0]] == self.parent.iscell[ichosen]:
                                if ichosen not in self.parent.imerge:
                                    self.parent.imerge.append(ichosen)
                                    self.parent.ichosen = ichosen
                                    merged = True
                                elif ichosen in self.parent.imerge and len(
                                        self.parent.imerge) > 1:
                                    self.parent.imerge.remove(ichosen)
                                    self.parent.ichosen = self.parent.imerge[0]
                                    merged = True
                        if not merged:
                            self.parent.imerge = [ichosen]
                            self.parent.ichosen = ichosen

                    if self.parent.isROI:
                        self.parent.ROI_remove()
                    if not self.parent.sizebtns.button(1).isChecked():
                        for btn in self.parent.topbtns.buttons():
                            if btn.isChecked():
                                btn.setStyleSheet(self.parent.styleUnpressed)
                    self.parent.update_plot()

    def zoom_plot(self):
        self.setXRange(0, self.parent.ops["Lx"])
        self.setYRange(0, self.parent.ops["Ly"])
        self.parent.update_roi_view_sync(source_view=self)
        self.parent.show()


def init_range(parent):
    parent.p1.setXRange(0, parent.ops["Lx"])
    parent.p1.setYRange(0, parent.ops["Ly"])
    parent.p2.setXRange(0, parent.ops["Lx"])
    parent.p2.setYRange(0, parent.ops["Ly"])
    parent.p3.setLimits(xMin=0, xMax=parent.Fcell.shape[1])
    parent.trange = np.arange(0, parent.Fcell.shape[1])


def ROI_index(settings, stat):
    """matrix Ly x Lx where each pixel is an ROI index (-1 if no ROI present)"""
    ncells = len(stat) - 1
    Ly = settings["Ly"]
    Lx = settings["Lx"]
    iROI = -1 * np.ones((Ly, Lx), dtype=np.int32)
    for n in range(ncells):
        ypix = stat[n]["ypix"][~stat[n]["overlap"]]
        if ypix is not None:
            xpix = stat[n]["xpix"][~stat[n]["overlap"]]
            iROI[ypix, xpix] = n
    return iROI

def _outline_contains_pixel(stat, ypix, xpix, width=1):
    if "ycirc" not in stat or "xcirc" not in stat:
        return False
    ycirc = np.asarray(stat["ycirc"])
    xcirc = np.asarray(stat["xcirc"])
    return np.any(
        (np.abs(ycirc - ypix) <= width)
        & (np.abs(xcirc - xpix) <= width)
    )

def merged_roi_at_pixel(parent, ypix, xpix):
    if parent.merged_view_mode == 0:
        iplot = 0
        outline_rois = np.where(~parent.iscell)[0]
    else:
        iplot = 1
        outline_rois = np.where(parent.iscell)[0]
    for n in outline_rois:
        if _outline_contains_pixel(parent.stat[n], ypix, xpix):
            return int(n)
    return int(parent.rois["iROI"][iplot, 0, ypix, xpix])