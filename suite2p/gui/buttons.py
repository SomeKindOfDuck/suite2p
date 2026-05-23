"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QPushButton, QButtonGroup, QLabel, QLineEdit, QCheckBox

def make_selection(parent):
    """ buttons to draw a square on view """
    parent.topbtns = QButtonGroup()
    ql = QLabel("select cells")
    ql.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
    parent.l0.addWidget(ql, 0, 2, 1, 2)
    pos = [2, 3, 4]
    for b in range(3):
        btn = TopButton(b, parent)
        btn.setFont(QtGui.QFont("Arial", 8))
        parent.topbtns.addButton(btn, b)
        parent.l0.addWidget(btn, 0, (pos[b]) * 2, 1, 2)
        btn.setEnabled(False)
    parent.topbtns.setExclusive(True)
    parent.isROI = False
    parent.ROIplot = 0
    ql = QLabel("n=")
    ql.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    ql.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
    parent.l0.addWidget(ql, 0, 10, 1, 1)
    parent.topedit = QLineEdit(parent)
    parent.topedit.setValidator(QtGui.QIntValidator(0, 500))
    parent.topedit.setText("40")
    parent.ntop = 40
    parent.topedit.setFixedWidth(35)
    parent.topedit.setAlignment(QtCore.Qt.AlignRight)
    parent.topedit.returnPressed.connect(parent.top_number_chosen)
    parent.l0.addWidget(parent.topedit, 0, 11, 1, 1)


def _sync_views_checked(parent):
    return (
        hasattr(parent, "syncViewsCheckBox")
        and parent.syncViewsCheckBox.isChecked()
    )


def _both_view_checked(parent):
    return (
        hasattr(parent, "sizebtns")
        and parent.sizebtns.button(1).isChecked()
    )


def _roi_view_sync_active(parent):
    return _sync_views_checked(parent)


def _has_roi_views(parent):
    return hasattr(parent, "p1") and hasattr(parent, "p2")


def _unlink_builtin_view_links(parent):
    """Remove pyqtgraph's one-way ViewBox links."""
    if not _has_roi_views(parent):
        return

    parent.p1.linkView(parent.p1.XAxis, view=None)
    parent.p1.linkView(parent.p1.YAxis, view=None)
    parent.p2.linkView(parent.p2.XAxis, view=None)
    parent.p2.linkView(parent.p2.YAxis, view=None)


def _mirror_view_range(parent, src, dst, force=False):
    """Copy x/y range from src ViewBox to dst ViewBox."""
    if getattr(parent, "_syncing_roi_views", False):
        return

    if not force and not _roi_view_sync_active(parent):
        return

    parent._syncing_roi_views = True
    try:
        x_range, y_range = src.viewRange()
        dst.setRange(xRange=x_range, yRange=y_range, padding=0)
    finally:
        parent._syncing_roi_views = False


def _ensure_roi_view_sync_connections(parent):
    """Connect p1 <-> p2 range synchronization only once."""
    if not _has_roi_views(parent):
        return

    if getattr(parent, "_roi_view_sync_connected", False):
        return

    parent._syncing_roi_views = False

    parent.p1.sigRangeChanged.connect(
        lambda *_: _mirror_view_range(parent, parent.p1, parent.p2)
    )
    parent.p2.sigRangeChanged.connect(
        lambda *_: _mirror_view_range(parent, parent.p2, parent.p1)
    )

    parent._roi_view_sync_connected = True


def _current_visible_view(parent):
    """Return the ViewBox whose current range should be treated as source."""
    if not _has_roi_views(parent):
        return None

    try:
        bid = parent.sizebtns.checkedId()
    except Exception:
        bid = 1

    if bid == 2:
        return parent.p2
    return parent.p1


def _update_cell_view_sync(parent, source_view=None):
    """Apply current sync setting to cells / not-cells views."""
    if not _has_roi_views(parent):
        return

    _ensure_roi_view_sync_connections(parent)

    _unlink_builtin_view_links(parent)

    if not _sync_views_checked(parent):
        if hasattr(parent, "win"):
            parent.win.show()
        parent.show()
        return

    if source_view is None:
        source_view = _current_visible_view(parent)

    if source_view is parent.p2:
        _mirror_view_range(parent, parent.p2, parent.p1, force=True)
    else:
        _mirror_view_range(parent, parent.p1, parent.p2, force=True)

    if hasattr(parent, "win"):
        parent.win.show()
    parent.show()


def _text_width(button, text):
    fm = button.fontMetrics()
    if hasattr(fm, "horizontalAdvance"):
        return fm.horizontalAdvance(text)
    return fm.width(text)


def _set_fixed_width_for_texts(button, texts, padding=28):
    width = max(_text_width(button, text) for text in texts) + padding
    button.setMinimumWidth(width)
    button.setMaximumWidth(width)


def set_merged_view_buttons_enabled(parent, enabled):
    """Enable merged-view buttons only after ROI data are loaded."""
    if not hasattr(parent, "mergedViewButton"):
        return

    parent.mergedViewButton.setEnabled(enabled)
    parent.mergedViewButton.setStyleSheet(
        parent.styleUnpressed if enabled else parent.styleInactive
    )

    # mode button is enabled only while merged view itself is active
    mode_enabled = enabled and getattr(parent, "merged_view", False)
    parent.mergedModeButton.setEnabled(mode_enabled)
    parent.mergedModeButton.setStyleSheet(
        parent.styleUnpressed if mode_enabled else parent.styleInactive
    )


# minimize view
def make_cellnotcell(parent):
    """ buttons for cell / not cell views at top """
    # number of ROIs in each image
    parent.lcell0 = QLabel("")
    parent.lcell0.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    parent.l0.addWidget(parent.lcell0, 0, 12, 1, 2)
    parent.lcell1 = QLabel("")
    parent.l0.addWidget(parent.lcell1, 0, 20, 1, 2)

    parent.sizebtns = QButtonGroup(parent)
    b = 0
    labels = [" cells", " both", " not cells"]
    for l in labels:
        btn = SizeButton(b, l, parent)
        parent.sizebtns.addButton(btn, b)
        parent.l0.addWidget(btn, 0, 14 + 2 * b, 1, 2)
        btn.setEnabled(False)
        if b == 1:
            btn.setChecked(True)
        b += 1
    parent.sizebtns.setExclusive(True)

    parent.syncViewsCheckBox = QCheckBox(" sync views")
    parent.syncViewsCheckBox.setStyleSheet("color: white;")
    parent.syncViewsCheckBox.setToolTip(
        "Synchronize pan and zoom between cells and not-cells views"
    )
    parent.syncViewsCheckBox.setChecked(False)
    parent.syncViewsCheckBox.stateChanged.connect(
        lambda _: _update_cell_view_sync(parent)
    )
    parent.l0.addWidget(parent.syncViewsCheckBox, 0, 22, 1, 4)

    parent.mergedViewButton = QPushButton(" merged")
    parent.mergedViewButton.setCheckable(True)
    parent.mergedViewButton.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
    parent.mergedViewButton.clicked.connect(lambda: toggle_merged_view(parent))
    parent.l0.addWidget(parent.mergedViewButton, 0, 26, 1, 2)

    parent.mergedModeButton = QPushButton(" cells mask")
    parent.mergedModeButton.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
    _set_fixed_width_for_texts(
        parent.mergedModeButton,
        [" cells mask", " not-cells mask"],
    )
    parent.mergedModeButton.clicked.connect(lambda: toggle_merged_mode(parent))
    parent.l0.addWidget(parent.mergedModeButton, 0, 28, 1, 2)

    # no data loaded yet, so keep them inactive like cells/both/not-cells buttons
    set_merged_view_buttons_enabled(parent, False)


def toggle_merged_view(parent):
    if not getattr(parent, "loaded", False):
        parent.mergedViewButton.setChecked(False)
        set_merged_view_buttons_enabled(parent, False)
        return

    parent.merged_view = parent.mergedViewButton.isChecked()

    if parent.merged_view:
        parent._pre_merged_size_id = parent.sizebtns.checkedId()
        parent.win.ci.layout.setColumnStretchFactor(0, 100)
        parent.win.ci.layout.setColumnStretchFactor(1, 0)
    else:
        bid = getattr(parent, "_pre_merged_size_id", 1)
        if bid < 0:
            bid = 1
        parent.sizebtns.button(bid).setChecked(True)
        parent.sizebtns.button(bid).press(parent)

    set_merged_view_buttons_enabled(parent, True)
    parent.update_plot()
    

def toggle_merged_mode(parent):
    parent.merged_view_mode = 1 - int(parent.merged_view_mode)
    if parent.merged_view_mode == 0:
        parent.mergedModeButton.setText(" cells mask")
    else:
        parent.mergedModeButton.setText(" not-cells mask")
    parent.update_plot()


def make_quadrants(parent):
    """ make quadrant buttons """
    parent.quadbtns = QButtonGroup(parent)
    for b in range(9):
        btn = QuadButton(b, " " + str(b + 1), parent)
        parent.quadbtns.addButton(btn, b)
        parent.l0.addWidget(btn, 0 + parent.quadbtns.button(b).ypos,
                            29 + parent.quadbtns.button(b).xpos, 1, 1)
        btn.setEnabled(False)
        b += 1
    parent.quadbtns.setExclusive(True)


class QuadButton(QPushButton):
    """ custom QPushButton class for quadrant plotting
        requires buttons to put into a QButtonGroup (parent.quadbtns)
         allows only 1 button to pressed at a time
    """

    def __init__(self, bid, Text, parent=None):
        super(QuadButton, self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.setMaximumWidth(22)
        self.xpos = bid % 3
        self.ypos = int(np.floor(bid / 3))
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        self.xrange = np.array([self.xpos - .15, self.xpos + 1.15]) * parent.ops["Lx"] / 3
        self.yrange = np.array([self.ypos - .15, self.ypos + 1.15]) * parent.ops["Ly"] / 3

        parent.p1.setXRange(self.xrange[0], self.xrange[1])
        parent.p1.setYRange(self.yrange[0], self.yrange[1])
        parent.p2.setXRange(self.xrange[0], self.xrange[1])
        parent.p2.setYRange(self.yrange[0], self.yrange[1])

        _update_cell_view_sync(parent, source_view=parent.p1)
        parent.show()


# size of view
class SizeButton(QPushButton):
    """ buttons to make trace box bigger or smaller """

    def __init__(self, bid, Text, parent=None):
        super(SizeButton, self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent))
        self.bid = bid
        self.show()

    def press(self, parent):
        bid = self.bid
        ts = 100

        if bid == 0:
            # cells view
            parent.win.ci.layout.setColumnStretchFactor(0, ts)
            parent.win.ci.layout.setColumnStretchFactor(1, 0)

            # not cells側から戻ってきた可能性があるので、
            # sync ONなら p2 の視野を p1 に反映
            _update_cell_view_sync(parent, source_view=parent.p2)

        elif bid == 1:
            # both view
            parent.win.ci.layout.setColumnStretchFactor(0, ts)
            parent.win.ci.layout.setColumnStretchFactor(1, ts)

            # both viewでは常に同期
            _update_cell_view_sync(parent, source_view=parent.p1)

        elif bid == 2:
            # not cells view
            parent.win.ci.layout.setColumnStretchFactor(0, 0)
            parent.win.ci.layout.setColumnStretchFactor(1, ts)

            # cells側から来た可能性があるので、
            # sync ONなら p1 の視野を p2 に反映
            _update_cell_view_sync(parent, source_view=parent.p1)

        # only enable selection buttons when not in "both" view
        if bid != 1:
            if parent.ops_plot["color"] != 0:
                for btn in parent.topbtns.buttons():
                    btn.setEnabled(True)
            else:
                parent.topbtns.button(0).setEnabled(True)
        else:
            parent.ROI_remove()
            for btn in parent.topbtns.buttons():
                btn.setEnabled(False)

        parent.win.show()
        parent.show()


#
class TopButton(QPushButton):
    """ selection of top neurons"""

    def __init__(self, bid, parent=None):
        super(TopButton, self).__init__(parent)
        text = [" draw selection", " select top n", " select bottom n"]
        self.bid = bid
        self.setText(text[bid])
        self.setCheckable(True)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent))
        self.show()

    def press(self, parent):
        bid = self.bid
        if not parent.sizebtns.button(1).isChecked():
            if parent.ops_plot["color"] == 0:
                for b in [1, 2]:
                    parent.topbtns.button(b).setEnabled(False)
            else:
                for b in [1, 2]:
                    parent.topbtns.button(b).setEnabled(True)
        else:
            for b in range(3):
                parent.topbtns.button(b).setEnabled(False)
        if bid == 0:
            parent.ROI_selection()
        else:
            self.top_selection(parent)

    def top_selection(self, parent):
        bid = self.bid
        parent.ROI_remove()
        draw = False
        ncells = len(parent.stat)
        icells = np.minimum(ncells, parent.ntop)
        if bid == 1:
            top = True
        elif bid == 2:
            top = False
        if parent.sizebtns.button(0).isChecked():
            wplot = 0
            draw = True
        elif parent.sizebtns.button(2).isChecked():
            wplot = 1
            draw = True
        if draw:
            if parent.ops_plot["color"] != 0:
                c = parent.ops_plot["color"]
                istat = parent.colors["istat"][c]
                if wplot == 0:
                    icell = np.array(parent.iscell.nonzero()).flatten()
                    istat = istat[parent.iscell]
                else:
                    icell = np.array((~parent.iscell).nonzero()).flatten()
                    istat = istat[~parent.iscell]
                inds = istat.argsort()
                if top:
                    inds = inds[-icells:]
                    parent.ichosen = icell[inds[-1]]
                else:
                    inds = inds[:icells]
                    parent.ichosen = icell[inds[0]]
                parent.imerge = []
                for n in inds:
                    parent.imerge.append(icell[n])
                # draw choices
                parent.update_plot()
                parent.show()
