"""Scrolling ticker bar — continuous horizontal scroll of asset prices and changes."""

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QPainter, QColor, QFontMetrics
from PySide6.QtWidgets import QWidget

from portopt.constants import Colors, Fonts


class TickerBar(QWidget):
    """Horizontal scrolling ticker bar displayed at the top of the terminal."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._offset = 0.0
        self._items: list[dict] = []
        self._rendered_text = ""
        self._text_width = 0
        self._speed = 1.0  # pixels per tick

        self._font = QFont(Fonts.MONO, Fonts.SIZE_TICKER)
        self._font.setBold(True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(30)  # ~33 fps

        # Default demo tickers
        self.set_items([
            {"symbol": "SPY", "price": 0.0, "change_pct": 0.0},
            {"symbol": "QQQ", "price": 0.0, "change_pct": 0.0},
            {"symbol": "DIA", "price": 0.0, "change_pct": 0.0},
            {"symbol": "IWM", "price": 0.0, "change_pct": 0.0},
        ])

    def set_items(self, items: list[dict]):
        """Set ticker items. Each dict: {symbol, price, change_pct}."""
        self._items = items
        self._build_text()
        if items and not self._timer.isActive():
            self._timer.start()

    def _build_text(self):
        """Build the full scrolling text string and measure its width."""
        parts = []
        for item in self._items:
            sym = item["symbol"]
            price = item.get("price", 0.0)
            chg = item.get("change_pct", 0.0)
            sign = "+" if chg >= 0 else ""
            if price > 0:
                parts.append(f"  {sym}  ${price:,.2f}  {sign}{chg:.2f}%  ")
            else:
                parts.append(f"  {sym}  ---  {sign}{chg:.2f}%  ")
        separator = "  |  "
        self._rendered_text = separator.join(parts)
        if self._rendered_text:
            self._rendered_text += separator  # trailing separator for seamless loop

        fm = QFontMetrics(self._font)
        self._text_width = fm.horizontalAdvance(self._rendered_text) if self._rendered_text else 0

    def _tick(self):
        self._offset -= self._speed
        if self._text_width > 0 and abs(self._offset) >= self._text_width:
            self._offset = 0.0
        self.update()

    def paintEvent(self, event):
        if not self._rendered_text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._font)

        # Background
        painter.fillRect(self.rect(), QColor(Colors.BG_SECONDARY))

        # Draw bottom border
        painter.setPen(QColor(Colors.BORDER))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

        y = self.height() // 2 + 4

        # Draw two copies for seamless scrolling
        for x_base in [self._offset, self._offset + self._text_width]:
            x = x_base
            for item in self._items:
                sym = item["symbol"]
                price = item.get("price", 0.0)
                chg = item.get("change_pct", 0.0)
                sign = "+" if chg >= 0 else ""

                # Symbol in accent color
                painter.setPen(QColor(Colors.ACCENT))
                sym_text = f"  {sym}  "
                painter.drawText(int(x), y, sym_text)
                x += QFontMetrics(self._font).horizontalAdvance(sym_text)

                # Price in white
                painter.setPen(QColor(Colors.TEXT_PRIMARY))
                if price > 0:
                    price_text = f"${price:,.2f}  "
                else:
                    price_text = "---  "
                painter.drawText(int(x), y, price_text)
                x += QFontMetrics(self._font).horizontalAdvance(price_text)

                # Change % in green/red
                color = Colors.PROFIT if chg >= 0 else Colors.LOSS
                painter.setPen(QColor(color))
                chg_text = f"{sign}{chg:.2f}%"
                painter.drawText(int(x), y, chg_text)
                x += QFontMetrics(self._font).horizontalAdvance(chg_text)

                # Separator
                painter.setPen(QColor(Colors.TEXT_MUTED))
                sep = "    |  "
                painter.drawText(int(x), y, sep)
                x += QFontMetrics(self._font).horizontalAdvance(sep)

        painter.end()
