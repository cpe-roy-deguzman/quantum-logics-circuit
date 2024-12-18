import sys
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from string import ascii_uppercase as ABC
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

WIN_WIDTH = 1280
WIN_HEIGHT = 720

TYPE_QUBIT = 0
TYPE_GATE = 1
TYPE_MEASURE = 2

GATE_PAULI_X: int = 0
GATE_PAULI_Y: int = 1
GATE_PAULI_Z: int = 2

buffer = BytesIO()

class CircuitComponent(QGraphicsPixmapItem):
    """
    Base class for circuit components (Qubit or Gate).
    """
    def __init__(self, pixmap, component_type: int, component_name: str, grid_size: int):
        super().__init__(pixmap)
        self.component_name = component_name
        self.component_type = component_type
        self.grid_size = grid_size

        self.connected_prev = None
        self.connected_next = None

        self.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setFlags(
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setAcceptDrops(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Ensure the item is selected on click
            if self.scene():
                for item in self.scene().selectedItems():
                    item.setSelected(item == self)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Snap the component to the nearest grid point when the mouse is released.
        """
        # Get the current position of the component
        curr_pos = self.pos()

        # Calculate the nearest grid position
        snapped_x = round(curr_pos.x() / self.grid_size) * self.grid_size
        snapped_y = round(curr_pos.y() / self.grid_size) * self.grid_size

        # Set the new position to the snapped grid position
        self.setPos(QPointF(snapped_x, snapped_y))

        # Check for potential connections
        if self.scene():
            self.scene().check_connections(self)

        # Call the base class implementation to ensure default behavior
        return super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)

        if self.isSelected() and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.connected_next != None:
                self.connected_next.connected_prev = None

            if self.connected_prev != None:
                self.connected_prev.connected_next = None

            self.deleted()

            self.scene().removeItem(self)
            
    def deleted(self, *args, **kwargs):
        """
        Overwrite function to be called when the CircuitComponent is deleted from the GraphicsScene
        """
        pass
    
class QubitComponent(CircuitComponent):
    def __init__(self, id: int, value: int, pixmap, grid_size):
        match value:
            case 1:
                self.value = np.array([[0], [1]])
                component_name = "qubit-1"
            case _:
                self.value = np.array([[1], [0]])
                component_name = "qubit-0"
                
        super().__init__(pixmap, TYPE_QUBIT, component_name, grid_size)

        self.__id: int = id

    def setID(self, value: int):
        self.__id = value

    def getID(self) -> int:
        return self.__id
    
    def deleted(self):
        if hasattr(self.scene(), "componentDeleted"):
            self.scene().componentDeleted.emit(self.__id)
    

class GateComponent(CircuitComponent):
    lookup = {
        GATE_PAULI_X: {
            "name": "Pauli-X",
            "value": np.array([
                [0, 1],
                [1, 0]
            ])
        },
        GATE_PAULI_Y: {
            "name": "Pauli-Y",
            "value": np.array([
                [0, -1j],
                [1j, 0]
            ])
        },
        GATE_PAULI_Z: {
            "name": "Pauli-Z",
            "value": np.array([
                [1, 0],
                [0, -1]
            ])
        }
    }

    def __init__(self, gate_type: str | int, pixmap: QPixmap, grid_size: int):
        # Initial gate type
        self.gate_type: int = GATE_PAULI_X

        if isinstance(gate_type, str):
            value = gate_type.upper()
            match value:
                case "PAULI-Y":
                    self.gate_type = GATE_PAULI_Y
                case "PAULI-Z":
                    self.gate_type = GATE_PAULI_Z
                case _:
                    self.gate_type = GATE_PAULI_X
        
        elif isinstance(gate_type, int):
            self.gate_type = gate_type if gate_type in GateComponent.lookup.keys() else GATE_PAULI_X
        
        # Set gate component name
        component_name = GateComponent.lookup.get(self.gate_type).get("name", "Pauli-X")

        super().__init__(pixmap, TYPE_GATE, component_name, grid_size)

        self.value = GateComponent.lookup.get(self.gate_type).get("value", [[0, 1], [1, 0]])

class MeasureComponent(CircuitComponent):
    def __init__(self, pixmap: QPixmap, grid_size: int):
        super().__init__(pixmap, TYPE_MEASURE, "measure", grid_size)

class ConnectionLine(QGraphicsLineItem):
    """
    Base class for connection line graphics item
    """
    def __init__(self, x1: float, y1: float, x2: float, y2: float, parent=None) -> None:
        super().__init__(x1, y1, x2, y2, parent=parent)
        self.setPen(QPen(QColor("black"), 2))
        self.__prev = None
        self.__next = None

        self.setFlag(QGraphicsLineItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsLineItem.GraphicsItemFlag.ItemIsFocusable, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def setPrev(self, qubit: CircuitComponent):
        self.__prev = qubit

    def setNext(self, gate: CircuitComponent):
        self.__next= gate

    def getPrev(self):
        return self.__prev

    def getNext(self):
        return self.__next

    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)

        if self.isSelected() and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.__prev != None:
                self.__prev.connected_next = None

            if self.__next != None:
                self.__next.connected_prev = None

            self.scene().removeItem(self)

class ComponentIcon(QWidget):
    """
    A QWidget subclass for draggable component icons.
    """
    def __init__(self, pixmap: QPixmap, component_name: str, component_type: int, display_text: str = "Component"):
        super().__init__()
        text = QLabel(display_text)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setFixedHeight(15)
        text.setFont(QFont("Montserrat Medium", 9))
        icon = QLabel()
        icon.setPixmap(pixmap)

        self.setFixedSize(pixmap.width(), pixmap.height() + text.height())
        self.setObjectName("component-icon")
        self.setContentsMargins(0, 0, 0, 0)

        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addWidget(icon)
        self.mainLayout.addWidget(text)

        self.component_name = component_name
        self.component_type = component_type

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Component Dragging Operation
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(str(self.component_type) + "," + self.component_name)
            drag.setMimeData(mime_data)

            # Drag preview
            drag_pixmap = self.grab()
            drag.setPixmap(drag_pixmap)
            drag.setHotSpot(event.pos())
            drag.exec(Qt.DropAction.CopyAction)


class QuantumToolbar(QScrollArea):
    def __init__(self, parent) -> None:
        super().__init__(parent=parent)
        self.setObjectName("toolbar")

        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.setContentsMargins(0, 0, 0, 0)
        self.setFixedWidth(196)

        # Enumerate quantum circuit components
        components = [
            {"name": "qubit-0", "type": TYPE_QUBIT, "img": "qubit-0.png", "pos": (1, 0)},
            {"name": "qubit-1", "type": TYPE_QUBIT, "img": "qubit-1.png", "pos": (1, 1)},
            {"name": "Pauli-X", "type": TYPE_GATE, "img": "Pauli-X.png", "pos": (3, 0)},
            {"name": "Pauli-Y", "type": TYPE_GATE, "img": "Pauli-Y.png", "pos": (3, 1)},
            {"name": "Pauli-Z", "type": TYPE_GATE, "img": "Pauli-Z.png", "pos": (4, 0)},
            {"name": "measure", "type": TYPE_MEASURE, "img": "measure.png", "pos": (6, 0)}
        ]

        main = QWidget(self)

        grid = QGridLayout(main)
        grid.addWidget(QLabel("Qubits"), 0, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Quantum Gates"), 2, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(QLabel("Measurement"), 5, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        grid.setRowStretch(0, 0)
        grid.setRowStretch(2, 0)
        grid.setRowStretch(5, 0)
        grid.setRowStretch(7, 1)

        # Populate palette with components
        for component in components:
            grid.addWidget(ComponentIcon(QPixmap(component.get("img", "null.png")), component["name"], component["type"]), *component["pos"])
        
        self.setWidget(main)
        self.setWidgetResizable(True)

class SimulationScene(QGraphicsScene):
    componentDeleted = pyqtSignal(int)

    def __init__(self, parent):
        super().__init__(parent=parent)

    def check_connections(self, component: CircuitComponent):
        """
        Check if the given component should connect to another.
        """

        if component.component_type == TYPE_QUBIT:
            return  # Qubits cannot connect to other qubits

        # Iterate through all items in the scene
        for item in self.items():
            if item == component or not isinstance(item, CircuitComponent):
                continue

            qubitToGate = component.component_type == TYPE_GATE and item.component_type == TYPE_QUBIT
            gateToMeasure = component.component_type == TYPE_MEASURE and item.component_type == TYPE_GATE
            gateToGate = component.component_type == TYPE_GATE and item.component_type == TYPE_GATE
            qubitToMeasure = component.component_type == TYPE_MEASURE and item.component_type == TYPE_QUBIT

            # Ensure connection is valid (qubit <-> gate)
            if qubitToGate or gateToMeasure or gateToGate or qubitToMeasure:
                # Check distance
                yDistance = abs(component.pos().y() - item.pos().y())
                distance = (component.pos() - item.pos()).manhattanLength() + yDistance * 2

                if distance <= component.grid_size * 8:  # Allow some tolerance
                    # Replace existing connection if there is any
                    if item.connected_next not in (component, None) :
                        self.removeItem(item.connected_next)

                    # Align component (gate) to item (qubit)
                    component.setPos(item.x() + component.grid_size * 6, item.y())

                    # Establish two-way connection
                    component.connected_prev = item
                    item.connected_next = component
                    self.add_connection_line(item, component)

                    # Return immediately to only connect to one (1) qubit
                    break

                # If not close enough but is still conencted, disconnect
                elif component.connected_next == item or component.connected_prev == item:
                    for line in self.items():
                        if isinstance(line, ConnectionLine):
                            if line.getNext() == component and line.getPrev() == item:
                                item.connected_next = None
                                component.connected_prev = None
                                self.removeItem(line)
                            elif line.getNext() == item and line.getPrev() == component:
                                item.connected_prev = None
                                component.connected_next = None
                                self.removeItem(line)

            
                    
                
    def add_connection_line(self, qubit: CircuitComponent, gate: CircuitComponent):
        """
        Draw a line to represent the connection between a qubit and a gate.
        """
        # Remove any existing connection lines
        for item in self.items():
            if isinstance(item, ConnectionLine):
                if item.getPrev() == qubit and item.getNext() == gate:
                    self.removeItem(item)

        # Add a new connection line
        line: ConnectionLine = ConnectionLine(
            qubit.x() + qubit.pixmap().width(),
            qubit.y() + qubit.pixmap().height() / 2,
            gate.x(),
            gate.y() + gate.pixmap().height() / 2
        )
        line.setPrev(qubit)
        line.setNext(gate)
        self.addItem(line)

class SimulationWindow(QGraphicsView):
    gridColor = {
        "fg": "#D7D7D7",
        "bg": "#F5F5F5"
    }

    def __init__(self, parent) -> None:
        super().__init__(parent=parent)
        self.setObjectName("simulation-window")
        self.setContentsMargins(0, 0, 0, 0)

        self.scene : SimulationScene = SimulationScene(self)
        self.scene.setSceneRect(0, 0, WIN_WIDTH, WIN_HEIGHT)
        self.grid_scale = 24
        self.qubits: list[QubitComponent] = []
        self._initGridBackground(self.scene, self.grid_scale)
        self.setScene(self.scene)

        self.scene.componentDeleted.connect(self._removeQubit)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.setAcceptDrops(True)

    def _removeQubit(self, id: int):
        self.qubits.pop(id)

    def _initGridBackground(self, scene: QGraphicsScene, scale: int = 16):
        # Create a transparent texture of size W * H
        gridTexture = QPixmap(scale, scale)
        gridTexture.fill(QColor(SimulationWindow.gridColor["bg"]))

        # Draw perpendicular lines across the texture
        painter : QPainter = QPainter(gridTexture)
        painter.setPen(QColor(SimulationWindow.gridColor["fg"]))
        painter.drawLine(0, 0, 0, scale - 1)
        painter.drawLine(0, 0, scale - 1, 0)
        painter.end()

        self.scene.setBackgroundBrush(QBrush(gridTexture))

    def __labelPixmap(self, pixmap: QPixmap, label: str, fontsize:int = 12) -> QPixmap:
        """
        Method to label a given QPixmap (left side)
        """
        w = pixmap.width()
        h = pixmap.height()
        output = QPixmap(w, h)
        output.fill(QColor("transparent"))  # Fill with transparency

        # Start painting on the new pixmap
        painter = QPainter(output)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  # Smooth edges

        # Set font and color for the letter
        font = QFont("Cambria Math", fontsize)  # Choose a font and size
        painter.setFont(font)
        painter.setPen(QColor("black"))  # Set letter color

        # Draw the letter on the left side
        letter_x = 0 # Position with a margin from the left
        letter_y = h // 2 + fontsize // 2 - 2  # Center vertically
        painter.drawText(letter_x, letter_y, label)

        # Draw the original pixmap to the right of the letter
        painter.drawPixmap(10, 4, pixmap.scaled(w - 8, h - 8, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        # End painting
        painter.end()
        return output

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasText():
            data = event.mimeData().text().split(",")

            component_type: int = int(data[0]) if data[0].isnumeric() else TYPE_QUBIT
            
            component_name: str = data[1]

            pixmap = QPixmap(f"{component_name}.png")

            if component_type == TYPE_QUBIT:
                qubit_value: int = 1 if component_name == "qubit-1" else 0
                # Find the least value for ID among existing qubits
                curr_id: int = 0
                for qubit in self.qubits:
                    if qubit and curr_id < qubit.getID() :
                        break
                    else:
                        curr_id += 1
                item = QubitComponent(curr_id, qubit_value, self.__labelPixmap(pixmap, ABC[curr_id % len(ABC)]), self.grid_scale)
                self.qubits.insert(curr_id, item)

            elif component_type == TYPE_GATE:
                item = GateComponent(component_name, pixmap, self.grid_scale)
            elif component_type == TYPE_MEASURE:
                item = MeasureComponent(pixmap, self.grid_scale)
            else:
                return
            
            itemPosition = self.mapToScene(event.position().toPoint())

            item.setPos(itemPosition - QPointF(itemPosition.x() % self.grid_scale, itemPosition.y() % self.grid_scale))
            self.scene.addItem(item)
            event.acceptProposedAction()

            self.scene.check_connections(item)

        

class ResultsWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setObjectName("result")
        self.setFixedWidth(320)
        self.setContentsMargins(0, 0, 0, 0)

        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QStackedLayout(self)

        # Create default page
        defaultPage = QWidget(self)
        defaultLayout = QHBoxLayout(defaultPage)
        defaultLayout.addStretch(1)
        defaultLayout.addWidget(QLabel("No results to show"))
        defaultLayout.addStretch(1)

        # Initialize results page
        self.resultsPage = QWidget(parent=self)
        self.resultsPage.setContentsMargins(0, 0, 0, 0)

        resultsTitle = QLabel("Results")
        resultsTitle.setFont(QFont("Montserrat Medium", 16, QFont.Weight.Bold))

        self.measuredOutputLabel = QLabel(self.resultsPage)
        self.measuredOutputLabel.setFont(QFont("Montserrat Medium", 12, QFont.Weight.Bold))

        self.probabilitiesTable = QTableWidget(self.resultsPage)
        self.probabilitiesTable.setFrameShape(QFrame.Shape.NoFrame)
        self.probabilitiesTable.setFixedWidth(200)
        self.probabilitiesTable.setColumnCount(2)
        self.probabilitiesTable.horizontalHeader().hide()
        self.probabilitiesTable.verticalHeader().hide()
        self.probabilitiesTable.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.probabilitiesTable.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.probabilitiesTable.setShowGrid(False)
        self.probabilitiesTable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.probabilitiesTable.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        self.stateVectorFigure = Figure(figsize=(3, 1), dpi=100)
        self.stateVectorCanvas = FigureCanvas(self.stateVectorFigure)
        self.stateVectorArea = self.stateVectorFigure.add_subplot(1, 1, 1)
        self.stateVectorArea.axis("off")

        qubitsTable = QWidget(self.resultsPage)
        self.qubitsLayout = QGridLayout(qubitsTable)
        self.qubitsLayout.setColumnMinimumWidth(0, 40)
        self.qubitsLayout.setColumnMinimumWidth(2, 80)
        self.qubitsLayout.setColumnStretch(0, 0)
        self.qubitsLayout.setColumnStretch(1, 1)
        self.qubitsLayout.setColumnStretch(2, 0)

        resultsLayout = QVBoxLayout(self.resultsPage)
        resultsLayout.addWidget(resultsTitle, 0, Qt.AlignmentFlag.AlignLeft)
        resultsLayout.addWidget(QLabel("Measured Output", self.resultsPage), 0, Qt.AlignmentFlag.AlignLeft)
        resultsLayout.addWidget(self.measuredOutputLabel, 0, Qt.AlignmentFlag.AlignCenter)
        resultsLayout.addSpacing(16)
        resultsLayout.addWidget(QLabel("Circuit Probabilities", self.resultsPage), 0, Qt.AlignmentFlag.AlignLeft)
        resultsLayout.addWidget(self.probabilitiesTable, 0, Qt.AlignmentFlag.AlignCenter)
        resultsLayout.addSpacing(16)
        resultsLayout.addWidget(QLabel("State Vector", self.resultsPage), 0, Qt.AlignmentFlag.AlignLeft)
        resultsLayout.addWidget(self.stateVectorCanvas, 0, Qt.AlignmentFlag.AlignCenter)
        resultsLayout.addSpacing(16)
        resultsLayout.addWidget(QLabel("Information per Qubit", self.resultsPage), 0, Qt.AlignmentFlag.AlignLeft)
        resultsLayout.addWidget(qubitsTable, 0, Qt.AlignmentFlag.AlignCenter)
        resultsLayout.addStretch()

        gateInfoPage = QWidget(parent=self)

        layout.insertWidget(0, self.resultsPage)
        layout.insertWidget(1, gateInfoPage)
        layout.insertWidget(2, defaultPage)

        layout.setCurrentIndex(2)

    def process_results(self, final_state: np.ndarray, qubit_states: dict[int, np.ndarray]):
        """
        Process and display the results based on the final state.
        """
        num_qubits = len(qubit_states)              # Number of qubits
        probabilities = np.abs(final_state) ** 2    # Probabilities for each basis state

        # Calculate for measured output using probabilities
        basis_states = list(range(2 ** num_qubits))
        output = np.random.choice(basis_states, p=probabilities.transpose().flatten())
        self.measuredOutputLabel.setText(format(output, f"0{num_qubits}b"))

        # Display probabilities for all basis states
        self.probabilitiesTable.clearContents()
        self.probabilitiesTable.setRowCount(len(probabilities))
        for i, p in enumerate(probabilities):
            item_state = QTableWidgetItem(format(i, f"0{num_qubits}b"))
            item_probability = QTableWidgetItem(f"{p.item() * 100:.2f}%")
            self.probabilitiesTable.setItem(i, 0, item_state)
            self.probabilitiesTable.setItem(i, 1, item_probability)
        
        total_height = 2 * self.probabilitiesTable.frameWidth()
        for y in range(self.probabilitiesTable.rowCount()):
            total_height += self.probabilitiesTable.rowHeight(y)

        self.probabilitiesTable.setFixedHeight(total_height)

        # Display the final state vector
        self.display_state_vector(final_state, num_qubits)

        # Reset and generate results for each qubit
        self.reset_qubits_info()
        self.display_qubits_info(qubit_states, num_qubits)

        # Set index of stacked layout to show the results page
        self.layout().setCurrentIndex(0)

    def display_state_vector(self, full_state: np.ndarray, num_qubits: int):
        """
        Display the resulting quantum state vector.
        """
        
        data = r"$\left.|\psi\right\rangle" + f"={"+".join([f'{x} \\left.|{format(i, f'0{num_qubits}b')}\\right\\rangle' for i, x in enumerate(full_state.flatten())])}$"

        self.stateVectorArea.clear()
        self.stateVectorArea.axis("off")
        self.stateVectorArea.text(0.5, 0.5, data, fontsize=12, ha="center", va="center")
        self.stateVectorCanvas.draw()

    def reset_qubits_info(self):
        # Reconfigure the qubits table
        while self.qubitsLayout.count() > 0:
            self.qubitsLayout.removeWidget(self.qubitsLayout.itemAt(0).widget())

    def display_qubits_info(self, qubit_states: dict[int, np.ndarray], num_qubits: int):
        for i, q in enumerate(qubit_states.items()):
            id, state = q

            # Display qubit ID
            qid_label = QLabel(ABC[id % len(ABC)])
            qid_label.setFont(QFont("Cambria Math", 15))
            self.qubitsLayout.addWidget(qid_label, i, 0, Qt.AlignmentFlag.AlignCenter)

            # Display qubit Bloch sphere
            bs_label = QLabel()
            bs_label.setPixmap(self.get_bloch_sphere(state))
            self.qubitsLayout.addWidget(bs_label, i, 1, Qt.AlignmentFlag.AlignCenter)

            # Display state as matrix
            qs_label = QLabel()
            qs_label.setPixmap(self.get_qubit_state(state))
            self.qubitsLayout.addWidget(qs_label, i, 2, Qt.AlignmentFlag.AlignCenter)

    def get_bloch_sphere(self, qubit_state: np.ndarray) -> QPixmap:
        """
        Retrieve the Bloch sphere representation of a qubit state as a QPixmap.
        """
        # Initialize figures for Bloch Sphere
        bs_figure = Figure(figsize=(4, 4), dpi=100)
        ax = bs_figure.add_subplot(1, 1, 1, projection='3d')
        ax.autoscale(enable=False)
        ax.axis("off")
        ax.set_box_aspect((1, 1, 1))

        # Shift view (pan)
        shift_x, shift_y = 0.1, 0
        ax.set_xlim([-1.2 + shift_x, 1.2 + shift_x])  # X-axis range
        ax.set_ylim([-1.2 + shift_y, 1.2 + shift_y])  # Y-axis range
        ax.set_zlim([-1.2, 1.2])  # Z-axis range

        # Compute Bloch sphere coordinates
        alpha, beta = qubit_state
        theta = 2 * np.arccos(np.abs(alpha))
        phi = np.angle(beta) - np.angle(alpha)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Draw Bloch sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.sin(v), np.cos(u))
        y_sphere = np.outer(np.sin(v), np.sin(u))
        z_sphere = np.outer(np.cos(v), np.ones_like(u))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='c', alpha=0.1)

        # Draw the x, y, and z axes
        axis_length = 1.2  # Extend slightly beyond the sphere radius
        # X-axis (black)
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='k', linewidth=1.5, arrow_length_ratio=0.1, alpha=0.2)
        ax.quiver(0, 0, 0, -axis_length, 0, 0, color='k', linewidth=1.5, arrow_length_ratio=0.1, alpha=0.2) 
        # Y-axis (black) 
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='k', linewidth=1.5, arrow_length_ratio=0.1, alpha=0.2)
        ax.quiver(0, 0, 0, 0, -axis_length, 0, color='k', linewidth=1.5, arrow_length_ratio=0.1, alpha=0.2)
        # Z-axis (black)
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='k', linewidth=1.5, arrow_length_ratio=0.1, alpha=0.2)
        ax.quiver(0, 0, 0, 0, 0, -axis_length, color='k', linewidth=1.5, arrow_length_ratio=0.1, alpha=0.2)

        # Draw latitude lines (parallels)
        for lat_z in np.linspace(-1, 1, 5):  # 9 lines including the poles
            lat_r = np.sqrt(1 - lat_z**2)  # Radius of the circle at height z
            lat_u = np.linspace(0, 2 * np.pi, 100)
            lat_x = lat_r * np.cos(lat_u)
            lat_y = lat_r * np.sin(lat_u)
            ax.plot(lat_x, lat_y, lat_z, color='k', linewidth=0.5, alpha=0.3 if lat_z == 0 else 0.08)

        # Draw longitude lines (meridians)
        for lon_i, lon_phi in enumerate(np.linspace(0, 2 * np.pi, 12, False)):  # 12 meridians
            lon_u = np.linspace(-1, 1, 100)  # z-values from -1 to 1
            lon_x = np.sqrt(1 - lon_u**2) * np.cos(lon_phi)
            lon_y = np.sqrt(1 - lon_u**2) * np.sin(lon_phi)
            lon_z = lon_u
            ax.plot(lon_x, lon_y, lon_z, color='k', linewidth=0.5, alpha=0.3 if lon_i % 3 == 0 else 0.08)

        # Plot the state vector
        ax.quiver(0, 0, 0, x, y, z, color='r', linewidth=2)
        ax.set_proj_type("ortho")

        # Reset buffer to prepare for saving the plot as image
        global buffer
        buffer.seek(0)
        buffer.truncate(0)

        # Save plot as image to a bytes buffer
        bs_figure.savefig(buffer, format="png", bbox_inches=Bbox.from_bounds(1, 1, 2, 2))
        buffer.seek(0)

        bs_pixmap = QPixmap()
        bs_pixmap.loadFromData(buffer.getvalue(), "PNG")
        return bs_pixmap.scaled(150, 150, transformMode=Qt.TransformationMode.SmoothTransformation)
            
    def get_qubit_state(self, qubit_state: np.ndarray, size:tuple[int, int] = (50, 100), text_box_size: tuple[int, int]= (40, 20), mar_y: int = 20, b_l: int = 10) -> QPixmap:
        """
        Display the qubit state as a LaTeX matrix using matplotlib.
        """
        w, h = size
        t_w, t_h = text_box_size
        alpha, beta = qubit_state.flatten()

        output = QPixmap(*size)
        output.fill(QColorConstants.Transparent)
        painter = QPainter(output)
        painter.setPen(QPen(QColor("black"), 2))
        painter.setFont(QFont("Cambria Math", 18))

        painter.drawText(w // 2 - t_w // 2, mar_y, t_w, t_h, Qt.AlignmentFlag.AlignCenter, f"{alpha}")
        painter.drawText(w // 2 - t_w // 2, h - mar_y - t_h, t_w, t_h, Qt.AlignmentFlag.AlignCenter, f"{beta}")

        painter.drawPolyline(
            QPoint(b_l, 0),
            QPoint(0, 0),
            QPoint(0, h),
            QPoint(b_l, h)
        )

        painter.drawPolyline(
            QPoint(w - b_l, 0),
            QPoint(w, 0),
            QPoint(w, h),
            QPoint(w - b_l, h)
        )

        painter.end()

        return output


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(WIN_WIDTH, WIN_HEIGHT)
        self.setWindowTitle("Quantum Circuits Simulation Software")
        self.setContentsMargins(0, 0, 0, 0)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        defFont = QFont("Montserrat Medium", 9)
        defFont.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.36)
        defFont.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        self.setFont(defFont)

        # Initialize main widget container
        main = QWidget(self)
        main.setObjectName("main")

        middle = QWidget(self)
        middle.setObjectName("main-middle")
        middle.setAutoFillBackground(True)
        middle.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        middle.setContentsMargins(0, 0, 0, 0)

        self.quantumToolbar = QuantumToolbar(main)
        self.simulationWindow = SimulationWindow(middle)

        self.simulationTools = QWidget(middle)
        self.simulationTools.setFixedHeight(64)

        self.simulateButton = QPushButton("Simulate", self.simulationTools)
        self.simulateButton.setFixedHeight(32)
        self.simulateButton.clicked.connect(self.calculateResults)

        simulationToolsLayout = QHBoxLayout(self.simulationTools)
        simulationToolsLayout.addStretch(1)
        simulationToolsLayout.addWidget(self.simulateButton)

        self.resultsWindow = ResultsWidget(main)
        
        middleLayout = QVBoxLayout(middle)
        middleLayout.setSpacing(0)
        middleLayout.addWidget(self.simulationWindow, 1)
        middleLayout.addWidget(self.simulationTools, 0)

        # Add all layers to "main"
        layout = QHBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.quantumToolbar)
        layout.addWidget(middle)
        layout.addWidget(self.resultsWindow)

        # Set "main" as main widget
        self.setCentralWidget(main)

        # Open and read the css as stylesheet of the app
        with open("styles.css") as f:
            style = f.read()
        self.setStyleSheet(style)

    def calculateResults(self):
        # Find all QubitComponents in the circuit
        qubits: list[QubitComponent] = [comp for comp in self.simulationWindow.scene.items() if isinstance(comp, QubitComponent)]
        qubits.sort(key=lambda q: q.getID())

        if not qubits:
            QMessageBox.warning(self, "Error", "No qubits found in the circuit!")
            return

        # Traverse and simulate each qubit
        qubit_states : dict = {}
        for qubit in qubits:
            curr_state: np.ndarray = qubit.value  # Initial state of the qubit
            curr_component = qubit.connected_next

            while curr_component:
                if isinstance(curr_component, GateComponent):
                    curr_state = np.dot(curr_component.value, curr_state)  # Apply gate matrix
                elif isinstance(curr_component, MeasureComponent):
                    # Simulate measurement
                    qubit_states.update({qubit.getID(): curr_state})
                    break

                curr_component = curr_component.connected_next

        # Calculate for the final state using tensor (kronecker) product
        if qubit_states:
            states: list[np.ndarray] = list(qubit_states.values())
            final_state: np.ndarray = states[0]

            for i, state in enumerate(states):
                if i >= 1:
                    final_state = np.kron(final_state, state)

            # Process the final state and qubit states to display (in the ResultsWidget)
            self.resultsWindow.process_results(final_state, qubit_states)
        else:
            QMessageBox.warning(self, "Error", "Nothing to measure! Make sure to use the \"Measure\" Component in the Components Palette")

def runApp():
    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont("Montserrat-Medium.ttf")

    

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    runApp()