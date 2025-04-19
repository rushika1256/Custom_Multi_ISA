import sys
import time 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTextEdit, QPushButton, QComboBox, QPlainTextEdit, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QTimer
import re
sys.path.append(r"D:\Lib\site-packages")



class BaseISA:
    def __init__(self):
        self.registers = {"R" + str(i): i % 8 for i in range(32)}
        self.pc = 0  # Program counter
        # Initialize memory with test values
        self.memory = [
            10, 20, 30, 40, 50, 60, 70, 80,    # 0-7
            15, 25, 35, 45, 55, 65, 75, 85,    # 8-15
            5, 15, 25, 35, 45, 55, 65, 75,     # 16-23
            100, 200, 300, 400, 500, 600, 700, 800  # 24-31
        ]
        self.logs = []
        self.instructions = []
        self.cycles = 0  # Track total cycles
        self.instruction_cycles = []  # Track cycles per instruction
        
    def load_program(self, instructions):
        self.instructions = instructions.strip().splitlines()
        self.logs.append(f"Program loaded with {len(self.instructions)} instruction(s).")
        self.cycles = 0
        self.instruction_cycles = [0] * len(self.instructions)
        self.pc = 0  # Reset program counter
        
    def execute(self):
        self.logs.append("Starting generic execution...")
        for idx, instr in enumerate(self.instructions):
            self.logs.append(f"[Line {idx+1}] Executing: {instr}")
            self.registers["R0"] += 1
            self.memory[idx % 32] = self.registers["R0"]
            self.cycles += 1
            self.instruction_cycles[idx] += 1
        self.logs.append("Generic execution finished.")
        self.logs.append(f"Total cycles: {self.cycles}")
        
    def get_logs(self):
        return "\n".join(self.logs)


class RISCISA(BaseISA):
    def execute(self):
        self.logs.append("Starting RISC ISA execution...")
        for idx, instr in enumerate(self.instructions):
            tokens = [token.strip().upper() for token in instr.split()]
            opcode = tokens[0] if tokens else ""
            
            # Check specifically for memory-to-memory operations (MEM[x], MEM[y])
            if "MEM" in instr and instr.count("MEM") > 1:
                self.logs.append(f"[RISC] Error: Memory-to-memory operations are not supported in RISC architecture!")
                self.logs.append("Suggestion: Please use load/store instructions instead. Examples:")
                self.logs.append("  LD R1, 5      # Load from memory address 5 to R1")
                self.logs.append("  ST R1, 10     # Store R1 to memory address 10")
                self.logs.append("  LDR R2, 15    # Load from memory address 15 to R2")
                continue

            # Allow load/store and register operations
            if opcode in ["LD", "LDR", "LW", "ST", "STR", "SW"]:
                # Handle load/store operations
                if len(tokens) >= 3:
                    reg = tokens[1]
                    addr = int(tokens[2])
                    if opcode in ["LD", "LDR", "LW"]:
                        self.registers[reg] = self.memory[addr]
                        self.logs.append(f"Loaded value {self.memory[addr]} from memory[{addr}] to {reg}")
                    else:  # ST, STR, SW
                        self.memory[addr] = self.registers[reg]
                        self.logs.append(f"Stored value {self.registers[reg]} from {reg} to memory[{addr}]")
            else:
                # Handle regular register operations
                self.logs.append(f"[RISC] Executing: {instr}")
                # Your existing register operation code
                self.registers["R1"] += (idx + 1)
                self.cycles += 1
                self.instruction_cycles[idx] += 1
            
        self.logs.append("RISC execution finished.")
        self.logs.append(f"Total cycles: {self.cycles}")

    def get_instruction_cycles(self, instr):
        # Parse the instruction to determine its type
        tokens = instr.strip().upper().split()
        opcode = tokens[0] if tokens else ""
        
        # Load/Store operations take 2 cycles
        if opcode in ["LD", "LDR", "LW", "ST", "STR", "SW"]:
            return 2
        # Branch operations take 2 cycles
        elif opcode in ["B", "BR", "BEQ", "BNE", "BLT", "BGT", "BLE", "BGE", "J", "JMP"]:
            return 2
        # ALU operations take 1 cycle
        else:
            return 1


class CISCISA(BaseISA):
    def execute(self):
        self.logs.append("Starting CISC ISA execution...")
        for idx, instr in enumerate(self.instructions):
            self.logs.append(f"[CISC] Executing: {instr}")
            
            # Split instruction into parts
            parts = [part.strip() for part in instr.split()]
            
            if len(parts) >= 1:
                opcode = parts[0].upper()
                
                # Handle memory to memory operations
                if "MEM" in instr:
                    self.handle_mem_operation(parts)
                else:
                    # Regular register operations
                    self.handle_register_operation(parts, idx)
                    
            self.cycles += 2  # CISC instructions take 2 cycles
            self.instruction_cycles[idx] += 2
            
        self.logs.append("CISC execution finished.")
        self.logs.append(f"Total cycles: {self.cycles} (2 cycles per instruction)")

    def handle_mem_operation(self, parts):
        opcode = parts[0].upper()
        
        # Existing memory-to-memory pattern
        mem_pattern = r"MEM\[(\d+)\]"
        
        # Check if it's a register-to-memory operation
        if len(parts) == 3 and "MEM" in parts[2]:  # Format: ADD R1, MEM[0]
            reg = parts[1]
            mem_addr = int(re.findall(mem_pattern, parts[2])[0])
            
            if 0 <= mem_addr < 32:
                if opcode == "MOV":
                    self.memory[mem_addr] = self.registers[reg]
                    self.logs.append(f"Moved value {self.registers[reg]} from {reg} to MEM[{mem_addr}]")
                elif opcode == "ADD":
                    self.memory[mem_addr] += self.registers[reg]
                    self.logs.append(f"Added {reg} to MEM[{mem_addr}], result: {self.memory[mem_addr]}")
                elif opcode == "SUB":
                    self.memory[mem_addr] -= self.registers[reg]
                    self.logs.append(f"Subtracted {reg} from MEM[{mem_addr}], result: {self.memory[mem_addr]}")
            else:
                self.logs.append(f"Error: Memory address out of bounds (max: 31)")
            
        # Check if it's a memory-to-register operation
        elif len(parts) == 3 and "MEM" in parts[1]:  # Format: ADD R1, MEM[0]
            reg = parts[2]
            mem_addr = int(re.findall(mem_pattern, parts[1])[0])
            
            if 0 <= mem_addr < 32:
                if opcode == "MOV":
                    self.registers[reg] = self.memory[mem_addr]
                    self.logs.append(f"Moved value {self.memory[mem_addr]} from MEM[{mem_addr}] to {reg}")
                elif opcode == "ADD":
                    self.registers[reg] += self.memory[mem_addr]
                    self.logs.append(f"Added MEM[{mem_addr}] to {reg}, result: {self.registers[reg]}")
                elif opcode == "SUB":
                    self.registers[reg] -= self.memory[mem_addr]
                    self.logs.append(f"Subtracted MEM[{mem_addr}] from {reg}, result: {self.registers[reg]}")
            else:
                self.logs.append(f"Error: Memory address out of bounds (max: 31)")
            
        # Existing memory-to-memory operations
        elif "MEM" in " ".join(parts):
            # Your existing memory-to-memory operation code...
            operands = re.findall(mem_pattern, " ".join(parts))
            if len(operands) == 2:
                dest_addr = int(operands[0])
                src_addr = int(operands[1])
                
                if max(dest_addr, src_addr) < 32:
                    if opcode == "MOV":
                        self.memory[dest_addr] = self.memory[src_addr]
                        self.logs.append(f"Moved value {self.memory[src_addr]} from MEM[{src_addr}] to MEM[{dest_addr}]")
                    
                    elif opcode == "ADD":
                        self.memory[dest_addr] = self.memory[dest_addr] + self.memory[src_addr]
                        self.logs.append(f"Added MEM[{src_addr}] to MEM[{dest_addr}], result: {self.memory[dest_addr]}")
                    
                    elif opcode == "SUB":
                        self.memory[dest_addr] = self.memory[dest_addr] - self.memory[src_addr]
                        self.logs.append(f"Subtracted MEM[{src_addr}] from MEM[{dest_addr}], result: {self.memory[dest_addr]}")
                else:
                    self.logs.append(f"Error: Memory address out of bounds (max: 31)")

    def handle_register_operation(self, parts, idx):
        # Existing register operations
        if len(parts) >= 3:
            opcode = parts[0].upper()
            if opcode in ["ADD", "SUB", "AND", "OR", "MOV", "MUL", "DIV"]:
                # Your existing register operation logic
                self.registers["R2"] += (idx * 2)
                self.memory[idx % 32] = self.registers["R2"]

    def get_instruction_cycles(self, instr):
        # Parse the instruction to determine its type
        tokens = instr.strip().upper().split()
        opcode = tokens[0] if tokens else ""
        
        # Complex operations take more cycles
        if "MEM" in instr:  # Memory-to-memory operations
            return 4
        elif opcode in ["MUL", "DIV", "MOD"]:  # Multiplication, division
            return 3
        elif opcode in ["LD", "LDR", "LW", "ST", "STR", "SW"]:  # Load/Store
            return 3
        elif opcode in ["B", "BR", "BEQ", "BNE", "BLT", "BGT", "BLE", "BGE", "J", "JMP"]:  # Branch
            return 3
        else:  # Regular ALU operations
            return 2




class CrazyISA(BaseISA):
    def execute(self):
        for idx, instr in enumerate(self.instructions):
            parts = instr.replace(",", "").split()
            if not parts:
                continue
            opcode = parts[0].upper()
            step_info = f"[CrazyISA] Step {idx+1}: Executing '{instr}' => "
            if opcode == "ADD" and len(parts) == 4:
                d, s1, s2 = parts[1], parts[2], parts[3]
                self.registers[d] = self.registers[s1] - self.registers[s2]
                self.logs.append(step_info + f"{d} = {self.registers[s1]} - {self.registers[s2]} = {self.registers[d]}")
            elif opcode == "SUB" and len(parts) == 4:
                d, s1, s2 = parts[1], parts[2], parts[3]
                self.registers[d] = self.registers[s1] + self.registers[s2]
                self.logs.append(step_info + f"{d} = {self.registers[s1]} + {self.registers[s2]} = {self.registers[d]}")
            elif opcode == "MUL" and len(parts) == 4:
                d, s1, s2 = parts[1], parts[2], parts[3]
                if self.registers[s2] != 0:
                    self.registers[d] = self.registers[s1] // self.registers[s2]
                    self.logs.append(step_info + f"{d} = {self.registers[s1]} // {self.registers[s2]} = {self.registers[d]}")
            elif opcode == "DIV" and len(parts) == 4:
                d, s1, s2 = parts[1], parts[2], parts[3]
                self.registers[d] = self.registers[s1] * self.registers[s2]
                self.logs.append(step_info + f"{d} = {self.registers[s1]} * {self.registers[s2]} = {self.registers[d]}")
            elif opcode == "LOAD" and len(parts) == 3:
                r, addr = parts[1], int(parts[2])
                self.memory[addr] = self.registers.get(r, 0)
                self.logs.append(step_info + f"MEM[{addr}] = {self.memory[addr]} (LOAD -> STORE)")
            elif opcode == "STORE" and len(parts) == 3:
                r, addr = parts[1], int(parts[2])
                self.registers[r] = self.memory[addr]
                self.logs.append(step_info + f"{r} = MEM[{addr}] = {self.registers[r]} (STORE -> LOAD)")
            elif opcode == "PRINT" and len(parts) == 2:
                r = parts[1]
                self.logs.append(step_info + f"{r} = {self.registers.get(r, 0)}")
            else:
                self.logs.append(step_info + "Unknown or unsupported instruction.")
            self.cycles += 1
        self.logs.append(f"Crazy ISA execution finished. Total cycles: {self.cycles}")


    def get_instruction_cycles(self, instr):
        # Every Crazy ISA instruction takes 1 cycle
        return 1

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Multi-ISA Simulator")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self.get_stylesheet())
        self.paused = False
        self.current_step = 0
        self.simulator = None
        self.instruction_start_time = None 
        self.total_simulated_time = 0.0     
        self.simulation_start_time = None    
        self.execution_multiplier = 1.0    

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.top_layout = QHBoxLayout()
        self.main_layout.addLayout(self.top_layout)
        
        self.isa_selection_layout = QVBoxLayout()
        self.top_layout.addLayout(self.isa_selection_layout)
        isa_label = QLabel("Select ISA:")
        self.isa_selection_layout.addWidget(isa_label)
        self.isa_combo = QComboBox()
        self.isa_combo.addItems(["RISC", "CISC", "Crazy"])
        self.isa_selection_layout.addWidget(self.isa_combo)
        
        self.execute_button = QPushButton("Execute Program")
        self.execute_button.clicked.connect(self.execute_program)
        self.isa_selection_layout.addWidget(self.execute_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.isa_selection_layout.addWidget(self.pause_button)

        self.instruction_editor = QTextEdit()
        self.instruction_editor.setPlaceholderText("Enter assembly-like instructions here...")
        self.top_layout.addWidget(self.instruction_editor, 1)

        self.middle_layout = QHBoxLayout()
        self.main_layout.addLayout(self.middle_layout)
        
        # Create a container for registers and memory text displays
        self.text_displays_container = QWidget()
        self.text_displays_layout = QHBoxLayout()
        self.text_displays_container.setLayout(self.text_displays_layout)
        
        # Registers display
        self.registers_display = QPlainTextEdit()
        self.registers_display.setReadOnly(True)
        self.registers_display.setPlaceholderText("Registers")
        self.text_displays_layout.addWidget(self.registers_display)
        
        # Memory display (new)
        self.memory_text_display = QPlainTextEdit()
        self.memory_text_display.setReadOnly(True)
        self.memory_text_display.setPlaceholderText("Memory Contents")
        self.text_displays_layout.addWidget(self.memory_text_display)

        self.middle_layout.addWidget(self.text_displays_container)
        
        self.memory_widget = QWidget()
        self.memory_layout = QGridLayout()
        self.memory_widget.setLayout(self.memory_layout)
        self.memory_layout.setSpacing(10)
        self.build_memory_display()
        self.memory_scroll = QScrollArea()
        self.memory_scroll.setWidgetResizable(True)
        self.memory_scroll.setWidget(self.memory_widget)
        self.middle_layout.addWidget(self.memory_scroll)
        
        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText("Execution Log")
        self.middle_layout.addWidget(self.log_display)

        self.log_animation = QPropertyAnimation(self.log_display, b"geometry")
        self.log_animation.setDuration(500)
        self.log_animation.setEasingCurve(QEasingCurve.InOutQuad)

        self.step_timer = QTimer()
        self.step_timer.timeout.connect(self.execute_one_step)
        self.step_timer.setInterval(1000)
        
    def build_memory_display(self):
        for i in reversed(range(self.memory_layout.count())):
            widget_to_remove = self.memory_layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()
        
        self.memory_labels = []
        # Add memory display section
        memory_title = QLabel("Memory Values:")
        memory_title.setStyleSheet("color: white; font-weight: bold;")
        self.memory_layout.addWidget(memory_title, 0, 0, 1, 8)
        
        # Display both registers and memory
        for row in range(8):
            for col in range(4):
                idx = row * 4 + col
                # Register display
                reg_label = QLabel(f"R{idx}\nValue: {self.simulator.registers['R'+str(idx)]}" if self.simulator else f"R{idx}\nValue: 0")
                reg_label.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
                reg_label.setAlignment(Qt.AlignCenter)
                reg_label.setMinimumSize(80, 50)
                self.memory_layout.addWidget(reg_label, row + 1, col)
                self.memory_labels.append(reg_label)
                
                # Memory display
                mem_label = QLabel(f"M{idx}\nValue: {self.simulator.memory[idx]}" if self.simulator else f"M{idx}\nValue: 0")
                mem_label.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
                mem_label.setAlignment(Qt.AlignCenter)
                mem_label.setMinimumSize(80, 50)
                mem_label.setStyleSheet("background-color: #2d2d2d;")
                self.memory_layout.addWidget(mem_label, row + 1, col + 4)
                self.memory_labels.append(mem_label)

    def update_memory_display(self, registers, memory, highlight_registers=None, highlight_memory=None):
        # Hide registers >15 in CISC mode
        for idx in range(len(registers)):
            label_idx = idx * 2
            reg_label = self.memory_labels[label_idx]
            if isinstance(self.simulator, CISCISA) and idx >= 16:
                reg_label.hide()
            else:
                reg_label.show()
        if highlight_registers is None:
            highlight_registers = []
        if highlight_memory is None:
            highlight_memory = []
            
        for idx in range(32):
            reg_label = self.memory_labels[idx * 2]
            mem_label = self.memory_labels[idx * 2 + 1]
            
            reg_name = f"R{idx}"
            reg_label.setText(f"{reg_name}\nValue: {registers[reg_name]}")
            mem_label.setText(f"M{idx}\nValue: {memory[idx]}")
            
            # Reset styles first
            reg_label.setStyleSheet("")
            mem_label.setStyleSheet("background-color: #2d2d2d;")
            
            # Apply highlights
            if reg_name in highlight_registers:
                reg_label.setStyleSheet("background-color: #f1c40f;")
            if idx in highlight_memory:
                mem_label.setStyleSheet("background-color: #f1c40f;")
    
    def update_registers_display(self, registers):
        text = "\n".join(f"{reg}: {val}" for reg, val in registers.items())
        self.registers_display.setPlainText(text)
    
    def update_memory_text_display(self, memory):
        # Format memory contents with consistent spacing
        text = []
        for i, val in enumerate(memory):
            text.append(f"MEM[{i:2d}]: {val:4d}")  # This will align the numbers nicely
        self.memory_text_display.setPlainText("\n".join(text))
    
    def execute_program(self):
        if self.isa_combo.currentText() == "Crazy":
            self.simulator = CrazyISA()
            code = self.instruction_editor.toPlainText()
            self.simulator.load_program(code)
            self.simulator.execute()
            self.build_memory_display()
            self.update_registers_display(self.simulator.registers)
            self.update_memory_text_display(self.simulator.memory)
            self.update_memory_display(self.simulator.registers, self.simulator.memory)
            self.log_display.setPlainText("\n".join(self.simulator.logs))
            return

        self.log_display.clear()
        self.registers_display.clear()
        self.memory_text_display.clear()

        code = self.instruction_editor.toPlainText()
        if not code.strip():
            sample_instructions = [
                "ADD R1, R2, R3",
                "SUB R4, R5, R6",
                "AND R7, R8, R9",
                "OR R10, R11, R12",
                "MOV R13, R14",
                "LDR R1, 5",    # Load from memory address 5 into R1
                "STR R2, 10",   # Store R2 into memory address 10
                "MUL R3, R4, R5",
                "DIV R6, R7, R8",
                "BEQ R1, R2, 5", # Branch if R1 == R2 to instruction 5
                "BNE R3, R4, 7", # Branch if R3 != R4 to instruction 7
                "XOR R9, R10, R11",
                "NOT R12, R13",
                "SHL R14, R15, 2", # Shift left R15 by 2 bits, store in R14
                "JMP 3"         # Jump to instruction 3
            ]
            code = "\n".join(sample_instructions)
            self.instruction_editor.setPlainText(code)

        selected_isa = self.isa_combo.currentText()
        if selected_isa == "RISC":
            self.simulator = RISCISA()
            self.execution_multiplier = 1.0 
        elif selected_isa == "CISC":
            self.simulator = CISCISA()
            self.execution_multiplier = 2.0
        elif selected_isa == "Crazy":
            self.simulator = CrazyISA()
        else:
            self.simulator = BaseISA()
            self.execution_multiplier = 1.0

        self.simulator.load_program(code)
        self.simulator.logs.append("Starting simulation (step-by-step) execution.")
        self.current_step = 0
        self.paused = False
        self.pause_button.setText("Pause")
        self.build_memory_display()
        self.total_simulated_time = 0.0
        self.simulation_start_time = time.perf_counter()
        self.step_timer.start()
        self.execute_one_step()
    
    def execute_one_step(self):
        if self.paused:
            return
        
        if self.current_step < len(self.simulator.instructions):
            self.instruction_start_time = time.perf_counter()
            
            instr = self.simulator.instructions[self.current_step]
            step_number = self.current_step + 1
            tokens = [token.replace(",", "").upper() for token in instr.split()]
            opcode = tokens[0]
            used_registers = []
            used_memory = []
            
            # Check which ISA we're using for cycle count
            if isinstance(self.simulator, RISCISA):
                cycles_for_instr = self.simulator.get_instruction_cycles(instr)
            elif isinstance(self.simulator, CISCISA):
                cycles_for_instr = self.simulator.get_instruction_cycles(instr)
            elif isinstance(self.simulator, CrazyISA):
                cycles_for_instr = self.simulator.get_instruction_cycles(instr)
            else:
                cycles_for_instr = 1
                
            # Check if it's a memory operation
            if "MEM" in instr:
                if isinstance(self.simulator, RISCISA):
                    # For RISC, show error message
                    self.simulator.logs.append(f"[RISC] Error: Memory-to-memory operations are not supported in RISC architecture!")
                    self.simulator.logs.append("Suggestion: Please use register-to-register instructions instead. Examples:")
                    self.simulator.logs.append("  ADD R1, R2, R3    # Add contents of R2 and R3, store in R1")
                    self.simulator.logs.append("  MOV R1, R2        # Move contents of R2 to R1")
                    self.simulator.logs.append("  SUB R4, R5, R6    # Subtract R6 from R5, store in R4")
                    self.simulator.cycles += 1  # Count as a cycle
                    self.simulator.instruction_cycles[self.current_step] += 1
                elif isinstance(self.simulator, CISCISA):
                    # For CISC, handle memory operations
                    pattern = r"MEM\[(\d+)\]"
                    memory_addresses = re.findall(pattern, instr)
                    used_memory = [int(addr) for addr in memory_addresses]
                    self.simulator.handle_mem_operation(tokens)
                    self.simulator.cycles += cycles_for_instr  # CISC memory operations take more cycles
                    self.simulator.instruction_cycles[self.current_step] += cycles_for_instr
                    
                    # Update both the visualization and plain text displays
                    self.update_memory_display(self.simulator.registers, self.simulator.memory, 
                                            highlight_registers=None, highlight_memory=used_memory)
                    self.update_memory_text_display(self.simulator.memory)
            else:
                # Handle various instruction types
                if opcode == "ADD" and len(tokens) == 4:
                    dest, src1, src2 = tokens[1], tokens[2], tokens[3]
                    result = self.simulator.registers[src1] + self.simulator.registers[src2]
                    self.simulator.registers[dest] = result
                    self.simulator.logs.append(
                        f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src1]} + {self.simulator.registers[src2]} = {result}."
                    )
                    used_registers = [dest, src1, src2]
                
                elif opcode == "SUB" and len(tokens) == 4:
                    dest, src1, src2 = tokens[1], tokens[2], tokens[3]
                    result = self.simulator.registers[src1] - self.simulator.registers[src2]
                    self.simulator.registers[dest] = result
                    self.simulator.logs.append(
                        f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src1]} - {self.simulator.registers[src2]} = {result}."
                    )
                    used_registers = [dest, src1, src2]
                
                elif opcode == "AND" and len(tokens) == 4:
                    dest, src1, src2 = tokens[1], tokens[2], tokens[3]
                    result = self.simulator.registers[src1] & self.simulator.registers[src2]
                    self.simulator.registers[dest] = result
                    self.simulator.logs.append(
                        f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src1]} & {self.simulator.registers[src2]} = {result}."
                    )
                    used_registers = [dest, src1, src2]
                
                elif opcode == "OR" and len(tokens) == 4:
                    dest, src1, src2 = tokens[1], tokens[2], tokens[3]
                    result = self.simulator.registers[src1] | self.simulator.registers[src2]
                    self.simulator.registers[dest] = result
                    self.simulator.logs.append(
                        f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src1]} | {self.simulator.registers[src2]} = {result}."
                    )
                    used_registers = [dest, src1, src2]
                
                elif opcode == "XOR" and len(tokens) == 4:
                    dest, src1, src2 = tokens[1], tokens[2], tokens[3]
                    result = self.simulator.registers[src1] ^ self.simulator.registers[src2]
                    self.simulator.registers[dest] = result
                    self.simulator.logs.append(
                        f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src1]} ^ {self.simulator.registers[src2]} = {result}."
                    )
                    used_registers = [dest, src1, src2]
                
                elif opcode == "NOT" and len(tokens) == 3:
                    dest, src = tokens[1], tokens[2]
                    result = ~self.simulator.registers[src]
                    self.simulator.registers[dest] = result
                    self.simulator.logs.append(
                        f"Step {step_number}: Executed '{instr}'. {dest} = ~{self.simulator.registers[src]} = {result}."
                    )
                    used_registers = [dest, src]
                
                elif opcode == "MUL" and len(tokens) == 4:
                    dest, src1, src2 = tokens[1], tokens[2], tokens[3]
                    result = self.simulator.registers[src1] * self.simulator.registers[src2]
                    self.simulator.registers[dest] = result
                    self.simulator.logs.append(
                        f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src1]} * {self.simulator.registers[src2]} = {result}."
                    )
                    used_registers = [dest, src1, src2]
                
                elif opcode == "DIV" and len(tokens) == 4:
                    dest, src1, src2 = tokens[1], tokens[2], tokens[3]
                    if self.simulator.registers[src2] != 0:
                        result = self.simulator.registers[src1] // self.simulator.registers[src2]
                        self.simulator.registers[dest] = result
                        self.simulator.logs.append(
                            f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src1]} // {self.simulator.registers[src2]} = {result}."
                        )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Division by zero error in '{instr}'."
                        )
                    used_registers = [dest, src1, src2]
                
                elif opcode == "MOD" and len(tokens) == 4:
                    dest, src1, src2 = tokens[1], tokens[2], tokens[3]
                    if self.simulator.registers[src2] != 0:
                        result = self.simulator.registers[src1] % self.simulator.registers[src2]
                        self.simulator.registers[dest] = result
                        self.simulator.logs.append(
                            f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src1]} % {self.simulator.registers[src2]} = {result}."
                        )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Modulo by zero error in '{instr}'."
                        )
                    used_registers = [dest, src1, src2]
                
                elif opcode == "SHL" and len(tokens) == 4:
                    dest, src, shift = tokens[1], tokens[2], tokens[3]
                    if shift.isdigit():
                        shift_val = int(shift)
                        result = self.simulator.registers[src] << shift_val
                        self.simulator.registers[dest] = result
                        self.simulator.logs.append(
                            f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src]} << {shift_val} = {result}."
                        )
                    else:
                        shift_val = self.simulator.registers[shift]
                        result = self.simulator.registers[src] << shift_val
                        self.simulator.registers[dest] = result
                        self.simulator.logs.append(
                            f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src]} << {shift_val} = {result}."
                        )
                    used_registers = [dest, src]
                
                elif opcode == "SHR" and len(tokens) == 4:
                    dest, src, shift = tokens[1], tokens[2], tokens[3]
                    if shift.isdigit():
                        shift_val = int(shift)
                        result = self.simulator.registers[src] >> shift_val
                        self.simulator.registers[dest] = result
                        self.simulator.logs.append(
                            f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src]} >> {shift_val} = {result}."
                        )
                    else:
                        shift_val = self.simulator.registers[shift]
                        result = self.simulator.registers[src] >> shift_val
                        self.simulator.registers[dest] = result
                        self.simulator.logs.append(
                            f"Step {step_number}: Executed '{instr}'. {dest} = {self.simulator.registers[src]} >> {shift_val} = {result}."
                        )
                    used_registers = [dest, src]
                elif opcode == "MOV" and len(tokens) == 3:
                    dest, src = tokens[1], tokens[2]
                    if src in self.simulator.registers:
                        result = self.simulator.registers[src]
                        self.simulator.registers[dest] = result
                        self.simulator.logs.append(
                            f"Step {step_number}: Executed '{instr}'. {dest} = {result}."
                        )
                    elif src.isdigit():  # Immediate value
                        result = int(src)
                        self.simulator.registers[dest] = result
                        self.simulator.logs.append(
                            f"Step {step_number}: Executed '{instr}'. {dest} = {result} (immediate)."
                        )
                    used_registers = [dest]
                    if src in self.simulator.registers:
                        used_registers.append(src)
                
                # Load and Store operations
                elif opcode in ["LD", "LDR", "LW"] and len(tokens) >= 3:
                    dest = tokens[1]
                    # Check if the source is a memory address
                    if tokens[2].isdigit():
                        addr = int(tokens[2])
                        if 0 <= addr < len(self.simulator.memory):
                            self.simulator.registers[dest] = self.simulator.memory[addr]
                            self.simulator.logs.append(
                                f"Step {step_number}: Executed '{instr}'. Loaded {self.simulator.memory[addr]} from MEM[{addr}] to {dest}."
                            )
                            used_registers = [dest]
                            used_memory = [addr]
                        else:
                            self.simulator.logs.append(
                                f"Step {step_number}: Error: Memory address {addr} out of bounds (max: {len(self.simulator.memory)-1})."
                            )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Error: Invalid memory address format in '{instr}'."
                        )
                
                elif opcode in ["ST", "STR", "SW"] and len(tokens) >= 3:
                    src = tokens[1]
                    # Check if the destination is a memory address
                    if tokens[2].isdigit():
                        addr = int(tokens[2])
                        if 0 <= addr < len(self.simulator.memory):
                            self.simulator.memory[addr] = self.simulator.registers[src]
                            self.simulator.logs.append(
                                f"Step {step_number}: Executed '{instr}'. Stored {self.simulator.registers[src]} from {src} to MEM[{addr}]."
                            )
                            used_registers = [src]
                            used_memory = [addr]
                        else:
                            self.simulator.logs.append(
                                f"Step {step_number}: Error: Memory address {addr} out of bounds (max: {len(self.simulator.memory)-1})."
                            )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Error: Invalid memory address format in '{instr}'."
                        )
                
                # Branch operations
                elif opcode in ["B", "BR", "J", "JMP"] and len(tokens) >= 2:
                    if tokens[1].isdigit():
                        target = int(tokens[1])
                        if 0 <= target < len(self.simulator.instructions):
                            self.simulator.logs.append(
                                f"Step {step_number}: Executed '{instr}'. Jumping to instruction {target+1}."
                            )
                            # Update cycle count before changing step
                            self.simulator.cycles += cycles_for_instr
                            self.simulator.instruction_cycles[self.current_step] += cycles_for_instr
                            
                            # Update UI before changing instruction pointer
                            self.update_registers_display(self.simulator.registers)
                            self.update_memory_text_display(self.simulator.memory)
                            self.update_memory_display(self.simulator.registers, self.simulator.memory, 
                                                    highlight_registers=used_registers, 
                                                    highlight_memory=used_memory)
                            self.current_step = target - 1  # -1 because we'll increment at the end of this method
                            return  # Skip the rest of the current step processing
                        else:
                            self.simulator.logs.append(
                                f"Step {step_number}: Error: Branch target {target} out of bounds (max: {len(self.simulator.instructions)-1})."
                            )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Error: Invalid branch target in '{instr}'."
                        )
                
                elif opcode == "BEQ" and len(tokens) == 4:
                    reg1, reg2, target = tokens[1], tokens[2], tokens[3]
                    used_registers = [reg1, reg2]
                    
                    if target.isdigit():
                        target_num = int(target)
                        if 0 <= target_num < len(self.simulator.instructions):
                            if self.simulator.registers[reg1] == self.simulator.registers[reg2]:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} == {reg2}, branching to instruction {target_num+1}."
                                )
                                # Update cycle count before changing step
                                self.simulator.cycles += cycles_for_instr
                                self.simulator.instruction_cycles[self.current_step] += cycles_for_instr
                                
                                # Update UI before changing instruction pointer
                                self.update_registers_display(self.simulator.registers)
                                self.update_memory_text_display(self.simulator.memory)
                                self.update_memory_display(self.simulator.registers, self.simulator.memory, 
                                                        highlight_registers=used_registers, 
                                                        highlight_memory=used_memory)
                                self.current_step = target_num - 1  # -1 because we'll increment at the end
                                return  # Skip the rest of the current step processing
                            else:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} != {reg2}, not branching."
                                )
                        else:
                            self.simulator.logs.append(
                                f"Step {step_number}: Error: Branch target {target_num} out of bounds (max: {len(self.simulator.instructions)-1})."
                            )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Error: Invalid branch target in '{instr}'."
                        )
                
                elif opcode == "BNE" and len(tokens) == 4:
                    reg1, reg2, target = tokens[1], tokens[2], tokens[3]
                    used_registers = [reg1, reg2]
                    
                    if target.isdigit():
                        target_num = int(target)
                        if 0 <= target_num < len(self.simulator.instructions):
                            if self.simulator.registers[reg1] != self.simulator.registers[reg2]:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} != {reg2}, branching to instruction {target_num+1}."
                                )
                                # Update cycle count before changing step
                                self.simulator.cycles += cycles_for_instr
                                self.simulator.instruction_cycles[self.current_step] += cycles_for_instr
                                
                                # Update UI before changing instruction pointer
                                self.update_registers_display(self.simulator.registers)
                                self.update_memory_text_display(self.simulator.memory)
                                self.update_memory_display(self.simulator.registers, self.simulator.memory, 
                                                        highlight_registers=used_registers, 
                                                        highlight_memory=used_memory)
                                self.current_step = target_num - 1  # -1 because we'll increment at the end
                                return  # Skip the rest of the current step processing
                            else:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} == {reg2}, not branching."
                                )
                        else:
                            self.simulator.logs.append(
                                f"Step {step_number}: Error: Branch target {target_num} out of bounds (max: {len(self.simulator.instructions)-1})."
                            )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Error: Invalid branch target in '{instr}'."
                        )
                
                elif opcode == "BLT" and len(tokens) == 4:
                    reg1, reg2, target = tokens[1], tokens[2], tokens[3]
                    used_registers = [reg1, reg2]
                    
                    if target.isdigit():
                        target_num = int(target)
                        if 0 <= target_num < len(self.simulator.instructions):
                            if self.simulator.registers[reg1] < self.simulator.registers[reg2]:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} < {reg2}, branching to instruction {target_num+1}."
                                )
                                # Update cycle count before changing step
                                self.simulator.cycles += cycles_for_instr
                                self.simulator.instruction_cycles[self.current_step] += cycles_for_instr
                                
                                # Update UI before changing instruction pointer
                                self.update_registers_display(self.simulator.registers)
                                self.update_memory_text_display(self.simulator.memory)
                                self.update_memory_display(self.simulator.registers, self.simulator.memory, 
                                                        highlight_registers=used_registers, 
                                                        highlight_memory=used_memory)
                                self.current_step = target_num - 1  # -1 because we'll increment at the end
                                return  # Skip the rest of the current step processing
                            else:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} >= {reg2}, not branching."
                                )
                        else:
                            self.simulator.logs.append(
                                f"Step {step_number}: Error: Branch target {target_num} out of bounds (max: {len(self.simulator.instructions)-1})."
                            )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Error: Invalid branch target in '{instr}'."
                        )
                
                elif opcode == "BGT" and len(tokens) == 4:
                    reg1, reg2, target = tokens[1], tokens[2], tokens[3]
                    used_registers = [reg1, reg2]
                    
                    if target.isdigit():
                        target_num = int(target)
                        if 0 <= target_num < len(self.simulator.instructions):
                            if self.simulator.registers[reg1] > self.simulator.registers[reg2]:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} > {reg2}, branching to instruction {target_num+1}."
                                )
                                # Update cycle count before changing step
                                self.simulator.cycles += cycles_for_instr
                                self.simulator.instruction_cycles[self.current_step] += cycles_for_instr
                                
                                # Update UI before changing instruction pointer
                                self.update_registers_display(self.simulator.registers)
                                self.update_memory_text_display(self.simulator.memory)
                                self.update_memory_display(self.simulator.registers, self.simulator.memory, 
                                                        highlight_registers=used_registers, 
                                                        highlight_memory=used_memory)
                                self.current_step = target_num - 1  # -1 because we'll increment at the end
                                return  # Skip the rest of the current step processing
                            else:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} <= {reg2}, not branching."
                                )
                        else:
                            self.simulator.logs.append(
                                f"Step {step_number}: Error: Branch target {target_num} out of bounds (max: {len(self.simulator.instructions)-1})."
                            )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Error: Invalid branch target in '{instr}'."
                        )
                
                elif opcode == "BLE" and len(tokens) == 4:
                    reg1, reg2, target = tokens[1], tokens[2], tokens[3]
                    used_registers = [reg1, reg2]
                    
                    if target.isdigit():
                        target_num = int(target)
                        if 0 <= target_num < len(self.simulator.instructions):
                            if self.simulator.registers[reg1] <= self.simulator.registers[reg2]:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} <= {reg2}, branching to instruction {target_num+1}."
                                )
                                # Update cycle count before changing step
                                self.simulator.cycles += cycles_for_instr
                                self.simulator.instruction_cycles[self.current_step] += cycles_for_instr
                                
                                # Update UI before changing instruction pointer
                                self.update_registers_display(self.simulator.registers)
                                self.update_memory_text_display(self.simulator.memory)
                                self.update_memory_display(self.simulator.registers, self.simulator.memory, 
                                                        highlight_registers=used_registers, 
                                                        highlight_memory=used_memory)
                                self.current_step = target_num - 1  # -1 because we'll increment at the end
                                return  # Skip the rest of the current step processing
                            else:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} > {reg2}, not branching."
                                )
                        else:
                            self.simulator.logs.append(
                                f"Step {step_number}: Error: Branch target {target_num} out of bounds (max: {len(self.simulator.instructions)-1})."
                            )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Error: Invalid branch target in '{instr}'."
                        )
                
                elif opcode == "BGE" and len(tokens) == 4:
                    reg1, reg2, target = tokens[1], tokens[2], tokens[3]
                    used_registers = [reg1, reg2]
                    
                    if target.isdigit():
                        target_num = int(target)
                        if 0 <= target_num < len(self.simulator.instructions):
                            if self.simulator.registers[reg1] >= self.simulator.registers[reg2]:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} >= {reg2}, branching to instruction {target_num+1}."
                                )
                                # Update cycle count before changing step
                                self.simulator.cycles += cycles_for_instr
                                self.simulator.instruction_cycles[self.current_step] += cycles_for_instr
                                
                                # Update UI before changing instruction pointer
                                self.update_registers_display(self.simulator.registers)
                                self.update_memory_text_display(self.simulator.memory)
                                self.update_memory_display(self.simulator.registers, self.simulator.memory, 
                                                        highlight_registers=used_registers, 
                                                        highlight_memory=used_memory)
                                self.current_step = target_num - 1  # -1 because we'll increment at the end
                                return  # Skip the rest of the current step processing
                            else:
                                self.simulator.logs.append(
                                    f"Step {step_number}: Executed '{instr}'. {reg1} < {reg2}, not branching."
                                )
                        else:
                            self.simulator.logs.append(
                                f"Step {step_number}: Error: Branch target {target_num} out of bounds (max: {len(self.simulator.instructions)-1})."
                            )
                    else:
                        self.simulator.logs.append(
                            f"Step {step_number}: Error: Invalid branch target in '{instr}'."
                        )
                        
                else:
                    # Handle unsupported or unknown instructions
                    self.simulator.logs.append(
                        f"Step {step_number}: Unknown or unsupported instruction format: '{instr}'. Skipping."
                    )

                # Update cycle count based on the instruction type and ISA
                self.simulator.cycles += cycles_for_instr
                self.simulator.instruction_cycles[self.current_step] += cycles_for_instr

                # Update both the visualization and text displays
                self.update_registers_display(self.simulator.registers)
                self.update_memory_text_display(self.simulator.memory)
                self.update_memory_display(self.simulator.registers, self.simulator.memory, 
                                        highlight_registers=used_registers, 
                                        highlight_memory=used_memory)

            # Display timing and cycle information
            measured_time = (time.perf_counter() - self.instruction_start_time) * 1000
            simulated_elapsed_time = measured_time * self.execution_multiplier
            self.total_simulated_time += simulated_elapsed_time
            
            # Display cycle count information based on ISA
            if isinstance(self.simulator, RISCISA):
                cycles_info = f" (RISC: ALU=1 cycle, Load/Store=2 cycles, Branch=2 cycles)"
            elif isinstance(self.simulator, CISCISA):
                cycles_info = f" (CISC: ALU=2 cycles, Mem-to-Mem=4 cycles, Load/Store=3 cycles, Branch=3 cycles)"
            elif isinstance(self.simulator, CrazyISA):
                cycles_info = f" (Crazy ISA: 1 cycle per instruction)"
            else:
                cycles_info = ""
                
            self.simulator.logs.append(
                f"Step {step_number} completed in {simulated_elapsed_time:.3f} ms (simulated){cycles_info}. "
                f"Cycles for this instruction: {self.simulator.instruction_cycles[self.current_step]}. "
                f"Total cycles: {self.simulator.cycles}"
            )
            
            self.log_display.setPlainText("\n".join(self.simulator.logs))
            self.animate_log_display()
            self.current_step += 1
        else:
            total_cycles = self.simulator.cycles
            cycles_per_instruction = ", ".join([f"{i+1}:{c}" for i, c in enumerate(self.simulator.instruction_cycles)])
            self.simulator.logs.append(f"Program execution finished. Total simulated execution time: {self.total_simulated_time:.3f} ms.")
            self.simulator.logs.append(f"Total cycles: {total_cycles}")
            self.simulator.logs.append(f"Cycles per instruction: {cycles_per_instruction}")
            
            # Add ISA-specific summary
            if isinstance(self.simulator, RISCISA):
                self.simulator.logs.append("\nRISC ISA Summary:")
                self.simulator.logs.append("- ALU operations: 1 cycle")
                self.simulator.logs.append("- Load/Store operations: 2 cycles")
                self.simulator.logs.append("- Branch operations: 2 cycles")
            elif isinstance(self.simulator, CISCISA):
                self.simulator.logs.append("\nCISC ISA Summary:")
                self.simulator.logs.append("- ALU operations: 2 cycles")
                self.simulator.logs.append("- Memory-to-memory operations: 4 cycles")
                self.simulator.logs.append("- Load/Store operations: 3 cycles")
                self.simulator.logs.append("- Branch operations: 3 cycles")
                self.simulator.logs.append("- Multiplication/Division: 3 cycles")
            elif isinstance(self.simulator, CrazyISA):
                self.simulator.logs.append("\nCrazy ISA Summary:")
                self.simulator.logs.append("- All operations: 1 cycle (reversed operations per instruction)")
                
            self.log_display.setPlainText("\n".join(self.simulator.logs))
            self.step_timer.stop()

    def toggle_pause(self):
        if self.paused:
            self.paused = False
            self.pause_button.setText("Pause")
        else:
            self.paused = True
            self.pause_button.setText("Resume")
    
    def animate_log_display(self):
        orig_geom = self.log_display.geometry()
        start_geom = QRect(orig_geom.x(), orig_geom.y() + 50, orig_geom.width(), orig_geom.height())
        self.log_display.setGeometry(start_geom)
        self.log_animation.setStartValue(start_geom)
        self.log_animation.setEndValue(orig_geom)
        self.log_animation.start()
    
    def get_stylesheet(self):
        return """
        QMainWindow {
            background-color: #2b2b2b;
        }
        QLabel {
            color: #e0e0e0;
            font-size: 14px;
        }
        QTextEdit, QPlainTextEdit {
            background-color: #3c3f41;
            color: #ffffff;
            border: 1px solid #5c5c5c;
            border-radius: 5px;
            padding: 5px;
        }
        QPushButton {
            background-color: #007acc;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #005f99;
        }
        QComboBox {
            background-color: #3c3f41;
            color: #ffffff;
            border: 1px solid #5c5c5c;
            border-radius: 5px;
            padding: 5px;
        }
        QScrollArea {
            background-color: #3c3f41;
            border: none;
        }
        """
        
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()
                    