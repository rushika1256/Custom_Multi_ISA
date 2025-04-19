# Custom_Multi_ISA
A Python-based educational simulator that demonstrates the differences between RISC (Reduced Instruction Set Computing) and CISC (Complex Instruction Set Computing) architectures, along with a "Crazy" ISA for experimental purposes.

 Features
- Multiple ISA Support:
  - RISC: Simple instructions with 1-2 cycles per operation
  - CISC: Complex instructions with 2-4 cycles per operation
  - Crazy ISA: Experimental architecture with reversed operations

- Instruction Types:
  - Arithmetic operations (ADD, SUB, MUL, DIV)
  - Logical operations (AND, OR, XOR, NOT)
  - Memory operations (LD/ST for RISC, Memory-to-Memory for CISC)
  - Branch operations (BEQ, BNE, BLT, BGT, etc.)
  - Shift operations (SHL, SHR)

- Real-time Visualization:
  - Register contents display
  - Memory contents display
  - Instruction execution logging
  - Cycle counting and timing information

 Requirements
- Python must be installed

How to use ?
1. Run the simulator: `python main.py`
2. Select an ISA (RISC/CISC/Crazy)
3. Enter assembly instructions in the editor
4. Click "Execute Program" to run
5. Use "Pause/Resume" to control execution

Sample Assembly Instructions:
- ADD R1, R2, R3    # RISC: Add R2 and R3, store in R1
- MOV MEM[1], MEM[0] # CISC: Move value from memory[0] to memory[1]
- LDR R1, 5         # Load from memory address 5 to R1
- STR R2, 10        # Store R2 to memory address 10
- BEQ R1, R2, 5     # Branch to instruction 5 if R1 equals R2

Architecture Details
- RISC: Emphasizes simple instructions, register-based operations
- CISC: Supports complex memory-to-memory operations
- Memory: 32 locations (0-31)
- Registers: 32 registers (R0-R31, CISC limited to R0-R15)
