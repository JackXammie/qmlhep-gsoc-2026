# ===============================================
# GSoC Quantum ML Example: Single-Qubit PQC
# ===============================================

# 1. IMPORT LIBRARIES
# Cirq for quantum circuits, Sympy for symbolic parameters,
# NumPy for calculations, TensorFlow and TFQ for training
import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

# 2. DEFINE QUBIT AND PARAMETER
# Single qubit at grid position (0,0)
# 'theta' is the trainable parameter for RY rotation
qubit = cirq.GridQubit(0, 0)
theta = sympy.Symbol('theta')

# 3. BUILD PARAMETERIZED CIRCUIT
# Apply RY(theta) rotation and measure
circuit = cirq.Circuit(
    cirq.ry(theta)(qubit),
    cirq.measure(qubit, key='result')
)
print("Quantum Circuit:")
print(circuit)

# 4. PREPARE PQC INPUTS AND OBSERVABLE
# TFQ requires circuits as tensors
inputs = tfq.convert_to_tensor([circuit])
observable = cirq.Z(qubit)  # Target expectation: Z

# 5. CREATE PQC MODEL
# Use TFQ PQC layer with the circuit and observable
model = tf.keras.Sequential([
    tfq.layers.PQC(circuit, observable)
])

# 6. COMPILE MODEL
# Adam optimizer and mean squared error loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss=tf.keras.losses.MeanSquaredError()
)

# Target expectation for the qubit
target = np.array([[-1]])

# 7. TRAIN MODEL
# Run 20 epochs, printing loss and expectation each step
for epoch in range(20):
    history = model.fit(inputs, target, epochs=1, verbose=0)
    trained_exp = model.predict(inputs)
    print(f"Epoch {epoch+1}/20 - Loss: {history.history['loss'][0]:.4f} - Expectation: {trained_exp[0][0]:.4f}")

# 8. FINAL TRAINED EXPECTATION
print("Trained expectation:", model.predict(inputs))
