import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

# -----------------------------
# 1. Define qubits
# -----------------------------
n_qubits = 4
qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]

# -----------------------------
# 2. Create quantum circuit
# -----------------------------
def create_quantum_circuit():
    circuit = cirq.Circuit()
    symbols = sympy.symbols(f'theta0:{n_qubits}')
    
    for i, q in enumerate(qubits):
        circuit.append(cirq.rx(symbols[i])(q))
    
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[2], qubits[3]))
    
    return circuit, symbols

quantum_circuit, symbols = create_quantum_circuit()

# -----------------------------
# 3. FIXED classical → quantum encoding
# -----------------------------
def classical_to_circuit(x):
    batch_size = tf.shape(x)[0]
    empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
    return tf.tile(empty_circuit, [batch_size])

# -----------------------------
# 4. Build Hybrid Model
# -----------------------------
# Classical input (e.g., soil, rainfall, crop type)
x_classical = tf.keras.Input(shape=(8,), dtype=tf.float32)

# Convert to quantum circuits
quantum_input = tf.keras.layers.Lambda(classical_to_circuit)(x_classical)

# Quantum layer
quantum_layer = tfq.layers.PQC(
    quantum_circuit,
    operators=[cirq.Z(q) for q in qubits]
)(quantum_input)

# Classical layers after quantum
dense = tf.keras.layers.Dense(16, activation='relu')(quantum_layer)
output = tf.keras.layers.Dense(1)(dense)

# Final model
model = tf.keras.Model(inputs=x_classical, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse'
)

# -----------------------------
# 5. Dummy Multi-Farm Dataset
# -----------------------------
# Each row = farm with multiple crop features
X_train = np.random.rand(20, 8)
y_train = np.random.rand(20, 1)

# -----------------------------
# 6. Train Model
# -----------------------------
print("Training model...")
model.fit(X_train, y_train, epochs=5)

# -----------------------------
# 7. Test Prediction
# -----------------------------
test_sample = np.random.rand(1, 8)
prediction = model.predict(test_sample)

print("Test Prediction:", prediction)
