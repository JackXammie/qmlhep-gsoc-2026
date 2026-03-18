#1 Imports
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

#2 Farm Data (same as before)
X = np.array([
    [6.4, 28, 45, 180, 3.2, 12.5],       # Soil
    [1792, 3.1, 52.5, 0, 0, 0],          # Plant
    [1, 1, 1, 1, 0, 0],                  # Routine
    [1, 1, 1, 0, 0, 0],                  # Nutrients
    [110, 20, 3, 0, 0, 0]                # Harvest
], dtype=np.float32)

# Normalize (VERY IMPORTANT for quantum rotations)
X = X / np.max(X)

#3 Graph edges
edges = [(0,1), (2,1), (3,1), (1,4), (0,4)]

#4 Label
y = np.array([[110]], dtype=np.float32)

#5 Define qubits per node
qubits = [
    cirq.GridQubit(0,0), cirq.GridQubit(0,1),  # Soil (2)
    cirq.GridQubit(1,0), cirq.GridQubit(1,1),  # Plant (2)
    cirq.GridQubit(2,0),                      # Routine (1)
    cirq.GridQubit(3,0),                      # Nutrients (1)
    cirq.GridQubit(4,0), cirq.GridQubit(4,1)   # Harvest (2)
]

#6 Build DATA-ENCODING circuit
def create_circuit(features):
    circuit = cirq.Circuit()

    # Flatten all features
    flat_features = features.flatten()

    # Encode into qubits (angle encoding)
    for i, val in enumerate(flat_features[:len(qubits)]):
        circuit.append(cirq.ry(val * np.pi)(qubits[i]))

    # Entangle based on graph edges
    for src, tgt in edges:
        q_src = qubits[src]
        q_tgt = qubits[tgt]
        circuit.append(cirq.CNOT(q_src, q_tgt))

    return circuit

#7 Create quantum dataset (ONE graph sample)
circuits = [create_circuit(X)]
quantum_data = tfq.convert_to_tensor(circuits)

#8 Parameterized circuit (trainable)
theta = sympy.symbols('theta0:8')

pqc = cirq.Circuit()
for i, q in enumerate(qubits):
    pqc.append(cirq.ry(theta[i])(q))

# Observable
observable = sum([cirq.Z(q) for q in qubits])

#9 Build model
quantum_input = tf.keras.Input(shape=(), dtype=tf.string)

quantum_layer = tfq.layers.PQC(pqc, observable)
x = quantum_layer(quantum_input)

output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=quantum_input, outputs=output)

#10 Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='mean_squared_error'
)

#11 Train
print("Training realistic quantum farm model...")
model.fit(quantum_data, y, epochs=20)

#12 Predict
pred = model.predict(quantum_data)
print("Prediction:", pred)
