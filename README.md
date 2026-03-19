🌾 Quantum Machine Learning + Agriculture System

📌 Overview

This repository contains my implementations for the QMLHEP GSoC 2026 evaluation tasks, extended into a real-world application: agriculture systems modeling using Quantum Machine Learning (QML).

The work combines:

- Quantum circuits
- Hybrid quantum–classical models
- Graph-based learning
- Multi-system (farm) modeling

---

🧠 Project Idea

  I developed a system where:

- Each farm = node
- Each node contains crop + environmental data
- Data is encoded into quantum circuits (qubits)
- Farms can interact and scale

This turns the project into a real-world QML application, not just isolated tasks.

---

📁 Repository Structure

Task1_Quantum_Circuits/
    train_qml.py

hybrid_quantum_agri_model.py      # Task 2: Hybrid Quantum Model

multi_farm_quantum_model.py       # Task 3: Multi-Farm System

requirements.txt
README.md

---

⚙️ What Each Task Does

🔹 Task 1 – Quantum Circuits

- Basic quantum circuit design
- Parameterized quantum models
- Learning simple patterns

---

🔹 Task 2 – Hybrid Quantum Model

- Combines classical + quantum layers
- Encodes agricultural features into qubits
- Produces predictions via measurement

---

🔹 Task 3 – Multi-Farm System

- Models multiple farms simultaneously
- Each farm = independent + scalable unit
- Foundation for graph-based interactions

---

🚀 Getting Started

git clone https://github.com/JackXammie/agric-qml.git
cd agric-qml

python3 -m venv tfq-env
source tfq-env/bin/activate

pip install -r requirements.txt

python multi_farm_quantum_model.py

---

🔬 Key Concepts

- Qubits: Represent farm states
- Quantum Encoding: Converts classical farm data
- Hybrid Learning: Classical + Quantum models
- Scalability: Extendable to many farms

---

🌍 Future Work

- Graph Neural Networks (farm interactions)
- Real agricultural datasets
- Optimization (yield, irrigation, nutrients)
- Quantum Graph Neural Networks

---

🎯 Goal

To explore how quantum machine learning can move from theory → real-world systems, using agriculture as a scalable testbed.

---

🧰 Tools Used

- Cirq
- TensorFlow Quantum
- Python

---

📧 Contact

For questions or collaboration, reach out via GitHub.
