# 🔐 ADSA Project: CCTV Camera Placement using Greedy Approximation for Set Cover

This project simulates the optimal placement of CCTV cameras in a surveillance area using a greedy approximation approach to the classic **Set Cover Problem (SCP)**.

## 📌 Project Overview

The **Set Cover Problem** is a well-known NP-hard problem in computer science and operations research. In this context, we model a surveillance grid as a set of demand points and attempt to cover all points using the **minimum number of CCTV cameras**, each with a fixed range of coverage.

📹 Each camera can cover a circular area.  
🧠 The greedy algorithm chooses cameras that cover the most uncovered points at each step.  
📉 The goal is to reduce total cameras while maintaining full coverage.

This project showcases how **approximation algorithms** can solve complex combinatorial problems efficiently in real-world scenarios.

## 🧠 Problem Statement

> _Given a set of locations that require surveillance (demand points), and a set of possible CCTV placements (with known coverage range), choose the smallest subset of placements such that all areas are covered._

The Set Cover Problem is NP-hard, meaning that **exact algorithms** are computationally infeasible for large inputs. Hence, a **greedy heuristic** is used for an efficient and near-optimal solution.

## ⚙️ Features

- Models a 2D grid of surveillance points.
- Simulates camera placement with customizable range.
- Implements a greedy approximation algorithm.
- Visualizes the selected camera positions and their coverage.
- Interactive and reproducible for academic or demo purposes.

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/aeiou-sudo/ADSA.git
cd ADSA
```

### 2. Set up the environment
This project requires Python 3.x. Use the following to create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Then install the required libraries:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, just ensure `matplotlib`, `numpy`, and `random` are installed.

### 3. Run the simulation

```bash
python CCTV.py
```

This will launch a simulation where cameras are placed to cover the grid using the greedy algorithm. The final result is shown using `matplotlib` as a plot.

## 🖼️ Demo
You can find a video demo of the simulation here *[[LinkedIn](https://www.linkedin.com/in/paul-jose-016)]*.

## 📚 References
This project is inspired by and loosely based on:

F. Barriga-Gallegos, A. Lüer-Villagra, and G. Gutiérrez-Jarpa, "The Angular Set Covering Problem," *IEEE Access*, vol. 12, pp. 87181–87198, 2024, doi: 10.1109/ACCESS.2024.3416871.

## 🙋‍♂️ Author
**Paul Jose** - M.Tech CSE - Mar Athanasius College of Engineering - [[LinkedIn](https://www.linkedin.com/in/paul-jose-016)] | [[Email](mailto:pauljose513@gmail.com)]

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.