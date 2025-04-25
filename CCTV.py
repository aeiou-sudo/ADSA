import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QPushButton, QSpinBox, QDoubleSpinBox, QTextEdit, QComboBox)
from PyQt5.QtCore import Qt
import sys
import time
import pulp

def visualize_camera_placement(ax, grid_size, demand_grid, site_positions, option_list, chosen_options, configs):
    ax.clear()
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)

    for i in range(grid_size):
        for j in range(grid_size):
            if demand_grid[i, j] == 1:
                ax.add_patch(patches.Rectangle((j, grid_size - i - 1), 1, 1, color='red', alpha=0.4))

    for idx, (si, sj) in enumerate(site_positions):
        ax.add_patch(patches.Rectangle((sj, grid_size - si - 1), 1, 1, color='blue', alpha=0.3))
        ax.text(sj + 0.2, grid_size - si - 0.8, f"S{idx}", fontsize=9, color='black')

    covered_demand = np.zeros_like(demand_grid)
    for opt_idx in chosen_options:
        site_id, fov, ori = option_list[opt_idx]
        si, sj = site_positions[site_id]
        x, y = sj + 0.5, grid_size - si - 0.5
        radius = configs[fov]
        
        wedge = patches.Wedge((x, y), radius, ori - fov / 2, ori + fov / 2,
                            color='green', alpha=0.2)
        ax.add_patch(wedge)
        ax.plot(x, y, 'ko')
        
        for i in range(grid_size):
            for j in range(grid_size):
                if demand_grid[i, j] == 1 and not covered_demand[i, j]:
                    if is_point_covered_vis(si, sj, i, j, radius, fov, ori, grid_size):
                        covered_demand[i, j] = 1
                        ax.add_patch(patches.Rectangle(
                            (j, grid_size - i - 1), 1, 1, 
                            facecolor='none', edgecolor='green', linewidth=2))

    ax.set_title("CCTV Angular Coverage on Grid\nRed: Demand | Blue: Camera Site | Green: FOV")
    ax.set_aspect('equal')

def is_point_covered_vis(si, sj, di, dj, radius, fov, ori, grid_size):
    dx, dy = dj - sj, (grid_size - di - 1) - (grid_size - si - 1)
    dist = np.hypot(dx, dy)
    
    if dist <= radius:
        angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        angle_diff = min((angle - ori) % 360, (ori - angle) % 360)
        return angle_diff <= fov / 2
    return False

def is_point_covered(si, sj, di, dj, radius, fov, ori):
    dx, dy = dj - sj, si - di  # Corrected to match visualization coordinates
    dist = np.hypot(dx, dy)
    
    if dist <= radius:
        angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        angle_diff = min((angle - ori) % 360, (ori - angle) % 360)
        return angle_diff <= fov / 2
    return False

def get_coverage_matrix(grid_size, demand_grid, site_positions, configs):
    demand_positions = [(i, j) for i in range(grid_size) for j in range(grid_size) if demand_grid[i, j] == 1]
    
    option_list = []
    a_matrix = []
    
    for site_id, (si, sj) in enumerate(site_positions):
        for fov, radius in configs.items():
            ori_step = min(10, fov // 4)
            
            targeted_orientations = set()
            for di, dj in demand_positions:
                if np.hypot(dj - sj, si - di) <= radius:
                    angle = (np.degrees(np.arctan2(si - di, dj - sj)) + 360) % 360
                    for offset in [-ori_step/2, 0, ori_step/2]:
                        targeted_orientations.add(int((angle + offset) % 360))
            
            for ori in range(0, 360, ori_step):
                targeted_orientations.add(ori)
            
            orientations = sorted(targeted_orientations)
            
            for ori in orientations:
                coverage_vector = [0] * len(demand_positions)
                
                for pos_idx, (di, dj) in enumerate(demand_positions):
                    if is_point_covered(si, sj, di, dj, radius, fov, ori):
                        coverage_vector[pos_idx] = 1
                
                if sum(coverage_vector) > 0:
                    option_list.append((site_id, fov, ori))
                    a_matrix.append(coverage_vector)
    
    return option_list, np.array(a_matrix)

def run_optimization(grid_size, demand_grid, site_positions, configs, g_cost, c_cost, method='column_generation'):
    if not site_positions or np.sum(demand_grid) == 0:
        return [], [], 0
    
    demand_positions = [(i, j) for i in range(grid_size) for j in range(grid_size) if demand_grid[i, j] == 1]
    m = len(demand_positions)
    
    option_list, a_matrix = get_coverage_matrix(grid_size, demand_grid, site_positions, configs)
    
    if len(option_list) == 0:
        return [], [], 0
    
    site_to_options = {}
    for idx, (site_id, _, _) in enumerate(option_list):
        if site_id not in site_to_options:
            site_to_options[site_id] = []
        site_to_options[site_id].append(idx)
    
    if method == 'column_generation' and len(option_list) > 100:
        chosen_options = column_generation_solver(
            a_matrix, option_list, site_to_options, m, g_cost, c_cost)
    else:
        chosen_options = direct_ilp_solver(
            a_matrix, option_list, site_to_options, m, g_cost, c_cost)
    

    used_sites = set(option_list[idx][0] for idx in chosen_options)
    total_cost = g_cost * len(used_sites) + c_cost * len(chosen_options)
    
    if chosen_options:
        covered_demands = set()
        for idx in chosen_options:
            row = a_matrix[idx]
            for j in range(m):
                if row[j] == 1:
                    covered_demands.add(j)
        
        coverage_percent = len(covered_demands) / m * 100
        
        print(f"Demand points covered: {len(covered_demands)}/{m} ({coverage_percent:.1f}%)")
        print(f"Cameras used: {len(chosen_options)}")
        print(f"Sites utilized: {len(used_sites)}")
        print(f"Total cost: {total_cost}")
    
    return option_list, chosen_options, total_cost

def direct_ilp_solver(a_matrix, option_list, site_to_options, m, g_cost, c_cost):
    n = len(option_list)
    sites = list(site_to_options.keys())
    
    prob = pulp.LpProblem("CCTV_Placement", pulp.LpMinimize)
    
    x = {j: pulp.LpVariable(f"x_{j}", cat=pulp.LpBinary) for j in range(n)}
    y = {i: pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary) for i in sites}
    
    prob += pulp.lpSum(g_cost * y[i] for i in sites) + pulp.lpSum(c_cost * x[j] for j in range(n))
    
    for i in range(m):
        prob += pulp.lpSum(a_matrix[j][i] * x[j] for j in range(n)) >= 1
    
    for site_id in sites:
        for j in site_to_options[site_id]:
            prob += x[j] <= y[site_id]
    
    total_cameras = min(m * 2, n)
    prob += pulp.lpSum(x[j] for j in range(n)) <= total_cameras
    
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    chosen_options = [j for j in range(n) if pulp.value(x[j]) > 0.5]
    return chosen_options

def column_generation_solver(a_matrix, option_list, site_to_options, m, g_cost, c_cost):
    current_options = set()
    covered_demands = set()
    
    while len(covered_demands) < m:
        best_option = None
        best_efficiency = -1
        
        for j, row in enumerate(a_matrix):
            if j in current_options:
                continue
                
            new_coverage = sum(1 for i in range(m) if row[i] == 1 and i not in covered_demands)
            
            if new_coverage == 0:
                continue
            
            site_id = option_list[j][0]
            additional_site_cost = g_cost if site_id not in {option_list[i][0] for i in current_options} else 0
            option_cost = additional_site_cost + c_cost
            efficiency = new_coverage / option_cost
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_option = j
        
        if best_option is None or best_efficiency <= 0:
            break
            
        current_options.add(best_option)
        for i in range(m):
            if a_matrix[best_option][i] == 1:
                covered_demands.add(i)
    
    if len(covered_demands) < m:
        return direct_ilp_solver(a_matrix, option_list, site_to_options, m, g_cost, c_cost)
    
    max_iterations = 30
    improvement_threshold = 0.001
    
    best_cost = float('inf')
    best_solution = list(current_options)
    
    for iteration in range(max_iterations):
        restricted_options = list(current_options)
        restricted_a_matrix = a_matrix[restricted_options]
        
        restricted_site_to_options = {}
        for idx, opt_idx in enumerate(restricted_options):
            site_id = option_list[opt_idx][0]
            if site_id not in restricted_site_to_options:
                restricted_site_to_options[site_id] = []
            restricted_site_to_options[site_id].append(idx)
        
        chosen_indices = direct_ilp_solver(
            restricted_a_matrix, 
            [option_list[idx] for idx in restricted_options], 
            restricted_site_to_options, 
            m, g_cost, c_cost
        )
        
        chosen_options = [restricted_options[idx] for idx in chosen_indices]
        
        used_sites = set(option_list[idx][0] for idx in chosen_options)
        current_cost = g_cost * len(used_sites) + c_cost * len(chosen_options)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = chosen_options
            improvement = (best_cost - current_cost) / best_cost if best_cost > 0 else 1.0
            if improvement < improvement_threshold:
                break
        
        coverage_count = np.zeros(m)
        for opt_idx in chosen_options:
            coverage_count += a_matrix[opt_idx]
        
        critical_demands = np.where(coverage_count == 1)[0]
        
        added_new_option = False
        
        for demand_idx in critical_demands:
            current_covering_option = None
            for opt_idx in chosen_options:
                if a_matrix[opt_idx][demand_idx] == 1:
                    current_covering_option = opt_idx
                    break
            
            best_new_option = None
            best_additional_coverage = -1
            
            for j in range(len(option_list)):
                if j in current_options:
                    continue
                
                if a_matrix[j][demand_idx] == 1:
                    additional_coverage = 0
                    for d_idx in range(m):
                        if a_matrix[j][d_idx] == 1 and coverage_count[d_idx] == 1:
                            additional_coverage += 1
                    
                    if additional_coverage > best_additional_coverage:
                        best_additional_coverage = additional_coverage
                        best_new_option = j
            
            if best_new_option is not None and best_additional_coverage > 0:
                current_options.add(best_new_option)
                added_new_option = True
        
        if not added_new_option:
            option_coverages = []
            for opt_idx in chosen_options:
                covered_demands = {i for i in range(m) if a_matrix[opt_idx][i] == 1}
                option_coverages.append((opt_idx, covered_demands))
            
            for j in range(len(option_list)):
                if j in current_options:
                    continue
                
                new_coverage = {i for i in range(m) if a_matrix[j][i] == 1}
                
                if len(new_coverage) == 0:
                    continue
                
                for opt_idx, opt_coverage in option_coverages:
                    if new_coverage.issuperset(opt_coverage) and len(new_coverage) > len(opt_coverage):
                        current_options.add(j)
                        added_new_option = True
                        break
            
            if not added_new_option:
                break
    
    return best_solution

class CCTVOptimizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV Angular Coverage Optimizer")
        self.setGeometry(100, 100, 1000, 600)
        self.grid_size = 10

        self.demand_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.site_positions = []
        self.placing_demand = True
        self.option_list = []
        self.chosen_opts = []

        layout = QHBoxLayout()
        control_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 3)
        layout.addLayout(control_layout, 1)
        self.setLayout(layout)

        control_layout.addWidget(QLabel("<b>Grid Configuration</b>"))
        
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid Size:"))
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(5, 30)
        self.grid_spin.setValue(10)
        self.grid_spin.valueChanged.connect(self.on_grid_size_change)
        grid_layout.addWidget(self.grid_spin)
        control_layout.addLayout(grid_layout)

        self.demand_btn = QPushButton("Place Demands")
        self.demand_btn.setCheckable(True)
        self.demand_btn.setChecked(True)
        self.demand_btn.clicked.connect(lambda: self.set_mode(True))
        
        self.site_btn = QPushButton("Place Sites")
        self.site_btn.setCheckable(True)
        self.site_btn.clicked.connect(lambda: self.set_mode(False))
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.demand_btn)
        btn_layout.addWidget(self.site_btn)
        control_layout.addLayout(btn_layout)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        control_layout.addWidget(self.clear_btn)

        control_layout.addWidget(QLabel("<b>Optimization Settings</b>"))
        
        cost_layout = QHBoxLayout()
        cost_layout.addWidget(QLabel("Site Cost:"))
        self.site_cost_spin = QSpinBox()
        self.site_cost_spin.setRange(1, 100)
        self.site_cost_spin.setValue(10)
        cost_layout.addWidget(self.site_cost_spin)
        
        cost_layout.addWidget(QLabel("Camera Cost:"))
        self.camera_cost_spin = QSpinBox()
        self.camera_cost_spin.setRange(1, 100)
        self.camera_cost_spin.setValue(5)
        cost_layout.addWidget(self.camera_cost_spin)
        control_layout.addLayout(cost_layout)

        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Column Generation", "Direct ILP"])
        algo_layout.addWidget(self.algo_combo)
        control_layout.addLayout(algo_layout)

        fov_layout = QHBoxLayout()
        fov_layout.addWidget(QLabel("FOV Configuration:"))
        self.fov_combo = QComboBox()
        self.fov_combo.addItems(["Standard (90°/60°)", "Wide (120°/90°/60°)", "Narrow (60°/45°/30°)"])
        fov_layout.addWidget(self.fov_combo)
        control_layout.addLayout(fov_layout)

        self.run_btn = QPushButton("Run Optimization")
        self.run_btn.clicked.connect(self.run)
        control_layout.addWidget(self.run_btn)

        control_layout.addWidget(QLabel("<b>Results</b>"))
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        control_layout.addWidget(self.output)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.draw_grid()

    def set_mode(self, is_demand):
        self.placing_demand = is_demand
        self.demand_btn.setChecked(is_demand)
        self.site_btn.setChecked(not is_demand)
        mode_text = "Demands (Red)" if self.placing_demand else "Sites (Blue)"
        self.ax.set_title(f"Click to Place {mode_text}")
        self.canvas.draw()

    def on_grid_size_change(self):
        self.clear_all()

    def clear_all(self):
        grid_size = self.grid_spin.value()
        self.grid_size = grid_size
        self.demand_grid = np.zeros((grid_size, grid_size), dtype=int)
        self.site_positions = []
        self.option_list = []
        self.chosen_opts = []
        self.output.clear()
        self.draw_grid()

    def draw_grid(self):
        fov_config = self.get_fov_config()
        
        visualize_camera_placement(self.ax, self.grid_size, self.demand_grid, 
                                  self.site_positions, self.option_list, 
                                  self.chosen_opts, fov_config)
        self.canvas.draw()

    def get_fov_config(self):
        fov_option = self.fov_combo.currentText()
        if fov_option == "Wide (120°/90°/60°)":
            return {120: 3, 90: 4, 60: 5}
        elif fov_option == "Narrow (60°/45°/30°)":
            return {60: 4, 45: 5, 30: 6}
        else:  # Standard
            return {90: 4, 60: 5}

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        i = self.grid_size - int(event.ydata) - 1
        j = int(event.xdata)
        if not (0 <= i < self.grid_size and 0 <= j < self.grid_size):
            return
            
        self.option_list = []
        self.chosen_opts = []
        
        if self.placing_demand:
            self.demand_grid[i, j] = 1 - self.demand_grid[i, j]
        else:
            pos = (i, j)
            if pos in self.site_positions:
                self.site_positions.remove(pos)
            else:
                self.site_positions.append(pos)
        
        self.draw_grid()

    def run(self):
        if not self.site_positions:
            self.output.setText("Error: No camera sites placed")
            return
            
        if np.sum(self.demand_grid) == 0:
            self.output.setText("Error: No demand points placed")
            return
        
        configs = self.get_fov_config()
        g_cost = self.site_cost_spin.value()
        c_cost = self.camera_cost_spin.value()
        
        method = 'column_generation' if self.algo_combo.currentText() == "Column Generation" else 'direct'
        
        self.output.setText("Optimizing camera placement...\nThis may take a moment.")
        QApplication.processEvents()
        
        start_time = time.time()
        
        self.option_list, self.chosen_opts, total_cost = run_optimization(
            self.grid_size, self.demand_grid, self.site_positions, configs, g_cost, c_cost, method)
        
        end_time = time.time()
        
        self.draw_grid()

        demand_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) 
                           if self.demand_grid[i, j] == 1]
        
        if self.chosen_opts:
            covered_demands = set()
            for di, dj in demand_positions:
                for opt_idx in self.chosen_opts:
                    site_id, fov, ori = self.option_list[opt_idx]
                    si, sj = self.site_positions[site_id]
                    radius = configs[fov]
                    
                    if is_point_covered(si, sj, di, dj, radius, fov, ori):
                        covered_demands.add((di, dj))
                        break
            
            coverage_percent = len(covered_demands) / len(demand_positions) * 100 if demand_positions else 0
            
            used_sites = {self.option_list[k][0] for k in self.chosen_opts}
            
            self.output.setText(
                f"Optimization Results:\n\n"
                f"Total Cost: {total_cost}\n"
                f"Camera Sites Used: {len(used_sites)}/{len(self.site_positions)}\n"
                f"Cameras Deployed: {len(self.chosen_opts)}\n"
                f"Demand Coverage: {len(covered_demands)}/{len(demand_positions)} ({coverage_percent:.1f}%)\n"
                f"Computation Time: {end_time - start_time:.2f} seconds\n\n"
                f"Algorithm: {self.algo_combo.currentText()}\n\n"
                f"Legend:\n"
                f"- Red: Demand Points\n"
                f"- Blue: Potential Camera Sites\n"
                f"- Green Wedge: Selected Camera FOV\n"
                f"- Green Border: Covered Demand Point"
            )
        else:
            self.output.setText("No optimal solution found.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CCTVOptimizer()
    win.show()
    sys.exit(app.exec_())